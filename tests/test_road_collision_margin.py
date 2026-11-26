"""Tolerance changes contact detection, preserving the nominal lane geometry."""
import math

import pytest
import torch

from utilities.helper_scenario import interX
from utilities.road_collision import lane_boundary_collision, RoadBoundaryCollision


def geometry(offsets, yaws):
    offsets = torch.tensor(offsets)
    yaw = torch.tensor(yaws)
    t = torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
    n = torch.stack((-torch.sin(yaw), torch.cos(yaw)), -1)
    c = torch.stack((torch.zeros_like(offsets), offsets), -1)
    vertices = torch.stack([c+s*.08*t+r*.04*n for s, r in
                            ((-1,-1),(1,-1),(1,1),(-1,1),(-1,-1))], 1)
    x = torch.linspace(-1., 1., 31)
    left = torch.stack((x, torch.full_like(x, .09)), -1).expand(len(offsets), -1, -1).clone()
    right = left.clone()
    right[..., 1] = -.09
    return vertices, left, right


def test_small_crossing_is_tolerated_but_larger_crossing_is_not():
    v, left, right = geometry([0., .06, -.06, .08, -.08], [0.]*5)
    before = [t.clone() for t in (v, left, right)]
    assert lane_boundary_collision(v, left, right).tolist() == [False, True, True, True, True]
    assert lane_boundary_collision(v, left, right, .02).tolist() == [False, False, False, True, True]
    for actual, original in zip((v, left, right), before):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_rotated_front_and_rear_corners_still_trigger_contacts():
    v, left, right = geometry([.02, .04, -.02, -.04], [math.pi/6]*4)
    assert lane_boundary_collision(v, left, right).all()
    assert lane_boundary_collision(v, left, right, .02).tolist() == [False, True, False, True]


def test_zero_margin_preserves_original_curved_boundary_intersections():
    v, left, right = geometry([0., .06, -.06], [0., .2, -.4])
    left[..., 1] += .025*left[..., 0]**2
    right[..., 1] += .025*right[..., 0]**2
    expected = interX(v, left, False) | interX(v, right, False)
    torch.testing.assert_close(lane_boundary_collision(v, left, right, 0.), expected)
    # A rigid rotation must not change which lateral excursions are tolerated.
    theta = torch.tensor(.8)
    rotation = torch.stack((torch.stack((theta.cos(), -theta.sin())),
                            torch.stack((theta.sin(), theta.cos()))))
    rotated = lane_boundary_collision(v@rotation.T, left@rotation.T, right@rotation.T, .02)
    torch.testing.assert_close(rotated, lane_boundary_collision(v, left, right, .02))


@pytest.mark.parametrize("margin", [-.01, float("nan"), float("inf")])
def test_invalid_margins_are_rejected(margin):
    with pytest.raises(ValueError):
        lane_boundary_collision(*geometry([0.], [0.]), margin)


def test_white_divider_is_not_a_physical_road_boundary():
    outer = [[-1.,-.18],[1.,-.18],[1.,.18],[-1.,.18]]
    road = RoadBoundaryCollision([[outer]])
    v, left, right = geometry([0., .08, -.08, .15, -.15, .4], [0.]*6)
    # The vehicle can straddle the white divider between the two lanes.
    assert road(v).tolist() == [False, False, False, True, True, True]
    assert road(v.reshape(2, 3, 5, 2)).shape == (2, 3)
    expanded = [[-1.02,-.20],[1.02,-.20],[1.02,.20],[-1.02,.20]]
    assert RoadBoundaryCollision([[expanded]])(v).tolist() == [False, False, False, False, False, True]


def test_island_and_disconnected_roads_do_not_create_false_bridge_segments():
    outer = [[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]
    tiny_island = [[-.02,-.02],[.02,-.02],[.02,.02],[-.02,.02]]
    v, _, _ = geometry([0., .3, 2.], [0., math.pi/6, 0.])
    # Island wholly within the body: all four car corners are on road.
    assert RoadBoundaryCollision([[outer, tiny_island]])(v).tolist() == [True, False, True]
    other = [[-1.,1.8],[1.,1.8],[1.,2.2],[-1.,2.2]]
    assert RoadBoundaryCollision([[outer], [other]])(v).tolist() == [False, False, False]


def test_concave_curb_crossing_is_detected_between_vehicle_corners():
    # A very thin notch enters the rectangle between its front and rear.
    outer = [[-1.,-1.],[1.,-1.],[1.,1.],[.01,1.],[.01,0.],
             [-.01,0.],[-.01,1.],[-1.,1.]]
    v, _, _ = geometry([0.], [0.])
    assert RoadBoundaryCollision([[outer]])(v).item()
