"""Lane-contact tolerance without changing the vehicle or observation geometry."""
import math
import json
from pathlib import Path

import torch

from utilities.helper_scenario import interX


def lane_boundary_collision(vertices, left, right, margin=0.):
    """Intersect the real vehicle with optional outward-offset route edges.

    `margin` is metres per side, not a penalty multiplier or vehicle shrink.
    Left/right arrays have corresponding samples, including padded tails.
    Zero retains the original intersection calls and results exactly.
    """
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Lane collision margin must be finite and non-negative")
    if margin:
        normal = left-right
        normal = normal/torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-12)
        left, right = left+margin*normal, right-margin*normal
    return (interX(L1=vertices, L2=left, is_return_points=False)
            | interX(L1=vertices, L2=right, is_return_points=False))


class RoadBoundaryCollision:
    """Vehicle containment in pavement polygons, with all island holes kept.

    Polygon rings come from the physical road union, independent of lane
    assignment/paint. The asset builder applies any curb tolerance once.
    """

    def __init__(self, regions, device="cpu"):
        starts, ends = [], []
        for region in regions:
            for ring in region:
                p = torch.as_tensor(ring, dtype=torch.float32, device=device)
                if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] != 2 or not torch.isfinite(p).all():
                    raise ValueError("Invalid physical road boundary ring")
                starts.append(p)
                ends.append(p.roll(-1, 0))
        if not starts:
            raise ValueError("Physical road collision regions are empty")
        self.start, self.end = torch.cat(starts), torch.cat(ends)
        valid = torch.linalg.vector_norm(self.end-self.start, dim=-1) > 1e-9
        self.start, self.end = self.start[valid], self.end[valid]

    @classmethod
    def from_asset(cls, filename, device="cpu"):
        path = Path(filename)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1]/path
        asset = json.loads(path.read_text())
        if asset.get("coordinates") != "simulation_world":
            raise ValueError("Road collision asset must use simulation world coordinates")
        return cls(asset["collision_regions"], device=device)

    @staticmethod
    def _cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]

    @torch.no_grad()
    def __call__(self, vertices):
        original_shape = vertices.shape[:-2]
        rectangles = vertices[..., :4, :].reshape(-1, 4, 2)
        if not rectangles.numel():
            return torch.zeros(original_shape, dtype=torch.bool, device=vertices.device)
        return torch.cat([self._check(chunk) for chunk in rectangles.split(128)]).reshape(original_shape)

    def _check(self, corners):
        a, b = self.start, self.end
        x, y = corners[..., 0, None], corners[..., 1, None]
        dy = b[:, 1]-a[:, 1]
        safe_dy = torch.where(dy.abs() > 1e-12, dy, torch.ones_like(dy))
        x_at_y = a[:, 0]+(y-a[:, 1])*(b[:, 0]-a[:, 0])/safe_dy
        crossings = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < x_at_y)
        inside = crossings.sum(-1).remainder(2).bool()

        # Exact rectangle-edge/road-edge intersections catch an island or
        # notch crossing the body between corners, which sampling would miss.
        u = corners.roll(-1, 1)-corners
        v = b-a
        delta = a[None, None]-corners[:, :, None]
        denominator = self._cross(u[:, :, None], v)
        safe = torch.where(denominator.abs() > 1e-12, denominator, torch.ones_like(denominator))
        t = self._cross(delta, v)/safe
        s = self._cross(delta, u[:, :, None])/safe
        intersects = ((denominator.abs() > 1e-12) & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)).any(-1).any(-1)

        # Also catch a small island entirely enclosed by the vehicle body.
        signs = self._cross(u[:, :, None], delta)
        boundary_inside_body = ((signs >= -1e-9).all(1) | (signs <= 1e-9).all(1)).any(-1)
        return ~inside.all(-1) | intersects | boundary_inside_body
