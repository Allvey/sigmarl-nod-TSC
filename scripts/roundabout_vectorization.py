"""Vectorize deterministic road-design masks and tessellate polygons with holes.

The geometry source is roundabout_width_design.py, not a generated picture.
OpenCV is used only by the offline asset builder; runtime reads JSON.
"""
import cv2
import numpy as np


def contour_groups(mask, min_area=5., tolerance=1.):
    """Return each outer polygon with its holes, without filling closed strokes."""
    contours, hierarchy = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    groups = []
    for i, contour in enumerate(contours):
        if hierarchy[0, i, 3] != -1 or cv2.contourArea(contour) < min_area:
            continue
        rings = [cv2.approxPolyDP(contour, tolerance, True).reshape(-1, 2).astype(float)]
        child = int(hierarchy[0, i, 2])
        while child != -1:
            if cv2.contourArea(contours[child]) >= min_area:
                rings.append(cv2.approxPolyDP(contours[child], tolerance, True).reshape(-1, 2).astype(float))
            child = int(hierarchy[0, child, 0])
        rings = [r for r in rings if len(r) >= 3]
        if rings:
            groups.append(rings)
    return groups


def trapezoids(rings):
    """Even-odd scan tessellation: convex quads with holes, safe for OpenGL.

    Each strip ends at polygon vertices; linear edges cannot cross within a
    strip. Pair sorted intersections to preserve arbitrary concavities/holes.
    """
    edges = np.vstack([np.stack((r, np.roll(r, -1, axis=0)), axis=1) for r in rings])
    edges = edges[abs(edges[:, 1, 1]-edges[:, 0, 1]) > 1e-9]
    levels = np.unique(np.concatenate([r[:, 1] for r in rings]))
    for y0, y1 in zip(levels, levels[1:]):
        mid = (y0+y1)/2
        e = edges[(np.minimum(edges[:, 0, 1], edges[:, 1, 1]) < mid)
                  & (np.maximum(edges[:, 0, 1], edges[:, 1, 1]) > mid)]
        slope = (e[:, 1, 0]-e[:, 0, 0])/(e[:, 1, 1]-e[:, 0, 1])
        xmid = e[:, 0, 0]+slope*(mid-e[:, 0, 1])
        order = np.argsort(xmid)
        if len(order) % 2:
            raise ValueError("Invalid vector contour intersections")
        lo = e[:, 0, 0]+slope*(y0-e[:, 0, 1])
        hi = e[:, 0, 0]+slope*(y1-e[:, 0, 1])
        for a, b in zip(order[::2], order[1::2]):
            if xmid[b]-xmid[a] > 1e-7:
                yield np.array([[lo[a], y0], [lo[b], y0], [hi[b], y1], [hi[a], y1]])
