"""Vehicle-footprint distances and containment for a route's own lane corridor.

Boundaries follow route direction. Left plus reversed right also describes a
closed-loop corridor: the two connecting seam segments cancel in the ray test.
"""
import torch


def point_segment_distance(points, line):
    delta = line[1:] - line[:-1]
    u = ((points[..., None, :] - line[:-1]) * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-12)
    projection = line[:-1] + u.clamp(0, 1)[..., None] * delta
    return (points[..., None, :] - projection).norm(dim=-1)


def side_clearance(vertices, line, prepared=None, frame=None):
    # Segment distance is attained at an endpoint unless the segments intersect.
    if prepared is None:
        a = point_segment_distance(vertices[:, :4], line).amin(dim=(-1, -2))
    else:
        delta, length2 = prepared
        u = ((vertices[:, :4, None] - line[:-1]) * delta).sum(-1) / length2
        projection = line[:-1] + u.clamp(0, 1)[..., None] * delta
        a = (vertices[:, :4, None] - projection).norm(dim=-1).amin(dim=(-1, -2))
    if frame is None:
        edges = vertices[:, 1:] - vertices[:, :-1]
        edge_length2 = edges.square().sum(-1)[:, None].clamp_min(1e-12)
        origin = vertices[:, 0]
        basis = edges[:, :2] / edges[:, :2].norm(dim=-1, keepdim=True)
        extent = edges[:, :2].norm(dim=-1)[:, None]
    else:
        edges, edge_length2, origin, basis, extent = frame
    diff = line[None, :, None, :] - vertices[:, None, :-1, :]
    u = (diff * edges[:, None]).sum(-1) / edge_length2
    b = (diff - u.clamp(0, 1)[..., None] * edges[:, None]).norm(dim=-1).amin(dim=(-1, -2))
    starts = torch.einsum('bsi,bji->bsj', line[None, :-1] - origin[:, None], basis)
    ends = torch.einsum('bsi,bji->bsj', line[None, 1:] - origin[:, None], basis)
    delta = ends - starts
    parallel = delta.abs() < 1e-10
    denom = torch.where(parallel, torch.ones_like(delta), delta)
    t1, t2 = -starts / denom, (extent - starts) / denom
    enter = torch.where(parallel, -torch.inf, torch.minimum(t1, t2)).amax(-1).clamp_min(0)
    leave = torch.where(parallel, torch.inf, torch.maximum(t1, t2)).amin(-1).clamp_max(1)
    outside = (parallel & ((starts < -1e-7) | (starts > extent + 1e-7))).any(-1)
    intersects = ((enter <= leave + 1e-7) & ~outside).any(-1)
    return torch.where(intersects, torch.zeros_like(a), torch.minimum(a, b)), intersects


def corridor_geometry(vertices, left, right, is_loop=False):
    """Return unsigned left/right footprint distances [B,2], violation [B].

    Containment detects a footprint entirely outside, even without contact.
    Safety callers must use the violation flag as well as these distances.
    """
    if not is_loop:
        # The route's entry/exit is not a lateral wall. Extend side boundaries
        # beyond a vehicle length so crossing the goal is not a lane violation.
        extension = 2 * (vertices[:, 1:] - vertices[:, :-1]).norm(dim=-1).amax()
        def extend(line):
            delta = line[1:] - line[:-1]
            delta = delta[delta.norm(dim=-1) > 1e-8]
            if not len(delta):
                raise ValueError('Navigation boundary has no nonzero segment')
            start, end = delta[0] / delta[0].norm(), delta[-1] / delta[-1].norm()
            return torch.cat(((line[0] - extension * start)[None], line,
                              (line[-1] + extension * end)[None]))
        left, right = extend(left), extend(right)
    distances, touches = zip(*(side_clearance(vertices, line) for line in (left, right)))
    polygon = torch.cat((left, right.flip(0), left[:1]))
    a, b = polygon[:-1], polygon[1:]
    points = vertices[:, :4, None]
    y = points[..., 1]
    straddles = (a[:, 1] > y) != (b[:, 1] > y)
    dy = b[:, 1] - a[:, 1]
    denominator = torch.where(dy.abs() < 1e-12, torch.ones_like(dy), dy)
    crossing_x = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / denominator
    inside = ((straddles & (points[..., 0] < crossing_x)).sum(-1) % 2) == 1
    violation = ~inside.all(-1) | touches[0] | touches[1]
    return torch.stack(distances, -1), violation


def prepare_corridor(left, right, is_loop, extension):
    """Precompute only map/vehicle-shape constants, once per assigned route."""
    if not is_loop:
        def extend(line):
            delta = line[1:] - line[:-1]
            delta = delta[delta.norm(dim=-1) > 1e-8]
            if not len(delta):
                raise ValueError('Navigation boundary has no nonzero segment')
            start, end = delta[0] / delta[0].norm(), delta[-1] / delta[-1].norm()
            return torch.cat(((line[0] - extension * start)[None], line,
                              (line[-1] + extension * end)[None]))
        left, right = extend(left), extend(right)
    sides = []
    for line in (left, right):
        delta = line[1:] - line[:-1]
        sides.append((line, (delta, delta.square().sum(-1).clamp_min(1e-12))))
    polygon = torch.cat((left, right.flip(0), left[:1]))
    a, b = polygon[:-1], polygon[1:]
    dy = b[:, 1] - a[:, 1]
    denominator = torch.where(dy.abs() < 1e-12, torch.ones_like(dy), dy)
    return sides, (a, b, b[:, 0] - a[:, 0], denominator)


def prepared_geometry(vertices, prepared):
    sides, (a, b, dx, denominator) = prepared
    edges = vertices[:, 1:] - vertices[:, :-1]
    edge_length2 = edges.square().sum(-1)[:, None].clamp_min(1e-12)
    lengths = edges[:, :2].norm(dim=-1, keepdim=True)
    frame = edges, edge_length2, vertices[:, 0], edges[:, :2] / lengths, lengths.squeeze(-1)[:, None]
    distances, touches = zip(*(side_clearance(vertices, line, static, frame) for line, static in sides))
    points = vertices[:, :4, None]
    y = points[..., 1]
    straddles = (a[:, 1] > y) != (b[:, 1] > y)
    crossing_x = a[:, 0] + (y - a[:, 1]) * dx / denominator
    inside = ((straddles & (points[..., 0] < crossing_x)).sum(-1) % 2) == 1
    return torch.stack(distances, -1), ~inside.all(-1) | touches[0] | touches[1]


class NavigationBoundaryCache(dict):
    """Route assignments invalidate grouping only at reset, never each step.

    Same-route cars across all environments/agent slots share one geometry call.
    Rigid vehicle dimensions fix the open-end extension; no states or RNG are
    cached. Map polylines are immutable for the lifetime of the environment.
    """
    def __init__(self, batch_dim, n_agents, extension, device):
        super().__init__()
        self.batch_dim, self.n_agents = batch_dim, n_agents
        self.extension, self.device = extension, device
        self.prepared = {}
        self.groups = None

    def __setitem__(self, key, route):
        if self.get(key) is route:
            return
        super().__setitem__(key, route)
        self.groups = None
        if id(route) not in self.prepared:
            # Retain the route itself to prevent id reuse following reassignment.
            self.prepared[id(route)] = (route, prepare_corridor(
                route['left_boundary'], route['right_boundary'],
                route.get('is_loop', False), self.extension))

    def evaluate(self, vertices):
        if self.groups is None:
            indices = {}
            for e in range(self.batch_dim):
                for i in range(self.n_agents):
                    indices.setdefault(id(self[e, i]), []).append(e * self.n_agents + i)
            self.groups = [(torch.tensor(ids, device=self.device), self.prepared[key][1])
                           for key, ids in indices.items()]
        flat = vertices.reshape(-1, 5, 2)
        distances = flat.new_empty((len(flat), 2))
        violations = torch.empty(len(flat), device=flat.device, dtype=torch.bool)
        for indices, prepared in self.groups:
            d, v = prepared_geometry(flat[indices], prepared)
            distances[indices], violations[indices] = d, v
        return (distances.reshape(self.batch_dim, self.n_agents, 2),
                violations.reshape(self.batch_dim, self.n_agents))
