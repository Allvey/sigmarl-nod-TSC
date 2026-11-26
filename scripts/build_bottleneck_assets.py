"""Build a straight, divided 2 -> 1 -> 2 road without image tracing.

Lanelets, pavement, collision rings and SVG share one analytic design.
Only offline geometry/preview generation is performed; no policy is run.
"""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NAME = "bottleneck_1"
DESIGN_PATH = ROOT / "assets/maps" / (NAME + ".design.json")


def load_design(filename=DESIGN_PATH):
    design = json.loads(Path(filename).read_text())
    for key in ("lane_width", "median_width", "approach_length", "taper_length",
                "throat_length", "exit_length", "world_margin", "world_height",
                "terminal_pavement_extension", "sample_spacing"):
        if not math.isfinite(design[key]) or design[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if design["lane_width"] < 2 * design["vehicle_size"][1]:
        raise ValueError("Lane width must leave room around the real vehicle")
    if design["world_height"] <= 4 * design["lane_width"] + design["median_width"]:
        raise ValueError("Road does not fit the world height")
    if design["terminal_pavement_extension"] >= design["world_margin"]:
        raise ValueError("Terminal pavement must stay inside the world")
    if design["n_agents"] < 1 or int(design["n_agents"]) != design["n_agents"]:
        raise ValueError("n_agents must be a positive integer")
    return design


def smoothstep(t):
    """C2 transition: zero first/second derivative at both ends."""
    t = np.clip(t, 0., 1.)
    return t**3 * (10. - 15.*t + 6.*t*t)


def stations(design):
    return np.cumsum([design["world_margin"], design["approach_length"],
                      design["taper_length"], design["throat_length"],
                      design["taper_length"], design["exit_length"]])


def outer_lane_fraction(progress, design):
    """Extra pavement width in units of one lane, in the driving direction."""
    a = design["approach_length"]
    b = a + design["taper_length"]
    c = b + design["throat_length"]
    return (1. - smoothstep((np.asarray(progress) - a) / design["taper_length"])
            + smoothstep((np.asarray(progress) - c) / design["taper_length"]))


def sample_curve(start, end, function, spacing):
    parameter = np.linspace(start, end, 2001)
    points = function(parameter)
    arc = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    target = np.linspace(0., arc[-1], max(3, math.ceil(arc[-1] / spacing) + 1))
    return np.column_stack([np.interp(target, arc, points[:, axis]) for axis in (0, 1)])


def build_geometry(design):
    x = stations(design)
    width, median = design["lane_width"], design["median_width"]
    center = design["world_height"] / 2
    ways, descriptions, routes, metadata = {}, {}, [], []

    def add(points, direction, lane, kind):
        key = str(len(ways) + 1)
        ways[key] = points
        descriptions[key] = dict(direction=direction, lane=lane, kind=kind)
        return key

    # Progress increases in the travel direction. Upper carriageway is the
    # exact 180-degree rotation of the lower one, not a shared oncoming lane.
    for direction in ("eastbound", "westbound"):
        east = direction == "eastbound"

        def curve(progress, lane):
            distance = median/2 + width/2 + lane*width*outer_lane_fraction(progress, design)
            return np.column_stack((x[0] + progress if east else x[-1] - progress,
                                    center - distance if east else center + distance))

        lengths = np.cumsum([0., design["approach_length"], design["taper_length"],
                             design["throat_length"], design["taper_length"], design["exit_length"]])
        segment = {}
        for i, kind in enumerate(("entry", "merge", "throat", "split", "exit")):
            for lane in ((0,) if kind == "throat" else (0, 1)):
                points = sample_curve(lengths[i], lengths[i+1], lambda p: curve(p, lane),
                                      design["sample_spacing"])
                segment[kind, lane] = add(points, direction, lane, kind)
        for entry_lane in (0, 1):
            for exit_lane in (0, 1):
                entry = segment["entry", entry_lane]
                # Stagger both lanes on a common safe lattice. Any subset of
                # these slots meets the existing 1.2 * body diagonal rule;
                # random insertion cannot jam the remaining empty spaces.
                spacing = float(np.linalg.norm(ways[entry][1]-ways[entry][0]))
                separation = 1.2*math.hypot(*design["vehicle_size"])
                half_period = max(separation/2, math.sqrt(max(0., separation**2-width**2)))
                step = 2*math.ceil((half_period+1e-5)/spacing)
                start, stop = 4, len(ways[entry])-6
                spawn_indices = list(range(start+entry_lane*(step//2), stop, step))
                routes.append([entry, segment["merge", entry_lane], segment["throat", 0],
                               segment["split", exit_lane], segment["exit", exit_lane]])
                metadata.append(dict(direction=direction, origin="west" if east else "east",
                                     destination="east" if east else "west",
                                     entry_lane=entry_lane, exit_lane=exit_lane,
                                     spawn_start_index=start, spawn_end_index=stop,
                                     spawn_point_indices=spawn_indices))
    # Same-direction traffic on both entry lanes must see one another before
    # the merge. Opposite carriageways are physically separated throughout.
    neighbors = {key: [other for other, q in ways.items()
                       if descriptions[key]["direction"] == descriptions[other]["direction"]
                       and np.linalg.norm(p[:, None]-q[None, :], axis=-1).min() < .8]
                 for key, p in ways.items()}
    dimensions = np.array([x[-1] + design["world_margin"], design["world_height"]])
    return ways, descriptions, routes, metadata, neighbors, dimensions


def road_sections(design):
    x = stations(design)
    end_pad = design["terminal_pavement_extension"]
    # Include all transition junctions exactly, plus terminal body overhang.
    samples = np.unique(np.r_[np.linspace(x[0]-end_pad, x[-1]+end_pad, 241), x])
    middle = design["world_height"]/2
    width, median = design["lane_width"], design["median_width"]
    result = []
    for east in (True, False):
        progress = samples-x[0] if east else x[-1]-samples
        w = width*(1. + outer_lane_fraction(progress, design))
        inner = np.full_like(samples, middle-median/2 if east else middle+median/2)
        outer = inner-w if east else inner+w
        bottom, top = (outer, inner) if east else (inner, outer)
        polygon = np.vstack((np.column_stack((samples, bottom)),
                             np.column_stack((samples[::-1], top[::-1]))))
        result.append(dict(x=samples, bottom=bottom, top=top, polygon=polygon))
    return result


def body_samples(points, size):
    vec = np.diff(points, axis=0)
    tangent = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
    tangent = np.vstack((tangent, tangent[-1]))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return np.stack([points+s*size[0]/2*tangent+r*size[1]/2*normal
                     for s in np.linspace(-1., 1., 5) for r in np.linspace(-1., 1., 3)])


def validate_geometry(design, ways, descriptions, routes, dimensions):
    x = stations(design)
    max_curvature, minimum_clearance = 0., float("inf")
    road = road_sections(design)
    for route in routes:
        for previous, following in zip(route, route[1:]):
            if not np.allclose(ways[previous][-1], ways[following][0], atol=1e-9):
                raise ValueError("Disconnected reference route")
        points = np.vstack([ways[route[0]]] + [ways[key][1:] for key in route[1:]])
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if not np.isfinite(points).all() or lengths.min() <= .015:
            raise ValueError("Non-finite or duplicate route samples")
        yaw = np.unwrap(np.arctan2(*np.diff(points, axis=0)[:, ::-1].T))
        curvature = np.abs(np.diff(yaw))/((lengths[:-1]+lengths[1:])/2)
        max_curvature = max(max_curvature, float(curvature.max()))
        body = body_samples(points, design["vehicle_size"])
        section = road[0 if descriptions[route[0]]["direction"] == "eastbound" else 1]
        low = np.interp(body[..., 0], section["x"], section["bottom"])
        high = np.interp(body[..., 0], section["x"], section["top"])
        minimum_clearance = min(minimum_clearance, float(np.minimum(body[..., 1]-low,
                                                                    high-body[..., 1]).min()))
        if np.any(body[..., 0] <= section["x"][0]) or np.any(body[..., 0] >= section["x"][-1]):
            raise ValueError("Vehicle extends beyond the terminal pavement")
    if minimum_clearance < .025 or max_curvature > 2.:
        raise ValueError("Road requires more vehicle clearance or gentler tapers")
    if not all(np.all(p > 0.) and np.all(p < dimensions) for p in ways.values()):
        raise ValueError("Route exceeds the viewport")
    return dict(route_count=len(routes), lanelet_count=len(ways),
                max_curvature_per_m=round(max_curvature, 6),
                min_body_vertical_clearance_m=round(minimum_clearance, 6),
                lane_width_m=design["lane_width"], median_width_m=design["median_width"],
                shared_oncoming_lane=False, vehicle_size_m=design["vehicle_size"],
                throat_interval_x=x[2:4].tolist())


def build_artwork(design, dimensions):
    shapes = []

    def polygon(points, role, color):
        shapes.append(dict(kind="polygon", role=role,
                           points=np.asarray(points).round(8).tolist(), color=color, width=1.))

    def ribbon(points, role, color, width=.009):
        points = np.asarray(points)
        tangent = np.gradient(points, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) * width/2
        for i in range(len(points)-1):
            polygon([points[i]-normal[i], points[i+1]-normal[i+1],
                     points[i+1]+normal[i+1], points[i]+normal[i]], role, color)

    green, asphalt, white = [.88, .93, .84], [.40, .42, .44], [.98, .98, .96]
    polygon([[0., 0.], [dimensions[0], 0.], dimensions, [0., dimensions[1]]], "background", green)
    sections = road_sections(design)
    for section in sections:
        x, bottom, top = section["x"], section["bottom"], section["top"]
        for i in range(len(x)-1):
            polygon([[x[i], bottom[i]], [x[i+1], bottom[i+1]],
                     [x[i+1], top[i+1]], [x[i], top[i]]], "road", asphalt)
    x = stations(design)
    cy, median, width = design["world_height"]/2, design["median_width"], design["lane_width"]
    pad = design["terminal_pavement_extension"]
    polygon([[x[0]-pad, cy-median/2], [x[-1]+pad, cy-median/2],
             [x[-1]+pad, cy+median/2], [x[0]-pad, cy+median/2]], "median", [.47, .66, .36])
    for section in sections:
        for side in ("bottom", "top"):
            ribbon(np.column_stack((section["x"], section[side])), "curb", white)
    for side in (-1, 1):
        # The taper is unmarked pavement: no painted sliver is presented as
        # an additional, impossibly narrow lane after the outer lane ends.
        y_divider = cy+side*(median/2+width)
        for start, stop in ((x[0], x[1]-.06), (x[4]+.06, x[5])):
            for a in np.arange(start, stop-.04, .13):
                ribbon([[a, y_divider], [min(a+.07, stop), y_divider]], "lane_divider", white, .007)
        for a, lane in ((x[0]+.6, 0), (x[0]+.6, 1), (x[1]-.3, 0), (x[1]-.3, 1),
                        ((x[2]+x[3])/2, 0), (x[4]+.5, 0), (x[4]+.5, 1),
                        (x[-1]-.45, 0), (x[-1]-.45, 1)):
            y = cy+side*(median/2+width/2+lane*width)
            direction = -side  # lower moves east, upper moves west
            arrow = np.array([[-.065, -.008], [.025, -.008], [.025, -.023],
                              [.065, 0.], [.025, .023], [.025, .008], [-.065, .008]])
            arrow[:, 0] *= direction
            polygon(arrow+[a, y], "direction_arrow", white)
    return dict(schema_version=1, coordinates="simulation_world", shapes=shapes), sections


def write_svg(render, dimensions, filename):
    scale = 170.
    root = ET.Element("svg", xmlns="http://www.w3.org/2000/svg",
                      viewBox=f"0 0 {dimensions[0]*scale:.3f} {dimensions[1]*scale:.3f}")
    layers = {}
    for shape in render["shapes"]:
        role = shape["role"]
        if role not in layers:
            layers[role] = ET.SubElement(root, "g", id=role)
        points = [(p[0]*scale, (dimensions[1]-p[1])*scale) for p in shape["points"]]
        color = "#"+"".join(f"{round(c*255):02x}" for c in shape["color"])
        ET.SubElement(layers[role], "polygon", points=" ".join(f"{a:.3f},{b:.3f}" for a, b in points), fill=color)
    ET.indent(root)
    ET.ElementTree(root).write(filename, encoding="utf-8", xml_declaration=True)


def write_assets(output_dir=ROOT/"assets/maps", design_path=DESIGN_PATH, preview=None):
    design = load_design(design_path)
    ways, descriptions, routes, metadata, neighbors, dimensions = build_geometry(design)
    checks = validate_geometry(design, ways, descriptions, routes, dimensions)
    spawn_points = np.unique(np.vstack([ways[r[0]][m["spawn_point_indices"]]
                                       for r, m in zip(routes, metadata)]), axis=0)
    separation = np.linalg.norm(spawn_points[:, None]-spawn_points[None, :], axis=-1)
    np.fill_diagonal(separation, np.inf)
    if separation.min() < 1.2*math.hypot(*design["vehicle_size"]) or len(spawn_points) < design["n_agents"]:
        raise ValueError("Entry spawn slots do not fit the requested traffic: extend approaches or reduce n_agents")
    checks["unique_spawn_slots"] = len(spawn_points)
    checks["min_spawn_center_distance_m"] = round(float(separation.min()), 6)
    render, sections = build_artwork(design, dimensions)
    aspect = design["viewer_size"][0]/design["viewer_size"][1]
    scene = dict(map_path=f"assets/maps/{NAME}.osm", n_agents=design["n_agents"],
                 name="Straight divided two-to-one-to-two bottleneck", reference_paths_ids=routes,
                 neighboring_lanelet_ids=neighbors, figsize_x=6.,
                 viewer_zoom=math.sqrt(max(dimensions[0]/(2*aspect), dimensions[1]/2))*1.02,
                 viewer_size=design["viewer_size"], compact_hud=True, lane_width=design["lane_width"],
                 lane_collision_margin=0., road_collision_mode="road_edges", scale=1e5,
                 map_margin=design["world_margin"], world_x_dim=float(dimensions[0]),
                 world_y_dim=float(dimensions[1]), render_asset_path=f"assets/maps/{NAME}.render.json",
                 geometry_asset_path=f"assets/maps/{NAME}.boundaries.json", route_metadata=metadata)
    meta = dict(schema_version=1, scene=scene, lanelets=descriptions, geometry_checks=checks,
                design_asset_path=f"assets/maps/{NAME}.design.json",
                note="Approved schematic implemented analytically; no surveyed-road or policy-performance claim.")
    boundaries = dict(schema_version=1, coordinates="simulation_world", geometry_source="analytic_divided_bottleneck",
                      lane_width=design["lane_width"], median_width=design["median_width"],
                      collision_margin_m=0.,
                      collision_regions=[[section["polygon"].round(8).tolist()] for section in sections])
    osm = ET.Element("osm", version="0.6", generator="sigmarl-nod bottleneck asset builder")
    nodes = {}

    def node(point):
        key = tuple(np.round(point, 9))
        if key not in nodes:
            nodes[key] = len(nodes)+1
            ET.SubElement(osm, "node", id=str(nodes[key]), lat=f"{50+point[0]/1e5:.13f}",
                          lon=f"{6+point[1]/1e5:.13f}")
        return str(nodes[key])

    # Same coordinate origin as the renderer, including unused HUD margins.
    margin = design["world_margin"]
    node([margin, margin])
    node(dimensions-margin)
    for points in ways.values():
        for point in points:
            node(point)
    for key, points in ways.items():
        way = ET.SubElement(osm, "way", id=key)
        for point in points:
            ET.SubElement(way, "nd", ref=node(point))
        ET.SubElement(way, "tag", k="lanes", v=key)
        ET.SubElement(way, "tag", k="highway", v="service")
    ET.indent(osm)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(osm).write(output_dir/(NAME+".osm"), encoding="utf-8", xml_declaration=True)
    for suffix, data in ((".json", meta), (".render.json", render), (".boundaries.json", boundaries)):
        (output_dir/(NAME+suffix)).write_text(json.dumps(data, indent=2 if suffix == ".json" else None)+"\n")
    write_svg(render, dimensions, output_dir/(NAME+".svg"))
    if preview:
        write_preview(render, design, ways, descriptions, dimensions, preview)
    return meta


def write_preview(render, design, ways, descriptions, dimensions, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    ax.add_collection(PolyCollection([s["points"] for s in render["shapes"]],
                                    facecolors=[s["color"] for s in render["shapes"]],
                                    edgecolors="none", antialiaseds=False))
    for key, points in ways.items():
        if descriptions[key]["kind"] not in ("entry", "throat", "exit"):
            continue
        j = len(points)//2
        body = body_samples(points, design["vehicle_size"])
        # Corners of a real-size body, not oversized illustration cars.
        corners = body[[0, 12, 14, 2], j]
        color = "#1679b4" if descriptions[key]["direction"] == "eastbound" else "#eeaa35"
        ax.add_patch(Polygon(corners, facecolor=color, edgecolor="#1e293b", linewidth=.6))
    x = stations(design)
    for a, label in (((x[0]+x[1])/2, "2 lanes / direction"),
                     ((x[2]+x[3])/2, "1 lane / direction"),
                     ((x[4]+x[5])/2, "2 lanes / direction")):
        ax.text(a, dimensions[1]/2+.72, label, ha="center", fontsize=12)
    ax.text(dimensions[0]/2, dimensions[1]-.4, "DIVIDED BOTTLENECK: 2 -> 1 -> 2", ha="center", fontsize=17)
    ax.text(dimensions[0]/2, .5, "Static geometry preview | lane 0.18 m | vehicle 0.16 x 0.08 m\n"
            "White dashes are passable; curbs and the median are physical boundaries.",
            ha="center", fontsize=10)
    ax.set(xlim=(0., dimensions[0]), ylim=(0., dimensions[1]), aspect="equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"assets/maps")
    parser.add_argument("--design", type=Path, default=DESIGN_PATH)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    meta = write_assets(args.output_dir, args.design, args.preview)
    print(f"Built {NAME}: {len(meta['lanelets'])} lanelets, "
          f"{len(meta['scene']['reference_paths_ids'])} routes; {meta['geometry_checks']}")


if __name__ == "__main__":
    main()
