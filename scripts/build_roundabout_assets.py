"""Build native road assets from a width-constrained curved road design.

Aerial landmarks guide the spines. Road edges, markings and OSM lanelets
share parallel curves; the superseded generated diagram is not geometry input.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import cv2

from scripts.roundabout_vectorization import contour_groups, trapezoids
from scripts.roundabout_width_design import (
    ROOT, NAME, TRACE, REFERENCE, WIDTH, PIXEL_SCALE, MARGIN, SPACING,
    RASTER_SCALE, SPINE_AXES, RADII, ISLAND_RADIUS, world, pixels,
    build_geometry, footprint_corners, clearance_values, design_layers,
    carriageways, ring_frame, validate_design,
)


def build_render_asset(ways, descriptions, center, dimensions):
    road, layers = design_layers()
    shapes = [dict(kind="polygon", role="background", points=[[0., 0.], [float(dimensions[0]), 0.],
                  dimensions.tolist(), [0., float(dimensions[1])]], color=[1., 1., 1.], width=1.)]
    contours_by_role = {}
    for role, mask, color, minimum_area in layers:
        groups = contour_groups(mask, minimum_area*RASTER_SCALE**2)
        contours_by_role[role] = [[world(r/RASTER_SCALE).round(7).tolist() for r in rings] for rings in groups]
        for rings in groups:
            for tile in trapezoids(rings):
                shapes.append(dict(kind="polygon", role=role, points=world(tile/RASTER_SCALE).round(7).tolist(),
                                   color=color, width=1.))
    render = dict(schema_version=1, coordinates="simulation_world", shapes=shapes)
    boundaries = dict(schema_version=2, coordinates="simulation_world", image_size=TRACE["image_size"],
                      image_to_world=[[PIXEL_SCALE, 0., MARGIN],
                                      [0., -PIXEL_SCALE, MARGIN+TRACE["image_size"][1]*PIXEL_SCALE], [0., 0., 1.]],
                      reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                      geometry_source="analytic_width_constrained_spines", lane_width=WIDTH,
                      raster_scale=RASTER_SCALE, contour_tolerance_pixels=1/RASTER_SCALE, layers=contours_by_role)
    # Expand physical pavement for curb-contact tolerance. Lane paint is not
    # included: crossing an internal white dash must never count as a crash.
    margin = TRACE.get("lane_collision_margin", 0.)
    if margin < 0 or margin >= WIDTH/2:
        raise ValueError("Road contact margin must be non-negative and below half a lane width")
    radius = round(margin/PIXEL_SCALE*RASTER_SCALE)
    if radius:
        road = cv2.dilate(road, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
    groups = contour_groups(road, 20.*RASTER_SCALE**2)
    boundaries["collision_regions"] = [[world(r/RASTER_SCALE).round(7).tolist()
                                        for r in rings] for rings in groups]
    boundaries["collision_margin_m"] = margin
    return render, boundaries


def write_svg(boundaries, render, filename):
    """An editable vector preview; use even-odd paths to preserve all holes."""
    w, h = TRACE["image_size"]
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", viewBox=f"0 0 {w} {h}")
    ET.SubElement(svg, "rect", width=str(w), height=str(h), fill="white")
    colors = {item["role"]: item["color"] for item in render["shapes"]}
    for role, groups in boundaries["layers"].items():
        if not groups:
            continue
        layer = ET.SubElement(svg, "g", id=role)
        color = "#"+"".join(f"{int(round(channel*255)):02x}" for channel in colors[role])
        for rings in groups:
            commands = []
            for ring in rings:
                p = pixels(ring)
                commands.append("M "+" L ".join(f"{x:.3f},{y:.3f}" for x, y in p)+" Z")
            ET.SubElement(layer, "path", d=" ".join(commands), fill=color, attrib={"fill-rule": "evenodd"})
    ET.indent(svg)
    ET.ElementTree(svg).write(filename, encoding="utf-8", xml_declaration=True)


def write_assets(output_dir, preview=None, route_preview=None, vehicle_preview=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ways, descriptions, routes, info, neighbors, center, dims = build_geometry()
    checks = validate_design(ways, routes)
    root = ET.Element("osm", version="0.6", generator="sigmarl-nod width-constrained road builder")
    nodes = {}

    def node(point):
        key = tuple(np.round(point, 9))
        if key not in nodes:
            nodes[key] = len(nodes)+1
            ET.SubElement(root, "node", id=str(nodes[key]), lat=f"{50+point[0]/1e5:.13f}",
                          lon=f"{6+point[1]/1e5:.13f}")
        return str(nodes[key])

    # Unreferenced coordinate anchors align the parser to the diagram frame,
    # including margins not occupied by a route. They are not drivable ways.
    node([MARGIN, MARGIN])
    node(dims-MARGIN)
    for points in ways.values():
        for point in points:
            node(point)
    for key, points in ways.items():
        way = ET.SubElement(root, "way", id=key)
        for point in points:
            ET.SubElement(way, "nd", ref=node(point))
        ET.SubElement(way, "tag", k="lanes", v=key)
        ET.SubElement(way, "tag", k="highway", v="service")
    ET.indent(root)
    ET.ElementTree(root).write(output_dir/(NAME+".osm"), encoding="utf-8", xml_declaration=True)
    scene = dict(map_path=f"assets/maps/{NAME}.osm", n_agents=16,
                 name="Width-constrained curved four-arm roundabout", reference_paths_ids=routes,
                 neighboring_lanelet_ids=neighbors, figsize_x=4.,
                 viewer_zoom=math.sqrt(float(max(dims))/2)*1.02,
                 viewer_size=TRACE["viewer_size"], compact_hud=True, lane_width=WIDTH,
                 lane_collision_margin=TRACE.get("lane_collision_margin", 0.),
                 road_collision_mode=TRACE.get("road_collision_mode", "reference_lane"),
                 scale=1e5, map_margin=MARGIN, world_x_dim=float(dims[0]), world_y_dim=float(dims[1]),
                 render_asset_path=f"assets/maps/{NAME}.render.json", route_metadata=info,
                 geometry_asset_path=f"assets/maps/{NAME}.boundaries.json")
    meta = dict(schema_version=2, scene=scene, lanelets=descriptions, ring_center=center.tolist(),
                ring_radii=RADII.tolist(), ring_spine_axes=SPINE_AXES.tolist(),
                reference_trace=f"assets/maps/{NAME}.trace.json",
                reference_image=TRACE["reference_image"],
                geometry_checks=checks,
                note="Aerial-inspired width-constrained redesign; markings, curbs and routes share parallel geometry. Not surveyed reconstruction.")
    render, boundaries = build_render_asset(ways, descriptions, center, dims)
    (output_dir/(NAME+".json")).write_text(json.dumps(meta, indent=2)+"\n")
    for suffix, data in ((".render.json", render), (".boundaries.json", boundaries)):
        (output_dir/(NAME+suffix)).write_text(json.dumps(data, separators=(",", ":"))+"\n")
    write_svg(boundaries, render, output_dir/(NAME+".svg"))
    if preview or route_preview or vehicle_preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.add_collection(PolyCollection([s["points"] for s in render["shapes"]],
                                        facecolors=[s["color"] for s in render["shapes"]],
                                        edgecolors="none", antialiaseds=False))
        ax.set(xlim=(0, dims[0]), ylim=(0, dims[1]), aspect="equal")
        ax.axis("off")
        fig.tight_layout(pad=0)
        if preview:
            Path(preview).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview, dpi=150)
        if vehicle_preview:
            # Static true-size vehicles in both lanes at the same progress.
            # This is a geometric illustration, never a simulated trajectory.
            from matplotlib.patches import Polygon
            sections = [value for (_, kind), value in carriageways().items() if kind == "entry"]
            sections.append(ring_frame(np.linspace(0, 2*np.pi, 1001)))
            for section_id, (p, t, n) in enumerate(sections):
                samples = [300] if section_id < 4 else [140, 390, 640, 890]
                for j in samples:
                    for lane, offset in enumerate((WIDTH/2, -WIDTH/2)):
                        c = p[j]+offset*n[j]
                        half_length, half_width = np.asarray(TRACE["vehicle_size"])/2
                        corners = np.array([c+s*half_length*t[j]+r*half_width*n[j]
                                            for s, r in ((-1,-1),(1,-1),(1,1),(-1,1))])
                        ax.add_patch(Polygon(corners, facecolor=("#1565c0", "#f9c74f")[lane],
                                             edgecolor="#333333", linewidth=.65, zorder=5))
            ax.text(.02, .98, "STATIC SIZE CHECK\nLane: 0.18 m   Vehicle: 0.16 x 0.08 m",
                    transform=ax.transAxes, va="top", fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
            Path(vehicle_preview).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(vehicle_preview, dpi=150)
        if route_preview:
            for key, p in ways.items():
                ax.plot(p[:, 0], p[:, 1], color=("#2962ff", "#a112e8")[descriptions[key]["lane"]],
                        lw=.85, alpha=.85)
            Path(route_preview).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(route_preview, dpi=150)
        plt.close(fig)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"assets/maps")
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--route-preview", type=Path)
    parser.add_argument("--vehicle-preview", type=Path)
    args = parser.parse_args()
    meta = write_assets(args.output_dir, args.preview, args.route_preview, args.vehicle_preview)
    print(f"Built {NAME}: {len(meta['lanelets'])} lanelets, "
          f"{len(meta['scene']['reference_paths_ids'])} routes in {args.output_dir}")


if __name__ == "__main__":
    main()
