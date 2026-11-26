"""Static JSON road artwork, shared by the offline preview and VMAS rendering."""
import json
import textwrap
from pathlib import Path


def overlay_text_rows(lines, width, font_size):
    """Wrap pixel-space HUD text, including long diagnostic suffixes.

    One font-size per character leaves room beyond Courier's usual pixel
    advance. Preserve every field instead of cutting the end of a line off.
    """
    columns = max(12, int(width / font_size))
    return [row for line in lines for row in
            (textwrap.wrap(line, width=columns, subsequent_indent="  ") or [""])]


def load_render_asset(filename):
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    with path.open(encoding="utf-8") as stream:
        asset = json.load(stream)
    if asset.get("schema_version") != 1 or asset.get("coordinates") != "simulation_world":
        raise ValueError(f"Unsupported map artwork format: {path}")
    return asset


def static_map_geometries(parser, rendering):
    """Build once per map, retaining existing trajectory/agent overlays above it."""
    if getattr(parser, "render_asset", None) is None:
        return None
    if not hasattr(parser, "_render_asset_geometries"):
        geometries = []
        for item in parser.render_asset["shapes"]:
            if item["kind"] == "polygon":
                geom = rendering.make_polygon(v=item["points"], filled=True, draw_border=False)
            elif item["kind"] == "line":
                geom = rendering.PolyLine(v=item["points"], close=False)
                geom.set_linewidth(item["width"])
            else:
                raise ValueError(f"Unknown map artwork primitive: {item['kind']}")
            geom.set_color(*item["color"])
            geometries.append(geom)
        parser._render_asset_geometries = geometries
    return parser._render_asset_geometries
