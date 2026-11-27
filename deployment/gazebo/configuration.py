"""Load experiment profiles with explicit, relative JSON inheritance."""
import copy
import json
from pathlib import Path


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path, visited=None):
    path = Path(path).resolve()
    visited = set() if visited is None else visited
    if path in visited:
        raise ValueError("Circular experiment profile inheritance")
    visited.add(path)
    config = json.loads(path.read_text())
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, visited), config)
    return config
