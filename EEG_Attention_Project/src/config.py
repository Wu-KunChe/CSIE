"""Central YAML configuration loading and path resolution."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f'Expected a YAML mapping at {path}')
    return obj


def load_project_config(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    return {
        'project_root': root,
        'paths': read_yaml(root / 'configs' / 'paths.yaml'),
        'baseline': read_yaml(root / 'configs' / 'baseline.yaml'),
        'channels': read_yaml(root / 'configs' / 'channels.yaml'),
    }


def resolve_project_path(project_root: str | Path, path: str | Path) -> Path:
    """Resolve local relative project paths; absolute Windows drives stay literal on Windows."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(project_root) / p
