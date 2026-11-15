"""Turn a raw reconstruction mesh into a printable one: repair, scale, flat base, export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

SUPPORTED_EXPORTS = {".stl", ".3mf"}


class PrepError(RuntimeError):
    """The mesh cannot be made printable by the automatic steps."""


@dataclass
class RepairReport:
    faces_before: int
    faces_after: int
    components_before: int
    components_removed: int
    watertight_before: bool
    watertight_after: bool
    notes: list[str] = field(default_factory=list)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file; a multi-part scene is merged into one mesh."""
    loaded = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise PrepError(f"{path}: no triangle mesh found")
    return loaded


def repair(mesh: trimesh.Trimesh, min_component_ratio: float = 0.01) -> tuple[trimesh.Trimesh, RepairReport]:
    """Clean a scan mesh.

    Drops floating fragments whose surface area is below `min_component_ratio`
    of the largest part (reconstruction noise), removes degenerate/duplicate faces, fixes
    normals and fills simple holes. Holes that trimesh cannot close leave the
    mesh non-watertight; that is reported, not hidden.
    """
    m = mesh.copy()
    faces_before = len(m.faces)
    watertight_before = m.is_watertight

    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()

    parts = m.split(only_watertight=False)
    components_before = len(parts)
    if components_before == 0:
        raise PrepError("mesh has no faces after cleanup")
    # Surface area, not face count: dense noise blobs can have more faces than a clean flat part.
    largest = max(p.area for p in parts)
    keep = [p for p in parts if p.area >= largest * min_component_ratio]
    removed = components_before - len(keep)
    m = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]

    trimesh.repair.fix_normals(m)
    trimesh.repair.fill_holes(m)
    m.merge_vertices()

    report = RepairReport(
        faces_before=faces_before,
        faces_after=len(m.faces),
        components_before=components_before,
        components_removed=removed,
        watertight_before=watertight_before,
        watertight_after=m.is_watertight,
    )
    if not m.is_watertight:
        report.notes.append("mesh is still not watertight; holes too large or complex for automatic fill")
    if len(keep) > 1:
        report.notes.append(f"{len(keep)} separate parts kept; check they are intended")
    return m, report


def principal_extents(mesh: trimesh.Trimesh) -> np.ndarray:
    """Oriented-bounding-box extents sorted largest first (orientation-independent)."""
    return np.sort(mesh.bounding_box_oriented.primitive.extents)[::-1]


def scale_to_known(mesh: trimesh.Trimesh, known_mm: float, extent_index: int = 0) -> trimesh.Trimesh:
    """Uniformly scale so the chosen principal extent (0 = longest) equals `known_mm`.

    Photogrammetry output has arbitrary units; one real measurement fixes the scale.
    """
    if known_mm <= 0:
        raise ValueError("known_mm must be positive")
    current = principal_extents(mesh)[extent_index]
    if current <= 0:
        raise PrepError("mesh has zero extent along the chosen axis")
    m = mesh.copy()
    m.apply_scale(known_mm / current)
    return m


def flatten_base(mesh: trimesh.Trimesh, cut_mm: float = 0.0) -> trimesh.Trimesh:
    """Rest the mesh on z=0. With `cut_mm` > 0, slice that much off the bottom and cap it.

    Assumes the scan is already upright (z up). Cutting needs a watertight mesh,
    otherwise the cap cannot be built.
    """
    m = mesh.copy()
    m.apply_translation([0, 0, -m.bounds[0][2]])
    if cut_mm > 0:
        if not m.is_watertight:
            raise PrepError("cannot cut a flat base on a non-watertight mesh")
        if cut_mm >= m.extents[2]:
            raise PrepError("cut_mm is not smaller than the mesh height")
        m = m.slice_plane(plane_origin=[0, 0, cut_mm], plane_normal=[0, 0, 1], cap=True)
        m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def export(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Write STL or 3MF. Units are millimetres."""
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_EXPORTS:
        raise ValueError(f"unsupported export format {p.suffix}; use one of {sorted(SUPPORTED_EXPORTS)}")
    p.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(p))
    return p
