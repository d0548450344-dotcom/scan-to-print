"""Phase 0 accuracy check: compare scanned dimensions with caliper measurements.

Input CSV columns: object_id, mesh_path, dim1_mm, dim2_mm, dim3_mm
(the object's three caliper dimensions, in any order). The longest caliper
dimension sets the scale, so only the two remaining dimensions are evaluated:
the scaled one is correct by construction and would inflate the pass rate.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .prepare import PrepError, load_mesh, principal_extents, repair, scale_to_known

# Go/no-go from docs/feasibility.md
TOLERANCE_MM_PER_100MM = 1.0
REQUIRED_PASS_RATIO = 0.7


@dataclass
class DimResult:
    object_id: str
    dim_rank: int  # 1 = second-longest, 2 = shortest
    reference_mm: float
    scanned_mm: float
    error_mm: float
    allowed_mm: float
    passed: bool


@dataclass
class ObjectResult:
    object_id: str
    passed: bool
    watertight: bool
    error: str = ""


def allowed_error(reference_mm: float, tol_per_100: float = TOLERANCE_MM_PER_100MM) -> float:
    return tol_per_100 * reference_mm / 100.0


def evaluate_extents(object_id: str, scanned_raw: np.ndarray, reference: list[float],
                     tol_per_100: float = TOLERANCE_MM_PER_100MM) -> list[DimResult]:
    """Scale raw extents by the longest reference dimension and compare the other two."""
    ref = np.sort(np.asarray(reference, dtype=float))[::-1]
    raw = np.sort(np.asarray(scanned_raw, dtype=float))[::-1]
    if ref.shape != (3,) or raw.shape != (3,) or np.any(ref <= 0) or raw[0] <= 0:
        raise ValueError(f"{object_id}: need three positive reference and scanned dimensions")
    scanned = raw * (ref[0] / raw[0])
    results = []
    for i in (1, 2):
        err = float(scanned[i] - ref[i])
        allowed = allowed_error(ref[i], tol_per_100)
        results.append(DimResult(object_id, i, float(ref[i]), float(scanned[i]), err, allowed, abs(err) <= allowed))
    return results


def evaluate_batch(csv_path: str | Path, report_path: str | Path | None = None) -> dict:
    """Run repair + evaluation for every row; return a summary with the go/no-go verdict."""
    csv_path = Path(csv_path)
    dims: list[DimResult] = []
    objects: list[ObjectResult] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            oid = row["object_id"]
            mesh_path = (csv_path.parent / row["mesh_path"]).resolve()
            try:
                mesh, rep = repair(load_mesh(mesh_path))
                ref = [float(row[k]) for k in ("dim1_mm", "dim2_mm", "dim3_mm")]
                res = evaluate_extents(oid, principal_extents(mesh), ref)
                dims.extend(res)
                objects.append(ObjectResult(oid, all(r.passed for r in res), rep.watertight_after))
            except (PrepError, ValueError, KeyError, OSError) as e:
                # A failed scan counts as a failed object, not a skipped one.
                objects.append(ObjectResult(oid, False, False, str(e)))

    n = len(objects)
    n_pass = sum(o.passed for o in objects)
    summary = {
        "objects": n,
        "passed": n_pass,
        "pass_ratio": (n_pass / n) if n else 0.0,
        "required_ratio": REQUIRED_PASS_RATIO,
        "tolerance_mm_per_100mm": TOLERANCE_MM_PER_100MM,
        "go": n > 0 and n_pass / n >= REQUIRED_PASS_RATIO,
        "failed_objects": [asdict(o) for o in objects if not o.passed],
    }
    if report_path:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with rp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(DimResult.__dataclass_fields__))
            w.writeheader()
            for d in dims:
                w.writerow(asdict(d))
    return summary
