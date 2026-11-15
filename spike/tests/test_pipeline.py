import csv

import numpy as np
import pytest
import trimesh

from scan2print import measure, prepare
from scan2print.cli import main


def box(extents=(100.0, 60.0, 40.0)):
    return trimesh.creation.box(extents=extents)


def holed_box_with_debris():
    m = box()
    m.update_faces(np.arange(1, len(m.faces)))  # punch a hole: drop one triangle
    debris = trimesh.creation.icosphere(subdivisions=0, radius=1.0)
    debris.apply_translation([500, 0, 0])
    # second real part: far more faces than the box, smaller area -> must be kept
    big = trimesh.creation.icosphere(subdivisions=4, radius=30)
    big.apply_translation([0, 0, 200])
    return trimesh.util.concatenate([m, big, debris])


def test_repair_fills_hole_and_drops_debris():
    raw = holed_box_with_debris()
    assert not raw.is_watertight
    fixed, rep = prepare.repair(raw)
    assert rep.components_removed == 1
    assert rep.watertight_after
    assert any("separate parts" in n for n in rep.notes)


def test_scale_to_known_uses_principal_extent_regardless_of_rotation():
    m = box((10.0, 6.0, 4.0))
    m.apply_transform(trimesh.transformations.random_rotation_matrix())
    scaled = prepare.scale_to_known(m, 100.0)
    assert np.allclose(prepare.principal_extents(scaled), [100, 60, 40], atol=1e-6)


def test_flatten_base_rests_on_zero_and_cuts():
    m = box()
    flat = prepare.flatten_base(m, cut_mm=5.0)
    assert flat.bounds[0][2] == pytest.approx(0)
    assert flat.extents[2] == pytest.approx(35.0)
    assert flat.is_watertight


def test_flatten_cut_refuses_open_mesh():
    m = box()
    m.update_faces(np.arange(1, len(m.faces)))
    with pytest.raises(prepare.PrepError):
        prepare.flatten_base(m, cut_mm=1.0)


@pytest.mark.parametrize("suffix", [".stl", ".3mf"])
def test_export_roundtrip(tmp_path, suffix):
    out = prepare.export(box(), tmp_path / f"part{suffix}")
    back = prepare.load_mesh(out)
    assert np.allclose(sorted(back.extents), [40, 60, 100], atol=1e-4)


def test_export_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        prepare.export(box(), tmp_path / "part.obj")


def test_evaluate_extents_ignores_scaling_axis_and_applies_tolerance():
    # raw units arbitrary (x0.37); second dim off by 0.5 mm on 60 -> allowed 0.6 -> pass;
    # third off by 0.5 mm on 40 -> allowed 0.4 -> fail
    raw = np.array([100.0, 60.5, 40.5]) * 0.37
    res = measure.evaluate_extents("o1", raw, [40, 100, 60])
    assert [r.dim_rank for r in res] == [1, 2]
    assert res[0].passed and not res[1].passed


def _write_batch(tmp_path, n_good, n_bad):
    rows = []
    for i in range(n_good + n_bad):
        err = 0.0 if i < n_good else 3.0
        m = box((100.0, 60.0 + err, 40.0))
        m.apply_scale(0.05)  # arbitrary scan units
        name = f"obj{i}.stl"
        m.export(tmp_path / name)
        rows.append({"object_id": f"o{i}", "mesh_path": name, "dim1_mm": 100, "dim2_mm": 60, "dim3_mm": 40})
    rows.append({"object_id": "missing", "mesh_path": "nope.stl", "dim1_mm": 1, "dim2_mm": 1, "dim3_mm": 1})
    p = tmp_path / "scans.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p


def test_batch_go_no_go_counts_missing_scan_as_failure(tmp_path):
    p = _write_batch(tmp_path, n_good=7, n_bad=2)  # 7 of 10 incl. missing -> exactly 0.7 -> go
    s = measure.evaluate_batch(p, tmp_path / "report.csv")
    assert s["objects"] == 10 and s["passed"] == 7 and s["go"]
    assert any(o["object_id"] == "missing" and o["error"] for o in s["failed_objects"])
    assert (tmp_path / "report.csv").read_text().count("\n") == 1 + 9 * 2


def test_batch_no_go(tmp_path):
    p = _write_batch(tmp_path, n_good=6, n_bad=3)
    assert not measure.evaluate_batch(p)["go"]


def test_cli_prepare(tmp_path, capsys):
    src = tmp_path / "raw.stl"
    m = box((10.0, 6.0, 4.0))
    m.export(src)
    code = main(["prepare", str(src), str(tmp_path / "out.3mf"), "--known-mm", "100", "--cut-mm", "2"])
    assert code == 0
    assert '"watertight_after": true' in capsys.readouterr().out


def test_cli_reports_errors_without_traceback(tmp_path, capsys):
    assert main(["prepare", str(tmp_path / "none.stl"), str(tmp_path / "o.stl")]) == 3
    assert "error:" in capsys.readouterr().err
