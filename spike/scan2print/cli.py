"""Command line: scan2print {prepare,evaluate,reconstruct} ..."""

from __future__ import annotations

import argparse
import json
import sys

from . import measure, prepare


def _prepare(a: argparse.Namespace) -> int:
    mesh, rep = prepare.repair(prepare.load_mesh(a.input), a.min_component_ratio)
    if a.known_mm:
        mesh = prepare.scale_to_known(mesh, a.known_mm, a.extent_index)
    mesh = prepare.flatten_base(mesh, a.cut_mm)
    out = prepare.export(mesh, a.output)
    print(json.dumps({"output": str(out), "extents_mm": prepare.principal_extents(mesh).round(2).tolist(),
                      "repair": rep.__dict__}, ensure_ascii=False, indent=2))
    return 0 if rep.watertight_after else 2


def _evaluate(a: argparse.Namespace) -> int:
    summary = measure.evaluate_batch(a.csv, a.report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["go"] else 1


def _reconstruct(a: argparse.Namespace) -> int:
    from .reconstruct import colmap_mesh
    print(colmap_mesh(a.images, a.workspace, a.colmap, a.quality))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scan2print")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="repair, scale, flatten and export a mesh")
    pp.add_argument("input")
    pp.add_argument("output", help=".stl or .3mf")
    pp.add_argument("--known-mm", type=float, help="real length of one principal dimension")
    pp.add_argument("--extent-index", type=int, default=0, choices=(0, 1, 2), help="0 = longest dimension")
    pp.add_argument("--cut-mm", type=float, default=0.0, help="slice this much off the bottom for a flat base")
    pp.add_argument("--min-component-ratio", type=float, default=0.01)
    pp.set_defaults(func=_prepare)

    pe = sub.add_parser("evaluate", help="Phase 0 accuracy go/no-go over a CSV of scans")
    pe.add_argument("csv")
    pe.add_argument("--report", help="per-dimension CSV report path")
    pe.set_defaults(func=_evaluate)

    pr = sub.add_parser("reconstruct", help="run COLMAP on a folder of photos (GPU machine only)")
    pr.add_argument("images")
    pr.add_argument("workspace")
    pr.add_argument("--colmap", default="colmap")
    pr.add_argument("--quality", default="medium", choices=("low", "medium", "high", "extreme"))
    pr.set_defaults(func=_reconstruct)

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except Exception as e:  # CLI boundary: report, don't dump a traceback on users
        print(f"error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
