# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

The project is in **Phase 0 (feasibility spike)**. The only code is `spike/`, a Python package (`scan2print`) for print preparation and the accuracy go/no-go check. It is throwaway-quality by intent but its repair/scale/export logic is expected to seed the MVP worker. The protocol the user follows with real objects is `docs/phase0-protocol.md`.

## Commands (run from `spike/`)

```bash
python -m venv ../.venv && ../.venv/bin/pip install -e '.[dev]'
../.venv/bin/pytest -q                                   # all tests
../.venv/bin/pytest -q tests/test_pipeline.py -k scale   # single test
../.venv/bin/scan2print prepare raw.ply out.3mf --known-mm 120 --cut-mm 1
../.venv/bin/scan2print evaluate scans.csv --report out/report.csv   # exit 0 = go, 1 = no-go
```

CI (`.github/workflows/spike.yml`) runs pytest on PRs. `scan2print reconstruct` wraps the COLMAP CLI and is **untested**: the dev container has no COLMAP and no GPU.

## spike/ structure

- `prepare.py` – load → `repair` (drop debris by surface-area ratio, fix normals, fill simple holes; reports instead of hiding non-watertight results) → `scale_to_known` (uniform scale from one real dimension, via oriented-bbox principal extents so rotation doesn't matter) → `flatten_base` (assumes z-up) → `export` (STL/3MF, mm).
- `measure.py` – go/no-go: scale by the longest caliper dimension, evaluate only the other two (the scaling axis is correct by construction), tolerance 1 mm per 100 mm, pass if ≥70% of objects pass. Failed/missing scans count as failures. Thresholds are constants mirroring `docs/feasibility.md`; change both together.

The product vision (Hebrew) and the decisions below are in `docs/`:
- `docs/feasibility.md` – feasibility analysis, risks, and the Phase 0 go/no-go criterion
- `docs/architecture.md` – component choices (existing vs. built in-house) and licensing
- `docs/roadmap.md` – phased roadmap and MVP scope

The user writes in Hebrew; reply in Hebrew unless asked otherwise. Code, identifiers and commit messages are in English.

## Product in one line

Phone/desktop camera photos of a physical object → 3D reconstruction → interactive viewer → automatic print preparation → **explicit user approval** → STL/3MF export. Cloud-first, multi-device.

## Decisions already made (don't relitigate without new evidence)

- **Do not build a reconstruction engine.** Use COLMAP (BSD-3) in a cloud GPU worker for the MVP; Apple Object Capture on iOS later. The product's differentiator is the print-preparation pipeline, not scanning.
- **MVP exports STL/3MF only.** No G-code and no printer connectivity in the MVP. Users slice in their own slicer.
- **Scale comes from the user or a reference object.** Photogrammetry has no absolute scale; never present dimensions as accurate without one.
- **Approval gate is mandatory:** the final print file is produced only after an explicit user confirmation step. Keep model creation and print preparation as separate stages/modules.
- Viewer: Google `model-viewer`; three.js only when editing tools are needed.
- Mesh repair/prep is built in-house on top of `manifold` (Apache-2.0) and `trimesh` (MIT).

## Licensing constraints (check before adding any dependency)

- **AGPL-3.0:** OrcaSlicer, PrusaSlicer, CuraEngine, OpenMVS, OctoPrint. Running these as part of a hosted service triggers source-disclosure obligations. Do not link them into the backend; any use (Phase 2+) must be a separate process and requires an explicit licensing decision recorded in `docs/architecture.md`.
- **GPL-3.0:** PyMeshLab/MeshLab, Moonraker – server-side only, never shipped in client apps.
- **Non-commercial – do not use:** Inria `graphdeco-inria/gaussian-splatting`, NVIDIA instant-ngp. Use `gsplat` (Apache-2.0) if splatting is ever needed (viewing only; splats are not printable meshes).
- AI image-to-3D models (TRELLIS, Hunyuan3D): check the **weights'** license separately from the code license.

## Known technical limits to respect in design and UX

- Phone photogrammetry accuracy is roughly ±0.5–2 mm at best; not suitable for precision mechanical parts. Do not market or message it as such.
- Shiny, transparent and textureless objects fail reconstruction; detect/warn rather than silently produce bad meshes.
- The cloud dev container for these sessions has no GPU; heavy reconstruction cannot be tested here.
