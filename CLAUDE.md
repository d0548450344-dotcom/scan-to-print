# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Pre-code. The project is in **Phase 0 (feasibility spike)**. There is no build, lint or test tooling yet. Do not invent commands; add them here when the first code lands.

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
