"""Thin wrapper around the COLMAP CLI.

NOT TESTED in the dev container: it has no COLMAP binary and no GPU, and dense
reconstruction needs CUDA. Run on a GPU machine with COLMAP installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class ReconstructError(RuntimeError):
    pass


def colmap_mesh(image_dir: str | Path, workspace: str | Path, colmap_bin: str = "colmap",
                quality: str = "medium") -> Path:
    """Run COLMAP's automatic reconstructor and return the Poisson mesh path."""
    if shutil.which(colmap_bin) is None:
        raise ReconstructError(f"'{colmap_bin}' not found on PATH; install COLMAP (needs CUDA for dense)")
    image_dir, workspace = Path(image_dir), Path(workspace)
    if not image_dir.is_dir() or not any(image_dir.iterdir()):
        raise ReconstructError(f"no images in {image_dir}")
    workspace.mkdir(parents=True, exist_ok=True)
    cmd = [colmap_bin, "automatic_reconstructor",
           "--workspace_path", str(workspace),
           "--image_path", str(image_dir),
           "--quality", quality,
           "--dense", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ReconstructError(f"colmap failed ({proc.returncode}): {proc.stderr[-2000:]}")
    mesh = workspace / "dense" / "0" / "meshed-poisson.ply"
    if not mesh.exists():
        raise ReconstructError(f"colmap finished but {mesh} is missing; check the log in {workspace}")
    return mesh
