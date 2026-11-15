"""Photos → printable-candidate mesh, using pycolmap.

Pipeline: normalise photos (EXIF rotation, size) → SIFT + exhaustive matching →
incremental SfM → undistort → PatchMatch stereo (CUDA) → fusion → crop to the
object (remove the table plane, re-orient z-up) → Poisson mesh.

Needs `pycolmap-cuda12` and an NVIDIA GPU for the dense steps (e.g. Google Colab
with a T4). Only the pure-numpy parts (PLY I/O, cropping) are tested in CI; the
pycolmap calls are exercised in the Colab notebook.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class ReconstructError(RuntimeError):
    pass


@dataclass
class ReconReport:
    images_total: int = 0
    images_registered: int = 0
    fused_points: int = 0
    object_points: int = 0
    mesh_path: str = ""
    warnings: list[str] = field(default_factory=list)


# ---------- photos ----------

def prepare_images(src: str | Path, dst: str | Path, max_size: int = 3200) -> int:
    """Copy photos with EXIF rotation applied and the long side capped at `max_size`.

    Android phones store portrait shots rotated with an EXIF flag; COLMAP reads raw
    pixels, so without this mixed orientations break reconstruction.
    """
    from PIL import Image, ImageOps

    src, dst = Path(src), Path(dst)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if len(files) < 10:
        raise ReconstructError(f"{src}: only {len(files)} photos; need at least 10 (40-80 recommended)")
    dst.mkdir(parents=True, exist_ok=True)
    for p in files:
        with Image.open(p) as im:
            exif = im.getexif()
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_size, max_size))
            exif[0x0112] = 1  # orientation now normal; keep focal-length tags for COLMAP
            im.save(dst / (p.stem + ".jpg"), quality=95, exif=exif.tobytes())
    return len(files)


# ---------- PLY (vertex-only, as written by COLMAP fusion) ----------

_PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1",
              "int": "i4", "int32": "i4", "uint": "u4", "short": "i2", "ushort": "u2", "char": "i1"}


def read_ply_points(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read xyz (and nx,ny,nz if present) from a binary little-endian vertex PLY."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ReconstructError(f"{path}: not a PLY file")
        props, n, fmt = [], 0, None
        in_vertex = False
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ReconstructError(f"{path}: list properties on vertices not supported")
                props.append((parts[2], "<" + _PLY_TYPES[parts[1]]))
        if fmt != "binary_little_endian":
            raise ReconstructError(f"{path}: unsupported PLY format {fmt}")
        data = np.frombuffer(f.read(n * np.dtype(props).itemsize), dtype=props, count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    names = {p[0] for p in props}
    normals = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(np.float64) \
        if {"nx", "ny", "nz"} <= names else None
    return xyz, normals


def write_ply_points(path: str | Path, xyz: np.ndarray, normals: np.ndarray) -> None:
    rec = np.empty(len(xyz), dtype=[(k, "<f4") for k in ("x", "y", "z", "nx", "ny", "nz")])
    for i, k in enumerate("xyz"):
        rec[k] = xyz[:, i]
        rec["n" + k] = normals[:, i]
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
              + "".join(f"property float {k}\n" for k in ("x", "y", "z", "nx", "ny", "nz"))
              + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


# ---------- cropping ----------

def fit_plane_ransac(xyz: np.ndarray, threshold: float, iters: int = 500,
                     rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """Dominant plane n·x + d = 0 (unit n) by RANSAC."""
    rng = rng or np.random.default_rng(0)
    best_n, best_d, best_count = None, 0.0, -1
    for _ in range(iters):
        a, b, c = xyz[rng.choice(len(xyz), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        d = -n @ a
        count = int(np.sum(np.abs(xyz @ n + d) < threshold))
        if count > best_count:
            best_n, best_d, best_count = n, d, count
    if best_n is None:
        raise ReconstructError("could not fit a ground plane")
    # least-squares refinement on the inliers (RANSAC's 3-point plane is slightly tilted)
    inl = xyz[np.abs(xyz @ best_n + best_d) < threshold]
    c = inl.mean(axis=0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][-1]
    if n @ best_n < 0:
        n = -n
    return n, float(-n @ c)


def _rotation_to_z(n: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), float(n @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def crop_to_object(xyz: np.ndarray, normals: np.ndarray, camera_centers: np.ndarray,
                   plane_threshold_ratio: float = 0.005, cluster_radius_ratio: float = 0.01
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Remove the table plane and background, return the object points z-up with the table at z=0.

    Thresholds are fractions of the scene size because SfM units are arbitrary.
    The object side of the plane is the side the cameras are on. The object is the
    connected cluster (above the plane) closest to the point all cameras look at,
    approximated by the cluster nearest the cameras' centroid projected onto the plane.
    """
    if len(xyz) < 100:
        raise ReconstructError(f"only {len(xyz)} fused points; reconstruction too sparse")
    scale = float(np.linalg.norm(np.percentile(xyz, 95, axis=0) - np.percentile(xyz, 5, axis=0)))
    n, d = fit_plane_ransac(xyz, plane_threshold_ratio * scale)
    if np.mean(camera_centers @ n + d) < 0:
        n, d = -n, -d

    height = xyz @ n + d
    above = height > 2 * plane_threshold_ratio * scale
    if above.sum() < 50:
        raise ReconstructError("no object found above the table plane")
    pts, nrm = xyz[above], normals[above]

    tree = cKDTree(pts)
    pairs = tree.query_pairs(cluster_radius_ratio * scale, output_type="ndarray")
    from scipy.sparse import coo_matrix
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts), len(pts)))
    _, labels = connected_components(adj, directed=False)

    target = camera_centers.mean(axis=0)
    target = target - (target @ n + d) * n  # onto the table
    # among sizeable clusters, take the one whose footprint centre is nearest the target
    counts = np.bincount(labels)
    candidates = np.flatnonzero(counts >= max(50, 0.05 * counts.max()))
    def dist(lbl):
        c = pts[labels == lbl].mean(axis=0)
        c = c - (c @ n + d) * n
        return np.linalg.norm(c - target)
    best = min(candidates, key=dist)
    obj_mask = np.zeros(len(xyz), dtype=bool)
    obj_mask[np.flatnonzero(above)[labels == best]] = True

    # The "above" margin also shaved the object's lowest slice, which would shorten
    # every measured height. Add back near-table points that lie inside the object's
    # footprint (its 2D hull on the table). Known limit: under an overhang (a mug
    # handle) this also pulls in a sliver of visible table.
    R = _rotation_to_z(n)
    flat = (xyz @ R.T)[:, :2]
    band = (~obj_mask) & (height > -plane_threshold_ratio * scale) & ~above
    if band.any():
        hull = Delaunay(flat[obj_mask])
        inside = np.zeros(len(xyz), dtype=bool)
        inside[band] = hull.find_simplex(flat[band]) >= 0
        obj_mask |= inside
    pts, nrm = xyz[obj_mask], normals[obj_mask]

    origin = -d * n  # a point on the plane
    return (pts - origin) @ R.T, nrm @ R.T


# ---------- pycolmap pipeline ----------

def reconstruct(photo_dir: str | Path, work: str | Path, max_image_size: int = 2000,
                poisson_depth: int = 10) -> ReconReport:
    """Full photos → mesh run. Returns a report; the mesh is at report.mesh_path (z-up, table at z=0)."""
    import pycolmap

    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    images, sparse_dir, mvs = work / "images", work / "sparse", work / "mvs"
    sparse_dir.mkdir(parents=True)
    rep = ReconReport()
    rep.images_total = prepare_images(photo_dir, images)

    db = work / "database.db"
    pycolmap.extract_features(db, images)
    pycolmap.match_exhaustive(db)
    recs = pycolmap.incremental_mapping(db, images, sparse_dir)
    if not recs:
        raise ReconstructError("structure-from-motion failed: no images could be registered")
    best_id = max(recs, key=lambda k: recs[k].num_reg_images())
    rec = recs[best_id]
    rep.images_registered = rec.num_reg_images()
    if rep.images_registered < 0.8 * rep.images_total:
        rep.warnings.append(f"only {rep.images_registered}/{rep.images_total} photos registered; "
                            "add overlap or texture to the background")
    centers = np.array([rec.images[i].projection_center() for i in rec.reg_image_ids()])

    uopts = pycolmap.UndistortCameraOptions()
    uopts.max_image_size = max_image_size
    pycolmap.undistort_images(mvs, sparse_dir / str(best_id), images, undistort_options=uopts)
    pycolmap.patch_match_stereo(mvs)  # CUDA only
    fused = mvs / "fused.ply"
    pycolmap.stereo_fusion(fused, mvs, output_type="PLY")

    xyz, normals = read_ply_points(fused)
    rep.fused_points = len(xyz)
    if normals is None:
        raise ReconstructError("fused point cloud has no normals")
    obj, obj_n = crop_to_object(xyz, normals, centers)
    rep.object_points = len(obj)
    cropped = work / "object_points.ply"
    write_ply_points(cropped, obj, obj_n)

    mesh = work / "mesh.ply"
    mopts = pycolmap.PoissonMeshingOptions()
    mopts.depth = poisson_depth
    pycolmap.poisson_meshing(cropped, mesh, mopts)
    rep.mesh_path = str(mesh)
    return rep
