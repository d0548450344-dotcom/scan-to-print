import numpy as np
import pytest
from PIL import Image

from scan2print import reconstruct as r


def scene(rng):
    """Tilted table (plane) + a 40x20x10 box on it + a distant clutter blob; cameras above."""
    table = np.c_[rng.uniform(-200, 200, (4000, 2)), np.zeros(4000)]
    bx = rng.uniform([-20, -10, 0], [20, 10, 10], (3000, 3))
    clutter = rng.uniform([150, 150, 0], [170, 170, 30], (600, 3))
    pts = np.vstack([table, bx, clutter])
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    cams = np.c_[rng.uniform(-60, 60, (30, 2)), rng.uniform(80, 120, 30)]
    # apply an arbitrary rigid transform + scale, as SfM output would have
    a = 0.7
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    t = np.array([5.0, -3.0, 12.0])
    s = 0.013
    return (pts @ R.T + t) * s, nrm @ R.T, (cams @ R.T + t) * s, s


def test_crop_keeps_box_drops_table_and_clutter_and_reorients():
    rng = np.random.default_rng(1)
    pts, nrm, cams, s = scene(rng)
    obj, obj_n = r.crop_to_object(pts, nrm, cams)
    ext = np.ptp(obj, axis=0) / s
    assert ext[2] == pytest.approx(10, abs=1.5)  # height along new z
    assert sorted(ext[:2]) == pytest.approx([20, 40], abs=2)
    assert obj[:, 2].min() / s == pytest.approx(0, abs=0.5)  # table at z=0
    assert np.allclose(obj_n.mean(axis=0), [0, 0, 1], atol=2e-3)


def test_crop_flips_plane_normal_towards_cameras():
    rng = np.random.default_rng(2)
    pts, nrm, cams, _ = scene(rng)
    obj, _ = r.crop_to_object(pts, nrm, cams)
    assert obj[:, 2].mean() > 0


def test_crop_rejects_too_few_points():
    with pytest.raises(r.ReconstructError):
        r.crop_to_object(np.zeros((10, 3)), np.zeros((10, 3)), np.ones((3, 3)))


def test_ply_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    xyz, n = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    r.write_ply_points(tmp_path / "p.ply", xyz, n)
    x2, n2 = r.read_ply_points(tmp_path / "p.ply")
    assert np.allclose(x2, xyz, atol=1e-5) and np.allclose(n2, n, atol=1e-5)


def test_prepare_images_applies_exif_rotation_and_resizes(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(10):
        im = Image.new("RGB", (400, 200), (i, 0, 0))
        exif = im.getexif()
        exif[0x0112] = 6  # stored rotated: portrait shot
        im.save(src / f"IMG_{i}.JPG", exif=exif.tobytes())
    assert r.prepare_images(src, tmp_path / "dst", max_size=100) == 10
    with Image.open(tmp_path / "dst" / "IMG_0.jpg") as out:
        assert out.size == (50, 100)  # rotated to portrait, long side 100
        assert out.getexif().get(0x0112) == 1


def test_prepare_images_needs_enough_photos(tmp_path):
    (tmp_path / "few").mkdir()
    Image.new("RGB", (10, 10)).save(tmp_path / "few" / "a.jpg")
    with pytest.raises(r.ReconstructError):
        r.prepare_images(tmp_path / "few", tmp_path / "out")
