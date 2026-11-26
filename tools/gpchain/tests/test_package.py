"""VES-3585 package.py: the 1 cm bar length follows from the mesh node spacing, the image size and the voxel size."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package as P  # noqa: E402

tifffile = pytest.importorskip("tifffile")


def flat_mesh(d, rows=40, cols=60, step_vx=5.0):
    """a flat tifxyz with nodes every step_vx voxels (scale = 1 / step_vx)."""
    d.mkdir()
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float32)
    for k, a in zip("xyz", (c * step_vx + 100, np.full_like(r, 50.0), r * step_vx + 1000)):
        tifffile.imwrite(d / f"{k}.tif", a)
    (d / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [1 / step_vx, 1 / step_vx]}))
    return d


def test_px_um_one_px_per_voxel(tmp_path):
    m = flat_mesh(tmp_path / "m.tifxyz")
    ur, uc, u = P.px_um(m, (200, 300), 9.6)          # 5 px per node, 5 vx per node -> 1 px = 1 vx
    assert ur == pytest.approx(9.6) and uc == pytest.approx(9.6) and u == pytest.approx(9.6)


def test_px_um_half_res_render(tmp_path):
    m = flat_mesh(tmp_path / "m.tifxyz")
    assert P.px_um(m, (100, 150), 9.6)[2] == pytest.approx(19.2)


def test_bar_is_one_cm():
    img = np.zeros((300, 2000), np.uint8)
    arr, bar, strip = P.with_scale_bar(img, 9.6)
    assert bar == round(10_000 / 9.6) == 1042
    assert arr.shape == (300 + strip, 2000, 3)
    assert (arr[:300] == 0).all()                      # nothing drawn on the papyrus
    row = arr[300 + strip // 3 + 1, :, 0]
    blk = np.nonzero(row[:10 + bar + 6] == 0)[0]          # the label text starts 12 px after the bar
    assert blk.min() == 10 and blk.max() == 10 + bar - 1


def test_bar_too_long_refused():
    with pytest.raises(ValueError):
        P.with_scale_bar(np.zeros((100, 500), np.uint8), 9.6)


def test_banner_order_and_numbers():
    a = [np.full((100, 50 * (i + 1)), 40 * i, np.uint8) for i in range(3)]
    arr, xs = P.banner(a, ["01", "02", "03"], height=100)
    assert [x[1] - x[0] for x in xs] == [50, 100, 150]
    assert xs[1][0] == xs[0][1] + 6


def test_falsepos_fill_and_missing():
    txt, miss = P.fill_template("AUC {auc} vs {baseline_auc}; {x}", {"auc": 0.943, "baseline_auc": 0.929})
    assert txt == "AUC 0.943 vs 0.929; MISSING(x)" and miss == ["x"]
    tpl = (Path(P.__file__).parent / "templates/false_positives.md").read_text()
    _, miss = P.fill_template(tpl, {})
    assert {"auc", "shift_max", "perm_mean", "heldout_rows", "human_minutes"} <= set(miss)
    assert "field" not in miss                          # the template's own comment holds no placeholder


def cyl_mesh(d, rows=30, cols=80, R=200.0, step_vx=5.0):
    """nodes every step_vx voxels of arc on a cylinder of radius R (axis z): px_um must give arc, not chord."""
    d.mkdir()
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float64)
    th = c * step_vx / R
    for k, a in zip("xyz", (R * np.cos(th) + 1000, R * np.sin(th) + 1000, r * step_vx + 500)):
        tifffile.imwrite(d / f"{k}.tif", a.astype(np.float32))
    (d / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [1 / step_vx, 1 / step_vx]}))
    return d


def test_px_um_curved_is_arc_length(tmp_path):
    m = cyl_mesh(tmp_path / "c.tifxyz")
    ur, uc, _ = P.px_um(m, (150, 400), 9.6)            # 5 px per node
    assert ur == pytest.approx(9.6, rel=1e-4)
    chord = 2 * 200.0 * np.sin(5.0 / 400.0)            # 5 vx of arc -> chord
    assert uc == pytest.approx(chord / 5 * 9.6, rel=1e-4) and uc == pytest.approx(9.6, rel=1e-3)


def test_to8_keeps_uint8_and_scales_float():
    a = np.array([[0, 7, 255]], np.uint8)
    assert P.to8(a) is a
    assert P.to8(np.array([[0.0, 0.5, 1.0]], np.float32)).tolist() == [[0, 127, 255]]
    assert P.to8(np.array([[0, 1000, 2000]], np.uint16)).tolist() == [[0, 127, 255]]


def test_letter_sizes_known_boxes():
    s = P.letter_sizes([{"name": "a", "box": [10, 20, 110, 70]}], 9.6, 10.0)
    assert s[0]["h_px"] == 100 and s[0]["w_px"] == 50
    assert s[0]["h_mm"] == pytest.approx(0.96) and s[0]["w_mm"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        P.letter_sizes([{"name": "x", "box": [5, 5, 5, 9]}], 9.6, 9.6)


def test_letters_figure_leaves_letter_px(tmp_path):
    img = (np.arange(200 * 1200) % 251).astype(np.uint8).reshape(200, 1200)
    arr, sizes = P.letters_figure(img, 9.6, 9.6, [{"name": "a", "box": [50, 100, 90, 140]}])
    assert (arr[50:90, 100:140, 0] == img[50:90, 100:140]).all()      # outline drawn outside the box
    assert arr[49, 120].tolist() == [255, 0, 255] and sizes[0]["w_mm"] == pytest.approx(0.384)


def test_build_package(tmp_path):
    sub = tmp_path / "sub"
    (sub / "images").mkdir(parents=True)
    (sub / "meshes").mkdir()
    for i in (1, 2):
        flat_mesh(sub / "meshes" / f"column_0{i}.tifxyz", rows=40, cols=260)
        tifffile.imwrite(sub / "images" / f"column_0{i}.tif", np.full((200, 1300), 30 * i, np.uint8))
    meta = P.build(sub, tmp_path / "pkg", 9.6, "PHercTest", "scan1", height=50)
    pkg = tmp_path / "pkg"
    assert sorted(p.name for p in (pkg / "images").iterdir()) == ["column_01.png", "column_02.png"]
    for i in (1, 2):
        for k in "xyz":
            a = (sub / "meshes" / f"column_0{i}.tifxyz" / f"{k}.tif").read_bytes()
            assert (pkg / "meshes" / f"column_0{i}.tifxyz" / f"{k}.tif").read_bytes() == a
    from PIL import Image
    a = np.asarray(Image.open(pkg / "images" / "column_02.png"))
    assert (a[:200, :, 0] == 60).all() and meta["images"]["column_02"]["bar_px"] == 1042
    assert (pkg / "banner.png").is_file() and json.loads((pkg / "banner.json").read_text())["columns"]["column_02"]
    with pytest.raises(FileExistsError):
        P.build(sub, pkg, 9.6, "PHercTest", "scan1")
