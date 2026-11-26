"""VES-3585 orient.py: synthetic rolled scrolls with text written by A1-A2 (recto toward the core, outer -> inner)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orient as O  # noqa: E402


rolled = O.synthetic_scroll


def test_ops_group():
    for a in O.OPS:
        assert O.compose(a, O.inverse(a)) == "r0"
        for b in O.OPS:
            img = np.arange(12).reshape(3, 4)
            assert np.array_equal(O.apply_op(O.apply_op(img, a), b), O.apply_op(img, O.compose(a, b)))
    assert len({O.T[o].tobytes() for o in O.OPS}) == 8


@pytest.mark.parametrize("sigma", [1, -1])
def test_handedness(sigma):
    g, v = rolled(sigma, turns=1.3)
    h = O.handedness(g, v, (1000.0, 800.0))
    assert h["h"] == -sigma


def test_handedness_undecided_under_one_turn():
    g, v = rolled(1, turns=0.8)
    assert O.handedness(g, v, (1000.0, 800.0))["h"] is None


@pytest.mark.parametrize("sigma", [1, -1])
@pytest.mark.parametrize("op", O.OPS)
def test_rule_recovers_upright(sigma, op):
    g0, v0 = rolled(sigma, tilt=0.3)
    g, v = O.apply_op(g0, op), O.apply_op(v0, op)
    r = O.rule_op(g, v, (1000.0, 800.0), -sigma)
    assert r["op"] == O.inverse(op)
    assert np.allclose(O.apply_op(g, r["op"]), g0)
    assert r["mirror_by_normal"] == (op[0] == "m")
    assert r["node_agree"] > 0.95


def test_wrong_h_turns_180():
    g0, v0 = rolled(1)
    assert O.rule_op(g0, v0, (1000.0, 800.0), +1)["op"] == "r180"   # true h = -1


def test_orient_tifxyz_roundtrip(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    g0, _ = rolled(1, n_s=50, n_t=20)
    src = tmp_path / "a.tifxyz"
    src.mkdir()
    for i, c in enumerate("xyz"):
        tifffile.imwrite(src / f"{c}.tif", g0[..., i].astype(np.float32))
    (src / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [0.2, 0.1]}))
    dst = O.orient_tifxyz(src, tmp_path / "b.tifxyz", "m90", {"h": -1})
    g, _ = O.load_grid(dst)
    assert np.allclose(g, O.apply_op(g0.astype(np.float32), "m90"))
    m = json.loads((dst / "meta.json").read_text())
    assert m["orient"]["op"] == "m90" and m["scale"] == [0.1, 0.2]


def _cfg(tmp_path, status, scrolls=None):
    p = tmp_path / "orient_config.json"
    p.write_text(json.dumps({"rule": {"status": status, "why": "test"}, "scrolls": scrolls or {}}))
    return p


def _piece(tmp_path, name, g):
    tifffile = pytest.importorskip("tifffile")
    d = tmp_path / name
    d.mkdir()
    for i, c in enumerate("xyz"):
        tifffile.imwrite(d / f"{c}.tif", g[..., i].astype(np.float32))
    (d / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [0.2, 0.2]}))
    return d


@pytest.mark.parametrize("status", ["pending", "not-confirmed"])
def test_orient_stage_rule_refuses_unconfirmed_without_measured(tmp_path, status):
    import chain
    g0, _ = rolled(1, n_s=50, n_t=20)
    m = _piece(tmp_path, "column_01.tifxyz", g0)
    with pytest.raises(SystemExit):
        chain.orient_stage(m, [m], "rule", [1000.0, 800.0], _cfg(tmp_path, status), "PHercParis4")
    with pytest.raises(SystemExit):
        chain.orient_stage(m, [m], "rule", [1000.0, 800.0],
                           _cfg(tmp_path, status, {"PHercParis4": {"op": "m180"}}), None)


def test_orient_stage_rule_uses_measured_op_when_unconfirmed(tmp_path):
    import chain
    g0, _ = rolled(1, n_s=50, n_t=20)
    m = _piece(tmp_path, "column_01.tifxyz", g0)
    sec = chain.orient_stage(m, [m], "rule", [1000.0, 800.0],
                             _cfg(tmp_path, "not-confirmed", {"PHercParis4": {"op": "m180"}}), "PHercParis4")
    assert sec["pieces"]["column_01.tifxyz"]["op"] == "m180"
    assert sec["pieces"]["column_01.tifxyz"]["from_config"] == "PHercParis4"
    g, _ = O.load_grid(m)
    assert np.allclose(g, O.apply_op(g0.astype(np.float32), "m180"))


def test_orient_stage_rule_confirmed_uses_geometry(tmp_path):
    import chain
    g0, v0 = rolled(-1, tilt=0.3)
    fit = _piece(tmp_path, "fit.tifxyz", rolled(-1, turns=1.3)[0])
    m = _piece(tmp_path, "column_01.tifxyz", O.apply_op(g0, "r90"))
    sec = chain.orient_stage(fit, [m], "rule", [1000.0, 800.0], _cfg(tmp_path, "confirmed"), None)
    assert sec["handedness"]["h"] == 1
    assert sec["pieces"]["column_01.tifxyz"]["op"] == O.inverse("r90")
