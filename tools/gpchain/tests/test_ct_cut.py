"""Tests for tools/gpchain/ct_cut.py and the --umbilicus radial reference (VES-3760, F-610 attempt 3).

Run: <#2401 venv>/bin/python -m pytest tools/gpchain/tests -q
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ct_cut  # noqa: E402
import render_contract as rc  # noqa: E402
from test_render_contract import CFG, cylinder, write_tifxyz  # noqa: E402

CH = 128


def store(tmp_path, ct, absent=()):
    """zarr v2 group like the real CT root (group 0, 128^3 raw chunks, '/' keys, fill 0); `absent` chunk keys
    (z, y, x) are not written."""
    root = tmp_path / 'ct.zarr'
    (root / '0').mkdir(parents=True)
    (root / '.zgroup').write_text(json.dumps({'zarr_format': 2}))
    (root / '0' / '.zarray').write_text(json.dumps(dict(
        shape=list(ct.shape), chunks=[CH] * 3, dtype='|u1', fill_value=0, order='C', filters=None,
        dimension_separator='/', compressor=None, zarr_format=2)))
    n = [-(-s // CH) for s in ct.shape]
    for z in range(n[0]):
        for y in range(n[1]):
            for x in range(n[2]):
                if (z, y, x) in absent:
                    continue
                blk = np.zeros((CH,) * 3, np.uint8)
                sub = ct[z * CH:(z + 1) * CH, y * CH:(y + 1) * CH, x * CH:(x + 1) * CH]
                blk[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
                d = root / '0' / str(z) / str(y)
                d.mkdir(parents=True, exist_ok=True)
                (d / str(x)).write_bytes(blk.tobytes())
    return root


def plane(tmp_path, x0=20.0, n_c=50, n_r=20, step=5.0, z=80.0, name='plane'):
    g = np.zeros((n_r, n_c, 3))
    g[..., 0] = x0 + np.arange(n_c)[None] * step
    g[..., 1] = 20 + np.arange(n_r)[:, None] * step
    g[..., 2] = z
    return write_tifxyz(tmp_path / name, g, 1 / step)


def shas(d):
    return {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in ('x.tif', 'y.tif', 'z.tif', 'meta.json')}


def test_full_ct_leaves_the_piece_bytes_unchanged(tmp_path):
    ct = np.full((256, 256, 384), 90, np.uint8)
    root, p = store(tmp_path, ct), plane(tmp_path)
    before = shas(p)
    row = ct_cut.cut_piece(p, root)
    assert row['nodes_cut'] == 0 and row['kept'] and not row['written'] and shas(p) == before


def test_ct_zero_beyond_x_is_cut_and_the_rest_kept(tmp_path):
    ct = np.full((256, 256, 384), 90, np.uint8)
    ct[:, :, 150:] = 0                                  # masked past x 150 (vx)
    root, p = store(tmp_path, ct), plane(tmp_path)      # nodes x 20 .. 265, step 5
    row = ct_cut.cut_piece(p, root)
    _g, v = rc.load_grid(p)
    xs = 20.0 + np.arange(50) * 5.0                     # node x before the cut (cut nodes now read -1)
    # the 16 layers run along z (plane normal), so a node is CT-valid iff the trilinear CT at x is > 0: x < 150
    assert row['written'] and row['kept'] and row['cut_ct0_only'] == row['nodes_cut'] > 0
    assert v[:, xs < 149].all() and not v[:, xs > 150].any()
    assert json.loads((p / 'meta.json').read_text())['ct_cut']['nodes_cut'] == row['nodes_cut']


def test_ct_zero_in_some_layers_is_kept(tmp_path):
    """a node is kept when one or more of its 16 render-layer positions has CT > 0 (villa then renders data)."""
    ct = np.zeros((256, 256, 384), np.uint8)
    ct[84:, :, :] = 90                                  # data only above z 84: layers j >= 12 at z 80 + (j - 7.5)
    root, p = store(tmp_path, ct), plane(tmp_path)
    row = ct_cut.cut_piece(p, root)
    assert row['nodes_cut'] == 0


def test_absent_chunk_is_cut_even_where_a_layer_reads_the_next_chunk(tmp_path):
    ct = np.full((256, 256, 384), 90, np.uint8)
    root = store(tmp_path, ct, absent={(0, 0, 2)})     # x 256-383 in z 0-127, y 0-127
    p = plane(tmp_path)
    row = ct_cut.cut_piece(p, root)
    _g, v = rc.load_grid(p)
    xs = 20.0 + np.arange(50) * 5.0
    assert row['cut_no_chunk'] == row['nodes_cut'] > 0
    assert not v[:, xs >= 256].any() and v[:, xs < 256].all()


def test_piece_with_no_quad_left_is_cut_away(tmp_path):
    ct = np.full((256, 256, 384), 90, np.uint8)
    ct[:, :, :] = 0
    root, p = store(tmp_path, ct), plane(tmp_path)
    before = shas(p)
    row = ct_cut.cut_piece(p, root)
    assert not row['kept'] and not row['written'] and shas(p) == before
    m2 = ct_cut.ct_cut_stage(tmp_path, [p], root)
    assert m2[0] == [] and (tmp_path / 'ct-cut-away' / 'plane').is_dir() and not p.exists()


def test_node_normals_match_the_plane():
    g = np.zeros((4, 5, 3))
    g[..., 0] = np.arange(5)[None] * 5.0
    g[..., 1] = np.arange(4)[:, None] * 5.0
    n, has = ct_cut.node_normals(g, np.ones((4, 5), bool))
    assert has.all() and np.allclose(np.abs(n[..., 2]), 1.0)


# ---------------------------------------------------------------- --umbilicus
def umb_file(tmp_path, pts):
    f = tmp_path / 'umbilicus.json'
    f.write_text(json.dumps(dict(control_points=[dict(x=x, y=y, z=z, score=100) for x, y, z in pts])))
    return f


def test_umbilicus_constant_equals_the_fixed_axis(tmp_path):
    piece = write_tifxyz(tmp_path / 'piece', cylinder(900.0, outward=False, centre=(40.0, -30.0)))
    a = rc.layer_order(piece, None, axis_xy=(40.0, -30.0), cfg=CFG)
    u = rc.Umbilicus(umb_file(tmp_path, [(40, -30, 0), (40, -30, 1000)]))
    b = rc.layer_order(piece, None, cfg=CFG, umbilicus=u)
    assert a['order'] == b['order'] == 'villa' and b['reference'] == 'radial'
    assert a['radial']['dot_median'] == b['radial']['dot_median'] and a['radial']['agree'] == b['radial']['agree']
    assert b['radial']['umbilicus_sha256'] == u.sha256 and 'axis_xy' not in b['radial']


def test_umbilicus_follows_the_axis_where_a_fixed_point_refuses(tmp_path):
    """a small winding around an axis that sits 300 vx off the fixed point: the fixed point mixes the dot signs."""
    piece = write_tifxyz(tmp_path / 'piece', cylinder(250.0, n_cols=300, outward=False, centre=(300.0, 0.0)))
    a = rc.layer_order(piece, None, axis_xy=(0.0, 0.0), cfg=CFG)
    u = rc.Umbilicus(umb_file(tmp_path, [(300, 0, 100), (300, 0, 400)]))
    b = rc.layer_order(piece, None, cfg=CFG, umbilicus=u)
    assert a['order'] is None and b['order'] == 'villa' and b['radial']['agree'] > 0.99


def test_umbilicus_interpolates_in_z_and_holds_past_the_ends(tmp_path):
    u = rc.Umbilicus(umb_file(tmp_path, [(10, 0, 200), (0, 0, 100), (20, 10, 300)]))
    assert np.allclose(u(np.array([50.0, 150.0, 250.0, 900.0])), [[0, 0], [5, 0], [15, 5], [20, 10]])


def test_axis_and_umbilicus_together_refuse(tmp_path):
    import pytest
    piece = write_tifxyz(tmp_path / 'piece', cylinder(900.0, outward=False))
    with pytest.raises(ValueError):
        rc.layer_order(piece, None, axis_xy=(0.0, 0.0), cfg=CFG, umbilicus=rc.Umbilicus(umb_file(tmp_path, [(0, 0, 0)])))


# ---------------------------------------------------------------- quad-level test (--px, VES-3782)
def test_chunk_ct_equals_ct_sample_across_chunk_faces(tmp_path):
    rng = np.random.default_rng(0)
    ct = rng.integers(0, 255, (200, 140, 300), dtype=np.uint8)
    root = store(tmp_path, ct, absent={(1, 0, 1)})
    q = np.concatenate([rng.uniform([0, 0, 0], [299, 139, 199], (3000, 3)),
                        np.array([[127.5, 127.5, 127.5], [255.9, 127.2, 127.99], [128.0, 0.0, 0.0]])])
    Z = rc.open_array(root / '0')
    assert np.allclose(ct_cut.ChunkCT(root)(q), rc.ct_sample(Z, q), atol=1e-9)


def test_villa_normals_need_all_4_neighbours():
    g = np.zeros((5, 6, 3))
    g[..., 0] = np.arange(6)[None] * 5.0
    g[..., 1] = np.arange(5)[:, None] * 5.0
    v = np.ones((5, 6), bool)
    v[2, 4] = False
    n, has = ct_cut.villa_normals(g, v)
    assert not has[0].any() and not has[-1].any() and not has[:, 0].any() and not has[:, -1].any()
    assert not has[2, 3] and not has[1, 4] and not has[3, 4] and not has[2, 4] and has[1, 1] and has[2, 2]
    assert np.allclose(np.abs(n[has][:, 2]), 1.0)


def test_px_test_full_ct_leaves_the_piece_bytes_unchanged(tmp_path):
    ct = np.full((256, 256, 384), 90, np.uint8)
    root, p = store(tmp_path, ct), plane(tmp_path)
    before = shas(p)
    row = ct_cut.cut_piece(p, root, px=True)
    assert row['nodes_cut'] == 0 and row['px']['px_ct0'] == 0 and not row['written'] and shas(p) == before


def test_px_test_cuts_a_ct0_hole_between_passing_nodes(tmp_path):
    """a CT 0 box (all 16 layers) between nodes: the node test keeps every node, the px test drops the quad."""
    ct = np.full((256, 256, 384), 90, np.uint8)
    ct[60:101, 42:44, 102:104] = 0                      # z 60-100 (layers at z 72.5-87.5), y 42-43, x 102-103
    root, p = store(tmp_path, ct), plane(tmp_path)      # nodes x 20 + 5k, y 20 + 5k: the box sits inside the quad
    node_only = ct_cut.cut_piece(p, root, dry_run=True)  # x 100-105, y 40-45; px at x, y 102.5, 103.5
    assert node_only['nodes_cut'] == 0
    row = ct_cut.cut_piece(p, root, px=True)
    assert row['written'] and row['kept'] and row['cut_node_test'] == 0 and row['cut_px_test'] >= 1
    assert row['px']['px_ct0'] > 0 and row['px']['quads_with_ct0'] >= 1
    g, v = rc.load_grid(p)
    ok, nrm, ct0, _ni, _nj = ct_cut.px_model(g, v, rc.scale(p), ct_cut.ChunkCT(root))
    assert ok.any() and not ct0.any()                   # no CT0 px left inside a kept quad
    assert json.loads((p / 'meta.json').read_text())['ct_cut']['cut_px_test'] == row['cut_px_test']


def test_px_test_keeps_a_px_with_data_in_one_layer(tmp_path):
    ct = np.full((256, 256, 384), 90, np.uint8)
    ct[60:87, 42:44, 102:104] = 0                       # layer z 87.5 (j = 15) still reads CT > 0 there
    root, p = store(tmp_path, ct), plane(tmp_path)
    row = ct_cut.cut_piece(p, root, px=True, dry_run=True)
    assert row['px']['px_ct0'] == 0 and row['nodes_cut'] == 0


def test_px_test_counts_trilinear_below_1_as_zero(tmp_path):
    """villa writes uint8(trilinear) (truncation): 0.5 everywhere renders 0 (and is < ZERO_BELOW)."""
    ct = np.zeros((256, 256, 384), np.uint8)
    ct[::2] = 1                                         # trilinear 0.5 at half-integer z
    root, p = store(tmp_path, ct), plane(tmp_path)      # layers at z 72.5 .. 87.5
    row = ct_cut.cut_piece(p, root, px=True, dry_run=True)
    assert row['cut_node_test'] == 0                    # the node test (trilinear > 0) keeps every node
    assert row['px']['px_ct0'] == row['px']['px_tested'] > 0


def test_px_test_margin_counts_one_grey_level_as_zero(tmp_path):
    """ZERO_BELOW 2.0 (README attempt 4 step M): trilinear 1.5 in every layer counts as CT0, 2.0 does not."""
    ct = np.zeros((256, 256, 384), np.uint8)
    ct[::2] = 1
    ct[1::2] = 2                                        # trilinear 1.5 at half-integer z
    root, p = store(tmp_path, ct), plane(tmp_path)
    row = ct_cut.cut_piece(p, root, px=True, dry_run=True)
    assert row['px']['px_ct0'] == row['px']['px_tested'] > 0
    ct[:] = 2
    root2, p2 = store(tmp_path / 'b', ct), plane(tmp_path / 'b')
    assert ct_cut.cut_piece(p2, root2, px=True, dry_run=True)['px']['px_ct0'] == 0
