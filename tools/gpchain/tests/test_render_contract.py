"""Tests for tools/gpchain/render_contract.py (VES-3519): per-mesh layer order, FC on render-data px + rim gate.

Run: <#2401 venv>/bin/python -m pytest tools/gpchain/tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render_contract as rc  # noqa: E402

CFG = rc.load_config()


def write_tifxyz(d, g, s=0.2):
    import tifffile
    d.mkdir(parents=True)
    for k, c in enumerate('xyz'):
        tifffile.imwrite(d / f'{c}.tif', g[..., k].astype(np.float32))
    (d / 'meta.json').write_text(json.dumps({'scale': [s, s], 'type': 'seg', 'format': 'tifxyz'}))
    return d


def cylinder(R, n_rows=40, n_cols=200, z0=100.0, step=5.0, outward=True, centre=(0.0, 0.0)):
    """Grid on a cylinder: rows = z, cols = angle (growing). n = (dg/dcol) x (dg/drow) points outwards when z grows
    with the row (outward=True), inwards when it falls."""
    th = np.arange(n_cols) * step / R
    z = z0 + np.arange(n_rows) * step
    if not outward:
        z = z[::-1]
    g = np.zeros((n_rows, n_cols, 3))
    g[..., 0] = centre[0] + R * np.cos(th)[None]
    g[..., 1] = centre[1] + R * np.sin(th)[None]
    g[..., 2] = z[:, None]
    return g


# ---------------------------------------------------------------- layer order
def test_inward_piece_vs_outward_segment_reads_villa_order(tmp_path):
    seg = write_tifxyz(tmp_path / 'seg', cylinder(500.0, outward=True))
    piece = write_tifxyz(tmp_path / 'piece', cylinder(503.0, outward=False))     # 3 vx away, normal inwards
    o = rc.layer_order(piece, rc.SegmentNormals([seg]), axis_xy=(0.0, 0.0), cfg=CFG)
    assert o['order'] == 'villa' and o['reference'] == 'segments'
    assert o['segments']['dot_median'] < -0.99 and o['radial']['dot_median'] < -0.99


def test_outward_piece_reads_reversed_like_the_segments(tmp_path):
    seg = write_tifxyz(tmp_path / 'seg', cylinder(500.0, outward=True))
    piece = write_tifxyz(tmp_path / 'piece', cylinder(503.0, outward=True))
    o = rc.layer_order(piece, rc.SegmentNormals([seg]), axis_xy=(0.0, 0.0), cfg=CFG)
    assert o['order'] == 'reversed' and o['segments']['dot_median'] > 0.99


def test_order_is_per_mesh_not_copied(tmp_path):
    """Two pieces next to the same segment with opposite grid orientation get opposite orders."""
    seg = rc.SegmentNormals([write_tifxyz(tmp_path / 'seg', cylinder(500.0, outward=True))])
    a = write_tifxyz(tmp_path / 'a', cylinder(502.0, outward=False))
    b = write_tifxyz(tmp_path / 'b', cylinder(502.0, outward=False)[:, ::-1])  # cols reversed: normal flips
    assert rc.layer_order(a, seg, cfg=CFG)['order'] == 'villa'
    assert rc.layer_order(b, seg, cfg=CFG)['order'] == 'reversed'


def test_radial_fallback_when_no_segment_is_near(tmp_path):
    seg = rc.SegmentNormals([write_tifxyz(tmp_path / 'seg', cylinder(500.0, outward=True))])
    piece = write_tifxyz(tmp_path / 'piece', cylinder(900.0, outward=False))     # 400 vx from the segment
    o = rc.layer_order(piece, seg, axis_xy=(0.0, 0.0), cfg=CFG)
    assert o['segments']['n_near'] == 0 and o['reference'] == 'radial' and o['order'] == 'villa'


def test_no_reference_refuses(tmp_path):
    piece = write_tifxyz(tmp_path / 'piece', cylinder(900.0, outward=False))
    o = rc.layer_order(piece, None, None, cfg=CFG)
    assert o['order'] is None and 'no reference' in o['why']


def test_radial_disagreeing_with_segments_refuses(tmp_path):
    seg = rc.SegmentNormals([write_tifxyz(tmp_path / 'seg', cylinder(500.0, outward=True))])
    piece = write_tifxyz(tmp_path / 'piece', cylinder(503.0, outward=False))
    # an axis on the far side of the piece turns the radial direction round on most nodes
    o = rc.layer_order(piece, seg, axis_xy=(1100.0, 0.0), cfg=CFG)
    if (o['radial']['dot_median'] < 0) != (o['segments']['dot_median'] < 0):
        assert o['order'] is None and 'disagrees' in o['why']
    else:
        pytest.skip('synthetic axis did not flip the radial median')


def test_mixed_segment_signs_refuse(tmp_path):
    """Half the piece lies on an outward segment, half on an inward one: no clear sign -> refused."""
    g = cylinder(500.0, n_cols=200, outward=False)              # normal inwards
    s_out = write_tifxyz(tmp_path / 's1', g[:, :100][:, ::-1])  # first half, cols reversed: normal outwards
    s_in = write_tifxyz(tmp_path / 's2', g[:, 100:])            # second half, normal inwards
    piece = write_tifxyz(tmp_path / 'piece', cylinder(503.0, n_cols=200, outward=False))
    o = rc.layer_order(piece, rc.SegmentNormals([s_out, s_in]), cfg=CFG)
    assert o['segments']['agree'] < CFG['layer_order']['min_agree'] and o['order'] is None


def test_ordered_stack():
    V = np.arange(16 * 2 * 3, dtype=np.uint8).reshape(16, 2, 3)
    assert np.array_equal(rc.ordered_stack(V, 'villa'), V)
    assert np.array_equal(rc.ordered_stack(V, 'reversed'), V[::-1])
    assert rc.ordered_stack(V, 'reversed').flags['C_CONTIGUOUS']
    with pytest.raises(ValueError):
        rc.ordered_stack(V, None)


# ---------------------------------------------------------------- FC + rim gate
def ct_field(shape=(160, 200, 200)):
    z, y, x = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing='ij')
    return (100 + 40 * np.sin(x / 7.0) * np.cos(y / 9.0) + 0.2 * z).astype(np.float32)


def plane_piece(tmp_path, s=0.2):
    """Flat piece at z = 80: rows along y, cols along x, 5 vx per node."""
    n_r, n_c = 30, 30
    g = np.zeros((n_r, n_c, 3))
    g[..., 0] = 20 + np.arange(n_c)[None] / s
    g[..., 1] = 20 + np.arange(n_r)[:, None] / s
    g[..., 2] = 80.0
    return write_tifxyz(tmp_path / 'plane', g, s)


def render_from_ct(piece, Z):
    """Exact render: layers 7 and 8 = CT at the px xyz (registered convention), others a copy."""
    g, v = rc.load_grid(piece)
    s = rc.scale(piece)
    shape = tuple(int(round(n / s)) for n in v.shape)
    ok = rc.px_valid(v, shape, s)
    rr, cc = np.nonzero(ok)
    img = np.zeros(shape, np.float32)
    img[rr, cc] = rc.ct_sample(Z, rc.px_xyz(g, rr, cc, s, 0.5))
    V = np.repeat(np.clip(img, 1, 255).astype(np.uint8)[None], 16, 0)
    V[:, ~ok] = 0
    return V, ok


def small_cfg():
    c = json.loads(json.dumps(CFG))
    c['frame']['n'] = 5000
    return c


def test_frame_check_passes_on_a_registered_render(tmp_path):
    Z = ct_field()
    piece = plane_piece(tmp_path)
    V, _ = render_from_ct(piece, Z)
    f = rc.frame_check(piece, V, Z, small_cfg())
    assert f['ok'] and f['r'] > 0.99 and f['rim']['share'] == 0 and f['shape_is_nodes_over_scale']


def test_rim_at_the_edge_is_dropped_and_passes(tmp_path):
    from scipy import ndimage
    Z = ct_field()
    piece = plane_piece(tmp_path)
    V, ok = render_from_ct(piece, Z)
    edge = ok & ~ndimage.binary_erosion(ok, iterations=3)
    rr, cc = np.nonzero(edge)
    k = np.random.default_rng(0).choice(len(rr), int(0.005 * ok.sum()), replace=False)   # 0.5 % at the edge
    V[:, rr[k], cc[k]] = 0
    f = rc.frame_check(piece, V, Z, small_cfg())
    assert f['ok'] and f['rim']['at_edge'] == 1.0 and 0 < f['rim']['share'] <= 0.01
    assert f['reading']['r_all_px'] < f['r']                 # the 0s pull r down when kept (the w025 VOID)


def test_interior_no_data_fails_the_rim_gate(tmp_path):
    Z = ct_field()
    piece = plane_piece(tmp_path)
    V, ok = render_from_ct(piece, Z)
    V[:, 60:64, 60:64] = 0                                     # a hole in the middle
    f = rc.frame_check(piece, V, Z, small_cfg())
    assert not f['ok'] and f['rim']['at_edge'] < 1.0 and f['r'] > 0.99


def fixed_cfg():
    c = small_cfg()
    c['frame']['rim_limit'] = 'fixed'                          # the F-576 .. F-623 rule
    return c


def test_rim_share_above_limit_fails(tmp_path):
    from scipy import ndimage
    Z = ct_field()
    piece = plane_piece(tmp_path)
    V, ok = render_from_ct(piece, Z)
    edge = ok & ~ndimage.binary_erosion(ok, iterations=3)
    V[:, edge] = 0                                             # whole 3 px rim: > 1 % of px
    f = rc.frame_check(piece, V, Z, fixed_cfg())
    assert f['rim']['share'] > 0.01 and f['rim']['at_edge'] == 1.0 and not f['ok']
    assert f['rim']['limit'] == 'fixed'


# ---------------------------------------------------------------- edge-band rim limit (VES-3868, F-623 attempt 2)
def test_config_uses_the_edge_band_limit():
    assert CFG['frame']['rim_limit'] == 'edge_band'
    c = small_cfg()
    c['frame'].pop('rim_limit')                                # a config without the key keeps the fixed rule
    assert c['frame'].get('rim_limit', 'fixed') == 'fixed'


def test_whole_edge_band_no_data_passes_the_edge_band_limit(tmp_path):
    from scipy import ndimage
    Z = ct_field()
    piece = plane_piece(tmp_path)
    V, ok = render_from_ct(piece, Z)
    edge = ok & ~ndimage.binary_erosion(ok, iterations=3)
    V[:, edge] = 0                                             # the case the fixed rule fails (test above)
    f = rc.frame_check(piece, V, Z, small_cfg())
    assert f['ok'] and f['rim']['limit'] == 'edge_band' and f['rim']['at_edge'] == 1.0
    assert abs(f['rim']['share'] - f['rim']['band_share']) < 1e-12 and f['rim']['max'] == f['rim']['band_share']
    assert f['rim']['band_fill'] == 1.0 and f['rim']['fixed_max'] == 0.01


def test_edge_band_limit_still_fails_interior_no_data(tmp_path):
    """negative control NC-IN: no-data px planted outside the edge band fail, though the share is far below the
    band share and the rim itself is full."""
    from scipy import ndimage
    Z = ct_field()
    piece = plane_piece(tmp_path)
    V, ok = render_from_ct(piece, Z)
    edge = ok & ~ndimage.binary_erosion(ok, iterations=3)
    rr, cc = np.nonzero(edge)
    k = np.random.default_rng(3).choice(len(rr), len(rr) // 2, replace=False)
    V[:, rr[k], cc[k]] = 0                                     # half the band unrendered
    V[:, 70, 70] = 0                                           # one interior px
    f = rc.frame_check(piece, V, Z, small_cfg())
    assert not f['ok'] and f['rim']['at_edge'] < 1.0 and f['rim']['share'] < f['rim']['band_share']


def test_edge_band_limit_still_fails_a_shifted_render(tmp_path):
    """negative control NC-SHIFT: a render rolled 6 px fails on FC r, the rim gate does not rescue it."""
    Z = ct_field()
    piece = plane_piece(tmp_path)
    V, ok = render_from_ct(piece, Z)
    V = np.roll(V, 6, axis=2)
    V[:, ~ok] = 0
    f = rc.frame_check(piece, V, Z, small_cfg())
    assert f['r'] < 0.95 and not f['ok']


def test_edge_band_limit_on_a_long_thin_strip(tmp_path):
    """the F-623 failure: a strip with a long edge per area and the half-cell rim unrendered. The fixed rule fails
    it at FC r > 0.99, the edge-band rule passes it."""
    Z = ct_field((120, 260, 260))
    s, n_r, n_c = 0.05, 3, 12
    g = np.zeros((n_r, n_c, 3))
    g[..., 0] = 10 + np.arange(n_c)[None] / s
    g[..., 1] = 10 + np.arange(n_r)[:, None] / s
    g[..., 2] = 80.0
    piece = write_tifxyz(tmp_path / 'strip', g, s)
    V, ok = render_from_ct(piece, Z)
    rr, cc = np.nonzero(band_px(ok, 0, 10))
    k = np.random.default_rng(2).choice(len(rr), int(0.75 * len(rr)), replace=False)   # 75 % band fill (F-623 median 76 %)
    V[:, rr[k], cc[k]] = 0
    f = rc.frame_check(piece, V, Z, small_cfg())
    g2 = rc.frame_check(piece, V, Z, fixed_cfg())
    assert f['r'] > 0.99 and f['ok'] and not g2['ok'] and g2['rim']['share'] > g2['rim']['max']
    assert abs(f['rim']['band_fill'] - 0.75) < 0.01


def test_unknown_rim_limit_is_refused(tmp_path):
    Z = ct_field()
    piece = plane_piece(tmp_path)
    V, _ = render_from_ct(piece, Z)
    c = small_cfg()
    c['frame']['rim_limit'] = 'none'
    with pytest.raises(ValueError):
        rc.frame_check(piece, V, Z, c)


def test_misregistered_render_fails_fc(tmp_path):
    Z = ct_field()
    piece = plane_piece(tmp_path)
    V, _ = render_from_ct(piece, Z)
    V = np.roll(V, 6, axis=2)                                  # 6 px off along the columns
    f = rc.frame_check(piece, V, Z, small_cfg())
    assert f['r'] < 0.95 and not f['ok']


def test_render_cmd_has_the_f576_arguments(tmp_path):
    cmd = rc.render_cmd(tmp_path / 'p.tifxyz', tmp_path / 'ct.zarr', tmp_path / 'out' / 'r.zarr', CFG, '1:1')
    s = ' '.join(cmd)
    for frag in ('--group-idx 0', '--scale 1', '--num-slices 16', '--slice-step 1', '--cache-gb 4',
                 '--pyramid false', 'inkcheck:ves572', '--cpus 4', '--memory 8g'):
        assert frag in s, frag


# ---------------------------------------------------------------- rim width in node cells (VES-3539 attempt 2)
def coarse_piece(tmp_path, s=0.05, n=12):
    """Flat piece at z = 80, 20 vx per node (the f429 route's grid step)."""
    g = np.zeros((n, n, 3))
    g[..., 0] = 10 + np.arange(n)[None] / s
    g[..., 1] = 10 + np.arange(n)[:, None] / s
    g[..., 2] = 80.0
    return write_tifxyz(tmp_path / 'coarse', g, s)


def band_px(ok, lo, hi):
    """mesh-valid px whose erosion depth is in (lo, hi] (cross erosions, as frame_check)."""
    from scipy import ndimage
    a = ok & ~ndimage.binary_erosion(ok, iterations=hi)
    b = ok & ~ndimage.binary_erosion(ok, iterations=lo) if lo else np.zeros_like(ok)
    return a & ~b


def coarse_case(tmp_path, lo, hi, share):
    Z = ct_field((120, 260, 260))
    piece = coarse_piece(tmp_path)
    V, ok = render_from_ct(piece, Z)
    rr, cc = np.nonzero(band_px(ok, lo, hi))
    k = np.random.default_rng(1).choice(len(rr), int(share * ok.sum()), replace=False)
    V[:, rr[k], cc[k]] = 0
    return piece, V, Z


def test_half_cell_rim_on_a_coarse_grid_passes_and_the_px_rule_refused_it(tmp_path):
    piece, V, Z = coarse_case(tmp_path, 0, 10, 0.019)          # the attempt-1 pieces: 1.9 %, all within 10 px
    f = rc.frame_check(piece, V, Z, fixed_cfg())
    assert f['rim']['px'] == 10 and abs(f['rim']['max'] - 0.01 * 10 / 3) < 1e-12
    assert f['ok'] and f['rim']['at_edge'] == 1.0 and f['r'] > 0.99
    old = fixed_cfg()
    old['frame'].pop('rim_cells')                              # the attempt-1 rule (3 px, 1 %)
    g = rc.frame_check(piece, V, Z, old)
    assert not g['ok'] and g['rim']['px'] == 3 and g['rim']['at_edge'] < 1.0


def test_no_data_deeper_than_half_a_cell_fails_on_a_coarse_grid(tmp_path):
    piece, V, Z = coarse_case(tmp_path, 10, 14, 0.005)
    f = rc.frame_check(piece, V, Z, small_cfg())
    assert not f['ok'] and f['rim']['at_edge'] == 0.0


def test_interior_hole_fails_on_a_coarse_grid(tmp_path):
    piece, V, Z = coarse_case(tmp_path, 0, 10, 0.01)
    V[:, 100:110, 100:110] = 0
    f = rc.frame_check(piece, V, Z, small_cfg())
    assert not f['ok'] and f['rim']['at_edge'] < 1.0


def test_coarse_rim_share_above_the_scaled_limit_fails(tmp_path):
    piece, V, Z = coarse_case(tmp_path, 0, 10, 0.04)            # 4 % > 3.33 %
    f = rc.frame_check(piece, V, Z, fixed_cfg())
    assert f['rim']['at_edge'] == 1.0 and f['rim']['share'] > f['rim']['max'] and not f['ok']


def test_rim_width_is_3_px_and_1_percent_at_scale_0_2(tmp_path):
    Z = ct_field()
    piece = plane_piece(tmp_path)
    V, _ = render_from_ct(piece, Z)
    f = rc.frame_check(piece, V, Z, fixed_cfg())
    assert f['rim']['px'] == 3 and f['rim']['max'] == 0.01
