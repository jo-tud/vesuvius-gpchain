#!/usr/bin/env python3
"""GP chain render contract: villa render of a band piece -> per-mesh layer order + frame check (FC) -> ink map.

Why: F-576 (VES-3453). Two contract gaps showed up between our fit's band pieces and the ink models:
  1. Layer order. villa `vc_render_tifxyz` layer j sits at offset (j - 7.5) vx along the piece's grid normal.
     The hecate-family models want the depth order running inwards (F-005, F-184): on the label segments, whose
     normal points outwards, that is villa layers reversed (F-561). Our F-438 fit's normal points inwards (dot -0.99
     with the segments' normals), so the same reversal read the stack upside down (S 0.554). Attempt 2, villa
     order: S 0.932; confirmed 0.969 on fresh w026. So the order is derived per mesh from its own normal, never
     copied between meshes.
  2. FC. villa leaves a ~3 px no-data rim (0 in all layers, ~0.6 % of mesh-valid px) that our 4-valid-node px mask
     counts as valid; those 0s pulled FC r from 0.990 to 0.949 (w025, VOID). FC is computed on render-data px, and a
     rim gate covers what that hides: no-data <= 1 % of mesh-valid px, all within 3 px of the mask edge.

Layer-order rule (render_config.json "layer_order"; each piece on its own)
  grid normal n = (dg/dcol) x (dg/drow) at `n` random valid nodes (seed). Reference "segments": dot with the normal of
  the nearest node of a label segment mesh (<= d_max vx). The segments' villa render needs reversing (F-561), so
  the model's inward order is -n_seg: median dot < 0 -> "villa" (model plane j = villa layer j), > 0 -> "reversed"
  (plane j = layer 15 - j). Needs >= min_near nodes near a segment and >= min_agree of them on the median's side.
  Reference "radial" (only when no segment is near, needs --axis-xy): dot with the outward direction from the
  axis; inward (< 0) -> "villa". If both are available, the radial median sign must agree with the segments' order
  or the order is None (refused). Order None -> `infer` refuses that piece.

FC (render_config.json "frame")
  `n` random mesh-valid px (seed); px p <-> node (p + 0.5) * scale (F-561 P0). Pearson r between the mean of villa
  layers 7 and 8 and our trilinear CT sample at the px xyz, on px where the render has data (not 0 in all layers):
  r >= bar. Rim gate: render no-data px all within w px of the mask edge (the edge band), w = max(rim_px,
  ceil(rim_cells / scale)) (half a node cell: villa's unrendered outer half cell, VES-3539), and a no-data share of
  mesh-valid px <= the limit: "rim_limit" "edge_band" (VES-3868) = the piece's own band share (implied by the first
  test), "fixed" (F-576 .. F-623, the default when the key is missing) = rim_max * w / rim_px.
  Readings: r on all sampled px, r half a px off. ok = both.

Usage (python -I):
  render_contract.py order PIECE --segments SEG_TIFXYZ... [--axis-xy X Y]          -> JSON on stdout
  render_contract.py frame PIECE RENDER.zarr CT_ARRAY                              -> JSON on stdout
  render_contract.py infer CHAIN_OUT --ckpt PTH --hecate HECATE_PY --name NAME [--device cuda] [--cap-h H]
      for every piece in CHAIN_OUT/chain.json "render" with frame ok and an order: ink map into
      CHAIN_OUT/maps/<NAME>_<piece>.npy (uint8, 0 outside mesh-valid px) + CHAIN_OUT/maps/<NAME>.json (seconds,
      refusals). Refuses pieces without FC ok or an order. --cap-h: stop before a piece once the seconds in this
      log and in --cap-files plus --piece-h pass the cap.
Functions are used by chain.py --render. CPU parts run in the villa #2401 venv; infer needs torch (ves663-gpu).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = HERE / 'render_config.json'
NSL = 16


def load_config(path=CONFIG):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- tifxyz (same rules as F-576 run.py)
def load_grid(d):
    import tifffile
    g = np.stack([tifffile.imread(Path(d) / f'{c}.tif').astype(np.float64) for c in 'xyz'], -1)
    v = np.all(np.isfinite(g), -1) & ~np.all(g == -1, -1) & (g[..., 2] >= 0)
    return g, v


def scale(d):
    return float(json.loads((Path(d) / 'meta.json').read_text())['scale'][0])


def px_valid(v, shape, s, half=0.5):
    """Image px whose 4 surrounding nodes are valid; px p <-> node (p + half) * s."""
    vv = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
    r = np.floor((np.arange(shape[0]) + half) * s).astype(int)
    c = np.floor((np.arange(shape[1]) + half) * s).astype(int)
    ok_r, ok_c = r < vv.shape[0], c < vv.shape[1]
    out = np.zeros(shape, bool)
    out[np.ix_(ok_r, ok_c)] = vv[np.ix_(r[ok_r], c[ok_c])]
    return out


def px_xyz(g, rr, cc, s, half=0.5):
    """xyz at image px (rr, cc): bilinear in the node grid at ((p + half) * s)."""
    from scipy import ndimage
    u, w = (rr + half) * s, (cc + half) * s
    return np.stack([ndimage.map_coordinates(g[..., k], [u, w], order=1) for k in range(3)], 1)


def ct_sample(Z, xyz):
    """Trilinear CT at xyz (n, 3) from a (z, y, x) array Z, cval 0; one read per 128^3 cell (+ halo)."""
    from scipy import ndimage
    out = np.zeros(len(xyz), np.float64)
    cell = np.floor(xyz / 128).astype(int)
    _, inv = np.unique(cell, axis=0, return_inverse=True)
    for c in range(int(inv.max()) + 1):
        m = inv.ravel() == c
        q = xyz[m]
        a = np.maximum(np.floor(q.min(0)).astype(int) - 2, 0)
        b = np.minimum(np.ceil(q.max(0)).astype(int) + 3, np.array(Z.shape[::-1]))
        sub = np.asarray(Z[a[2]:b[2], a[1]:b[1], a[0]:b[0]]).astype(np.float32)
        out[m] = ndimage.map_coordinates(sub, [q[:, 2] - a[2], q[:, 1] - a[1], q[:, 0] - a[0]], order=1,
                                         mode='constant', cval=0.0)
    return out


def open_array(path):
    import zarr
    return zarr.open_array(str(path), mode='r')


def open_render(path):
    import zarr
    return zarr.open_group(str(path), mode='r')['0']


# ---------------------------------------------------------------- layer order
def grid_normals(g, v):
    """(node xyz, unit normal) at nodes with their +col and +row neighbours valid; n = (dg/dcol) x (dg/drow)."""
    a = g[:-1, 1:] - g[:-1, :-1]
    b = g[1:, :-1] - g[:-1, :-1]
    n = np.cross(a, b)
    ok = v[:-1, :-1] & v[:-1, 1:] & v[1:, :-1]
    L = np.linalg.norm(n, axis=-1)
    ok &= L > 0
    n = n / np.where(L > 0, L, 1)[..., None]
    return g[:-1, :-1][ok], n[ok]


class SegmentNormals:
    """Nodes and grid normals of the label segment meshes (the reference whose villa render needs reversing)."""

    def __init__(self, seg_dirs):
        from scipy.spatial import cKDTree
        P, N = [], []
        self.names = []
        for s in seg_dirs:
            p, n = grid_normals(*load_grid(s))
            P.append(p)
            N.append(n)
            self.names.append(Path(s).name)
        self.P, self.N = np.concatenate(P), np.concatenate(N)
        self.tree = cKDTree(self.P)


class Umbilicus:
    """Scroll axis xy at z, linear between the control points of an umbilicus.json (x, y, z; held past the ends), as
    F-610's why_refused.py `umb` (VES-3760: the radial reference of F-610 attempt 3)."""

    def __init__(self, path):
        b = Path(path).read_bytes()
        self.path, self.sha256 = str(Path(path).resolve()), hashlib.sha256(b).hexdigest()
        self.a = np.array(sorted((c['z'], c['x'], c['y']) for c in json.loads(b)['control_points']), float)

    def __call__(self, z):
        return np.stack([np.interp(z, self.a[:, 0], self.a[:, 1]), np.interp(z, self.a[:, 0], self.a[:, 2])], -1)


def layer_order(piece, segs=None, axis_xy=None, cfg=None, umbilicus=None):
    """Per-mesh layer order of a villa render of `piece` for the inward-reading ink models: "villa", "reversed" or
    None (refused). segs: SegmentNormals or None. axis_xy: scroll axis (x, y) for the radial reference or None.
    umbilicus: Umbilicus (axis xy at each node's z) instead of axis_xy, or None."""
    if axis_xy is not None and umbilicus is not None:
        raise ValueError('layer_order: axis_xy or umbilicus, not both')
    c = (cfg or load_config())['layer_order']
    rng = np.random.default_rng(c['seed'])
    p, n = grid_normals(*load_grid(piece))
    k = rng.choice(len(p), min(c['n'], len(p)), replace=False)
    p, n = p[k], n[k]
    res = dict(n=int(len(p)), reference=None, order=None, why=None)
    rad = None
    if axis_xy is not None or umbilicus is not None:
        r = np.zeros_like(p)
        r[:, :2] = p[:, :2] - (umbilicus(p[:, 2]) if umbilicus is not None else np.asarray(axis_xy, float))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        rad = np.einsum('ij,ij->i', n, r)
        ref = (dict(umbilicus=umbilicus.path, umbilicus_sha256=umbilicus.sha256) if umbilicus is not None
               else dict(axis_xy=[float(x) for x in axis_xy]))
        res['radial'] = dict(**ref, dot_median=float(np.median(rad)), pos_share=float((rad > 0).mean()))
    if segs is not None:
        d, i = segs.tree.query(p, distance_upper_bound=c['d_max'], workers=c.get('workers', 8))
        m = np.isfinite(d)
        sd = np.einsum('ij,ij->i', n[m], segs.N[i[m]]) if m.any() else np.zeros(0)
        res['segments'] = dict(names=segs.names, d_max=c['d_max'], n_near=int(m.sum()),
                               dot_median=float(np.median(sd)) if m.any() else None,
                               pos_share=float((sd > 0).mean()) if m.any() else None)
        if m.sum() >= c['min_near']:
            agree = float(max((sd > 0).mean(), (sd < 0).mean()))
            res['segments']['agree'] = agree
            if agree < c['min_agree']:
                res['why'] = f'segment normal dot signs mixed (agree {agree:.3f} < {c["min_agree"]})'
                return res
            order = 'villa' if np.median(sd) < 0 else 'reversed'
            if rad is not None and (np.median(rad) < 0) != (order == 'villa'):
                res['why'] = 'radial reference disagrees with the segment reference'
                return res
            res.update(reference='segments', order=order,
                       why=f'piece normal {"opposite to" if order == "villa" else "along"} the label segments '
                           f'(median dot {np.median(sd):+.3f}); segments read reversed (F-561)')
            return res
    if rad is not None:
        agree = float(max((rad > 0).mean(), (rad < 0).mean()))
        res['radial']['agree'] = agree
        if agree < c['min_agree_radial']:
            res['why'] = f'radial dot signs mixed (agree {agree:.3f} < {c["min_agree_radial"]})'
            return res
        order = 'villa' if np.median(rad) < 0 else 'reversed'
        res.update(reference='radial', order=order,
                   why=f'piece normal points {"inwards" if order == "villa" else "outwards"} '
                       f'(median radial dot {np.median(rad):+.3f}); the models read inwards (F-005, F-184)')
        return res
    res['why'] = 'no reference: no segment within d_max on >= min_near nodes and no --axis-xy / --umbilicus'
    return res


def ordered_stack(V, order):
    """The villa render (NSL, H, W) in model plane order (contiguous uint8). order must be "villa" or "reversed"."""
    if order == 'villa':
        return np.ascontiguousarray(np.asarray(V[:]))
    if order == 'reversed':
        return np.ascontiguousarray(np.asarray(V[:])[::-1])
    raise ValueError(f'no layer order ({order!r}): refused')


# ---------------------------------------------------------------- FC + rim gate
def frame_check(piece, V, Z, cfg=None):
    """FC on render-data px + rim gate (F-576 confirmation attempt 2 rule). V: villa render (NSL, H, W); Z: CT."""
    from scipy import ndimage
    c = (cfg or load_config())['frame']
    rng = np.random.default_rng(c['seed'])
    assert V.shape[0] == NSL, V.shape
    g, v = load_grid(piece)
    s = scale(piece)
    ok = px_valid(v, V.shape[1:], s)
    full = np.asarray(V[:])
    nodata = ok & (full == 0).all(0)
    # rim width: villa leaves the outer half node cell unrendered (VES-3539: 100 % of no-data within 0.5 / scale px of
    # the mask edge on f439 (5 px / node) and f429 (20 px / node) pieces), so the width is in node cells, at least
    # rim_px; the share limit grows with the width. At scale 0.2 this is 3 px and rim_max, F-576 / F-585 unchanged.
    width = max(int(c['rim_px']), int(np.ceil(round(c.get('rim_cells', 0.0) / s, 6))))
    edge = ok & ~ndimage.binary_erosion(ok, iterations=width)
    rim_share = float(nodata.sum() / ok.sum())
    band_share = float(edge.sum() / ok.sum())
    # share limit. "fixed" (F-576 .. F-623): rim_max * w / rim_px of mesh-valid px. "edge_band" (VES-3868, F-623
    # attempt 2): the piece's own edge-band share. Villa's unrendered half cell fills 62-85 % of the band, so its share
    # follows the band share (r 0.973, 214 pieces); a fixed share failed CT-cut strips and small pieces with FC r
    # >= 0.989. With at_edge == 1.0 the no-data px are a subset of the band, so share <= band share then holds by
    # construction: the edge-band gate is FC r >= bar and every no-data px inside the half-cell band.
    limit = c.get('rim_limit', 'fixed')
    fixed_max = c['rim_max'] * width / c['rim_px']
    if limit == 'fixed':
        rim_max = fixed_max
    elif limit == 'edge_band':
        rim_max = band_share
    else:
        raise ValueError(f'rim_limit {limit!r}: not "fixed" or "edge_band"')
    # share of no-data px within rim_px of the mask edge; no no-data px at all passes (F-576 confirm.py gave 0 here,
    # which would have refused a render with no rim; w026 had 0.65 % no-data, so its numbers are unchanged)
    rim_at_edge = float((nodata & edge).sum() / nodata.sum()) if nodata.any() else 1.0
    rr, cc = np.nonzero(ok)
    k = rng.choice(len(rr), min(c['n'], len(rr)), replace=False)
    rr, cc = rr[k], cc[k]
    m78 = full[7:9].astype(np.float64)
    mid = (m78[0][rr, cc] + m78[1][rr, cc]) / 2
    d = ~nodata[rr, cc]
    x, x0 = ct_sample(Z, px_xyz(g, rr, cc, s, 0.5)), ct_sample(Z, px_xyz(g, rr, cc, s, 0.0))
    r = float(np.corrcoef(x[d], mid[d])[0, 1])
    return dict(render_shape=list(V.shape), nodes=list(v.shape), n_sampled=int(len(rr)), n=int(d.sum()), r=r,
                bar=c['bar'], rim=dict(share=rim_share, max=rim_max, at_edge=rim_at_edge, px=width,
                                              rim_px=c['rim_px'], rim_cells=c.get('rim_cells', 0.0), rim_max=c['rim_max'],
                                              limit=limit, band_share=band_share, fixed_max=fixed_max,
                                              band_fill=float(nodata.sum() / edge.sum()) if edge.any() else 0.0),
                shape_is_nodes_over_scale=[round(n / s) for n in v.shape] == list(V.shape[1:]),
                reading=dict(r_all_px=float(np.corrcoef(x, mid)[0, 1]),
                             r_half_px_off_data_px=float(np.corrcoef(x0[d], mid[d])[0, 1])),
                ok=bool(r >= c['bar'] and rim_share <= rim_max and rim_at_edge == 1.0))


# ---------------------------------------------------------------- render (villa, docker)
def render_cmd(piece, ct_root, dst, cfg=None, uid_gid=None):
    """docker command for villa vc_render_tifxyz on one piece (F-576 run.py render arguments)."""
    import os
    c = (cfg or load_config())['render']
    assert c['num_slices'] == NSL, c['num_slices']
    ug = uid_gid or f'{os.getuid()}:{os.getgid()}'
    mounts = sorted({str(Path(p).resolve().parent) for p in (piece, ct_root, dst)})
    cmd = ['docker', 'run', '--rm', '--memory', c['memory'], '--cpus', str(c['threads']), '--user', ug]
    for m in mounts:
        cmd += ['-v', f'{m}:{m}']
    home = Path(dst).resolve().parent
    cmd += ['-e', f'HOME={home}', '-e', f'VC3D_CONFIG_DIR={home}/vc3dcfg', '-e', f'OMP_NUM_THREADS={c["threads"]}',
            '--entrypoint', '/opt/vc3d/bin/vc_render_tifxyz', c['image'],
            '--volume', str(Path(ct_root).resolve()), '--group-idx', '0', '--scale', str(c['scale']),
            '--segmentation', str(Path(piece).resolve()), '--num-slices', str(c['num_slices']),
            '--slice-step', str(c['slice_step']), '--cache-gb', str(c['cache_gb']), '--pyramid', 'false',
            '--zarr-output', str(Path(dst).resolve())]
    return cmd


def render_image_ok(cfg=None):
    import subprocess
    c = (cfg or load_config())['render']
    got = subprocess.run(['docker', 'image', 'inspect', '-f', '{{.Id}}', c['image']], text=True,
                         capture_output=True).stdout.strip()
    return got[7:19] == c['image_id'], got


# ---------------------------------------------------------------- inference
def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_hecate(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('hecate_ref', path)
    hec = importlib.util.module_from_spec(spec)
    sys.modules['hecate_ref'] = hec
    spec.loader.exec_module(hec)
    return hec


def cmd_infer(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = a.chain_out.resolve()
    man = json.loads((out / 'chain.json').read_text())
    assert man.get('render'), 'chain.json has no render section (run chain.py --render)'
    md = out / 'maps'
    md.mkdir(exist_ok=True)
    logf = md / f'{a.name}.json'
    log = json.loads(logf.read_text()) if logf.exists() else dict(name=a.name, seconds={}, refused={})
    ck_sha = sha256(a.ckpt)
    if a.ckpt_sha and ck_sha != a.ckpt_sha:
        sys.exit(f'infer: {a.ckpt} sha256 {ck_sha} != {a.ckpt_sha}')
    log.update(ckpt=str(a.ckpt), ckpt_sha256=ck_sha, hecate=str(a.hecate), hecate_sha256=sha256(a.hecate))
    hec = model = None
    for name, r in man['render'].items():
        dst = md / f'{a.name}_{name}.npy'
        if dst.exists():
            continue
        if not (r['frame']['ok'] and r['order']['order'] in ('villa', 'reversed')):
            log['refused'][name] = dict(frame_ok=r['frame']['ok'], order=r['order']['order'], why=r['order']['why'])
            continue
        if a.cap_h is not None:
            others = [f for f in a.cap_files if f.resolve() != logf]
            spent = (sum(log['seconds'].values())
                     + sum(sum(json.loads(f.read_text())['seconds'].values()) for f in others if f.exists())) / 3600
            if spent + a.piece_h > a.cap_h:
                log['stopped'] = f'GPU cap: {spent:.3f} h used + {a.piece_h} h > {a.cap_h} h before {name}'
                print(log['stopped'], flush=True)
                break
        if model is None:
            hec = load_hecate(a.hecate)
            model = hec.load_model(str(a.ckpt), a.device)
        t0 = time.time()
        V = open_render(out / r['render'])
        v = ordered_stack(V, r['order']['order'])
        assert v.shape[0] == NSL, v.shape
        g, vv = load_grid(out / r['piece'])
        okp = px_valid(vv, v.shape[1:], scale(out / r['piece']))
        o2 = np.zeros(v.shape[1:], np.uint8)
        hec.predict(model, v, o2, reverse=False, batch_size=4)
        o2[~okp] = 0
        np.save(dst, o2)
        log['seconds'][name] = round(time.time() - t0, 1)
        logf.write_text(json.dumps(log, indent=1))
        print(a.name, name, log['seconds'][name], 's', flush=True)
    logf.write_text(json.dumps(log, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sp = ap.add_subparsers(dest='cmd', required=True)
    o = sp.add_parser('order')
    o.add_argument('piece', type=Path)
    o.add_argument('--segments', type=Path, nargs='+')
    o.add_argument('--axis-xy', type=float, nargs=2)
    f = sp.add_parser('frame')
    f.add_argument('piece', type=Path)
    f.add_argument('render', type=Path)
    f.add_argument('ct', type=Path, help='CT zarr array (z, y, x), e.g. ROOT/0')
    i = sp.add_parser('infer')
    i.add_argument('chain_out', type=Path)
    i.add_argument('--ckpt', type=Path, required=True)
    i.add_argument('--ckpt-sha', help='refuse unless the checkpoint has this sha256')
    i.add_argument('--hecate', type=Path, required=True)
    i.add_argument('--name', required=True)
    i.add_argument('--device', default='cuda')
    i.add_argument('--cap-h', type=float)
    i.add_argument('--piece-h', type=float, default=0.07, help='GPU-h estimate per piece for the cap check')
    i.add_argument('--cap-files', type=Path, nargs='*', default=[],
                   help='other infer logs (maps/*.json) whose seconds count towards --cap-h with this one\'s')
    a = ap.parse_args(argv)
    if a.cmd == 'order':
        segs = SegmentNormals(a.segments) if a.segments else None
        print(json.dumps(layer_order(a.piece, segs, a.axis_xy), indent=1))
    elif a.cmd == 'frame':
        print(json.dumps(frame_check(a.piece, open_render(a.render), open_array(a.ct)), indent=1))
    else:
        cmd_infer(a)


if __name__ == '__main__':
    main()
