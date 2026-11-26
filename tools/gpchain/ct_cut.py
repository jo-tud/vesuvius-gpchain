#!/usr/bin/env python3
"""GP chain stage "ct_cut": cut each band piece to the CT-valid region before the render (VES-3760, F-610 attempt 3).

Why: F-610 (VES-3706). On a full Paris 4 band the outer 38 windings of the fit run past the CT's valid region (the
masked open-data volume is 0 there, or its chunk is not in the store); villa renders 0 in all 16 layers there and the
rim gate refuses the whole piece (78 pieces: CT0 47, NOCHUNK 31, DATA 0). The cut removes those nodes first, so the
gate judges the part of the piece that has CT.

Rule (fixed, no parameter; README "Attempt 3")
  a valid node is CT-valid iff
    (a) its level-0 (render group) 128^3 chunk file exists in the CT store (absent = the remote store has no chunk:
        the fetch wrote every chunk under the meshes it could get, 404 = absent), and
    (b) the trilinear CT (cval 0) is > 0 at one or more of its 16 render-layer positions p + (j - 7.5) n, j = 0..15,
        n = unit mean normal of the complete grid quads at the node (quad normal = (dg/dcol) x (dg/drow), the quad's
        mean edge vectors; the sign does not matter, the 16 positions are symmetric): villa's no-data px are the px
        that are 0 in all 16 layers (render_contract frame_check), so this is the same test at the node.
  Nodes that touch no complete quad render no px and are left as they are. CT-invalid nodes are set to -1 (the tifxyz
  invalid value) in x/y/z.tif. A piece with no node cut is not written (bytes unchanged). A piece with no complete quad
  left is cut away: moved to OUT/ct-cut-away/ and not rendered. meta.json of a cut piece gets bbox of what is left and
  "ct_cut" (counts); area_cm2 / area_vx2 still describe the piece before the cut (meta "ct_cut" says so).

Quad-level test (--px, VES-3782, F-610 attempt 4; README "Attempt 4"). Why: attempt 3 left CT0 px inside quads whose 4
nodes passed (b) (75 pieces refused on at_edge alone). After the node cut, every render px of a complete quad is
tested the way villa 47111d1a renders it (QuadSurface::gen linear + readMultiSliceImpl, uint8):
  px p <-> grid (p + 0.5) * scale (render_contract); position = bilinear in its quad; normal = the unit normal of the
  NEAREST node (round), (g[r, c+1] - g[r, c-1]) x (g[r+1, c] - g[r-1, c]), none at the grid border or if any of the 4
  neighbours is invalid (villa then writes no data: the half-cell rim, not tested here); layer j at
  position + (j - 7.5) n; villa writes uint8(trilinear) (truncation): 0 iff trilinear CT < 1; the test uses
  trilinear < ZERO_BELOW = 2.0 (one grey level of margin for float32 positions, README attempt 4 step M).
  A tested px is CT0 iff all 16 layers are < 1. For every CT0 px its nearest node (the node villa took the normal
  from, a corner of its quad) is set to -1: the quad is dropped, and every px left in a complete quad either keeps the
  normal it was tested with (it has data) or loses its normal (rim). No tolerance, no parameter.

Contract
  input   band pieces SUBMISSION/meshes/column_NN.tifxyz (volume voxels of the render CT group 0), CT_ROOT zarr
  output  the same pieces, cut in place; returns the chain.json "ct_cut" section (per piece: nodes valid before /
          after, cut, cut by (a) / by (b), quads before / after, kept). human_inputs [] (0 minutes).
Runs in the villa #2401 venv (numpy, scipy, tifffile, zarr). Deterministic.

Usage: python -I ct_cut.py PIECE.tifxyz CT_ROOT.zarr [--px] [--dry-run]   (one piece; JSON on stdout)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # python -I: no script dir on the path
import render_contract as rcm  # noqa: E402

STAGE = 'ct_cut'
CH = 128
BATCH = 1 << 19                    # px per sampling batch (memory only; the result does not depend on it)
ZERO_BELOW = 2.0                   # quad-level test: a layer counts as 0 iff trilinear CT < ZERO_BELOW. villa writes 0
                                   # below 1 (uint8 truncation); README attempt 4 step M found villa no-data px the model
                                   # read as 1.0002-1.0069 (float32 positions), so by its fixed rule: 2.0 (one grey level)


def quads(v):
    return v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]


def node_normals(g, v):
    """(unit mean normal per node, has): mean of the unit normals of the complete quads that touch the node."""
    q = quads(v)
    dc = 0.5 * ((g[:-1, 1:] - g[:-1, :-1]) + (g[1:, 1:] - g[1:, :-1]))
    dr = 0.5 * ((g[1:, :-1] - g[:-1, :-1]) + (g[1:, 1:] - g[:-1, 1:]))
    n = np.cross(dc, dr)
    L = np.linalg.norm(n, axis=-1)
    q &= L > 0
    n = np.where(q[..., None], n / np.where(L > 0, L, 1)[..., None], 0.0)
    acc = np.zeros(g.shape)
    for a, b in ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(None, -1)),
                 (slice(None, -1), slice(1, None)), (slice(1, None), slice(1, None))):
        acc[a, b] += n
    L = np.linalg.norm(acc, axis=-1)
    has = L > 0
    return acc / np.where(has, L, 1)[..., None], has


def ct_valid(g, v, ct_root, Z):
    """(test mask, ok_a, ok_b) over the node grid: test = valid nodes touching a complete quad; ok_* on those."""
    n, has = node_normals(g, v)
    test = v & has
    p, nn = g[test], n[test]
    shape = np.array(Z.shape[::-1], float)                   # x, y, z
    inside = np.all(p >= 0, 1) & np.all(p <= shape - 1, 1)
    key = np.floor(np.clip(p, 0, shape - 1)[:, ::-1] / CH).astype(int)    # z, y, x chunk index
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    have = np.array([(Path(ct_root) / '0' / str(k[0]) / str(k[1]) / str(k[2])).exists() for k in uk])
    ok_a = inside & have[inv.ravel()]
    off = (np.arange(rcm.NSL) - (rcm.NSL - 1) / 2.0)          # villa layer j at (j - 7.5) vx along the normal
    q = (p[:, None, :] + off[None, :, None] * nn[:, None, :]).reshape(-1, 3)
    mx = rcm.ct_sample(Z, q).reshape(len(p), rcm.NSL).max(1)
    ok_b = mx > 0
    return test, ok_a, ok_b


# ---------------------------------------------------------------- quad-level test (--px, VES-3782)
class ChunkCT:
    """Trilinear uint8 CT from the raw 128^3 chunk files of CT_ROOT/0 (absent chunk or outside the array = 0, as the
    zarr fill value and villa's sampler); chunks cached (LRU)."""

    def __init__(self, ct_root, cache=128):
        import functools
        a = json.loads((Path(ct_root) / '0' / '.zarray').read_text())
        assert a['chunks'] == [CH] * 3 and a['dtype'] == '|u1' and a['compressor'] is None and a['order'] == 'C', a
        assert a.get('dimension_separator') == '/' and a['fill_value'] == 0, a
        self.root, self.shape = Path(ct_root) / '0', np.array(a['shape'])          # z, y, x
        self.chunk = functools.lru_cache(maxsize=cache)(self._chunk)

    def _chunk(self, k):
        f = self.root / str(k[0]) / str(k[1]) / str(k[2])
        if any(k[i] < 0 or k[i] * CH >= self.shape[i] for i in range(3)) or not f.exists():
            return np.zeros((CH,) * 3, np.uint8)
        return np.fromfile(f, np.uint8).reshape((CH,) * 3)

    def block(self, k):
        """(CH + 1)^3 voxels from chunk k (z, y, x) on, the +1 face from the next chunks."""
        b = np.empty((CH + 1,) * 3, np.uint8)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    c = self.chunk((k[0] + dz, k[1] + dy, k[2] + dx))
                    b[dz * CH:dz * CH + (1 if dz else CH), dy * CH:dy * CH + (1 if dy else CH),
                      dx * CH:dx * CH + (1 if dx else CH)] = c[:1 if dz else CH, :1 if dy else CH, :1 if dx else CH]
        return b

    def __call__(self, xyz):
        """trilinear CT at xyz (n, 3), float64."""
        q = xyz[:, ::-1]                                                     # z, y, x
        i0 = np.floor(q).astype(np.int64)
        f = q - i0
        out = np.zeros(len(q))
        key = np.floor_divide(i0, CH)
        order = np.lexsort(key.T[::-1])
        ks = key[order]
        cut = np.flatnonzero(np.any(ks[1:] != ks[:-1], 1)) + 1
        for grp in np.split(order, cut):
            if not len(grp):
                continue
            k = tuple(int(x) for x in key[grp[0]])
            b = self.block(k)
            li, lf = i0[grp] - np.array(k) * CH, f[grp]
            acc = np.zeros(len(grp))
            for dz in (0, 1):
                wz = lf[:, 0] if dz else 1 - lf[:, 0]
                for dy in (0, 1):
                    wy = lf[:, 1] if dy else 1 - lf[:, 1]
                    for dx in (0, 1):
                        wx = lf[:, 2] if dx else 1 - lf[:, 2]
                        acc += wz * wy * wx * b[li[:, 0] + dz, li[:, 1] + dy, li[:, 2] + dx]
            out[grp] = acc
        # villa ChunkSampler::inBounds: outside the array no sample (0); the store covers every piece here
        out[np.any(q < 0, 1) | np.any(q > self.shape - 1, 1)] = 0.0
        return out


def villa_normals(g, v):
    """(unit normal, has) per node as villa 47111d1a grid_normal_int: (g[r, c+1] - g[r, c-1]) x (g[r+1, c] - g[r-1, c]);
    none at the grid border, at an invalid node, if any of the 4 neighbours is invalid or the cross is 0."""
    n = np.zeros(g.shape)
    has = np.zeros(v.shape, bool)
    c = np.cross(g[1:-1, 2:] - g[1:-1, :-2], g[2:, 1:-1] - g[:-2, 1:-1])
    L = np.linalg.norm(c, axis=-1)
    ok = v[1:-1, 1:-1] & v[1:-1, 2:] & v[1:-1, :-2] & v[2:, 1:-1] & v[:-2, 1:-1] & (L > 0)
    has[1:-1, 1:-1] = ok
    n[1:-1, 1:-1] = np.where(ok[..., None], c / np.where(L > 0, L, 1)[..., None], 0.0)
    return n, has


def px_grid(v, s, has):
    """render px grid (round(nodes / scale)): (ok = px in a complete quad, normal = ok and its nearest node has a villa
    normal (`has`), quad row / col, nearest node row / col, grid coordinate u (rows) / w (cols))."""
    H, W = (int(round(n / s)) for n in v.shape)
    q = quads(v)
    u = (np.arange(H) + 0.5) * s
    w = (np.arange(W) + 0.5) * s
    qi, qj = np.floor(u).astype(int), np.floor(w).astype(int)
    ni, nj = np.floor(u + 0.5).astype(int), np.floor(w + 0.5).astype(int)
    okr, okc = qi < q.shape[0], qj < q.shape[1]
    ok = np.zeros((H, W), bool)
    ok[np.ix_(okr, okc)] = q[np.ix_(qi[okr], qj[okc])]
    nrm_ok = np.zeros((H, W), bool)
    nrm_ok[np.ix_(ni < v.shape[0], nj < v.shape[1])] = has[np.ix_(ni[ni < v.shape[0]], nj[nj < v.shape[1]])]
    nrm_ok &= ok
    return ok, nrm_ok, qi, qj, ni, nj, u, w


def px_model(g, v, s, ct):
    """villa's render of the piece, modelled per px (docstring "Quad-level test"):
    (ok, normal, ct0, nearest node row, col) on the render px grid (round(nodes / scale)); ok = px in a complete quad,
    normal = its nearest node has a villa normal, ct0 = tested (ok & normal) and all 16 layers trilinear < 1."""
    nrm, has = villa_normals(g, v)
    ok, nrm_ok, qi, qj, ni, nj, u, w = px_grid(v, s, has)
    ct0 = np.zeros(ok.shape, bool)
    off = np.arange(rcm.NSL) - (rcm.NSL - 1) / 2.0                           # villa buildOffsetList, step 1
    rows = np.flatnonzero(nrm_ok.any(1))
    per = np.cumsum(nrm_ok[rows].sum(1))
    for blk in np.split(rows, np.searchsorted(per, np.arange(BATCH, per[-1], BATCH)) if len(rows) else []):
        rr, cc = np.nonzero(nrm_ok[blk])                                     # px batch (~BATCH px of whole rows)
        rr = blk[rr]
        i, a = qi[rr], (u[rr] - qi[rr])[:, None]
        j, b = qj[cc], (w[cc] - qj[cc])[:, None]
        p = (1 - a) * ((1 - b) * g[i, j] + b * g[i, j + 1]) + a * ((1 - b) * g[i + 1, j] + b * g[i + 1, j + 1])
        n = nrm[ni[rr], nj[cc]]
        # pass 1: the two middle layers (offsets -0.5, +0.5); a px with data there needs no more samples
        mid = np.maximum(ct(p - 0.5 * n), ct(p + 0.5 * n)) >= ZERO_BELOW
        rest = np.flatnonzero(~mid)
        if len(rest):
            lay = (p[rest, None, :] + off[None, :, None] * n[rest, None, :]).reshape(-1, 3)
            ct0[rr[rest], cc[rest]] = ct(lay).reshape(len(rest), rcm.NSL).max(1) < ZERO_BELOW
    return ok, nrm_ok, ct0, ni, nj


def px_cut(g, v, s, ct):
    """nodes to cut so that no CT0 px is left in a complete quad: the nearest node of every CT0 px."""
    ok, nrm_ok, ct0, ni, nj = px_model(g, v, s, ct)
    rr, cc = np.nonzero(ct0)
    bad = np.zeros(v.shape, bool)
    bad[ni[rr], nj[cc]] = True
    q = quads(v)
    qb = np.zeros(q.shape, bool)
    qb[np.floor((rr + 0.5) * s).astype(int), np.floor((cc + 0.5) * s).astype(int)] = True
    info = dict(px_in_quads=int(ok.sum()), px_tested=int(nrm_ok.sum()), px_no_normal=int((ok & ~nrm_ok).sum()),
                px_ct0=int(len(rr)), quads_with_ct0=int(qb.sum()))
    return bad, info


def cut_piece(piece, ct_root, Z=None, dry_run=False, px=False, ct=None):
    """cut one piece in place; returns its row for chain.json "ct_cut". px: + the quad-level test (VES-3782)."""
    import tifffile
    piece = Path(piece)
    Z = Z if Z is not None else rcm.open_array(Path(ct_root) / '0')
    g, v = rcm.load_grid(piece)
    test, ok_a, ok_b = ct_valid(g, v, ct_root, Z)
    bad = np.zeros(v.shape, bool)
    bad[test] = ~(ok_a & ok_b)
    v1 = v & ~bad
    if px:
        bad_px, px_info = px_cut(g, v1, rcm.scale(piece), ct if ct is not None else ChunkCT(ct_root))
        bad |= bad_px
    v2 = v & ~bad
    row = dict(nodes_valid_before=int(v.sum()), nodes_tested=int(test.sum()), nodes_cut=int(bad.sum()),
               cut_no_chunk=int((~ok_a).sum()), cut_ct0_only=int((ok_a & ~ok_b).sum()),
               nodes_valid_after=int(v2.sum()), quads_before=int(quads(v).sum()), quads_after=int(quads(v2).sum()))
    if px:
        row.update(cut_node_test=int((v & ~v1).sum()), cut_px_test=int((v1 & ~v2).sum()),
                   quads_after_node_test=int(quads(v1).sum()), px=px_info)
    row['kept'] = row['quads_after'] > 0
    row['written'] = bool(row['nodes_cut'] and row['kept'] and not dry_run)
    if row['written']:
        for k, c in enumerate('xyz'):
            f = piece / f'{c}.tif'
            a = tifffile.imread(f)
            a[bad] = -1
            tifffile.imwrite(f, a)
        mp = piece / 'meta.json'
        meta = json.loads(mp.read_text())
        good = g[v2]
        meta['bbox'] = [good.min(0).tolist(), good.max(0).tolist()]
        meta['ct_cut'] = dict(row, rule='tools/gpchain/ct_cut.py (VES-3760' + (', --px VES-3782)' if px else ')'),
                              note='area_cm2 / area_vx2 are the piece before the cut')
        mp.write_text(json.dumps(meta, indent=4) + '\n')
    return row


def ct_cut_stage(out, meshes, ct_root, px=False):
    """cut every band piece; cut-away pieces move to OUT/ct-cut-away. Returns (kept meshes, chain.json section).
    px: + the quad-level test (VES-3782)."""
    Z = rcm.open_array(Path(ct_root) / '0')
    ct = ChunkCT(ct_root) if px else None
    sec = dict(stage=STAGE, ct_root=str(ct_root), rule='ct_cut.py docstring (VES-3760' + (
               '; quad-level test --px, VES-3782)' if px else ')'), px=px, pieces={}, human_inputs=[], human_minutes=0)
    kept = []
    for m in meshes:
        row = cut_piece(m, ct_root, Z, px=px, ct=ct)
        sec['pieces'][m.name] = row
        if row['kept']:
            kept.append(m)
        else:
            away = out / 'ct-cut-away'
            away.mkdir(exist_ok=True)
            shutil.move(str(m), str(away / m.name))
        print(f"chain: {m.name} ct_cut {row['nodes_cut']}/{row['nodes_valid_before']} nodes "
              f"(no chunk {row['cut_no_chunk']}, CT 0 {row['cut_ct0_only']}"
              + (f", px test {row['cut_px_test']} for {row['px']['px_ct0']} CT0 px" if px else '')
              + f"), kept {row['kept']}", flush=True)
    return kept, sec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('piece', type=Path)
    ap.add_argument('ct_root', type=Path)
    ap.add_argument('--dry-run', action='store_true', help='count only, do not write')
    ap.add_argument('--px', action='store_true', help='+ the quad-level test (VES-3782)')
    a = ap.parse_args(argv)
    print(json.dumps(cut_piece(a.piece, a.ct_root, dry_run=a.dry_run, px=a.px), indent=1))


if __name__ == '__main__':
    main()
