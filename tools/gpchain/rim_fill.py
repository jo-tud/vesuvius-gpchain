#!/usr/bin/env python3
"""GP output chain, stage "rim-fill": give back the rim the stock regrid drops, after vc_obj2tifxyz.

Why: VES-2570 (F-437 fix 2b). The isometric stage (isometric_step.py) keeps 1.000 of the 3D area
through vc_tifxyz2obj and flatboi, then `vc_obj2tifxyz` re-grids the flat OBJ: a grid node is valid
only if it falls inside a flat triangle, and a quad only if all 4 corners are valid. On w130 (F-437)
that drops the outer ~0.5-1 grid step all round the band, 2.75 % of the footprint (2 121 partial
quads). The rule below makes the grid edge round to the nearest node instead of truncating inward.

Rule (geometry only, one parameter R in grid steps, default 0.5)
  grid map   the one vc_obj2tifxyz used: uv_min / uv_max over the face UVs (float32, as it parses
             them), node (row, col) <-> uv_min + (col, row) * (uv_max - uv_min) / (W - 1, H - 1).
  fill       an invalid node within R grid steps (UV distance / grid step, per axis) of the flat
             mesh footprint gets the 3D position of the nearest flat triangle's affine map, extended
             past its edge (barycentric weights may be negative). The affine map of a triangle is its
             own isometric flattening, so the new grid edges keep the triangle's stretch.
             Nearest = smallest point-triangle distance in grid coordinates over every face whose
             node bounding box (grown by R) holds the node; ties go to the lowest face index
             (vc_obj2tifxyz's own "first triangle wins" order). Exact for any triangle size.
  R = 0.5    a node is valid iff it lies within half a grid step of the surface: the grid edge then
             falls on average on the surface edge (for a straight edge at a random offset, mean
             undercover = mean overshoot = 1/8 step), instead of 1/2 step inside it (stock).
  inside     a node inside a flat triangle that the stock tool left invalid is filled too: it sits within
             the stock tool's fixed-point vertex snap (1/256 step) of an edge (VES-2570: 2 of 70 293 on
             the F-429 strip, depth <= 0.001 step); counted as inside_filled with its largest depth.
  never      nodes valid in the stock output are not touched (same float32 values); only nodes with
             a valid node within 2 grid steps are candidates, so the fill cannot grow islands.
  spike guard  --max-edge K (default 0 = off; VES-2570 attempt 4): a filled node is dropped again if its
             3D distance to any stock-valid 4-neighbour exceeds K x the median 3D edge of the
             stock grid (4-neighbour pairs, both valid). One pass, no iteration. Catches the affine map
             extended across a narrow UV slit whose two sides are far apart in 3D (w130: filled edges
             4.6-17.8 median edges, VES-2570 diag_fold.py). Counted as guard_dropped.
Contract
  input   IN.tifxyz written by `vc_obj2tifxyz OBJ IN.tifxyz ...` and that OBJ (3D vertices in volume
          voxels, x y z order; UVs in voxels; one or more faces).
  output  OUT.tifxyz, same grid, scale, meta and frame; x/y/z.tif rewritten (zlib), only filled nodes
          differ. OUT/rim_fill.json: parameters, sha256s, filled node count, valid 4-connected
          component counts before / after and the OBJ surface's own connected pieces (after < surface
          = the fill bridged separate pieces; after < before alone = it re-joined a stock raster
          that broke one piece, e.g. F-429 strip band piece 2: 4 -> 1, surface 1),
          largest filled distance, and the human-input ledger entry (none: 0 minutes).
Usage: python3 tools/gpchain/rim_fill.py IN.tifxyz OBJ OUT.tifxyz [--r 0.5] [--max-edge 0]
       (numpy, scipy, tifffile; runs in the villa #2401 venv)   Deterministic.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components

STAGE = "rim-fill"


def read_obj(path):
    """vertices (n, 3) x y z, uvs (m, 2) as float32 (vc_obj2tifxyz parses floats), faces (f, 3, 2) [v, vt]."""
    v, vt, f = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("vt "):
                vt.append(line.split()[1:3])
            elif line.startswith("f "):
                f.append([[int(t) - 1 for t in c.split("/")[:2]] for c in line.split()[1:4]])
    return (np.array(v, np.float32).astype(np.float64), np.array(vt, np.float32),
            np.array(f, np.int64))


def load(d):
    x, y, z = (tifffile.imread(Path(d) / f"{c}.tif") for c in "xyz")
    v = (x != -1) & (y != -1) & (z != -1) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x, y, z, v


def point_triangle(G, A, B, C, edge=False):
    """distance from points G to triangles ABC (2D, all (n, 2)) and the unclamped barycentric weights;
    edge=True also returns the distance to the nearest triangle edge (inside or out)."""
    v0, v1, v2 = B - A, C - A, G - A
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    with np.errstate(invalid="ignore", divide="ignore"):
        wb = (d11 * d20 - d01 * d21) / den
        wc = (d00 * d21 - d01 * d20) / den
    wa = 1.0 - wb - wc

    def seg(X, Y):
        e = Y - X
        t = np.clip(((G - X) * e).sum(-1) / np.maximum((e * e).sum(-1), 1e-30), 0.0, 1.0)
        return np.linalg.norm(G - (X + t[:, None] * e), axis=-1)

    inside = (wa >= 0) & (wb >= 0) & (wc >= 0)
    e = np.minimum(np.minimum(seg(A, B), seg(B, C)), seg(C, A))
    dist = np.where(inside, 0.0, e)
    dist = np.where(np.isfinite(den) & (den > 0), dist, np.inf)        # degenerate UV triangle: never
    w = np.stack([wa, wb, wc], -1)
    return (dist, w, e) if edge else (dist, w)


def fill(xyz, v, verts, uvs, faces, r, chunk=20000):
    """xyz (H, W, 3) x y z float32, v valid. Returns new xyz, new valid, filled mask, filled distance."""
    H, W = v.shape
    fuv = uvs[faces[..., 1]].astype(np.float64)                          # (f, 3, 2)
    umin = fuv.reshape(-1, 2).min(0).astype(np.float32).astype(np.float64)
    umax = fuv.reshape(-1, 2).max(0).astype(np.float32).astype(np.float64)
    step = np.maximum(umax - umin, 1e-12) / np.array([W - 1, H - 1], np.float64)
    g = (fuv - umin) / step                                              # grid coordinates (col, row)
    p3 = verts[faces[..., 0]]                                            # (f, 3, 3) x y z
    cand = ndimage.binary_dilation(v, structure=np.ones((3, 3), bool), iterations=2) & ~v
    best_d = np.full(v.shape, np.inf)
    best_f = np.full(v.shape, -1, np.int64)
    lo = np.floor(g.min(1) - r).astype(np.int64)                         # per-face node bbox (col, row)
    hi = np.ceil(g.max(1) + r).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array([W - 1, H - 1]))
    for f0 in range(0, len(g), chunk):                                   # every (face, node in its bbox)
        f = np.arange(f0, min(f0 + chunk, len(g)))
        nx = np.maximum(hi[f, 0] - lo[f, 0] + 1, 0)
        ny = np.maximum(hi[f, 1] - lo[f, 1] + 1, 0)
        n = nx * ny
        fi = np.repeat(f, n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        cc = lo[fi, 0] + k % np.repeat(nx, n)
        rr = lo[fi, 1] + k // np.repeat(nx, n)
        m = cand[rr, cc]
        fi, cc, rr = fi[m], cc[m], rr[m]
        d, _ = point_triangle(np.stack([cc, rr], -1).astype(np.float64), g[fi, 0], g[fi, 1], g[fi, 2])
        o = np.lexsort((fi, d))                                          # nearest first, ties: lowest face
        fi, cc, rr, d = fi[o], cc[o], rr[o], d[o]
        _, first = np.unique(rr * W + cc, return_index=True)
        fi, cc, rr, d = fi[first], cc[first], rr[first], d[first]
        better = (d < best_d[rr, cc]) | ((d == best_d[rr, cc]) & (fi < best_f[rr, cc]))
        best_d[rr[better], cc[better]] = d[better]
        best_f[rr[better], cc[better]] = fi[better]
    take = cand & (best_d <= r)
    rr, cc = np.nonzero(take)
    t = best_f[rr, cc]
    _, w, e = point_triangle(np.stack([cc, rr], -1).astype(np.float64), g[t, 0], g[t, 1], g[t, 2], edge=True)
    inside_depth = float(e[best_d[rr, cc] <= 0].max()) if (best_d[rr, cc] <= 0).any() else 0.0
    out = xyz.copy()
    out[rr, cc] = np.einsum("nk,nkc->nc", w, p3[t]).astype(np.float32)
    nv = v | take
    dist = np.where(take, best_d, 0.0)
    return out, nv, take, dist, inside_depth


def surface_components(faces):
    """connected pieces of the OBJ surface (faces joined by shared vertices)."""
    f = faces[..., 0]
    used, inv = np.unique(f, return_inverse=True)
    inv = inv.reshape(-1, 3)
    a = np.concatenate([inv[:, 0], inv[:, 1]])
    b = np.concatenate([inv[:, 1], inv[:, 2]])
    g = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(len(used), len(used)))
    return int(connected_components(g, directed=False)[0])


def sha256s(d):
    return {f: hashlib.sha256((Path(d) / f).read_bytes()).hexdigest() for f in ("x.tif", "y.tif", "z.tif")}


def spike_guard(xyz, v, nv, filled, k):
    """drop filled nodes with a 3D edge to a stock-valid 4-neighbour > k x the median stock edge (one pass;
    edges between two filled nodes are not tested, so one spike does not take its good neighbours along)."""
    p = xyz.astype(np.float64)
    eh = np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1)
    ev = np.linalg.norm(p[1:] - p[:-1], axis=-1)
    med = float(np.median(np.concatenate([eh[v[:, 1:] & v[:, :-1]], ev[v[1:] & v[:-1]]])))
    worst = np.zeros(v.shape)
    for e, a, b in ((eh, (slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                    (ev, (slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        worst[a] = np.maximum(worst[a], np.where(v[b], e, 0.0))     # node at a, stock neighbour at b
        worst[b] = np.maximum(worst[b], np.where(v[a], e, 0.0))
    drop = filled & (worst > k * med)
    return nv & ~drop, filled & ~drop, int(drop.sum()), med


def rim_fill(src, obj, out, r=0.5, max_edge=0.0):
    """the stage as a function (band_cuts.py calls it per piece). Returns the manifest dict."""
    src, obj, out = Path(src), Path(obj), Path(out)
    if out.exists():
        raise FileExistsError(f"rim_fill: {out} exists; refusing to overwrite")
    x, y, z, v = load(src)
    verts, uvs, faces = read_obj(obj)
    xyz, nv, filled, dist, inside_depth = fill(np.stack([x, y, z], -1), v, verts, uvs, faces, r)
    guard = {}
    if max_edge > 0:
        nv, filled, dropped, med = spike_guard(xyz, v, nv, filled, max_edge)
        dist = np.where(filled, dist, 0.0)
        guard = {"guard_dropped": dropped, "median_edge_vx": round(med, 4)}
    shutil.copytree(src, out)
    for i, c in enumerate("xyz"):
        a = np.where(nv, xyz[..., i], np.float32(-1)).astype(x.dtype)
        tifffile.imwrite(out / f"{c}.tif", a, compression="zlib")
    four = ndimage.generate_binary_structure(2, 1)
    man = {"stage": STAGE, "src": str(src.resolve()), "obj": str(obj.resolve()), "out": str(out.resolve()),
           "params": {"r_grid_steps": r, **({"max_edge_x_median": max_edge} if max_edge > 0 else {})},
           "in_sha256": sha256s(src), "obj_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
           "out_sha256": sha256s(out), "shape": list(v.shape),
           "valid_in": int(v.sum()), "filled": int(filled.sum()), "valid_out": int(nv.sum()),
           "components_in": int(ndimage.label(v, four)[1]), "components_out": int(ndimage.label(nv, four)[1]),
           "surface_components": surface_components(faces),
           "max_fill_dist_steps": round(float(dist.max()), 4),
           "inside_filled": int((filled & (dist <= 0)).sum()),
           "inside_filled_max_depth_steps": inside_depth,
           "stock_nodes_unchanged": bool(all(np.array_equal(np.where(v, a, 0), np.where(v, xyz[..., i], 0))
                                             for i, a in enumerate((x, y, z)))),
           **guard, "human_inputs": [], "human_minutes": 0}
    (out / "rim_fill.json").write_text(json.dumps(man, indent=1) + "\n")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("obj", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--r", type=float, default=0.5, help="fill distance, grid steps")
    ap.add_argument("--max-edge", type=float, default=0.0, help="spike guard K x median edge (0 = off)")
    a = ap.parse_args(argv)
    try:
        man = rim_fill(a.src, a.obj, a.out, a.r, a.max_edge)
    except FileExistsError as e:
        sys.exit(str(e))
    print(json.dumps({k: man[k] for k in ("valid_in", "filled", "components_in", "components_out",
                                          "max_fill_dist_steps")}))


if __name__ == "__main__":
    main()
