#!/usr/bin/env python3
"""GP output chain, stage "fold-repair": refill folded fit patches with a smooth surface, same grid,
no vertex dropped, before the isometric stage.

Why: F-437 (VES-2547 / VES-2560). The start of the outermost winding (w130) of the ves1904 fit is a
folded corner (rows 13-24 x columns 0-65: flipped quads, edges from 0.8 to 378 vx on a 20 vx grid).
No flat map is isometric there (mesh_isometry p05 0.84), and masking it (crumple_mask.py) costs area
and leaves a non-developable rim (p05 0.891, area guard 0.921). This stage repairs instead of dropping.

Rule (geometry only, no CT, no ink)
  bad quad     all 4 corners valid, and any of
                 fold     its normal (a-b-c triangle) makes > 90 deg with the mean unit normal of the
                          (2 NR + 1)^2 quads around it (dot < 0), or with the normal of a valid
                          4-neighbour quad (a crease sharper than 90 deg);
                 spike    its 3D area > SPIKE x the median quad area of the input;
                 collapse its shortest edge < COLLAPSE x the median edge length of the input.
  zone         corners of bad quads, grown by DILATE grid steps (8-neighbour), valid vertices only.
  refill       zone positions are the minimiser of the discrete bi-Laplacian energy sum_i |L p_i|^2
               over the valid grid (L = uniform graph Laplacian over the valid 4-neighbours, so grid
               edges get the natural boundary condition); positions outside the zone are fixed and
               copied unchanged. Solved per coordinate (x, y, z) with a sparse direct solver. A zone
               component with no fixed vertex next to it is left unchanged and reported.
  frame        xyz (default): the refill solves volume (z, y, x). cyl: it solves (z, r, theta) about a
               local scroll axis parallel to z, one axis per zone component (Kasa circle fit to the valid
               (y, x) of the grid columns within CYL_COLS of the component). Why: a bi-Laplacian refill in
               xyz pulls a curved wrap toward its chord; on inner windings (pitch ~15 vx) that crosses the
               neighbouring wrap (VES-2575 dev: 15-25 % of zone vertices on the wrong side, 3 % floor).
  Positions outside the zone are byte-identical to the input; the valid mask is unchanged.

Contract
  input   a tifxyz (x/y/z.tif + meta.json), volume voxels, any grid scale; invalid = -1 (x, y, z).
  output  a tifxyz with the same grid, scale, meta, valid mask and dtype; only zone vertices move.
  manifest  OUT/fold_repair.json: parameters, input / output sha256, bad-quad counts by kind, zone
          size and columns, displacement percentiles, 3D area in / out, the human-input ledger entry
          (none: 0 minutes).

Usage: python3 tools/gpchain/fold_repair.py IN.tifxyz OUT.tifxyz [--cols 0 653] [--nr 4]
       [--spike 4] [--collapse 0.25] [--dilate 4] [--frame xyz|cyl] [--cyl-cols 325]
       (numpy, scipy, tifffile; the villa #2401 venv)
  --cols U0 U1  only bad quads with a corner in grid columns [U0, U1) seed the zone (default: all);
                repeat for several ranges.
Deterministic: no random numbers.
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
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crumple_mask import angle_defect, area3d_per_quad, load  # noqa: E402

STAGE = "fold-repair"


def quad_valid(v):
    return v[:-1, :-1] & v[:-1, 1:] & v[1:, :-1] & v[1:, 1:]


def bad_quads(p, v, nr, spike, collapse):
    """boolean (H-1, W-1) arrays fold, spike, collapse."""
    q = quad_valid(v)
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
    n = np.cross(b - a, c - a)
    nn = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)
    nn[~q] = 0
    k = 2 * nr + 1
    sm = np.stack([ndimage.uniform_filter(nn[..., i], k, mode="constant") for i in range(3)], -1)
    fold = q & ((nn * sm).sum(-1) < 0)
    for ax in (0, 1):
        sl0 = (slice(None, -1), slice(None)) if ax == 0 else (slice(None), slice(None, -1))
        sl1 = (slice(1, None), slice(None)) if ax == 0 else (slice(None), slice(1, None))
        crease = q[sl0] & q[sl1] & ((nn[sl0] * nn[sl1]).sum(-1) < 0)
        fold[sl0] |= crease
        fold[sl1] |= crease
    area = area3d_per_quad(p, v)
    spk = q & (area > spike * np.median(area[q]))
    e = np.stack([np.linalg.norm(b - a, axis=-1), np.linalg.norm(c - a, axis=-1),
                  np.linalg.norm(d - b, axis=-1), np.linalg.norm(d - c, axis=-1)], -1)
    med = np.median(e[q])
    col = q & (e.min(-1) < collapse * med)
    return fold, spk, col


def col_ranges(cols):
    """None, one (U0, U1) pair or a list of pairs -> list of pairs or None."""
    if cols is None:
        return None
    return [tuple(cols)] if np.isscalar(cols[0]) else [tuple(c) for c in cols]


def zone_from(bad, v, dilate, cols=None):
    cols = col_ranges(cols)
    if cols is not None:
        keep = np.zeros_like(bad)
        for u0, u1 in cols:
            keep[:, max(u0 - 1, 0):u1] = True
        bad = bad & keep
    z = np.zeros(v.shape, bool)
    for dr in (0, 1):
        for dc in (0, 1):
            z[dr:dr + bad.shape[0], dc:dc + bad.shape[1]] |= bad
    if dilate:
        z = ndimage.binary_dilation(z, np.ones((3, 3), bool), iterations=dilate)
    return z & v


def laplacian(v):
    """uniform graph Laplacian over valid 4-neighbours; rows/cols index the flattened grid."""
    H, W = v.shape
    idx = np.arange(H * W).reshape(H, W)
    rows, cols, vals = [], [], []
    deg = np.zeros(H * W)
    for dr, dc in ((0, 1), (1, 0)):
        a = v[:H - dr, :W - dc] & v[dr:, dc:]
        i, j = idx[:H - dr, :W - dc][a], idx[dr:, dc:][a]
        rows += [i, j]
        cols += [j, i]
        np.add.at(deg, i, 1)
        np.add.at(deg, j, 1)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(H * W, H * W))
    with np.errstate(divide="ignore"):
        inv = np.where(deg > 0, 1.0 / deg, 0.0)
    return sparse.diags(inv) @ A - sparse.diags((deg > 0).astype(float)), deg


def axis_center(p, v, c0, c1):
    """Kasa least-squares circle centre (y, x) of the valid vertices in grid columns [c0, c1)."""
    q = p[:, max(c0, 0):c1][v[:, max(c0, 0):c1]]
    y, x = q[:, 1], q[:, 2]
    A = np.stack([2 * y, 2 * x, np.ones_like(y)], -1)
    sol = np.linalg.lstsq(A, y * y + x * x, rcond=None)[0]
    return sol[:2]


def to_cyl(p, c, th0):
    d = p[..., 1:] - c
    th = np.angle(np.exp(1j * (np.arctan2(d[..., 0], d[..., 1]) - th0)))
    return np.stack([p[..., 0], np.hypot(d[..., 0], d[..., 1]), th], -1)


def from_cyl(q, c, th0):
    th = q[..., 2] + th0
    return np.stack([q[..., 0], c[0] + q[..., 1] * np.sin(th), c[1] + q[..., 1] * np.cos(th)], -1)


def refill(p, v, zone, frame="xyz", cyl_cols=325, info=None):
    """bi-Laplacian refill of the zone; returns new positions and the list of unanchored components."""
    H, W = v.shape
    L, deg = laplacian(v)
    lab, n = ndimage.label(zone)
    fixed = v & ~zone
    near_fixed = ndimage.binary_dilation(fixed, np.ones((3, 3), bool))
    unanchored = [int(i) for i in range(1, n + 1) if not (near_fixed & (lab == i)).any()]
    solve = zone & ~np.isin(lab, unanchored)
    out = p.copy()
    if not solve.any():
        return out, unanchored
    if frame == "cyl":
        # one local axis per component; each component is solved with everything else at its input position
        for i in range(1, n + 1):
            if i in unanchored:
                continue
            cc = np.flatnonzero((lab == i).any(0))
            c = axis_center(p, v, cc[0] - cyl_cols, cc[-1] + 1 + cyl_cols)
            ring = near_fixed & fixed & ndimage.binary_dilation(lab == i, np.ones((3, 3), bool))
            d = p[ring][:, 1:] - c
            th0 = float(np.angle(np.exp(1j * np.arctan2(d[:, 0], d[:, 1])).mean()))
            q = to_cyl(p, c, th0)
            qs = _solve(L, q, v, lab == i, v & (lab != i))
            out[lab == i] = from_cyl(qs, c, th0)[lab == i]
            if info is not None:
                info.append({"component": i, "cols": [int(cc[0]), int(cc[-1]) + 1],
                             "axis_yx": [round(float(x), 1) for x in c],
                             "r_median": round(float(np.median(q[lab == i][:, 1])), 1)})
        return out, unanchored
    return _solve(L, p, v, solve, fixed), unanchored


def _solve(L, p, v, solve, fixed):
    out = p.copy()
    u = np.flatnonzero(solve.ravel())
    f = np.flatnonzero(fixed.ravel())
    # energy rows: valid vertices within 2 steps of an unknown (the only rows that depend on it)
    rel = ndimage.binary_dilation(solve, np.ones((3, 3), bool), iterations=2) & v
    r = np.flatnonzero(rel.ravel())
    Lr = L[r]
    Lu, Lf = Lr[:, u], Lr[:, f]
    M = (Lu.T @ Lu).tocsc()
    flat = p.reshape(-1, 3)
    o = out.reshape(-1, 3)
    for k in range(3):
        rhs = -(Lu.T @ (Lf @ flat[f, k]))
        o[u, k] = spsolve(M, rhs)
    return out


def sha256s(d):
    return {f: hashlib.sha256((Path(d) / f).read_bytes()).hexdigest() for f in ("x.tif", "y.tif", "z.tif")}


def repair(p, v, cols=None, nr=4, spike=4.0, collapse=0.25, dilate=4, frame="xyz", cyl_cols=325, info=None):
    fold, spk, col = bad_quads(p, v, nr, spike, collapse)
    zone = zone_from(fold | spk | col, v, dilate, cols)
    out, unanchored = refill(p, v, zone, frame, cyl_cols, info)
    return out, zone, {"fold": fold, "spike": spk, "collapse": col}, unanchored


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cols", type=int, nargs=2, action="append", default=None)
    ap.add_argument("--nr", type=int, default=4)
    ap.add_argument("--spike", type=float, default=4.0)
    ap.add_argument("--collapse", type=float, default=0.25)
    ap.add_argument("--dilate", type=int, default=4)
    ap.add_argument("--frame", choices=("xyz", "cyl"), default="xyz")
    ap.add_argument("--cyl-cols", type=int, default=325)
    a = ap.parse_args(argv)
    cols = a.cols[0] if a.cols and len(a.cols) == 1 else a.cols
    if a.out.exists():
        sys.exit(f"fold_repair: {a.out} exists; refusing to overwrite")
    p, v = load(a.src)
    axes = []
    out, zone, bad, unanchored = repair(p, v, cols, a.nr, a.spike, a.collapse, a.dilate, a.frame, a.cyl_cols, axes)
    shutil.copytree(a.src, a.out)
    for i, c in enumerate("zyx"):
        arr = tifffile.imread(a.src / f"{c}.tif")
        new = np.where(zone, out[..., i], arr).astype(arr.dtype)
        tifffile.imwrite(a.out / f"{c}.tif", new, compression="zlib")
    p2, v2 = load(a.out)
    assert (v2 == v).all(), "valid mask changed"
    disp = np.linalg.norm(p2 - p, axis=-1)[zone]
    ad_in, ad_out = np.abs(angle_defect(p, v)), np.abs(angle_defect(p2, v2))
    zc = np.flatnonzero(zone.any(0))
    qa_in, qa_out = area3d_per_quad(p, v).sum(), area3d_per_quad(p2, v2).sum()
    f2, s2, c2 = bad_quads(p2, v2, a.nr, a.spike, a.collapse)
    man = {"stage": STAGE, "src": str(a.src.resolve()), "out": str(a.out.resolve()),
           "params": {"cols": cols, "nr": a.nr, "spike": a.spike, "collapse": a.collapse, "dilate": a.dilate,
                      **({"frame": a.frame, "cyl_cols": a.cyl_cols} if a.frame != "xyz" else {})},
           **({"cyl_axes": axes} if a.frame != "xyz" else {}),
           "in_sha256": sha256s(a.src), "out_sha256": sha256s(a.out),
           "valid": int(v.sum()), "valid_unchanged": True,
           "bad_quads_in": {k: int(x.sum()) for k, x in bad.items()},
           "bad_quads_out": {"fold": int(f2.sum()), "spike": int(s2.sum()), "collapse": int(c2.sum())},
           "zone_vertices": int(zone.sum()), "zone_cols": [int(zc[0]), int(zc[-1]) + 1] if zc.size else None,
           "unanchored_components": unanchored,
           "displacement_vx": {q: round(float(np.percentile(disp, q)), 2) for q in (50, 95, 100)} if disp.size else None,
           "zone_mean_abs_angle_defect_mrad": {
               "in": round(float(np.nanmean(ad_in[zone])) * 1e3, 1) if zone.any() else None,
               "out": round(float(np.nanmean(ad_out[zone])) * 1e3, 1) if zone.any() else None},
           "area3d_in_vx2": round(float(qa_in), 1), "area3d_out_vx2": round(float(qa_out), 1),
           "area_kept": round(float(qa_out / qa_in), 4) if qa_in else None,
           "human_inputs": [], "human_minutes": 0}
    (a.out / "fold_repair.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("bad_quads_in", "bad_quads_out", "zone_vertices", "zone_cols",
                                          "displacement_vx", "zone_mean_abs_angle_defect_mrad", "area_kept")}))


if __name__ == "__main__":
    main()
