#!/usr/bin/env python3
"""GP output chain, stage "crumple-mask": drop crumpled (non-developable) fit patches before the
isometric stage, with a geometry-only angle-defect rule.

Why: no flat map is isometric on a crumpled patch. F-437 (VES-2547): the start of the outermost
winding of the ves1904 fit has mean |angle defect| 103 mrad against <= 54 elsewhere, and SLIM leaves
its band cut out of the villa #2401 `mesh_isometry` band at any iteration count.

Rule (geometry only, no CT, no ink)
  angle defect  per interior vertex (all 4 incident quads valid): 2 pi - sum of the 6 triangle angles
                at the vertex, quads split a-b-c / b-d-c as in F-429 metrics.area3d. Radians.
  local mean    mean |angle defect| over the interior vertices in a (2R+1) x (2R+1) grid window.
  mask          valid vertices whose local mean > TAU (vertices with no interior vertex in their
                window are never masked), then valid 4-connected components smaller than
                MIN_KEEP of the valid vertices are dropped too (no stranded islands).
  TAU is an input, never derived from the surface being masked. VES-2560 froze R = 16 and
  TAU = 0.12094 rad, the p99 over windings 10-129 of the ves1904 fit (so ~1 % of a typical winding
  is masked); a "2x the median" bar would mask 7-11 % of those windings (heavy tail).

Contract
  input   a tifxyz (x/y/z.tif + meta.json), volume voxels, any grid scale; invalid = -1 (x, y, z).
  output  a tifxyz with the same grid, scale, meta and frame; masked vertices set to -1 in x/y/z.tif.
  manifest  OUT/crumple_mask.json: parameters, input / output sha256, masked vertex and 3D-area counts,
          and the human-input ledger entry (none: 0 minutes).

Usage: python3 tools/gpchain/crumple_mask.py IN.tifxyz OUT.tifxyz --tau 0.12094 [--radius 16]
       [--min-keep 0.01]   (needs numpy, scipy, tifffile; runs in the villa #2401 venv)
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
from scipy import ndimage

STAGE = "crumple-mask"


def load(d):
    d = Path(d)
    x, y, z = (tifffile.imread(d / f"{c}.tif") for c in "xyz")
    p = np.stack([z, y, x], -1).astype(np.float64)
    v = (x != -1) & (y != -1) & (z != -1) & np.isfinite(p).all(-1)
    return p, v


def _angle(a, b, c):
    """angle at a in triangle a-b-c, arrays (..., 3)."""
    u, w = b - a, c - a
    cs = (u * w).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1) + 1e-12)
    return np.arccos(np.clip(cs, -1.0, 1.0))


def angle_defect(p, v):
    """per-vertex angle defect (rad), NaN where the vertex is not interior."""
    H, W = v.shape
    q = v[:-1, :-1] & v[:-1, 1:] & v[1:, :-1] & v[1:, 1:]
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
    s = np.zeros((H, W))
    # triangle a-b-c: corners a (r, c), b (r, c+1), c (r+1, c)
    # triangle b-d-c: corners b (r, c+1), d (r+1, c+1), c (r+1, c)
    for corner, ang in (((0, 0), _angle(a, b, c)), ((0, 1), _angle(b, c, a)), ((1, 0), _angle(c, a, b)),
                        ((0, 1), _angle(b, d, c)), ((1, 1), _angle(d, c, b)), ((1, 0), _angle(c, b, d))):
        dr, dc = corner
        s[dr:H - 1 + dr, dc:W - 1 + dc] += np.where(q, ang, 0.0)
    nq = np.zeros((H, W), int)
    for dr in (0, 1):
        for dc in (0, 1):
            nq[dr:H - 1 + dr, dc:W - 1 + dc] += q
    interior = nq == 4
    return np.where(interior, 2 * np.pi - s, np.nan)


def local_mean_abs(ad, radius):
    """mean |ad| over the non-NaN entries in a (2R+1)^2 window; NaN where the window has none."""
    ok = np.isfinite(ad)
    k = 2 * radius + 1
    num = ndimage.uniform_filter(np.where(ok, np.abs(ad), 0.0), k, mode="constant")
    den = ndimage.uniform_filter(ok.astype(float), k, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-9, num / np.maximum(den, 1e-12), np.nan)


def area3d_per_quad(p, v):
    q = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
    t = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1) + 0.5 * np.linalg.norm(np.cross(b - d, c - d), axis=-1)
    return np.where(q, t, 0.0)


def mask(p, v, tau, radius, min_keep):
    lm = local_mean_abs(angle_defect(p, v), radius)
    crumpled = v & np.isfinite(lm) & (lm > tau)
    keep = v & ~crumpled
    lab, n = ndimage.label(keep)
    island = np.zeros_like(keep)
    if n:
        sizes = ndimage.sum(keep, lab, index=np.arange(1, n + 1))
        small = np.flatnonzero(sizes < min_keep * v.sum()) + 1
        island = np.isin(lab, small)
    return keep & ~island, crumpled, island, lm


def sha256s(d):
    return {f: hashlib.sha256((Path(d) / f).read_bytes()).hexdigest() for f in ("x.tif", "y.tif", "z.tif")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tau", type=float, required=True, help="local mean |angle defect| bar, radians")
    ap.add_argument("--radius", type=int, default=16)
    ap.add_argument("--min-keep", type=float, default=0.01)
    a = ap.parse_args(argv)
    if a.out.exists():
        sys.exit(f"crumple_mask: {a.out} exists; refusing to overwrite")
    p, v = load(a.src)
    keep, crumpled, island, lm = mask(p, v, a.tau, a.radius, a.min_keep)
    shutil.copytree(a.src, a.out)
    for c in "xyz":
        arr = tifffile.imread(a.src / f"{c}.tif")
        arr = np.where(keep, arr, np.asarray(-1, arr.dtype)).astype(arr.dtype)
        tifffile.imwrite(a.out / f"{c}.tif", arr, compression="zlib")
    qa_in, qa_out = area3d_per_quad(p, v).sum(), area3d_per_quad(p, keep).sum()
    cols = np.flatnonzero(crumpled.any(0))
    man = {"stage": STAGE, "src": str(a.src.resolve()), "out": str(a.out.resolve()),
           "params": {"tau_rad": a.tau, "radius": a.radius, "min_keep": a.min_keep},
           "in_sha256": sha256s(a.src), "out_sha256": sha256s(a.out),
           "valid_in": int(v.sum()), "crumpled": int(crumpled.sum()), "island": int(island.sum()),
           "valid_out": int(keep.sum()),
           "area3d_in_vx2": round(float(qa_in), 1), "area3d_out_vx2": round(float(qa_out), 1),
           "area_kept": round(float(qa_out / qa_in), 4) if qa_in else None,
           "crumpled_cols": [int(cols[0]), int(cols[-1]) + 1] if cols.size else None,
           "human_inputs": [], "human_minutes": 0}
    (a.out / "crumple_mask.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("valid_in", "crumpled", "island", "area_kept", "crumpled_cols")}))


if __name__ == "__main__":
    main()
