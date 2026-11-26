#!/usr/bin/env python3
"""Reading orientation of GP meshes and images from scroll geometry alone (VES-3585).

Rule (experiments/2026-11-08-ves3585-gp-orient/README.md). A1: the text is on the face toward the core. A2: it reads
from the outer end to the inner end. A3: the volume's (x, y, z) voxel frame is not mirrored. For each mesh node:
n_in = the sheet normal on the core side, right = the in-sheet tangent pointing inward along the spiral (its angular
part has sign h; h = +1 if the inward spiral turns counter-clockwise about +z), up = n_in x right. The image op is the
dihedral element that maps image +x onto right and image +y (down) onto -up.

Ops act on the (row, col) axes of an image or grid: 'r<deg>' = np.rot90(a, k), 'm<deg>' = np.rot90(np.fliplr(a), k).
op(a) is the upright array. The render of an oriented tifxyz grid is the oriented render (rows/cols follow the grid).

Usage:
  python -I orient.py handedness FIT.tifxyz --axis-xy X Y
  python -I orient.py rule MESH.tifxyz --axis-xy X Y --h {-1,1}
  python -I orient.py apply SRC.tifxyz DST.tifxyz --op OP
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

OPS = ("r0", "r90", "r180", "r270", "m0", "m90", "m180", "m270")


def apply_op(a, op):
    """op on the first two axes of a (H, W[, C]) array."""
    k = int(op[1:]) // 90
    if op[0] == "m":
        a = a[:, ::-1]
    return np.rot90(a, k, axes=(0, 1))


def op_matrix(op):
    """2x2 integer T with old (drow, dcol) = T @ new (drow', dcol') for b = op(a)."""
    i, j = np.mgrid[0:3, 0:3]
    bi, bj = apply_op(i, op), apply_op(j, op)
    dy = (bi[2, 1] - bi[1, 1], bj[2, 1] - bj[1, 1])      # new +row
    dx = (bi[1, 2] - bi[1, 1], bj[1, 2] - bj[1, 1])      # new +col
    return np.array([[dy[0], dx[0]], [dy[1], dx[1]]], dtype=int)


T = {op: op_matrix(op) for op in OPS}


def op_of_matrix(m):
    """The op whose T is nearest (Frobenius) to the 2x2 matrix m (any scale); also the angle residual in degrees."""
    m = np.asarray(m, float)
    u, _, vt = np.linalg.svd(m)
    q = u @ vt                                           # nearest orthogonal matrix (polar part)
    best = max(OPS, key=lambda o: float(np.sum(T[o] * q)))
    c = np.clip(np.sum(T[best] * q) / 2.0, -1, 1)        # cos of the residual angle (same det)
    det_ok = np.sign(np.linalg.det(q)) == np.sign(np.linalg.det(T[best]))
    return best, (float(np.degrees(np.arccos(c))) if det_ok else 180.0)


def compose(first, then):
    """op equal to applying `first`, then `then`."""
    return op_of_matrix(T[first] @ T[then])[0]


def inverse(op):
    return op_of_matrix(np.linalg.inv(T[op]))[0]


def load_grid(d):
    import tifffile
    g = np.stack([tifffile.imread(Path(d) / f"{c}.tif").astype(np.float64) for c in "xyz"], -1)
    v = np.all(np.isfinite(g), -1) & ~np.all(g == -1, -1) & (g[..., 2] > 0)
    return g, v


def tangents(g, v):
    """unit e_r = d xyz / d row, e_c = d xyz / d col (central differences where both neighbours are valid) + mask."""
    er = np.full(g.shape, np.nan)
    ec = np.full(g.shape, np.nan)
    ok_r = v[2:, :] & v[:-2, :]
    ok_c = v[:, 2:] & v[:, :-2]
    er[1:-1][ok_r] = (g[2:] - g[:-2])[ok_r]
    ec[:, 1:-1][ok_c] = (g[:, 2:] - g[:, :-2])[ok_c]
    m = v & np.isfinite(er).all(-1) & np.isfinite(ec).all(-1)
    er, ec = er[m], ec[m]
    er /= np.linalg.norm(er, axis=1, keepdims=True)
    ec /= np.linalg.norm(ec, axis=1, keepdims=True)
    return g[m], er, ec


def handedness(g, v, axis_xy, tol=0.1, min_pairs=200, agree=0.9, row_step=1):
    """h = -sign(r(theta + 2 pi) - r(theta)) from node pairs in the same row; dict with h (None if undecided)."""
    cx, cy = axis_xy
    votes, n_pairs, spans = [], 0, []
    for r in range(0, g.shape[0], row_step):
        idx = np.where(v[r])[0]
        if idx.size < 20:
            continue
        p = g[r, idx]
        th = np.unwrap(np.arctan2(p[:, 1] - cy, p[:, 0] - cx))
        rad = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
        spans.append(float(abs(th[-1] - th[0])))
        o = np.argsort(th)
        th, rad = th[o], rad[o]
        j = np.searchsorted(th, th + 2 * np.pi)
        j = np.clip(j, 0, th.size - 1)
        jm = np.clip(j - 1, 0, th.size - 1)
        jj = np.where(np.abs(th[jm] - th - 2 * np.pi) < np.abs(th[j] - th - 2 * np.pi), jm, j)
        ok = np.abs(th[jj] - th - 2 * np.pi) <= tol
        if ok.sum() == 0:
            continue
        n_pairs += int(ok.sum())
        votes.append(float(np.sign(np.median(rad[jj[ok]] - rad[ok]))))
    out = {"rows_voting": len(votes), "pairs": n_pairs,
           "theta_span_med": float(np.median(spans)) if spans else None, "h": None}
    if n_pairs < min_pairs or not votes:
        out["why"] = f"{n_pairs} pairs at 2 pi +- {tol} rad (< {min_pairs}): rows do not span a full turn"
        return out
    s = np.array(votes)
    share = max((s > 0).mean(), (s < 0).mean())
    out["agree"] = float(share)
    if share < agree:
        out["why"] = f"rows agree {share:.2f} < {agree}"
        return out
    out["h"] = int(-np.sign(np.sum(s)))                 # r grows with +theta -> inward is -theta -> h = -1
    return out


def frame_vectors(p, er, ec, axis_xy, h):
    """per node: n_in, right, up (unit) by the rule."""
    radial = np.zeros_like(p)
    radial[:, :2] = p[:, :2] - np.asarray(axis_xy, float)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    n = np.cross(ec, er)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n_in = n * -np.sign(np.sum(n * radial, 1))[:, None]
    theta_hat = np.cross(np.array([0.0, 0.0, 1.0]), radial)  # counter-clockwise about +z
    right = h * theta_hat
    right -= n_in * np.sum(right * n_in, 1)[:, None]
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    up = np.cross(n_in, right)
    return n_in, right, up


def rule_op(g, v, axis_xy, h, max_nodes=200_000, seed=3585):
    """The op that makes the mesh's image upright by the rule, with per-op scores and node agreement."""
    p, er, ec = tangents(g, v)
    if p.shape[0] > max_nodes:
        s = np.random.default_rng(seed).choice(p.shape[0], max_nodes, replace=False)
        p, er, ec = p[s], er[s], ec[s]
    n_in, right, up = frame_vectors(p, er, ec, axis_xy, h)
    per = {}
    for op in OPS:
        t = T[op]
        new_y = t[0, 0] * er + t[1, 0] * ec            # new +row direction in 3D
        new_x = t[0, 1] * er + t[1, 1] * ec            # new +col direction in 3D
        per[op] = np.sum(new_x * right, 1) + np.sum(new_y * -up, 1)
    S = np.stack([per[o] for o in OPS])
    best_node = np.argmax(S, 0)
    score = {o: float(per[o].mean()) for o in OPS}
    op = max(OPS, key=score.get)
    mirror = bool(np.median(np.sum(np.cross(ec, er) * n_in, 1)) > 0)
    return {"op": op, "h": int(h), "axis_xy": [float(x) for x in axis_xy], "score": score,
            "node_agree": float(np.mean(best_node == OPS.index(op))), "n_nodes": int(p.shape[0]),
            "mirror_by_normal": mirror,
            "dots": {"ec.right": float(np.median(np.sum(ec * right, 1))), "er.up": float(np.median(np.sum(er * up, 1))),
                     "up.z": float(np.median(up[:, 2])), "ecxer.n_in": float(np.median(np.sum(np.cross(ec, er) * n_in, 1)))}}


def synthetic_scroll(sigma, turns=1.3, n_s=400, n_t=40, ell=0.6, r0=400.0, pitch=30.0, axis=(1000.0, 800.0), tilt=0.0):
    """Upright grid G0 (row 0 = top line, col 0 = start of text = outer end) of a scroll whose up is sigma * z.
    Reading direction s goes inward; A1 fixes the turning sense: phi' * sigma < 0."""
    s = np.linspace(0, 1, n_s)
    t = np.linspace(1, 0, n_t)                       # row 0 = top
    phi = -sigma * 2 * np.pi * turns * s              # inward turning sense = -sigma (h = -sigma)
    r = r0 + pitch * turns * (1 - s)                  # outer -> inner
    S, Tt = np.meshgrid(phi, t)
    R = np.broadcast_to(r, S.shape)
    x = axis[0] + R * np.cos(S + tilt)
    y = axis[1] + ell * R * np.sin(S + tilt)
    z = 5000 + sigma * 600 * Tt
    return np.stack([x, y, z], -1), np.ones(S.shape, bool)


def orient_tifxyz(src, dst, op, info=None):
    """write op(src grid) as a tifxyz at dst; meta.json gets "orient" (op + info). Scale entries swap on 90/270."""
    import tifffile
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=False)
    for c in "xyz":
        a = tifffile.imread(src / f"{c}.tif")
        tifffile.imwrite(dst / f"{c}.tif", np.ascontiguousarray(apply_op(a, op)))
    meta = json.loads((src / "meta.json").read_text())
    if op[1:] in ("90", "270") and isinstance(meta.get("scale"), list) and len(meta["scale"]) == 2:
        meta["scale"] = meta["scale"][::-1]
    meta["orient"] = {"op": op, "rule": "VES-3585", **(info or {})}
    (dst / "meta.json").write_text(json.dumps(meta, indent=4) + "\n")
    for f in src.iterdir():
        if f.name not in ("x.tif", "y.tif", "z.tif", "meta.json") and f.is_file():
            shutil.copyfile(f, dst / f.name)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("handedness")
    a1.add_argument("fit", type=Path)
    a1.add_argument("--axis-xy", type=float, nargs=2, required=True)
    a2 = sp.add_parser("rule")
    a2.add_argument("mesh", type=Path)
    a2.add_argument("--axis-xy", type=float, nargs=2, required=True)
    a2.add_argument("--h", type=int, choices=(-1, 1), required=True)
    a3 = sp.add_parser("apply")
    a3.add_argument("src", type=Path)
    a3.add_argument("dst", type=Path)
    a3.add_argument("--op", choices=OPS, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "handedness":
        g, v = load_grid(a.fit)
        print(json.dumps(handedness(g, v, a.axis_xy)))
    elif a.cmd == "rule":
        g, v = load_grid(a.mesh)
        print(json.dumps(rule_op(g, v, a.axis_xy, a.h)))
    else:
        orient_tifxyz(a.src, a.dst, a.op)


if __name__ == "__main__":
    sys.exit(main())
