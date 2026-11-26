#!/usr/bin/env python3
"""GP output chain, stage "band_cuts": cut a flat (isometric) surface into column meshes that follow
the bent valid band, so each column grid holds less empty space and keeps the 3D area.

Why: after the isometric stage (tools/gpchain/isometric_step.py) the valid band of a fit strip is
not straight in the flat grid. Our fit bands are z-slabs, so the band edges unroll as a wave of about
one winding period (F-429 strip: +-40 rows on a 45-row band, flat grid 44 % valid). villa #2401
`cut-columns` cuts full-height grid columns, so every column mesh carries that empty space.

Rule (fixed; two parameters, --piece and --smooth)
  1. centreline: per flat grid column (one grid step of u), the mean row of the surface there;
     moving average over --smooth columns (edge padded). Arc length s along it, in grid cells.
  2. N = max(1, round(L / --piece)) pieces of equal arc length (L = centreline length).
     Piece boundaries are the lines through the centreline point at s_k, perpendicular to the
     centreline there, so neighbouring pieces share one boundary (no gap, no overlap).
  3. each piece is turned rigidly so the chord of its centreline is horizontal, and gets its own
     grid at the input's grid step (meta "scale" unchanged). A rigid turn of an isometric map stays
     isometric.
Three ways to make the piece grids (same centreline and boundaries):
  --obj FLAT.obj (default path, "obj-rotated"): flatboi's OBJ from `isometric_step.py --keep-obj`.
     Faces go to the piece that holds their UV centroid; the piece's UVs are turned and the stock
     `vc_obj2tifxyz --tifxyz-source=FLAT.tifxyz` rasterises each piece once (as the stage does).
     --rim-fill R (VES-2570, default 0 = off): then rim_fill.py fills each piece's invalid nodes
     within R grid steps of that piece's own faces (the stock raster drops ~1/2 step all round).
     --piece-objs DIR keeps the piece OBJs (turned UVs) there, for an area audit.
  no --obj ("grid-rotated"): resample the flat tifxyz on the turned grid, linear on the flat mesh's
     triangles; valid where every flat vertex with non-zero weight is valid (a second rasterisation).
  --no-rotate ("plain"): vertical boundaries at integer flat columns and a per-piece row window
     (no resampling; cell values copied byte for byte).

Contract
  input   FLAT.tifxyz from isometric_step (volume voxels of the fit frame) [+ FLAT.obj].
  output  SUBMISSION/meshes/column_NN.tifxyz (#2401 layout), grid step = input step, meta extra:
          kind, source, u0/u1 (piece arc range along the centreline in grid cells, so #2401's
          no_overlapping_cuts gate applies), cut_rule, angle_deg.
          SUBMISSION/band_cuts.json: parameters, per-piece numbers, human_inputs [] (0 minutes).
Runs in a python with villa `vesuvius` installed (the #2401 venv); --obj also needs docker and
the villa-ink:ves1471 image. Deterministic.

Usage: python band_cuts.py FLAT.tifxyz SUBMISSION [--obj FLAT.obj] [--piece 300] [--smooth 151]
       [--no-rotate] [--image villa-ink:ves1471] [--threads 4] [--rim-fill 0.5] [--rim-max-edge 0] [--piece-objs DIR]
"""
import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from vesuvius.tifxyz import Tifxyz, read_tifxyz, write_tifxyz

STAGE = "band_cuts"


def centreline(col, row, weight, smooth):
    """Weighted mean row per integer column bin of points (col, row), smoothed; arc length s."""
    b = np.floor(col).astype(np.int64)
    c0, c1 = int(b.min()), int(b.max())
    n = np.bincount(b - c0, weights=weight, minlength=c1 - c0 + 1)
    sr = np.bincount(b - c0, weights=weight * row, minlength=c1 - c0 + 1)
    c = np.arange(c0, c1 + 1)
    have = n > 0
    m = np.interp(c, c[have], sr[have] / n[have])
    k = max(1, min(int(smooth), len(c)))
    pad = np.pad(m, (k // 2, k - 1 - k // 2), mode="edge")
    ms = np.convolve(pad, np.ones(k) / k, mode="valid")
    dm = np.gradient(ms) if len(c) > 1 else np.zeros_like(ms)
    ds = np.sqrt(1.0 + dm ** 2)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]))])
    return c.astype(np.float64), ms, dm, s


def at(s_query, s, arr):
    return float(np.interp(s_query, s, arr))


def sample(P, V, qr, qc):
    """Linear xyz on the mesh triangles at flat points (qr, qc): each grid quad is split along the
    01-10 diagonal (as the area sum does); valid where every vertex with weight > 0 is valid."""
    H, W = V.shape
    r0, c0 = np.floor(qr).astype(np.int64), np.floor(qc).astype(np.int64)
    fr, fc = qr - r0, qc - c0
    up = fr + fc > 1
    corners = (((0, 0), np.where(up, 0.0, 1 - fr - fc)), ((0, 1), np.where(up, 1 - fr, fc)),
               ((1, 0), np.where(up, 1 - fc, fr)), ((1, 1), np.where(up, fr + fc - 1, 0.0)))
    out = np.zeros(qr.shape + (3,), np.float64)
    ok = np.ones(qr.shape, bool)
    for (dr, dc), w in corners:
        need = w > 0
        rr, cc = r0 + dr, c0 + dc
        inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        ok &= ~need | inb
        rr, cc = np.clip(rr, 0, H - 1), np.clip(cc, 0, W - 1)
        ok &= ~need | V[rr, cc]
        out += np.where(need[..., None], w[..., None] * P[rr, cc], 0.0)
    return out, ok


def read_obj(path):
    v, vt, f = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("vt "):
                vt.append(line.split()[1:3])
            elif line.startswith("f "):
                f.append([x.split("/")[:2] for x in line.split()[1:4]])
    f = np.asarray(f, np.int64) - 1                              # (nf, 3, [v, vt])
    return np.asarray(v, np.float64), np.asarray(vt, np.float64), f


def write_obj(path, v, vt, f):
    """Faces f (nf, 3, [v, vt]) with only the vertices they use; one vt per face corner."""
    used, inv = np.unique(f[..., 0], return_inverse=True)
    with open(path, "w") as fh:
        fh.write("".join("v %.17g %.17g %.17g\n" % tuple(x) for x in v[used]))
        fh.write("".join("vt %.17g %.17g\n" % tuple(x) for x in vt[f[..., 1].reshape(-1)]))
        vi = inv.reshape(-1, 3) + 1
        ti = np.arange(1, 3 * len(f) + 1).reshape(-1, 3)
        fh.write("".join("f %d/%d %d/%d %d/%d\n" % (a[0], b[0], a[1], b[1], a[2], b[2])
                         for a, b in zip(vi, ti)))


def rasterise(work, names, src, image, threads):
    """Stock vc_obj2tifxyz on each piece OBJ in one container (same tool and flags as the stage)."""
    import os
    import subprocess
    script = "set -e; cd /w; " + "; ".join(
        f"vc_obj2tifxyz {n}.obj {n}.tifxyz --tifxyz-source=/src" for n in names)
    r = subprocess.run(["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                        "--cpus", str(threads), "--memory", "8g", "--network", "none",
                        "-e", f"OMP_NUM_THREADS={threads}", "-v", f"{work}:/w", "-v", f"{src}:/src:ro",
                        "--entrypoint", "sh", image, "-c", script], text=True, capture_output=True)
    if r.returncode != 0:
        sys.stderr.write((r.stdout + r.stderr)[-4000:])
        sys.exit(f"band_cuts: vc_obj2tifxyz failed (exit {r.returncode})")
    return r.stdout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("flat", type=Path)
    ap.add_argument("submission", type=Path)
    ap.add_argument("--obj", type=Path, help="flatboi OBJ from isometric_step.py --keep-obj")
    ap.add_argument("--piece", type=float, default=300.0, help="target piece length, grid cells")
    ap.add_argument("--smooth", type=int, default=151, help="centreline moving average, grid columns")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--image", default="villa-ink:ves1471")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rim-fill", type=float, default=0.0, help="--obj only: rim_fill.py R, grid steps (0 = off)")
    ap.add_argument("--rim-max-edge", type=float, default=0.0, help="--rim-fill only: rim_fill.py --max-edge (0 = off)")
    ap.add_argument("--piece-objs", type=Path, help="--obj only: keep the piece OBJs in this directory")
    a = ap.parse_args(argv)
    if (a.rim_fill or a.piece_objs) and not a.obj:
        sys.exit("band_cuts: --rim-fill and --piece-objs need --obj")
    if a.obj and a.no_rotate:
        sys.exit("band_cuts: --obj and --no-rotate do not go together")
    if a.threads > 12:
        sys.exit("band_cuts: at most 12 threads on the shared machine")
    rule = "obj-rotated" if a.obj else "plain" if a.no_rotate else "grid-rotated"
    mesh_dir = a.submission / "meshes"
    if mesh_dir.exists() and any(mesh_dir.iterdir()):
        sys.exit(f"band_cuts: {mesh_dir} is not empty; refusing to overwrite")

    surf = read_tifxyz(a.flat)
    P = surf.get_zyxs(stored_resolution=True).astype(np.float64)
    V = np.asarray(surf.valid_vertex_mask, bool) & np.isfinite(P).all(-1)
    H, W = V.shape
    sy, sx = surf.get_scale_tuple()
    if a.obj:                                       # UV units are voxels; cells = UV * scale
        ov, ovt, of = read_obj(a.obj)
        uvc = ovt[of[..., 1]] * np.array([sx, sy])                  # (nf, 3, 2) in cells
        pts = uvc.mean(1)                                           # face centroids (col, row)
        e1, e2 = uvc[:, 1] - uvc[:, 0], uvc[:, 2] - uvc[:, 0]
        wts = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    else:
        rr, cc = np.nonzero(V)
        pts = np.stack([cc, rr], 1).astype(np.float64)              # valid cells (col, row)
        wts = np.ones(len(pts))
    c, m, dm, s = centreline(pts[:, 0], pts[:, 1], wts, a.smooth)
    L = float(s[-1])
    N = max(1, int(round(L / a.piece)))
    bounds = [L * k / N for k in range(N + 1)]

    lines = []                                                   # interior boundaries
    for k in range(1, N):
        if a.no_rotate:
            cb = float(round(at(bounds[k], s, c)))
            lines.append((np.array([cb - 0.5, 0.0]), np.array([1.0, 0.0])))
        else:
            t = np.array([1.0, at(bounds[k], s, dm)])
            lines.append((np.array([at(bounds[k], s, c), at(bounds[k], s, m)]), t / np.linalg.norm(t)))

    def member(k, p):
        ok = np.ones(len(p), bool)
        if k >= 1:
            B, t = lines[k - 1]
            ok &= (p - B) @ t >= 0
        if k < N - 1:
            B, t = lines[k]
            ok &= (p - B) @ t < 0
        return ok

    owner = sum(member(k, pts).astype(np.int64) for k in range(N))
    a.submission.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(exist_ok=True)
    pieces, names = [], [f"column_{k + 1:02d}" for k in range(N)]
    work = Path(tempfile.mkdtemp(prefix="band-cuts-", dir=a.submission)) if a.obj else None
    for k in range(N):
        mine = member(k, pts)
        O = np.array([at(bounds[k], s, c), at(bounds[k], s, m)])
        E = np.array([at(bounds[k + 1], s, c), at(bounds[k + 1], s, m)])
        theta = 0.0 if a.no_rotate else math.atan2(E[1] - O[1], E[0] - O[0])
        eu = np.array([math.cos(theta), math.sin(theta)])
        ev = np.array([-math.sin(theta), math.cos(theta)])
        pieces.append({"mesh": names[k] + ".tifxyz", "angle_deg": round(math.degrees(theta), 3),
                       "s0": round(bounds[k], 2), "s1": round(bounds[k + 1], 2),
                       "units_owned": int(mine.sum())})
        if a.obj:                                   # turn the piece's UVs about O, back to UV units
            fk = of[mine]
            uv = ovt[fk[..., 1]] * np.array([sx, sy]) - O            # (nfk, 3, 2) cells
            turned = np.stack([uv @ eu, uv @ ev], -1) / np.array([sx, sy])
            fk2 = np.stack([fk[..., 0], np.arange(3 * len(fk)).reshape(-1, 3)], -1)
            write_obj(work / f"{names[k]}.obj", ov, turned.reshape(-1, 2), fk2)
            continue
        cells = pts[mine]
        if a.no_rotate:
            j0, j1 = int(cells[:, 0].min()), int(cells[:, 0].max())
            i0, i1 = int(cells[:, 1].min()), int(cells[:, 1].max())
            Q = P[i0:i1 + 1, j0:j1 + 1].copy()
            ok = V[i0:i1 + 1, j0:j1 + 1].copy()
            g = np.stack(np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1)), -1).reshape(-1, 2)
            ok &= member(k, g.astype(np.float64)).reshape(ok.shape)
        else:
            pa, pb = (cells - O) @ eu, (cells - O) @ ev
            j0, j1 = int(math.floor(pa.min())), int(math.ceil(pa.max()))
            i0, i1 = int(math.floor(pb.min())), int(math.ceil(pb.max()))
            jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1))
            q = O + jj[..., None] * eu + ii[..., None] * ev       # (col, row)
            Q, ok = sample(P, V, q[..., 1], q[..., 0])
            ok &= member(k, q.reshape(-1, 2)).reshape(ok.shape)
        Q[~ok] = -1.0
        good = Q[ok]
        write_tifxyz(mesh_dir / pieces[k]["mesh"], Tifxyz(
            _z=Q[..., 0].astype(np.float32), _y=Q[..., 1].astype(np.float32),
            _x=Q[..., 2].astype(np.float32), uuid=names[k], _scale=surf.get_scale_tuple(),
            bbox=tuple(good[:, ::-1].min(0).tolist() + good[:, ::-1].max(0).tolist())), overwrite=False)
    if a.obj:
        log = rasterise(work, names, a.flat.resolve(), a.image, a.threads)
        for k, n in enumerate(names):
            if a.rim_fill > 0:
                sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: no script dir
                from rim_fill import rim_fill
                man = rim_fill(work / f"{n}.tifxyz", work / f"{n}.obj", work / f"{n}.rim.tifxyz", a.rim_fill,
                                a.rim_max_edge)
                (work / f"{n}.rim.tifxyz" / "rim_fill.json").unlink()
                pieces[k]["rim_fill"] = {key: man[key] for key in (
                    "valid_in", "filled", "components_in", "components_out", "surface_components",
                    "max_fill_dist_steps", "stock_nodes_unchanged") + (("guard_dropped",) if a.rim_max_edge > 0 else ())}
                shutil.copytree(work / f"{n}.rim.tifxyz", mesh_dir / f"{n}.tifxyz")
            else:
                shutil.copytree(work / f"{n}.tifxyz", mesh_dir / f"{n}.tifxyz")
            if a.piece_objs:
                a.piece_objs.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(work / f"{n}.obj", a.piece_objs / f"{n}.obj")
        shutil.rmtree(work)
    for k, pc in enumerate(pieces):                 # #2401 cut keys, overwriting copied source keys
        mp = mesh_dir / pc["mesh"] / "meta.json"
        meta = json.loads(mp.read_text())
        meta.update({"kind": "column", "source": str(a.flat.resolve()),
                     "u0": int(math.floor(bounds[k])), "u1": int(math.floor(bounds[k + 1])),
                     "cut_rule": rule, "angle_deg": pc["angle_deg"]})
        mp.write_text(json.dumps(meta, indent=4) + "\n")
        g = read_tifxyz(mesh_dir / pc["mesh"])
        pc["shape"] = list(g.valid_vertex_mask.shape)
        pc["valid_frac"] = round(float(np.asarray(g.valid_vertex_mask).mean()), 4)
    manifest = {"stage": STAGE, "rule": rule, "flat": str(a.flat.resolve()),
                "obj": str(a.obj.resolve()) if a.obj else None, "flat_shape": [H, W],
                "flat_valid_frac": round(float(V.mean()), 4),
                "params": {"piece": a.piece, "smooth": a.smooth, "no_rotate": a.no_rotate,
                           "image": a.image if a.obj else None, "rim_fill": a.rim_fill},
                "centreline_len_cells": round(L, 2), "n_pieces": N,
                "units": "obj faces" if a.obj else "flat valid cells",
                "units_owned_once": int((owner == 1).sum()), "units_total": int(len(pts)),
                "pieces": pieces, "human_inputs": [], "human_minutes": 0}
    (a.submission / "band_cuts.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("stage", "rule", "n_pieces", "human_minutes")}))


if __name__ == "__main__":
    main()
