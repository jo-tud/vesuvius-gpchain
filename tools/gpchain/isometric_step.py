#!/usr/bin/env python3
"""GP output chain, stage "isometric": make a fit tifxyz arc-length isometric with stock VC3D tools.

Why: villa spiral-fit exports space grid columns by fit angle, not arc length (F-143, F-424:
u stretch median 1.35, p95 2.58), so villa #2401 `gp_submission check` warns `mesh_isometry`
and a render squeezes letters about 1.35x along the line. One SLIM flatten fixes it (F-426).

Contract
  input   a tifxyz (x/y/z.tif + meta.json), points in volume voxels of the fit frame, any grid scale.
  output  a tifxyz in the same voxel frame, re-gridded at the input's grid scale (meta "scale"),
          grid (row, col) * 1/scale = flat-map position in voxels (isometric to the 3D surface).
          The flat strip may bend in UV, so the output grid is taller and has more invalid cells.
  tools   (all stock, image villa-ink:ves1471, villa 6bbe6e21; nothing reimplemented here)
          vc_tifxyz2obj IN OUT.obj ; flatboi OUT.obj ITERS ENERGY ; vc_obj2tifxyz OUT_flatboi.obj
          flat.tifxyz --tifxyz-source=IN   (the same tools VC3D's SLIM-flatten action uses)
  manifest  OUT.stage.json: input / output sha256, image id, parameters, wall time, flatboi's own
          final stretch report, and the human-input ledger entry for this stage (none: 0 minutes).

Usage: python3 tools/gpchain/isometric_step.py IN.tifxyz OUT.tifxyz [--iters 50]
       [--energy symmetric_dirichlet] [--threads 8] [--memory 16g] [--image villa-ink:ves1471] [--keep-obj]
       [--density 1]
--density K (VES-2570, default 1 = unchanged): the stock vc_obj2tifxyz re-grids K x denser per axis
(its --tifxyz-source is a copy of IN whose meta "scale" is K x), so OUT has scale K x IN's.
The stock re-grid drops ~1/2 grid step all round the surface (VES-2570); tools/gpchain/rim_fill.py
gives that rim back as a separate stage after this one.
Deterministic: two runs give byte-identical x/y/z.tif (VES-2481). CPU, ~15 s per 66 x 900 strip.
More than one winding: use --iters 200 (VES-2547: 50 iterations leave 10 windings unconverged,
L2 median 1.10; 200 give 1.001 in 435 s). A crumpled fit patch stays non-isometric at any count.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STAGE = "isometric"
DEFAULT_IMAGE = "villa-ink:ves1471"


def sha256s(d):
    return {f: hashlib.sha256((Path(d) / f).read_bytes()).hexdigest() for f in ("x.tif", "y.tif", "z.tif")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--energy", default="symmetric_dirichlet")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory", default="16g")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--density", type=int, default=1, help="output grid K x the input density per axis")
    ap.add_argument("--keep-obj", action="store_true",
                    help="also keep flatboi's OBJ (3D + flat UV) as OUT.flat.obj, for band_cuts.py --obj")
    a = ap.parse_args(argv)
    if a.threads > 12:
        sys.exit("isometric_step: at most 12 threads on the shared machine")
    if a.density < 1:
        sys.exit("isometric_step: --density must be >= 1")
    src, out = a.src.resolve(), a.out.resolve()
    for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
        if not (src / f).is_file():
            sys.exit(f"isometric_step: {src} is not a tifxyz (missing {f})")
    if out.exists():
        sys.exit(f"isometric_step: {out} exists; refusing to overwrite")
    image_id = subprocess.run(["docker", "image", "inspect", a.image, "--format", "{{.Id}}"],
                              text=True, capture_output=True).stdout.strip()
    if not image_id:
        sys.exit(f"isometric_step: docker image {a.image} not found")

    with tempfile.TemporaryDirectory(prefix="gpchain-iso-") as tmp:
        w = Path(tmp)
        shutil.copytree(src, w / "src.tifxyz")
        grid_src = "src.tifxyz"
        if a.density != 1:                      # same points, meta scale x K: sizes the stock re-grid
            shutil.copytree(src, w / "grid.tifxyz")
            meta = json.loads((src / "meta.json").read_text())
            meta["scale"] = [float(x) * a.density for x in meta["scale"]]
            (w / "grid.tifxyz" / "meta.json").write_text(json.dumps(meta, indent=4) + "\n")
            grid_src = "grid.tifxyz"
        script = (f"set -e; cd /w; vc_tifxyz2obj src.tifxyz src.obj; "
                  f"flatboi src.obj {a.iters} {a.energy}; "
                  f"vc_obj2tifxyz src_flatboi.obj flat.tifxyz --tifxyz-source={grid_src}")
        t0 = time.time()
        r = subprocess.run(["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                            "--cpus", str(a.threads), "--memory", a.memory, "--network", "none",
                            "-e", f"OMP_NUM_THREADS={a.threads}", "-v", f"{w}:/w", "--entrypoint", "sh",
                            a.image, "-c", script], text=True, capture_output=True)
        wall = time.time() - t0
        log = r.stdout + "\n--- stderr ---\n" + r.stderr
        if r.returncode != 0:
            sys.stderr.write(log[-4000:])
            sys.exit(f"isometric_step: tool chain failed (exit {r.returncode})")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(w / "flat.tifxyz", out)
        out.with_name(out.name + ".log").write_text(log)
        if a.keep_obj:
            shutil.copyfile(w / "src_flatboi.obj", out.with_name(out.name + ".flat.obj"))
        report = [l for l in log.splitlines()
                  if re.search(r"stretch|area error|energy|Final", l, re.I)][-12:]

    manifest = {
        "stage": STAGE, "src": str(src), "out": str(out),
        "src_sha256": sha256s(src), "out_sha256": sha256s(out),
        "src_meta_scale": json.loads((src / "meta.json").read_text()).get("scale"),
        "out_meta_scale": json.loads((out / "meta.json").read_text()).get("scale"),
        "tools": ["vc_tifxyz2obj", "flatboi", "vc_obj2tifxyz --tifxyz-source"],
        "image": a.image, "image_id": image_id, "villa_commit": "6bbe6e21",
        "params": {"iters": a.iters, "energy": a.energy, "threads": a.threads,
                   **({"density": a.density} if a.density != 1 else {})},
        "wall_s": round(wall, 1), "flatboi_report": report,
        "human_inputs": [], "human_minutes": 0,
    }
    if a.keep_obj:
        obj = out.with_name(out.name + ".flat.obj")
        manifest["flat_obj"] = {"path": str(obj), "sha256": hashlib.sha256(obj.read_bytes()).hexdigest()}
    out.with_name(out.name + ".stage.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("stage", "out", "wall_s", "human_minutes")}))


if __name__ == "__main__":
    main()
