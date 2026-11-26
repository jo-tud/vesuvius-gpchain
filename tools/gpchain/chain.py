#!/usr/bin/env python3
"""GP output chain driver: fit tifxyz -> (one winding) -> isometric stage -> band cuts, with the
route picked by the fit's repair state and the human-input ledger summed per run.

Why: VES-2611 (decision on F-439). The F-439 stage (isometric_step.py --density 4, then rim_fill.py
--r 0.5 --max-edge 2 after every stock vc_obj2tifxyz) is our default, but only on repaired fits
(F-438 route, fold_repair.py). On an unrepaired fit stock density-4 quads still cross the tears
(F-439: 36 quads > 4x median on unrepaired w130, 0 on the repaired fit), so the chain keeps the
F-429 path there and logs why. Parameters live in chain_config.json next to this file.

Routes (chain_config.json)
  repaired   FIT/fold_repair.json exists, its stage is "fold-repair" and its out_sha256 equals the
             fit's own x/y/z.tif. Anything else is unrepaired; chain.json names the reason.
  f439       isometric_step.py --keep-obj --density 4 -> rim_fill.py (stock output + flatboi OBJ)
             --r 0.5 --max-edge 2 -> band_cuts.py --obj --piece 1200 --rim-fill 0.5 --rim-max-edge 2
  f429       isometric_step.py --keep-obj -> band_cuts.py --obj --piece 300 (no fill)
  Both piece lengths are 57.6 mm at 9.6 um (300 cells at the fit's grid step, x density).
Report-only, never a route switch: quads with 3D area > 4x the median complete quad (the F-439
tear signature) in the stock stage output, the stage output and each band piece.

Render stage (--render CT_ROOT, F-576 / VES-3519; render_contract.py, render_config.json)
  per band piece: villa vc_render_tifxyz (16 layers, docker) -> OUT/render/column_NN.zarr; the per-mesh layer
  order (piece normal vs the --segments label meshes, else vs --axis-xy; never copied between meshes) and FC on
  render-data px + the rim gate, both into chain.json "render". render_contract.py infer reads that section and
  refuses pieces with FC not ok or no order.

CT cut (--ct-cut, VES-3760; ct_cut.py) and --umbilicus (VES-3760)
  --ct-cut: after the band cuts (and --orient) and before the render, each piece is cut to the CT-valid region of the
  render CT (chunk present and CT > 0 at one of the node's 16 render-layer positions; ct_cut.py); a piece with no
  complete quad left moves to OUT/ct-cut-away and is not rendered; chain.json "ct_cut". --ct-cut-px (VES-3782): also
  drop quads with a render px whose 16 layers are all CT < 1 (ct_cut.py quad-level test). --umbilicus FILE: the radial
  layer-order reference is the umbilicus xy at each node's z (render_contract.Umbilicus) instead of --axis-xy.

Orient stage (--orient rule|OP, VES-3585; orient.py, orient_config.json)
  after the band cuts and before --render: each piece's x/y/z grid is turned / mirrored by the dihedral op that puts
  its image into reading orientation (rule: text on the core side, read outer -> inner; h from the fit's spiral
  with --axis-xy), so the render of column_NN.tifxyz is upright by itself. The op and its inputs go into the
  piece's meta.json "orient" and chain.json "orient". Default none (pieces unchanged). "rule" runs the geometric
  rule only while orient_config.json has it "confirmed"; otherwise it uses that file's measured op for
  --orient-scroll, else it refuses.

Contract
  input   FIT.tifxyz (a spiral-fit export or a fold_repair.py output; volume voxels of the fit
          frame). --cols U0 U1: first cut grid columns [U0, U1) of FIT with villa #2401
          `cut-columns` (one piece, as F-437 / F-439 did for w130: 0 652).
  output  OUT/src (the cut, if --cols), OUT/stage-stock (f439 only: the stock stage),
          OUT/stage/flat.tifxyz (+ .flat.obj), OUT/submission/meshes/column_NN.tifxyz (#2401 layout,
          opens in VC3D unchanged), OUT/chain.json: route and reason, config sha256, every stage
          manifest, big-quad counts and the human-input ledger (all entries, minutes summed);
          with --render also OUT/render/column_NN.zarr and chain.json "render" (order, FC per piece).
Runs in the villa #2401 venv (vesuvius, numpy, scipy, tifffile) with docker and villa-ink:ves1471.
Deterministic: each stage is (VES-2481, VES-2570).

Usage: python -I tools/gpchain/chain.py FIT.tifxyz OUT [--cols U0 U1] [--config chain_config.json]
       [--iters N] [--piece-objs] [--render CT_ROOT.zarr [--segments SEG.tifxyz ...] [--axis-xy X Y]
       [--umbilicus UMBILICUS.json] [--ct-cut [--ct-cut-px]] [--render-config render_config.json]] [--orient {none,rule,OP} [--orient-config orient_config.json]]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # python -I: no script dir on the path

STAGE = "chain"


def sha256s(d):
    return {f: hashlib.sha256((Path(d) / f).read_bytes()).hexdigest() for f in ("x.tif", "y.tif", "z.tif")}


def repair_state(fit):
    """(repaired, reason, fold_repair summary or None)."""
    mf = fit / "fold_repair.json"
    if not mf.is_file():
        return False, "no fold_repair.json in the fit (not an F-438 repaired fit)", None
    m = json.loads(mf.read_text())
    summ = {k: m.get(k) for k in ("src", "params", "zone_cols", "bad_quads_in", "bad_quads_out", "area_kept")}
    if m.get("stage") != "fold-repair":
        return False, f"fold_repair.json stage is {m.get('stage')!r}, not 'fold-repair'", summ
    if m.get("out_sha256") != sha256s(fit):
        return False, "fold_repair.json out_sha256 does not match the fit's x/y/z.tif", summ
    return True, "fold_repair.json matches the fit bytes (F-438 repaired fit)", summ


def big_quads(d, k):
    """complete quads with 3D area > k x the median complete quad (areacheck.py's rule)."""
    import numpy as np
    from crumple_mask import area3d_per_quad, load
    p, v = load(d)
    qa = area3d_per_quad(p, v)
    q = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
    med = float(np.median(qa[q])) if q.any() else 0.0
    return int((q & (qa > k * med)).sum())


def run(name, cmd):
    r = subprocess.run(list(map(str, cmd)), text=True, capture_output=True)
    if r.returncode != 0:
        sys.stderr.write((r.stdout + r.stderr)[-4000:])
        sys.exit(f"chain: {name} failed (exit {r.returncode})")
    return r


def render_stage(out, meshes, ct_root, segments, axis_xy, cfg_path, umbilicus=None):
    """villa render + per-mesh layer order + FC per band piece (render_contract.py)."""
    import render_contract as rcm
    cfg = rcm.load_config(cfg_path)
    umb = rcm.Umbilicus(umbilicus) if umbilicus else None
    ok, got = rcm.render_image_ok(cfg)
    if not ok:
        sys.exit(f"chain: render image {cfg['render']['image']} is {got}, not {cfg['render']['image_id']}")
    rd = out / "render"
    (rd / "vc3dcfg").mkdir(parents=True, exist_ok=True)
    segs = rcm.SegmentNormals(segments) if segments else None
    Z = rcm.open_array(Path(ct_root) / "0")
    res = {}
    for m in meshes:
        name = m.name[:-len(".tifxyz")]
        dst, part = rd / f"{name}.zarr", rd / f"{name}.zarr.part"
        t0 = time.time()
        r = run(f"render {name}", rcm.render_cmd(m, ct_root, part, cfg))
        (rd / f"{name}.log").write_text(r.stdout + r.stderr)
        part.rename(dst)
        t_render = round(time.time() - t0, 1)
        V = rcm.open_render(dst)
        res[name] = {"piece": str(m.relative_to(out)), "render": str(dst.relative_to(out)), "render_s": t_render,
                     "order": rcm.layer_order(m, segs, axis_xy, cfg, umb), "frame": rcm.frame_check(m, V, Z, cfg)}
        print(f"chain: {name} order {res[name]['order']['order']} ({res[name]['order']['reference']}), "
              f"FC r {res[name]['frame']['r']:.4f} rim {res[name]['frame']['rim']['share']:.4f} "
              f"ok {res[name]['frame']['ok']}", flush=True)
    return res, hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()


def orient_stage(fit, meshes, mode, axis_xy, cfg_path, scroll=None):
    """turn / mirror each band piece grid into reading orientation (orient.py); returns the chain.json section.
    mode rule: the geometric rule only if orient_config.json has it confirmed; else the scroll's measured op from
    orient_config.json "scrolls", else refuse (fail closed)."""
    import numpy as np
    import orient as om
    cfg = json.loads(Path(cfg_path).read_text())
    sec = {"mode": mode, "config_sha256": hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest(),
           "rule_status": cfg["rule"]["status"], "axis_xy": axis_xy, "pieces": {}}
    if mode == "rule" and cfg["rule"]["status"] != "confirmed":
        got = (cfg.get("scrolls", {}).get(scroll) or {}).get("op") if scroll else None
        if got not in om.OPS:
            sys.exit(f"chain: the orientation rule is {cfg['rule']['status']} ({cfg['rule'].get('why')}) and "
                     f"{cfg_path} has no measured op for scroll {scroll!r} (--orient-scroll); pass --orient OP")
        sec.update(scroll=scroll, from_config=cfg["scrolls"][scroll])
        mode = got
    if mode == "rule":
        gf, vf = om.load_grid(fit)
        hd = om.handedness(gf, vf, axis_xy)
        sec["handedness"] = hd
        if hd["h"] is None:
            sys.exit(f"chain: --orient rule needs the fit's turning sense: {hd['why']}")
    for m in meshes:
        if mode == "rule":
            g, v = om.load_grid(m)
            r = om.rule_op(g, v, axis_xy, sec["handedness"]["h"])
            info = {k: r[k] for k in ("h", "axis_xy", "node_agree", "mirror_by_normal", "dots")}
            op = r["op"]
        else:
            op, info = mode, ({"from_config": scroll} if "from_config" in sec else {"given": True})
        tmp = m.with_name(m.name + ".orient")
        om.orient_tifxyz(m, tmp, op, info)
        shutil.rmtree(m)
        tmp.rename(m)
        sec["pieces"][m.name] = {"op": op, **info}
        print(f"chain: {m.name} orient {op}", flush=True)
    return sec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fit", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cols", type=int, nargs=2, metavar=("U0", "U1"))
    ap.add_argument("--config", type=Path, default=HERE / "chain_config.json")
    ap.add_argument("--iters", type=int, help="flatboi iterations (default: the route's; 200 for > 1 winding)")
    ap.add_argument("--piece-objs", action="store_true", help="keep band piece OBJs in OUT/piece-objs (area audit)")
    ap.add_argument("--render", type=Path, metavar="CT_ROOT", help="zarr root (group 0 = the fit's voxel frame): "
                    "render each piece, layer order + FC into chain.json")
    ap.add_argument("--segments", type=Path, nargs="+", help="label segment tifxyz dirs (layer-order reference)")
    ap.add_argument("--axis-xy", type=float, nargs=2, help="scroll axis x y (radial layer-order reference)")
    ap.add_argument("--umbilicus", type=Path, help="umbilicus.json (control points x y z): radial layer-order "
                    "reference at each node's z, instead of --axis-xy (VES-3760)")
    ap.add_argument("--ct-cut", action="store_true", help="--render only: cut each piece to the CT-valid region "
                    "before the render (ct_cut.py, VES-3760); cut-away pieces go to OUT/ct-cut-away")
    ap.add_argument("--ct-cut-px", action="store_true", help="with --ct-cut: + the quad-level test (every render px "
                    "of a kept quad has CT >= 1 in one of its 16 layers; ct_cut.py, VES-3782)")
    ap.add_argument("--render-config", type=Path, default=HERE / "render_config.json")
    ap.add_argument("--orient", default="none", help="none (default), rule (needs --axis-xy) or a dihedral op "
                    "(r0 r90 r180 r270 m0 m90 m180 m270): put each piece into reading orientation (orient.py)")
    ap.add_argument("--orient-config", type=Path, default=HERE / "orient_config.json")
    ap.add_argument("--orient-scroll", help="scroll key in orient_config.json scrolls (used by --orient rule while "
                    "the rule is not confirmed)")
    a = ap.parse_args(argv)
    import orient as om
    if a.orient not in ("none", "rule", *om.OPS):
        sys.exit(f"chain: --orient {a.orient}: not none, rule or one of {om.OPS}")
    if a.orient == "rule" and not a.axis_xy:
        sys.exit("chain: --orient rule needs --axis-xy")
    fit, out = a.fit.resolve(), a.out.resolve()
    for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
        if not (fit / f).is_file():
            sys.exit(f"chain: {fit} is not a tifxyz (missing {f})")
    if a.segments and not a.render:
        sys.exit("chain: --segments needs --render")
    if a.axis_xy and not (a.render or a.orient == "rule"):
        sys.exit("chain: --axis-xy needs --render or --orient rule")
    if (a.umbilicus or a.ct_cut) and not a.render:
        sys.exit("chain: --umbilicus and --ct-cut need --render")
    if a.ct_cut_px and not a.ct_cut:
        sys.exit("chain: --ct-cut-px needs --ct-cut")
    if a.umbilicus and a.axis_xy:
        sys.exit("chain: --umbilicus or --axis-xy, not both")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"chain: {out} is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    cfg_bytes = a.config.read_bytes()
    cfg = json.loads(cfg_bytes)
    t0 = time.time()

    repaired, reason, repair = repair_state(fit)
    route = "f439" if repaired else "f429"
    rc = cfg["routes"][route]
    if not repaired:
        reason += "; " + rc["why"]
    print(f"chain: route {route} ({rc['finding']}): {reason}", flush=True)

    bindir = Path(sys.executable).parent
    src = fit
    if a.cols:
        sub = out / "src"
        sub.mkdir()
        csv = out / "src_cuts.csv"
        csv.write_text(f"column,u0,u1\n1,{a.cols[0]},{a.cols[1]}\n")
        run("cut-columns", [bindir / "vesuvius.gp_submission", "cut-columns", "--submission", sub,
                             "--surface", fit, "--cuts", csv])
        src = sub / "meshes" / "column_01.tifxyz"

    iso, band = rc["isometric_step"], rc["band_cuts"]
    iters = a.iters or iso["iters"]
    flat = out / "stage" / "flat.tifxyz"
    stock = out / "stage-stock" / "flat.tifxyz" if rc["rim_fill"] else flat
    cmd = [sys.executable, "-I", HERE / "isometric_step.py", src, stock, "--keep-obj", "--iters", iters]
    if iso["density"] != 1:
        cmd += ["--density", iso["density"]]
    run("isometric_step", cmd)
    obj = Path(str(stock) + ".flat.obj")
    stages = [json.loads(Path(str(stock) + ".stage.json").read_text())]
    if rc["rim_fill"]:
        from rim_fill import rim_fill
        rf = rc["rim_fill"]
        m = rim_fill(stock, obj, flat, rf["r"], rf["max_edge"])
        (flat / "rim_fill.json").rename(flat.parent / "rim_fill.json")
        shutil.copyfile(obj, Path(str(flat) + ".flat.obj"))
        stages.append(m)
    flat_obj = Path(str(flat) + ".flat.obj")

    sub = out / "submission"
    cmd = [sys.executable, "-I", HERE / "band_cuts.py", flat, sub, "--obj", flat_obj,
           "--piece", band["piece"], "--smooth", band["smooth"]]
    if band["rim_fill"]:
        cmd += ["--rim-fill", band["rim_fill"]]
        if band["rim_max_edge"]:
            cmd += ["--rim-max-edge", band["rim_max_edge"]]
    if a.piece_objs:
        cmd += ["--piece-objs", out / "piece-objs"]
    run("band_cuts", cmd)
    bm = json.loads((sub / "band_cuts.json").read_text())
    stages.append(bm)

    k = cfg["big_quad_factor"]
    meshes = sorted((sub / "meshes").glob("column_*.tifxyz"))
    bq = {"factor": k, "stage": big_quads(flat, k),
          "pieces": {m.name: big_quads(m, k) for m in meshes}}
    if rc["rim_fill"]:
        bq["stage_stock"] = big_quads(stock, k)
    orient = None
    if a.orient != "none":
        orient = orient_stage(fit, meshes, a.orient, a.axis_xy, a.orient_config.resolve(), a.orient_scroll)
    render = render_sha = ct_cut = None
    if a.ct_cut:
        from ct_cut import ct_cut_stage
        meshes, ct_cut = ct_cut_stage(out, meshes, a.render.resolve(), px=a.ct_cut_px)
        stages.append(ct_cut)
    if a.render:
        render, render_sha = render_stage(out, meshes, a.render.resolve(), a.segments, a.axis_xy,
                                          a.render_config.resolve(), a.umbilicus)
    human = [h for s in stages for h in s.get("human_inputs", [])]
    man = {"stage": STAGE, "decision": cfg.get("decision"), "fit": str(fit), "fit_sha256": sha256s(fit),
           "cols": a.cols, "config": str(a.config.resolve()),
           "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
           "route": route, "finding": rc["finding"], "route_reason": reason, "repair": repair,
           "params": {"isometric_step": {**iso, "iters": iters}, "rim_fill": rc["rim_fill"], "band_cuts": band},
           "stage_out": str(flat), "stage_out_sha256": sha256s(flat),
           "pieces": {m.name: sha256s(m) for m in meshes}, "big_quads": bq,
           "render_ct": str(a.render.resolve()) if a.render else None, "render_config_sha256": render_sha,
           "render": render, "orient": orient, "ct_cut": ct_cut,
           "umbilicus": str(a.umbilicus.resolve()) if a.umbilicus else None,
           "stages": stages, "wall_s": round(time.time() - t0, 1),
           "human_inputs": human, "human_minutes": sum(s["human_minutes"] for s in stages)}
    (out / "chain.json").write_text(json.dumps(man, indent=1, default=str) + "\n")
    print(json.dumps({x: man[x] for x in ("route", "route_reason", "big_quads", "wall_s", "human_minutes")}))


if __name__ == "__main__":
    main()
