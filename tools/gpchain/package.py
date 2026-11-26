#!/usr/bin/env python3
"""GP package images (VES-3585 part 2; 34_prizes.md :146-147, :223): 1 cm scale bar on every image, one banner of all
columns with their numbers, letter sizes for a First Letters entry.

Pixel size. An image rendered from a tifxyz has (grid rows / scale) x (grid cols / scale) px for villa
vc_render_tifxyz --scale 1, i.e. one px per voxel along the flattened surface. px_um() does not trust that: it takes
the median 3D distance between neighbouring grid nodes (voxels) over the px per node of the image, times the voxel
size, separately along rows and columns. The bar uses the mean of the two; their difference is reported (the
flattening's stretch).

Nothing is drawn over the papyrus: the bar and the column numbers sit in a strip added below / above the image.

A First Letters entry also needs "the pixel and millimeter dimensions of a few representative letters" (:223), but its
image must carry no annotations of characters: `letters` writes a separate figure; boxes come from a person or labels.

Usage:
  python -I package.py build SUBMISSION --voxel-um 9.6 --scroll NAME --scan ID --out PKG   (new package: meshes copied,
      images/column_NN.png with the 1 cm bar, banner.png, package.json; SUBMISSION is not changed)
      [--chain CHAIN --store KEPT --image-json IMG --results RESULTS]   fields from records (records.py, VES-3629):
      package.json "fields" + false_positives.md; exit 3 while any field is MISSING (the package is still written)
      [--orient measure --orient-scroll KEY --orient-python PY]   F-596 measure on the package images
      (orient_measure.py); its op is applied to meshes and images only if the F-596 bars pass, else orientation MISSING
  python -I package.py scalebar SUBMISSION --voxel-um 9.6 [--out DIR]   images/column_NN.tif -> DIR/column_NN.png
  python -I package.py banner SUBMISSION --voxel-um 9.6 --out banner.png [--height 400]
  python -I package.py letters IMAGE MESH --boxes BOXES.json --voxel-um 9.6 --out LETTERS.png
      BOXES.json: [{"name": "alpha 1", "box": [row0, col0, row1, col1]}, ...] in IMAGE px (end exclusive)
  python -I package.py falsepos FIELDS.json --out submission/false_positives.md   (templates/false_positives.md)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def px_um(mesh_dir, img_shape, voxel_um):
    """(um per px along rows, along cols, mean) from the mesh's node spacing and the image size."""
    from orient import load_grid
    g, v = load_grid(mesh_dir)
    dr = np.linalg.norm(g[1:] - g[:-1], axis=-1)[v[1:] & v[:-1]]
    dc = np.linalg.norm(g[:, 1:] - g[:, :-1], axis=-1)[v[:, 1:] & v[:, :-1]]
    ppn_r = img_shape[0] / g.shape[0]
    ppn_c = img_shape[1] / g.shape[1]
    ur = float(np.median(dr) / ppn_r * voxel_um)
    uc = float(np.median(dc) / ppn_c * voxel_um)
    return ur, uc, (ur + uc) / 2


def font(size):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def to8(img):
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    m = float(np.nanmax(a)) if a.size else 0.0
    if m > 255 or 0 < m <= 1.0:
        a = a / m * 255
    return np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)


def with_scale_bar(img, um_per_px, label=None):
    """img (H, W) -> RGB array with a strip below holding a 1 cm bar (exact length in px) and its label.
    Returns (array, bar_px, strip_h)."""
    from PIL import Image, ImageDraw
    a = to8(img)
    H, W = a.shape
    bar = int(round(10_000.0 / um_per_px))             # 1 cm
    if bar > W - 20:
        raise ValueError(f"1 cm = {bar} px does not fit the image width {W}")
    strip = max(48, int(0.06 * H))
    out = Image.new("RGB", (W, H + strip), (255, 255, 255))
    out.paste(Image.fromarray(a).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(out)
    x0, y0 = 10, H + strip // 3
    th = max(4, strip // 8)
    d.rectangle([x0, y0, x0 + bar - 1, y0 + th - 1], fill=(0, 0, 0))
    for x in (x0, x0 + bar - 1):
        d.rectangle([x, y0 - th, x, y0 + 2 * th], fill=(0, 0, 0))
    f = font(max(12, strip // 3))
    d.text((x0 + bar + 12, y0 - th), f"1 cm ({bar} px, {um_per_px:.2f} µm/px)" + (f"  {label}" if label else ""),
           fill=(0, 0, 0), font=f)
    return np.asarray(out), bar, strip


def banner(images, labels, height=400):
    """images in reading order (list of 2D arrays, already oriented) -> one RGB banner, each resized to `height`
    px of papyrus with a strip above holding its column number; columns separated by a 6 px white gap."""
    from PIL import Image, ImageDraw
    strip = max(28, height // 8)
    tiles, xs = [], []
    for a in images:
        a = to8(a)
        w = max(1, int(round(a.shape[1] * height / a.shape[0])))
        tiles.append(Image.fromarray(a).resize((w, height), Image.BOX))
    W = sum(t.width for t in tiles) + 6 * (len(tiles) - 1)
    out = Image.new("RGB", (W, height + strip), (255, 255, 255))
    d = ImageDraw.Draw(out)
    f = font(int(strip * 0.7))
    x = 0
    for t, lab in zip(tiles, labels):
        out.paste(t.convert("RGB"), (x, strip))
        d.text((x + 4, 2), lab, fill=(0, 0, 0), font=f)
        xs.append([x, x + t.width])
        x += t.width + 6
    return np.asarray(out), xs


def letter_sizes(boxes, um_per_px_rows, um_per_px_cols):
    """[{'name', 'box': [r0, c0, r1, c1]}] -> rows with width / height in px and mm (rows = height, cols = width)."""
    out = []
    for b in boxes:
        r0, c0, r1, c1 = (int(x) for x in b["box"])
        if not (r1 > r0 and c1 > c0):
            raise ValueError(f"letter box {b} is empty")
        h, w = r1 - r0, c1 - c0
        out.append({"name": b["name"], "box": [r0, c0, r1, c1], "w_px": w, "h_px": h,
                    "w_mm": round(w * um_per_px_cols / 1000, 3), "h_mm": round(h * um_per_px_rows / 1000, 3)})
    return out


def letters_figure(img, um_rows, um_cols, boxes):
    """a separate First Letters figure: the image with the 1 cm bar, a 1 px outline around each box (no fill) and a
    size table in a strip below. Returns (array, sizes)."""
    from PIL import Image, ImageDraw
    sizes = letter_sizes(boxes, um_rows, um_cols)
    arr, bar, strip = with_scale_bar(img, (um_rows + um_cols) / 2)
    H = np.asarray(img).shape[0]
    f = font(max(12, strip // 3))
    lines = [f"{s['name']}: {s['w_px']} x {s['h_px']} px = {s['w_mm']:.2f} x {s['h_mm']:.2f} mm (w x h)" for s in sizes]
    lh = max(14, strip // 3) + 4
    out = Image.new("RGB", (arr.shape[1], arr.shape[0] + lh * len(lines) + 8), (255, 255, 255))
    out.paste(Image.fromarray(arr), (0, 0))
    d = ImageDraw.Draw(out)
    for s in sizes:
        r0, c0, r1, c1 = s["box"]
        d.rectangle([c0 - 1, r0 - 1, c1, r1], outline=(255, 0, 255))      # outside the box: letter px untouched
    for i, t in enumerate(lines):
        d.text((10, H + strip + 4 + i * lh), t, fill=(0, 0, 0), font=f)
    return np.asarray(out), sizes


def build(sub, out, voxel_um, scroll, scan, height=400, orient=None, chain=None, store=None, image_json=None,
          results=None, argv=None):
    """new GP package from a #2401 submission: meshes copied byte-equal (or taken through the measured op),
    images/column_NN.png with the 1 cm bar (the raw tif's sha256 in package.json), banner.png + banner.json.
    orient: None (= --orient none) or an orient_measure.py result {"op": OP or None, "why": ...}; an op is applied to
    the mesh grids (orient.orient_tifxyz) and the images alike, else orientation stays MISSING.
    chain: the chain output root (images.json, wNNN/maps, wNNN/chain.json). With it, package.json gets every field
    from records.py and false_positives.md is filled from them; fields with no record stay MISSING(name)."""
    import hashlib
    import shutil
    import tifffile
    from PIL import Image
    from orient import apply_op, orient_tifxyz
    sub, out = Path(sub), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; build writes a new package")
    names = sorted(p.stem for p in (sub / "images").glob("column_*.tif"))
    if not names:
        raise FileNotFoundError(f"no images/column_*.tif in {sub}")
    op = (orient or {}).get("op")
    (out / "images").mkdir(parents=True)
    if op:
        (out / "meshes").mkdir()
        for n in names:
            orient_tifxyz(sub / "meshes" / f"{n}.tifxyz", out / "meshes" / f"{n}.tifxyz", op,
                          {"source": "orient_measure.py (F-596 measure on the package images)"})
    else:
        shutil.copytree(sub / "meshes", out / "meshes")
    meta = {"scroll": scroll, "scan": scan, "voxel_um": voxel_um, "source": str(sub), "images": {},
            "orient": {"op": op, "why": (orient or {}).get("why", "--orient none")}}
    imgs = []
    for n in names:
        src = sub / "images" / f"{n}.tif"
        img = tifffile.imread(src)
        if op:
            img = np.ascontiguousarray(apply_op(img, op))
        ur, uc, u = px_um(out / "meshes" / f"{n}.tifxyz", img.shape, voxel_um)
        arr, bar, strip = with_scale_bar(img, u, f"{scroll} {n}")
        Image.fromarray(arr).save(out / "images" / f"{n}.png")
        imgs.append(img)
        meta["images"][n] = {"um_per_px_rows": ur, "um_per_px_cols": uc, "um_per_px": u, "bar_px": bar,
                             "strip_px": strip, "shape": list(img.shape), "orient_op": op,
                             "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
    arr, xs = banner(imgs, [n.split("_")[1] for n in names], height)
    Image.fromarray(arr).save(out / "banner.png")
    (out / "banner.json").write_text(json.dumps({"columns": dict(zip(names, xs)), "height": height}, indent=1) + "\n")
    if chain is not None:
        import records
        um = float(np.mean([m["um_per_px"] for m in meta["images"].values()]))
        F = records.all_fields(chain, store, image_json, um)
        F["orientation"] = (records.field(f"{op} applied to meshes and images ({orient.get('source') or 'F-596 bars passed: '
                                          + str(orient['C_MEAS']['n_measured']) + '/' + str(orient['C_MEAS']['n_columns'])
                                          + ' measured'})", [orient.get("record", "orient.json")]) if op else
                            records.field(why=f"--orient none: {meta['orient']['why']}"))
        values = {k: v["value"] for k, v in F.items()}
        if results is not None:
            values.update(records.band_fields(results, chain, um, scroll, scan, records.git_head()))
        txt, miss = fill_template((HERE / "templates/false_positives.md").read_text(), values)
        (out / "false_positives.md").write_text(txt)
        meta.update(fields=F, chain=str(chain), store=str(store), results=str(results) if results else None,
                    code_commit=records.git_head(), argv=argv,
                    filled_from_records=sorted(k for k, v in F.items() if v["value"] is not None),
                    missing=sorted(set(miss) | {k for k, v in F.items() if v["value"] is None}))
    (out / "package.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def orient_from_config(path, scroll):
    """--orient config: the scroll's measured op from orient_config.json "scrolls" (written only by a passing
    measure, e.g. F-596 attempt 5); none there -> {"op": None} and orientation stays MISSING."""
    ent = json.loads(Path(path).read_text()).get("scrolls", {}).get(scroll)
    if not (ent and ent.get("op")):
        return {"op": None, "why": f"orient_config.json has no measured op for scroll {scroll!r}"}
    return {"op": ent["op"], "why": None, "source": ent.get("source"), "record": "tools/gpchain/orient_config.json"}


def fill_template(text, fields):
    """{name} -> fields[name]; unknown names become MISSING(name). Returns (text, missing names)."""
    import re
    missing = []

    def rep(m):
        k = m.group(1)
        if k in fields and fields[k] is not None:
            return str(fields[k])
        missing.append(k)
        return f"MISSING({k})"
    return re.sub(r"\{([a-z0-9_]+)\}", rep, text), sorted(set(missing))


def main(argv=None):
    import tifffile
    from PIL import Image
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build")
    p.add_argument("submission", type=Path)
    p.add_argument("--voxel-um", type=float, required=True)
    p.add_argument("--scroll", required=True)
    p.add_argument("--scan", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--height", type=int, default=400)
    p.add_argument("--chain", type=Path, help="chain output root: fill package fields from records (records.py)")
    p.add_argument("--store", type=Path, default=Path("kept"),
                   help="tools/provenance kept store")
    p.add_argument("--image-json", type=Path, help="docker/gpchain/package.sh image record")
    p.add_argument("--results", type=Path, help="band.py results.json (score fields of false_positives.md)")
    p.add_argument("--orient", choices=("none", "measure", "config"), default="none",
                   help="measure: run orient_measure.py (F-596) on the package images; an op only if its bars pass. "
                        "config: the measured op of --orient-scroll in orient_config.json (none there = MISSING)")
    p.add_argument("--orient-scroll", help="scroll key of the reading references, e.g. PHercParis4")
    p.add_argument("--orient-python", default=sys.executable,
                   help="python with F-596's pins (torch, scipy 1.16.2): the image's /opt/venvs/cpu")
    p = sp.add_parser("letters")
    p.add_argument("image", type=Path)
    p.add_argument("mesh", type=Path)
    p.add_argument("--boxes", type=Path, required=True)
    p.add_argument("--voxel-um", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    for n in ("scalebar", "banner"):
        p = sp.add_parser(n)
        p.add_argument("submission", type=Path)
        p.add_argument("--voxel-um", type=float, required=True)
        p.add_argument("--out", type=Path)
        p.add_argument("--height", type=int, default=400)
    p = sp.add_parser("falsepos")
    p.add_argument("fields", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--template", type=Path, default=HERE / "templates/false_positives.md")
    a = ap.parse_args(argv)
    if a.cmd == "falsepos":
        txt, miss = fill_template(a.template.read_text(), json.loads(a.fields.read_text()))
        a.out.write_text(txt)
        if miss:
            sys.exit(f"package: false_positives.md has MISSING fields: {', '.join(miss)}")
        return 0
    if a.cmd == "build":
        import subprocess
        import time
        t0 = time.time()
        orient = None
        if a.orient == "measure":
            if a.out.exists():
                raise FileExistsError(f"{a.out} exists; build writes a new package")
            oj = a.out.with_name(a.out.name + ".orient.json")
            r = subprocess.run([a.orient_python, str(HERE / "orient_measure.py"), str(a.submission),
                                "--scroll", a.orient_scroll or a.scroll, "--out", str(oj)])
            orient = json.loads(oj.read_text()) if r.returncode == 0 and oj.is_file() else \
                {"op": None, "why": f"orient_measure.py exit {r.returncode}"}
        elif a.orient == "config":
            orient = orient_from_config(HERE / "orient_config.json", a.orient_scroll or a.scroll)
        t1 = time.time()
        m = build(a.submission, a.out, a.voxel_um, a.scroll, a.scan, a.height, orient=orient, chain=a.chain,
                  store=a.store, image_json=a.image_json, results=a.results, argv=sys.argv)
        if orient is not None:
            (a.out / "orient.json").write_text(json.dumps(orient, indent=1, default=str) + "\n")
        m["wall_s"] = {"orient": round(t1 - t0, 1), "build": round(time.time() - t1, 1)}
        (a.out / "package.json").write_text(json.dumps(m, indent=1) + "\n")
        print(f"package: {len(m['images'])} images with a 1 cm bar, banner.png -> {a.out}; orient "
              f"{m['orient']['op'] or 'none'}; wall {m['wall_s']}")
        if m.get("missing"):
            print(f"package: MISSING {len(m['missing'])}: {', '.join(m['missing'])}", file=sys.stderr)
            return 3
        return 0
    if a.cmd == "letters":
        img = tifffile.imread(a.image) if a.image.suffix in (".tif", ".tiff") else np.asarray(Image.open(a.image).convert("L"))
        ur, uc, _ = px_um(a.mesh, img.shape, a.voxel_um)
        arr, sizes = letters_figure(img, ur, uc, json.loads(a.boxes.read_text()))
        Image.fromarray(arr).save(a.out)
        a.out.with_suffix(".json").write_text(json.dumps({"um_per_px_rows": ur, "um_per_px_cols": uc,
                                                          "letters": sizes}, indent=1) + "\n")
        return 0
    sub = a.submission
    names = sorted(p.stem for p in (sub / "images").glob("column_*.tif"))
    if a.cmd == "scalebar":
        out = a.out or sub / "images_scalebar"
        out.mkdir(parents=True, exist_ok=True)
        rep = {}
        for n in names:
            img = tifffile.imread(sub / "images" / f"{n}.tif")
            ur, uc, u = px_um(sub / "meshes" / f"{n}.tifxyz", img.shape, a.voxel_um)
            arr, bar, strip = with_scale_bar(img, u, n)
            Image.fromarray(arr).save(out / f"{n}.png")
            rep[n] = {"um_per_px_rows": ur, "um_per_px_cols": uc, "um_per_px": u, "bar_px": bar, "strip_px": strip}
        (out / "scalebar.json").write_text(json.dumps(rep, indent=1) + "\n")
    else:
        imgs = [tifffile.imread(sub / "images" / f"{n}.tif") for n in names]
        arr, xs = banner(imgs, [n.split("_")[1] for n in names], a.height)
        Image.fromarray(arr).save(a.out)
        (a.out.with_suffix(".json")).write_text(json.dumps({"columns": dict(zip(names, xs)), "height": a.height},
                                                           indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
