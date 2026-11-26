#!/usr/bin/env python3
"""Fetch the CT chunks that a set of fitted winding meshes touch, from the public Vesuvius Challenge bucket, into a
local OME-Zarr root whose group 0 is the source level (here level 2 = 9.6 um, the fit's voxel frame).

Rule (unchanged from the stored runs): every valid mesh node, shifted by -PAD, 0, +PAD voxels on each axis (27
offsets), floor-divided by the chunk edge, gives a (z, y, x) chunk key; chunks outside the array are skipped; a chunk
the bucket does not have (HTTP 404) is left out, which zarr reads as the fill value 0.

  python fetch_ct.py ROOT MESH.tifxyz [MESH.tifxyz ...] [--src URL] [--level 2] [--pad 24] [--max-gb 40]
  (run in the w2401 venv: numpy, tifffile)
Writes ROOT/.zattrs, ROOT/.zgroup, ROOT/0/.zarray, ROOT/0/<z>/<y>/<x> and ROOT/fetch.json (counts, GB, seconds).
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

SRC = ('https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/'
       '20260411134726-2.400um-0.2m-78keV-masked.zarr')


def load_grid(d):
    g = np.stack([tifffile.imread(Path(d) / f'{c}.tif').astype(np.float64) for c in 'xyz'], -1)
    v = np.all(np.isfinite(g), -1) & ~np.all(g == -1, -1) & (g[..., 2] >= 0)
    return g, v


def touched(meshes, ch, pad):
    keys = set()
    for m in meshes:
        g, v = load_grid(m)
        p = g[v]
        for dz in (-pad, 0, pad):
            for dy in (-pad, 0, pad):
                for dx in (-pad, 0, pad):
                    q = np.floor((p[:, ::-1] + np.array([dz, dy, dx])) / ch).astype(int)   # (z, y, x) keys
                    keys |= set(map(tuple, np.unique(q, axis=0)))
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('root', type=Path)
    ap.add_argument('meshes', type=Path, nargs='+')
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--level', type=int, default=2)
    ap.add_argument('--pad', type=int, default=24)
    ap.add_argument('--max-gb', type=float, default=40)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(argv)
    root, src = a.root, a.src.rstrip('/')
    (root / '0').mkdir(parents=True, exist_ok=True)
    meta = json.loads(urllib.request.urlopen(src + '/.zattrs').read())
    ms = meta['multiscales'][0]
    d = dict(ms['datasets'][a.level])
    d['path'] = '0'
    ms['datasets'] = [d]
    ms['name'] = f'{src.rsplit("/", 1)[-1]} re-rooted: 0 = source level {a.level}, touched chunks only'
    (root / '.zattrs').write_text(json.dumps(meta, indent=1))
    (root / '.zgroup').write_bytes(urllib.request.urlopen(src + '/.zgroup').read())
    za = urllib.request.urlopen(f'{src}/{a.level}/.zarray').read()
    (root / '0' / '.zarray').write_bytes(za)
    zj = json.loads(za)
    ch = zj['chunks'][0]
    assert zj['chunks'] == [ch] * 3, zj['chunks']
    shp = zj['shape']
    keys = sorted(k for k in touched(a.meshes, ch, a.pad) if all(0 <= k[i] < -(-shp[i] // ch) for i in range(3)))
    est = len(keys) * ch ** 3 * np.dtype(zj['dtype']).itemsize / 1e9
    print(len(keys), 'chunks, at most', round(est, 1), 'GB', flush=True)
    if est > a.max_gb:
        raise SystemExit(f'fetch_ct: {est:.1f} GB > --max-gb {a.max_gb}')

    def get(k):
        rel = f'{k[0]}/{k[1]}/{k[2]}'
        dst = root / '0' / rel
        if dst.exists():
            return 0, True
        dst.parent.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            try:
                b = urllib.request.urlopen(f'{src}/{a.level}/{rel}', timeout=60).read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return 0, False                          # fill value 0, as zarr reads it
                time.sleep(2 ** i)
            except Exception:                                # noqa: BLE001 (retry any network error)
                time.sleep(2 ** i)
        else:
            raise RuntimeError(f'fetch failed {k}')
        tmp = dst.with_suffix('.part')
        tmp.write_bytes(b)
        tmp.replace(dst)
        return len(b), True

    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(get, keys))
    info = dict(src=src, level=a.level, pad_vx=a.pad, meshes=[str(m) for m in a.meshes], n_chunks=len(keys),
                n_missing_remote=sum(not r[1] for r in res), gb=round(sum(r[0] for r in res) / 1e9, 2),
                seconds=round(time.time() - t0, 1))
    (root / 'fetch.json').write_text(json.dumps(info, indent=1) + '\n')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
