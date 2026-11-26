#!/usr/bin/env python3
"""Chain outputs (one dir per winding) -> a Grand Prize submission folder -> the format check of villa PR #2401
(a draft pull request, closed by its author on 2026-11-24, not merged; not in villa, not an organiser tool).

Every piece that has an S map becomes column_NN (windings in the given order, pieces by name):
meshes/column_NN.tifxyz (the chain's flattened piece, copied unchanged) and images/column_NN.tif (the S map, uint8,
zlib), the layout of villa PR #2401. Then `vesuvius.gp_submission check --submission SUB` (villa PR #2401 head
ea48ce6b) writes SUB/check_report.json. Same steps as the stored runs' `images` step.

  python make_folder.py SUB WINDING_OUT [WINDING_OUT ...] [--map S]     (run in the w2401 venv)
Writes SUB/ and SUB.folder.json (columns -> winding/piece, check exit, required gates). Exit = the check's exit.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

REQUIRED = ('tifxyz_format', 'column_naming', 'image_names_match_meshes')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('sub', type=Path)
    ap.add_argument('windings', type=Path, nargs='+')
    ap.add_argument('--map', default='S')
    a = ap.parse_args(argv)
    sub = a.sub
    if sub.exists():
        raise SystemExit(f'{sub} exists')
    (sub / 'meshes').mkdir(parents=True)
    (sub / 'images').mkdir()
    cols = {}
    for d in a.windings:
        for piece in sorted((d / 'submission/meshes').glob('column_*.tifxyz')):
            name = piece.name[:-len('.tifxyz')]
            m = d / 'maps' / f'{a.map}_{name}.npy'
            if not m.exists():
                continue
            col = f'column_{len(cols) + 1:02d}'                 # check rule: column_01..99, then column_100
            cols[col] = f'{d.name}/{name}'
            shutil.copytree(piece, sub / 'meshes' / f'{col}.tifxyz')
            tifffile.imwrite(sub / 'images' / f'{col}.tif', np.load(m), compression='zlib')
    exe = Path(sys.executable).parent / 'vesuvius.gp_submission'
    r = subprocess.run([str(exe), 'check', '--submission', str(sub)], text=True, capture_output=True)
    rep = json.loads((sub / 'check_report.json').read_text())
    req = {g['name']: bool(g['passed']) for g in rep['gates'] if g['required']}
    res = dict(columns=cols, exit=r.returncode, status=rep.get('status'), required=req,
               required_set_ok=sorted(req) == sorted(REQUIRED), n_required_pass=sum(req.values()),
               warnings=[g['name'] for g in rep['gates'] if not g['required'] and not g['passed']])
    res['ok'] = bool(res['exit'] == 0 and res['required_set_ok'] and res['n_required_pass'] == len(REQUIRED))
    sub.with_name(sub.name + '.folder.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: res[k] for k in ('exit', 'status', 'required', 'ok')}, indent=1))
    return r.returncode


if __name__ == '__main__':
    sys.exit(main())
