#!/usr/bin/env python3
"""Get every input gpchain reads for the Paris 4 example and check each file's sha256 against inputs/manifest.json.

  python get_inputs.py DEST --windings w011 w108 w129 --fit-from DIR --ckpt-from S.pth [--frozen]
  (run in the w2401 venv; needs network for the public files)

DEST/fit/wNNN_spliced/   our fit's per-winding meshes   (--fit-from: a folder holding wNNN_spliced/; HOSTING: to add)
DEST/segments/<id>/      4 organiser segment meshes at level 2, made from the public S3 meshes by segments_to_l2.py
DEST/umbilicus.json      the organisers' Paris 4 umbilicus (dl.ash2txt.org spiral_datasets)
DEST/models/hecate.py    hecate's model code (Hugging Face scrollprize/hecate, pinned revision)
DEST/models/S.pth        the S checkpoint (--ckpt-from; HOSTING: the S release)
DEST/models/hecate_9.6um.pth   with --frozen: the frozen hecate weights (Hugging Face scrollprize/hecate)
Writes DEST/inputs.json (each file: sha256 wanted, got, ok). Exit 1 if any file differs.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / 'inputs/manifest.json'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def download(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    for a in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dst.with_suffix('.part'), 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            dst.with_suffix('.part').replace(dst)
            return
        except Exception as e:                                  # noqa: BLE001 (retry any network error)
            if a == 5:
                raise RuntimeError(f'download failed: {url}: {e}')
            time.sleep(2 ** a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dest', type=Path)
    ap.add_argument('--windings', nargs='+', required=True, help='e.g. w011 w108 w129, or "all"')
    ap.add_argument('--fit-from', type=Path, required=True, help='folder with wNNN_spliced/ (the unpacked release asset, README "Hosting")')
    ap.add_argument('--ckpt-from', type=Path, required=True, help='the S checkpoint file hecate_onsheet_S.pth from https://huggingface.co/jo-tud/hecate-onsheet')
    ap.add_argument('--frozen', action='store_true', help='also download the frozen hecate weights')
    a = ap.parse_args(argv)
    m = json.loads(MANIFEST.read_text())
    wind = sorted(m['fit']['windings']) if a.windings == ['all'] else a.windings
    dest = a.dest.resolve()
    rec = {}

    def check(key, p, want):
        got = sha(p)
        rec[key] = dict(want=want, got=got, ok=got == want)

    for w in wind:                                              # fit
        d = dest / 'fit' / f'{w}_spliced'
        d.mkdir(parents=True, exist_ok=True)
        for f, want in m['fit']['windings'][w].items():
            shutil.copyfile(a.fit_from / f'{w}_spliced' / f, d / f)
            check(f'fit/{w}_spliced/{f}', d / f, want)
    sys.path.insert(0, str(HERE))
    import segments_to_l2
    with tempfile.TemporaryDirectory(prefix='gpchain-seg-') as tmp:  # segments: public 2.4 um mesh -> level 2
        for sid, files in m['segments']['ids'].items():
            src = Path(tmp) / sid
            for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
                download(m['segments']['source'].replace('{meta.json,x.tif,y.tif,z.tif}', f).format(id=sid), src / f)
            segments_to_l2.main(src, dest / 'segments' / sid)
            for f, want in files.items():
                check(f'segments/{sid}/{f}', dest / 'segments' / sid / f, want)
    download(m['umbilicus']['url'], dest / 'umbilicus.json')
    check('umbilicus.json', dest / 'umbilicus.json', m['umbilicus']['sha256'])
    download(m['hecate']['url'], dest / 'models/hecate.py')
    check('models/hecate.py', dest / 'models/hecate.py', m['hecate']['sha256'])
    shutil.copyfile(a.ckpt_from, dest / 'models/S.pth')
    check('models/S.pth', dest / 'models/S.pth', m['ckpt']['S']['sha256'])
    if a.frozen:
        download(m['ckpt']['frozen']['url'], dest / 'models/hecate_9.6um.pth')
        check('models/hecate_9.6um.pth', dest / 'models/hecate_9.6um.pth', m['ckpt']['frozen']['sha256'])
    ok = all(r['ok'] for r in rec.values())
    (dest / 'inputs.json').write_text(json.dumps(dict(ok=ok, windings=wind, files=rec), indent=1) + '\n')
    bad = [k for k, r in rec.items() if not r['ok']]
    print(f'{len(rec)} files, {len(bad)} differ' + (f': {bad}' if bad else ''))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
