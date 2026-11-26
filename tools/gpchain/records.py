#!/usr/bin/env python3
"""GP package fields read from machine records, never typed in (VES-3629, task E of VES-3584).

A field is filled only when a record names it. Records are matched by sha256, never by path or name:
  - the chain output (CHAIN/images.json: column -> wNNN/piece; wNNN/maps/<model>.json: ckpt_sha256, hecate_sha256;
    wNNN/maps/<model>_<piece>.npy; wNNN/chain.json: render_config_sha256, render.frame);
  - the provenance store (tools/provenance/prov.py: STORE/MANIFEST.jsonl kept rounds, STORE/runs/*/run.json,
    STORE/blobs/sha256/..);
  - the git history of this repo (a config file version whose sha256 equals the one the chain recorded);
  - image.json, written by docker/gpchain/package.sh from `docker image inspect` (tag, id, revision label, argv).
Each field is {"value": text or None, "source": [record paths / ids], "why": reason when value is None}.
A None value is written MISSING(name) in the package text, and the build reports it.

Fields from records (34_prizes.md lines in docs: tools/gpchain/CHECKLIST.md): model_ref, window_px, window_mm,
seeds, training_run, inference_runs, kept_rounds, docker_image, requirements, repro_cmd. No record exists for
heldout_rows, letter_mm or window_note: they stay MISSING with the reason (they need a real result).
"""
import collections
import hashlib
import json
import pickle
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NO_RECORD = {
    'heldout_rows': 'needs a real result: a held-out set with public ground truth on the submitted scroll',
    'letter_mm': 'needs a real result: letter boxes from a person or labels (package.py letters)',
    'window_note': 'needs a real result: window vs letter size and held-out text (34_prizes.md :152)',
}


def sha256(p, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(bufsize):
            h.update(b)
    return h.hexdigest()


def field(value=None, source=(), why=None):
    assert (value is None) == (why is not None), (value, why)
    return {'value': value, 'source': list(source), 'why': why}


# ------------------------------------------------------------------------------------------- record readers
def chain_columns(chain):
    """CHAIN/images.json -> {column_NN: (winding dir, piece name)} in column order."""
    cols = json.loads((Path(chain) / 'images.json').read_text())['columns']
    return {c: (Path(chain) / cols[c].split('/')[0], cols[c].split('/')[1]) for c in sorted(cols)}


def store_rounds(store):
    f = Path(store) / 'MANIFEST.jsonl'
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.is_file() else []


def store_runs(store):
    d = Path(store) / 'runs'
    return [json.loads(p.read_text()) for p in sorted(d.glob('*/run.json'))] if d.is_dir() else []


def blob(store, h):
    return Path(store) / 'blobs' / 'sha256' / h[:2] / h


def out_shas(run):
    return {o.get('sha256') for o in run.get('outputs', []) if o.get('sha256')}


class _NoTorch(pickle.Unpickler):
    """reads the plain-python part of a torch zip checkpoint (config, ints, strings) without torch: tensors and
    storages become None. Only the classes torch writes for tensors are accepted; anything else raises."""
    OK = {('torch._utils', '_rebuild_tensor_v2'), ('torch._utils', '_rebuild_parameter'),
          ('torch._utils', '_rebuild_tensor')}

    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if (module, name) in self.OK or (module == 'torch' and name.endswith('Storage')):
            return lambda *a, **k: None
        raise pickle.UnpicklingError(f'records: {module}.{name} is not allowed in a checkpoint payload')

    def persistent_load(self, pid):
        return None


def checkpoint_payload(path):
    with zipfile.ZipFile(path) as z:
        pk = [n for n in z.namelist() if n.endswith('/data.pkl') or n == 'data.pkl']
        assert len(pk) == 1, f'records: {path} is not a torch zip checkpoint ({pk})'
        with z.open(pk[0]) as f:
            return _NoTorch(f).load()


def git_version_with_sha(rel, want):
    """the newest commit whose copy of REPO/rel has sha256 `want` (None if no version does)."""
    log = subprocess.run(['git', '-C', str(REPO), 'log', '--format=%H', '--', rel], text=True,
                         capture_output=True).stdout.split()
    for c in log:
        b = subprocess.run(['git', '-C', str(REPO), 'show', f'{c}:{rel}'], capture_output=True).stdout
        if hashlib.sha256(b).hexdigest() == want:
            return c, json.loads(b)
    return None, None


def git_head():
    return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True, capture_output=True).stdout.strip()


# ------------------------------------------------------------------------------------------------- fields
def model_fields(chain, store, model='S', um_per_px=None):
    """model_ref, window_px, window_mm, seeds, training_run, inference_runs, kept_rounds."""
    cols = chain_columns(chain)
    logs = {c: json.loads((w / 'maps' / f'{model}.json').read_text()) for c, (w, _) in cols.items()}
    F = {}
    shas = {lg.get('ckpt_sha256') for lg in logs.values()}
    rounds = store_rounds(store)
    runs = store_runs(store)
    if len(shas) != 1 or None in shas:
        why = f'maps/{model}.json of the {len(cols)} columns name {len(shas)} checkpoints'
        for k in ('model_ref', 'window_px', 'window_mm', 'seeds', 'training_run', 'kept_rounds'):
            F[k] = field(why=why)
    else:
        ck = shas.pop()
        hec = {lg.get('hecate_sha256') for lg in logs.values()}
        src_logs = [f'{w.name}/maps/{model}.json' for w, _ in cols.values()]
        kept = [(r, f) for r in rounds for f in r['files'] if f['sha256'] == ck and f['role'] == 'checkpoint']
        b = blob(store, ck)
        if kept and b.is_file() and sha256(b) == ck:
            r, f = kept[0]
            F['model_ref'] = field(
                f"{model} = {Path(f['name']).name} sha256 {ck} (kept round {r['round']} of run {r['run_id']}); "
                f"code hecate.py sha256 {hec.pop() if len(hec) == 1 else 'MIXED'}",
                [*src_logs, f"kept/MANIFEST.jsonl {r['run_id']}/{r['round']}", f'kept/blobs/sha256/{ck[:2]}/{ck}'])
            pay = checkpoint_payload(b)
            ps = (pay.get('config') or {}).get('patch_size')
            if ps and len(ps) == 3:
                F['window_px'] = field(f'{ps[1]} x {ps[2]} px x {ps[0]} slices (patch_size {ps}, '
                                       f"sampling {pay.get('sampling_um')} um)",
                                       [f'kept checkpoint {ck[:12]} payload config.patch_size'])
                F['window_mm'] = (field(f'{ps[1] * um_per_px / 1000:.2f} x {ps[2] * um_per_px / 1000:.2f} mm '
                                        f'({ps[1]} px x {um_per_px:.2f} um/px, the package images\' mean)',
                                        ['window_px', 'package.json images um_per_px'])
                                  if um_per_px else field(why='no image px size measured'))
            else:
                F['window_px'] = F['window_mm'] = field(why=f'checkpoint {ck[:12]} payload has no config.patch_size')
            tr = [g for g in r['files'] if g['role'] == 'train_log' and Path(g['name']).name == 'train.json']
            tseed = json.loads(blob(store, tr[0]['sha256']).read_text()).get('seed') if tr else None
            pseed = (pay.get('ves3090') or {}).get('seed') if isinstance(pay.get('ves3090'), dict) else None
            F['_train_seed'] = (tseed, pseed)
            rr = r['files']
            roles = collections.Counter(g['role'] for g in rr)
            F['kept_rounds'] = field(
                f"round {r['round']} of run {r['run_id']}: {len(rr)} files "
                f"({', '.join(f'{k} x{v}' for k, v in sorted(roles.items()))}), {r['bytes'] / 2**30:.2f} GiB, "
                f"licence at submission {r.get('licence_at_submission')}",
                [f"kept/MANIFEST.jsonl {r['run_id']}/{r['round']}"])
        else:
            why = (f'checkpoint sha256 {ck[:12]} is in no kept round of {store}' if not kept
                   else f'kept blob of {ck[:12]} missing or changed')
            for k in ('model_ref', 'window_px', 'window_mm', 'kept_rounds'):
                F[k] = field(why=why)
            F['_train_seed'] = (None, None)
        trn = [x for x in runs if x['kind'] == 'train' and x['status'] == 'done' and ck in out_shas(x)]
        F['training_run'] = (field(', '.join(x['run_id'] for x in trn), [f"kept/runs/{x['run_id']}/run.json" for x in trn])
                             if trn else field(why=f'no done training run in {store} has checkpoint {ck[:12]} '
                                                   f'as an output (trained before tools/provenance; kept round only)'))
    # inference: a done infer run whose outputs hold this column's map bytes, for every model map of the column
    cover, src = {}, []
    for c, (w, piece) in cols.items():
        maps = sorted((w / 'maps').glob(f'*_{piece}.npy'))
        need = {sha256(m) for m in maps}
        hit = [x for x in runs if x['kind'] == 'infer' and x['status'] == 'done' and need and need <= out_shas(x)]
        if hit:
            cover[c] = hit[0]['run_id']
            src.append(f"kept/runs/{hit[0]['run_id']}/run.json")
    n = len(cols)
    F['inference_runs'] = (field('; '.join(f'{c}: {r}' for c, r in cover.items()), sorted(set(src)))
                           if len(cover) == n else
                           field(why=f'{len(cover)}/{n} columns have a done inference run whose outputs hold their '
                                     f'map bytes ({", ".join(f"{c} = {r}" for c, r in cover.items()) or "none"})'))
    # seeds: training (kept train.json and the checkpoint's own payload) + inference (render_config.json version
    # whose sha256 every column's chain.json recorded)
    rc = {json.loads((w / 'chain.json').read_text()).get('render_config_sha256') for w, _ in cols.values()}
    commit, cfg = git_version_with_sha('tools/gpchain/render_config.json', rc.pop()) if len(rc) == 1 else (None, None)
    tseed, pseed = F.pop('_train_seed', (None, None))
    if 'seeds' in F:
        pass
    elif tseed is None or (pseed is not None and pseed != tseed):
        F['seeds'] = field(why=f'training seed not recorded or inconsistent (train.json {tseed}, checkpoint {pseed})')
    elif cfg is None:
        F['seeds'] = field(why='the chain.json render_config_sha256 matches no version of tools/gpchain/render_config.json')
    else:
        F['seeds'] = field(f"training: seed {tseed} (kept train.json; checkpoint payload {pseed}); inference: "
                           f"layer-order draws seed {cfg['layer_order']['seed']}, frame-check draws seed "
                           f"{cfg['frame']['seed']} (render_config.json at {commit[:8]})",
                           ['kept train.json', 'checkpoint payload ves3090.seed',
                            f'tools/gpchain/render_config.json@{commit[:8]}', '*/chain.json render_config_sha256'])
    return F


def image_fields(image_json):
    """docker_image, requirements, repro_cmd from image.json (docker/gpchain/package.sh)."""
    F = {}
    p = Path(image_json) if image_json else None
    if not (p and p.is_file()):
        why = 'no image.json (the build did not run through docker/gpchain/package.sh)'
        return {k: field(why=why) for k in ('docker_image', 'requirements', 'repro_cmd')}
    m = json.loads(p.read_text())
    rev = (m.get('labels') or {}).get('org.opencontainers.image.revision')
    known = rev and subprocess.run(['git', '-C', str(REPO), 'cat-file', '-e', f'{rev}^{{commit}}'],
                                   capture_output=True).returncode == 0
    F['docker_image'] = (field(f"{m['tag']} (id {m['id']}, built from commit {rev[:8]}); package code: "
                               f"commit {m.get('code_commit', '?')[:8]} mounted read-only", [str(p.name)])
                         if m.get('id') and known else
                         field(why=f"image.json has no id or its revision {rev} is not a commit of this repo"))
    readme = REPO / 'docker/gpchain/README.md'
    txt = readme.read_text() if readme.is_file() else ''
    F['requirements'] = (field(f'docker/gpchain/README.md "System requirements" (sha256 {sha256(readme)[:16]}); '
                               f'venv pins: /opt/venvs/*.freeze.txt in the image', ['docker/gpchain/README.md'])
                         if '## System requirements' in txt else
                         field(why='docker/gpchain/README.md has no "System requirements" section'))
    argv = m.get('argv')
    F['repro_cmd'] = (field(' '.join(argv), [str(p.name)]) if argv else field(why='image.json has no argv'))
    return F


def band_fields(results, chain, um_per_px, scroll, scan, commit):
    """the score fields of false_positives.md from a band.py results.json (F-585 schema) and the chain's chain.json
    files; moved here from experiments/2026-11-08-ves3585-gp-package/run.py (F-598), values unchanged."""
    results = Path(results)
    r = json.loads(results.read_text())
    s = r['score']
    try:
        src = str(results.relative_to(REPO))
    except ValueError:
        src = str(results)
    fc = {}
    for c, w in r['P1']['columns'].items():
        ch = json.loads((Path(chain) / w.split('/')[0] / 'chain.json').read_text())
        fc[c] = ch['render'][w.split('/')[1]]['frame']
    rs = [f['r'] for f in fc.values()]
    shifts = s['N_shift']['auc']
    d = results.parent.name
    return {
        'scroll': scroll, 'scan': scan, 'region': r['arena'], 'commit': commit,
        'pc_summary': 'step-1 PC, F-576 PC2, checkpoint sha, PC-G: ' +
                      ', '.join(f"{k} {'pass' if v else 'FAIL'}" for k, v in r['controls'].items()
                                if k in ('PC_step1', 'PC2_f576', 'checkpoints', 'PC_G')),
        'pc_source': f'{src} (controls), {d}/pc.json',
        # SHIFTS (0, ±400), (±300, 0) map px (F-576 run.py:70); map px = image px, size measured by package.py
        'shift_um': f"{400 * um_per_px:.0f} along columns / {300 * um_per_px:.0f} along rows (400 / 300 map px)",
        'auc': round(s['auc']['S'], 4), 'shift_max': round(max(shifts.values()), 4),
        'perm_mean': round(s['N_perm']['mean'], 4), 'perm_max': round(s['N_perm']['max'], 4),
        'score_source': f'{src} (score)', 'model': 'S', 'baseline_auc': round(s['auc']['frozen'], 4),
        'fc_min': round(min(rs), 4), 'fc_max': round(max(rs), 4), 'fc_bar': fc['column_01']['bar'],
        'chain_source': 'CHAIN/wNNN/chain.json (render.frame)',
        # prose from F-598 (the F-585 README "Arms"); a record-backed overlap check is CHECKLIST.md item :124
        'overlap_note': ('Scored only on the 20260918 labels of the 4 HELD segments, which S never trained on; '
                         'training-segment labels (w035-w041) are not scored. These labels were read by earlier '
                         'tasks (F-555, F-561, F-576), so they are not a held-out set.'),
        'overlap_source': f'{d}/README.md ("Arms"), preflight_overlap.json',
        'human_minutes': r['ledger']['minutes'], 'ledger_source': f'{src} (ledger)',
    }


def all_fields(chain, store, image_json, um_per_px, model='S'):
    """every record field + the no-record fields, each {"value", "source", "why"}."""
    F = model_fields(chain, store, model, um_per_px)
    F.update(image_fields(image_json))
    F.update({k: field(why=w) for k, w in NO_RECORD.items()})
    return F


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('chain', type=Path)
    ap.add_argument('--store', type=Path, default=Path('kept'))
    ap.add_argument('--image-json', type=Path)
    ap.add_argument('--um-per-px', type=float)
    a = ap.parse_args()
    print(json.dumps(all_fields(a.chain, a.store, a.image_json, a.um_per_px), indent=1))
