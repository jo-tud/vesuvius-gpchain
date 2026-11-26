#!/usr/bin/env python3
"""Per-winding coverage of chain outputs, the rule behind "120 of 120 windings" (stored run F-626):

a winding is covered when the chain made at least one piece and every piece (a) passes the frame check, (b) has a
layer order ("villa" or "reversed"; a piece whose order could not be decided is refused) and (c) has a map of each
named model. A winding with no chain.json is not covered.

  python coverage.py WINDING_OUT [WINDING_OUT ...] [--maps S] [--out coverage.json]
"""
import argparse
import json
from pathlib import Path


def winding(d, maps):
    f = d / 'chain.json'
    if not f.exists():
        return dict(ok=False, why=['no chain.json'], pieces={})
    rend = json.loads(f.read_text()).get('render') or {}
    pieces = {name: dict(fc_ok=r['frame']['ok'], order=r['order']['order'], reference=r['order']['reference'],
                         maps={n: (d / 'maps' / f'{n}_{name}.npy').exists() for n in maps})
              for name, r in sorted(rend.items())}
    why = [] if pieces else ['chain.json has no render pieces']
    for name, p in pieces.items():
        if not p['fc_ok']:
            why.append(f'{name}: frame check not ok')
        if p['order'] not in ('villa', 'reversed'):
            why.append(f'{name}: no layer order')
        why += [f'{name}: no {n} map' for n, ok in p['maps'].items() if not ok]
    return dict(ok=not why, why=why, pieces=pieces)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('windings', type=Path, nargs='+')
    ap.add_argument('--maps', nargs='+', default=['S'])
    ap.add_argument('--out', type=Path)
    a = ap.parse_args(argv)
    res = {d.name: winding(d, a.maps) for d in a.windings}
    n = sum(r['ok'] for r in res.values())
    out = dict(n_covered=n, n=len(res), share=n / len(res), windings=res)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(f'{n} of {len(res)} windings covered')
    for k, r in res.items():
        if not r['ok']:
            print(' ', k, '; '.join(r['why']))


if __name__ == '__main__':
    main()
