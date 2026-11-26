#!/usr/bin/env -S python3 -I
"""`docker` for tools/gpchain without Docker (inside the gpchain image, or on a host: GPCHAIN_VCB). tools/gpchain calls villa tools as `docker run ... IMAGE ...` and
checks `docker image inspect IMAGE` ids. In this image those villa builds are already present, byte-copied from the
named images with their own libraries (docker/gpchain/bundle.sh -> /opt/vcb/<image>/), so this shim runs them
directly and tools/gpchain stays unchanged. Outside the image, GPCHAIN_VCB names the folder that holds
<image name with : and / as _>/{bin,IMAGE_ID} (scripts/build_villa.sh writes it); default /opt/vcb.

Supported, nothing else (anything else exits 125 and says why, so a new call form fails loudly):
  docker image inspect -f|--format '{{.Id}}' IMAGE      -> the source image id recorded at build time
  docker run --rm [--memory M] [--cpus N] [--user U] [--network none] [-e K=V]... [-v HOST:CTR[:ro]]...
             --entrypoint ENTRY IMAGE ARGS...
      ENTRY sh (with -c SCRIPT) or a villa binary name / path; container paths CTR in ARGS and SCRIPT are mapped
      back to HOST (path tokens only); PATH starts with /opt/vcb/<image>/bin. --memory / --cpus are not
      enforced here: the outer `docker run --memory` of this image is the limit. --user must be the current user.
Every call is appended to $GPCHAIN_SHIM_LOG when set.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

VCB = Path(os.environ.get('GPCHAIN_VCB', '/opt/vcb'))


def die(msg):
    sys.stderr.write(f'gpchain docker shim: {msg}\n')
    sys.exit(125)


def bundle(image):
    d = VCB / image.replace(':', '_').replace('/', '_')
    if not (d / 'IMAGE_ID').is_file():
        die(f'image {image!r} is not in this image (have: {sorted(p.name for p in VCB.iterdir())})')
    return d


def log(rec):
    f = os.environ.get('GPCHAIN_SHIM_LOG')
    if f:
        with open(f, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')


def inspect(args):
    fmt, rest = None, []
    it = iter(args)
    for x in it:
        if x in ('-f', '--format'):
            fmt = next(it)
        elif x.startswith('--format='):
            fmt = x.split('=', 1)[1]
        else:
            rest.append(x)
    if fmt != '{{.Id}}' or len(rest) != 1:
        die(f'image inspect: only -f {{{{.Id}}}} IMAGE is supported, got {args}')
    print((bundle(rest[0]) / 'IMAGE_ID').read_text().strip())
    log({'call': 'inspect', 'image': rest[0]})
    return 0


def mapper(mounts):
    """Rewrite container path tokens to host paths: CTR alone, CTR/..., KEY=CTR..., inside a shell script too."""
    pairs = sorted(((c.rstrip('/'), h.rstrip('/')) for h, c in mounts if h.rstrip('/') != c.rstrip('/')),
                   key=lambda p: -len(p[0]))
    if not pairs:
        return lambda s: s
    rx = re.compile(r'(?<![\w./-])(' + '|'.join(re.escape(c) for c, _ in pairs) + r')(?=$|[/\s;=:\'"])')
    m = dict(pairs)
    return lambda s: rx.sub(lambda g: m[g.group(1)], s)


def run(args):
    env, mounts, entry, i = {}, [], None, 0
    flags_ignored = {}
    while i < len(args):
        x = args[i]
        if x == '--rm':
            i += 1
        elif x in ('--memory', '--cpus', '--network', '--user', '-e', '-v', '--entrypoint'):
            if i + 1 >= len(args):
                die(f'{x} without a value')
            v = args[i + 1]
            if x == '-e':
                k, _, val = v.partition('=')
                env[k] = val
            elif x == '-v':
                parts = v.split(':')
                if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] not in ('ro', 'rw')):
                    die(f'-v {v}: only HOST:CTR[:ro|rw]')
                mounts.append((parts[0], parts[1]))
            elif x == '--entrypoint':
                entry = v
            elif x == '--network' and v != 'none':
                die('--network: only none')
            elif x == '--user' and v != f'{os.getuid()}:{os.getgid()}':
                die(f'--user {v} is not the current user {os.getuid()}:{os.getgid()}')
            else:
                flags_ignored[x] = v
            i += 2
        elif x.startswith('-'):
            die(f'run: unsupported flag {x}')
        else:
            break
    if entry is None or i >= len(args):
        die(f'run: need --entrypoint and an image: {args}')
    image, rest = args[i], args[i + 1:]
    b = bundle(image)
    mp = mapper(mounts)
    e = dict(os.environ, **env)
    e['PATH'] = f'{b / "bin"}:{e.get("PATH", "/usr/bin:/bin")}'
    if entry == 'sh':
        if len(rest) != 2 or rest[0] != '-c':
            die(f'run --entrypoint sh: only -c SCRIPT, got {rest}')
        cmd = ['/bin/sh', '-c', mp(rest[1])]
    else:
        exe = b / 'bin' / Path(entry).name
        if not exe.is_file():
            die(f'{entry}: not bundled from {image}')
        cmd = [str(exe), *map(mp, rest)]
    log({'call': 'run', 'image': image, 'entry': entry, 'cmd': cmd, 'mounts': mounts, 'env': env,
         'not_enforced': flags_ignored})
    return subprocess.run(cmd, env=e, cwd='/').returncode


def main(argv):
    if argv[:2] == ['image', 'inspect']:
        return inspect(argv[2:])
    if argv[:1] == ['run']:
        return run(argv[1:])
    die(f'unsupported command: docker {" ".join(argv[:3])}')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
