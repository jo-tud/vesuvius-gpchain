#!/bin/bash
# Build the four villa tools gpchain calls, from villa source at the pinned commits, and lay them out for
# scripts/docker-shim.py, so the chain then runs with no container at all.
#
#   bash scripts/build_villa.sh VCB [--jobs 8] [--no-cache] [--tag-prefix gpchain-villa]
#
# Needs Docker (build only) and network (GitHub, the Ubuntu and Debian snapshot archives). CPU only; about 1 h cold.
#   villa 6bbe6e21 -> vc_tifxyz2obj, flatboi, vc_obj2tifxyz   (build/villa-6bbe6e21, Ubuntu 26.04)
#   villa 47111d1a -> vc_render_tifxyz                         (build/vc3d-47111d1a, Debian 13)
# Writes VCB/villa-ink_ves1471/ and VCB/inkcheck_ves572/ (bin/ wrappers, real/ binaries, lib/ with every shared
# library they link, SHA256SUMS, IMAGE_ID = the id of the new build) and VCB/builds.json. The two folder names are
# the image names the chain asks for (isometric_step.py default, render_config.json); they hold the new builds.
# If you build villa yourself without Docker, lay the same folders out by hand (bin/<tool> runnable, IMAGE_ID).
set -euo pipefail
TREE=$(cd "$(dirname "$0")/.." && pwd)
V6=6bbe6e211c56079c763440105b299463a0c8e6d6
V47=47111d1a31381a5ff2c05046ed3e311b0dbeec64
vcb= jobs=8 nocache= prefix=gpchain-villa
die() { echo "build_villa.sh: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case $1 in
    --jobs) jobs=$2; shift 2 ;; --no-cache) nocache=--no-cache; shift ;; --tag-prefix) prefix=$2; shift 2 ;;
    -*) die "unknown flag $1" ;; *) [ -z "$vcb" ] || die "one VCB dir"; vcb=$1; shift ;;
  esac
done
[ -n "$vcb" ] || die "usage: build_villa.sh VCB [--jobs N] [--no-cache]"
[ ! -e "$vcb/villa-ink_ves1471" ] && [ ! -e "$vcb/inkcheck_ves572" ] || die "$vcb already holds a build"
mkdir -p "$vcb"; vcb=$(realpath "$vcb")
T6=$prefix:6bbe6e21 T47=$prefix:47111d1a
ctx=$(mktemp -d); trap 'rm -rf "$ctx"' EXIT
t0=$(date +%s)

# villa 6bbe6e21: volume-cartographer only, from GitHub, checked by hash
git init -q "$ctx/g" && git -C "$ctx/g" remote add origin https://github.com/ScrollPrize/villa.git
git -C "$ctx/g" fetch -q --depth 1 --filter=blob:none origin "$V6"
[ "$(git -C "$ctx/g" rev-parse FETCH_HEAD)" = "$V6" ] || die "fetched villa is not $V6"
mkdir -p "$ctx/c6/villa"
git -C "$ctx/g" archive "$V6" volume-cartographer | tar -x -C "$ctx/c6/villa"
echo "$V6" > "$ctx/c6/villa/COMMIT"
cp "$TREE/build/villa-6bbe6e21/Dockerfile" "$ctx/c6/"
docker build $nocache --progress=plain --build-arg JOBS="$jobs" --label villa_commit="$V6" -t "$T6" "$ctx/c6"
t1=$(date +%s)
# villa 47111d1a: the recipe fetches villa itself (sparse, checked by hash)
docker build $nocache --progress=plain --target vc3d --label villa_commit="$V47" -t "$T47" "$TREE/build/vc3d-47111d1a"
t2=$(date +%s)

bundle() {  # bundle TAG NAME BIN...
  local tag=$1 name=$2; shift 2
  docker run --rm --network none --user "$(id -u):$(id -g)" -v "$TREE/scripts/bundle.sh:/bundle.sh:ro" \
    -v "$vcb:/vcb" --entrypoint /bin/sh "$tag" /bundle.sh "/vcb/$name" "$@"
  docker image inspect -f '{{.Id}}' "$tag" > "$vcb/$name/IMAGE_ID"
}
bundle "$T6" villa-ink_ves1471 vc_tifxyz2obj flatboi vc_obj2tifxyz
bundle "$T47" inkcheck_ves572 /opt/vc3d/bin/vc_render_tifxyz
python3 - "$vcb" "$T6" "$T47" "$V6" "$V47" "$((t1 - t0))" "$((t2 - t1))" <<'EOF'
import json, sys
from pathlib import Path
vcb, t6, t47, v6, v47, s6, s47 = sys.argv[1:]
rec = {}
for name, tag, commit, s in (('villa-ink_ves1471', t6, v6, s6), ('inkcheck_ves572', t47, v47, s47)):
    d = Path(vcb) / name
    sums = dict(reversed(l.split()) for l in (d / 'SHA256SUMS').read_text().splitlines())
    rec[name] = dict(tag=tag, villa_commit=commit, image_id=(d / 'IMAGE_ID').read_text().strip(),
                     build_s=int(s), sha256=sums, n_libs=int((d / 'NLIBS').read_text()))
(Path(vcb) / 'builds.json').write_text(json.dumps(rec, indent=1) + '\n')
print(json.dumps(rec, indent=1))
EOF
