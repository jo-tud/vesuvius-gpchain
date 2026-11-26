#!/bin/bash
# Build the two Python environments gpchain runs in, from the hash-locked files in env/ (the same files and uv
# commands as the stored runs' image). Needs uv 0.11.16 and network (PyPI, download.pytorch.org, GitHub).
#
#   bash scripts/make_venvs.sh VENVS [--cpu]
#
#   VENVS/w2401  CPython 3.14.5, env/requirements-w2401.txt + villa PR #2401 head ea48ce6b `vesuvius` (editable):
#                chain.py, the GP check (vesuvius.gp_submission), make_folder.py           ~0.8 GB
#   VENVS/gpu    CPython 3.12.13, env/requirements-gpu.txt (torch cu130) + villa 47111d1a `vesuvius`:
#                render_contract.py infer (the ink maps)                                    ~5 GB
#   VENVS/cpu    (--cpu) the same with torch cpu: infer on CPU (slow), package.py --orient  ~1.3 GB
# The Python builds go to VENVS/python, the villa sources to VENVS/src. VENVS/*.freeze.txt list what is installed.
set -euo pipefail
TREE=$(cd "$(dirname "$0")/.." && pwd)
V=${1:?usage: make_venvs.sh VENVS [--cpu]}; shift
kinds="gpu"; [ "${1:-}" = --cpu ] && kinds="cpu gpu"
VILLA_COMMIT=47111d1a31381a5ff2c05046ed3e311b0dbeec64
VILLA_LOCK_SHA=753447a32a0567d54c7fdcc38f37ca83146d0a82eee4720a6184ba6f782c7273
PR_HEAD=ea48ce6ba32cbb854c292f42bc2ad3286fafc1b4                # villa PR #2401 head (closed 2026-11-24, not merged; fetched by hash)
[ "$(uv --version | cut -d' ' -f2)" = 0.11.16 ] || { echo "need uv 0.11.16 (pip install uv==0.11.16)" >&2; exit 2; }
[ ! -e "$V/w2401" ] || { echo "$V/w2401 exists" >&2; exit 2; }
mkdir -p "$V"; V=$(realpath "$V")
export UV_PYTHON_INSTALL_DIR="$V/python" UV_PYTHON_PREFERENCE=only-managed UV_NO_CACHE=1 \
       UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=1 UV_HTTP_RETRIES=8 UV_HTTP_TIMEOUT=120
fetch() {  # fetch COMMIT DEST: villa's vesuvius/ package at COMMIT from GitHub
  local g; g=$(mktemp -d)
  git init -q "$g" && git -C "$g" remote add origin https://github.com/ScrollPrize/villa.git
  git -C "$g" fetch -q --depth 1 --filter=blob:none origin "$1"
  [ "$(git -C "$g" rev-parse FETCH_HEAD)" = "$1" ] || { echo "fetched villa is not $1" >&2; exit 2; }
  mkdir -p "$2" && git -C "$g" archive "$1" vesuvius | tar -x -C "$2"
  rm -rf "$g"
}
fetch "$VILLA_COMMIT" "$V/src/villa-47111d1a"
echo "$VILLA_LOCK_SHA  $V/src/villa-47111d1a/vesuvius/uv.lock" | sha256sum -c --quiet
fetch "$PR_HEAD" "$V/src/villa-pr2401"
uv python install -q 3.12.13 3.14.5
for t in $kinds; do
  uv venv -q --python 3.12.13 "$V/$t"
  uv pip sync -q --python "$V/$t/bin/python" --require-hashes --torch-backend "$([ $t = cpu ] && echo cpu || echo cu130)" \
    "$TREE/env/requirements-$t.txt"
  uvx -q --from pip==25.2 pip --python "$V/$t/bin/python" install -q \
    --no-deps --no-index --no-build-isolation --ignore-requires-python "$V/src/villa-47111d1a/vesuvius"
done
uv venv -q --python 3.14.5 "$V/w2401"
uv pip sync -q --python "$V/w2401/bin/python" --require-hashes "$TREE/env/requirements-w2401.txt"
uv pip install -q --python "$V/w2401/bin/python" --no-deps -e "$V/src/villa-pr2401/vesuvius"
for t in $kinds w2401; do uv pip freeze --python "$V/$t/bin/python" > "$V/$t.freeze.txt"; done
du -sh "$V"/*
