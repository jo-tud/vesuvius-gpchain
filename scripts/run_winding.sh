#!/bin/bash
# One fitted winding -> flattened, cut, CT-cut, rendered, ordered, frame-checked pieces + one S ink map per piece,
# on the host with no container: villa calls go through scripts/docker-shim.py to the builds in VCB.
#
#   bash scripts/run_winding.sh --fit FIT/wNNN_spliced --out OUT/wNNN --ct CT_ROOT --inputs INPUTS \
#        --venvs VENVS --vcb VCB [--device cuda|cpu] [--no-maps | --maps-only]
#
# INPUTS: scripts/get_inputs.py output (segments/, umbilicus.json, models/). CT_ROOT: scripts/fetch_ct.py output.
# Same chain flags as the stored runs: --segments (4 organiser meshes) --umbilicus --ct-cut --ct-cut-px. The render
# config is tools/gpchain/render_config.json with only render.image_id set to the VCB render build's id.
set -euo pipefail
TREE=$(cd "$(dirname "$0")/.." && pwd)
fit= out= ct= inputs= venvs= vcb= device=cuda maps=1 chain=1
die() { echo "run_winding.sh: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case $1 in
    --fit) fit=$2; shift 2 ;; --out) out=$2; shift 2 ;; --ct) ct=$2; shift 2 ;; --inputs) inputs=$2; shift 2 ;;
    --venvs) venvs=$2; shift 2 ;; --vcb) vcb=$2; shift 2 ;; --device) device=$2; shift 2 ;; --no-maps) maps=; shift ;; --maps-only) chain=; shift ;;
    *) die "unknown flag $1" ;;
  esac
done
for v in fit out ct inputs venvs vcb; do [ -n "${!v}" ] || die "need --$v"; done
if [ -n "$chain" ]; then [ ! -e "$out" ] || die "$out exists (the chain writes a new dir)"
else [ -f "$out/chain.json" ] || die "--maps-only: no $out/chain.json"; fi
for d in "$vcb/villa-ink_ves1471" "$vcb/inkcheck_ves572"; do [ -f "$d/IMAGE_ID" ] || die "no build in $d"; done
fit=$(realpath "$fit"); ct=$(realpath "$ct"); inputs=$(realpath "$inputs"); venvs=$(realpath "$venvs")
vcb=$(realpath "$vcb"); out=$(realpath -m "$out"); mkdir -p "$(dirname "$out")"
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
printf '#!/bin/sh\nexec "%s" -I "%s" "$@"\n' "$venvs/w2401/bin/python" "$TREE/scripts/docker-shim.py" > "$work/docker"
chmod 755 "$work/docker"
export PATH="$work:$PATH" GPCHAIN_VCB="$vcb" GPCHAIN_SHIM_LOG="$out.shim.jsonl" OMP_NUM_THREADS=8
"$venvs/w2401/bin/python" -I - "$TREE/tools/gpchain/render_config.json" "$vcb/inkcheck_ves572/IMAGE_ID" \
  "$out.render_config.json" <<'EOF'
import json, sys
src, idf, dst = sys.argv[1:]
c = json.load(open(src))
c['render']['image_id'] = open(idf).read().strip()[7:19]       # the 12 hex digits render_image_ok compares
json.dump(c, open(dst, 'w'), indent=1)
EOF
segs=(); for s in $(ls "$inputs/segments" | sort); do segs+=("$inputs/segments/$s"); done
t0=$(date +%s)
if [ -n "$chain" ]; then
  "$venvs/w2401/bin/python" -I "$TREE/tools/gpchain/chain.py" "$fit" "$out" --render "$ct" --segments "${segs[@]}" \
    --umbilicus "$inputs/umbilicus.json" --ct-cut --ct-cut-px --render-config "$out.render_config.json"
fi
t1=$(date +%s)
if [ -n "$maps" ]; then
  "$venvs/gpu/bin/python" -I "$TREE/tools/gpchain/render_contract.py" infer "$out" --ckpt "$inputs/models/S.pth" \
    --ckpt-sha "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ckpt"]["S"]["sha256"])' \
                 "$TREE/inputs/manifest.json")" --hecate "$inputs/models/hecate.py" --name S --device "$device"
fi
t2=$(date +%s)
echo "{\"chain_s\": $((t1 - t0)), \"maps_s\": $((t2 - t1))}" > "$out.wall${chain:+-chain}${maps:+-maps}.json"
echo "run_winding.sh: $out (chain $((t1 - t0)) s, maps $((t2 - t1)) s)"
