#!/bin/sh
# Copy villa binaries out of their own image with every shared library they link (ldd closure, the image's own
# glibc and loader included), so one gpchain image can carry two villa builds on different OS bases unchanged.
#   sh bundle.sh DEST BIN [BIN ...]        (run inside the source image; BIN = a name on PATH or an absolute path)
# DEST/real/<bin> are the original bytes (SHA256SUMS lists them); DEST/bin/<bin> is a wrapper that starts the
# binary with DEST/lib/ld-linux-x86-64.so.2 --inhibit-cache --library-path DEST/lib, so nothing from the host image's
# /lib or ld.so.cache is mixed in. Fails if a library is missing ("not found" in ldd).
set -eu
D=$1; shift
mkdir -p "$D/real" "$D/lib" "$D/bin"
for b in "$@"; do
  p=$(command -v "$b" || true); [ -n "$p" ] || { echo "bundle.sh: $b not found" >&2; exit 1; }
  if ldd "$p" | grep -q 'not found'; then ldd "$p" >&2; exit 1; fi
  cp -L "$p" "$D/real/"
  ldd "$p" | awk '$2 == "=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}' | while read -r l; do
    [ -e "$D/lib/$(basename "$l")" ] || cp -L "$l" "$D/lib/"
  done
  n=$(basename "$p")
  cat > "$D/bin/$n" <<W
#!/bin/sh
H=\$(dirname "\$(dirname "\$(readlink -f "\$0")")")
exec "\$H/lib/ld-linux-x86-64.so.2" --inhibit-cache --library-path "\$H/lib" "\$H/real/$n" "\$@"
W
  chmod 755 "$D/bin/$n"
done
cd "$D" && sha256sum real/* > SHA256SUMS && ls lib | wc -l > NLIBS
echo "bundle.sh: $(ls real | wc -l) binaries, $(cat NLIBS) libs, $(du -sm . | cut -f1) MB in $D"
