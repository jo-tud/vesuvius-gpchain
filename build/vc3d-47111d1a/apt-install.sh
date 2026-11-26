#!/bin/sh
# apt-get install from the Debian snapshot matching the base image's date, so the packages
# (compiler included) are the same on every rebuild, not whatever the current point release has.
#
# Usage (in a Dockerfile RUN): DEBIAN_SNAPSHOT=20261005T000000Z sh docker/apt-install.sh PKG...
#
# Downloads APT_PARALLEL (default 12) .debs at a time before installing. From our build host, one
# connection to snapshot.debian.org (or any Debian mirror tried) gets 50-200 kB/s, so apt's one
# connection per host took ~2.5 h for the 493 MB the vc3d stage needs; parallel connections add
# up. apt-get install then checks every file against the snapshot's hashes and fetches anything
# missing or wrong itself, so this only changes speed, never what gets installed.
set -eu
: "${DEBIAN_SNAPSHOT:?set DEBIAN_SNAPSHOT, e.g. 20261005T000000Z}"
src=/etc/apt/sources.list.d/debian.sources
sed -i -E "s#^URIs: http://deb.debian.org/(debian(-security)?)\$#URIs: http://snapshot.debian.org/archive/\\1/${DEBIAN_SNAPSHOT}#" "$src"
grep -q "snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" "$src" \
  || { echo "apt-install.sh: could not point $src at the snapshot" >&2; exit 1; }
# Snapshot Release files are past their Valid-Until date by design.
apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Retries=5 update

# Pre-fetch: `apt-get download name=version` for every package the install would fetch, in
# parallel. Failures are ignored here; the install below retries them one at a time.
mkdir -p /var/cache/apt/archives/partial
cd /var/cache/apt/archives
apt-get install -y --no-install-recommends --print-uris "$@" \
  | sed -n "s/^'[^']*' \([^ ]*\.deb\) .*/\1/p" \
  | sed -e 's/%3a/:/g' -e 's/^\([^_]*\)_\(.*\)_[^_]*\.deb$/\1=\2/' \
  | xargs -r -n 1 -P "${APT_PARALLEL:-12}" apt-get -q -o Acquire::Retries=5 download >/dev/null 2>&1 \
  || true
echo "apt-install.sh: pre-fetched $(ls *.deb 2>/dev/null | wc -l) .debs, $(du -sm . | cut -f1) MB"
cd /

apt-get -o Acquire::Retries=5 install -y --no-install-recommends "$@"
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb
