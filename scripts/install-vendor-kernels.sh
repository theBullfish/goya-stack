#!/bin/bash
# Install Intel's TPC kernel library (6,799 kernels) from Intel's own server.
# This repo does not redistribute Intel's binaries; the script downloads habanalabs-graph 1.1.1-94,
# checks its sha256, and installs only libtpc_kernels.so.1 from it.
#   -> /usr/local/lib/goya-stack/vendor/libtpc_kernels.so.1
set -euo pipefail
URL=https://vault.habana.ai/artifactory/debian/focal/pool/main/h/habanagraph/habanalabs-graph-1.1.1-94_amd64.deb
SHA256=75f78d5fc9adb5e07553042b7644dcfc191d31cb68f7887eb7b82208a9c13075
LIB_MD5=444d3b4bfc487359212791acdb6c20f1
DEST=/usr/local/lib/goya-stack/vendor

[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
if [ -f "$DEST/libtpc_kernels.so.1" ] && echo "$LIB_MD5  $DEST/libtpc_kernels.so.1" | md5sum -c --status -; then
  echo "libtpc_kernels.so.1 is already installed in $DEST."; exit 0
fi
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
echo "Downloading habanalabs-graph 1.1.1-94 (192 MB) from vault.habana.ai ..."
curl -fL --retry 3 -o "$tmp/graph.deb" "$URL"
echo "$SHA256  $tmp/graph.deb" | sha256sum -c -
dpkg-deb --fsys-tarfile "$tmp/graph.deb" | tar -x -C "$tmp" ./usr/lib/habanalabs/libtpc_kernels.so.1
echo "$LIB_MD5  $tmp/usr/lib/habanalabs/libtpc_kernels.so.1" | md5sum -c -
install -d "$DEST"
install -m 0644 "$tmp/usr/lib/habanalabs/libtpc_kernels.so.1" "$DEST/"
echo "Installed $DEST/libtpc_kernels.so.1"
