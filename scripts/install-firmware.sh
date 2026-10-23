#!/bin/bash
# Install Intel's Goya firmware package (habanalabs-firmware 1.1.1-94) from Intel's own server.
# This repo does not redistribute Intel's firmware. The script downloads it and checks its sha256.
set -euo pipefail
URL=https://vault.habana.ai/artifactory/debian/focal/pool/main/h/habanafirmware/habanalabs-firmware-1.1.1-94.amd64.deb
SHA256=c8e122aa46b58f118fca79a7867faa191f04f2811755421c25e175ea7a1ec4c4

[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
if dpkg-query -W -f='${Version}' habanalabs-firmware 2>/dev/null | grep -qx '1.1.1-94'; then
  echo "habanalabs-firmware 1.1.1-94 is already installed."
else
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  echo "Downloading habanalabs-firmware 1.1.1-94 from vault.habana.ai ..."
  curl -fL --retry 3 -o "$tmp/habanalabs-firmware.deb" "$URL"
  echo "$SHA256  $tmp/habanalabs-firmware.deb" | sha256sum -c -
  dpkg -i "$tmp/habanalabs-firmware.deb"
fi
ls -la /lib/firmware/habanalabs/goya/goya-boot-fit.itb /lib/firmware/habanalabs/goya/goya-fit.itb
