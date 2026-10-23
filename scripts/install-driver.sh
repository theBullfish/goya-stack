#!/bin/bash
# Build the habanalabs driver in driver/habanalabs-1.7.1 for the running kernel and install it.
# Re-run this after any kernel update; a new kernel boots without it.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
K=$(uname -r)
SRC=$PWD/driver/habanalabs-1.7.1

case "$K" in
  6.8.*) ;;
  *) echo "STOP: this driver tree is built and tested for Ubuntu 24.04's 6.8 kernel, and this machine runs $K."
     echo "Install the 6.8 kernel and boot it: sudo apt install linux-image-generic linux-headers-generic"
     echo "(See README, 'Host setup'.)"; exit 2;;
esac
lockdown=$(cat /sys/kernel/security/lockdown 2>/dev/null || echo "[none]")
case "$lockdown" in
  *"[none]"*) ;;
  *) echo "STOP: kernel lockdown is on ($lockdown). Turn Secure Boot off in the BIOS; this driver is unsigned."; exit 2;;
esac
if find /lib/modules/"$K" -path "*/dkms/*" -name 'habanalabs*.ko*' 2>/dev/null | grep -q .; then
  echo "STOP: a DKMS habanalabs driver is installed for $K (from a Habana SDK package). Remove it first:"
  echo "  sudo apt remove habanalabs-dkms"; exit 2
fi

apt-get install -y build-essential "linux-headers-$K" >/dev/null
BUILD_USER=${SUDO_USER:-root}
sudo -u "$BUILD_USER" make -C "$SRC" -j"$(nproc)" KVERSION="$K" GIT_SHA=68c1a21 M="$SRC"

install -d "/lib/modules/$K/updates"
install -m 0644 "$SRC/drivers/net/ethernet/habanalabs/habanalabs_en.ko" \
                "$SRC/drivers/misc/habanalabs/habanalabs.ko" "/lib/modules/$K/updates/"
depmod -a "$K"
printf 'habanalabs_en\nhabanalabs\n' > /etc/modules-load.d/habanalabs.conf
echo "Installed habanalabs $(modinfo -F version "/lib/modules/$K/updates/habanalabs.ko"), srcversion $(modinfo -F srcversion "/lib/modules/$K/updates/habanalabs.ko"), for $K."

loaded=$(cat /sys/module/habanalabs/srcversion 2>/dev/null || true)
want=$(modinfo -F srcversion "/lib/modules/$K/updates/habanalabs.ko")
if [ "$loaded" = "$want" ]; then
  echo "This driver is already loaded."
elif [ -n "$loaded" ]; then
  echo "A different habanalabs driver is loaded right now (srcversion $loaded). Reboot to switch to this one."
  exit 3
else
  modprobe habanalabs_en && modprobe habanalabs && echo "Loaded. Give the card about 30 seconds to boot its firmware."
fi
