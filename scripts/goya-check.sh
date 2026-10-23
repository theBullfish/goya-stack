#!/bin/bash
# Read-only health check for Habana Goya cards. Safe to run any time, as often as you like.
# Reads lspci and the driver's cached sysfs files only. It never queries live sensors
# (no hl-smi, no *_clk_curr, no hwmon), because on these cards that can reset the device.
shopt -s nullglob

addrs=$(lspci -D -d 1da3: | cut -d' ' -f1)
if [ -z "$addrs" ]; then
  echo "No Goya (1da3:xxxx) on the PCI bus. See docs/TROUBLESHOOTING.md, 'The card is not in lspci'."
  exit 1
fi

echo "kernel $(uname -r)   driver: $( [ -e /sys/module/habanalabs ] && echo "habanalabs loaded, srcversion $(cat /sys/module/habanalabs/srcversion 2>/dev/null)" || echo "habanalabs NOT loaded")"
lockdown=$(cat /sys/kernel/security/lockdown 2>/dev/null)
[ -n "$lockdown" ] && echo "lockdown: $lockdown   (want [none]; anything else means Secure Boot is on)"
ls /lib/firmware/habanalabs/goya/goya-boot-fit.itb /lib/firmware/habanalabs/goya/goya-fit.itb >/dev/null 2>&1 \
  && echo "firmware files: present" || echo "firmware files: MISSING from /lib/firmware/habanalabs/goya/"

rc=0
for a in $addrs; do
  echo
  echo "== $a  $(lspci -s "$a" -nn | cut -d' ' -f2-)"
  lnk=$(lspci -s "$a" -vv 2>/dev/null | grep -E "LnkSta:" | sed 's/^\s*//; s/,\s*TrErr.*//')
  [ -n "$lnk" ] && echo "   $lnk" || echo "   link: run with sudo to read LnkSta"
  dev=/sys/bus/pci/devices/$a
  [ -e "$dev/driver" ] && echo "   bound to: $(basename "$(readlink "$dev/driver")")" || { echo "   bound to: nothing"; rc=1; }
  # Out-of-tree driver: /sys/class/habanalabs/hlN. In-tree accel driver: /sys/class/accel/accelN/device.
  s=""
  for c in /sys/class/habanalabs/hl[0-9]* /sys/class/accel/accel[0-9]*/device; do
    [ "$(cat "$c/pci_addr" 2>/dev/null)" = "$a" ] && { s=$c; break; }
  done
  if [ -z "$s" ]; then echo "   no driver status files for this card"; rc=1; continue; fi
  st=$(cat "$s/status" 2>/dev/null)
  echo "   status: $st   resets: soft=$(cat "$s/soft_reset_cnt" 2>/dev/null) hard=$(cat "$s/hard_reset_cnt" 2>/dev/null)"
  for f in preboot_btl_ver uboot_ver armcp_ver fuse_ver; do
    [ -r "$s/$f" ] && echo "   $f: $(head -c 120 "$s/$f" | tr -d '\n')"
  done
  [ "$st" = "Operational" ] || rc=1
done
exit $rc
