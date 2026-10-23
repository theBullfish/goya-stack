#!/bin/bash
# From a prepared Ubuntu 24.04 host (6.8 kernel, Secure Boot off, card seated and powered)
# to a verified card: firmware, driver, runtime, power cap, health check, GEMM test.
# Safe to run again; every step skips what is already done.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
[ "$(id -u)" = 0 ] || { echo "run with sudo: sudo ./setup.sh"; exit 1; }
step() { echo; echo "=================== $*"; }

step "1/6 firmware";  scripts/install-firmware.sh || exit 1
step "2/6 driver";    scripts/install-driver.sh; rc=$?
if [ $rc = 3 ]; then echo; echo "Reboot, then run sudo ./setup.sh again."; exit 3; fi
[ $rc = 0 ] || exit 1
step "3/6 runtime";   scripts/build-runtime.sh || exit 1
step "4/6 power cap (100 W per card)"
install -m 0755 deploy/goya-power-cap.sh /usr/local/sbin/goya-power-cap.sh
install -m 0644 deploy/goya-power-cap.service /etc/systemd/system/
install -m 0644 deploy/99-goya-power-cap.rules /etc/udev/rules.d/
systemctl daemon-reload && systemctl enable --now goya-power-cap.service && udevadm control --reload-rules
step "5/6 health check"
for _ in $(seq 1 12); do scripts/goya-check.sh >/dev/null 2>&1 && break; sleep 5; done
scripts/goya-check.sh || { echo; echo "The card is not Operational yet. See docs/TROUBLESHOOTING.md."; exit 1; }
step "6/6 GEMM on the card, checked element by element against the CPU"
goya_gemm auto verify
