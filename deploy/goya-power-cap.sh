#!/bin/bash
# Cap every Goya HL-1000 at its real ~100 W rating.
#
# The habanalabs driver defaults max_power to 200000 mW (MAX_POWER_DEFAULT in
# goya/goyaP.h). That 200 W figure is wrong for this part -- Goya is ~100 W --
# and running with it permitted is the likely cause of the soft-reset history on
# this fleet (hl0 134, hl3 110 as of 2026-09-02).
#
# Measured 2026-09-02: capping costs NOTHING. 200/150/100 W all give identical
# runtime within noise (headline 256^3 int8: 8.49 / 8.48 / 8.53 ms, all
# bit-exact) and 100 W holds under sustained four-card load with zero new resets.
#
# The setting does not survive a driver reload or reboot, hence this unit.
CAP_MW=${GOYA_POWER_CAP_MW:-100000}
rc=0
shopt -s nullglob
for s in /sys/class/habanalabs/hl[0-9]*; do
    [ -w "$s/max_power" ] || continue
    if echo "$CAP_MW" > "$s/max_power" 2>/dev/null; then
        logger -t goya-power-cap "$(basename "$s"): max_power=$(cat "$s/max_power")"
    else
        logger -t goya-power-cap "$(basename "$s"): FAILED to set max_power"; rc=1
    fi
done
exit $rc
