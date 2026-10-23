#!/bin/bash
# Build hl-thunk (Habana's userspace library) and libgoyamme, then install them under /usr/local.
#   lib:     /usr/local/lib/goya-stack/{libgoyamme.so,libhl-thunk.so*}
#   kernels: /usr/local/share/goya-stack/tpc-kernels/*.o
#   tool:    /usr/local/bin/goya_gemm
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
ROOT=$PWD
UAPI=$ROOT/driver/habanalabs-1.7.1/include/uapi
BUILD_USER=${SUDO_USER:-root}
as_user() { sudo -u "$BUILD_USER" "$@"; }

apt-get install -y build-essential cmake libcmocka-dev >/dev/null

echo "== hl-thunk (pinned at upstream commit cac1fd7, the newest that matches this driver)"
# hl-thunk's CMakeLists overwrites CMAKE_C_FLAGS, so the driver's uapi header goes in through CPATH,
# which gcc searches before /usr/include. That keeps any SDK copy of misc/habanalabs.h out of the build.
as_user env CPATH="$UAPI" cmake -S runtime/hl-thunk -B runtime/hl-thunk/build -DHLTESTS_LIB_MODE=1 >/dev/null
as_user env CPATH="$UAPI" cmake --build runtime/hl-thunk/build -j"$(nproc)"

echo "== libgoyamme + goya_gemm"
as_user make -C runtime/libgoyamme -j"$(nproc)"
make -C runtime/libgoyamme install

echo
echo "Installed. Next: sudo goya_gemm auto verify"
