# goya-stack

Run real software on a Habana Goya HL-100 inference card: a working driver, a runtime that
drives the card's matrix engine and its 8 vector cores, a test that checks every result against
the CPU, and a 30-billion-parameter language model generating text on one card.

Habana stopped supporting Goya after SDK 1.1.x, and that SDK never shipped a public way to run
your own work on it. Everything here was built and measured on our own HL-100 cards.

| | what | where |
|---|---|---|
| Driver | Intel's habanalabs 1.7.1 driver, built for Ubuntu 24.04, with one fix | [`driver/`](driver), [docs/DRIVER.md](docs/DRIVER.md) |
| Firmware | Intel's own package, downloaded from Intel and checksum-verified | [`scripts/install-firmware.sh`](scripts/install-firmware.sh) |
| Runtime | hl-thunk (Habana's userspace library) and **libgoyamme**: int8 GEMM on the MME, f32/int32 kernels on the TPC cores | [`runtime/`](runtime) |
| Test | `goya_gemm`: random and swept GEMM shapes, every output element compared with a CPU reference | [`examples/goya_gemm.c`](examples/goya_gemm.c) |
| Inference | Qwen3-30B-A3B generating text on one card, with a built-in reference check | [`examples/qwen3/`](examples/qwen3) |

## What you need

- **The card, powered correctly.** EPS / CPU 8-pin cable, never a PCIe 6+2. About 100 W.
  Forced air across the passive heatsink. Read [docs/HARDWARE.md](docs/HARDWARE.md) and
  [docs/SAFETY.md](docs/SAFETY.md) before powering it.
- **A PC with an x16 slot.** BIOS: **Above 4G Decoding on, Secure Boot off.**
- **Ubuntu 24.04 LTS running its 6.8 kernel.** Easiest is a fresh install from the Ubuntu 24.04
  *Server* image, which uses 6.8. Recent *Desktop* images default to a newer kernel. To move a
  Desktop install to 6.8 for good:
  ```bash
  sudo apt install linux-generic
  sudo apt purge linux-generic-hwe-24.04 linux-image-generic-hwe-24.04 linux-headers-generic-hwe-24.04
  sudo reboot          # once: GRUB > "Advanced options for Ubuntu" > the 6.8.0-... entry
  sudo apt autoremove --purge     # run from 6.8; removes the newer kernels so 6.8 stays the default
  uname -r             # must start with 6.8
  ```
- For the language model: 32 GB of RAM and 35 GB of free disk.

## Setup

```bash
git clone https://github.com/theBullfish/goya-stack.git
cd goya-stack
sudo ./setup.sh
```

`setup.sh` installs the firmware, builds and installs the driver, builds the runtime, caps the
card at 100 W, checks the card and runs a GEMM test on it. It is safe to run again. If another
habanalabs driver is already loaded (for example from SDK 1.1.1-94), it stops and asks you to
reboot once, then you run it again.

Success looks like this:

```
   LnkSta: Speed 16GT/s, Width x16
   bound to: habanalabs
   status: Operational   resets: soft=0 hard=0
...
all exact vs CPU reference
```

Then run the language model, which takes a few minutes per run on one card:

```bash
cd examples/qwen3            # model download is in its README
sudo python3 generate.py --model /path/to/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf --check
```

```
=== OUTPUT ===
The capital of France is Paris. The capital
CHECK PASS: the 4 tokens match the reference run exactly.
```

Any time, `scripts/goya-check.sh` is a read-only health check that is safe to run as often as
you like.

## Using the card from your own code

libgoyamme is a C library. The core call is an int8 matrix multiply on the MME with exact int32
results:

```c
#include <goya_mme.h>       /* installed to /usr/local/include */

gmme *g = gmme_open(NULL);                       /* first free card, or "0000:03:00.0" */
if (!g || gmme_error(g)[0]) { fprintf(stderr, "open: %s\n", gmme_error(g)); return 1; }
/* C[M][N] (int32) = A[M][K] (int8) x B[K][N] (int8), all row-major */
if (gmme_gemm_i8(g, A, B, C, M, K, N)) fprintf(stderr, "gemm: %s\n", gmme_error(g));
gmme_close(g);
```

```bash
cc my_app.c -L/usr/local/lib/goya-stack -Wl,-rpath,/usr/local/lib/goya-stack -lgoyamme -o my_app
sudo ./my_app
```

Limits measured on the hardware:

- **N, the output width, is at most 256 per call.** That is the MME array width. Tile wider
  outputs in 256-column slices, as `examples/qwen3/generate.py` does. A wider call breaks the
  open context until the card is reopened.
- **K is verified bit-exact up to 16384.**
- **M is unlimited**, because libgoyamme tiles it.
- **One process can have a card open at a time.** Run it with `sudo`.

The rest of the API is in [`runtime/libgoyamme/goya_mme.h`](runtime/libgoyamme/goya_mme.h). It
covers weights that stay resident in the card's 8 GB of memory, batched GEMMs, and the TPC
kernels in [`runtime/tpc-kernels/`](runtime/tpc-kernels). Those kernels are element-wise ops,
row reductions, RMSNorm pieces, RoPE, softmax pieces, SwiGLU and int8 requantise. Python can
reach all of it through `ctypes`, which is how the Qwen3 example works.

## Intel's TPC kernel library

Intel's SDK ships 6,799 prebuilt TPC kernels: element-wise math, activations, reductions,
softmax and more. This installs Intel's library from Intel's server, checksum-verified:

```bash
sudo scripts/install-vendor-kernels.sh     # -> /usr/local/lib/goya-stack/vendor/libtpc_kernels.so.1
```

libgoyamme fires vendor kernels that take one f32 input tensor and write one f32 output
(`gmme_tpc_fire_vendor`). A tool to pull any kernel out of the library by name is being added.

## What is verified, and on what

Measured on our HL-100 cards, firmware `28.2.2-sec-0`, Ubuntu 24.04, kernel `6.8.0-138`. Every
GEMM and kernel result was compared with a CPU reference.

| claim | evidence |
|---|---|
| GEMM exact, current fast path | 10 shapes from 64³ to 256³, including non-powers of two, 4 runs each with fresh data. K exact from 256 to 16,384. M exact from 256 to 2,048. Wide outputs tiled to N=256 across 9 shapes, 416,604 elements, zero mismatches (2 Sep 2026) |
| GEMM exact, earlier path | 336 shapes × 3 runs on four cards, then 285,370 random GEMMs with zero mismatches (28 Aug 2026) |
| TPC kernels exact | element-wise ops exact on all 8 cores. RMSNorm, SwiGLU, residual add and int8 requantise bit-exact. One attention head bit-exact (Aug–Sep 2026) |
| Qwen3-30B-A3B generates text on one card | "The capital of France is Paris. The capital", 4 tokens in 158.5 s (2 Sep 2026) |
| This repo builds | driver and runtime built from this repo on kernel 6.8.0-138. The Qwen3 example's model reader agrees with the reader used in the proven run on all 579 tensors (23 Oct 2026) |

The runs above used our earlier driver build. That build carried extra debugging changes this
repo removes; [docs/DRIVER.md](docs/DRIVER.md) lists each one. **A full on-card rerun of this
repo's exact build is pending.** It covers `setup.sh`, the GEMM sweep and the Qwen3 check, and
this line changes when it is done.

## Already installed Habana SDK 1.1.1-94?

You don't need it. This repo takes only the firmware package from that release, and
`setup.sh` installs that directly from Intel. If the SDK's `habanalabs-dkms` driver is installed,
remove it first (`sudo apt remove habanalabs-dkms`). Its driver cannot build on the 6.8 kernel,
and it would compete with this one.

## Support

Brad Svenson, Codex Labs LLC. Email: **watchdogpodcast@gmail.com**.
Questions and problems are welcome. If something fails, send the output of
`sudo scripts/goya-check.sh` and `sudo dmesg | grep -iE "habana|goya" | tail -50`.

## Licenses

- `driver/`: GPL-2.0, Intel's habanalabs driver with our changes.
- `runtime/hl-thunk/`: MIT, Habana's hl-thunk.
- Everything else (libgoyamme, the TPC kernels, examples, scripts, docs): Codex Labs LLC. See
  [LICENSE](LICENSE).
- Intel's firmware is not in this repo. `setup.sh` downloads it from Intel.
