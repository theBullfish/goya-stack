# The driver

`driver/habanalabs-1.7.1` is Intel's habanalabs kernel driver, release 1.7.1 (internal version
`1.7.1-68c1a21`), as shipped in `habanalabs-dkms-1.7.1-85.all.deb` from vault.habana.ai. That
package's sha256 is `769688706052439ff558f3cfe9e13368d6f43f835c168f8677d14d1d8af8e67d`. It still
supports Goya and builds as an out-of-tree module. It creates
`/dev/hl0`, `/dev/hl1`, ... and `/sys/class/habanalabs/`, which is what hl-thunk expects.

The git history shows every change, one commit each:

1. **`driver: Intel habanalabs-dkms 1.7.1-85, unmodified`.** Intel's 1,206 files as shipped.
2. **`driver: build fixes for current kernels`.** API renames only, so 2022 source compiles on
   Ubuntu 24.04's 6.8 kernel:
   - `vm_flags_set()`, and `class_create()` and `netif_napi_add()` without the removed argument
   - `get_user_pages*()` without the `vmas` argument, and `eventfd_signal()` without the count
   - `get_random_u32()`, `MAX_PAGE_ORDER` and `mmap_lock`
   - the PCIe error-reporting calls the kernel now makes itself
   - header probes in `Makefile.compat` that look under `$(srctree)`

   Nothing about what the driver does changes.
3. **`driver: map Goya command buffers page by page`.** This is the one functional change.

## The one functional change, and why it exists

Intel's `goya_mmap()` hands command-buffer memory to `dma_mmap_coherent()`. On our hosts that
call hits `kernel BUG at mm/memory.c:2049` the moment a program maps a command buffer. We
reproduced it twice on Ubuntu's own unmodified in-tree driver, kernel 6.8.0-138, from two
different programs. A kernel `BUG()` kills the task while it holds memory-map locks. The machine
keeps answering ping but stops accepting logins, and only a power cycle clears it.

Our reading of the kernel source is that the BUG happens when the IOMMU translates DMA, because
`dma_mmap_coherent()` then maps the pages one at a time on a mapping the driver has already
marked as a raw PFN map. That reading is from the source, not measured.

The fix maps the buffer page by page with `remap_pfn_range()`. It uses `vmalloc_to_pfn()` for
vmap'd memory and `virt_to_phys()` for linear memory, and `PAGE_ALIGN`s the length so the
countdown cannot underflow. An earlier version without the alignment hung a host by spinning in
the kernel.

## What our earlier driver build had, and this one does not

Every measurement in the README's verification table ran on our earlier build, srcversion
`1FACA53B6E83AD31C2CBD1F`. Rebuilt from its exact source, it reproduces that srcversion, which
is how we know which source produced the results. Compared with Intel's source, that build
carried these changes on top of the two above. This repo leaves all of them out:

| removed | what it did | why it is out |
|---|---|---|
| MME error suppression | turned the MME's fatal ECC error into a warning and left the engine running | Stock handling resets the card and reports. The suppression could leave a wedged engine and a stuck process. The kernel log for the proven runs (29 Aug to 5 Sep 2026, 971 driver lines) has zero MME ECC events, so the change never fired on the normal path. |
| hard-reset retry | retried a failed hard reset 3 times | In the one recorded case, all four cards failed all three retries. No measured benefit. |
| MME firewall widening | widened the MME's protected DRAM range from 512 MB to 8 GB in `goya_security.c` | A debugging experiment from May 2026, and it is not stock. |
| four extra ioctls | let a program write MME, router and DMA registers through the kernel, plus three debug reads | Nothing in this repo calls them. They also let any program write engine registers. |
| an unused variable | left over from debugging | Dead code. |

## Checking what you built

```bash
modinfo -F version   /lib/modules/$(uname -r)/updates/habanalabs.ko   # 1.7.1-68c1a21
modinfo -F srcversion /lib/modules/$(uname -r)/updates/habanalabs.ko  # 7F1C0DF94FEEB09421EB15F on 6.8.0-138
cat /sys/module/habanalabs/srcversion                                  # what is loaded now
```

srcversion is a hash of the driver source. The same source gives the same srcversion on any
6.8 kernel, so a different value means the source in your checkout differs from this repo.

## After a kernel update

A new kernel boots without this driver. Run `sudo scripts/install-driver.sh` again after any
kernel update, or hold the kernel:

```bash
sudo apt-mark hold linux-generic linux-image-generic linux-headers-generic
```

Run `sudo apt-mark unhold ...` to undo it. The tree builds on the 6.8 kernel series. Newer series
(6.11 and up) need more API updates, so `install-driver.sh` stops on them.
