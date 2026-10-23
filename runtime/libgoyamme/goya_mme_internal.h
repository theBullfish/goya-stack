/* goya_mme_internal.h -- shared between the MME and TPC compilation units.
   Not part of the public API; goya_mme.h is. */
#ifndef GOYA_MME_INTERNAL_H
#define GOYA_MME_INTERNAL_H
#include <stdint.h>
#include "hlthunk.h"
#include "hlthunk_tests.h"

struct gmme {
    int fd; uint64_t sram; uint32_t mq, dq;
    void *dA,*dB,*dC,*hb; uint64_t A,B,C,hva;
    int idle, fires, idle_worst, max_tiles, reread_differs; uint32_t eiter, bubbles; int pad, rows_ovr; char err[256];
    int b_pinned, b_K, b_N;   /* weights resident on the card */
    uint64_t b_off_cur;       /* byte offset into B for the current fire */
    char kernels[60][512];    /* kernel arena: every registered kernel, resident */
    int  n_kernels;
    void *dW; uint64_t W, warena_size, warena_used;
    /* Slice table: grown on demand. It used to be a fixed slice[4096], sized
       when a slot was 64 KB; with K-deep slots a 7 GB arena holds 14,336 slots
       at K=2048, so a fixed 4096 names became the binding limit rather than the
       card's memory. */
    struct gmme_slice { char name[128]; void *host; int card_slot; uint64_t used;
                        int K, N, Kp, Np; } *slice;
    int  slice_cap;
    /* Bytes per arena slot, decided at RUNTIME from the largest slice actually
       loaded -- NOT from GMME_MAX_K. Slots stay uniform so eviction never
       fragments, but sizing them to the compile-time maximum K wasted the arena
       by the ratio of that maximum to the model's real K (8x at K=2048 once
       GMME_MAX_K became 16384). Growing it re-partitions the arena, which is
       safe because a slice always keeps its host copy. */
    uint64_t wslot;
    int  n_slices, cur_slice, n_card, n_card_slots;
    uint64_t clock, promotes, spills;
    /* Command buffers are pure scratch: their contents are refilled every
       fire, but hltests_create_cb(INTERNAL) mmaps host memory AND maps it
       into the device MMU -- 1.095 ms measured, per call. Allocating them
       per fire was 88% of the cost of every TPC operation. Cached here,
       created lazily on first fire, freed in gmme_close. */
    void *cb_cfg[8], *cb_eng[8];  /* TPC scratch, one pair per core */
    void *cb_tdma;                /* TPC DMA CB, cached PER CONTEXT (was a static) */
    /* 0 = legacy single shared region (known good, but serializes the cores).
       Non-zero = per-core partitions starting at this SRAM offset. Set with
       GMME_TPC_PART=<hex offset>; GMME_TPC_STRIDE overrides the 4 MB stride. */
    uint32_t tpc_part_base, tpc_part_stride;
    void *cb_mcfg, *cb_meng;      /* MME scratch (cfg is CFG_CB_SZ) */
    int      cur_M;               /* total rows of the CURRENT gemm, for GMME_FULLM */
    int      hb_huge;             /* staging buffer really got 2 MB pages */
    uint32_t eng_staged_o;        /* the cfg-CB length baked into the staged CP_DMA.
                                     The engine CB embeds cp_dma.size = o, so it DOES
                                     change when the tile count changes. Caching on
                                     eng_staged alone left a stale short CP_DMA and
                                     only the first descriptor ever fired. */
    int      sram_dirty;          /* another engine staged into SRAM since the
                                     MME last staged its CBs; forces a re-stage
                                     even if the sizes match. */
    uint32_t eng_staged;          /* bytes of cb_meng already resident in SRAM: the
                                     engine CB is a constant CP_DMA, upload it ONCE */
    void *cb_mdma;                /* the per-fire EXTERNAL CB, cached: creating and
                                     destroying it every fire is pure overhead */
};

/* SRAM map. 50.59 MB total (measured, hw_ip_info). The kernel arena is
   SHARED and read-only -- every core can fire every resident kernel with
   no reload. Per-core scratch partitions live in goya_tpc.c; the MME keeps
   its own CB region below them. Nothing is reused across engines any more:
   sharing scratch is what made concurrency impossible. */
/* Shared between the two translation units so the MME and the TPCs can be
   in flight at the same time. Submitting without waiting is the whole point:
   the device-idle poll is device-WIDE, so paying it per operation is what
   forced one engine at a time. */
struct gmme_tpc_job_s;
int  gmme_submit_async(gmme *g, void *eng, uint32_t eo, uint32_t q,
                       uint64_t *seq, void **dma);
int  gmme_submit_multiq(gmme *g, void *const *eng, const uint32_t *eo,
                        const uint32_t *q, int n, uint64_t *seq, void **dma);
int  gmme_wait_all(gmme *g, uint64_t *seq, void **dma, int n);
int  gmme_settle(gmme *g);
int  gmme_in_chain(void);
/* Several host->device transfers in ONE command submission. hltests_dma_transfer
   is create-CB + submit + WAIT + destroy PER TRANSFER; batching is what lets
   independent engines stage in parallel instead of one at a time. */
struct stage_req { uint64_t src, dst; uint32_t size; int dir; };
int  gmme_stage_multi(gmme *g, const struct stage_req *r, int n);

int  tpc_validate  (gmme *g, void *jobs, int n);
int  tpc_stage_all (gmme *g, void *jobs, int n, uint64_t *seq, void **dma, uint32_t *tq);
int  tpc_exec_all  (gmme *g, void *jobs, int n, uint64_t *seq, void **dma, uint32_t *tq);
void tpc_collect_all(gmme *g, void *jobs, int n);

#define SRAM_TPC_KERNEL_OFF 0x000000
#define SRAM_TPC_IN_OFF     0x100000
#define SRAM_TPC_OUT_OFF    0x600000
#endif
