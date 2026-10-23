/* goya_mme.c -- see goya_mme.h. Descriptor form proven on HL-100 cards 2026-08-28.
 *
 * The two facts this file exists to encode:
 *   1. VALID_ELEMENTS are CUMULATIVE PRODUCTS ({K, K*M, ...}), not per-dim sizes.
 *      Writing per-dim sizes describes a one-row tensor and silently caps output at 1 row.
 *   2. The MME trigger needs EB=1. The vendor's EB=0 resets a gen-1 card.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "goya_mme.h"
#include "hlthunk.h"
#include "hlthunk_tests.h"


/* SPMU hardware counters bracketing the REAL command-buffer fire, via the stock
   HL_DEBUG ioctl only -- no BAR0, no debugfs, no reset. Only the ioctl mechanics
   of an earlier research tool are reused; that tool drove the MME by raw BAR0
   writes and is deliberately NOT used.
   Points: 6 MME1_SBA (operand A read) 7 MME3_SBB (operand B read)
           8 MME4_WACS2               9 MME4_WACS (write-back). out[4] = cycles.
   The question: does write-back fire once per OUTPUT ROW, or once per ACCUMULATE
   STEP (rows*K)? The latter means C round-trips every step. */
static int e_spmu = -1, e_sram = 0;
static void exp_init(void);
/* GMME_SRAM=1 puts the MME OPERANDS in on-die SRAM instead of DRAM.
   Until now only the command buffers lived in SRAM; A, B and C were all
   hltests_allocate_device_mem() = DRAM, so every operand fetch and every
   write-back crossed the DRAM bus. Measured cost is ~145 ns per accumulate
   step, which is DRAM-latency territory. A 256^3 int8 GEMM needs
   A 64K + B 64K + C 256K = 384 KB, against ~50.59 MB of SRAM (measured,
   hw_ip_info) of which only the TPC regions and the two MME CBs are used.
   SRAM map already in use: 0x000000 TPC kernels, 0x100000 TPC in,
   0x600000 TPC out, 0x700000 MME cfg CB, 0x710000 MME eng CB. */
#define SRAM_MME_A_OFF 0x800000
#define SRAM_MME_B_OFF 0x900000
#define SRAM_MME_C_OFF 0xA00000
#define GMME_DIR_IN  (e_sram ? HL_DMA_HOST_TO_SRAM : HL_DMA_HOST_TO_DRAM)
#define GMME_DIR_OUT (e_sram ? HL_DMA_SRAM_TO_HOST : HL_DMA_DRAM_TO_HOST)
struct gmme_dbg_spmu { __u64 ev[10]; __u32 n; __u32 pmtrc; __u32 tch; __u32 teh; };
static int gmme_dbg(int fd, __u32 op, __u32 idx, __u32 en,
                    void *in, __u32 is, void *out, __u32 os)
{
    struct hl_debug_args a; memset(&a, 0, sizeof a);
    a.op = op; a.reg_idx = idx; a.enable = en;
    a.input_ptr = (__u64)(uintptr_t)in; a.input_size = is;
    a.output_ptr = (__u64)(uintptr_t)out; a.output_size = os;
    return hlthunk_debug(fd, &a);
}
#define CFGCB 0x700000
#define ENGCB 0x710000
#define SPHDR 0x80128140u   /* spatial: TRANS_A set (conv wants it CLEAR) */
#define HDR       0x80128100u   /* STORE_OUT | ACCUM_MASK=1 | int8 in / int32 out.
                                  TRANS_A (bit 6, MME_ARCH_HEADER_TRANS_A) CLEARED
                                  2026-09-01: we were asking the engine to transpose
                                  the A operand every fire. It is a no-op for our
                                  layout -- bit-exact either way -- and cost ~30-50%.
                                  K=256 rows=4: 12.75 -> 6.10 ms. It also specifically
                                  penalised multi-row fires, which is why every rows
                                  sweep looked flat. */
#define ASSOC0    0x03690B48u   /* loop0->K (A0,B1,CIN5,COUT5); loop1->M (A1,B5,CIN5,COUT1) */
#define CAP_A (32u<<20)   /* one A tile per K tile, staged together */
#define CAP_B (32u<<20)   /* 8 resident weight matrices: a layer needs more than one */
#define CAP_C (96u<<20)   /* K-tiled partials + accumulator, AND eight per-lane
                              operand partitions (8 MB each from 8 MB) for the wave */
#define CAP_H (72u<<20)
#define HSTAGE_B (36u<<20)   /* B's slice of the host bounce buffer.
                              MUST be < CAP_H -- it is a HOST offset,
                              and tying it to CAP_A broke the moment
                              CAP_A grew past the host buffer. */
/* WEIGHT ARENA -- many slices resident in the card's DRAM at once.
 * The card has 7.5 GB and we were using 64 KB of it, holding exactly ONE
 * matrix. Slices are bump-allocated here and addressed by name, so a card can
 * hold a chunk of a model rather than a single tensor. Size via GMME_ARENA_MB. */
#define WARENA_DEFAULT_MB 256
/* There is no fixed name limit any more. WARENA_MAX_SLOTS was 4096, sized when
   a slot was 64 KB; the slice table now grows on demand (slice_table_room). */
/* Slots are uniform so eviction never fragments, and weights are read-only, so
   a slice always keeps its HOST copy: spilling is just forgetting, promoting is
   one DMA, and host RAM is the overflow tier under the card's own memory.
   THE SLOT SIZE IS A RUNTIME VALUE (g->wslot), set from the largest slice
   actually loaded. It used to be GMME_MAX_K * GMME_MAX_N -- fine at 256x256,
   but when GMME_MAX_K was corrected 256 -> 16384 every slot silently became
   4 MB, so a real K=2048 column block (512 KB) burned 8x the arena it needed
   and a 256 MB arena fell from 4096 slots to 64. Sizing off a compile-time
   ceiling is the same class of error as believing that ceiling was hardware. */
#define WSLOT_MIN   4096u        /* one page; also the granularity slots round to */
/* A descriptor is 107 WREG32 packets. A config CB must NOT cross a 64 KB
   boundary: the descriptor that straddles 0x10000 comes back corrupted, and the
   engine then writes wrong results for exactly that tile. Measured: with 77
   descriptors per CB one row is wrong, with 76 none are. Keep the whole CB
   inside one 64 KB block. */
#define CFG_CB_SZ        0x10000u
#define DESC_BYTES          (107*8)
#define MAX_TILES_PER_CB  (CFG_CB_SZ / DESC_BYTES)

#include "goya_mme_internal.h"
#include "vdesc.h"
#include <misc/habanalabs.h>
#include <errno.h>


/* One PACKET_WREG_BULK covering n_dw CONSECUTIVE 32-bit registers starting at
   reg_start. Goya packet layout (include/specs/goya/goya_packets.h):
     dw0: size64 (count of 64-bit values, low 16 bits)
     dw1: ctl = reg_offset[15:0] | opcode<<24 | eb<<29 | reg_barrier<<30 | mb<<31
     then size64 64-bit values.
   n_dw must be even. Written straight into the CB, which is a plain buffer. */
static uint32_t wbulk(void *cb, uint32_t o, uint32_t reg_start,
                      const uint32_t *vals, int n_dw, int eb)
{
    uint8_t *b = (uint8_t *)cb + o;
    uint32_t hdr[2];
    if (n_dw & 1) return o;                     /* must be a whole 64-bit count */
    hdr[0] = (uint32_t)(n_dw / 2) & 0xFFFFu;    /* size64 */
    hdr[1] = (reg_start & 0xFFFFu)
           | (2u  << 24)                        /* PACKET_WREG_BULK */
           | ((eb ? 1u : 0u) << 29)             /* eng_barrier */
           | (1u  << 30)                        /* reg_barrier, must be 1 */
           | (1u  << 31);                       /* msg_barrier */
    memcpy(b, hdr, 8);
    memcpy(b + 8, vals, (size_t)n_dw * 4);
    return o + 8 + (uint32_t)n_dw * 4;
}

static int vreg_override(uint32_t reg, uint32_t *out);   /* fwd decl */
static uint32_t w(int fd, void *cb, uint32_t o, uint32_t r, uint32_t v, int eb) {
    struct hltests_pkt_info p; memset(&p, 0, sizeof(p));
    { uint32_t ov; if (vreg_override(r, &ov)) v = ov; }   /* GMME_VREG applies on EVERY path */
    p.eb = eb ? EB_TRUE : EB_FALSE; p.mb = MB_TRUE;
    p.wreg32.reg_addr = (uint16_t)(r & 0xffff); p.wreg32.value = v;
    return hltests_add_wreg32_pkt(fd, cb, o, &p);
}

int gmme_rows_per_fire(int K) {
    /* MEASURED 2026-09-01, after clearing TRANS_A (HDR bit 6).
     *
     * The old formula was `256/K`, described as an "A-tile limit: rows*K <= 256".
     * That was never a hardware limit -- it was an empirical fit to the behaviour
     * of a descriptor that transposed A on every fire, which specifically
     * penalised multi-row work. With the spurious transpose gone the curve is
     * monotonic and MORE rows is simply better:
     *
     *   K=32   rows 8 -> 64 : 1.20 -> 1.07 ms
     *   K=64   rows 4 -> 64 : 1.99 -> 1.74 ms
     *   K=128  rows 2 -> 16 : 3.81 -> 3.18 ms
     *   K=256  rows 1 -> 64 : 7.51 -> 5.78 ms
     *
     * All bit-exact. 64 is the measured optimum across every K tested; beyond it
     * the curve is flat, so there is no reason to go higher and pay larger tiles.
     */
    if (K < 1 || K > GMME_MAX_K) return 0;
    return GMME_ROWS_PER_FIRE;
}

const char *gmme_error(const gmme *g) { return g ? g->err : "no context"; }
int gmme_idle (const gmme *g) { return g ? g->idle  : 0; }
int gmme_fires(const gmme *g) { return g ? g->fires : 0; }
int gmme_idle_polls(const gmme *g) { return g ? g->idle_worst : 0; }
int gmme_reread_differs(const gmme *g) { return g ? g->reread_differs : 0; }

gmme *gmme_open(const char *busid) {
    gmme *g = calloc(1, sizeof *g);
    if (!g) return NULL;
    if (hltests_init()) { snprintf(g->err,sizeof g->err,"hltests_init failed"); return g; }
    g->fd = hltests_open(busid);
    if (g->fd < 0) { snprintf(g->err,sizeof g->err,"open(%s) = %d", busid?busid:"auto", g->fd); return g; }
    struct hlthunk_hw_ip_info hw; memset(&hw,0,sizeof hw);
    hlthunk_get_hw_ip_info(g->fd, &hw);
    g->sram = hw.sram_base_address;
    g->mq = hltests_get_mme_qid(g->fd, 0, STREAM0);
    g->dq = hltests_get_dma_down_qid(g->fd, STREAM0);
    g->dA = hltests_allocate_device_mem(g->fd, CAP_A, NOT_CONTIGUOUS);
    g->dB = hltests_allocate_device_mem(g->fd, CAP_B, NOT_CONTIGUOUS);
    g->dC = hltests_allocate_device_mem(g->fd, CAP_C, NOT_CONTIGUOUS);
    /* The staging buffer is DMA'd to the card on every operand upload. On 4 KB
       pages a 16 MB transfer is ~4000 scatter-gather entries and the link runs at
       ~2.8 GB/s on a Gen4 x16 card (~25 GB/s of capability). 2 MB huge pages cut
       the entry count by 512x. GMME_NOHUGE=1 forces the old path for A/B. */
    g->hb = getenv("GMME_NOHUGE") ? NULL
          : hltests_allocate_host_mem(g->fd, CAP_H, HUGE_MAP);
    g->hb_huge = (g->hb != NULL);
    if (!g->hb) g->hb = hltests_allocate_host_mem(g->fd, CAP_H, NOT_HUGE_MAP);
    if (getenv("GMME_VERBOSE"))
        fprintf(stderr, "  [gmme] staging buffer %u MB on %s pages\n",
                (unsigned)(CAP_H >> 20), g->hb_huge ? "HUGE(2MB)" : "4KB");
    if (!g->dA || !g->dB || !g->dC || !g->hb) {
        snprintf(g->err,sizeof g->err,"device/host allocation failed"); return g; }
    g->A = (uint64_t)(uintptr_t)g->dA; g->B = (uint64_t)(uintptr_t)g->dB;
    g->C = (uint64_t)(uintptr_t)g->dC;
    if (e_sram) {   /* keep the DRAM allocations mapped; just aim the engine at SRAM */
        g->A = g->sram + SRAM_MME_A_OFF;
        g->B = g->sram + SRAM_MME_B_OFF;
        g->C = g->sram + SRAM_MME_C_OFF;
    }
    g->hva = hltests_get_device_va_for_host_ptr(g->fd, g->hb);
    g->cur_slice = -1;   /* calloc gives 0, which would look like slice 0 is active */
    {   const char *mb = getenv("GMME_ARENA_MB");
        g->warena_size = (uint64_t)(mb ? atoi(mb) : WARENA_DEFAULT_MB) << 20;
        g->dW = hltests_allocate_device_mem(g->fd, g->warena_size, NOT_CONTIGUOUS);
        if (g->dW) { g->W = (uint64_t)(uintptr_t)g->dW;
                     g->wslot = 0; g->n_card_slots = 0; }   /* sized on first load_slice */
        else { g->warena_size = 0;
               snprintf(g->err,sizeof g->err,"weight arena alloc failed"); } }
    exp_init();
    if (e_spmu >= 0) gmme_dbg(g->fd, HL_DEBUG_OP_SET_MODE, 0, 1, NULL, 0, NULL, 0);
    {   const char *e = getenv("GMME_MAX_TILES");   /* 1 = one submit per tile */
        g->max_tiles = e ? atoi(e) : MAX_TILES_PER_CB;
        if (g->max_tiles < 1) g->max_tiles = 1;
        if (g->max_tiles > MAX_TILES_PER_CB) g->max_tiles = MAX_TILES_PER_CB;
        e = getenv("GMME_EITER");   g->eiter   = e ? (uint32_t)strtoul(e,NULL,0) : 0;
        e = getenv("GMME_BUBBLES"); g->bubbles = e ? (uint32_t)strtoul(e,NULL,0) : 0;
        /* Padding defaults ON. Without it any tile or block ending on a
           remainder computes a wrong answer -- measured: 32 of 704 shapes
           silently wrong, every one with K not a multiple of 32 (K=8,16,33).
           GMME_PAD=0 still disables it for experiments. */
        e = getenv("GMME_TPC_PART");   g->tpc_part_base   = e ? (uint32_t)strtoul(e,NULL,0) : 0;
        e = getenv("GMME_TPC_STRIDE"); g->tpc_part_stride = e ? (uint32_t)strtoul(e,NULL,0) : 0x400000u;
        e = getenv("GMME_PAD");     g->pad     = e ? atoi(e) : 1;
        e = getenv("GMME_ROWS");    g->rows_ovr= e ? atoi(e) : 0;
        }
    return g;
}

void gmme_close(gmme *g) {
    if (g && g->fd >= 0) {
        int ci;
        for (ci = 0; ci < 8; ci++) {
            if (g->cb_cfg[ci]) { hltests_destroy_cb(g->fd, g->cb_cfg[ci]); g->cb_cfg[ci] = NULL; }
            if (g->cb_eng[ci]) { hltests_destroy_cb(g->fd, g->cb_eng[ci]); g->cb_eng[ci] = NULL; }
        }
        if (g->cb_mcfg) hltests_destroy_cb(g->fd, g->cb_mcfg);
        if (g->cb_meng) hltests_destroy_cb(g->fd, g->cb_meng);
        if (g->cb_mdma) { hltests_destroy_cb(g->fd, g->cb_mdma); g->cb_mdma = NULL; }
        g->cb_mcfg = g->cb_meng = NULL;
        g->eng_staged = g->eng_staged_o = 0;
    }
    if (!g) return;
    if (g->slice) {                      /* host copies are ours; they were leaked before */
        int si; for (si = 0; si < g->n_slices; si++) free(g->slice[si].host);
        free(g->slice); g->slice = NULL;
    }
    if (g->fd >= 0) hltests_close(g->fd);
    free(g);
}


/* GMME_VREG="0xREG=0xVAL,..." : override individual registers in the VDESC
   replay. Lets us correct vendor extents that do not match OUR buffers without
   a rebuild per experiment. Returns 1 and sets *out if reg is overridden. */
static int vreg_override(uint32_t reg, uint32_t *out) {
    static char buf[1024]; static int inited = 0;
    const char *p; char *q;
    if (!inited) { const char *e = getenv("GMME_VREG");
        buf[0] = 0; if (e) { strncpy(buf, e, sizeof buf - 1); buf[sizeof buf - 1] = 0; }
        inited = 1; }
    if (!buf[0]) return 0;
    for (p = buf; *p; ) {
        unsigned long r = strtoul(p, &q, 0);
        if (q == p || *q != '=') break;
        p = q + 1;
        { unsigned long v = strtoul(p, &q, 0);
          if (q == p) break;
          if ((uint32_t)r == reg) { *out = (uint32_t)v;
              fprintf(stderr,"      [VREG override 0x%05X -> 0x%08X]\n", reg, (uint32_t)v);
              return 1; } }
        p = q; while (*p == ',' || *p == ' ') p++;
    }
    return 0;
}


/* Stage several host->device transfers in ONE command submission.
   hltests_dma_transfer creates a CB, adds one packet, submits, waits and
   destroys -- a full round trip PER TRANSFER. A and B are adjacent and both go
   to the down queue, so one CB with two packets costs one submission instead of
   two. Packets in a CB execute in order, so both are resident before we return. */
int gmme_stage_multi(gmme *g, const struct stage_req *r, int n)
{
    void *cb; uint32_t off = 0; int i, rc;
    struct hltests_pkt_info p;
    if (n <= 0) return 0;
    cb = hltests_create_cb(g->fd, 0x8000, EXTERNAL, 0);   /* room for many packets */
    if (!cb) return -1;
    for (i = 0; i < n; i++) {
        memset(&p, 0, sizeof p);
        p.eb = EB_FALSE; p.mb = MB_TRUE;
        p.dma.src_addr = r[i].src; p.dma.dst_addr = r[i].dst;
        p.dma.size = r[i].size;   p.dma.dma_dir = r[i].dir;
        off = hltests_add_dma_pkt(g->fd, cb, off, &p);
    }
    rc = hltests_submit_and_wait_cs(g->fd, cb, off, g->dq,
                                    DESTROY_CB_TRUE, HL_WAIT_CS_STATUS_COMPLETED);
    return rc;
}

/* Append one full descriptor + EXECUTE for a tile. Returns new cb offset.
   EB=1 on the trigger is what serialises tile N against tile N+1 overwriting ARCH. */
/* EXPERIMENT KNOBS (DABS round 2, 2026-09-01). Default = EXACT current behaviour.
   GMME_SPATIAL=0 : write 0 instead of rows-1 into A/C SPATIAL_SIZE_MINUS_1 (b+0x6C).
     Tests whether SPATIAL_SIZE_MINUS_1 is an independent loop count that
     double-counts rows against KERNEL_SIZE_MINUS_1.DIM_1 (the O(rows^2) claim).
   GMME_AROI=<n>  : write n instead of 1 into A_ROI_SIZE_0 (b+0x3C).
     Tests whether ROI_SIZE is per-step fetch granularity (the ~1000x claim).
     Corpus warning: ROI_SIZE must never exceed VALID_ELEMENTS. A_VALID_ELEMENTS_0
     is K, so n=K sits AT the limit, not past it. */
static int e_aw = 0, e_accum = -1;
static uint32_t e_hdr = 0, e_assoc = 0;
static int e_spmode = 0, e_bspat = 0, e_bstr = -1, e_bconv = 0, e_vdesc = 0, e_bulk = 0, e_signal = 0, e_fence = 0, e_sob = 0, e_mmeidle = 0;
static int e_avb = 0, e_braw = 0, e_sobval = 0;
static int in_chain = 0;
static int acc_cin  = 0;   /* LOAD_CIN: C += A@B instead of C = A@B (K-tiling) */   /* inside gmme_chain_begin/end: settle ONCE, not per fire */
static int e_spatial = -1, e_aroi = -1, e_dim0 = 0, e_dim1 = 0, e_ave = 0, e_poison = 0, e_swapa = 0, e_uncap = 0, e_sparse = 0, e_credits = 0;
/* SOUND-AND-TRACE bit-map: poison one field GROUP at a time by SHRINKING it
   (never enlarging -- an enlarged extent would read out of bounds). A group
   whose poisoning leaves the GEMM bit-exact is a DEAD WIRE: the engine is not
   consuming that field. Bits: 1 A_VE 2 A_STRIDE 4 A_ROI 8 A_SPATIAL
   16 B_VE 32 B_STRIDE 64 B_ROI 128 B_SPATIAL 256 C_VE 512 C_STRIDE
   1024 C_ROI 2048 C_SPATIAL 4096 KERNEL_SIZE 8192 ASSOC */
#define PZ(bit,val) ((e_poison & (bit)) ? 1u : (uint32_t)(val))
static void exp_init(void){
    const char *e;
    if (e_spatial >= 0) return;
    e = getenv("GMME_SPATIAL"); e_spatial = e ? atoi(e) : 0;   /* DEFAULT 0 since 2026-09-01: writing rows-1 here double-counts the row loop against KERNEL_SIZE_MINUS_1.DIM_1, making per-descriptor cost O(rows^2). Zeroing it is bit-exact and up to 12x faster at high rows. */
    /* GMME_AW=w : the COHERENT TRIPLE. ROI_SIZE is a DIVISOR of the extent in
       the loop-count derivation (DIM = ceil((VALID-BASE)/ROI_SIZE)-1), so it
       cannot move alone: the step count must fall by w and the pointer must
       advance by w. Changing ROI_SIZE by itself made the engine walk K steps
       of w elements over a K-element operand -- out of bounds, wrong answers,
       643x slower. Sets A_ROI_SIZE_0=w, A_LOOP_STRIDE_0=w, DIM_0=K/w-1. */
    /* GMME_ACCUM=n : override HDR bits 11:8 (ACCUM_MASK) -- the bitmask of which
       conv loops keep their partial sums in the accumulator GRF. We have always
       written 1 (loop0/K only). Per the gen-1 field map this is SEPARATE from
       address motion, which is why sweeping strides/ROI never moved it.
       GMME_HDR=0xVALUE overrides the whole header. */
    /* GMME_ASSOC=0xVALUE : override ASSOCIATED_DIMS_0. Default binds loop0->K
       (inner) and loop1->M (outer). The K loop is a dependency chain -- every
       step accumulates what the last one wrote -- so with it innermost nothing
       fills the pipeline. Swapping puts independent rows innermost.
       Must be paired with GMME_DIM0/GMME_DIM1, since the loop counts swap too. */
    /* GMME_SPMODE=1 : emit the SPATIAL-form descriptor instead of the conv-loop
       form. Captured 2026-09-02 from a live SynapseAI 1.1.1 GEMM: the vendor
       ZEROES KERNEL_SIZE_MINUS_1 and ASSOCIATED_DIMS_0 -- no conv loops at all --
       and drives the whole GEMM through SPATIAL_SIZE_MINUS_1 with ROI_SIZE = K.
       Its B is COMPRESSED (header bit 21), which we cannot copy, so B keeps our
       dense geometry here. See docs/VENDOR-DESCRIPTOR-DIFF.md. */
    /* B geometry knobs for the spatial form. B must be STATIONARY across spatial
       steps (every output row multiplies the same K x N weights), so its spatial
       size should be 0, not K-1. Left tunable while the form is being found. */
    /* GMME_BCONV=1 : in spatial mode, emit the B block EXACTLY as the proven
       conv-form does. The vendor B is COMPRESSED (header bit 21) so its numbers
       never applied to our dense weights; our conv B geometry is known good. */
    e = getenv("GMME_BCONV");   e_bconv   = e ? atoi(e) : 3;   /* DEFAULT 3: spatial B geometry */
    e = getenv("GMME_BSPAT");   e_bspat   = e ? atoi(e) : 0;   /* B_SPATIAL_SIZE_MINUS_1 */
    e = getenv("GMME_BSTR");    e_bstr    = e ? atoi(e) : -1;  /* B_SPATIAL_STRIDE_0 */
    /* GMME_VDESC=1 : replay the VENDOR descriptor verbatim (vdesc.h), captured
       from the live registers after a VERIFIED bit-exact SynapseAI 1.1.1 raw
       int8 256x256x256 GEMM at 0.213 ms / 50 MACs/cycle. Only the six base
       address dwords are substituted with our buffers. Valid for 256x256x256
       only -- it is one captured shape, not a formula. */
    /* GMME_BULK=1 : emit the descriptor as ONE PACKET_WREG_BULK instead of 106
       individual PACKET_WREG_32s. The vendor does this -- goya::getRegForLoadDesc
       returns 0x8, which is exactly the 16-bit reg_offset of a bulk write starting
       at MME_ARCH_A_BASE_ADDR_HIGH (0xD0008), and addLoadDesc() writes by SECTION.
       107 separate writes let the engine observe a HALF-UPDATED descriptor; a bulk
       write lands as a unit. Same values, different atomicity. */
    /* GMME_SIGNAL=1 : make the MME SIGNAL a sync object on completion, the way
       the vendor does. Decoded from mme_masks.h and the vendor descriptor:
         HEADER bit 5 = SIGNAL_EN
         SYNC_OBJECT_MESSAGE (0xD01A0): value[15:0] | so_offset[30:16] | op[31]
         vendor writes 0x80000001 = increment SOB 0 by 1
       Without this nothing tells us the engine finished, so we poll device-idle
       20000 times. With it we can wait on the object instead. */
    /* GMME_FENCE=1 : the vendor structure in miniature. The MME signals a sync
       object on completion; the DMA queue CB arms a monitor on that object and
       FENCES, so the CS itself cannot complete until the engine is genuinely
       done. hltests_wait_for_cs then becomes authoritative and the device-idle
       settle should return on its FIRST poll instead of spinning 20000 times.
       The settle is deliberately KEPT -- it is what returns the card to low-power
       idle (thermal), see docs/GOYA-POWER-AND-THERMAL.md. We are making it free,
       not removing it.  GMME_SOB picks the object (default 0). */
    e = getenv("GMME_MMEIDLE"); e_mmeidle = e ? atoi(e) : 0;
    e = getenv("GMME_FENCE");   e_fence   = e ? atoi(e) : 0;
    e = getenv("GMME_SOB");     e_sob     = e ? atoi(e) : 0;
    e = getenv("GMME_SIGNAL");  e_signal  = e ? atoi(e) : 0;
    e = getenv("GMME_BULK");    e_bulk    = e ? atoi(e) : 0;
    e = getenv("GMME_VDESC");   e_vdesc   = e ? atoi(e) : 0;
    e = getenv("GMME_SPMODE");  e_spmode  = e ? atoi(e) : 1;   /* DEFAULT ON 2026-09-02: the 12x spatial path */
    e = getenv("GMME_ASSOC");   e_assoc   = e ? (uint32_t)strtoul(e,NULL,0) : 0;
    e = getenv("GMME_ACCUM");   e_accum   = e ? (int)strtol(e,NULL,0) : -1;
    e = getenv("GMME_HDR");     e_hdr     = e ? (uint32_t)strtoul(e,NULL,0) : 0;
    e = getenv("GMME_AW");      e_aw      = e ? atoi(e) : 0;
    e = getenv("GMME_AROI");    e_aroi    = e ? atoi(e) : 0;
    e = getenv("GMME_AVB");     e_avb     = e ? (int)strtol(e,NULL,0) : 0;
    e = getenv("GMME_BRAW");    e_braw    = e ? atoi(e) : 1;
    e = getenv("GMME_SOBVAL"); e_sobval  = e ? atoi(e) : 0;
    /* MEASURED 2026-09-02: the MME adds 1 to the SOB per fire, NOT 8.
       The old tiles*8 came from the conv path and makes the fence wait
       for a count that never arrives -> submit/wait rc=1. */   /* DEFAULT 1: row-major B (TRANS_A already transposes) */
    /* GMME_BRAW=1: keep the spatial B GEOMETRY (256-byte pitch, full ROI) but
       upload B ROW-MAJOR instead of transposed. With HEADER TRANS_A set the
       engine already reads A down K and B transposed; transposing the upload
       as well is a double transpose, which the one-hot probe showed directly
       (C[0] == B[:,k] instead of B[k,:]). */
    /* GMME_AVB: transplant vendor A-block fields onto the CONV descriptor.
       bit0 = A_VALID_ELEMENTS_2..4 -> 1        (vendor 1, ours K*rows)
       bit1 = A_ROI_SIZE_1          -> K*rows   (vendor 0x10000 at rows=256, ours 1)
       bit2 = A_LOOP_STRIDE_0       -> 0        (vendor 0, ours 1) */   /* 0 = leave as 1 */
    e = getenv("GMME_DIM0");    e_dim0    = e ? atoi(e) : 0;   /* 0 = use K    */
    e = getenv("GMME_DIM1");    e_dim1    = e ? atoi(e) : 0;   /* 0 = use rows */
    e = getenv("GMME_AVE");     e_ave     = e ? atoi(e) : 0;   /* 1 = A VALID_ELEMENTS_1..4 = K, not K*rows */
    e = getenv("GMME_POISON");  e_poison  = e ? (int)strtol(e,NULL,0) : 0;
    e = getenv("GMME_SWAPA");   e_swapa   = e ? atoi(e) : 0;
    /* GMME_UNCAP=1 bypasses the GMME_MAX_N front-end guard. GMME_MAX_N=256 is
       commented "hardware limit" but was never tested -- the same comment on
       GMME_MAX_ROWS=8 was DISPROVED 2026-09-01 (rows=128 is bit-exact). The
       8-bit KERNEL_SIZE_MINUS_1 field genuinely bounds K, but N lives in
       32-bit tensor extents, so nothing structural caps it at 256.
       Buffer caps still apply and are checked separately. */
    e = getenv("GMME_UNCAP");   e_uncap   = e ? atoi(e) : 0;
    e = getenv("GMME_SRAM");    e_sram    = e ? atoi(e) : 0;
    e = getenv("GMME_SPMU");    e_spmu    = e ? (int)strtol(e,NULL,0) : -1;
    e = getenv("GMME_CREDITS"); e_credits = e ? (int)strtol(e,NULL,0) : 0;
    e = getenv("GMME_SPARSE");  e_sparse  = e ? atoi(e) : 1;   /* DEFAULT 1 since 2026-09-01: MME_ARCH latches, so only A base, C base and the trigger need rewriting per tile. Bit-exact, saves ~61ns x 102 writes per descriptor. */
}

/* ENGINE CREDIT REGISTERS (2026-09-01) -- outside the descriptor, never written
   by this library until now. From goya/asic_reg/mme_regs.h + mme_masks.h:
     0xD0300 MME_STORE_MAX_CREDIT  V              [5:0]  max 63
     0xD0304 MME_AGU  SBA_MAX_CREDIT [4:0]  max 31   (operand A in)
                      SBB_MAX_CREDIT [12:8] max 31   (operand B in)
                      SBC_MAX_CREDIT [20:16]max 31   (CIN in)
                      WBC_MAX_CREDIT [29:24]max 63   (write-back out)
     0xD0314 MME_WBC, 0xD0324 MME_WBC_CONTROL_DATA
   These gate how many operand fetches and stores the engine may keep IN FLIGHT.
   If they come out of reset at 0/1 the engine serialises on memory latency for
   every accumulate step, which is exactly the measured ~145 ns/step.
   Values 1 = the vendor-derived set used by that earlier research tool
   (STORE 0x21, AGU 0x0f0f0f10, WBC 0x3200, WBC_CTL 0x401).
   Values 2 = every credit field at its architectural maximum.
   Written as ordinary WREG32 into the config CB -- same path as the descriptor,
   no BAR0. These are credit/config registers, NOT reset or boot registers. */
static uint32_t credits(gmme *g, void *cfg, uint32_t o)
{
    uint32_t store, agu, wbc, wbcd;
    if (e_credits == 2) {
        store = 0x3Fu;                                   /* V = 63 */
        agu   = (31u) | (31u << 8) | (31u << 16) | (63u << 24);
        wbc   = 0x3200u; wbcd = 0x401u;
    } else {
        store = 0x21u; agu = 0x0f0f0f10u; wbc = 0x3200u; wbcd = 0x401u;
    }
    /* GMME_CREDITS=3: also program the STREAMING BUFFER DEPTH registers, which
       neither this library nor goya_init_golden_registers() has ever written:
         0xD0308 MME_SBA  MAX_SIZE [9:0] max 1023, EU_MAX_CREDIT [20:16] max 31
         0xD030C MME_SBB  same
         0xD0310 MME_SBC  same
         0xD0314 MME_WBC  MAX_OUTSTANDING [11:0] max 4095,
                          DISABLE_FAST_END_PE [12], LD_INSERT_BUBBLE_DIS [13]
       MAX_SIZE is how much the buffer can HOLD, as opposed to the AGU's request
       credits (already at their reset values and proven not to be the limiter).
       A small SB cannot retain operands across steps, forcing a memory access per
       accumulate step -- which is the measured ~145 ns/step. Vendor MME_WBC=0x3200
       decodes to only 512 outstanding with both bubble bits set. */
    if (e_credits >= 3) {
        uint32_t sb = (1023u) | (31u << 16);
        o = w(g->fd, cfg, o, 0xD0308, sb, 0);              /* SBA */
        o = w(g->fd, cfg, o, 0xD030C, sb, 0);              /* SBB */
        o = w(g->fd, cfg, o, 0xD0310, sb, 0);              /* SBC */
        wbc   = 0xFFFu | (1u << 12) | (1u << 13);          /* 4095 outstanding */
        store = 0x3Fu;
        agu   = (31u) | (31u << 8) | (31u << 16) | (63u << 24);
    }
    o = w(g->fd, cfg, o, 0xD0300, store, 0);
    o = w(g->fd, cfg, o, 0xD0304, agu,   0);
    o = w(g->fd, cfg, o, 0xD0314, wbc,   0);
    o = w(g->fd, cfg, o, 0xD0324, wbcd,  0);
    return o;
}

static uint32_t desc(gmme *g, void *cfg, uint32_t o,
                     uint64_t Abase, uint64_t Cbase, int rows, int K, int N, int first) {
    uint32_t b; int i;
    exp_init();
    if (e_vdesc) {
        /* Verbatim vendor descriptor with our A/B/C substituted. */
        int q;
        uint64_t Bb = (g->cur_slice >= 0)
            ? (g->W + (uint64_t)g->slice[g->cur_slice].card_slot * g->wslot) : g->B;
        uint32_t bulkvals[VDESC_N];
        for (q = 0; q < VDESC_N; q++) {
            uint32_t reg = VDESC[q].reg, val = VDESC[q].val;
            switch (reg) {
            case 0xD0008: val = (uint32_t)(Abase >> 32); break;
            case 0xD001C: val = (uint32_t)Abase;         break;
            case 0xD000C: val = (uint32_t)(Bb    >> 32); break;
            case 0xD0020: val = (uint32_t)Bb;            break;
            case 0xD0014: val = (uint32_t)(Cbase >> 32); break;
            case 0xD0028: val = (uint32_t)Cbase;         break;
            default: break;
            }
            { uint32_t ov; if (vreg_override(reg, &ov)) val = ov; }
            bulkvals[q] = val;
            if (!e_bulk) o = w(g->fd, cfg, o, reg, val, 0);
        }
        if (e_bulk)   /* one bulk write for the whole contiguous 0xD0008..0xD01AC block */
            o = wbulk(cfg, o, VDESC[0].reg, bulkvals, VDESC_N, 0);
        o = w(g->fd, cfg, o, 0xD0200, 1, 1);   /* EXECUTE, EB=1 mandatory on gen-1 */
        return o;
    }

    if (e_spmode) {
        /* ---- SPATIAL form, modelled on the captured vendor descriptor ------ */
        uint32_t b2;
        /* GMME_FULLM=1: declare the WHOLE tensor in ONE descriptor instead of one
           per 256 rows. Live descriptor diff against SynapseAI at M=8192 K=2048
           N=256 showed the vendor carries A_VALID_ELEMENTS_1 = K*M (0x01000000 =
           16,777,216, exactly M*K) and C_VALID_ELEMENTS_1 = N*M (0x00200000), with
           SPATIAL_SIZE_MINUS_1 still 255 and E_NUM_ITERATION_MINUS_1 = 10. We
           carried K*rows / N*rows and issued 32 descriptors, each re-reading the
           whole KxN weight block (measured: cost proportional to K, ~2.8 GB/s).
           EXPERIMENTAL -- correctness is the oracle, not the register match. */
        static int e_fullm = -1;
        uint32_t RF, SPN, EIT;
        if (e_fullm < 0) { const char *fm = getenv("GMME_FULLM"); e_fullm = fm ? atoi(fm) : 0; }
        {   /* NOT `cur_M > rows`: under FULLM the caller sets mt = Mp so rows ==
               cur_M, which would make that test false and silently fall back to
               rows-1 in an 8-bit field. Gate on the mode and the shape only. */
            int fm = e_fullm && g->cur_M >= 256 && (g->cur_M % 256) == 0;
            RF  = fm ? (uint32_t)g->cur_M : (uint32_t)rows;
            SPN = fm ? 255u : (uint32_t)(rows - 1);
            EIT = fm ? (uint32_t)(RF / 256u - 1u) : g->eiter;
        }
        o=w(g->fd,cfg,o,0xD0008,(uint32_t)(Abase>>32),0); o=w(g->fd,cfg,o,0xD001C,(uint32_t)Abase,0);
        {   uint64_t Bb = (g->cur_slice >= 0)
                ? (g->W + (uint64_t)g->slice[g->cur_slice].card_slot * g->wslot)
                : (g->B + g->b_off_cur);
            o=w(g->fd,cfg,o,0xD000C,(uint32_t)(Bb>>32),0); o=w(g->fd,cfg,o,0xD0020,(uint32_t)Bb,0); }
        o=w(g->fd,cfg,o,0xD0014,(uint32_t)(Cbase>>32),0); o=w(g->fd,cfg,o,0xD0028,(uint32_t)Cbase,0);
        /* LOAD_CIN reads the existing C as the accumulator seed, so a K-tiled
           GEMM can do C += A_tile @ B_tile. CIN and COUT share the C block on
           Goya (separate bases, one set of strides), so CIN base = C base. */
        o=w(g->fd,cfg,o,0xD0010,acc_cin?(uint32_t)(Cbase>>32):0u,0);
        o=w(g->fd,cfg,o,0xD0024,acc_cin?(uint32_t)Cbase:0u,0);
        o=w(g->fd,cfg,o,0xD0018,0,0); o=w(g->fd,cfg,o,0xD002C,0,0);
        {   uint32_t h = e_hdr ? e_hdr : SPHDR;
            if (acc_cin) h |= (1u << 13);            /* LOAD_CIN */
            o=w(g->fd,cfg,o,0xD0030, h, 0); }
        o=w(g->fd,cfg,o,0xD0034,0,0);           /* NO conv loops */
        o=w(g->fd,cfg,o,0xD0038,0,0);           /* NO loop-to-dim bindings */
        o=w(g->fd,cfg,o,0xD003C,0,0);
        o=w(g->fd,cfg,o,0xD0040,0,0); o=w(g->fd,cfg,o,0xD0044,0,0);
        o=w(g->fd,cfg,o,0xD0048,0,0); o=w(g->fd,cfg,o,0xD004C,0,0);

        b2 = 0xD0050;                            /* A: rows x K, spatial over rows */
        for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x00+4*i,0,0);
        for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x14+4*i,i==0?(uint32_t)K:(i==1?(uint32_t)(K*RF):1u),0);
        for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x28+4*i,
            i==1?(uint32_t)K:0u,0);   /* LOOP_STRIDE_1 = K: the row advance (vendor 0x100) */
        for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x3C+4*i,
            i==0?(uint32_t)K:(i==1?(uint32_t)(K*RF):1u),0);   /* ROI_SIZE_0=K, ROI_SIZE_1=K*RF (vendor 0x10000) */
        for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x4C+4*i,0,0);
        for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x5C+4*i,i==0?(uint32_t)K:1u,0);   /* spatial step = one row */
        o=w(g->fd,cfg,o,b2+0x6C,SPN,0);                                        /* rows spatial steps */

        b2 = 0xD00C0;                            /* B: K x N */
        if (e_bconv == 3) {
            /* DERIVED from the vendor, 2026-09-02, with pass-through weights:
                 B_VALID_ELEMENTS_0 = K        B_ROI_SIZE_0       = K
                 B_VALID_ELEMENTS_1 = K*N      B_ROI_SIZE_1       = K*N
                 B_LOOP_STRIDE_0    = K        B_SPATIAL_STRIDE_0 = K
                 B_SPATIAL_SIZE_MINUS_1 = N-1
               B is stored TRANSPOSED, B[n][k], row length K -- proven by stamping
               each weight with its own k or n index and reading the buffer back:
               k stamps came out sequential, n stamps came out all zero. */
            /* Derived from pass-through vendor runs (WQUANT=1), 2026-09-02:
                 B_VE0 = K              (valid bytes per row)
                 B_VE1 = N * 256        (N rows, each PADDED to 256)
                 LOOP_STRIDE_0 = ROI_SIZE_0 = SPATIAL_STRIDE_0 = 256  (constant,
                   independent of K -- that constancy is what revealed the pad)
                 SPATIAL_SIZE_MINUS_1 = N-1
               B is stored TRANSPOSED as B[n][k] with a 256-byte row pitch. */
            {   /* GMME_BRAW uploads B ROW-MAJOR as B[k][n] with a 256-byte pitch,
                   so the extents are K rows of N valid elements. Without BRAW the
                   buffer is transposed B[n][k]: N rows of K. These coincide only at
                   K == N == 256, which is why every other shape stalled. */
                uint32_t v0   = e_braw ? (uint32_t)N        : (uint32_t)K;
                uint32_t v1   = e_braw ? (uint32_t)(K*256)  : (uint32_t)(N*256);
                uint32_t nsp  = e_braw ? (uint32_t)(K-1)    : (uint32_t)(N-1);
                for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x00+4*i,0,0);
                for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x14+4*i,i==0?v0:(i==1?v1:1u),0);
                for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x28+4*i,i==0?256u:0u,0);
                for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x3C+4*i,i==0?256u:(i==1?v1:1u),0);
                for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x4C+4*i,0,0);
                for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x5C+4*i,i==0?256u:1u,0);
                o=w(g->fd,cfg,o,b2+0x6C,nsp,0);
            }
        } else if (e_bconv == 2) {
            /* B given the SAME STRUCTURE as the C block, with K where C has rows.
               Brad's idea: C works spatially, so paste C's shape onto B and swap
               the row-extent for the K-extent. B participates in the shared
               spatial loop (so it does not stall) but with stride 0 it does not
               move, because every output row multiplies the same K x N weights.
               ROI_SIZE_1 = N*K so each step may read the whole of B. */
            for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x00+4*i,0,0);
            for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x14+4*i,i==0?(uint32_t)N:(i==1?(uint32_t)(N*K):1u),0);
            for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x28+4*i,i==0?(uint32_t)N:0u,0);
            for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x3C+4*i,i==0?(uint32_t)N:(i==1?(uint32_t)(N*K):1u),0);
            for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x4C+4*i,0,0);
            for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x5C+4*i,
                i==0?(uint32_t)(e_bstr>=0?e_bstr:0):1u,0);
            o=w(g->fd,cfg,o,b2+0x6C,(uint32_t)e_bspat,0);
        } else if (e_bconv) {                    /* the proven conv-form B, verbatim */
            for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x00+4*i,0,0);
            for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x14+4*i,i==0?(uint32_t)N:(uint32_t)(N*K),0);
            for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x28+4*i,i==1?(uint32_t)N:0u,0);
            for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x3C+4*i,i==0?(uint32_t)N:(i==1?(uint32_t)K:1u),0);
            for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x4C+4*i,0,0);
            for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x5C+4*i,
                i==0?(uint32_t)(e_bstr>=0?e_bstr:1):1u,0);
            o=w(g->fd,cfg,o,b2+0x6C,(uint32_t)e_bspat,0);
        } else {
            for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x00+4*i,0,0);
            for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x14+4*i,i==0?(uint32_t)N:(i==1?(uint32_t)(N*K):1u),0);
            for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x28+4*i,i==0?(uint32_t)N:0u,0);
            for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x3C+4*i,i==0?(uint32_t)N:(i==1?(uint32_t)(N*K):1u),0);
            for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x4C+4*i,0,0);
            for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x5C+4*i,
                i==0?(uint32_t)(e_bstr>=0?e_bstr:0):1u,0);
            o=w(g->fd,cfg,o,b2+0x6C,(uint32_t)e_bspat,0);
        }

        b2 = 0xD0130;                            /* C: rows x N, spatial over rows */
        for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x00+4*i,0,0);
        for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x14+4*i,i==0?(uint32_t)N:(i==1?(uint32_t)(N*RF):1u),0);
        for(i=0;i<5;i++) o=w(g->fd,cfg,o,b2+0x28+4*i,i==0?(uint32_t)N:0u,0);   /* stride on _0 like vendor */
        for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x3C+4*i,i==0?(uint32_t)N:(i==1?(uint32_t)(N*RF):1u),0);
        for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x4C+4*i,0,0);
        for(i=0;i<4;i++) o=w(g->fd,cfg,o,b2+0x5C+4*i,i==0?(uint32_t)N:1u,0);
        o=w(g->fd,cfg,o,b2+0x6C,SPN,0);

        /* SYNC_OBJECT_MESSAGE. The vendor writes 0x80000001 here and sets
           HEADER bit 5 (SIGNAL_EN); we wrote 0 and left SIGNAL_EN clear. Without
           it the engine completes the work but never signals, so the settle loop
           spins its full 20000 polls x 83 us = 1.66 s -- exactly the 1651-1661 ms
           "stall" every spatial run showed. GMME_SIG=0 restores the old behaviour. */
        {   const char *sg = getenv("GMME_SIG");
            uint32_t sv = sg ? (uint32_t)strtoul(sg,NULL,0) : 0x80000001u;
            o=w(g->fd,cfg,o,0xD01A0,sv,0); }
        o=w(g->fd,cfg,o,0xD01A4,0,0);
        o=w(g->fd,cfg,o,0xD01A8,EIT,0);
        o=w(g->fd,cfg,o,0xD01AC,g->bubbles,0);
        o=w(g->fd,cfg,o,0xD0200,1,1);            /* EXECUTE, EB=1 mandatory on gen-1 */
        return o;
    }


    /* SOUND-AND-TRACE ROOT CHECK: GMME_SWAPA=1 points the A operand at the B
       buffer -- a valid, mapped device address holding the WRONG data. If the
       GEMM is still bit-exact under this, the engine is not consuming our
       descriptor at all and every field-level result is meaningless. */
    {   uint64_t Ab = e_swapa ? g->B : Abase;
        o=w(g->fd,cfg,o,0xD0008,(uint32_t)(Ab>>32),0); o=w(g->fd,cfg,o,0xD001C,(uint32_t)Ab,0); }
    {   uint64_t Bb = (g->cur_slice >= 0) ? (g->W + (uint64_t)g->slice[g->cur_slice].card_slot * g->wslot) : g->B;
        o=w(g->fd,cfg,o,0xD000C,(uint32_t)(Bb>>32),0); o=w(g->fd,cfg,o,0xD0020,(uint32_t)Bb,0); }
    o=w(g->fd,cfg,o,0xD0014,(uint32_t)(Cbase>>32),0); o=w(g->fd,cfg,o,0xD0028,(uint32_t)Cbase,0);
    o=w(g->fd,cfg,o,0xD0010,0,0); o=w(g->fd,cfg,o,0xD0024,0,0);
    o=w(g->fd,cfg,o,0xD0018,0,0); o=w(g->fd,cfg,o,0xD002C,0,0);
    /* GMME_SPARSE=1: of the 107 WREG32 writes in a descriptor, only the A base,
       the C base and the trigger differ between tiles of one GEMM. The shapes,
       strides, ASSOC and header are identical every fire, yet we rewrite all of
       them -- and the manual says dispatch, not arithmetic, is what we measure.
       Skipping the invariant writes on every tile after the first drops dispatch
       from 107 writes to 5. If MME_ARCH latches per fire this stays bit-exact;
       if the engine needs them re-presented, EXACT goes False and the idea dies. */
    if (!e_sparse || first) {
    {   uint32_t hdr = e_hdr ? e_hdr : HDR;
        if (e_accum >= 0) hdr = (hdr & ~0xF00u) | (((uint32_t)e_accum & 0xF) << 8);
        if (e_signal || e_fence) hdr |= 0x20u;   /* SIGNAL_EN */
        o=w(g->fd,cfg,o,0xD0030,hdr,0); }
    /* SOUND-AND-TRACE (2026-09-01): KERNEL_SIZE_MINUS_1 normally tracks the data
       geometry exactly. GMME_DIM0/GMME_DIM1 decouple the LOOP COUNT from the
       tensors, so we can see whether time follows the LOOP or the DATA VOLUME.
       Only ever set SMALLER than the real extent -- larger would read OOB.
       Answers are expected WRONG under these: we are tracing the clock. */
    {   uint32_t d0 = e_dim0 ? (uint32_t)e_dim0
                    : (e_aw ? (uint32_t)(K / e_aw) : (uint32_t)K);
        uint32_t d1 = e_dim1 ? (uint32_t)e_dim1 : (uint32_t)rows;
        o=w(g->fd,cfg,o,0xD0034,PZ(4096, (((d1-1)&0xff)<<8)|((d0-1)&0xff)),0); }
    o=w(g->fd,cfg,o,0xD0038,PZ(8192, e_assoc ? e_assoc : ASSOC0),0); o=w(g->fd,cfg,o,0xD003C,0,0);
    o=w(g->fd,cfg,o,0xD0040,0,0); o=w(g->fd,cfg,o,0xD0044,0,0);
    o=w(g->fd,cfg,o,0xD0048,0,0); o=w(g->fd,cfg,o,0xD004C,0,0);

    b = 0xD0050;                                   /* A: rows x K, int8 */
    for(i=0;i<5;i++) o=w(g->fd,cfg,o,b+0x00+4*i,0,0);
    /* SOUND-AND-TRACE: the canonical field map says loop counts are DERIVED,
       not declared -- DIM_0 = A_VALID_ELEMENTS_1 - 1. This field is the only
       A-side term carrying a `rows` factor, so if the K-loop bound comes from
       here it runs K*rows times instead of K, giving the measured K*rows^2.
       GMME_AVE=1 drops the rows factor to test exactly that. */
    for(i=0;i<5;i++) o=w(g->fd,cfg,o,b+0x14+4*i,
        PZ(1, i==0?(uint32_t)K
            : ((e_avb&1) && i>=2) ? 1u
            : (e_ave?(uint32_t)K:(uint32_t)(K*rows))),0);
    for(i=0;i<5;i++) o=w(g->fd,cfg,o,b+0x28+4*i,
        PZ(2, i==0?((e_avb&4)?0u:(e_aw?(uint32_t)e_aw:1u)):(i==1?(uint32_t)K:0u)),0);
    for(i=0;i<4;i++) o=w(g->fd,cfg,o,b+0x3C+4*i,
        PZ(4, (i==0&&e_aw)?(uint32_t)e_aw
            : (i==0&&e_aroi)?(uint32_t)e_aroi
            : ((e_avb&2) && i==1)?(uint32_t)(K*rows) : 1u),0);
    for(i=0;i<4;i++) o=w(g->fd,cfg,o,b+0x4C+4*i,0,0);
    for(i=0;i<4;i++) o=w(g->fd,cfg,o,b+0x5C+4*i,PZ(8, i==0?(uint32_t)K:1u),0);
    o=w(g->fd,cfg,o,b+0x6C,e_spatial?(uint32_t)(rows-1):0u,0);

    b = 0xD00C0;                                   /* B: K x N, int8 */
    for(i=0;i<5;i++) o=w(g->fd,cfg,o,b+0x00+4*i,0,0);
    for(i=0;i<5;i++) o=w(g->fd,cfg,o,b+0x14+4*i,PZ(16, i==0?(uint32_t)N:(uint32_t)(N*K)),0);
    for(i=0;i<5;i++) o=w(g->fd,cfg,o,b+0x28+4*i,PZ(32, i==1?(uint32_t)N:0u),0);
    for(i=0;i<4;i++) o=w(g->fd,cfg,o,b+0x3C+4*i,PZ(64, i==0?(uint32_t)N
        : (i==1 ? ((e_avb&8)?(uint32_t)(K*N):(uint32_t)K) : 1u)),0);
    for(i=0;i<4;i++) o=w(g->fd,cfg,o,b+0x4C+4*i,0,0);
    for(i=0;i<4;i++) o=w(g->fd,cfg,o,b+0x5C+4*i,PZ(128,1),0);
    o=w(g->fd,cfg,o,b+0x6C,0,0);

    b = 0xD0130;                                   /* C: rows x N, int32 */
    for(i=0;i<5;i++) o=w(g->fd,cfg,o,b+0x00+4*i,0,0);
    for(i=0;i<5;i++) o=w(g->fd,cfg,o,b+0x14+4*i,PZ(256, i==0?(uint32_t)N:(uint32_t)(N*rows)),0);
    for(i=0;i<5;i++) o=w(g->fd,cfg,o,b+0x28+4*i,PZ(512, i==1?(uint32_t)N:0u),0);
    for(i=0;i<4;i++) o=w(g->fd,cfg,o,b+0x3C+4*i,PZ(1024, i==0?(uint32_t)N:1u),0);
    for(i=0;i<4;i++) o=w(g->fd,cfg,o,b+0x4C+4*i,0,0);
    for(i=0;i<4;i++) o=w(g->fd,cfg,o,b+0x5C+4*i,PZ(2048, i==0?(uint32_t)N:1u),0);
    o=w(g->fd,cfg,o,b+0x6C,e_spatial?(uint32_t)(rows-1):0u,0);

    }
    o=w(g->fd,cfg,o,0xD01A0,
        (e_signal || e_fence)
            ? (1u | (((uint32_t)e_sob * 4u) << 16) | 0x80000000u)
            : 0u, 0);
    o=w(g->fd,cfg,o,0xD01A4,0,0);
    o=w(g->fd,cfg,o,0xD01A8,g->eiter,0);   /* E_NUM_ITERATION_MINUS_1: the spatial loop */
    o=w(g->fd,cfg,o,0xD01AC,g->bubbles,0); /* E_BUBBLES_PER_SPLIT: pipeline spacing */
    o=w(g->fd,cfg,o,0xD0200,1,1);          /* EXECUTE. EB=1 is mandatory on gen-1. */
    return o;
}

/* GMME_PROF=1 prints a per-step breakdown. Same scaffold as the TPC path;
   this is what found the per-fire CB allocation. */
#include <time.h>
static int mprof = -1;
static struct timespec mp_t0;
static void PB(void){ if(mprof<0) mprof = getenv("GMME_PROF")?1:0;
                      if(mprof) clock_gettime(CLOCK_MONOTONIC,&mp_t0); }
static void PE(const char *lbl){ struct timespec e; if(!mprof) return;
    clock_gettime(CLOCK_MONOTONIC,&e);
    fprintf(stderr,"    %-26s %8.3f ms\n", lbl,
        (e.tv_sec-mp_t0.tv_sec)*1e3+(e.tv_nsec-mp_t0.tv_nsec)/1e6); }

/* Submit and return immediately; caller waits on *seq then frees *dma. */
/* ONE command submission spanning N engine queues. hltests_submit_cs takes an
   ARRAY of chunks, each with its own queue index -- so eight TPC cores are one
   submission, not eight. Issuing one CS per core was costing ~0.07 ms each and
   made a 458-thousand-add fold slower than 134 million MACs of GEMM. */
int gmme_submit_multiq(gmme *g, void *const *eng, const uint32_t *eo,
                       const uint32_t *q, int n, uint64_t *seq, void **dma)
{
    struct hltests_cs_chunk ex[9];
    uint32_t dof; int i;
    if (!g || g->fd < 0 || n < 1 || n > 8) return -1;
    *dma = hltests_create_cb(g->fd, 0x1000, EXTERNAL, 0);
    if (!*dma) return -1;
    dof = hltests_add_nop_pkt(g->fd, *dma, 0, EB_FALSE, MB_TRUE);
    memset(ex, 0, sizeof ex);
    for (i = 0; i < n; i++) {
        ex[i].cb_ptr = eng[i]; ex[i].cb_size = eo[i]; ex[i].queue_index = q[i];
    }
    ex[n].cb_ptr = *dma; ex[n].cb_size = dof; ex[n].queue_index = g->dq;
    return hltests_submit_cs(g->fd, NULL, 0, ex, n + 1, 0, seq);
}

int gmme_submit_async(gmme *g, void *eng, uint32_t eo, uint32_t q,
                      uint64_t *seq, void **dma)
{
    uint32_t dof; struct hltests_cs_chunk ex[2];
    *dma = hltests_create_cb(g->fd, 0x1000, EXTERNAL, 0);
    if (!*dma) return -1;
    dof = hltests_add_nop_pkt(g->fd, *dma, 0, EB_FALSE, MB_TRUE);
    memset(ex,0,sizeof ex);
    ex[0].cb_ptr=eng;  ex[0].cb_size=eo;  ex[0].queue_index=q;
    ex[1].cb_ptr=*dma; ex[1].cb_size=dof; ex[1].queue_index=g->dq;
    return hltests_submit_cs(g->fd, NULL, 0, ex, 2, 0, seq);
}

int gmme_wait_all(gmme *g, uint64_t *seq, void **dma, int n)
{
    int i, rc = 0;
    for (i = 0; i < n; i++) {
        if (hltests_wait_for_cs(g->fd, seq[i], 5000000) != HL_WAIT_CS_STATUS_COMPLETED)
            rc = -1;
        hltests_destroy_cb(g->fd, dma[i]);
    }
    return rc;
}


/* Per-ENGINE idle, instead of hlthunk_is_device_idle()'s device-WIDE answer.
 * HL_INFO_HW_IDLE returns busy_engines_mask; GOYA_ENGINE_ID_MME_0 is bit 5
 * (DMA_0..4 = 0..4, MME_0 = 5, TPC_0..7 = 6..13). The aggregate call reports
 * busy while any DMA is still winding down, so we spun ~83-96 times waiting for
 * engines we do not care about. Sourced from our own corpus note (CAPABILITIES.md:
 * "35 HL_INFO sub-opcodes catalogued ... our wrapper code should expose all of
 * these"), which we had and never used.
 *
 * NOTE: this narrows WHAT we wait for, it does not skip waiting. The device-wide
 * settle still runs before release -- that is the thermal requirement. */
#define GOYA_ENGINE_BIT_MME0 5u
static int mme_is_idle(int fd)
{
    struct hl_info_args args;
    struct hl_info_hw_idle idle;
    memset(&args, 0, sizeof args);
    memset(&idle, 0, sizeof idle);
    args.op = HL_INFO_HW_IDLE;
    args.return_pointer = (__u64)(uintptr_t)&idle;
    args.return_size = (__u32)sizeof idle;
    if (hlthunk_get_info(fd, &args))
        return -1;                       /* fall back to the device-wide poll */
    return (idle.busy_engines_mask & (1u << GOYA_ENGINE_BIT_MME0)) ? 0 : 1;
}

/* Poll until the ENGINE settles. Device-WIDE, so it belongs once per batch,
   never once per operation. Releasing a busy device soft-resets the card. */
int gmme_settle(gmme *g)
{
    int t;
    /* MEASURED 2026-09-01: hlthunk_is_device_idle costs 83 us per call on a
       QUIET device -- 2.6x the entire CS wait (32 us). 44 polls = 3.65 ms of
       pure ioctl, which was 92% of every GEMM. The poll cannot resolve when the
       device went idle because each look is slower than the work.
       GMME_NO_SETTLE=1 skips it; submit+wait already blocks on CS completion. */
    static int nosettle = -1;
    if (nosettle < 0) nosettle = getenv("GMME_NO_SETTLE") ? 1 : 0;
    if (nosettle) { g->idle = 1; return 0; }
    for (t = 0; t < 20000; t++) { g->idle = hlthunk_is_device_idle(g->fd); if (g->idle) break; }
    if (t > g->idle_worst) g->idle_worst = t;
    return g->idle ? 0 : -1;
}

/* Submit one cfg CB (which may hold many tiles) as a single command submission. */
static int submit(gmme *g, void *cfg, uint32_t o, int tiles) {
    uint32_t eo, dof; int rc; uint64_t sq;

    if (!g->cb_meng) {
        g->cb_meng = hltests_create_cb(g->fd, 0x1000, INTERNAL, g->sram + ENGCB);
        if (!g->cb_meng) { snprintf(g->err,sizeof g->err,"create eng cb"); return -1; }
    }
    void *eng = g->cb_meng;
    struct hltests_pkt_info p; memset(&p,0,sizeof p);
    eo = 0;
    if (e_fence) {   /* zero the sync object before the engine bumps it */
        struct hltests_pkt_info sp; memset(&sp,0,sizeof sp);
        sp.eb = EB_FALSE; sp.mb = MB_TRUE;
        sp.write_to_sob.sob_id = (uint16_t)e_sob;
        sp.write_to_sob.value  = 0;
        sp.write_to_sob.mode   = SOB_SET;
        eo = hltests_add_write_to_sob_pkt(g->fd, eng, eo, &sp);
    }
    p.eb=EB_FALSE; p.mb=MB_TRUE; p.cp_dma.src_addr=g->sram+CFGCB; p.cp_dma.size=o;
    eo = hltests_add_cp_dma_pkt(g->fd, eng, eo, &p);
    if (!g->cb_mdma) {
        g->cb_mdma = hltests_create_cb(g->fd, 0x1000, EXTERNAL, 0);
        if (!g->cb_mdma) { snprintf(g->err,sizeof g->err,"create dma cb"); return -1; }
    }
    void *dma = g->cb_mdma;   /* cached; packets are rewritten from offset 0 */
    struct hltests_pkt_info np; memset(&np,0,sizeof np); np.eb=EB_FALSE; np.mb=MB_TRUE;
    dof = 0;
    if (e_fence) {   /* the CS cannot retire until the MME has signalled */
        struct hltests_monitor_and_fence mf; memset(&mf,0,sizeof mf);
        mf.queue_id    = (uint8_t)g->dq;
        mf.cmdq_fence  = false;
        mf.dec_fence   = true;
        mf.sob_id      = (uint16_t)e_sob;
        mf.mon_id      = 0;
        mf.sob_val     = (uint16_t)(e_sobval ? e_sobval : (e_spmode ? tiles : tiles * 8));
        mf.mon_address = 0;
        mf.mon_payload = 1;
        dof = hltests_add_monitor_and_fence(g->fd, dma, dof, &mf);
    }
    dof = hltests_add_nop_pkt(g->fd, dma, dof, EB_FALSE, MB_TRUE);

PB();
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+CFGCB, o, HL_DMA_HOST_TO_SRAM);
PE("submit: cfg->SRAM");
    /* The engine CB is a CP_DMA (plus an optional SOB pre-zero) whose LENGTH is the
       cfg-CB size o -- so it changes whenever the tile count changes, even though the
       engine CB's own size eo does not. Keying the cache on eo alone meant a 1-tile
       call left a short CP_DMA staged, and the next multi-tile call copied only the
       first descriptor: rows past the first tile were never computed and read back as
       untouched device memory. Key on BOTH. */
    /* Belt and braces after the 0x700000 collision: any engine that stages into
       SRAM behind our back invalidates this cache. The offsets no longer
       overlap, but a silent stale-CB execution is expensive enough to find
       twice that the guard stays. */
    if (g->eng_staged != eo || g->eng_staged_o != o || g->sram_dirty) {
        g->sram_dirty = 0;
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+ENGCB, eo, HL_DMA_HOST_TO_SRAM);
        g->eng_staged = eo;
        g->eng_staged_o = o;
    }

    struct hltests_cs_chunk ex[2]; memset(ex,0,sizeof ex);
    ex[0].cb_ptr=eng; ex[0].cb_size=eo;  ex[0].queue_index=g->mq;
    ex[1].cb_ptr=dma; ex[1].cb_size=dof; ex[1].queue_index=g->dq;
    struct gmme_dbg_spmu spin_; __u64 spout_[5]; int sparmed_ = 0;
    if (e_spmu >= 0) {
        memset(&spin_,0,sizeof spin_); spin_.n = 3;
        spin_.ev[0]=0x0; spin_.ev[1]=0x1; spin_.ev[2]=0x2;
        memset(spout_,0,sizeof spout_);
        sparmed_ = (gmme_dbg(g->fd, HL_DEBUG_OP_SPMU, (__u32)e_spmu, 1,
                             &spin_, sizeof spin_, spout_, sizeof spout_) == 0);
        if (!sparmed_) fprintf(stderr,"      [spmu %d ARM FAILED: %s]\n", e_spmu, strerror(errno));
    }
PB();
    rc = hltests_submit_cs(g->fd, NULL, 0, ex, 2, 0, &sq);
    if (!rc) rc = hltests_wait_for_cs(g->fd, sq, 5000000);
PE("submit: submit+wait");
    /* wait_for_cs means the QUEUE drained, not that the ENGINE finished. Poll until
       settled instead of sampling once -- a single sample is a racy oracle. */
    /* Poll until the ENGINE settles, not just the queue. This is not optional:
     * the driver soft-resets the card on release if the device is not idle, so
     * skipping it costs real wear. A GMME_NOIDLE escape hatch used to live here;
     * it produced both wrong answers and 12 soft resets on hl2, and is gone. */
PB();
    /* GMME_PROBE_US=N: sleep N us, then poll ONCE, to separate real engine
       time from the cost of the poll itself (82 us per ioctl, measured).
       The full settle loop still runs afterwards -- releasing a non-idle
       device makes the driver soft-reset it. */
    if (getenv("GMME_PROBE_US")) {
        int us = atoi(getenv("GMME_PROBE_US"));
        struct timespec ts; ts.tv_sec = us/1000000; ts.tv_nsec = (us%1000000)*1000L;
        nanosleep(&ts, NULL);
        fprintf(stderr,"      [after %6d us: %s]\n", us,
                hlthunk_is_device_idle(g->fd) ? "IDLE" : "busy");
    }
    {   int tries;
        static int cap = -1;
        /* GMME_SPINCAP: this settle sits BEFORE the C readback, so nothing has
           drained the queue yet. In the spatial form the device never reports
           idle here and we burn the full 20000 polls x 83 us = 1.66 s -- which is
           the entire apparent "stall" (work itself is 0.032 ms). A low cap lets
           the readback DMA, which is its own waiting CS, do the draining. */
        if (cap < 0) { const char *ec = getenv("GMME_SPINCAP"); cap = ec ? atoi(ec) : 20000; }
        /* In a chain the next op is another device-side fire; the engine only
           has to be settled before the RESULT leaves the card. Paying a
           device-idle poll per matmul is the single largest per-op cost. */
        /* NEVER mutate the static: chain mode once set cap=0 for the rest
           of the process, so every UNCHAINED fire after the first chain
           (gmme_apply_batch) skipped its settle and read C back one fire
           behind -- the batch answered with the previous op's output.
           Found 2026-09-03: a batch call returned the previous op's result
           when it followed a chained call in the same process. */
        int spins = in_chain ? 0 : cap;
        for (tries = 0; tries < spins; tries++) {
            if (e_mmeidle) {
                int mi = mme_is_idle(g->fd);
                if (mi >= 0) { g->idle = mi; if (g->idle) break; continue; }
            }
            g->idle = hlthunk_is_device_idle(g->fd);
            if (g->idle) break;
        }
        if (tries > g->idle_worst) g->idle_worst = tries;
        if (mprof) fprintf(stderr,"      [spun %d ioctls]\n", tries);
    }
PE("submit: settle spin");
    if (e_spmu >= 0 && sparmed_) {
        memset(spout_,0,sizeof spout_);
        if (gmme_dbg(g->fd, HL_DEBUG_OP_SPMU, (__u32)e_spmu, 0,
                     &spin_, sizeof spin_, spout_, sizeof spout_))
            fprintf(stderr, "      [spmu %d READ FAILED: %s]\n", e_spmu, strerror(errno));
        else
            fprintf(stderr, "      [spmu%d tiles=%d ev0=%llu ev1=%llu ev2=%llu CYCLES=%llu]\n",
                e_spmu, tiles,
                (unsigned long long)spout_[0], (unsigned long long)spout_[1],
                (unsigned long long)spout_[2], (unsigned long long)spout_[4]);
    }
    g->fires += tiles;
    /* dma CB is cached in g->cb_mdma; freed in gmme_close */
    if (rc) { snprintf(g->err,sizeof g->err,"submit/wait rc=%d",rc); return -2; }
    return 0;
}

/* Upload B once and leave it on the card. For inference B is a weight matrix
   that does not change between tokens; re-sending it over PCIe every call puts
   the bus in the critical path of every op. Pass K,N exactly as the later
   gmme_gemm_i8 calls will. */
/* Make room for a slice of `need` bytes. If it does not fit the current slot
   size, the arena is re-partitioned at the larger size. That invalidates every
   card residency -- slot i is at a different address once the stride changes --
   so everything is spilled. This is CHEAP AND SAFE precisely because a slice
   always keeps its host copy: spilling is forgetting, and each slice re-promotes
   on its next use with one DMA. Returns -1 only if a single slice cannot fit the
   whole arena. */
static int arena_fit(gmme *g, uint64_t need)
{
    int i;
    need = (need + (WSLOT_MIN - 1)) & ~(uint64_t)(WSLOT_MIN - 1);
    if (need <= g->wslot) return 0;
    if (need > g->warena_size) {
        snprintf(g->err, sizeof g->err,
                 "slice needs %llu B but the whole arena is %llu B (raise GMME_ARENA_MB)",
                 (unsigned long long)need, (unsigned long long)g->warena_size);
        return -1;
    }
    g->wslot = need;
    g->n_card_slots = (int)(g->warena_size / g->wslot);
    for (i = 0; i < g->n_slices; i++) g->slice[i].card_slot = -1;
    g->n_card = 0;
    g->cur_slice = -1;          /* its card_slot is now -1; never index the arena with it */
    return 0;
}

/* The slice table grows on demand; there is no fixed name limit any more. */
static int slice_table_room(gmme *g)
{
    struct gmme_slice *n;
    int cap;
    if (g->n_slices < g->slice_cap) return 0;
    cap = g->slice_cap ? g->slice_cap * 2 : 256;
    n = realloc(g->slice, (size_t)cap * sizeof *n);
    if (!n) { snprintf(g->err, sizeof g->err, "slice table alloc failed"); return -1; }
    memset(n + g->slice_cap, 0, (size_t)(cap - g->slice_cap) * sizeof *n);
    g->slice = n; g->slice_cap = cap;
    return 0;
}

/* Bring a slice onto the card, evicting the least recently used one if the
   arena is full. Weights are read-only so eviction writes nothing back. */
static int promote(gmme *g, int idx)
{
    int victim = -1, i; uint64_t oldest = ~0ull;
    if (g->slice[idx].card_slot >= 0) { g->slice[idx].used = ++g->clock; return 0; }
    if (g->n_card < g->n_card_slots) {
        g->slice[idx].card_slot = g->n_card++;
    } else {
        for (i = 0; i < g->n_slices; i++)
            if (g->slice[i].card_slot >= 0 && g->slice[i].used < oldest) {
                oldest = g->slice[i].used; victim = i; }
        if (victim < 0) { snprintf(g->err,sizeof g->err,"no slice to evict"); return -1; }
        g->slice[idx].card_slot = g->slice[victim].card_slot;
        g->slice[victim].card_slot = -1;          /* spilled: host copy is the truth */
        g->spills++;
    }
    memcpy(g->hb, g->slice[idx].host, (size_t)g->slice[idx].Kp * g->slice[idx].Np);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
        g->W + (uint64_t)g->slice[idx].card_slot * g->wslot,
        (uint32_t)((size_t)g->slice[idx].Kp * g->slice[idx].Np), GMME_DIR_IN);
    {   int t; for (t=0;t<20000;t++){ g->idle=hlthunk_is_device_idle(g->fd); if(g->idle) break; } }
    g->slice[idx].used = ++g->clock;
    g->promotes++;
    return 0;
}

/* Place a named weight slice in the arena. Many can be resident at once; the
   card is addressed by slot name from then on. */
int gmme_load_slice(gmme *g, const char *name, const int8_t *W, int K, int N) {
    int Kp = K, Np = N, i;
    uint64_t need, off;
    if (!g || g->fd < 0 || !name) return -1;
    if (K < 1 || N < 1 || (K > GMME_MAX_K && !e_uncap) || (N > GMME_MAX_N && !e_uncap)) {
        snprintf(g->err,sizeof g->err,"load_slice: bad shape"); return -1; }
    if (!g->warena_size) { snprintf(g->err,sizeof g->err,"no weight arena"); return -1; }
    if (g->pad) {
        Kp = ((K + 31) / 32) * 32; if (Kp < 32) Kp = 32;
        Np = (N <= 256) ? 256 : ((N + 31) / 32) * 32;  /* N MUST pad to exactly 256: */
            /* B_ROI_SIZE_0 is the fixed 256-byte pitch while B_VALID_ELEMENTS_0 was N,
               and ROI_SIZE must never exceed VALID_ELEMENTS. With the old
               ceil(N/32)*32 any N<=224 padded to less than 256 and produced ZERO
               correct columns, silently. Measured: N=224 wrong, N=225 exact.
               The copy-out already slices N columns from Np-wide rows, so padding
               to 256 costs discarded MACs but is correct. GMME_NPAD=0 restores the
               old behaviour for experiments only -- it reopens the bug. */
    }
    (void)need; (void)off;
    if (arena_fit(g, (uint64_t)Kp * Np) < 0) return -1;
    for (i = 0; i < g->n_slices; i++)
        if (strcmp(g->slice[i].name, name) == 0) break;
    if (i == g->n_slices) {
        if (slice_table_room(g) < 0) return -1;
        g->n_slices++;
    }
    snprintf(g->slice[i].name,sizeof g->slice[i].name,"%s",name);
    g->slice[i].K = K; g->slice[i].N = N; g->slice[i].Kp = Kp; g->slice[i].Np = Np;
    g->slice[i].card_slot = -1;                    /* starts in host RAM */

    /* Host copy, always kept. This IS the overflow tier. */
    free(g->slice[i].host);
    g->slice[i].host = calloc(1, (size_t)Kp*Np);
    if (!g->slice[i].host) { snprintf(g->err,sizeof g->err,"host copy alloc failed"); return -1; }
    {   int r; for (r = 0; r < K; r++)
            memcpy((int8_t*)g->slice[i].host + (size_t)r*Np, W + (size_t)r*N, N); }

    if (promote(g, i) < 0) return -1;
    return i;
}

int gmme_slice_count(const gmme *g) { return g ? g->n_slices : 0; }
int gmme_slice_slots(const gmme *g) { return g ? g->n_card_slots : 0; }
uint64_t gmme_arena_used(const gmme *g) { return g ? g->warena_used : 0; }
uint64_t gmme_arena_size(const gmme *g) { return g ? g->warena_size : 0; }
uint64_t gmme_slot_bytes(const gmme *g) { return g ? g->wslot : 0; }
int gmme_slices_on_card(const gmme *g) { return g ? g->n_card : 0; }
uint64_t gmme_promotions(const gmme *g) { return g ? g->promotes : 0; }
uint64_t gmme_spills(const gmme *g) { return g ? g->spills : 0; }

/* Stream activations against a named resident slice. */
int gmme_apply_slice(gmme *g, const char *name, const int8_t *X, int32_t *Y, int M) {
    int i;
    if (!g || g->fd < 0 || !name) return -1;
    for (i = 0; i < g->n_slices; i++)
        if (strcmp(g->slice[i].name, name) == 0) {
            if (promote(g, i) < 0) return -1;       /* pull it back if spilled */
            g->cur_slice = i;
            {   int rc = gmme_gemm_i8(g, X, NULL, Y, M, g->slice[i].K, g->slice[i].N);
                g->cur_slice = -1; return rc; }
        }
    snprintf(g->err,sizeof g->err,"slice %s not resident",name);
    return -1;
}


/* ---- gmme_apply_batch --------------------------------------------------
 * N INDEPENDENT components in ONE command buffer, ONE submission, ONE settle.
 *
 * Every MME descriptor carries its own A, B and C base address, so a single
 * config CB can hold descriptors naming DIFFERENT resident weight slices with
 * different activations and different outputs. Nothing in the silicon requires
 * the tiles in a CB to belong to the same GEMM -- that was only ever a property
 * of how we drove it.
 *
 * Why this exists: gmme_apply_slice() runs one component per call and drains the
 * device each time. Measured floor is ~0.315 ms per call, so a 6,000-component
 * model costs ~1.9 s in call overhead alone before any arithmetic. Packing up to
 * MAX_TILES_PER_CB (76) components per fire amortises that floor across all of
 * them.
 *
 * All slices in one batch must share K and N (true for an expert family).
 * M is 1 row per component -- the decode case, which is where the floor hurts.
 */
int gmme_apply_batch(gmme *g, const char **names, const int8_t *X,
                     int32_t *Y, int n)
{
    int i, idx, K = 0, N = 0, Kp, Np, rc;
    uint32_t o = 0, eo;
    void *cfg, *eng;
    int slots[MAX_TILES_PER_CB];

    if (!g || g->fd < 0 || !names || !X || !Y) return -1;
    if (n < 1 || n > (int)MAX_TILES_PER_CB) {
        snprintf(g->err,sizeof g->err,"batch of %d, limit is %d",n,(int)MAX_TILES_PER_CB);
        return -1; }
    exp_init();

    /* resolve every name first: a half-issued batch is worse than none */
    for (i = 0; i < n; i++) {
        for (idx = 0; idx < g->n_slices; idx++)
            if (strcmp(g->slice[idx].name, names[i]) == 0) break;
        if (idx >= g->n_slices) {
            snprintf(g->err,sizeof g->err,"slice %s not resident",names[i]); return -1; }
        if (promote(g, idx) < 0) return -1;
        if (i == 0) { K = g->slice[idx].K; N = g->slice[idx].N; }
        else if (g->slice[idx].K != K || g->slice[idx].N != N) {
            snprintf(g->err,sizeof g->err,"batch needs one K,N; %s is %dx%d not %dx%d",
                     names[i], g->slice[idx].K, g->slice[idx].N, K, N); return -1; }
        slots[i] = idx;
    }
    Kp = K; Np = N;
    if (g->pad) { Kp = ((K+31)/32)*32; if (Kp<32) Kp=32;
                  Np = ((N+31)/32)*32; if (Np<32) Np=32; }
    if ((size_t)n*Kp > CAP_A || (size_t)n*Np*4 > CAP_C) {
        snprintf(g->err,sizeof g->err,"batch buffers too large"); return -1; }

    /* one packed upload for every component's activation row */
    memset(g->hb, 0, (size_t)n*Kp);
    for (i = 0; i < n; i++)
        memcpy((int8_t*)g->hb + (size_t)i*Kp, X + (size_t)i*K, K);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,g->A,
                         (uint32_t)(n*Kp), GMME_DIR_IN);

    if (!g->cb_mcfg) {
        g->cb_mcfg = hltests_create_cb(g->fd, CFG_CB_SZ, INTERNAL, g->sram + CFGCB);
        if (!g->cb_mcfg) { snprintf(g->err,sizeof g->err,"create cfg cb"); return -1; } }
    if (!g->cb_meng) {
        g->cb_meng = hltests_create_cb(g->fd, 0x1000, INTERNAL, g->sram + ENGCB);
        if (!g->cb_meng) { snprintf(g->err,sizeof g->err,"create eng cb"); return -1; } }
    cfg = g->cb_mcfg; eng = g->cb_meng;

    /* one descriptor per component; cur_slice selects that descriptor's B */
    for (i = 0; i < n; i++) {
        g->cur_slice = slots[i];
        o = desc(g, cfg, o, g->A + (uint64_t)i*Kp,
                 g->C + (uint64_t)i*Np*4, 1, Kp, Np, i == 0);
    }
    g->cur_slice = -1;

    {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
        p.eb=EB_FALSE; p.mb=MB_TRUE;
        p.cp_dma.src_addr = g->sram + CFGCB; p.cp_dma.size = o;
        eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+CFGCB, o, GMME_DIR_IN==HL_DMA_HOST_TO_SRAM?GMME_DIR_IN:HL_DMA_HOST_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+ENGCB, eo, HL_DMA_HOST_TO_SRAM);

    rc = submit(g, cfg, o, n);
    if (rc) return rc;

    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->C,g->hva,
                         (uint32_t)(n*Np*4), GMME_DIR_OUT);
    for (i = 0; i < n; i++)
        memcpy((int8_t*)Y + (size_t)i*N*4, (int8_t*)g->hb + (size_t)i*Np*4, (size_t)N*4);
    g->fires = n;
    return 0;
}

int gmme_load_weights(gmme *g, const int8_t *W, int K, int N) {
    int Kp = K, Np = N, i;
    if (!g || g->fd < 0) return -1;
    if (K < 1 || N < 1 || (K > GMME_MAX_K && !e_uncap) || (N > GMME_MAX_N && !e_uncap)) {
        snprintf(g->err,sizeof g->err,"pin_b: bad shape"); return -1; }
    if (g->pad) {
        Kp = ((K + 31) / 32) * 32; if (Kp < 32) Kp = 32;
        Np = (N <= 256) ? 256 : ((N + 31) / 32) * 32;  /* N MUST pad to exactly 256: */
            /* B_ROI_SIZE_0 is the fixed 256-byte pitch while B_VALID_ELEMENTS_0 was N,
               and ROI_SIZE must never exceed VALID_ELEMENTS. With the old
               ceil(N/32)*32 any N<=224 padded to less than 256 and produced ZERO
               correct columns, silently. Measured: N=224 wrong, N=225 exact.
               The copy-out already slices N columns from Np-wide rows, so padding
               to 256 costs discarded MACs but is correct. GMME_NPAD=0 restores the
               old behaviour for experiments only -- it reopens the bug. */
    }
    if ((size_t)Kp*Np > CAP_B) { snprintf(g->err,sizeof g->err,"pin_b: too large"); return -1; }
    memset(g->hb, 0, (size_t)Kp*Np);
    for (i = 0; i < K; i++) memcpy((int8_t*)g->hb + (size_t)i*Np, W + (size_t)i*N, N);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,g->B,
                         (uint32_t)(Kp*Np),GMME_DIR_IN);
    /* Settle before returning. Every entry point leaves the device idle: the
       driver soft-resets a card that is not idle on release, and a shell that
       can leave it hot is not a shell. */
    {   int tries;
        for (tries = 0; tries < 20000; tries++) {
            g->idle = hlthunk_is_device_idle(g->fd);
            if (g->idle) break;
        }
        if (tries > g->idle_worst) g->idle_worst = tries;
    }
    g->b_pinned = 1; g->b_K = K; g->b_N = N;
    return 0;
}

void gmme_drop_weights(gmme *g) { if (g) g->b_pinned = 0; }
int  gmme_weights_k(const gmme *g) { return (g && g->b_pinned) ? g->b_K : 0; }
int  gmme_weights_n(const gmme *g) { return (g && g->b_pinned) ? g->b_N : 0; }

/* Stream activations against the resident weights. No weight pointer exists in
   this signature by design. */
int gmme_apply(gmme *g, const int8_t *X, int32_t *Y, int M) {
    if (!g || g->fd < 0) return -1;
    if (!g->b_pinned) {
        snprintf(g->err,sizeof g->err,
                 "gmme_apply: no resident weights -- call gmme_load_weights first");
        return -1;
    }
    return gmme_gemm_i8(g, X, NULL, Y, M, g->b_K, g->b_N);
}

/* ---- All nine engines at once ---------------------------------------- */

int gmme_fire_all(gmme *g, const int8_t *A, const int8_t *B, int32_t *C,
                  int M, int K, int N, gmme_tpc_job *jobs, int njobs)
{
    uint64_t tseq[8], mseq = 0; void *tdma[8], *mdma = NULL; uint32_t tq[8];
    int have_mme = (A != NULL && C != NULL), rc = 0, i, r;
    int Kp = K, Np = N, Mp = M, mt = 0, tiles = 0, done = 0, rows;
    uint32_t o = 0, eo = 0;
    void *cfg = NULL, *eng = NULL;

    if (!g || g->fd < 0) return -1;
    g->err[0] = 0; g->idle_worst = 0;
    if (njobs < 0 || njobs > 8) { snprintf(g->err,sizeof g->err,"njobs 0..8"); return -1; }
    if (!have_mme && njobs == 0) { snprintf(g->err,sizeof g->err,"nothing to do"); return -1; }
    if (njobs > 0 && tpc_validate(g, jobs, njobs)) return -1;

    /* ---- MME leg: program and submit, do NOT wait --------------------- */
    if (have_mme) {
        if (M<1||K<1||N<1)      { snprintf(g->err,sizeof g->err,"bad shape"); return -1; }
        if (K > GMME_MAX_K && !e_uncap) { snprintf(g->err,sizeof g->err,"K=%d exceeds %d",K,GMME_MAX_K); return -1; }
        if (N > GMME_MAX_N)     { snprintf(g->err,sizeof g->err,"N=%d exceeds %d",N,GMME_MAX_N); return -1; }
        if ((size_t)M*K > CAP_A){ snprintf(g->err,sizeof g->err,"A too large"); return -1; }
        if ((size_t)M*N*4>CAP_C){ snprintf(g->err,sizeof g->err,"C too large"); return -1; }
        if (g->pad) {
            Kp = ((K + 31) / 32) * 32; if (Kp < 32) Kp = 32;
            Np = (N <= 256) ? 256 : ((N + 31) / 32) * 32;  /* N MUST pad to exactly 256: */
            /* B_ROI_SIZE_0 is the fixed 256-byte pitch while B_VALID_ELEMENTS_0 was N,
               and ROI_SIZE must never exceed VALID_ELEMENTS. With the old
               ceil(N/32)*32 any N<=224 padded to less than 256 and produced ZERO
               correct columns, silently. Measured: N=224 wrong, N=225 exact.
               The copy-out already slices N columns from Np-wide rows, so padding
               to 256 costs discarded MACs but is correct. GMME_NPAD=0 restores the
               old behaviour for experiments only -- it reopens the bug. */
            if ((Kp > GMME_MAX_K && !e_uncap) || (Np > GMME_MAX_N && !e_uncap)) {
                snprintf(g->err,sizeof g->err,"padded K=%d N=%d exceeds 256",Kp,Np); return -1; }
        }
        mt = g->rows_ovr ? g->rows_ovr
           : (e_spmode ? (M > 256 ? 256 : M)    /* spatial: one descriptor covers all rows,
                                                   but SPATIAL_SIZE_MINUS_1 is 8 bits, so
                                                   256 rows is the ceiling -- above that we
                                                   tile into several descriptors again.
                                                   Setting mt=M unconditionally broke M>256. */
                       : gmme_rows_per_fire(Kp));
        Mp = g->pad ? ((M + mt - 1) / mt) * mt : M;
        if ((size_t)Mp*Kp > CAP_A || (size_t)Kp*Np > CAP_B || (size_t)Mp*Np*4 > CAP_C) {
            snprintf(g->err,sizeof g->err,"padded buffers too large"); return -1; }
        if ((Mp + mt - 1) / mt > g->max_tiles) {
            snprintf(g->err,sizeof g->err,
                "gemm needs %d tiles, one CB holds %d -- use a smaller M",
                (Mp+mt-1)/mt, g->max_tiles);
            return -1; }

        if (Kp == K && Mp == M) {          /* no padding: one contiguous copy, no memset */
            memcpy(g->hb, A, (size_t)M*K);
        } else {
            memset(g->hb, 0, (size_t)Mp*Kp);
            for (i = 0; i < M; i++) memcpy((int8_t*)g->hb + (size_t)i*Kp, A + (size_t)i*K, K);
        }
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,g->A,
                             (uint32_t)(Mp*Kp),GMME_DIR_IN);
        if (g->cur_slice < 0 && B && !(g->b_pinned && g->b_K == K && g->b_N == N)) {
PB();
            memset(g->hb, 0, (size_t)Kp*Np);
            if ((e_spmode && e_bconv == 3) || e_vdesc) {
                /* B goes to the card TRANSPOSED for the spatial form: B[n][k],
                   row length Kp. Source is row-major B[k][n]. */
                int kk, nn;
                if (e_braw) {
                    /* row-major into a 256-byte pitch. A byte-at-a-time loop over
                       K*N was costing more than the DMA it feeds. */
                    if (N == 256) {
                        memcpy(g->hb, B, (size_t)K*256);          /* pitch == row: one copy */
                    } else {
                        for (kk = 0; kk < K; kk++)
                            memcpy((int8_t*)g->hb + (size_t)kk*256, B + (size_t)kk*N, (size_t)N);
                    }
                    (void)nn;
                } else
                for (nn = 0; nn < N; nn++)
                    for (kk = 0; kk < K; kk++)
                        ((int8_t*)g->hb)[(size_t)nn*256 + kk] = B[(size_t)kk*N + nn];
            } else
            for (i = 0; i < K; i++) memcpy((int8_t*)g->hb + (size_t)i*Np, B + (size_t)i*N, N);
            hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,g->B,
                                 (uint32_t)(((e_spmode&&e_bconv==3)||e_vdesc)?(e_braw?(size_t)Kp*256:(size_t)Np*256):(size_t)Kp*Np),GMME_DIR_IN);
PE("gemm: B stage+DMA");
        }
        if (!g->cb_mcfg) {
            g->cb_mcfg = hltests_create_cb(g->fd, CFG_CB_SZ, INTERNAL, g->sram + CFGCB);
            if (!g->cb_mcfg) { snprintf(g->err,sizeof g->err,"create cfg cb"); return -1; } }
        if (!g->cb_meng) {
            g->cb_meng = hltests_create_cb(g->fd, 0x1000, INTERNAL, g->sram + ENGCB);
            if (!g->cb_meng) { snprintf(g->err,sizeof g->err,"create eng cb"); return -1; } }
        cfg = g->cb_mcfg; eng = g->cb_meng;
        while (done < Mp && tiles < g->max_tiles && o + DESC_BYTES <= CFG_CB_SZ) {
            rows = (Mp - done < mt) ? (Mp - done) : mt;
            o = desc(g, cfg, o, g->A + (uint64_t)done*Kp,
                     g->C + (uint64_t)done*Np*4, rows, Kp, Np, tiles == 0);
            done += rows; tiles++;
        }
        {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
            p.eb=EB_FALSE; p.mb=MB_TRUE;
            p.cp_dma.src_addr = g->sram + CFGCB; p.cp_dma.size = o;
            eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+CFGCB, o, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+ENGCB, eo, HL_DMA_HOST_TO_SRAM);
        if (gmme_submit_async(g, eng, eo, g->mq, &mseq, &mdma)) {
            snprintf(g->err,sizeof g->err,"MME submit failed"); gmme_settle(g); return -1; }
        /* MME is now running. Everything below overlaps with it. */
    }

    /* ---- TPC legs, interleaved with the MME --------------------------- */
    if (njobs > 0) {
        if (tpc_stage_all(g, jobs, njobs, tseq, tdma, tq)) rc = -1;
        if (!rc && gmme_wait_all(g, tseq, tdma, njobs)) {
            snprintf(g->err,sizeof g->err,"TPC CS#1 did not complete"); rc = -1; }
        if (!rc && tpc_exec_all(g, jobs, njobs, tseq, tdma, tq)) rc = -1;
        if (!rc && gmme_wait_all(g, tseq, tdma, njobs)) {
            snprintf(g->err,sizeof g->err,"TPC EXECUTE did not complete"); rc = -1; }
    }
    if (have_mme) {
        if (hltests_wait_for_cs(g->fd, mseq, 5000000) != HL_WAIT_CS_STATUS_COMPLETED) {
            snprintf(g->err,sizeof g->err,"MME did not complete"); rc = -1; }
        hltests_destroy_cb(g->fd, mdma);
    }
    /* One settle for the whole batch. */
    if (gmme_settle(g)) { snprintf(g->err,sizeof g->err,"never settled"); rc = -1; }
    if (rc) return -1;

    if (have_mme) {
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->C,g->hva,
                             (uint32_t)(Mp*Np*4),GMME_DIR_OUT);
        if (Np == N) {                     /* pitch matches: one contiguous copy */
            memcpy(C, g->hb, (size_t)M*N*4);
        } else {
            for (r = 0; r < M; r++)
                memcpy((int8_t*)C + (size_t)r*N*4, (int8_t*)g->hb + (size_t)r*Np*4, (size_t)N*4);
        }
        g->fires += tiles;
    }
    if (njobs > 0) tpc_collect_all(g, jobs, njobs);
    return 0;
}

/* Device-resident GEMM: operands ALREADY on the card, result stays on the card.
   No host upload, no readback, no per-op DMA. This is the entry point a chained
   layer needs -- gmme_gemm_i8() copies A down and C back every call because its
   signature takes host pointers, which is fine for one isolated op and ruinous
   for a sequence of them. Offsets are byte offsets into the existing A/B/C
   device allocations. */
/* Device-to-device copy inside the card: C -> A, no host memory. This is the
   hop that lets one GEMM's output become the next GEMM's input without a round
   trip through the host. */
/* Chain mode. Between begin/end the per-fire device-idle settle is skipped;
   gmme_chain_end() performs it once. The card must still be returned to idle
   before the context closes -- that is thermal AND correctness -- so end() is
   mandatory, not optional. */
int  gmme_in_chain(void) { return in_chain; }

void gmme_chain_begin(gmme *g) {
    (void)g; in_chain = 1;
    /* ordered completion instead of a polled one: the MME signals a sync
       object and the CS fences on it, so the next op cannot start early. */
    e_signal = 1; e_fence = 1;
    if (!e_hdr) e_hdr = SPHDR | 0x20u;   /* HEADER bit5 SIGNAL_EN */
}
int  gmme_chain_end(gmme *g)   {
    in_chain = 0; e_signal = 0; e_fence = 0;
    if (e_hdr == (SPHDR | 0x20u)) e_hdr = 0;
    return gmme_settle(g);
}

int gmme_dev_copy_c_to_a(gmme *g, uint64_t a_off, uint64_t c_off, uint32_t bytes)
{
    if (!g || g->fd < 0) return -1;
    if ((size_t)a_off + bytes > CAP_A || (size_t)c_off + bytes > CAP_C) {
        snprintf(g->err,sizeof g->err,"dev_copy: out of range"); return -1; }
    hltests_dma_transfer(g->fd, g->dq, EB_FALSE, MB_TRUE,
                         g->C + c_off, g->A + a_off, bytes, GOYA_DMA_DRAM_TO_DRAM);
    return 0;
}

/* Upload/download for the ENDS of a chain -- called once per layer, not per op. */
/* Stage N independent A/B regions in ONE submission. Calling gmme_put_a/put_b in
   a loop is N blocking DMA round trips for work that has no ordering between its
   pieces -- a serial loop over independent data is a defect on this hardware. */
int gmme_put_many(gmme *g, const uint64_t *dst_off, const void *const *src,
                  const uint32_t *bytes, const int *to_b, int n)
{
    struct stage_req rq[32]; int i, nq = 0; uint64_t hoff = 0;
    if (!g || g->fd < 0 || n < 1 || n > 32) return -1;
    for (i = 0; i < n; i++) {
        if (hoff + bytes[i] > CAP_H) {
            snprintf(g->err,sizeof g->err,"put_many: host staging overflow"); return -1; }
        memcpy((int8_t*)g->hb + hoff, src[i], bytes[i]);
        rq[nq].src  = g->hva + hoff;
        rq[nq].dst  = (to_b && to_b[i]) ? (g->B + dst_off[i]) : (g->A + dst_off[i]);
        rq[nq].size = bytes[i];
        rq[nq].dir  = GMME_DIR_IN;
        nq++; hoff += bytes[i];
    }
    return gmme_stage_multi(g, rq, nq);
}

int gmme_put_a(gmme *g, uint64_t a_off, const void *src, uint32_t bytes)
{
    if (!g || g->fd < 0 || (size_t)a_off + bytes > CAP_A) return -1;
    memcpy(g->hb, src, bytes);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,g->A+a_off,bytes,GOYA_DMA_HOST_TO_DRAM);
    return 0;
}
int gmme_put_b(gmme *g, uint64_t b_off, const void *src, uint32_t bytes)
{
    if (!g || g->fd < 0 || (size_t)b_off + bytes > CAP_B) return -1;
    memcpy(g->hb, src, bytes);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,g->B+b_off,bytes,GOYA_DMA_HOST_TO_DRAM);
    return 0;
}
/* Stage into the C region. put_a/put_b cover the GEMM operands, but a
   requantise needs its per-element scale vector RESIDENT alongside the int32
   result the MME just wrote, and that result lives in C. */
int gmme_put_c(gmme *g, uint64_t c_off, const void *src, uint32_t bytes)
{
    if (!g || g->fd < 0 || (size_t)c_off + bytes > CAP_C) return -1;
    memcpy(g->hb, src, bytes);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,g->C+c_off,bytes,GOYA_DMA_HOST_TO_DRAM);
    return 0;
}
int gmme_get_c(gmme *g, uint64_t c_off, void *dst, uint32_t bytes)
{
    if (!g || g->fd < 0 || (size_t)c_off + bytes > CAP_C) return -1;
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->C+c_off,g->hva,bytes,GOYA_DMA_DRAM_TO_HOST);
    memcpy(dst, g->hb, bytes);
    return 0;
}

int gmme_gemm_resident_acc(gmme *g, uint64_t a_off, uint64_t b_off, uint64_t c_off,
                           int M, int K, int N, int accumulate)
{
    int rc_; acc_cin = accumulate ? 1 : 0;
    rc_ = gmme_gemm_resident(g, a_off, b_off, c_off, M, K, N);
    acc_cin = 0; return rc_;
}

int gmme_gemm_resident(gmme *g, uint64_t a_off, uint64_t b_off, uint64_t c_off,
                       int M, int K, int N)
{
    int Kp = K, Np = N, Mp = M, mt, tiles, done = 0, rows, rc;
    uint32_t o = 0;
    void *cfg;
    if (!g || g->fd < 0) return -1;
    if (M<1||K<1||N<1) { snprintf(g->err,sizeof g->err,"bad shape"); return -1; }
    exp_init();
    g->err[0] = 0; g->fires = 0; g->idle_worst = 0;
    if (g->pad) {
        Kp = ((K+31)/32)*32; if (Kp<32) Kp=32;
        Np = ((N+31)/32)*32; if (Np<32) Np=32;
    }
    mt = g->rows_ovr ? g->rows_ovr : (e_spmode ? (M > 256 ? 256 : M) : gmme_rows_per_fire(Kp));
    Mp = g->pad ? ((M + mt - 1)/mt)*mt : M;
    if ((size_t)a_off + (size_t)Mp*Kp > CAP_A ||
        (size_t)c_off + (size_t)Mp*Np*4 > CAP_C ||
        (size_t)b_off + (size_t)Kp*Np > CAP_B) {
        snprintf(g->err,sizeof g->err,"resident: offset out of range"); return -1; }
    g->b_off_cur = b_off;

    while (done < Mp) {
        tiles = 0; o = 0;
        if (!g->cb_mcfg) {
            g->cb_mcfg = hltests_create_cb(g->fd, CFG_CB_SZ, INTERNAL, g->sram + CFGCB);
            if (!g->cb_mcfg) { snprintf(g->err,sizeof g->err,"create cfg cb"); return -1; }
        }
        cfg = g->cb_mcfg;
        if (e_credits) o = credits(g, cfg, o);
        while (done < Mp && tiles < g->max_tiles && o + DESC_BYTES <= CFG_CB_SZ) {
            rows = (Mp - done < mt) ? (Mp - done) : mt;
            o = desc(g, cfg, o, g->A + a_off + (uint64_t)done*Kp,
                     g->C + c_off + (uint64_t)done*Np*4, rows, Kp, Np, tiles == 0);
            done += rows; tiles++;
        }
        rc = submit(g, cfg, o, tiles);
        if (rc) { g->b_off_cur = 0; return rc; }
    }
    g->b_off_cur = 0;
    return 0;
}

int gmme_gemm_i8(gmme *g, const int8_t *A, const int8_t *B, int32_t *C,
                 int M, int K, int N) {
    int mt, done = 0, rows, rc;
    if (!g || g->fd < 0) return -1;
    if (M<1 || K<1 || N<1)          { snprintf(g->err,sizeof g->err,"bad shape"); return -1; }
    if (g->cur_slice < 0 && !B && !(g->b_pinned && g->b_K==K && g->b_N==N)) {
        snprintf(g->err,sizeof g->err,"no B given and no matching resident weights"); return -1; }
    if (K > GMME_MAX_K && !e_uncap) { snprintf(g->err,sizeof g->err,"K=%d exceeds %d",K,GMME_MAX_K); return -1; }
    exp_init();
    if (N > GMME_MAX_N && !e_uncap) { snprintf(g->err,sizeof g->err,"N=%d exceeds %d",N,GMME_MAX_N); return -1; }
    if ((size_t)M*K > CAP_A)        { snprintf(g->err,sizeof g->err,"A too large"); return -1; }
    if ((size_t)M*N*4 > CAP_C)      { snprintf(g->err,sizeof g->err,"C too large"); return -1; }
    exp_init();
    g->fires = 0; g->idle_worst = 0; g->err[0] = 0;

    /* Pad so no tile and no block ends on a remainder. Zero rows of A and zero
       columns of B contribute nothing to the sum, so the padded result is exact
       in the M x N window we hand back. */
    {   int Kp = K, Np = N, Mp, r, i;
        if (g->pad) {
            Kp = ((K + 31) / 32) * 32; if (Kp < 32) Kp = 32;
            Np = (N <= 256) ? 256 : ((N + 31) / 32) * 32;  /* N MUST pad to exactly 256: */
            /* B_ROI_SIZE_0 is the fixed 256-byte pitch while B_VALID_ELEMENTS_0 was N,
               and ROI_SIZE must never exceed VALID_ELEMENTS. With the old
               ceil(N/32)*32 any N<=224 padded to less than 256 and produced ZERO
               correct columns, silently. Measured: N=224 wrong, N=225 exact.
               The copy-out already slices N columns from Np-wide rows, so padding
               to 256 costs discarded MACs but is correct. GMME_NPAD=0 restores the
               old behaviour for experiments only -- it reopens the bug. */
            if ((Kp > GMME_MAX_K && !e_uncap) || (Np > GMME_MAX_N && !e_uncap)) {
                snprintf(g->err,sizeof g->err,"padded K=%d N=%d exceeds 256",Kp,Np); return -1; }
        }
        mt = g->rows_ovr ? g->rows_ovr
           : (e_spmode ? (M > 256 ? 256 : M)    /* spatial: one descriptor covers all rows,
                                                   but SPATIAL_SIZE_MINUS_1 is 8 bits, so
                                                   256 rows is the ceiling -- above that we
                                                   tile into several descriptors again.
                                                   Setting mt=M unconditionally broke M>256. */
                       : gmme_rows_per_fire(Kp));
        Mp = g->pad ? ((M + mt - 1) / mt) * mt : M;
        if ((size_t)Mp*Kp > CAP_A || (size_t)Kp*Np > CAP_B || (size_t)Mp*Np*4 > CAP_C) {
            snprintf(g->err,sizeof g->err,"padded buffers too large"); return -1; }

        /* A into hb[0..], B into hb[CAP_A..] so BOTH go in ONE submission.
           hltests_dma_transfer is create+submit+wait+destroy per call. */
        if (Kp == K && Mp == M) {
            memcpy(g->hb, A, (size_t)M*K);
        } else {
            memset(g->hb, 0, (size_t)Mp*Kp);
            for (i = 0; i < M; i++) memcpy((int8_t*)g->hb + (size_t)i*Kp, A + (size_t)i*K, K);
        }
PB();
        {   struct stage_req rq[2]; int nq = 0; uint32_t bsz = 0;
            int8_t *bh = (int8_t*)g->hb + HSTAGE_B;
            int need_b = (g->cur_slice < 0 && B &&
                          !(g->b_pinned && g->b_K == K && g->b_N == N));
            rq[nq].src = g->hva; rq[nq].dst = g->A; rq[nq].size = (uint32_t)(Mp*Kp); rq[nq].dir = GMME_DIR_IN; nq++;
            if (need_b) {
                memset(bh, 0, (size_t)Kp*Np);
                if ((e_spmode && e_bconv == 3) || e_vdesc) {
                    int kk, nn;
                    if (e_braw) {
                        if (N == 256) memcpy(bh, B, (size_t)K*256);
                        else for (kk = 0; kk < K; kk++)
                                 memcpy(bh + (size_t)kk*256, B + (size_t)kk*N, (size_t)N);
                        (void)nn;
                    } else
                    for (nn = 0; nn < N; nn++)
                        for (kk = 0; kk < K; kk++)
                            bh[(size_t)nn*256 + kk] = B[(size_t)kk*N + nn];
                } else
                for (i = 0; i < K; i++) memcpy(bh + (size_t)i*Np, B + (size_t)i*N, N);
                bsz = (uint32_t)(((e_spmode&&e_bconv==3)||e_vdesc)
                                 ? (e_braw?(size_t)Kp*256:(size_t)Np*256) : (size_t)Kp*Np);
                rq[nq].src = g->hva + HSTAGE_B; rq[nq].dst = g->B; rq[nq].size = bsz; rq[nq].dir = GMME_DIR_IN; nq++;
            }
            if (gmme_stage_multi(g, rq, nq)) {
                snprintf(g->err,sizeof g->err,"stage A/B failed"); return -1; }
        }
PE("gemm: A+B stage+DMA");

        g->cur_M = Mp;   /* desc() needs the TOTAL rows for GMME_FULLM */
        if (getenv("GMME_FULLM") && Mp > mt && (Mp % 256) == 0) mt = Mp;  /* one descriptor */
        while (done < Mp) {
            void *cfg;
            uint32_t o = 0; int tiles = 0;
            if (!g->cb_mcfg) {
                g->cb_mcfg = hltests_create_cb(g->fd, CFG_CB_SZ, INTERNAL, g->sram + CFGCB);
                if (!g->cb_mcfg) { snprintf(g->err,sizeof g->err,"create cfg cb"); return -1; }
            }
            cfg = g->cb_mcfg;
PB();
            if (e_credits) o = credits(g, cfg, o);
            while (done < Mp && tiles < g->max_tiles
                   && o + DESC_BYTES <= CFG_CB_SZ) {
                rows = (Mp - done < mt) ? (Mp - done) : mt;
                o = desc(g, cfg, o, g->A + (uint64_t)done*Kp,
                         g->C + (uint64_t)done*Np*4, rows, Kp, Np, tiles == 0);
                done += rows; tiles++;
            }
PE("gemm: build descriptors");
            rc = submit(g, cfg, o, tiles);   /* cfg cached in g */
            if (rc) return rc;
        }
PB();
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->C,g->hva,(uint32_t)(Mp*Np*4),GMME_DIR_OUT);
PE("gemm: C->host");
        for (r = 0; r < M; r++)
            memcpy((int8_t*)C + (size_t)r*N*4, (int8_t*)g->hb + (size_t)r*Np*4, (size_t)N*4);
    }
    g->reread_differs = 0;
    if (getenv("GMME_REREAD")) {
        /* Read the SAME device memory again. If the second read differs, the data in
           DRAM was still settling; if it matches, DRAM genuinely holds wrong values. */
        memset(g->hb, 0x5C, (size_t)M*N*4);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->C,g->hva,(uint32_t)(M*N*4),GMME_DIR_OUT);
        if (memcmp(C, g->hb, (size_t)M*N*4)) g->reread_differs = 1;
    }
    return 0;
}
