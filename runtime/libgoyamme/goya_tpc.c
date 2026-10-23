/* goya_tpc.c -- run a TPC kernel through the front door.
 *
 * Ported from tools/tpc/tpc_front_door.c, which is where every one of the
 * comments below was learned the hard way. Stock driver, stock firmware, no
 * debugfs, no BAR0 access: an engine CB carrying one CP_DMA at a config CB
 * staged in SRAM, plus a NOP on an external queue so the CS can complete.
 *
 * Verified on four HL-1000 cards, all 8 TPC cores, stock driver
 * 1FACA53B6E83AD31C2CBD1F, zero soft resets.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <elf.h>
#include <unistd.h>
#include "goya_mme.h"
#include "goya_mme_internal.h"

#define TPC_CFG_BASE          0xE06000
#define QM_TENSOR_BASE        0xE06A00
#define QM_TENSOR_STRIDE      0x4C
#define T_BASE_ADDR_LOW       0x00
#define T_BASE_ADDR_HIGH      0x04
#define T_PADDING_VALUE       0x08
#define T_TENSOR_CONFIG       0x0C
#define T_DIM_0_SIZE          0x10
#define T_DIM_0_STRIDE        0x14
#define T_DIM_0_BASE_OFFSET   0x18
#define QM_BASE_ADDR_LOW      0xE06C60
#define QM_BASE_ADDR_HIGH     0xE06C64
#define QM_TID_BASE_DIM_0     0xE06C68
#define QM_TID_SIZE_DIM_0     0xE06C6C
#define QM_TID_BASE_DIM_1     0xE06C70
#define QM_TID_SIZE_DIM_1     0xE06C74
#define QM_TID_BASE_DIM_2     0xE06C78
#define QM_TID_SIZE_DIM_2     0xE06C7C
#define QM_TID_BASE_DIM_3     0xE06C80
#define QM_TID_SIZE_DIM_3     0xE06C84
#define QM_TID_BASE_DIM_4     0xE06C88
#define QM_TID_SIZE_DIM_4     0xE06C8C
#define QM_KERNEL_CONFIG      0xE06D10
#define TPC0_CFG_ARUSER       0xE06D18
#define TPC0_CFG_AWUSER       0xE06D1C
#define ARUSER_MMBP           (1u << 10)
#define TPC_CMD               0xE06830
#define TPC_EXECUTE           0xE06838
#define ICACHE_BASE_LOW       0xE06840
#define ICACHE_BASE_HIGH      0xE06844
#define TPC_CMD_ICACHE_INV    (1u << 0)
#define TPC_CMD_ICACHE_PF64   (1u << 4)
/* KERNEL ARENA -- every registered kernel lives in SRAM at once.
 *
 * SRAM is 50.59 MB and a TPC kernel .text is 1-16 KB, so hundreds of kernels
 * fit simultaneously. Switching what a core runs is then just pointing that
 * core's ICACHE_BASE at a different resident slot -- invalidate, prefetch,
 * fire. No DMA, no re-upload.
 *
 * That is the point: EVERY core can fire EVERY registered kernel at any time.
 * The earlier one-slot-per-core layout meant a core owned a single verb and any
 * change cost a full upload. This is the rotating cylinder, per core. */
/* 32 KB. Was 16 KB, sized when every kernel was hand-authored (largest .text
   then: 15,776 B). VENDOR kernels are bigger -- softmax_dim0_f32 is 18,144 B and
   the largest goya softmax/layer_norm in libtpc_kernels.so.1 is 25,280 B -- so a
   16 KB slot made every vendor kernel unloadable ("cannot load .text").
   Slot count halved to keep the arena footprint IDENTICAL at 960 KB, still
   ending below SRAM_TPC_IN_OFF (0x100000). Cost: 30 resident kernels, not 60. */
#define KERNEL_SLOT_SIZE   0x8000u                    /* 32 KB */
#define KERNEL_ARENA_SLOTS 30                         /* 960 KB, unchanged */
#define KERNEL_SLOT(i)     (SRAM_TPC_KERNEL_OFF + (uint32_t)(i) * KERNEL_SLOT_SIZE)
/* Two SRAM maps for the TPC scratch.

   Legacy (tpc_part_base == 0, the default) is the original single shared
   region: every core stages through the same buffers, so they can never
   run at once. Partitioned (GMME_TPC_PART=<hex>) gives each core its own
   slice, which is what makes concurrency possible at all.

   The kernel arena (0x000000..0x0F0000) is SHARED in BOTH and that is
   deliberate: it is read-only once uploaded, so any core can fire any
   resident kernel with no reload. Sharing the arena is the point. Sharing
   the scratch was what forced serialization. */
#define LEGACY_IN_OFF         0x100000u
#define LEGACY_IN2_OFF        0x200000u
#define LEGACY_OUT_OFF        0x600000u
/* 0x700000 / 0x710000 WERE USED HERE AND THEY BELONG TO THE MME.
   goya_mme.c stages its config and engine command buffers at CFGCB 0x700000 and
   ENGCB 0x710000. The TPC scratch used exactly the same two addresses, so any
   TPC op overwrote the MME's staged CBs -- and because the MME caches
   `eng_staged`, it then believed its engine CB was still resident, skipped
   re-staging, and executed the TPC's leftover buffer as its own descriptor
   stream. Measured 2026-09-02: a GEMM was EXACT cold and WRONG after a single
   TPC op, |out| 1.18e9 against a true 1.36e6.
   This is why an interleaved layer (MME projections + TPC norms) could never
   have worked, and it was silent because the cache hid the missing re-stage. */
#define LEGACY_CFG_OFF        0x740000u
#define LEGACY_ENG_OFF        0x750000u

static uint32_t tpc_part(const gmme *g, int i) {
    return g->tpc_part_base + (uint32_t)i * g->tpc_part_stride;
}
static uint32_t tpc_in_off (const gmme *g, int i) {
    return g->tpc_part_base ? tpc_part(g,i) + 0x000000u : LEGACY_IN_OFF; }
static uint32_t tpc_in2_off(const gmme *g, int i) {
    return g->tpc_part_base ? tpc_part(g,i) + 0x100000u : LEGACY_IN2_OFF; }
static uint32_t tpc_out_off(const gmme *g, int i) {
    return g->tpc_part_base ? tpc_part(g,i) + 0x200000u : LEGACY_OUT_OFF; }
/* Third input, for kernels that genuinely take three operands. The legacy map
   leaves 4 MB unused between IN2 (0x200000) and OUT (0x600000); this takes the
   first megabyte of it. The PARTITIONED map has no room -- cfg/eng sit at
   +0x300000 -- so a three-input op refuses to run there rather than quietly
   overlapping another core's scratch. */
#define LEGACY_IN3_OFF        0x300000u
/* Partitioned third input. Each partition is 4 MB (GMME_TPC_STRIDE default
   0x400000) and uses +0 in, +0x100000 in2, +0x200000 out, +0x300000 cfg,
   +0x301000 eng -- leaving +0x310000..+0x400000, 960 KB, unused. A third input
   goes there. It is 64 KB short of the 1 MB the other inputs get, so op3 caps
   there rather than pretending; every real use (RMSNorm's x*invrms*weight over
   8x2048 = 64 KB) is far under it. */
#define PART_IN3_OFF          0x310000u
#define PART_IN3_CAP          0x0F0000u    /* 960 KB */
static uint32_t tpc_in3_off(const gmme *g, int i) {
    return g->tpc_part_base ? tpc_part(g,i) + PART_IN3_OFF : LEGACY_IN3_OFF; }
static uint32_t tpc_cfg_off(const gmme *g, int i) {
    return g->tpc_part_base ? tpc_part(g,i) + 0x300000u : LEGACY_CFG_OFF; }
static uint32_t tpc_eng_off(const gmme *g, int i) {
    return g->tpc_part_base ? tpc_part(g,i) + 0x301000u : LEGACY_ENG_OFF; }

static uint32_t cw(int fd, void *cb, uint32_t o, uint32_t reg, uint32_t v, int eb) {
    struct hltests_pkt_info p; memset(&p,0,sizeof p);
    p.eb = eb ? EB_TRUE : EB_FALSE; p.mb = MB_TRUE;
    p.wreg32.reg_addr = (uint16_t)(reg & 0xfff); p.wreg32.value = v;
    return hltests_add_wreg32_pkt(fd, cb, o, &p);
}

/* A TPC ELF's .text, raw. main must be at offset 0. */
static int load_text(const char *path, void *dst, uint32_t max, uint32_t *out) {
    FILE *f = fopen(path, "rb"); long n; uint8_t *e; int i;
    if (!f) return -1;
    fseek(f,0,SEEK_END); n = ftell(f); fseek(f,0,SEEK_SET);
    e = malloc(n);
    if (!e || fread(e,1,n,f) != (size_t)n) { fclose(f); free(e); return -1; }
    fclose(f);
    { Elf32_Ehdr *eh = (Elf32_Ehdr*)e;
      Elf32_Shdr *sh = (Elf32_Shdr*)(e + eh->e_shoff);
      const char *str = (const char*)(e + sh[eh->e_shstrndx].sh_offset);
      for (i = 0; i < eh->e_shnum; i++)
        if (!strcmp(str + sh[i].sh_name, ".text")) {
            if (sh[i].sh_size > max) { free(e); return -1; }
            memcpy(dst, e + sh[i].sh_offset, sh[i].sh_size);
            *out = sh[i].sh_size; free(e); return 0;
        } }
    free(e); return -1;
}

/* GMME_PROF=1 prints a per-step time breakdown of one fire. */
#include <time.h>
static int prof_on = -1;
static struct timespec prof_t0;
static void PB(void){ if(prof_on<0) prof_on = getenv("GMME_PROF")?1:0;
                      if(prof_on) clock_gettime(CLOCK_MONOTONIC,&prof_t0); }
static void PE(const char *lbl){ struct timespec e; if(!prof_on) return;
    clock_gettime(CLOCK_MONOTONIC,&e);
    fprintf(stderr,"    %-26s %8.3f ms\n", lbl,
        (e.tv_sec-prof_t0.tv_sec)*1e3+(e.tv_nsec-prof_t0.tv_nsec)/1e6); }

static int settle(gmme *g) {
    int t;
    for (t = 0; t < 20000; t++) { g->idle = hlthunk_is_device_idle(g->fd); if (g->idle) break; }
    if (t > g->idle_worst) g->idle_worst = t;
    return g->idle ? 0 : -1;
}

/* The cached DMA CB used to be a process-wide static. It was created against
   the FIRST card opened and then reused for every other card in the same
   process, so on a four-card host every TPC CS on cards 2-4 failed ("CS#1
   failed") while the same call in a fresh process passed. It lives on the
   context now. Found 2026-09-03 by the reflex sensing all four cards. */
static int submit_wait(gmme *g, void *eng, uint32_t eo, uint32_t tq) {
    void *dma; uint32_t dof; uint64_t seq; int rc;
    struct hltests_pkt_info np; struct hltests_cs_chunk ex[2];
    if (!g->cb_tdma) g->cb_tdma = hltests_create_cb(g->fd, 0x1000, EXTERNAL, 0);
    dma = g->cb_tdma;
    memset(&np,0,sizeof np); np.eb = EB_FALSE; np.mb = MB_TRUE;
    dof = hltests_add_nop_pkt(g->fd, dma, 0, EB_FALSE, MB_TRUE);
    memset(ex,0,sizeof ex);
    ex[0].cb_ptr=eng; ex[0].cb_size=eo;  ex[0].queue_index=tq;
    ex[1].cb_ptr=dma; ex[1].cb_size=dof; ex[1].queue_index=g->dq;
    rc = hltests_submit_cs(g->fd, NULL, 0, ex, 2, 0, &seq);
    if (!rc) rc = hltests_wait_for_cs(g->fd, seq, 5000000);
    if (rc != HL_WAIT_CS_STATUS_COMPLETED) return -1;
    /* MEASURED 2026-09-02: skipping this inside a chain is NOT safe. The MME has
       SIGNAL+FENCE ordering in chain mode; the TPC queue has no equivalent, so a
       following DMA can start before the vector core has finished. It passed at
       M=16 and failed at M=64 (5120/16384 wrong) -- a size-dependent race, which
       is the worst kind. The settle stays until the TPC path has a real fence. */
    return settle(g);
}

/* Allocate the two scratch CBs once. See goya_mme_internal.h for why. */
static int tpc_cbs(gmme *g, int i)
{
    if (!g->cb_cfg[i]) {
        g->cb_cfg[i] = hltests_create_cb(g->fd, 0x1000, INTERNAL, g->sram + tpc_cfg_off(g, i));
        if (!g->cb_cfg[i]) { snprintf(g->err,sizeof g->err,"tpc: cfg cb %d",i); return -1; }
    }
    if (!g->cb_eng[i]) {
        g->cb_eng[i] = hltests_create_cb(g->fd, 0x1000, INTERNAL, g->sram + tpc_eng_off(g, i));
        if (!g->cb_eng[i]) { snprintf(g->err,sizeof g->err,"tpc: eng cb %d",i); return -1; }
    }
    return 0;
}

/* One TPC fire. in_b may be NULL, in which case tensors 0/1/2 all alias in_a
   (what the reduce kernel expects). tid0/tid1 are the index-space extents. */
/* Upload a kernel into the arena once and return its handle. Re-registering
   the same path returns the existing handle without touching the card. */
int gmme_tpc_register_kernel(gmme *g, const char *path)
{
    uint32_t tsz = 0; int i;
    if (!g || g->fd < 0 || !path) return -1;
    for (i = 0; i < g->n_kernels; i++)
        if (strcmp(g->kernels[i], path) == 0) return i;      /* already resident */
    if (g->n_kernels >= KERNEL_ARENA_SLOTS) {
        snprintf(g->err,sizeof g->err,"kernel arena full (%d slots)",KERNEL_ARENA_SLOTS);
        return -1; }
    if (load_text(path, g->hb, KERNEL_SLOT_SIZE, &tsz)) {
        snprintf(g->err,sizeof g->err,"cannot load .text from %s",path); return -1; }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
                         g->sram+KERNEL_SLOT(g->n_kernels),tsz,HL_DMA_HOST_TO_SRAM);
    snprintf(g->kernels[g->n_kernels],sizeof g->kernels[0],"%s",path);
    return g->n_kernels++;
}

int gmme_tpc_kernel_count(const gmme *g) { return g ? g->n_kernels : 0; }

/* Build the descriptor + icache program into this core's cfg CB.

   Shared by the single blocking fire and the concurrent batch, deliberately:
   there must be exactly ONE descriptor code path or the two will drift and
   only one of them ends up being the tested one. Returns bytes written. */
static uint32_t tpc_desc(gmme *g, int tpc_idx, void *cfg, uint64_t kaddr,
                         int n_a, const int32_t *in_b, int n_b,
                         int tid0, int tid1, int n_out)
{
    uint32_t o = 0; int t, d;
    uint64_t taddr[4];
    taddr[0]=taddr[2]= g->sram + tpc_in_off(g, tpc_idx);
    taddr[1]= in_b ? g->sram + tpc_in2_off(g, tpc_idx) : g->sram + tpc_in_off(g, tpc_idx);
    taddr[3]= g->sram + tpc_out_off(g, tpc_idx);
    o = cw(g->fd,cfg,o,TPC0_CFG_ARUSER,ARUSER_MMBP,0);
    o = cw(g->fd,cfg,o,TPC0_CFG_AWUSER,ARUSER_MMBP,0);
    for (t = 0; t < 4; t++) {
        uint32_t b = QM_TENSOR_BASE + t*QM_TENSOR_STRIDE;
        o = cw(g->fd,cfg,o,b+T_BASE_ADDR_LOW ,(uint32_t)(taddr[t]&0xFFFFFFFF),0);
        o = cw(g->fd,cfg,o,b+T_BASE_ADDR_HIGH,(uint32_t)(taddr[t]>>32),0);
        o = cw(g->fd,cfg,o,b+T_PADDING_VALUE ,0,0);
        o = cw(g->fd,cfg,o,b+T_TENSOR_CONFIG ,(2u<<0)|(1u<<8),0);   /* INT32 */
        /* DIM_0_SIZE must come from the ACTUAL staged input. A constant here
           only matches one input length; anything else reads past the staged
           region and returns zeros. */
        o = cw(g->fd,cfg,o,b+T_DIM_0_SIZE      ,
               (uint32_t)((t==1 && in_b) ? n_b : (t==3 ? n_out : n_a)),0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_STRIDE    ,4,0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_BASE_OFFSET,0,0);
        /* DIM_1..4 PERSIST across runs. A previous 2-D experiment leaves
           DIM_1_SIZE/STRIDE set and this descriptor is then silently wrong. */
        for (d = 1; d <= 4; d++) {
            uint32_t doff = b + 0x1C + (d-1)*0x0C;
            o = cw(g->fd,cfg,o,doff+0,0,0);
            o = cw(g->fd,cfg,o,doff+4,0,0);
            o = cw(g->fd,cfg,o,doff+8,0,0);
        }
    }
    o = cw(g->fd,cfg,o,QM_BASE_ADDR_LOW ,(uint32_t)(kaddr&0xFFFFFFFF),0);
    o = cw(g->fd,cfg,o,QM_BASE_ADDR_HIGH,(uint32_t)(kaddr>>32),0);
    /* ALL FIVE index-space dims must be set. A stale TID_SIZE of 0 gives the
       index space zero extent: the kernel runs ZERO iterations and returns
       all-zero output from a perfectly successful CS. */
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_0,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_0,(uint32_t)tid0,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_1,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_1,(uint32_t)tid1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_2,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_2,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_3,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_3,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_4,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_4,1,0);
    o = cw(g->fd,cfg,o,QM_KERNEL_CONFIG,0x2002,0);
    o = cw(g->fd,cfg,o,ICACHE_BASE_LOW ,(uint32_t)(kaddr&0xFFFFFFFF),0);
    o = cw(g->fd,cfg,o,ICACHE_BASE_HIGH,(uint32_t)(kaddr>>32),0);
    /* Engine barrier around the icache prefetch, then settle NOPs. Without it
       the TPC starts before its icache is filled and results are run-to-run
       nondeterministic. */
    o = cw(g->fd,cfg,o,TPC_CMD,TPC_CMD_ICACHE_INV|TPC_CMD_ICACHE_PF64,1);
    for (t = 0; t < 8; t++) {
        struct hltests_pkt_info np; memset(&np,0,sizeof np);
        np.eb = EB_TRUE; np.mb = MB_TRUE;
        o = hltests_add_nop_pkt(g->fd, cfg, o, EB_TRUE, MB_TRUE);
    }

    return o;
}


/* ---------------------------------------------------------------------------
 * REQUANTISE ABI -- int32 in, float32 scale in, INT8 out.
 *
 * A separate descriptor builder for the same reason the vendor one is separate:
 * tpc_desc() writes DATA_TYPE = e_32 and DIM_0_STRIDE = 4 for all four tensors,
 * and every bit-exact result in this library rests on it. This ABI needs tensor
 * 3 to be one byte per element, so it gets its own builder rather than a dtype
 * argument threaded through the proven one.
 *
 * The encoding is from the vendor header, not inferred:
 *   habanatools/TensorDescriptor.h, TENSOR_CONFIG field 1:0 DATA_TYPE
 *   enum ElementSizeInBytes { e_8 = 0x0, e_16 = 0x1, e_32 = 0x2 }
 * so int32 and float32 are BOTH e_32 -- Goya gen-1 stores an element size here,
 * not a type -- and only the int8 output differs.
 *
 * Index space is n/256, not the n/64 every all-int32 op uses: the kernel packs
 * four 64-lane f32 groups into one 256-lane int8 vector per step.
 * -------------------------------------------------------------------------*/
#define TCFG_E8   0u   /* ElementSizeInBytes e_8  */
#define TCFG_E32  2u   /* ElementSizeInBytes e_32 */

static const char *kern_path(const char *env, const char *dflt);   /* defined below */

static uint32_t tpc_desc_requant(gmme *g, int tpc_idx, void *cfg, uint64_t kaddr,
                                 int n, int tid0)
{
    uint32_t o = 0; int t, d;
    uint64_t taddr[4];
    taddr[0] = taddr[2] = g->sram + tpc_in_off(g, tpc_idx);   /* a: int32   */
    taddr[1] =            g->sram + tpc_in2_off(g, tpc_idx);  /* b: float32 */
    taddr[3] =            g->sram + tpc_out_off(g, tpc_idx);  /* c: int8    */
    o = cw(g->fd,cfg,o,TPC0_CFG_ARUSER,ARUSER_MMBP,0);
    o = cw(g->fd,cfg,o,TPC0_CFG_AWUSER,ARUSER_MMBP,0);
    for (t = 0; t < 4; t++) {
        uint32_t b  = QM_TENSOR_BASE + t*QM_TENSOR_STRIDE;
        uint32_t dt = (t == 3) ? TCFG_E8 : TCFG_E32;
        o = cw(g->fd,cfg,o,b+T_BASE_ADDR_LOW ,(uint32_t)(taddr[t]&0xFFFFFFFF),0);
        o = cw(g->fd,cfg,o,b+T_BASE_ADDR_HIGH,(uint32_t)(taddr[t]>>32),0);
        o = cw(g->fd,cfg,o,b+T_PADDING_VALUE ,0,0);
        /* VALID_DIM_MASK (bits 12:8) = 1, the value every proven descriptor in
           this library uses. Setting it to 3 to declare a 2-D tensor changed
           nothing, so its documented meaning ("bit i set if dim i is NOT used")
           is not what this silicon does, and it is NOT what gates dim1. */
        o = cw(g->fd,cfg,o,b+T_TENSOR_CONFIG ,(dt<<0)|(1u<<8),0);
        /* All four tensors are FLAT. A 2-D input (dim0 = n/4 words at 16-byte
           stride, dim1 = 4 byte-lanes at 4-byte stride) would have undone the
           char256 byte-lane interleave for free; it does not work on this part.
           Measured: with DIM_1_SIZE=4 and DIM_1_STRIDE=4 written, every
           byte-lane still read the SAME word (out came back as src[p/4]
           repeated four times), unchanged by VALID_DIM_MASK and unchanged by
           giving the index space a matching dim1 extent. Address generation
           here is dim0 only. See k_requant_i32_i8.c for what that means. */
        o = cw(g->fd,cfg,o,b+T_DIM_0_SIZE      ,(uint32_t)n,0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_STRIDE    ,(t==3) ? 1u : 4u,0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_BASE_OFFSET,0,0);
        /* DIM_1..4 persist across runs; a stale extent silently changes the op. */
        for (d = 1; d <= 4; d++) {
            uint32_t doff = b + 0x1C + (d-1)*0x0C;
            o = cw(g->fd,cfg,o,doff+0,0,0);
            o = cw(g->fd,cfg,o,doff+4,0,0);
            o = cw(g->fd,cfg,o,doff+8,0,0);
        }
    }
    o = cw(g->fd,cfg,o,QM_BASE_ADDR_LOW ,(uint32_t)(kaddr&0xFFFFFFFF),0);
    o = cw(g->fd,cfg,o,QM_BASE_ADDR_HIGH,(uint32_t)(kaddr>>32),0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_0,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_0,(uint32_t)tid0,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_1,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_1,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_2,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_2,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_3,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_3,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_4,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_4,1,0);
    o = cw(g->fd,cfg,o,QM_KERNEL_CONFIG,0x2002,0);
    o = cw(g->fd,cfg,o,ICACHE_BASE_LOW ,(uint32_t)(kaddr&0xFFFFFFFF),0);
    o = cw(g->fd,cfg,o,ICACHE_BASE_HIGH,(uint32_t)(kaddr>>32),0);
    o = cw(g->fd,cfg,o,TPC_CMD,TPC_CMD_ICACHE_INV|TPC_CMD_ICACHE_PF64,1);
    for (t = 0; t < 8; t++)
        o = hltests_add_nop_pkt(g->fd, cfg, o, EB_TRUE, MB_TRUE);
    return o;
}

/* The fixed output permutation this ABI produces, within each 256-element
   block: out[p] = requant(in[perm[p]]). A char256 element p is byte p%4 of
   32-bit lane p/4, and Goya gen-1 has no pack/squeeze/partial store to undo it
   (all __gaudi_plus__ or later). Cancel it once, offline, by permuting the
   consuming GEMM's weight rows by the same order -- those weights already have
   to be requantised per column before they can be used at all. */
void gmme_requant_perm(int *perm256)
{
    int p;
    if (!perm256) return;
    for (p = 0; p < 256; p++) perm256[p] = (p % 4) * 64 + (p / 4);
}

/* Elementwise op where tensor 1 is BROADCAST: every lane reads the same
 * address. DIM_0_STRIDE = 0 for that tensor, so the address generator never
 * advances it. This is what turns a per-row scalar (a reduction result) into a
 * full-width operand WITHOUT a host round-trip -- the last thing standing
 * between RMSNorm and running entirely on the card.
 *
 * Separate builder, per tpc_desc()'s own standing note: no proven descriptor is
 * touched. dim1 addressing does not work on this part (see tpc_desc_requant),
 * so stride 0 on dim0 is the only broadcast available.
 */
static uint32_t tpc_desc_bcast(gmme *g, int tpc_idx, void *cfg, uint64_t kaddr,
                               int n, int tid0)
{
    uint32_t o = 0; int t, d;
    uint64_t taddr[4];
    taddr[0] = taddr[2] = g->sram + tpc_in_off(g, tpc_idx);
    taddr[1] =            g->sram + tpc_in2_off(g, tpc_idx);
    taddr[3] =            g->sram + tpc_out_off(g, tpc_idx);
    o = cw(g->fd,cfg,o,TPC0_CFG_ARUSER,ARUSER_MMBP,0);
    o = cw(g->fd,cfg,o,TPC0_CFG_AWUSER,ARUSER_MMBP,0);
    for (t = 0; t < 4; t++) {
        uint32_t b = QM_TENSOR_BASE + t*QM_TENSOR_STRIDE;
        o = cw(g->fd,cfg,o,b+T_BASE_ADDR_LOW ,(uint32_t)(taddr[t]&0xFFFFFFFF),0);
        o = cw(g->fd,cfg,o,b+T_BASE_ADDR_HIGH,(uint32_t)(taddr[t]>>32),0);
        o = cw(g->fd,cfg,o,b+T_PADDING_VALUE ,0,0);
        o = cw(g->fd,cfg,o,b+T_TENSOR_CONFIG ,(2u<<0)|(1u<<8),0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_SIZE      ,(uint32_t)n,0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_STRIDE    ,(t==1) ? 0u : 4u,0);  /* <-- broadcast */
        o = cw(g->fd,cfg,o,b+T_DIM_0_BASE_OFFSET,0,0);
        for (d = 1; d <= 4; d++) {
            uint32_t doff = b + 0x1C + (d-1)*0x0C;
            o = cw(g->fd,cfg,o,doff+0,0,0);
            o = cw(g->fd,cfg,o,doff+4,0,0);
            o = cw(g->fd,cfg,o,doff+8,0,0);
        }
    }
    o = cw(g->fd,cfg,o,QM_BASE_ADDR_LOW ,(uint32_t)(kaddr&0xFFFFFFFF),0);
    o = cw(g->fd,cfg,o,QM_BASE_ADDR_HIGH,(uint32_t)(kaddr>>32),0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_0,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_0,(uint32_t)tid0,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_1,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_1,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_2,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_2,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_3,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_3,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_4,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_4,1,0);
    o = cw(g->fd,cfg,o,QM_KERNEL_CONFIG,0x2002,0);
    o = cw(g->fd,cfg,o,ICACHE_BASE_LOW ,(uint32_t)(kaddr&0xFFFFFFFF),0);
    o = cw(g->fd,cfg,o,ICACHE_BASE_HIGH,(uint32_t)(kaddr>>32),0);
    o = cw(g->fd,cfg,o,TPC_CMD,TPC_CMD_ICACHE_INV|TPC_CMD_ICACHE_PF64,1);
    for (t = 0; t < 8; t++)
        o = hltests_add_nop_pkt(g->fd, cfg, o, EB_TRUE, MB_TRUE);
    return o;
}

int gmme_tpc_op_resident_bcast(gmme *g, const char *kpath, uint64_t a_off,
                               uint64_t b_off, uint64_t out_off, int n, int tpc_idx)
{
    uint32_t o = 0, eo, tq; int kslot, rc; void *cfg, *eng; uint64_t kaddr;
    int tid0 = (n + 63) / 64;
    if (!g || g->fd < 0 || n < 1) return -1;
    if (tpc_idx < 0 || tpc_idx > 7) { snprintf(g->err,sizeof g->err,"bcast: idx 0..7"); return -1; }
    if ((size_t)n*4 > 0x100000) { snprintf(g->err,sizeof g->err,"bcast: input too large"); return -1; }
    kslot = gmme_tpc_register_kernel(g, kpath);
    if (kslot < 0) return -1;
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->C + a_off,
        g->sram + tpc_in_off(g,tpc_idx),  (uint32_t)(n*4), HL_DMA_DRAM_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->C + b_off,
        g->sram + tpc_in2_off(g,tpc_idx), 4u, HL_DMA_DRAM_TO_SRAM);   /* ONE f32 */
    memset(g->hb, 0, 4096);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
        g->sram + tpc_out_off(g,tpc_idx), 4096, HL_DMA_HOST_TO_SRAM);
    if (tpc_cbs(g, tpc_idx)) return -1;
    g->sram_dirty = 1;   /* the MME must re-stage its CBs after this */
    cfg = g->cb_cfg[tpc_idx]; kaddr = g->sram + KERNEL_SLOT(kslot);
    o = tpc_desc_bcast(g, tpc_idx, cfg, kaddr, n, tid0);
    tq = hltests_get_tpc_qid(g->fd, tpc_idx, STREAM0);
    eng = g->cb_eng[tpc_idx];
    {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
        p.eb=EB_FALSE; p.mb=MB_TRUE;
        p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p.cp_dma.size = o;
        eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o, HL_DMA_HOST_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), eo, HL_DMA_HOST_TO_SRAM);
    rc = submit_wait(g, eng, eo, tq);
    if (rc) { snprintf(g->err,sizeof g->err,"bcast: CS#1 failed"); return -1; }
    {   uint32_t o2 = cw(g->fd,cfg,0,TPC_EXECUTE,1,1), e2;
        struct hltests_pkt_info p2; memset(&p2,0,sizeof p2);
        p2.eb=EB_FALSE; p2.mb=MB_TRUE;
        p2.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p2.cp_dma.size = o2;
        e2 = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p2);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), e2, HL_DMA_HOST_TO_SRAM);
        rc = submit_wait(g, eng, e2, tq);
        if (rc) { snprintf(g->err,sizeof g->err,"bcast: EXECUTE failed"); return -1; }
    }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        g->sram + tpc_out_off(g,tpc_idx), g->C + out_off,
        (uint32_t)(n*4), HL_DMA_SRAM_TO_DRAM);
    return 0;
}

/* Two-pass f32 row sum. Pass 1 (k_reduce_f32_strided) strides a full vector so
 * it costs K/64 loads; pass 2 (k_reduce_f32_1d) folds the 64 partials. Total
 * K/64 + 64 loads instead of K -- 96 against 2048 at K=2048.
 *
 * The one-pass version was 77% of RMSNorm's runtime (14.9 ms for 16 rows of
 * 2048) purely because it walked k one element at a time.
 *
 * The summation ORDER differs from the one-pass version, so it is a different
 * f32 answer -- correct, but not the same bits. Callers comparing against a host
 * reference must mirror THIS order. */
int gmme_tpc_reduce_f32_fast(gmme *g, uint64_t a_off, uint64_t scratch_off,
                             uint64_t out_off, int K, int tpc_idx)
{
    if (!g || g->fd < 0) return -1;
    if (K < 64 || (K % 64)) {
        snprintf(g->err,sizeof g->err,"reduce_f32_fast: K must be a multiple of 64"); return -1; }
    {   /* pass 1: K/64 strided loads -> 64 partials at scratch_off */
        const char *saved = getenv("GOYA_TPC_REDUCE_F32_KERNEL");
        (void)saved;
        int rc = gmme_tpc_reduce_f32_pass1(g, a_off, scratch_off, K, tpc_idx);
        if (rc) return rc;
    }
    /* pass 2: fold the 64 partials with the proven one-pass kernel */
    return gmme_tpc_reduce_f32_resident(g, scratch_off, out_off, 64, tpc_idx);
}

/* Row reduction, fully resident: out[0] = sum of a_off[0 .. K-1], f32.
 *
 * tpc_desc() is reused unchanged -- f32 and int32 are both ElementSizeInBytes
 * e_32 on this part, so the all-int32 descriptor is already the right one. Only
 * the kernel differs. Index space is (dim0 = 1, dim1 = K), the same shape the
 * proven int32 reduce uses.
 *
 * The kernel's sliding window reads up to a[K+62], so 63 zero elements are
 * staged past the row rather than letting it read whatever follows: lanes 1..63
 * are not the answer, but they must not be built from unmapped memory.
 */
static int reduce_f32_common(gmme *g, uint64_t a_off, uint64_t out_off,
                             int dma_elems, int tid0, int tid1, const char *kpath, int tpc_idx);
static int reduce2_f32_common(gmme *g, uint64_t a_off, uint64_t b_off,
                              uint64_t out_off, int dma_elems, int tid0,
                              int tid1, const char *kpath, int tpc_idx);

/* Every row in ONE fire. A TPC fire costs ~0.67 ms FIXED regardless of the work
   inside it (measured on hl0: 64 elements 0.675 ms, 65536 elements 1.096 ms --
   a thousand times the work for 62% more time). The 16 separate per-row fires
   this replaces were ~11 ms of pure overhead. Fire COUNT is the cost. */
/* Three-operand elementwise op: tensors 0, 1 and 2 are all distinct. Saves a
 * whole fire wherever two chained binary ops can become one ternary one, and at
 * ~0.67 ms fixed per fire that is the only optimisation on this part that
 * reliably pays. Separate descriptor builder, per tpc_desc()'s standing note. */
static uint32_t tpc_desc3(gmme *g, int tpc_idx, void *cfg, uint64_t kaddr,
                          int n, int tid0)
{
    uint32_t o = 0; int t, d;
    uint64_t taddr[4];
    taddr[0] = g->sram + tpc_in_off (g, tpc_idx);
    taddr[1] = g->sram + tpc_in2_off(g, tpc_idx);
    taddr[2] = g->sram + tpc_in3_off(g, tpc_idx);      /* the point */
    taddr[3] = g->sram + tpc_out_off(g, tpc_idx);
    o = cw(g->fd,cfg,o,TPC0_CFG_ARUSER,ARUSER_MMBP,0);
    o = cw(g->fd,cfg,o,TPC0_CFG_AWUSER,ARUSER_MMBP,0);
    for (t = 0; t < 4; t++) {
        uint32_t b = QM_TENSOR_BASE + t*QM_TENSOR_STRIDE;
        o = cw(g->fd,cfg,o,b+T_BASE_ADDR_LOW ,(uint32_t)(taddr[t]&0xFFFFFFFF),0);
        o = cw(g->fd,cfg,o,b+T_BASE_ADDR_HIGH,(uint32_t)(taddr[t]>>32),0);
        o = cw(g->fd,cfg,o,b+T_PADDING_VALUE ,0,0);
        o = cw(g->fd,cfg,o,b+T_TENSOR_CONFIG ,(2u<<0)|(1u<<8),0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_SIZE      ,(uint32_t)n,0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_STRIDE    ,4,0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_BASE_OFFSET,0,0);
        for (d = 1; d <= 4; d++) {
            uint32_t doff = b + 0x1C + (d-1)*0x0C;
            o = cw(g->fd,cfg,o,doff+0,0,0);
            o = cw(g->fd,cfg,o,doff+4,0,0);
            o = cw(g->fd,cfg,o,doff+8,0,0);
        }
    }
    o = cw(g->fd,cfg,o,QM_BASE_ADDR_LOW ,(uint32_t)(kaddr&0xFFFFFFFF),0);
    o = cw(g->fd,cfg,o,QM_BASE_ADDR_HIGH,(uint32_t)(kaddr>>32),0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_0,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_0,(uint32_t)tid0,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_1,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_1,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_2,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_2,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_3,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_3,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_4,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_4,1,0);
    o = cw(g->fd,cfg,o,QM_KERNEL_CONFIG,0x2002,0);
    o = cw(g->fd,cfg,o,ICACHE_BASE_LOW ,(uint32_t)(kaddr&0xFFFFFFFF),0);
    o = cw(g->fd,cfg,o,ICACHE_BASE_HIGH,(uint32_t)(kaddr>>32),0);
    o = cw(g->fd,cfg,o,TPC_CMD,TPC_CMD_ICACHE_INV|TPC_CMD_ICACHE_PF64,1);
    for (t = 0; t < 8; t++)
        o = hltests_add_nop_pkt(g->fd, cfg, o, EB_TRUE, MB_TRUE);
    return o;
}

int gmme_tpc_op3_resident(gmme *g, const char *kpath, uint64_t a_off,
                          uint64_t b_off, uint64_t c_off, uint64_t out_off,
                          int n, int tpc_idx)
{
    uint32_t o = 0, eo, tq; int kslot, rc; void *cfg, *eng; uint64_t kaddr;
    int tid0 = (n + 63) / 64;
    if (!g || g->fd < 0 || n < 1) return -1;
    {   size_t cap = g->tpc_part_base ? PART_IN3_CAP : 0x100000u;
        if ((size_t)n*4 > cap) { snprintf(g->err,sizeof g->err,
            "op3: %d elems exceeds the %s third-scratch cap of %zu bytes",
            n, g->tpc_part_base ? "partitioned" : "legacy", cap); return -1; } }
    kslot = gmme_tpc_register_kernel(g, kpath);
    if (kslot < 0) return -1;
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->C + a_off,
        g->sram + tpc_in_off(g,tpc_idx),  (uint32_t)(n*4), HL_DMA_DRAM_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->C + b_off,
        g->sram + tpc_in2_off(g,tpc_idx), (uint32_t)(n*4), HL_DMA_DRAM_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->C + c_off,
        g->sram + tpc_in3_off(g,tpc_idx), (uint32_t)(n*4), HL_DMA_DRAM_TO_SRAM);
    memset(g->hb, 0, 4096);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
        g->sram + tpc_out_off(g,tpc_idx), 4096, HL_DMA_HOST_TO_SRAM);
    if (tpc_cbs(g, tpc_idx)) return -1;
    g->sram_dirty = 1;   /* the MME must re-stage its CBs after this */
    cfg = g->cb_cfg[tpc_idx]; kaddr = g->sram + KERNEL_SLOT(kslot);
    o = tpc_desc3(g, tpc_idx, cfg, kaddr, n, tid0);
    tq = hltests_get_tpc_qid(g->fd, tpc_idx, STREAM0);
    eng = g->cb_eng[tpc_idx];
    {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
        p.eb=EB_FALSE; p.mb=MB_TRUE;
        p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p.cp_dma.size = o;
        eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o, HL_DMA_HOST_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), eo, HL_DMA_HOST_TO_SRAM);
    rc = submit_wait(g, eng, eo, tq);
    if (rc) { snprintf(g->err,sizeof g->err,"op3: CS#1 failed"); return -1; }
    {   uint32_t o2 = cw(g->fd,cfg,0,TPC_EXECUTE,1,1), e2;
        struct hltests_pkt_info p2; memset(&p2,0,sizeof p2);
        p2.eb=EB_FALSE; p2.mb=MB_TRUE;
        p2.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p2.cp_dma.size = o2;
        e2 = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p2);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), e2, HL_DMA_HOST_TO_SRAM);
        rc = submit_wait(g, eng, e2, tq);
        if (rc) { snprintf(g->err,sizeof g->err,"op3: EXECUTE failed"); return -1; }
    }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        g->sram + tpc_out_off(g,tpc_idx), g->C + out_off,
        (uint32_t)(n*4), HL_DMA_SRAM_TO_DRAM);
    return 0;
}

/* Two-fire sum of squares: strided pass (rows * hid/64 loads) then a 64-wide
   fold. ONE extra fire in total, taken only because measurement showed work had
   become dominant -- the single-fire version was 32768 vector loads. Not the
   same trade as k_reduce_f32_strided, which added a fire PER ROW. */
int gmme_tpc_sumsq_f32_rows_fast(gmme *g, uint64_t a_off, uint64_t scratch_off,
                                 uint64_t out_off, int rows, int hid, int tpc_idx)
{
    int rc;
    if (!g || g->fd < 0 || rows < 1 || hid < 64 || (hid % 64)) {
        if (g) snprintf(g->err,sizeof g->err,"sumsq_fast: hid must be a multiple of 64");
        return -1; }
    rc = reduce_f32_common(g, a_off, scratch_off, rows*hid, rows, hid/64,
        kern_path("GOYA_TPC_SUMSQ_F32_STRIDED","k_sumsq_f32_rows_strided.o"), tpc_idx);
    if (rc) return rc;
    return gmme_tpc_reduce_f32_rows(g, scratch_off, out_off, rows, 64, tpc_idx);
}

/* One dot product per row in ONE fire: out[i*64] = dot(a_row_i, b_row_i).
   The attention score primitive -- QK^T for a T-token head is T*T dots, laid
   out as T*T rows and fired once instead of T*T times. */
/* Row max in TWO fires: strided pass then a 64-wide fold. Softmax needs it --
   exp(score) overflows f32 past ~88 and real scores go well beyond, so the
   max-subtract is not optional. Max is associative, so unlike the f32 sums the
   fold order does not change the answer. */
int gmme_tpc_max_f32_rows(gmme *g, uint64_t a_off, uint64_t scratch_off,
                          uint64_t out_off, int rows, int hid, int tpc_idx)
{
    int rc;
    if (!g || g->fd < 0 || rows < 1 || hid < 64 || (hid % 64)) {
        if (g) snprintf(g->err,sizeof g->err,"max_rows: hid must be a multiple of 64");
        return -1; }
    rc = reduce_f32_common(g, a_off, scratch_off, rows*hid, rows, hid/64,
        kern_path("GOYA_TPC_MAX_F32_ROWS","k_max_rows_f32.o"), tpc_idx);
    if (rc) return rc;
    return reduce_f32_common(g, scratch_off, out_off, rows*64, rows, 64,
        kern_path("GOYA_TPC_MAX_F32_FOLD","k_max_fold_f32.o"), tpc_idx);
}

int gmme_tpc_dotrows_f32(gmme *g, uint64_t a_off, uint64_t b_off,
                         uint64_t out_off, int rows, int hid, int tpc_idx)
{
    if (!g || g->fd < 0 || rows < 1 || hid < 1) return -1;
    return reduce2_f32_common(g, a_off, b_off, out_off, rows*hid, rows, hid,
        kern_path("GOYA_TPC_DOTROWS_F32","k_dotrows_f32.o"), tpc_idx);
}

int gmme_tpc_sumsq_f32_rows(gmme *g, uint64_t a_off, uint64_t out_off,
                            int rows, int hid, int tpc_idx)
{
    if (!g || g->fd < 0 || rows < 1 || hid < 1) return -1;
    return reduce_f32_common(g, a_off, out_off, rows*hid, rows, hid,
        kern_path("GOYA_TPC_SUMSQ_F32_ROWS","k_sumsq_f32_rows.o"), tpc_idx);
}

int gmme_tpc_reduce_f32_rows(gmme *g, uint64_t a_off, uint64_t out_off,
                             int rows, int hid, int tpc_idx)
{
    if (!g || g->fd < 0 || rows < 1 || hid < 1) return -1;
    return reduce_f32_common(g, a_off, out_off, rows*hid, rows, hid,
        kern_path("GOYA_TPC_REDUCE_F32_ROWS","k_reduce_f32_rows.o"), tpc_idx);
}

int gmme_tpc_reduce_f32_pass1(gmme *g, uint64_t a_off, uint64_t out_off,
                              int K, int tpc_idx)
{
    if (!g || g->fd < 0 || K < 64 || (K % 64)) return -1;
    return reduce_f32_common(g, a_off, out_off, K, 1, K/64,
        kern_path("GOYA_TPC_REDUCE_F32_STRIDED","k_reduce_f32_strided.o"), tpc_idx);
}

int gmme_tpc_reduce_f32_resident(gmme *g, uint64_t a_off, uint64_t out_off,
                                 int K, int tpc_idx)
{
    return reduce_f32_common(g, a_off, out_off, K, 1, K,
        kern_path("GOYA_TPC_REDUCE_F32_KERNEL","k_reduce_f32_1d.o"), tpc_idx);
}

/* Two-input row reduction: same descriptor and index space as
   reduce_f32_common, but tensor 1 is a real second operand rather than a
   dummy, so it needs its own DRAM->SRAM stage. */
static int reduce2_f32_common(gmme *g, uint64_t a_off, uint64_t b_off,
                              uint64_t out_off, int dma_elems, int tid0,
                              int tid1, const char *kpath, int tpc_idx)
{
    int rc;
    if (!g || g->fd < 0) return -1;
    if ((size_t)(dma_elems+64)*4 > 0x100000) {
        snprintf(g->err,sizeof g->err,"dotrows: too large"); return -1; }
    /* stage operand B first; reduce_f32_common stages A and fires. */
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->C + b_off,
        g->sram + tpc_in2_off(g,tpc_idx), (uint32_t)(dma_elems*4), HL_DMA_DRAM_TO_SRAM);
    rc = reduce_f32_common(g, a_off, out_off, dma_elems, tid0, tid1, kpath, tpc_idx);
    return rc;
}

static int reduce_f32_common(gmme *g, uint64_t a_off, uint64_t out_off,
                             int dma_elems, int tid0, int tid1, const char *kpath, int tpc_idx)
{
    int K = dma_elems;
    uint32_t o = 0, eo, tq; int kslot, rc, dummy_b = 1; void *cfg, *eng; uint64_t kaddr;
    if (!g || g->fd < 0 || K < 1) return -1;
    if (tpc_idx < 0 || tpc_idx > 7) { snprintf(g->err,sizeof g->err,"reduce_f32: idx 0..7"); return -1; }
    if ((size_t)(K+64)*4 > 0x100000) { snprintf(g->err,sizeof g->err,"reduce_f32: K too large"); return -1; }
    kslot = gmme_tpc_register_kernel(g, kpath);
    if (kslot < 0) return -1;

    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->C + a_off,
        g->sram + tpc_in_off(g,tpc_idx), (uint32_t)(K*4), HL_DMA_DRAM_TO_SRAM);
    memset(g->hb, 0, (size_t)tid0*64*4 > 4096u ? 4096u : (size_t)tid0*64*4);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->hva,
        g->sram + tpc_in_off(g,tpc_idx) + (uint64_t)K*4, 64*4, HL_DMA_HOST_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->hva,
        g->sram + tpc_out_off(g,tpc_idx), (uint32_t)(tid0*64*4), HL_DMA_HOST_TO_SRAM);

    if (tpc_cbs(g, tpc_idx)) return -1;
    g->sram_dirty = 1;   /* the MME must re-stage its CBs after this */
    cfg = g->cb_cfg[tpc_idx]; kaddr = g->sram + KERNEL_SLOT(kslot);
    o = tpc_desc(g, tpc_idx, cfg, kaddr, K+64, &dummy_b, K+64, tid0, tid1, tid0*64);
    tq = hltests_get_tpc_qid(g->fd, tpc_idx, STREAM0);
    eng = g->cb_eng[tpc_idx];
    {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
        p.eb=EB_FALSE; p.mb=MB_TRUE;
        p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p.cp_dma.size = o;
        eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o, HL_DMA_HOST_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), eo, HL_DMA_HOST_TO_SRAM);
    rc = submit_wait(g, eng, eo, tq);
    if (rc) { snprintf(g->err,sizeof g->err,"reduce_f32: CS#1 failed"); return -1; }
    {   uint32_t o2 = cw(g->fd,cfg,0,TPC_EXECUTE,1,1), e2;
        struct hltests_pkt_info p2; memset(&p2,0,sizeof p2);
        p2.eb=EB_FALSE; p2.mb=MB_TRUE;
        p2.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p2.cp_dma.size = o2;
        e2 = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p2);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), e2, HL_DMA_HOST_TO_SRAM);
        rc = submit_wait(g, eng, e2, tq);
        if (rc) { snprintf(g->err,sizeof g->err,"reduce_f32: EXECUTE failed"); return -1; }
    }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        g->sram + tpc_out_off(g,tpc_idx), g->C + out_off,
        (uint32_t)(tid0*64*4), HL_DMA_SRAM_TO_DRAM);
    return 0;
}

int gmme_tpc_requant_i32_i8(gmme *g, uint64_t a_off, uint64_t b_off,
                            uint64_t out_off, int n, int tpc_idx)
{
    uint32_t o = 0, eo, tq; int kslot, rc; void *cfg, *eng; uint64_t kaddr;
    int tid0 = n / 256;
    if (!g || g->fd < 0 || n < 1) return -1;
    if (tpc_idx < 0 || tpc_idx > 7) { snprintf(g->err,sizeof g->err,"requant: idx 0..7"); return -1; }
    if (n % 256) { snprintf(g->err,sizeof g->err,
        "requant: n must be a multiple of 256 (one int8 vector); got %d", n); return -1; }
    if ((size_t)n*4 > 0x100000) { snprintf(g->err,sizeof g->err,"requant: input too large"); return -1; }
    kslot = gmme_tpc_register_kernel(g, kern_path("GOYA_TPC_REQUANT_KERNEL","k_requant_i32_i8.o"));
    if (kslot < 0) return -1;

    /* both operands staged DRAM -> SRAM; nothing goes through the host */
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->C + a_off,
        g->sram + tpc_in_off(g,tpc_idx),  (uint32_t)(n*4), HL_DMA_DRAM_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->C + b_off,
        g->sram + tpc_in2_off(g,tpc_idx), (uint32_t)(n*4), HL_DMA_DRAM_TO_SRAM);
    /* Clear the output region. It is n BYTES here, not n*4 -- and it must be
       cleared, or a kernel that runs zero iterations returns the previous
       result and looks like a pass. */
    memset(g->hb, 0, (size_t)n);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
        g->sram + tpc_out_off(g,tpc_idx), (uint32_t)n, HL_DMA_HOST_TO_SRAM);

    if (tpc_cbs(g, tpc_idx)) return -1;
    g->sram_dirty = 1;   /* the MME must re-stage its CBs after this */
    cfg = g->cb_cfg[tpc_idx]; kaddr = g->sram + KERNEL_SLOT(kslot);
    o = tpc_desc_requant(g, tpc_idx, cfg, kaddr, n, tid0);
    tq = hltests_get_tpc_qid(g->fd, tpc_idx, STREAM0);
    eng = g->cb_eng[tpc_idx];
    {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
        p.eb=EB_FALSE; p.mb=MB_TRUE;
        p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p.cp_dma.size = o;
        eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o, HL_DMA_HOST_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), eo, HL_DMA_HOST_TO_SRAM);
    rc = submit_wait(g, eng, eo, tq);
    if (rc) { snprintf(g->err,sizeof g->err,"requant: CS#1 failed"); return -1; }
    {   uint32_t o2 = cw(g->fd,cfg,0,TPC_EXECUTE,1,1), e2;
        struct hltests_pkt_info p2; memset(&p2,0,sizeof p2);
        p2.eb=EB_FALSE; p2.mb=MB_TRUE;
        p2.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p2.cp_dma.size = o2;
        e2 = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p2);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), e2, HL_DMA_HOST_TO_SRAM);
        rc = submit_wait(g, eng, e2, tq);
        if (rc) { snprintf(g->err,sizeof g->err,"requant: EXECUTE failed"); return -1; }
    }
    /* result stays on the card: SRAM -> DRAM, n BYTES */
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        g->sram + tpc_out_off(g,tpc_idx), g->C + out_off,
        (uint32_t)n, HL_DMA_SRAM_TO_DRAM);
    return 0;
}

/* ---------------------------------------------------------------------------
 * VENDOR TPC KERNELS.
 *
 * A vendor kernel declares its own convention in its ELF, section .IndexMap,
 * in plain text. softmax_dim0_f32 says:
 *     IndexSpace:[0].[Input] .{ 64, 4, 1, 1, 0 }
 *                [1].[Output].{ 64, 4, 1, 1, 0 }
 * i.e. INPUT is tensor 0 and OUTPUT is tensor 1 -- not the 0/1/3 layout our
 * hand-authored kernels use -- 64 lanes per index-space step, 4-byte elements.
 *
 * tpc_desc() is deliberately NOT reused or modified: every bit-exact result in
 * this library depends on it. This is a separate builder for a separate ABI.
 * -------------------------------------------------------------------------*/
static uint32_t tpc_desc_vendor(gmme *g, int tpc_idx, void *cfg, uint64_t kaddr,
                                int n, int tid0)
{
    int v_tid1 = 1, v_d1 = 1, v_d0 = n;
    { const char *e;
      if ((e=getenv("GMME_VENDOR_TID1"))) v_tid1 = atoi(e);
      if ((e=getenv("GMME_VENDOR_DIM1"))) v_d1   = atoi(e);
      if ((e=getenv("GMME_VENDOR_DIM0"))) v_d0   = atoi(e); }
    uint32_t o = 0; int t, d;
    uint64_t taddr[4];
    taddr[0] = g->sram + tpc_in_off (g, tpc_idx);   /* Input  (vendor slot 0) */
    taddr[1] = g->sram + tpc_in2_off(g, tpc_idx);   /* OUTPUT (vendor slot 1) */
    taddr[2] = g->sram + tpc_in_off (g, tpc_idx);
    taddr[3] = g->sram + tpc_out_off(g, tpc_idx);
    o = cw(g->fd,cfg,o,TPC0_CFG_ARUSER,ARUSER_MMBP,0);
    o = cw(g->fd,cfg,o,TPC0_CFG_AWUSER,ARUSER_MMBP,0);
    for (t = 0; t < 4; t++) {
        uint32_t b = QM_TENSOR_BASE + t*QM_TENSOR_STRIDE;
        o = cw(g->fd,cfg,o,b+T_BASE_ADDR_LOW ,(uint32_t)(taddr[t]&0xFFFFFFFF),0);
        o = cw(g->fd,cfg,o,b+T_BASE_ADDR_HIGH,(uint32_t)(taddr[t]>>32),0);
        o = cw(g->fd,cfg,o,b+T_PADDING_VALUE ,0,0);
        /* FP32. INT32 is 2 and is hardcoded in tpc_desc; the dtype enum order
           INT8,INT16,INT32,UINT8,UINT16,UINT32,BF16,FP32,FP16 puts FP32 at 7. */
        o = cw(g->fd,cfg,o,b+T_TENSOR_CONFIG ,(7u<<0)|(1u<<8),0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_SIZE      ,(uint32_t)v_d0,0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_STRIDE    ,4,0);
        o = cw(g->fd,cfg,o,b+T_DIM_0_BASE_OFFSET,0,0);
        /* DIM_1 must be a REAL row, not zero. softmax_dim0 reduces along dim 0
           and iterates rows along dim 1; DIM_1_SIZE=0 means no rows, so the
           kernel runs zero iterations and a perfectly successful CS writes
           nothing. One row of n elements: size 1, stride n*4 bytes. */
        o = cw(g->fd,cfg,o,b+0x1C+0,(uint32_t)v_d1,0);  /* DIM_1_SIZE          */
        o = cw(g->fd,cfg,o,b+0x1C+4,(uint32_t)(n*4),0);/* DIM_1_STRIDE = n*4   */
        o = cw(g->fd,cfg,o,b+0x1C+8,0,0);              /* DIM_1_BASE_OFFSET    */
        for (d = 2; d <= 4; d++) {                 /* DIM_2..4 persist: clear */
            uint32_t doff = b + 0x1C + (d-1)*0x0C;
            o = cw(g->fd,cfg,o,doff+0,0,0);
            o = cw(g->fd,cfg,o,doff+4,0,0);
            o = cw(g->fd,cfg,o,doff+8,0,0);
        }
    }
    o = cw(g->fd,cfg,o,QM_BASE_ADDR_LOW ,(uint32_t)(kaddr&0xFFFFFFFF),0);
    o = cw(g->fd,cfg,o,QM_BASE_ADDR_HIGH,(uint32_t)(kaddr>>32),0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_0,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_0,(uint32_t)tid0,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_1,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_1,(uint32_t)v_tid1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_2,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_2,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_3,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_3,1,0);
    o = cw(g->fd,cfg,o,QM_TID_BASE_DIM_4,0,0); o = cw(g->fd,cfg,o,QM_TID_SIZE_DIM_4,1,0);
    o = cw(g->fd,cfg,o,QM_KERNEL_CONFIG,0x2002,0);
    o = cw(g->fd,cfg,o,ICACHE_BASE_LOW ,(uint32_t)(kaddr&0xFFFFFFFF),0);
    o = cw(g->fd,cfg,o,ICACHE_BASE_HIGH,(uint32_t)(kaddr>>32),0);
    /* THE MISSING PIECE. Pointing ICACHE_BASE at a kernel does nothing on its
       own -- without INVALIDATE|PREFETCH the core executes whatever was already
       cached, so a vendor kernel gives rc=0 and writes nothing. Eight barrier
       NOPs after, or the TPC starts before the icache is filled. */
    o = cw(g->fd,cfg,o,TPC_CMD,TPC_CMD_ICACHE_INV|TPC_CMD_ICACHE_PF64,1);
    { int q; for (q = 0; q < 8; q++)
        o = hltests_add_nop_pkt(g->fd, cfg, o, EB_TRUE, MB_TRUE); }
    return o;
}

static int tpc_fire(gmme *g, const char *kpath,
                    const int32_t *in_a, int n_a,
                    const int32_t *in_b, int n_b,
                    int tid0, int tid1,
                    int32_t *out, int n_out, int tpc_idx)
{
    uint32_t o = 0, eo, tq; int kslot;
    void *cfg, *eng; int rc;
    uint64_t kaddr;

    if (!g || g->fd < 0) return -1;
    if (n_a < 1 || tid0 < 1 || tid1 < 1 || n_out < 1) {
        snprintf(g->err,sizeof g->err,"tpc: bad shape"); return -1; }
    if ((size_t)n_a*4 > 0x100000 || (in_b && (size_t)n_b*4 > 0x100000)) {
        snprintf(g->err,sizeof g->err,"tpc: input too large"); return -1; }
    if (tpc_idx < 0 || tpc_idx > 7) {
        snprintf(g->err,sizeof g->err,"tpc: tpc_idx must be 0..7"); return -1; }

    /* Stage kernel, input and a cleared output region into SRAM. An internal
       CB is CB_ALLOC_USER: the driver uses the SRAM address verbatim and copies
       nothing, so whatever is in SRAM is what runs. */
    /* Resolve to a resident slot. Uploads only the first time this kernel is
       seen; after that any core can fire it with no DMA. */
    kslot = gmme_tpc_register_kernel(g, kpath);
    if (kslot < 0) return -1;
    memcpy(g->hb, in_a, (size_t)n_a*4);
PB();
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
                         g->sram+tpc_in_off(g, tpc_idx),(uint32_t)(n_a*4),HL_DMA_HOST_TO_SRAM);
    PE("dma: stage input");
    if (in_b) {
        memcpy(g->hb, in_b, (size_t)n_b*4);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
                             g->sram+tpc_in2_off(g, tpc_idx),(uint32_t)(n_b*4),HL_DMA_HOST_TO_SRAM);
    }
    memset(g->hb, 0, 4096);
PB();
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
                         g->sram+tpc_out_off(g, tpc_idx),4096,HL_DMA_HOST_TO_SRAM);
    PE("dma: clear output");

    if (tpc_cbs(g, tpc_idx)) return -1;
    g->sram_dirty = 1;   /* the MME must re-stage its CBs after this */
    cfg = g->cb_cfg[tpc_idx];
    kaddr = g->sram + KERNEL_SLOT(kslot);
    { int tsize[4]; tsize[0]=n_a; tsize[1]= in_b ? n_b : n_a; tsize[2]=n_a; tsize[3]=n_out;
      (void)tsize; }

    o = tpc_desc(g, tpc_idx, cfg, kaddr, n_a, in_b, n_b, tid0, tid1, n_out);
PB();
    tq = hltests_get_tpc_qid(g->fd, tpc_idx, STREAM0);
    PE("build cfg CB (cw loop)");

    /* CS#1: program the descriptor and prefetch the icache. */
    eng = g->cb_eng[tpc_idx];
    {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
        p.eb=EB_FALSE; p.mb=MB_TRUE;
        p.cp_dma.src_addr = g->sram + tpc_cfg_off(g, tpc_idx); p.cp_dma.size = o;
        eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
PB();
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g, tpc_idx), o, HL_DMA_HOST_TO_SRAM);
    PE("dma: cfg->SRAM (CS1)");
PB();
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g, tpc_idx), eo, HL_DMA_HOST_TO_SRAM);
    PE("dma: eng->SRAM (CS1)");
PB();
    rc = submit_wait(g, eng, eo, tq);
    PE("submit_wait CS1");
    /* cfg/eng are cached in g -- not destroyed here. */
    if (rc) { snprintf(g->err,sizeof g->err,"tpc: CS#1 (descriptor+icache) failed or never settled"); return -1; }

    /* CS#2: fire EXECUTE. An engine barrier waits for the ENGINE, not for an
       icache prefetch to land, so the trigger must be a separate submission. */
    {   void *cfg2 = g->cb_cfg[tpc_idx];
        uint32_t o2 = cw(g->fd,cfg2,0,TPC_EXECUTE,1,1), e2;
        void *eng2 = g->cb_eng[tpc_idx];
        struct hltests_pkt_info p2; memset(&p2,0,sizeof p2);
        p2.eb=EB_FALSE; p2.mb=MB_TRUE;
        p2.cp_dma.src_addr = g->sram + tpc_cfg_off(g, tpc_idx); p2.cp_dma.size = o2;
        e2 = hltests_add_cp_dma_pkt(g->fd, eng2, 0, &p2);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg2), g->sram+tpc_cfg_off(g, tpc_idx), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng2), g->sram+tpc_eng_off(g, tpc_idx), e2, HL_DMA_HOST_TO_SRAM);
PB();
        rc = submit_wait(g, eng2, e2, tq);
    PE("submit_wait CS2 (EXECUTE)");
        /* cached -- not destroyed. */
        if (rc) { snprintf(g->err,sizeof g->err,"tpc: EXECUTE failed or never settled"); return -1; } }

PB();
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        g->sram+tpc_out_off(g, tpc_idx), g->hva, (uint32_t)(n_out*4), HL_DMA_SRAM_TO_HOST);
    PE("dma: read back result");
    memcpy(out, g->hb, (size_t)n_out*4);
    g->fires++;
    return 0;
}


/* Fire a vendor kernel: input -> tensor 0, output read back from tensor 1.
   Same two-CS protocol as tpc_fire (descriptor+icache, then EXECUTE); only the
   descriptor and the readback slot differ. */
int gmme_tpc_fire_vendor(gmme *g, const char *kpath,
                         const void *in, int n, void *out, int tpc_idx)
{
    uint32_t o, eo, tq; int kslot; void *cfg, *eng; int rc;
    uint64_t kaddr; int tid0;

    if (!g || g->fd < 0) return -1;
    if (n < 1) { snprintf(g->err,sizeof g->err,"vendor: bad n"); return -1; }
    if ((size_t)n*4 > 0x100000) { snprintf(g->err,sizeof g->err,"vendor: too large"); return -1; }
    if (tpc_idx < 0 || tpc_idx > 7) { snprintf(g->err,sizeof g->err,"vendor: core 0..7"); return -1; }

    kslot = gmme_tpc_register_kernel(g, kpath);
    if (kslot < 0) return -1;

    tid0 = (n + 63) / 64;                 /* .IndexMap: 64 lanes per step */
    { const char *e;
      if ((e=getenv("GMME_VENDOR_TID0"))) tid0 = atoi(e); }

    memcpy(g->hb, in, (size_t)n*4);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
                         g->sram+tpc_in_off(g,tpc_idx),(uint32_t)(n*4),HL_DMA_HOST_TO_SRAM);
    /* Clear EVERY candidate output slot, so "nonzero afterwards" unambiguously
       means the kernel wrote there rather than leftover from an earlier fire. */
    memset(g->hb, 0, (size_t)n*4);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
                         g->sram+tpc_in2_off(g,tpc_idx),(uint32_t)(n*4),HL_DMA_HOST_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
                         g->sram+tpc_out_off(g,tpc_idx),(uint32_t)(n*4),HL_DMA_HOST_TO_SRAM);

    if (tpc_cbs(g, tpc_idx)) return -1;
    g->sram_dirty = 1;   /* the MME must re-stage its CBs after this */
    cfg   = g->cb_cfg[tpc_idx];
    kaddr = g->sram + KERNEL_SLOT(kslot);
    o     = tpc_desc_vendor(g, tpc_idx, cfg, kaddr, n, tid0);
    tq    = hltests_get_tpc_qid(g->fd, tpc_idx, STREAM0);

    eng = g->cb_eng[tpc_idx];
    {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
        p.eb=EB_FALSE; p.mb=MB_TRUE;
        p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p.cp_dma.size = o;
        eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o, HL_DMA_HOST_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), eo, HL_DMA_HOST_TO_SRAM);
    rc = submit_wait(g, eng, eo, tq);
    if (rc) { snprintf(g->err,sizeof g->err,"vendor: CS#1 failed"); return -1; }

    {   void *cfg2 = g->cb_cfg[tpc_idx];
        uint32_t o2 = cw(g->fd,cfg2,0,TPC_EXECUTE,1,1), e2;
        void *eng2 = g->cb_eng[tpc_idx];
        struct hltests_pkt_info p2; memset(&p2,0,sizeof p2);
        p2.eb=EB_FALSE; p2.mb=MB_TRUE;
        p2.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p2.cp_dma.size = o2;
        e2 = hltests_add_cp_dma_pkt(g->fd, eng2, 0, &p2);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg2), g->sram+tpc_cfg_off(g,tpc_idx), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng2), g->sram+tpc_eng_off(g,tpc_idx), e2, HL_DMA_HOST_TO_SRAM);
        rc = submit_wait(g, eng2, e2, tq);
        if (rc) { snprintf(g->err,sizeof g->err,"vendor: EXECUTE failed"); return -1; } }

    /* READ BACK FROM TENSOR 1 -- the vendor kernel's declared Output slot. */
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        g->sram+tpc_in2_off(g,tpc_idx), g->hva, (uint32_t)(n*4), HL_DMA_SRAM_TO_HOST);
    memcpy(out, g->hb, (size_t)n*4);
    g->fires++;
    return 0;
}


/* Diagnostic: read any staged SRAM region back to host. slot 0=in, 1=in2,
   2=out, 3=cfg. Used to find out WHERE a vendor kernel actually wrote. */
int gmme_tpc_peek(gmme *g, int slot, int n, void *out, int tpc_idx)
{
    uint32_t off;
    if (!g || g->fd < 0 || n < 1) return -1;
    switch (slot) {
      case 0: off = tpc_in_off (g,tpc_idx); break;
      case 1: off = tpc_in2_off(g,tpc_idx); break;
      case 2: off = tpc_out_off(g,tpc_idx); break;
      default: return -1;
    }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        g->sram+off, g->hva, (uint32_t)(n*4), HL_DMA_SRAM_TO_HOST);
    memcpy(out, g->hb, (size_t)n*4);
    return 0;
}


/* Batched VENDOR fire: one 64-lane vector per core, all cores in flight.
   Mirrors gmme_tpc_fire_many exactly; only the descriptor builder, the cleared
   slot and the collect slot differ (vendor output = tensor 1, not 3).
   Requires per-core SRAM partitions (GMME_TPC_PART), same as the batch path. */
int gmme_tpc_fire_vendor_many(gmme *g, const char *kpath,
                              const float *in, float *out, int n_each, int njobs)
{
    uint64_t seq[8]; void *dma[8]; uint32_t tq[8];
    int i, kslot, rc = 0;
    if (!g || g->fd < 0 || njobs < 1 || njobs > 8 || n_each < 1) return -1;
    kslot = gmme_tpc_register_kernel(g, kpath);
    if (kslot < 0) return -1;

    for (i = 0; i < njobs; i++) {
        int c = i; uint32_t o, eo; void *cfg, *eng;
        if (tpc_cbs(g, c)) return -1;
        cfg = g->cb_cfg[c]; eng = g->cb_eng[c];
        memcpy(g->hb, in + (size_t)i*n_each, (size_t)n_each*4);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
            g->sram+tpc_in_off(g,c),(uint32_t)(n_each*4),HL_DMA_HOST_TO_SRAM);
        memset(g->hb, 0, (size_t)n_each*4);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
            g->sram+tpc_in2_off(g,c),(uint32_t)(n_each*4),HL_DMA_HOST_TO_SRAM);
        o = tpc_desc_vendor(g, c, cfg, g->sram + KERNEL_SLOT(kslot),
                            n_each, (n_each + 63) / 64);
        {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
            p.eb=EB_FALSE; p.mb=MB_TRUE;
            p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,c); p.cp_dma.size = o;
            eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,c), o, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,c), eo, HL_DMA_HOST_TO_SRAM);
        tq[i] = hltests_get_tpc_qid(g->fd, c, STREAM0);
        if (gmme_submit_async(g, eng, eo, tq[i], &seq[i], &dma[i])) {
            snprintf(g->err,sizeof g->err,"vendor_many: CS#1 submit failed (core %d)",c);
            return -1; }
    }
    if (gmme_wait_all(g, seq, dma, njobs)) {
        snprintf(g->err,sizeof g->err,"vendor_many: CS#1 never completed");
        gmme_settle(g); return -1; }

    for (i = 0; i < njobs; i++) {
        int c = i; void *cfg = g->cb_cfg[c], *eng = g->cb_eng[c];
        uint32_t o2 = cw(g->fd,cfg,0,TPC_EXECUTE,1,1), e2;
        {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
            p.eb=EB_FALSE; p.mb=MB_TRUE;
            p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,c); p.cp_dma.size = o2;
            e2 = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,c), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,c), e2, HL_DMA_HOST_TO_SRAM);
        if (gmme_submit_async(g, eng, e2, tq[i], &seq[i], &dma[i])) {
            snprintf(g->err,sizeof g->err,"vendor_many: EXECUTE submit failed (core %d)",c);
            return -1; }
    }
    if (gmme_wait_all(g, seq, dma, njobs)) rc = -1;
    if (gmme_settle(g)) rc = -1;
    if (rc) { snprintf(g->err,sizeof g->err,"vendor_many: EXECUTE failed or never settled"); return -1; }

    for (i = 0; i < njobs; i++) {          /* collect from tensor 1 */
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            g->sram+tpc_in2_off(g,i), g->hva, (uint32_t)(n_each*4), HL_DMA_SRAM_TO_HOST);
        memcpy(out + (size_t)i*n_each, g->hb, (size_t)n_each*4);
    }
    g->fires += njobs;
    return 0;
}


/* ON-CARD PATH. The MME leaves its result in device DRAM at g->C. This stages
   the TPC input straight from there with a DRAM->SRAM transfer -- the data
   never touches host memory. src_off is a byte offset into g->C.
   Output still comes back to host (the caller needs it); eliminating that too
   requires the second GEMM to read from SRAM, which is the next step. */
int gmme_tpc_fire_vendor_resident(gmme *g, const char *kpath,
                                  uint64_t src_off, int n, float *out, int tpc_idx)
{
    uint32_t o, eo, tq; int kslot; void *cfg, *eng; int rc; uint64_t kaddr; int tid0;
    if (!g || g->fd < 0 || n < 1) return -1;
    if (tpc_idx < 0 || tpc_idx > 7) return -1;
    kslot = gmme_tpc_register_kernel(g, kpath);
    if (kslot < 0) return -1;
    tid0 = (n + 63) / 64;
    { const char *e; if ((e=getenv("GMME_VENDOR_TID0"))) tid0 = atoi(e); }

    /* THE POINT: device-to-device. No host memory in this transfer. */
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        g->C + src_off, g->sram + tpc_in_off(g,tpc_idx),
        (uint32_t)(n*4), HL_DMA_DRAM_TO_SRAM);
    memset(g->hb, 0, (size_t)n*4);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
        g->sram+tpc_in2_off(g,tpc_idx),(uint32_t)(n*4),HL_DMA_HOST_TO_SRAM);

    if (tpc_cbs(g, tpc_idx)) return -1;
    g->sram_dirty = 1;   /* the MME must re-stage its CBs after this */
    cfg = g->cb_cfg[tpc_idx]; kaddr = g->sram + KERNEL_SLOT(kslot);
    o = tpc_desc_vendor(g, tpc_idx, cfg, kaddr, n, tid0);
    tq = hltests_get_tpc_qid(g->fd, tpc_idx, STREAM0);
    eng = g->cb_eng[tpc_idx];
    {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
        p.eb=EB_FALSE; p.mb=MB_TRUE;
        p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p.cp_dma.size = o;
        eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o, HL_DMA_HOST_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), eo, HL_DMA_HOST_TO_SRAM);
    rc = submit_wait(g, eng, eo, tq);
    if (rc) { snprintf(g->err,sizeof g->err,"resident: CS#1 failed"); return -1; }
    {   void *cfg2 = g->cb_cfg[tpc_idx]; uint32_t o2 = cw(g->fd,cfg2,0,TPC_EXECUTE,1,1), e2;
        void *eng2 = g->cb_eng[tpc_idx];
        struct hltests_pkt_info p2; memset(&p2,0,sizeof p2);
        p2.eb=EB_FALSE; p2.mb=MB_TRUE;
        p2.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p2.cp_dma.size = o2;
        e2 = hltests_add_cp_dma_pkt(g->fd, eng2, 0, &p2);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg2), g->sram+tpc_cfg_off(g,tpc_idx), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng2), g->sram+tpc_eng_off(g,tpc_idx), e2, HL_DMA_HOST_TO_SRAM);
        rc = submit_wait(g, eng2, e2, tq);
        if (rc) { snprintf(g->err,sizeof g->err,"resident: EXECUTE failed"); return -1; } }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        g->sram+tpc_in2_off(g,tpc_idx), g->hva, (uint32_t)(n*4), HL_DMA_SRAM_TO_HOST);
    memcpy(out, g->hb, (size_t)n*4);
    g->fires++;
    return 0;
}

#ifndef GOYA_TPC_DIR_DEFAULT
#define GOYA_TPC_DIR_DEFAULT "/usr/local/share/goya-stack/tpc-kernels"
#endif

/* A per-kernel env var wins, then $GOYA_TPC_DIR, then the install directory.
   No $HOME-relative default: under sudo HOME is /root, and a missing kernel
   only shows up later as "cannot load .text". */
static const char *kern_path(const char *env, const char *dflt) {
    static char buf[512];
    const char *p = getenv(env), *dir = getenv("GOYA_TPC_DIR");
    if (p) return p;
    snprintf(buf, sizeof buf, "%s/%s", dir ? dir : GOYA_TPC_DIR_DEFAULT, dflt);
    return buf;   /* let the caller report the real path it could not open */
}

/* Device-resident two-input elementwise op: BOTH inputs read from device DRAM
   (C at two offsets), result written back to device DRAM. No host memory in the
   data path at all. This is what a K-tiled GEMM needs to sum its partials
   without shipping them to the host and back. */
/* PARALLEL device-resident elementwise ops: N independent (a,b)->out jobs, each
   on its OWN core, staged in ONE submission and fired CONCURRENTLY. The serial
   version below folds 8 partials in 7 blocking steps on one core while seven
   cores idle. This does the same work in ceil(log2) levels with every core busy.
   Staging is device->SRAM throughout; no host memory in the data path. */
int gmme_tpc_op_resident_many(gmme *g, const char *kpath,
                              const uint64_t *a_off, const uint64_t *b_off,
                              const uint64_t *out_off, int n, int njobs)
{
    uint64_t seq[8]; void *dma[8]; uint32_t tq[8], eo[8]; void *eng[8];
    int i, kslot, rc, dummy_b = 1;
    int tid0 = (n + 63) / 64;
    if (!g || g->fd < 0 || njobs < 1 || njobs > 8 || n < 1) return -1;
    if (!g->tpc_part_base) {
        snprintf(g->err,sizeof g->err,"parallel fold needs GMME_TPC_PART"); return -1; }
    kslot = gmme_tpc_register_kernel(g, kpath);
    if (kslot < 0) return -1;

    /* ---- stage every core's operands in ONE submission ---- */
    {   struct stage_req rq[8*3]; int nq = 0;
        memset(g->hb, 0, 4096);
        for (i = 0; i < njobs; i++) {
            int c = i;
            rq[nq].src = g->C + a_off[i]; rq[nq].dst = g->sram + tpc_in_off(g,c);
            rq[nq].size = (uint32_t)(n*4); rq[nq].dir = HL_DMA_DRAM_TO_SRAM; nq++;
            rq[nq].src = g->C + b_off[i]; rq[nq].dst = g->sram + tpc_in2_off(g,c);
            rq[nq].size = (uint32_t)(n*4); rq[nq].dir = HL_DMA_DRAM_TO_SRAM; nq++;
            rq[nq].src = g->hva;          rq[nq].dst = g->sram + tpc_out_off(g,c);
            rq[nq].size = 4096;           rq[nq].dir = HL_DMA_HOST_TO_SRAM; nq++;
        }
PB();
        if (gmme_stage_multi(g, rq, nq)) {
            snprintf(g->err,sizeof g->err,"parallel fold: stage failed"); return -1; }
PE("fold: stage operands");
    }

    /* ---- program every core, submit CS#1 WITHOUT waiting ---- */
PB();
    for (i = 0; i < njobs; i++) {
        int c = i; uint32_t o;
        void *cfg;
        if (tpc_cbs(g, c)) return -1;
        cfg = g->cb_cfg[c]; eng[i] = g->cb_eng[c];
        o = tpc_desc(g, c, cfg, g->sram + KERNEL_SLOT(kslot),
                     n, &dummy_b, n, tid0, 1, n);
        {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
            p.eb=EB_FALSE; p.mb=MB_TRUE;
            p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,c); p.cp_dma.size = o;
            eo[i] = hltests_add_cp_dma_pkt(g->fd, eng[i], 0, &p); }
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,c), o, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng[i]), g->sram+tpc_eng_off(g,c), eo[i], HL_DMA_HOST_TO_SRAM);
        tq[i] = hltests_get_tpc_qid(g->fd, c, STREAM0);
    }
PE("fold: build+stage descriptors");
    /* ONE submission for every core's descriptor, not one per core. */
PB();
    if (gmme_submit_multiq(g, eng, eo, tq, njobs, &seq[0], &dma[0])) {
        snprintf(g->err,sizeof g->err,"parallel fold: CS#1 submit"); return -1; }
    if (gmme_wait_all(g, seq, dma, 1)) {
        snprintf(g->err,sizeof g->err,"parallel fold: CS#1 wait"); return -1; }
PE("fold: CS1 descriptor");

    /* ---- fire every EXECUTE, then wait once: THIS is the concurrent part ---- */
    for (i = 0; i < njobs; i++) {
        int c = i; void *cfg = g->cb_cfg[c]; uint32_t o2;
        o2 = cw(g->fd,cfg,0,TPC_EXECUTE,1,1);
        {   struct hltests_pkt_info p2; memset(&p2,0,sizeof p2);
            p2.eb=EB_FALSE; p2.mb=MB_TRUE;
            p2.cp_dma.src_addr = g->sram + tpc_cfg_off(g,c); p2.cp_dma.size = o2;
            eo[i] = hltests_add_cp_dma_pkt(g->fd, eng[i], 0, &p2); }
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,c), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng[i]), g->sram+tpc_eng_off(g,c), eo[i], HL_DMA_HOST_TO_SRAM);
    }
    /* ONE submission for every core's EXECUTE. */
PB();
    if (gmme_submit_multiq(g, eng, eo, tq, njobs, &seq[0], &dma[0])) {
        snprintf(g->err,sizeof g->err,"parallel fold: EXECUTE submit"); return -1; }
    rc = gmme_wait_all(g, seq, dma, 1);
PE("fold: EXECUTE");
    if (rc) { snprintf(g->err,sizeof g->err,"parallel fold: EXECUTE wait"); return -1; }
PB();
    if (settle(g)) return -1;
PE("fold: settle");

    /* ---- write every result back to DRAM in ONE submission ---- */
    {   struct stage_req rq[8]; int nq = 0;
        for (i = 0; i < njobs; i++) {
            rq[nq].src = g->sram + tpc_out_off(g,i); rq[nq].dst = g->C + out_off[i];
            rq[nq].size = (uint32_t)(n*4); rq[nq].dir = HL_DMA_SRAM_TO_DRAM; nq++;
        }
PB();
        if (gmme_stage_multi(g, rq, nq)) {
            snprintf(g->err,sizeof g->err,"parallel fold: writeback failed"); return -1; }
PE("fold: writeback");
    }
    return 0;
}

int gmme_tpc_op_resident(gmme *g, const char *kpath,
                         uint64_t a_off, uint64_t b_off, uint64_t out_off,
                         int n, int tpc_idx)
{
    uint32_t o = 0, eo, tq; int kslot, rc; void *cfg, *eng; uint64_t kaddr;
    int tid0 = (n + 63) / 64, dummy_b = 1;
    if (!g || g->fd < 0 || n < 1) return -1;
    if (tpc_idx < 0 || tpc_idx > 7) { snprintf(g->err,sizeof g->err,"tpc: idx 0..7"); return -1; }
    if ((size_t)n*4 > 0x100000) { snprintf(g->err,sizeof g->err,"tpc: input too large"); return -1; }
    kslot = gmme_tpc_register_kernel(g, kpath);
    if (kslot < 0) return -1;

    /* THE POINT: both operands staged DRAM -> SRAM, never through the host. */
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->C + a_off,
        g->sram + tpc_in_off(g,tpc_idx),  (uint32_t)(n*4), HL_DMA_DRAM_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE, g->C + b_off,
        g->sram + tpc_in2_off(g,tpc_idx), (uint32_t)(n*4), HL_DMA_DRAM_TO_SRAM);
    memset(g->hb, 0, 4096);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,g->hva,
        g->sram + tpc_out_off(g,tpc_idx), 4096, HL_DMA_HOST_TO_SRAM);

    if (tpc_cbs(g, tpc_idx)) return -1;
    g->sram_dirty = 1;   /* the MME must re-stage its CBs after this */
    cfg = g->cb_cfg[tpc_idx]; kaddr = g->sram + KERNEL_SLOT(kslot);
    o = tpc_desc(g, tpc_idx, cfg, kaddr, n, &dummy_b, n, tid0, 1, n);
    tq = hltests_get_tpc_qid(g->fd, tpc_idx, STREAM0);
    eng = g->cb_eng[tpc_idx];
    {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
        p.eb=EB_FALSE; p.mb=MB_TRUE;
        p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p.cp_dma.size = o;
        eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o, HL_DMA_HOST_TO_SRAM);
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), eo, HL_DMA_HOST_TO_SRAM);
    rc = submit_wait(g, eng, eo, tq);
    if (rc) { snprintf(g->err,sizeof g->err,"op_resident: CS#1 failed"); return -1; }
    {   uint32_t o2 = cw(g->fd,cfg,0,TPC_EXECUTE,1,1), e2;
        struct hltests_pkt_info p2; memset(&p2,0,sizeof p2);
        p2.eb=EB_FALSE; p2.mb=MB_TRUE;
        p2.cp_dma.src_addr = g->sram + tpc_cfg_off(g,tpc_idx); p2.cp_dma.size = o2;
        e2 = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p2);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,tpc_idx), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng), g->sram+tpc_eng_off(g,tpc_idx), e2, HL_DMA_HOST_TO_SRAM);
        rc = submit_wait(g, eng, e2, tq);
        if (rc) { snprintf(g->err,sizeof g->err,"op_resident: EXECUTE failed"); return -1; }
    }
    /* result stays on the card: SRAM -> DRAM */
    hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
        g->sram + tpc_out_off(g,tpc_idx), g->C + out_off,
        (uint32_t)(n*4), HL_DMA_SRAM_TO_DRAM);
    return 0;
}

int gmme_tpc_reduce_i32(gmme *g, const int32_t *in, int n_in, int K,
                        int32_t *out, int n_out, int tpc_idx) {
    if (!g || g->fd < 0) return -1;
    if (K < 1) { snprintf(g->err,sizeof g->err,"reduce: K must be >= 1"); return -1; }
    return tpc_fire(g, kern_path("GOYA_TPC_KERNEL","k_reduce_i32_1d.o"),
                    in, n_in, NULL, 0, 1, K, out, n_out, tpc_idx);
}

/* EW: c[n] = a[n] + b[n]. 64 int32 lanes per index-space step. */
int gmme_tpc_ew_add_i32(gmme *g, const int32_t *a, const int32_t *b, int n,
                        int32_t *out, int tpc_idx) {
    int steps;
    if (!g || g->fd < 0) return -1;
    if (n < 1) { snprintf(g->err,sizeof g->err,"ew: n must be >= 1"); return -1; }
    steps = (n + 63) / 64;
    return tpc_fire(g, kern_path("GOYA_TPC_EW_KERNEL","k_ew_add_i32.o"),
                    a, n, b, n, steps, 1, out, n, tpc_idx);
}

int gmme_tpc_cores(const gmme *g) { (void)g; return 8; }

/* ---- Concurrent multi-core fire --------------------------------------
 *
 * Split into phases so the MME can be in flight across them too. The batch
 * pays TWO barriers and ONE device-idle settle for the whole set, instead
 * of a wait plus a device-WIDE idle spin per operation -- that per-op spin
 * is what made running more than one engine impossible.
 */

int tpc_validate(gmme *g, void *jobsv, int n)
{
    gmme_tpc_job *jobs = (gmme_tpc_job *)jobsv;
    int used[8], i;
    if (n < 1 || n > 8) {
        snprintf(g->err,sizeof g->err,"fire_many: njobs must be 1..8"); return -1; }
    /* On the legacy map every core stages through the SAME SRAM buffers, so
       running them together would silently corrupt each other's input. */
    if (!g->tpc_part_base) {
        snprintf(g->err,sizeof g->err,
                 "concurrent fire needs per-core SRAM partitions (set GMME_TPC_PART)");
        return -1; }
    memset(used,0,sizeof used);
    for (i = 0; i < n; i++) {
        int c = jobs[i].core;
        if (c < 0 || c > 7 || used[c]) {
            snprintf(g->err,sizeof g->err,"fire_many: bad or duplicate core %d",c); return -1; }
        if (!jobs[i].kernel || jobs[i].n_a < 1 || jobs[i].n_out < 1 ||
            jobs[i].tid0 < 1 || jobs[i].tid1 < 1) {
            snprintf(g->err,sizeof g->err,"fire_many: bad job %d",i); return -1; }
        if ((size_t)jobs[i].n_a*4 > 0x100000 ||
            (jobs[i].b && (size_t)jobs[i].n_b*4 > 0x100000)) {
            snprintf(g->err,sizeof g->err,"fire_many: job %d input too large",i); return -1; }
        used[c] = 1;
    }
    return 0;
}

/* Phase 1: stage inputs and program every core, submitting CS#1 as we go.
   Staging DMAs are serial -- they share the host bounce buffer -- and that
   serial term is what limits scaling; the EXECUTEs are what overlap. */
int tpc_stage_all(gmme *g, void *jobsv, int n, uint64_t *seq, void **dma, uint32_t *tq)
{
    gmme_tpc_job *jobs = (gmme_tpc_job *)jobsv;
    /* Every core gets its OWN slice of the host bounce buffer, so all eight
       stage in ONE submission instead of blocking on each other.
       Previously each job memcpy'd into g->hb and had to WAIT for its DMA
       before the next job could touch the buffer: FIVE blocking round trips
       per job, forty for eight jobs, ~0.07 ms each. The cores, their SRAM
       partitions and their queues were always independent -- this one shared
       host buffer was the only thing making them serial. */
    const uint32_t SLICE = 0x40000;      /* 8 x 256 KB = the 2 MB host buffer */
    struct stage_req rq[8*5];
    uint32_t eo_a[8]; void *eng_a[8];
    int i, nq = 0, batched = 1;

    for (i = 0; i < n; i++) {
        size_t need = (size_t)jobs[i].n_a*4
                    + (jobs[i].b ? (size_t)jobs[i].n_b*4 : 0) + 4096;
        if (need > SLICE) { batched = 0; break; }   /* too big to slice: stay serial */
    }

    for (i = 0; i < n; i++) {
        gmme_tpc_job *j = &jobs[i];
        int c = j->core; uint32_t o, eo, off = 0; void *cfg, *eng;
        uint8_t *hs; uint64_t vs;
        int kslot = gmme_tpc_register_kernel(g, j->kernel);
        if (kslot < 0) return -1;
        if (tpc_cbs(g, c)) return -1;
        cfg = g->cb_cfg[c]; eng = g->cb_eng[c];

        hs = (uint8_t *)g->hb + (batched ? (uint64_t)i * SLICE : 0);
        vs = g->hva           + (batched ? (uint64_t)i * SLICE : 0);

        memcpy(hs + off, j->a, (size_t)j->n_a*4);
        if (batched) { rq[nq].src = vs + off; rq[nq].dst = g->sram + tpc_in_off(g,c);
                       rq[nq].size = (uint32_t)(j->n_a*4);
                       rq[nq].dir = HL_DMA_HOST_TO_SRAM; nq++; }
        else hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,vs+off,
                 g->sram+tpc_in_off(g,c),(uint32_t)(j->n_a*4),HL_DMA_HOST_TO_SRAM);
        off += (uint32_t)(j->n_a*4);

        if (j->b) {
            memcpy(hs + off, j->b, (size_t)j->n_b*4);
            if (batched) { rq[nq].src = vs + off; rq[nq].dst = g->sram + tpc_in2_off(g,c);
                           rq[nq].size = (uint32_t)(j->n_b*4);
                           rq[nq].dir = HL_DMA_HOST_TO_SRAM; nq++; }
            else hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,vs+off,
                     g->sram+tpc_in2_off(g,c),(uint32_t)(j->n_b*4),HL_DMA_HOST_TO_SRAM);
            off += (uint32_t)(j->n_b*4);
        }

        memset(hs + off, 0, 4096);
        if (batched) { rq[nq].src = vs + off; rq[nq].dst = g->sram + tpc_out_off(g,c);
                       rq[nq].size = 4096; rq[nq].dir = HL_DMA_HOST_TO_SRAM; nq++; }
        else hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,vs+off,
                 g->sram+tpc_out_off(g,c),4096,HL_DMA_HOST_TO_SRAM);

        o = tpc_desc(g, c, cfg, g->sram + KERNEL_SLOT(kslot),
                     j->n_a, j->b, j->n_b, j->tid0, j->tid1, j->n_out);
        {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
            p.eb=EB_FALSE; p.mb=MB_TRUE;
            p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,c); p.cp_dma.size = o;
            eo = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
        eo_a[i] = eo; eng_a[i] = eng;

        if (batched) {
            rq[nq].src = hltests_get_device_va_for_host_ptr(g->fd,cfg);
            rq[nq].dst = g->sram + tpc_cfg_off(g,c); rq[nq].size = o;
            rq[nq].dir = HL_DMA_HOST_TO_SRAM; nq++;
            rq[nq].src = hltests_get_device_va_for_host_ptr(g->fd,eng);
            rq[nq].dst = g->sram + tpc_eng_off(g,c); rq[nq].size = eo;
            rq[nq].dir = HL_DMA_HOST_TO_SRAM; nq++;
        } else {
            hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
                hltests_get_device_va_for_host_ptr(g->fd,cfg),
                g->sram+tpc_cfg_off(g,c), o, HL_DMA_HOST_TO_SRAM);
            hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
                hltests_get_device_va_for_host_ptr(g->fd,eng),
                g->sram+tpc_eng_off(g,c), eo, HL_DMA_HOST_TO_SRAM);
        }
    }

    /* ONE submission for every core's inputs, output-zero, config and engine CB. */
    if (batched && gmme_stage_multi(g, rq, nq)) {
        snprintf(g->err,sizeof g->err,"fire_many: batched stage failed"); return -1; }

    for (i = 0; i < n; i++) {
        tq[i] = hltests_get_tpc_qid(g->fd, jobs[i].core, STREAM0);
        if (gmme_submit_async(g, eng_a[i], eo_a[i], tq[i], &seq[i], &dma[i])) {
            snprintf(g->err,sizeof g->err,"fire_many: CS#1 submit failed (core %d)",
                     jobs[i].core);
            return -1; }
    }
    return 0;
}

/* Phase 2: trigger every core. This is the part that runs in parallel.
   EXECUTE must be its own submission -- an engine barrier waits for the
   ENGINE, not for the icache prefetch from CS#1 to land. */
int tpc_exec_all(gmme *g, void *jobsv, int n, uint64_t *seq, void **dma, uint32_t *tq)
{
    gmme_tpc_job *jobs = (gmme_tpc_job *)jobsv;
    int i;
    for (i = 0; i < n; i++) {
        int c = jobs[i].core;
        void *cfg = g->cb_cfg[c], *eng = g->cb_eng[c];
        uint32_t o2 = cw(g->fd,cfg,0,TPC_EXECUTE,1,1), e2;
        {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
            p.eb=EB_FALSE; p.mb=MB_TRUE;
            p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,c); p.cp_dma.size = o2;
            e2 = hltests_add_cp_dma_pkt(g->fd, eng, 0, &p); }
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg),
            g->sram+tpc_cfg_off(g,c), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng),
            g->sram+tpc_eng_off(g,c), e2, HL_DMA_HOST_TO_SRAM);
        if (gmme_submit_async(g, eng, e2, tq[i], &seq[i], &dma[i])) {
            snprintf(g->err,sizeof g->err,"fire_many: EXECUTE submit failed (core %d)",c);
            return -1; }
    }
    return 0;
}

void tpc_collect_all(gmme *g, void *jobsv, int n)
{
    gmme_tpc_job *jobs = (gmme_tpc_job *)jobsv;
    int i;
    for (i = 0; i < n; i++) {
        int c = jobs[i].core;
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            g->sram+tpc_out_off(g,c), g->hva,
            (uint32_t)(jobs[i].n_out*4), HL_DMA_SRAM_TO_HOST);
        memcpy(jobs[i].out, g->hb, (size_t)jobs[i].n_out*4);
    }
    g->fires += n;
}

int gmme_tpc_fire_many(gmme *g, gmme_tpc_job *jobs, int njobs)
{
    uint64_t seq[8]; void *dma[8]; uint32_t tq[8];
    int rc = 0;
    if (!g || g->fd < 0) return -1;
    if (tpc_validate(g, jobs, njobs)) return -1;
    if (tpc_stage_all(g, jobs, njobs, seq, dma, tq)) return -1;
    if (gmme_wait_all(g, seq, dma, njobs)) {
        snprintf(g->err,sizeof g->err,"fire_many: CS#1 did not complete");
        gmme_settle(g); return -1; }
    if (tpc_exec_all(g, jobs, njobs, seq, dma, tq)) { gmme_settle(g); return -1; }
    if (gmme_wait_all(g, seq, dma, njobs)) rc = -1;
    if (gmme_settle(g)) rc = -1;
    if (rc) { snprintf(g->err,sizeof g->err,"fire_many: EXECUTE failed or never settled");
              return -1; }
    tpc_collect_all(g, jobs, njobs);
    return 0;
}


/* ---- The wave -----------------------------------------------------------
   See goya_mme.h. Same shape as gmme_tpc_op_resident_many, with the kernel,
   operand set and index space chosen PER JOB. Every descriptor goes down in
   one CS, every EXECUTE in one CS, and the device-idle settle is paid once. */
int gmme_tpc_wave(gmme *g, const gmme_wave_job *jobs, int njobs)
{
    uint64_t seq[8]; void *dma[8]; uint32_t tq[8], eo[8]; void *eng[8];
    int kslot[8]; int i, rc; uint32_t zero_need = 4096; int dummy_b = 1;
    if (!g || g->fd < 0 || njobs < 1 || njobs > 8) return -1;
    if (!g->tpc_part_base) {
        snprintf(g->err,sizeof g->err,"wave needs per-core SRAM partitions (GMME_TPC_PART)"); return -1; }
    for (i = 0; i < njobs; i++) {
        int j;
        const gmme_wave_job *w = &jobs[i];
        if (!w->kernel || w->core < 0 || w->core > 7 || (w->nops != 2 && w->nops != 3)) {
            snprintf(g->err,sizeof g->err,"wave job %d: bad kernel/core/nops", i); return -1; }
        for (j = 0; j < i; j++) if (jobs[j].core == w->core) {
            snprintf(g->err,sizeof g->err,"wave: core %d used twice", w->core); return -1; }
        if (w->a_bytes + w->a_tail_zero > 0x100000u || w->b_bytes > 0x100000u ||
            w->out_bytes > 0x100000u || (w->nops == 3 && w->c_bytes > PART_IN3_CAP)) {
            snprintf(g->err,sizeof g->err,"wave job %d: operand exceeds the 1 MB lane scratch", i); return -1; }
        if (w->out_bytes > zero_need) zero_need = w->out_bytes;
        if (w->a_tail_zero > zero_need) zero_need = w->a_tail_zero;
        kslot[i] = gmme_tpc_register_kernel(g, w->kernel);
        if (kslot[i] < 0) return -1;
    }
    /* ---- stage every job's operands in ONE submission ---- */
    {   struct stage_req rq[8*5]; int nq = 0;
        memset(g->hb, 0, zero_need);
        for (i = 0; i < njobs; i++) {
            const gmme_wave_job *w = &jobs[i]; int c = w->core;
            rq[nq].src = g->C + w->a_off; rq[nq].dst = g->sram + tpc_in_off(g,c);
            rq[nq].size = w->a_bytes; rq[nq].dir = HL_DMA_DRAM_TO_SRAM; nq++;
            if (w->a_tail_zero) {
                rq[nq].src = g->hva; rq[nq].dst = g->sram + tpc_in_off(g,c) + w->a_bytes;
                rq[nq].size = w->a_tail_zero; rq[nq].dir = HL_DMA_HOST_TO_SRAM; nq++; }
            if (w->nops == 3 || w->b_bytes) {      /* has_b alone = point at in2, stage nothing */
                rq[nq].src = g->C + w->b_off; rq[nq].dst = g->sram + tpc_in2_off(g,c);
                rq[nq].size = w->b_bytes; rq[nq].dir = HL_DMA_DRAM_TO_SRAM; nq++; }
            if (w->nops == 3) {
                rq[nq].src = g->C + w->c_off; rq[nq].dst = g->sram + tpc_in3_off(g,c);
                rq[nq].size = w->c_bytes; rq[nq].dir = HL_DMA_DRAM_TO_SRAM; nq++; }
            rq[nq].src = g->hva; rq[nq].dst = g->sram + tpc_out_off(g,c);
            rq[nq].size = w->out_bytes < 4096u ? 4096u : w->out_bytes;
            rq[nq].dir = HL_DMA_HOST_TO_SRAM; nq++;
        }
        if (gmme_stage_multi(g, rq, nq)) {
            snprintf(g->err,sizeof g->err,"wave: stage failed"); return -1; }
    }
    g->sram_dirty = 1;   /* the MME must re-stage its CBs after this */
    /* ---- program every core, ONE descriptor submission ---- */
    for (i = 0; i < njobs; i++) {
        const gmme_wave_job *w = &jobs[i]; int c = w->core; uint32_t o; void *cfg;
        uint64_t kaddr = g->sram + KERNEL_SLOT(kslot[i]);
        if (tpc_cbs(g, c)) return -1;
        cfg = g->cb_cfg[c]; eng[i] = g->cb_eng[c];
        if (w->nops == 3)
            o = tpc_desc3(g, c, cfg, kaddr, w->n_a, w->tid0);
        else
            o = tpc_desc(g, c, cfg, kaddr, w->n_a, w->has_b ? &dummy_b : NULL,
                         w->n_b, w->tid0, w->tid1, w->n_out);
        {   struct hltests_pkt_info p; memset(&p,0,sizeof p);
            p.eb=EB_FALSE; p.mb=MB_TRUE;
            p.cp_dma.src_addr = g->sram + tpc_cfg_off(g,c); p.cp_dma.size = o;
            eo[i] = hltests_add_cp_dma_pkt(g->fd, eng[i], 0, &p); }
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,c), o, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng[i]), g->sram+tpc_eng_off(g,c), eo[i], HL_DMA_HOST_TO_SRAM);
        tq[i] = hltests_get_tpc_qid(g->fd, c, STREAM0);
    }
    if (gmme_submit_multiq(g, eng, eo, tq, njobs, &seq[0], &dma[0])) {
        snprintf(g->err,sizeof g->err,"wave: descriptor submit"); return -1; }
    if (gmme_wait_all(g, seq, dma, 1)) {
        snprintf(g->err,sizeof g->err,"wave: descriptor wait"); return -1; }
    /* ---- every EXECUTE in ONE submission, then settle ONCE ---- */
    for (i = 0; i < njobs; i++) {
        int c = jobs[i].core; void *cfg = g->cb_cfg[c]; uint32_t o2;
        o2 = cw(g->fd,cfg,0,TPC_EXECUTE,1,1);
        {   struct hltests_pkt_info p2; memset(&p2,0,sizeof p2);
            p2.eb=EB_FALSE; p2.mb=MB_TRUE;
            p2.cp_dma.src_addr = g->sram + tpc_cfg_off(g,c); p2.cp_dma.size = o2;
            eo[i] = hltests_add_cp_dma_pkt(g->fd, eng[i], 0, &p2); }
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,cfg), g->sram+tpc_cfg_off(g,c), o2, HL_DMA_HOST_TO_SRAM);
        hltests_dma_transfer(g->fd,g->dq,EB_FALSE,MB_TRUE,
            hltests_get_device_va_for_host_ptr(g->fd,eng[i]), g->sram+tpc_eng_off(g,c), eo[i], HL_DMA_HOST_TO_SRAM);
    }
    if (gmme_submit_multiq(g, eng, eo, tq, njobs, &seq[0], &dma[0])) {
        snprintf(g->err,sizeof g->err,"wave: EXECUTE submit"); return -1; }
    rc = gmme_wait_all(g, seq, dma, 1);
    if (rc) { snprintf(g->err,sizeof g->err,"wave: EXECUTE wait"); return -1; }
    if (settle(g)) { snprintf(g->err,sizeof g->err,"wave: device did not settle"); return -1; }
    /* ---- every result back to DRAM in ONE submission ---- */
    {   struct stage_req rq[8]; int nq = 0;
        for (i = 0; i < njobs; i++) {
            const gmme_wave_job *w = &jobs[i];
            rq[nq].src = g->sram + tpc_out_off(g,w->core); rq[nq].dst = g->C + w->out_off;
            rq[nq].size = w->out_bytes; rq[nq].dir = HL_DMA_SRAM_TO_DRAM; nq++;
        }
        if (gmme_stage_multi(g, rq, nq)) {
            snprintf(g->err,sizeof g->err,"wave: writeback failed"); return -1; }
    }
    return 0;
}
