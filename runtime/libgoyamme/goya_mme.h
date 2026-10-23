/* goya_mme.h -- integer GEMM on the Goya HL-1000 MME, through the front door.
 *
 *   C[M][N] (int32) = A[M][K] (int8) x B[K][N] (int8)
 *
 * All matrices are row-major. Stock driver, stock firmware, no debugfs, no BAR0.
 * M is tiled automatically; K and N must fit the engine (see GMME_MAX_K/N).
 */
#ifndef GOYA_MME_H
#define GOYA_MME_H
#include <stdint.h>
#include <stddef.h>

/* K is NOT capped by the engine. "256, hardware limit" was a header constant that
   was never true; K is bit-exact to 16384 (measured) and native deep-K is several
   times faster than tiling K. N=256 IS real -- it is the MME array width, and N>256
   poisons the process context until the device is reopened. */
#define GMME_MAX_K 16384   /* largest VERIFIED K; no engine ceiling was found */
#define GMME_MAX_N   256   /* output width -- REAL hardware limit (array width) */
/* NOT a hardware limit -- rows=128 is bit-exact and rows=64 is the measured
   optimum. The old value of 8 was folklore inherited from the transposed-A
   descriptor. Kept only as a sanity ceiling. */
#define GMME_MAX_ROWS      256
#define GMME_ROWS_PER_FIRE  64   /* measured optimum, all K, 2026-09-01 */

typedef struct gmme gmme;

/* busid like "0000:01:00.0", or NULL to take the first free card. */
gmme       *gmme_open(const char *busid);
void        gmme_close(gmme *g);
const char *gmme_error(const gmme *g);

/* Returns 0 on success, negative on error. M is unlimited (tiled internally). */
int gmme_gemm_i8(gmme *g, const int8_t *A, const int8_t *B, int32_t *C,
                 int M, int K, int N);

/* ---------------------------------------------------------------------------
 * Resident weights.
 *
 * The card holds the weights; the caller only ever hands it input. Load W once
 * (at placement time), then stream activations against it. gmme_apply() has no
 * way to pass weights, so it cannot accidentally re-send them over PCIe -- the
 * expensive mistake is unrepresentable rather than merely discouraged.
 *
 *     gmme_load_weights(g, W, K, N);        // once
 *     gmme_apply(g, X, Y, M);               // per token: X[MxK] -> Y[MxN]
 * -------------------------------------------------------------------------*/
/* Weight arena: MANY named slices resident in the card's DRAM at once. */
int      gmme_load_slice(gmme *g, const char *name, const int8_t *W, int K, int N);
int      gmme_apply_slice(gmme *g, const char *name, const int8_t *X, int32_t *Y, int M);
int      gmme_slice_count(const gmme *g);

/* N independent components -- different resident weight slices, different
   activations, different outputs -- in ONE command buffer and ONE barrier.
   Every descriptor names its own A/B/C base, so a CB need not belong to one
   GEMM. All slices must share K,N. One activation row per component.
   Returns 0, or -1 with gmme_error() set. Max n = 76. */
int gmme_apply_batch(gmme *g, const char **names, const int8_t *X,
                     int32_t *Y, int n);

int      gmme_slice_slots(const gmme *g);   /* arena capacity, in slices */
/* Host RAM is the overflow tier: a slice always keeps its host copy, so the
   arena holds the ACTIVE set and cold slices spill. Promotion is one DMA. */
int      gmme_slices_on_card(const gmme *g);
uint64_t gmme_promotions(const gmme *g);
uint64_t gmme_spills(const gmme *g);
uint64_t gmme_arena_used(const gmme *g);
uint64_t gmme_arena_size(const gmme *g);
/* Bytes per arena slot -- decided at runtime from the largest slice loaded,
   NOT from GMME_MAX_K. 0 until the first gmme_load_slice. */
uint64_t gmme_slot_bytes(const gmme *g);

int  gmme_load_weights(gmme *g, const int8_t *W, int K, int N);
void gmme_drop_weights(gmme *g);
/* K and N come from the resident weights. Returns -1 if none are loaded. */
int  gmme_apply(gmme *g, const int8_t *X, int32_t *Y, int M);
/* Shape of the resident weights, or 0 if none. */
int  gmme_weights_k(const gmme *g);
int  gmme_weights_n(const gmme *g);

/* Largest M that runs in a single fire for this K. */
int gmme_rows_per_fire(int K);

/* Engine reached idle after the last call: 1 healthy, 0 undrained. */
int gmme_idle(const gmme *g);
/* Fires issued by the last gmme_gemm_i8 call. */
int gmme_fires(const gmme *g);
/* ---------------------------------------------------------------------------
 * TPC. Eight programmable cores per card, verified on all 8.
 *
 * gmme_tpc_reduce_i32 runs k_reduce_i32_1d: out[j] = sum(in[j..j+K-1]).
 * The kernel binary comes from $GOYA_TPC_KERNEL, defaulting to
 * $GOYA_TPC_DIR/k_reduce_i32_1d.o.
 * -------------------------------------------------------------------------*/
int gmme_tpc_reduce_i32(gmme *g, const int32_t *in, int n_in, int K,
                        int32_t *out, int n_out, int tpc_idx);
/* EW: out[n] = a[n] + b[n]. Kernel from $GOYA_TPC_EW_KERNEL, default
   $GOYA_TPC_DIR/k_ew_add_i32.o */
int gmme_tpc_ew_add_i32(gmme *g, const int32_t *a, const int32_t *b, int n,
                        int32_t *out, int tpc_idx);
/* Upload a kernel into SRAM once; any core can then fire it with no DMA.
   Returns a handle, or the existing one if already resident. */
int gmme_tpc_register_kernel(gmme *g, const char *path);
int gmme_tpc_kernel_count(const gmme *g);
int gmme_tpc_cores(const gmme *g);

/* ---- Concurrent multi-core fire -------------------------------------
 *
 * One job per TPC core, fired so the EXECUTEs overlap instead of running
 * one after another. The batch costs TWO barriers total -- all descriptors,
 * then all EXECUTEs -- plus ONE device-idle settle, rather than a wait and
 * a settle per operation. The device-idle poll is device-WIDE, so paying it
 * per operation is what made concurrency impossible.
 *
 * Requires per-core SRAM partitions (GMME_TPC_PART=<hex offset>); with the
 * legacy shared map every core stages through the same buffers and this
 * would corrupt them, so it refuses to run.
 *
 * Cores must be distinct. Up to 8 jobs.
 */
typedef struct {
    const char *kernel;      /* path to the .o; must be non-NULL           */
    const int32_t *a;        /* first input                                */
    int n_a;
    const int32_t *b;        /* second input, or NULL                      */
    int n_b;
    int tid0, tid1;          /* index-space extents                        */
    int32_t *out;
    int n_out;
    int core;                /* 0..7, distinct across the batch            */
} gmme_tpc_job;

int gmme_tpc_fire_many(gmme *g, gmme_tpc_job *jobs, int njobs);

/* Fire a VENDOR kernel (input=tensor 0, output=tensor 1, f32, n/64 index
   space) as declared in its ELF .IndexMap section. Separate ABI, separate
   descriptor builder; tpc_desc and every proven result are untouched. */
int gmme_tpc_fire_vendor(gmme *g, const char *kpath,
                         const void *in, int n, void *out, int tpc_idx);
/* Two-input elementwise op with BOTH operands and the result on the card. */
/* ---- The wave: N resident TPC jobs, ANY kernels, one settle ----------
 *
 * Generalises gmme_tpc_op_resident_many (one kernel for every core) to one
 * kernel PER JOB with its own operands and index space, so a whole phase of
 * unrelated verbs -- an add on lane 0, a row-sum on lane 1, a norm's rsqrt on
 * lane 2 ... -- goes down as ONE staging submission, ONE descriptor CS, ONE
 * EXECUTE CS and ONE device-idle settle. The settle is device-wide, so paying
 * it per op is what serialised the lanes (goya-nine-engines-concurrent).
 *
 * All offsets are byte offsets into the C region (DRAM). Cores must be
 * distinct. Requires GMME_TPC_PART (per-core SRAM partitions). */
typedef struct {
    const char *kernel;          /* path to the .o                          */
    int nops;                    /* 2: tensors A,B(,A),OUT   3: A,B,C,OUT   */
    int has_b;                   /* 2-op: tensor1 -> lane scratch B (else = A);
                                    b_bytes > 0 additionally STAGES it        */
    uint64_t a_off, b_off, c_off, out_off;
    uint32_t a_bytes, b_bytes, c_bytes, out_bytes;
    uint32_t a_tail_zero;        /* zero bytes DMA'd right after A (rows kernels) */
    int n_a, n_b, n_out;         /* DIM_0_SIZE per tensor, elements (2-op)  */
    int tid0, tid1;              /* index space                             */
    int core;                    /* 0..7, distinct across the wave          */
} gmme_wave_job;

int gmme_tpc_wave(gmme *g, const gmme_wave_job *jobs, int njobs);

/* N independent resident ops, one per core, fired CONCURRENTLY. */
int gmme_tpc_op_resident_many(gmme *g, const char *kpath, const uint64_t *a_off,
                              const uint64_t *b_off, const uint64_t *out_off,
                              int n, int njobs);

int gmme_tpc_op_resident(gmme *g, const char *kpath, uint64_t a_off,
                         uint64_t b_off, uint64_t out_off, int n, int tpc_idx);

/* Requantise a resident int32 result to resident int8 with a per-element f32
   scale: out[i] = sat_i8(rint_half_even((float)a[i] * b[i])). All three buffers
   are DRAM offsets into the C region; nothing passes through the host. n must
   be a multiple of 256 -- one int8 vector -- and the index space is n/256, not
   the n/64 the all-int32 ops use. Kernel from $GOYA_TPC_REQUANT_KERNEL,
   default $GOYA_TPC_DIR/k_requant_i32_i8.o */
/* Elementwise op with tensor 1 BROADCAST: every lane reads b_off, because that
   tensor's DIM_0_STRIDE is 0. Turns a per-row scalar (a reduction result) into a
   full-width operand with no host round-trip. */
int gmme_tpc_op_resident_bcast(gmme *g, const char *kpath, uint64_t a_off,
                               uint64_t b_off, uint64_t out_off, int n, int tpc_idx);
/* Row max in two fires (strided pass + 64-wide fold). scratch_off needs
   rows*64 f32. Softmax's max-subtract depends on it. */
int gmme_tpc_max_f32_rows(gmme *g, uint64_t a_off, uint64_t scratch_off,
                          uint64_t out_off, int rows, int hid, int tpc_idx);
/* One dot product per row in ONE fire: out[i*64] = dot(a_row_i, b_row_i).
   The attention score primitive. */
int gmme_tpc_dotrows_f32(gmme *g, uint64_t a_off, uint64_t b_off,
                         uint64_t out_off, int rows, int hid, int tpc_idx);
/* Three-operand elementwise op (tensors 0,1,2 all distinct), so two chained
   binary ops become one fire. Legacy SRAM map only. */
int gmme_tpc_op3_resident(gmme *g, const char *kpath, uint64_t a_off,
                          uint64_t b_off, uint64_t c_off, uint64_t out_off,
                          int n, int tpc_idx);
/* Two-fire sum of squares: strided pass then a 64-wide fold. scratch_off needs
   rows*64 f32. Different summation order from the one-fire version. */
int gmme_tpc_sumsq_f32_rows_fast(gmme *g, uint64_t a_off, uint64_t scratch_off,
                                 uint64_t out_off, int rows, int hid, int tpc_idx);
/* Sum of SQUARES of every row in ONE fire: out[i*64] = sum(row_i ^ 2). */
int gmme_tpc_sumsq_f32_rows(gmme *g, uint64_t a_off, uint64_t out_off,
                            int rows, int hid, int tpc_idx);
/* Every row's sum in ONE fire: out_off[i*64] = sum of row i. A TPC fire costs
   ~0.67 ms fixed regardless of the work inside it, so fire COUNT is the cost. */
int gmme_tpc_reduce_f32_rows(gmme *g, uint64_t a_off, uint64_t out_off,
                             int rows, int hid, int tpc_idx);
/* Two-pass f32 row sum: K/64 strided loads then a 64-wide fold, so K/64 + 64
   loads instead of K. scratch_off needs 64 f32. The summation ORDER differs
   from gmme_tpc_reduce_f32_resident, so it is a different (equally correct)
   f32 answer -- a host reference must mirror this order, not the one-pass one. */
int gmme_tpc_reduce_f32_fast(gmme *g, uint64_t a_off, uint64_t scratch_off,
                             uint64_t out_off, int K, int tpc_idx);
int gmme_tpc_reduce_f32_pass1(gmme *g, uint64_t a_off, uint64_t out_off,
                              int K, int tpc_idx);
/* Fully resident f32 row reduction: out_off[0] = sum of a_off[0..K-1].
   Writes 64 f32 (the sliding window); element 0 is the answer. The sum is
   sequential in k, so a host reference must add in the SAME order to match --
   float addition is not associative. Kernel from $GOYA_TPC_REDUCE_F32_KERNEL,
   default $GOYA_TPC_DIR/k_reduce_f32_1d.o */
int gmme_tpc_reduce_f32_resident(gmme *g, uint64_t a_off, uint64_t out_off,
                                 int K, int tpc_idx);
/* The fixed within-256 output permutation of the requantise ABI:
   out[p] = requant(in[perm[p]]). Cancel it offline in the consuming GEMM's
   weight row order; it cannot be undone in the store on Goya gen-1. */
void gmme_requant_perm(int *perm256);
int gmme_tpc_requant_i32_i8(gmme *g, uint64_t a_off, uint64_t b_off,
                            uint64_t out_off, int n, int tpc_idx);
int gmme_tpc_peek(gmme *g, int slot, int n, void *out, int tpc_idx);
int gmme_tpc_fire_vendor_many(gmme *g, const char *kpath,
                              const float *in, float *out, int n_each, int njobs);
/* Stage the TPC input straight from the MME's DRAM result -- no host memory. */
int gmme_tpc_fire_vendor_resident(gmme *g, const char *kpath,
                                  uint64_t src_off, int n, float *out, int tpc_idx);

/* ---- All nine engines at once ---------------------------------------
 *
 * One GEMM on the MME plus up to 8 TPC jobs, all in flight together.
 * Pass A=NULL to skip the MME leg, or njobs=0 to skip the TPC leg.
 *
 * The GEMM must fit in a single command buffer (tiles <= gmme_max_tiles);
 * a larger one needs several sequential MME submits, which would have to
 * share the one MME cfg region and so cannot be queued ahead.
 *
 * Requires per-core SRAM partitions (GMME_TPC_PART) when njobs > 0.
 */
/* Device-resident GEMM: no host copies. Byte offsets into A/B/C. */
void gmme_chain_begin(gmme *g);
int  gmme_chain_end(gmme *g);

int gmme_dev_copy_c_to_a(gmme *g, uint64_t a_off, uint64_t c_off, uint32_t bytes);
/* Stage N independent regions in ONE submission (to_b[i]!=0 -> B, else A). */
int gmme_put_many(gmme *g, const uint64_t *dst_off, const void *const *src,
                  const uint32_t *bytes, const int *to_b, int n);

int gmme_put_a(gmme *g, uint64_t a_off, const void *src, uint32_t bytes);
int gmme_put_b(gmme *g, uint64_t b_off, const void *src, uint32_t bytes);
int gmme_put_c(gmme *g, uint64_t c_off, const void *src, uint32_t bytes);
int gmme_get_c(gmme *g, uint64_t c_off, void *dst, uint32_t bytes);

/* C += A@B when accumulate!=0 (K-tiling); C = A@B otherwise. */
int gmme_gemm_resident_acc(gmme *g, uint64_t a_off, uint64_t b_off, uint64_t c_off,
                           int M, int K, int N, int accumulate);

int gmme_gemm_resident(gmme *g, uint64_t a_off, uint64_t b_off, uint64_t c_off,
                       int M, int K, int N);

int gmme_fire_all(gmme *g, const int8_t *A, const int8_t *B, int32_t *C,
                  int M, int K, int N, gmme_tpc_job *jobs, int njobs);

/* Worst-case idle poll count in the last call (0 = engine was idle immediately). */
int gmme_idle_polls(const gmme *g);
/* With GMME_REREAD=1: did a second read of the same device memory differ? */
int gmme_reread_differs(const gmme *g);
#endif
