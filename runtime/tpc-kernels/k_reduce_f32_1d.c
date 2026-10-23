/*
 * k_reduce_f32_1d -- f32 sliding-window sum, the row reduction RMSNorm needs.
 *
 * Structurally identical to the proven k_reduce_i32_1d: same index-space walk,
 * same 1-D descriptor, same sliding-window semantics (lane j = sum of
 * a[j .. j+K-1], and lane 0 is the answer). Only the accumulator type and the
 * add change, because the int32 add is the wrong operation for f32 data --
 * which is the whole reason the existing kernel could not simply be reused.
 *
 * The f32 sum is sequential in k, so a host reference must add in the SAME
 * order to match bit for bit. Float addition is not associative; a different
 * summation order is a different answer, and pretending otherwise is how a
 * "bit-exact" claim quietly becomes a tolerance claim.
 *
 * --- original notes from the int32 kernel, still accurate ---
 * k_reduce_f32_1d — minimum-risk first-light variant of the K-reduction.
 *
 *   lane 0 of the output  =  sum over k = 0 .. K-1 of a[k]
 *
 * WHY A SECOND KERNEL
 * -------------------
 * k_reduce_i32.c needs a 2-D tensor descriptor: three register writes
 * (DIM_1_SIZE / DIM_1_STRIDE / DIM_1_BASE_OFFSET at TENSOR_n + 0x1C/0x20/0x24)
 * that the proven loader currently zeroes.  Small change, but it is a change to
 * code that already works on silicon.
 *
 * This variant needs ZERO new register writes.  It runs against the loader
 * exactly as it is proven today.  Two existing registers change VALUE only:
 *
 *      QM_TID_SIZE_DIM_0  (0xE06C6C)  =  1        <- already 1
 *      QM_TID_SIZE_DIM_1  (0xE06C74)  =  K        <- currently 1
 *
 * Fire this one FIRST.  If it produces the right number, the whole chain
 * (kernel load, ICACHE prefetch, index space, AGU, store path) is confirmed and
 * the only remaining unknown for the general kernel is the dim-1 descriptor.
 *
 * HOW IT WORKS
 * ------------
 * Tensor 0 is 1-D: dim 0 size >= K+63, stride 4 bytes.  That is precisely the
 * descriptor the loader already programs.  The K partial products sit at
 * consecutive int32 addresses — precisely what the MME already writes
 * (SRAM_C = 0 2 3 4 means bytes 0, 4, 8, 12).
 *
 * A vector load at coordinate {k} pulls elements k .. k+63 into lanes 0 .. 63.
 * (An edit here once said 32 lanes.  WRONG -- the VPU is 256 bytes folded into
 * four dual-groups, which is 64 lanes at INT32.  The kernel binary is
 * unaffected either way: it is the same 1408-byte .text that passed first
 * light.  See tools/tpc/README.md section 0, correction 2.)
 * Summing that over k = 0 .. K-1 gives
 *
 *      lane j  =  sum over k of a[k + j]
 *
 * which is a sliding-window sum of width K.  Window j=0 is exactly the answer
 * we want.  Lanes 1 .. 63 hold the other windows — not garbage, but not the
 * answer either; the host reads element 0 only.  (The reference oracle predicts
 * every one of them, so a full-vector match is a much stronger check than lane 0
 * alone — it confirms the address generator, not just the adder.)
 *
 * REQUIREMENTS ON THE HOST
 * ------------------------
 *   - (K + 63) * 4 bytes readable behind tensor 0.  For K=4: 268 bytes.
 *   - 256 bytes writable behind tensor 3.  Only the first 4 matter.
 *   - Tensors 1 and 2 unused; point them anywhere readable.
 *
 * The outer index-space loop is kept identical to k_reduce_i32.c so the two
 * kernels stay structurally the same.  With TID_SIZE_DIM_0 = 1 it executes once
 * with i = 0, so n0 = 0 and the store lands at element 0.
 *
 * Codex Labs LLC
 */

void main(tensor a, tensor unused1, tensor unused2, tensor c)
{
    /* Same index-space loop as k_reduce_i32.c — see that file for why this
     * must never be dropped. */
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();

    for (int i = idx[0]; i < end[0]; i++) {

        int n0 = i * 64;
        float64 acc = 0.0f;

        /* The ONLY difference from k_reduce_i32.c: k walks dim 0 alongside n
         * instead of walking a separate dim 1.  This is what removes the need
         * for a 2-D descriptor, and it is also what makes the result a sliding
         * window rather than 64 independent reductions. */
        for (int k = idx[1]; k < end[1]; k++) {
            int5 co = {n0 + k, 0, 0, 0, 0};
            acc = acc + v_f32_ld_tnsr_b(co, 0);
        }

        int5 oc = {n0, 0, 0, 0, 0};
        v_f32_st_tnsr(oc, 3, acc);
    }
}
