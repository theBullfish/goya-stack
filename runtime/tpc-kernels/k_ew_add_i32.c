/*
 * k_ew_add_i32 — elementwise add on the Goya TPC.  c[n] = a[n] + b[n]
 *
 * The Franca Tier-0 verb EW, the first one we authored rather than inherited.
 * Same shape as k_reduce_i32_1d: 1-D int32 tensors, no 2-D descriptor, so it
 * runs against the loader exactly as proven.
 *
 * The VPU is 256 bytes folded into four dual-groups = 64 lanes at INT32, so
 * each index-space step covers 64 elements.  TID_SIZE_DIM_0 = ceil(n / 64);
 * dim 1 is unused and must be 1.
 *
 * Tensor 0 = a, tensor 1 = b, tensor 3 = c.  Unlike the reduce kernel this
 * needs tensor 1 pointed at its OWN buffer, not aliased onto tensor 0.
 *
 * Codex Labs LLC
 */

void main(tensor a, tensor b, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();

    for (int i = idx[0]; i < end[0]; i++) {
        int5 co = {i * 64, 0, 0, 0, 0};
        int64 va = v_i32_ld_tnsr_b(co, 0);
        int64 vb = v_i32_ld_tnsr_b(co, 1);
        v_i32_st_tnsr(co, 3, v_i32_add_b(va, vb));
    }
}
