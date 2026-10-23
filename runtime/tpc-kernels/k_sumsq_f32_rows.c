/*
 * k_sumsq_f32_rows -- every row's SUM OF SQUARES in ONE fire.
 *   c[i*64] = sum over k of a[i*hid + k]^2
 *
 * Folds the separate square-then-reduce pair into a single fire. A TPC fire
 * costs ~0.67 ms fixed on this part regardless of the work in it, so removing a
 * fire is worth far more than removing arithmetic -- see k_reduce_f32_rows.
 *
 * Row width comes from the index space (dim1), so the kernel needs no
 * compile-time width. Lane 0 of each row's block is the answer; the other lanes
 * are shifted windows, exactly as in k_reduce_f32_1d.
 */
void main(tensor a, tensor unused1, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    int hid = end[1] - idx[1];

    for (int i = idx[0]; i < end[0]; i++) {
        float64 acc = 0.0f;
        for (int k = idx[1]; k < end[1]; k++) {
            int5 co = {i * hid + k, 0, 0, 0, 0};
            float64 v = v_f32_ld_tnsr_b(co, 0);
            acc = acc + v_f32_mul_b(v, v);
        }
        int5 oc = {i * 64, 0, 0, 0, 0};
        v_f32_st_tnsr(oc, 3, acc);
    }
}
