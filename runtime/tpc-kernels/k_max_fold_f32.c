/*
 * k_max_fold_f32 -- fold 64 lanes down to lane 0.  c[i*64] = max(a[i*64 .. i*64+63])
 * Pass 2 of the row max. Sliding-window style like k_reduce_f32_1d, but with
 * max, so every lane ends up holding the same value over a 64-wide window and
 * lane 0 is unambiguously the row maximum.
 */
void main(tensor a, tensor unused1, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    int hid = end[1] - idx[1];
    for (int i = idx[0]; i < end[0]; i++) {
        int5 c0 = {i * hid, 0, 0, 0, 0};
        float64 acc = v_f32_ld_tnsr_b(c0, 0);
        for (int k = idx[1] + 1; k < end[1]; k++) {
            int5 co = {i * hid + k, 0, 0, 0, 0};
            acc = v_f32_max_b(acc, v_f32_ld_tnsr_b(co, 0));
        }
        int5 oc = {i * 64, 0, 0, 0, 0};
        v_f32_st_tnsr(oc, 3, acc);
    }
}
