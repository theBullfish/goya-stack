/*
 * k_max_rows_f32 -- every row's MAXIMUM in ONE fire.  c[i*64] = max(row i)
 *
 * Softmax needs this. exp(score) computed directly overflows f32 at score > 88,
 * and real attention scores go well past that; the standard fix is
 * exp(score - rowmax), which needs the row max first.
 *
 * Strided like k_sumsq_f32_rows_strided: dim1 carries hid/64, each lane keeps a
 * running max over its own stride-64 column, then lane 0 of the 64-wide block
 * is folded by a second pass. Max IS associative and commutative, so unlike the
 * f32 sums the fold order does not change the answer.
 */
void main(tensor a, tensor unused1, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    int nvec = end[1] - idx[1];
    int hid  = nvec * 64;

    for (int i = idx[0]; i < end[0]; i++) {
        int5 c0 = {i * hid, 0, 0, 0, 0};
        float64 acc = v_f32_ld_tnsr_b(c0, 0);
        for (int t = idx[1] + 1; t < end[1]; t++) {
            int5 co = {i * hid + t * 64, 0, 0, 0, 0};
            acc = v_f32_max_b(acc, v_f32_ld_tnsr_b(co, 0));
        }
        int5 oc = {i * 64, 0, 0, 0, 0};
        v_f32_st_tnsr(oc, 3, acc);
    }
}
