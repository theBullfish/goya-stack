/*
 * k_ew_max_i32 — elementwise max.  c[n] = max(a[n], b[n])
 *
 * Not a demo verb: softmax needs a max before exponentiating, and ReLU is
 * max(x, 0). This is the first piece of the numerics an integer softmax needs.
 */
void main(tensor a, tensor b, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    for (int i = idx[0]; i < end[0]; i++) {
        int5 co = {i * 64, 0, 0, 0, 0};
        v_i32_st_tnsr(co, 3, v_i32_max_b(v_i32_ld_tnsr_b(co, 0),
                                         v_i32_ld_tnsr_b(co, 1)));
    }
}
