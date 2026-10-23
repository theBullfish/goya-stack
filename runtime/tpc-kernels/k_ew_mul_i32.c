/*
 * k_ew_mul_i32 — elementwise multiply.  c[n] = a[n] * b[n]
 *
 * Uses v_i32_mul_round_b, which returns a same-width int64 (64 int32 lanes).
 * The plain v_i32_mul_b returns int128 -- a high/low pair, because 32x32 is a
 * 64-bit product -- and is the one to reach for when the widening matters.
 */
void main(tensor a, tensor b, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    for (int i = idx[0]; i < end[0]; i++) {
        int5 co = {i * 64, 0, 0, 0, 0};
        v_i32_st_tnsr(co, 3, v_i32_mul_round_b(v_i32_ld_tnsr_b(co, 0),
                                               v_i32_ld_tnsr_b(co, 1)));
    }
}
