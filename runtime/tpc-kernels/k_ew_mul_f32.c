/*
 * k_ew_mul_f32 -- c[n] = a[n] * b[n], f32.
 * The int32 sibling k_ew_mul_i32 cannot be reused: its multiply is integer.
 * A single IEEE multiply, so it is bit-exact against any host by construction.
 */
void main(tensor a, tensor b, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    for (int i = idx[0]; i < end[0]; i++) {
        int5 co = {i * 64, 0, 0, 0, 0};
        v_f32_st_tnsr(co, 3, v_f32_mul_b(v_f32_ld_tnsr_b(co, 0),
                                         v_f32_ld_tnsr_b(co, 1)));
    }
}
