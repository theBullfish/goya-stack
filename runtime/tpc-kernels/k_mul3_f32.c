/*
 * k_mul3_f32 -- c[n] = a[n] * b[n] * t2[n], f32.
 *
 * RMSNorm's tail was two fires: (invrms * weight) then (* x). Tensor slot 2 is
 * otherwise aliased to tensor 0 and unused, so a three-input multiply collapses
 * those into one fire. At ~0.67 ms fixed per fire that is worth more than the
 * multiply it adds.
 *
 * Association is LEFT TO RIGHT -- (a*b)*t2 -- and float multiply is not
 * associative, so a host reference must group the same way.
 */
void main(tensor a, tensor b, tensor t2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    for (int i = idx[0]; i < end[0]; i++) {
        int5 co = {i * 64, 0, 0, 0, 0};
        float64 r = v_f32_mul_b(v_f32_ld_tnsr_b(co, 0), v_f32_ld_tnsr_b(co, 1));
        v_f32_st_tnsr(co, 3, v_f32_mul_b(r, v_f32_ld_tnsr_b(co, 2)));
    }
}
