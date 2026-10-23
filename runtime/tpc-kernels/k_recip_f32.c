/*
 * k_recip_f32 -- c[n] = 1/a[n], Newton from an exponent-field seed.
 *
 * Goya gen-1 has no divide and no reciprocal instruction (CALC_FP_SPECIAL,
 * which owns SW_RECIP, is __gaudi_plus__ -- see k_rsqrt_f32.c). Four Newton
 * steps from the classic 0x7EF311C3 seed, all exact IEEE operations, so a host
 * running the same sequence matches bit for bit. Same code already proven
 * inside k_swiglu_f32; this exposes it standalone for softmax's 1/rowsum.
 */
void main(tensor a, tensor unused1, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    for (int i = idx[0]; i < end[0]; i++) {
        int5 co = {i * 64, 0, 0, 0, 0};
        float64 d = v_f32_ld_tnsr_b(co, 0);
        int64 db = *(int64 *)&d;
        db = 0x7EF311C3 - db;
        float64 y = *(float64 *)&db;
        y = v_f32_mul_b(y, 2.0f - v_f32_mul_b(d, y));
        y = v_f32_mul_b(y, 2.0f - v_f32_mul_b(d, y));
        y = v_f32_mul_b(y, 2.0f - v_f32_mul_b(d, y));
        y = v_f32_mul_b(y, 2.0f - v_f32_mul_b(d, y));
        v_f32_st_tnsr(co, 3, y);
    }
}
