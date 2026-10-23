/*
 * k_exp_f32 -- c[n] = e^(a[n]), in pure f32 arithmetic.
 *
 * Goya gen-1 exposes no special-function instruction (see k_rsqrt_f32.c), so
 * this is Cody-Waite range reduction plus a degree-5 minimax polynomial and a
 * 2^k built from the exponent field. Every step is an exact IEEE f32 operation
 * and an integer bit reinterpret, so a host reference running the identical
 * sequence agrees BIT FOR BIT -- proven for rsqrt at 4096/4096.
 *
 *   k = round(x * log2e)
 *   r = x - k*ln2_hi - k*ln2_lo      (two-part ln2: one f32 constant loses bits)
 *   e^r ~ 1 + r(1 + r(c2 + r(c3 + r(c4 + r*c5))))
 *   e^x = 2^k * e^r                  (2^k = (k+127) << 23, reinterpreted)
 */
void main(tensor a, tensor unused1, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    for (int i = idx[0]; i < end[0]; i++) {
        int5 co = {i * 64, 0, 0, 0, 0};
        float64 x = v_f32_ld_tnsr_b(co, 0);

        /* k = round(x * log2(e)), via the round-half-even f32->i32 convert */
        float64 kf = v_f32_mul_b(x, 1.44269504088896340736f);
        int64   k  = v_convert_f32_to_i32_b(kf, SW_RHNE);
        float64 kr = v_convert_i32_to_f32_b(k);

        /* r = x - k*ln2, split so the product is exact in f32 */
        float64 r = x - v_f32_mul_b(kr, 0.693145751953125f);
        r = r - v_f32_mul_b(kr, 1.428606765330187e-06f);

        /* degree-5 minimax on e^r, |r| <= ln2/2 */
        float64 p = 0.0013981999f;
        p = 0.0083363998f + v_f32_mul_b(p, r);
        p = 0.0416573100f + v_f32_mul_b(p, r);
        p = 0.1666653900f + v_f32_mul_b(p, r);
        p = 0.5000000000f + v_f32_mul_b(p, r);
        p = 1.0000000000f + v_f32_mul_b(p, r);
        p = 1.0000000000f + v_f32_mul_b(r, p);

        /* 2^k straight into the exponent field */
        int64 e2 = (k + 127) << 23;
        float64 s = *(float64 *)&e2;

        v_f32_st_tnsr(co, 3, v_f32_mul_b(p, s));
    }
}
