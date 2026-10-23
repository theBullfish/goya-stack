/*
 * k_swiglu_f32 -- SwiGLU:  c[n] = silu(a[n]) * b[n],  silu(x) = x / (1 + e^-x)
 *
 * a = the gate projection, b = the up projection. Both f32, both resident.
 *
 * exp is the software Cody-Waite path from k_exp_f32.c (Goya gen-1 exposes no
 * special-function instruction -- see k_rsqrt_f32.c for the evidence), and the
 * divide is a Newton reciprocal seeded from the exponent field, because there
 * is no f32 divide intrinsic on this part either. Both are exact IEEE
 * sequences, so a host running the same steps matches BIT FOR BIT -- measured
 * 4096/4096 for exp and for rsqrt separately before this was assembled.
 */
void main(tensor a, tensor b, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    for (int i = idx[0]; i < end[0]; i++) {
        int5 co = {i * 64, 0, 0, 0, 0};
        float64 g = v_f32_ld_tnsr_b(co, 0);
        float64 u = v_f32_ld_tnsr_b(co, 1);

        /* t = e^(-g) */
        float64 x  = v_f32_mul_b(g, -1.0f);
        float64 kf = v_f32_mul_b(x, 1.44269504088896340736f);
        int64   k  = v_convert_f32_to_i32_b(kf, SW_RHNE);
        float64 kr = v_convert_i32_to_f32_b(k);
        float64 r  = x - v_f32_mul_b(kr, 0.693145751953125f);
        r = r - v_f32_mul_b(kr, 1.428606765330187e-06f);
        float64 p = 0.0013981999f;
        p = 0.0083363998f + v_f32_mul_b(p, r);
        p = 0.0416573100f + v_f32_mul_b(p, r);
        p = 0.1666653900f + v_f32_mul_b(p, r);
        p = 0.5000000000f + v_f32_mul_b(p, r);
        p = 1.0000000000f + v_f32_mul_b(p, r);
        p = 1.0000000000f + v_f32_mul_b(r, p);
        int64 e2 = (k + 127) << 23;
        float64 sc = *(float64 *)&e2;
        float64 t  = v_f32_mul_b(p, sc);

        /* d = 1 + t ; y = 1/d by Newton, seed from the exponent field */
        float64 d = 1.0f + t;
        int64 db = *(int64 *)&d;
        db = 0x7EF311C3 - db;
        float64 y = *(float64 *)&db;
        y = v_f32_mul_b(y, 2.0f - v_f32_mul_b(d, y));
        y = v_f32_mul_b(y, 2.0f - v_f32_mul_b(d, y));
        y = v_f32_mul_b(y, 2.0f - v_f32_mul_b(d, y));
        y = v_f32_mul_b(y, 2.0f - v_f32_mul_b(d, y));

        v_f32_st_tnsr(co, 3, v_f32_mul_b(v_f32_mul_b(g, y), u));
    }
}
