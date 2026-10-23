/*
 * k_rsqrt_f32 -- c[n] = 1/sqrt(a[n]), in pure f32 arithmetic.
 *
 * Goya gen-1 exposes NO special-function instruction through this toolchain:
 * CALC_FP_SPECIAL is the only consumer of SW_RSQRT/SW_EXP/SW_TANH and it is
 * __gaudi_plus__, and v_f32_get_lut_entry_and_interval_start is rejected for
 * -march=goya (its uint64_float64 pair type is not declared for this target).
 * The switch NAMES compile, which is a trap -- nothing on this part takes them.
 *
 * So: the classic exponent-halving seed plus Newton-Raphson, using only f32
 * multiply/subtract and an integer bit reinterpret (which does compile here).
 * Every step is an exact IEEE operation, so a host implementation running the
 * identical sequence agrees BIT FOR BIT -- which a hardware transcendental
 * never would. That is what makes "bit-exact vs CPU" reachable for item 7.
 */
void main(tensor a, tensor unused1, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    for (int i = idx[0]; i < end[0]; i++) {
        int5 co = {i * 64, 0, 0, 0, 0};
        float64 x = v_f32_ld_tnsr_b(co, 0);
        float64 xhalf = v_f32_mul_b(x, 0.5f);

        int64 bits = *(int64 *)&x;                 /* reinterpret, not convert */
        bits = 0x5f3759df - (bits >> 1);           /* seed */
        float64 y = *(float64 *)&bits;

        /* three Newton steps: y = y * (1.5 - xhalf*y*y) */
        y = v_f32_mul_b(y, 1.5f - v_f32_mul_b(xhalf, v_f32_mul_b(y, y)));
        y = v_f32_mul_b(y, 1.5f - v_f32_mul_b(xhalf, v_f32_mul_b(y, y)));
        y = v_f32_mul_b(y, 1.5f - v_f32_mul_b(xhalf, v_f32_mul_b(y, y)));

        v_f32_st_tnsr(co, 3, y);
    }
}
