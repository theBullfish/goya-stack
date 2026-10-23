/*
 * k_requant_i32_i8 -- resident int32 -> int8 requantise with a per-element scale.
 *
 *     c[n] = sat_i8( rint_half_to_even( (float)a[n] * b[n] ) )
 *
 *   a : int32   -- the MME's raw accumulator output
 *   b : float32 -- one scale per element. A per-output-column scale is simply
 *                  that column's value repeated down the rows, which is the
 *                  shape Q8_0 forces on us (its own scales run along K and so
 *                  cannot be applied after the sum -- see docs/THE-PILLAR.md
 *                  and item 5).
 *   c : int8    -- what the next layer's GEMM consumes.
 *
 * WHY ONE INDEX-SPACE UNIT IS 256 ELEMENTS, NOT 64:
 * an i32/f32 vector is 64 lanes but an int8 vector is 256. Converting 64 f32
 * yields one QUARTER of a char256, so four groups are packed by lane and stored
 * once. The caller must set the index space to n/256. Every all-int32 op here
 * uses n/64; this one cannot, and passing it an n/64 index space would run the
 * kernel four times over the buffer.
 *
 * THE OUTPUT IS BYTE-LANE INTERLEAVED, BY DESIGN OF THE PART:
 * a char256 is not four concatenated quarters -- element p is BYTE p%4 OF
 * 32-BIT LANE p//4. So filling lanes 0..3 from four contiguous f32 loads gives
 *
 *     out[p] = requant(src[(p % 4) * 64 + (p / 4)])        within each 256 block
 *
 * measured directly with a ramp. On Goya gen-1 there is no way to undo this in
 * the store: SW_PACK / SW_PCK_32_TO_8 are __gaudi_plus__ only, st_tnsr_sqz is
 * __gaudi2__ only, st_tnsr_partial is __gaudi_plus__ only, and a strided
 * dim1 gather on the input has no effect -- DIM_1_SIZE/STRIDE were set and the
 * address generator ignored them (all four byte-lanes read the same word), with
 * or without VALID_DIM_MASK and with or without a matching index-space extent.
 *
 * So the permutation is part of this ABI rather than a bug to be worked around
 * at runtime. It is fixed, known, and confined WITHIN each 256-element block,
 * and the consumer of these int8s is the next GEMM -- whose weights are
 * prepared offline anyway (item 5: they must be requantised per column before
 * they can be used at all). Permuting those weight ROWS by the same sigma costs
 * nothing at load time and cancels it exactly. gmme_requant_perm() hands the
 * caller sigma so no one has to rediscover it.
 *
 * Rounding is SW_RHNE -- round half to nearest even -- which is what numpy's
 * rint does, so the host reference and the card agree exactly rather than
 * approximately.
 */
void main(tensor a, tensor b, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();

    for (int i = idx[0]; i < end[0]; i++) {
        int base = i * 256;
        char256 packed = 0;

        int5 o0 = { base +   0, 0, 0, 0, 0 };
        int5 o1 = { base +  64, 0, 0, 0, 0 };
        int5 o2 = { base + 128, 0, 0, 0, 0 };
        int5 o3 = { base + 192, 0, 0, 0, 0 };

        packed = v_convert_f32_to_i8_b(
                     v_f32_mul_b(v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(o0, 0)),
                                 v_f32_ld_tnsr_b(o0, 1)), 0, SW_RHNE, packed);
        packed = v_convert_f32_to_i8_b(
                     v_f32_mul_b(v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(o1, 0)),
                                 v_f32_ld_tnsr_b(o1, 1)), 1, SW_RHNE, packed);
        packed = v_convert_f32_to_i8_b(
                     v_f32_mul_b(v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(o2, 0)),
                                 v_f32_ld_tnsr_b(o2, 1)), 2, SW_RHNE, packed);
        packed = v_convert_f32_to_i8_b(
                     v_f32_mul_b(v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(o3, 0)),
                                 v_f32_ld_tnsr_b(o3, 1)), 3, SW_RHNE, packed);

        int5 co = { i * 256, 0, 0, 0, 0 };
        v_i8_st_tnsr(co, 3, packed);
    }
}
