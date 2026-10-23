/*
 * k_rope_f32 -- rotary position embedding, NeoX split-half convention.
 *
 *   out[i]      = x[i]*cos[i] - x[i+H]*sin[i]
 *   out[i+H]    = x[i]*sin[i] + x[i+H]*cos[i]        H = head_dim/2
 *
 * tensor 0 = x, tensor 1 = cos, tensor 2 = sin, tensor 3 = out. Fired through
 * gmme_tpc_op3_resident because it genuinely takes three operands.
 *
 * NO sin/cos KERNEL IS NEEDED. The tables depend only on the position and the
 * model's rope base (1e7 for Qwen3-30B-A3B), never on the activations, so they
 * are constants uploaded once -- which is what llama.cpp and HF both do. Writing
 * software sin/cos on the card would have been work for its own sake.
 *
 * WHY SPLIT-HALF IS THE LUCKY CONVENTION HERE. Qwen pairs dimension i with
 * i + head_dim/2, not with i+1. At head_dim 128 that means the two halves are
 * 64 elements apart -- exactly one 64-lane vector -- so a pair is two aligned
 * vector loads. The interleaved convention would have needed adjacent LANES,
 * which is the gather this part does not have (see item 7's correction).
 *
 * One index-space step covers ONE head: it loads both halves and stores both.
 */
void main(tensor x, tensor cs, tensor sn, tensor out)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();

    for (int i = idx[0]; i < end[0]; i++) {
        int5 lo = {i * 128,      0, 0, 0, 0};   /* x: head_dim = 128 per head  */
        int5 hi = {i * 128 + 64, 0, 0, 0, 0};
        /* The TABLES are head_dim/2 = 64 wide per head, NOT 128. Indexing them
           with the data's stride made head 0 correct and every head after it
           wrong -- which is exactly how it presented: 382/512 wrong, first at
           element 128. */
        int5 tb = {i * 64, 0, 0, 0, 0};

        float64 xl = v_f32_ld_tnsr_b(lo, 0);
        float64 xh = v_f32_ld_tnsr_b(hi, 0);
        float64 c  = v_f32_ld_tnsr_b(tb, 1);
        float64 s  = v_f32_ld_tnsr_b(tb, 2);

        v_f32_st_tnsr(lo, 3, v_f32_mul_b(xl, c) - v_f32_mul_b(xh, s));
        v_f32_st_tnsr(hi, 3, v_f32_mul_b(xl, s) + v_f32_mul_b(xh, c));
    }
}
