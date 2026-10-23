/*
 * k_dotrows_f32 -- one dot product per row, in ONE fire.
 *   c[i*64] = sum over k of a[i*hid + k] * b[i*hid + k]
 *
 * This is the attention score primitive. QK^T for a T-token head is T*T dot
 * products of length head_dim; laying them out as T*T rows and firing once
 * costs one fire instead of T*T, and a fire is ~0.67 ms fixed on this part.
 *
 * Row width comes from the index space (dim1). Lane 0 of each row's 64-wide
 * output block is the answer.
 */
void main(tensor a, tensor b, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    int hid = end[1] - idx[1];
    for (int i = idx[0]; i < end[0]; i++) {
        float64 acc = 0.0f;
        for (int k = idx[1]; k < end[1]; k++) {
            int5 co = {i * hid + k, 0, 0, 0, 0};
            acc = acc + v_f32_mul_b(v_f32_ld_tnsr_b(co, 0), v_f32_ld_tnsr_b(co, 1));
        }
        int5 oc = {i * 64, 0, 0, 0, 0};
        v_f32_st_tnsr(oc, 3, acc);
    }
}
