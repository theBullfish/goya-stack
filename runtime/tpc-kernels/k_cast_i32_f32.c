/*
 * k_cast_i32_f32 -- c[n] = (float)a[n].
 *
 * The MME emits int32; everything after it in a layer (norms, RoPE, softmax) is
 * f32. This is the dequantise half of the boundary -- multiply by the scale
 * afterwards with k_ew_mul_f32, or fold both with k_mul3_f32.
 *
 * Both tensors are ElementSizeInBytes e_32 on this part, so the proven all-int32
 * descriptor already describes them and tpc_desc is reused untouched.
 */
void main(tensor a, tensor unused1, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    for (int i = idx[0]; i < end[0]; i++) {
        int5 co = {i * 64, 0, 0, 0, 0};
        v_f32_st_tnsr(co, 3, v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(co, 0)));
    }
}
