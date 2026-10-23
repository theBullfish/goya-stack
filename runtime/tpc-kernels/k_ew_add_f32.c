/* k_ew_add_f32 -- c[n] = a[n] + b[n], f32. The int32 sibling cannot be reused:
   its add is integer. One IEEE add, so bit-exact by construction. */
void main(tensor a, tensor b, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    for (int i = idx[0]; i < end[0]; i++) {
        int5 co = {i * 64, 0, 0, 0, 0};
        v_f32_st_tnsr(co, 3, v_f32_ld_tnsr_b(co, 0) + v_f32_ld_tnsr_b(co, 1));
    }
}
