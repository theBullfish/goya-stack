/*
 * k_sumsq_f32_rows_strided -- pass 1 of a 2-fire row sum-of-squares.
 *   part[i*64 + j] = sum over t of a[i*hid + j + t*64]^2
 *
 * Strides a full vector, so a row costs hid/64 loads instead of hid. Pass 2 is
 * k_reduce_f32_rows over the rows x 64 partials.
 *
 * WHY THIS IS NOT THE SAME MISTAKE AS k_reduce_f32_strided. That earlier
 * two-pass version added a fire PER ROW (16 rows -> 32 fires) to save loads
 * that were nearly free at the time, and measured slower. This adds exactly ONE
 * fire in total, and it is being applied only after measurement showed work --
 * not fire count -- had become the dominant term: the one-fire sum-of-squares
 * was 32768 vector loads and ~3.8 ms of the 11.5 ms chain.
 *
 * The index space carries hid/64 in dim1; the kernel recovers the true row
 * width as (end[1]-idx[1])*64.
 *
 * ORDER: lane j accumulates t = 0,1,2,... then pass 2 folds lanes in order.
 * A host reference must group the same way -- float addition is not associative.
 */
void main(tensor a, tensor unused1, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    int nvec = end[1] - idx[1];          /* hid / 64 */
    int hid  = nvec * 64;

    for (int i = idx[0]; i < end[0]; i++) {
        float64 acc = 0.0f;
        for (int t = idx[1]; t < end[1]; t++) {
            int5 co = {i * hid + t * 64, 0, 0, 0, 0};
            float64 v = v_f32_ld_tnsr_b(co, 0);
            acc = acc + v_f32_mul_b(v, v);
        }
        int5 oc = {i * 64, 0, 0, 0, 0};
        v_f32_st_tnsr(oc, 3, acc);
    }
}
