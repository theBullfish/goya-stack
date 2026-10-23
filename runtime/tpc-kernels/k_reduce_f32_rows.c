/*
 * k_reduce_f32_rows -- ALL rows in ONE fire. c[i*64] = sum of row i.
 *
 * Index space: dim0 = number of rows, dim1 = the row length. The row length is
 * read from the index space itself (end[1] - idx[1]), so the kernel needs no
 * compile-time width.
 *
 * WHY THIS SHAPE. Measured on hl0: a TPC fire costs ~0.67 ms FIXED, almost
 * independently of the work in it -- k_ew_mul_f32 over 64 elements takes
 * 0.675 ms and over 65536 takes 1.096 ms, a thousand times the work for 62%
 * more time. Fire COUNT is the cost, not loads.
 *
 * That killed the obvious optimisation: a two-pass reduce doing K/64 strided
 * loads instead of K was 21x less traffic and came out SLOWER (19.6 ms vs
 * 13.9 for 16 rows), because it needed two fires per row instead of one. The
 * lever is not work per fire. It is fires.
 */
void main(tensor a, tensor unused1, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    int hid = end[1] - idx[1];

    for (int i = idx[0]; i < end[0]; i++) {
        float64 acc = 0.0f;
        for (int k = idx[1]; k < end[1]; k++) {
            int5 co = {i * hid + k, 0, 0, 0, 0};
            acc = acc + v_f32_ld_tnsr_b(co, 0);
        }
        int5 oc = {i * 64, 0, 0, 0, 0};
        v_f32_st_tnsr(oc, 3, acc);
    }
}
