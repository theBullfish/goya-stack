/*
 * k_reduce_f32_strided -- pass 1 of a two-pass f32 row sum.
 *
 *   acc[j] = sum over k of a[j + k*64]        (64 partial sums)
 *
 * WHY. k_reduce_f32_1d walks k one element at a time, so summing a HID-long row
 * costs HID vector loads to move HID elements -- 64x more traffic than the data
 * requires. Measured: 16 rows of 2048 took 14.9 ms, which was 77% of the whole
 * RMSNorm, while the host round-trip I had been calling "the hole" was 1.1%.
 *
 * This pass strides by a full vector, so it costs K/64 loads. Pass 2 is the
 * existing k_reduce_f32_1d over just the 64 partials. Total K/64 + 64 loads
 * instead of K: at K=2048 that is 96 against 2048.
 *
 * ORDER MATTERS. Lane j sums a[j], a[j+64], a[j+128], ... in k order, then the
 * 64 partials are summed in lane order. That is a DIFFERENT order from a plain
 * sequential sum, so it is a different f32 answer -- correct, but not the same
 * bits. The host reference must mirror this exact order; float addition is not
 * associative and pretending the orders agree is how bit-exact quietly becomes
 * approximately-equal.
 */
void main(tensor a, tensor unused1, tensor unused2, tensor c)
{
    int5 idx = get_index_space_offset();
    int5 end = idx + get_index_space_size();
    for (int i = idx[0]; i < end[0]; i++) {
        int n0 = i * 64;
        float64 acc = 0.0f;
        for (int k = idx[1]; k < end[1]; k++) {
            int5 co = {n0 + k * 64, 0, 0, 0, 0};
            acc = acc + v_f32_ld_tnsr_b(co, 0);
        }
        int5 oc = {n0, 0, 0, 0, 0};
        v_f32_st_tnsr(oc, 3, acc);
    }
}
