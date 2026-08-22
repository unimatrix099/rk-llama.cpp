#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
// Arm Mali. Valhall and later execute in 16-wide warps and expose no vendor
// attribute for a required subgroup size, so the REQD_* macros expand to
// nothing and the compiler picks the hardware width. The tuning constants
// guarded by INTEL_GPU are reused because every kernel's Intel path assumes
// a 16-wide subgroup, which matches Valhall exactly (verified: no kernel
// sets N_SIMDWIDTH 32 under INTEL_GPU). INTEL_GPU guards tuning constants
// only - no Intel-specific intrinsics - so this is a width selection, not a
// vendor claim.
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#define INTEL_GPU 1
// Valhall is 16 wide, so BOTH requests are pinned to 16. Leaving the width
// unpinned is not an option: the kernels assume the subgroup size the host
// passes for local-memory sizing, and an unpinned compiler choice silently
// corrupts the reductions (measured: PPL 22.7 -> 107.7). Serving the 32-wide
// request at 16 is safe because those kernels reduce generically, via
// sub_group_reduce_add and get_num_sub_groups(), and the host sizes their
// scratch from the matching subgroup_size = 16.
#define REQD_SUBGROUP_SIZE_16 __attribute__((reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((reqd_sub_group_size(16)))
#endif

//------------------------------------------------------------------------------
// norm
//------------------------------------------------------------------------------
kernel void kernel_norm(
        global void * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        float eps,
        local float * sum
) {
    src0 = (global void*)((global char*)src0 + offset0);
    dst = (global void*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float * x = (global float *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);

    // MEAN
    // parallel sum
    sum[get_local_id(0)] = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        sum[get_local_id(0)] += x[i00];
    }
    // reduce
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = get_local_size(0)/2; i > 0; i /= 2) {
        if (get_local_id(0) < i) {
            sum[get_local_id(0)] += sum[get_local_id(0) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean  = sum[0] / ne00;

    // recenter and VARIANCE
    barrier(CLK_LOCAL_MEM_FENCE);
    global float * y = dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    sum[get_local_id(0)] = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        y[i00] = x[i00] - mean;
        sum[get_local_id(0)] += y[i00] * y[i00];
    }

    // reduce
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = get_local_size(0)/2; i > 0; i /= 2) {
        if (get_local_id(0) < i) {
            sum[get_local_id(0)] += sum[get_local_id(0) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float variance = sum[0] / ne00;

    float scale = 1.0f/sqrt(variance + eps);
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        y[i00] = y[i00] * scale;
    }
}

//------------------------------------------------------------------------------
// norm_mul_add
//------------------------------------------------------------------------------
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_norm_mul_add(
        global char * src0_ptr, ulong src0_offset,
        global char * src1_ptr, ulong src1_offset,
        global char * src2_ptr, ulong src2_offset,
        global char * dst_ptr,  ulong dst_offset,
        int ne00, int ne01, int ne02, int ne03,
        ulong nb01, ulong nb02, ulong nb03,
        int ne10, int ne11, int ne12, int ne13,
        ulong nb11, ulong nb12, ulong nb13,
        int ne20, int ne21, int ne22, int ne23,
        ulong nb21, ulong nb22, ulong nb23,
        ulong nbd1, ulong nbd2, ulong nbd3,
        float eps,
        local float2 * sums
) {
    const int i03 = get_group_id(2);
    const int i02 = get_group_id(1);
    const int i01 = get_group_id(0);

    global float4 * x = (global float4 *)(src0_ptr + src0_offset + i01*nb01 + i02*nb02 + i03*nb03);
    global float4 * w = (global float4 *)(src1_ptr + src1_offset + (i01%ne11)*nb11 + (i02%ne12)*nb12 + (i03%ne13)*nb13);
    global float4 * b = (global float4 *)(src2_ptr + src2_offset + (i01%ne21)*nb21 + (i02%ne22)*nb22 + (i03%ne23)*nb23);
    global float4 * y = (global float4 *)(dst_ptr  + dst_offset  + i01*nbd1 + i02*nbd2 + i03*nbd3);

    float p_sum = 0.0f;
    float p_sum_sq = 0.0f;

    const int n_chunks = ne00 / 4;
    for (int i00 = get_local_id(0); i00 < n_chunks; i00 += get_local_size(0)) {
        float4 val = x[i00];
        p_sum += val.x + val.y + val.z + val.w;
        p_sum_sq += dot(val, val);
    }

    p_sum = sub_group_reduce_add(p_sum);
    p_sum_sq = sub_group_reduce_add(p_sum_sq);

    if (get_sub_group_local_id() == 0) {
        sums[get_sub_group_id()] = (float2)(p_sum, p_sum_sq);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) == 0) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (uint i = 0; i < get_num_sub_groups(); ++i) {
            float2 s = sums[i];
            sum += s.x;
            sum_sq += s.y;
        }

        const float inv_ne00 = 1.0f / (float)ne00;
        const float mean = sum * inv_ne00;
        const float variance = mad(-mean, mean, sum_sq * inv_ne00);

        sums[0] = (float2)(mean, rsqrt(variance + eps));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float2 mean_scale = sums[0];
    const float mean = mean_scale.x;
    const float scale = mean_scale.y;
    const float neg_mean_scale = -mean * scale;

    for (int i00 = get_local_id(0); i00 < n_chunks; i00 += get_local_size(0)) {
        const int w_idx = ne10 > 1 ? i00 : 0;
        const int b_idx = ne20 > 1 ? i00 : 0;
        const float4 norm_x = mad(x[i00], (float4)scale, (float4)neg_mean_scale);
        y[i00] = mad(norm_x, w[w_idx], b[b_idx]);
    }
}
