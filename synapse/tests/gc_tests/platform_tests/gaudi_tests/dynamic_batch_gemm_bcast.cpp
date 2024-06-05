#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "gc_dynamic_shapes_infra.h"
#include "node_factory.h"
#include "timer.h"
#include <tuple>
#include <deque>
#include "tensor_io_indicies.h"


class SynGaudiDynamicBatchGemmBcast : public SynGaudiTestInfra
{
public:
    SynGaudiDynamicBatchGemmBcast()
    {
        setTestPackage(TEST_PACKAGE_DSD);
        setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
    }
};

TEST_F_GC(SynGaudiDynamicBatchGemmBcast, basic)
{
    // TODO The test fails with dimensions that are commented out. The output tensor is all zeros.
    //
    unsigned max_batch_dim1 = 14;
    unsigned max_batch_dim2 = 30;
    unsigned max_batch_dim3 = 88;
    unsigned max_common_dim = 20;
    unsigned max_out_rows   = 20;
    unsigned max_out_cols   = 40;

    unsigned min_batch_dim1 = 1;
    unsigned min_batch_dim2 = 1;
    unsigned min_batch_dim3 = 1;
    unsigned min_common_dim = 1;
    unsigned min_out_rows   = 1;
    unsigned min_out_cols   = 1;

    unsigned act_batch_dim1 = 14;
    unsigned act_batch_dim2 = 22;
    unsigned act_batch_dim3 = 68;
    unsigned act_common_dim = 18;
    unsigned act_out_rows   = 18;
    unsigned act_out_cols   = 17;

    unsigned max1sizes[5]   = {max_common_dim, max_out_rows, max_batch_dim1, max_batch_dim2, max_batch_dim3};
    unsigned max2sizes[5]   = {max_out_cols, max_common_dim, max_batch_dim1, max_batch_dim2, 1};
    unsigned maxoutsizes[5] = {max_out_cols, max_out_rows, max_batch_dim1, max_batch_dim2, max_batch_dim3};

    unsigned min1sizes[5]   = {min_common_dim, min_out_rows, min_batch_dim1, min_batch_dim2, min_batch_dim3};
    unsigned min2sizes[5]   = {min_out_cols, min_common_dim, min_batch_dim1, min_batch_dim2, 1};
    unsigned minoutsizes[5] = {min_out_cols, min_out_rows, min_batch_dim1, min_batch_dim2, min_batch_dim3};

    unsigned act1sizes[5]   = {act_common_dim, act_out_rows, act_batch_dim1, act_batch_dim2, act_batch_dim3};
    unsigned act2sizes[5]   = {act_out_cols, act_common_dim, act_batch_dim1, act_batch_dim2, 1};
    unsigned actoutsizes[5] = {act_out_cols, act_out_rows, act_batch_dim1, act_batch_dim2, act_batch_dim3};

    synGEMMParams params {true, true};

    if (params.transpose_a)
    {
        std::swap(max1sizes[0], max1sizes[1]);
        std::swap(min1sizes[0], min1sizes[1]);
        std::swap(act1sizes[0], act1sizes[1]);
    }

    if (params.transpose_b)
    {
        std::swap(max2sizes[0], max2sizes[1]);
        std::swap(min2sizes[0], min2sizes[1]);
        std::swap(act2sizes[0], act2sizes[1]);
    }

    auto t1 = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  max1sizes,
                                  5,
                                  syn_type_single,
                                  nullptr,
                                  nullptr,
                                  0,
                                  0,
                                  nullptr,
                                  min1sizes);

    auto t2 = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  max2sizes,
                                  5,  // only 3 dims -- broadcast along the last one
                                  syn_type_single,
                                  nullptr,
                                  nullptr,
                                  0,
                                  0,
                                  nullptr,
                                  min2sizes);

    auto out = createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_NONE,
                                   nullptr,
                                   maxoutsizes,
                                   5,
                                   syn_type_single,
                                   nullptr,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   minoutsizes);

    addNodeToGraph("batch_gemm", {t1, t2}, {out}, &params, sizeof(params));

    compileTopology();

    setActualSizes(t1, act1sizes);
    setActualSizes(t2, act2sizes);
    setActualSizes(out, actoutsizes);

    runTopology();

    const float* in1Buf = reinterpret_cast<const float*>(m_hostBuffers[t1]);
    const float* in2Buf = reinterpret_cast<const float*>(m_hostBuffers[t2]);
    const float* outBuf = reinterpret_cast<const float*>(m_hostBuffers[out]);

    auto check_gemm = [](unsigned             rows,
                         unsigned             common,
                         unsigned             cols,
                         const float*         a1,
                         const float*         a2,
                         const float*         out,
                         const synGEMMParams& params,
                         unsigned             batch_i,
                         unsigned             batch_j,
                         unsigned             batch_k) {
        unsigned rowstride1 = common, colstride1 = 1;
        unsigned rowstride2 = cols, colstride2 = 1;

        if (params.transpose_a)
        {
            rowstride1 = 1;
            colstride1 = rows;
        }

        if (params.transpose_b)
        {
            rowstride2 = 1;
            colstride2 = common;
        }

        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                float x = 0;
                for (unsigned k = 0; k < common; ++k)
                {
                    x += a1[i * rowstride1 + k * colstride1] * a2[k * rowstride2 + j * colstride2];
                }
                ASSERT_NEAR(x, out[i * cols + j], 1.0e-4) << "i=" << i << ",j=" << j << ";batch_i=" << batch_i
                                                          << ",batch_j=" << batch_j << ",batch_k=" << batch_k;
            }
        }
    };

    for (int k = 0; k < act_batch_dim3; ++k)
    {
        for (int j = 0; j < act_batch_dim2; ++j)
        {
            for (int i = 0; i < act_batch_dim1; ++i)
            {
                const float* in1 = in1Buf + i * act_out_rows * act_common_dim +
                                   j * act_out_rows * act_common_dim * act_batch_dim1 +
                                   k * act_out_rows * act_common_dim * act_batch_dim1 * act_batch_dim2;

                const float* in2 =
                    in2Buf + i * act_common_dim * act_out_cols + j * act_common_dim * act_out_cols * act_batch_dim1;

                const float* out = outBuf + i * act_out_rows * act_out_cols +
                                   j * act_out_rows * act_out_cols * act_batch_dim1 +
                                   k * act_out_rows * act_out_cols * act_batch_dim1 * act_batch_dim2;

                ASSERT_NO_FATAL_FAILURE(
                    check_gemm(act_out_rows, act_common_dim, act_out_cols, in1, in2, out, params, i, j, k));
            }
        }
    }
}
