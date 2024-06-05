#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "gc_dynamic_shapes_infra.h"
#include "node_factory.h"
#include "timer.h"
#include <tuple>
#include <deque>
#include "tensor_io_indicies.h"

class SynGaudiDynamicBatchGemmBias : public SynGaudiTestInfra
{
public:
    SynGaudiDynamicBatchGemmBias() { setTestPackage(TEST_PACKAGE_DSD); }
};

// This test is disabled because memcpy cannot run with the new implementation (where the
// internal null input is created early). TODO We need to replace this with a TPC node that is
// designed to handle these tensors, when such node is ready.
//
TEST_F_GC(SynGaudiDynamicBatchGemmBias, basic, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned max_batch_dim  = 10;
    unsigned max_common_dim = 15;
    unsigned max_out_rows   = 8;
    unsigned max_out_cols   = 11;

    unsigned min_batch_dim  = 2;
    unsigned min_common_dim = 2;
    unsigned min_out_rows   = 2;
    unsigned min_out_cols   = 2;

    unsigned act_batch_dim  = 7;
    unsigned act_common_dim = 9;
    unsigned act_out_rows   = 5;
    unsigned act_out_cols   = 6;

    unsigned max1sizes[3]    = {max_common_dim, max_out_rows, max_batch_dim};
    unsigned max2sizes[3]    = {max_out_cols, max_common_dim, max_batch_dim};
    unsigned maxoutsizes[3]  = {max_out_cols, max_out_rows, max_batch_dim};
    unsigned maxbiassizes[2] = {max_out_cols, max_out_rows};

    unsigned min1sizes[3]    = {min_common_dim, min_out_rows, min_batch_dim};
    unsigned min2sizes[3]    = {min_out_cols, min_common_dim, min_batch_dim};
    unsigned minoutsizes[3]  = {min_out_cols, min_out_rows, min_batch_dim};
    unsigned minbiassizes[2] = {min_out_cols, min_out_rows};

    unsigned act1sizes[3]    = {act_common_dim, act_out_rows, act_batch_dim};
    unsigned act2sizes[3]    = {act_out_cols, act_common_dim, act_batch_dim};
    unsigned actoutsizes[3]  = {act_out_cols, act_out_rows, act_batch_dim};
    unsigned actbiassizes[2] = {act_out_cols, act_out_rows};

    auto t1 = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  max1sizes,
                                  3,
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
                                  3,
                                  syn_type_single,
                                  nullptr,
                                  nullptr,
                                  0,
                                  0,
                                  nullptr,
                                  min2sizes);

    auto bias = createPersistTensor(INPUT_TENSOR,
                                    MEM_INIT_RANDOM_POSITIVE,
                                    nullptr,
                                    maxbiassizes,
                                    2,
                                    syn_type_single,
                                    nullptr,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    minbiassizes);

    auto out = createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_NONE,
                                   nullptr,
                                   maxoutsizes,
                                   3,
                                   syn_type_single,
                                   nullptr,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   minoutsizes);

    addNodeToGraph("batch_gemm", {t1, t2, bias}, {out});

    compileTopology();

    setActualSizes(t1, act1sizes);
    setActualSizes(t2, act2sizes);
    setActualSizes(bias, actbiassizes);
    setActualSizes(out, actoutsizes);

    runTopology();

    const float* in1Buf  = reinterpret_cast<const float*>(m_hostBuffers[t1]);
    const float* in2Buf  = reinterpret_cast<const float*>(m_hostBuffers[t2]);
    const float* biasBuf = reinterpret_cast<const float*>(m_hostBuffers[bias]);
    const float* outBuf  = reinterpret_cast<const float*>(m_hostBuffers[out]);

    auto check_gemm = [](unsigned     rows,
                         unsigned     common,
                         unsigned     cols,
                         const float* a1,
                         const float* a2,
                         const float* bias,
                         const float* out) {
        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                float x = bias[i * cols + j];
                for (unsigned k = 0; k < common; ++k)
                {
                    x += a1[i * common + k] * a2[k * cols + j];
                }
                ASSERT_NEAR(x, out[i * cols + j], 1.0e-4);
            }
        }
    };

    for (int i = 0; i < act_batch_dim; ++i)
    {
        const float* in1 = in1Buf + i * act_out_rows * act_common_dim;
        const float* in2 = in2Buf + i * act_common_dim * act_out_cols;
        const float* out = outBuf + i * act_out_rows * act_out_cols;

        check_gemm(act_out_rows, act_common_dim, act_out_cols, in1, in2, biasBuf, out);
    }
}
