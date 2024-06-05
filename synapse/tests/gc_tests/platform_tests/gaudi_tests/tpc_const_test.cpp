#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "timer.h"
#include <tuple>
#include <deque>
#include "tensor_io_indicies.h"

class SynTrainingTestConstant : public SynTrainingTestInfra
{
};

TEST_F_GC(SynTrainingTestConstant, constant_bf16)
{
    const unsigned d1 = 116, d2 = 2;
    const unsigned size = d1 * d2;
    unsigned sizes[] = {d1, d2};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, 2, syn_type_bf16);
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, sizes, 2, syn_type_bf16);
    addNodeToGraph("constant_bf16", {in}, {out});
    compileAndRun();

    const uint16_t* inBuf = static_cast<const uint16_t*>(m_hostBuffers[in]);
    const uint16_t* outBuf = static_cast<const uint16_t*>(m_hostBuffers[out]);

    for (unsigned i = 0; i < size; ++i)
    {
        ASSERT_EQ(inBuf[i], outBuf[i]) << "at index " << i;
    }
}
