#include "gc_gaudi_test_infra.h"
#include "synapse_common_types.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "scoped_configuration_change.h"

TEST_F_GC(SynGaudiTestInfra, identity_cast)
{
    // TODO: Remove once [SW-136998] is done
    ScopedConfigurationChange disableGCOpValidation("ENABLE_GC_NODES_VALIDATION_BY_OPS_DB", "false");
    const unsigned dims          = 4;
    unsigned       inputSizes[]  = {2, 3, 5, 10};
    unsigned       outputSizes[] = {2, 3, 5, 10};

    // Create input tensors
    auto input1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inputSizes, dims, syn_type_uint8);
    auto input2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inputSizes, dims, syn_type_int16);
    auto input3 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inputSizes, dims, syn_type_uint32);

    // Create output tensors
    auto output1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, dims, syn_type_int8);
    auto output2 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, dims, syn_type_uint16);
    auto output3 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, dims, syn_type_int32);

    addNodeToGraph("identity", {input1}, {output1});
    addNodeToGraph("identity", {input2}, {output2});
    addNodeToGraph("identity", {input3}, {output3});

    compileAndRun();

    ASSERT_EQ(std::memcmp(m_hostBuffers[input1], m_hostBuffers[output1], sizeof(inputSizes)), 0);
    ASSERT_EQ(std::memcmp(m_hostBuffers[input1], m_hostBuffers[output1], sizeof(inputSizes) * 2), 0);
    ASSERT_EQ(std::memcmp(m_hostBuffers[input1], m_hostBuffers[output1], sizeof(inputSizes) * 4), 0);
}
