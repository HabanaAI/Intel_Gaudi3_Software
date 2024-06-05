#include "tpc_elf_header_common_test.h"
#include "graph_optimizer_test.h"
#include "kernel_db.h"
#include "gaudi_graph.h"
#include "gtest/gtest.h"

class TPCElfHeaderTestGaudi
: public TPCCustomElfHeaderNodeCommon<GaudiGraph>,
  public GraphOptimizerTest
, public ::testing::WithParamInterface<TestParams>
{
public:
    TPCElfHeaderTestGaudi(){}
    void runTest();
};

void TPCElfHeaderTestGaudi::runTest()
{
    KernelInstantiationWrapper instanceWrapper;

    TpcElfTools::TPCProgramHeader programHeader = {0};
    TestParams params{GetParam()};

    programHeader.version             = params.m_elfParams.version;
    programHeader.specialFunctionUsed = params.m_elfParams.specialFunctionUsed;
    programHeader.unsetSmallVLM       = params.m_elfParams.unsetSmallVlm;

    auto node = TPCElfHeaderTestGaudi::create(programHeader);
    ASSERT_TRUE(node != nullptr);

    node->instantiate(instanceWrapper);
    auto smallVlmequired = node->isSmallVLMRequired();
    ASSERT_EQ(smallVlmequired, params.m_smallVlmRequired);
}

TEST_P(TPCElfHeaderTestGaudi, test_gaudi_tpc_elf_features)
{
    runTest();
}

// Params: version, specialFunctionUsed, unsetSmallVlm, expected configuration
// Test cases:
// 1. Gaudi1, supported version, specialFunctionUsed, kernel requires largeVlm => true => Set smallVLM
// 2. Gaudi1, supported version, specialFunctionUsed, kernel doesn't require smallVlm => Set smallVLM

INSTANTIATE_TEST_SUITE_P(
    small_vlm,
    TPCElfHeaderTestGaudi,
    testing::Values(TestParams {TestParams::ElfParams {KernelDB::MIN_SUPPORTED_VERSION_SMALL_VLM, true, true}, true},
                    TestParams {TestParams::ElfParams {KernelDB::MIN_SUPPORTED_VERSION_SMALL_VLM, true, false}, true}));
