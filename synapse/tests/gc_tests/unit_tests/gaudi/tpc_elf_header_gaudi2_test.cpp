#include "tpc_elf_header_common_test.h"
#include "graph_optimizer_test.h"
#include "kernel_db.h"
#include "gaudi2_graph.h"
#include "gtest/gtest.h"

class TPCElfHeaderTestGaudi2
: public TPCCustomElfHeaderNodeCommon<Gaudi2Graph>
, public GraphOptimizerTest
, public ::testing::WithParamInterface<TestParams>
{
public:
    TPCElfHeaderTestGaudi2() {}
    void runTest();
};

void TPCElfHeaderTestGaudi2::runTest()
{
    KernelInstantiationWrapper instanceWrapper;

    TpcElfTools::TPCProgramHeader programHeader = {0};
    TestParams params{GetParam()};

    programHeader.version             = params.m_elfParams.version;
    programHeader.specialFunctionUsed = params.m_elfParams.specialFunctionUsed;
    programHeader.unsetSmallVLM       = params.m_elfParams.unsetSmallVlm;

    auto node = TPCElfHeaderTestGaudi2::create(programHeader);
    ASSERT_TRUE(node != nullptr);

    node->instantiate(instanceWrapper);
    auto smallVlmequired = node->isSmallVLMRequired();

    ASSERT_EQ(smallVlmequired, params.m_smallVlmRequired);
}

TEST_P(TPCElfHeaderTestGaudi2, test_gaudi2_tpc_elf_features)
{
    runTest();
}

// Params: version, specialFunctionUsed, unsetSmallVlm, expected configuration
// Test cases:
// 1. Gaudi2, supported version, specialFunctionUsed, kernel requires largeVlm => set smallVLM
// 2. Gaudi2, supported version, specialFunctionUsed, kernel doesn't require smallVlm => Don't set smallVLM
// 3. Gaudi2, supported version, no specialFunction , kernel doesn't require smallVlm => Don't set smallVLM
// 4. gaudi2, old version, specialFunctionUsed, kernel doesn't require smallVlm => Set SmallVLM
// 5. gaudi2, old version, specialFunctionUsed, kernel requires smallVlm => Set SmallVLM

INSTANTIATE_TEST_SUITE_P(
    small_vlm,
    TPCElfHeaderTestGaudi2,
    testing::Values(
        TestParams {TestParams::ElfParams {KernelDB::MIN_SUPPORTED_VERSION_SMALL_VLM, true, false}, true},
        TestParams {TestParams::ElfParams {KernelDB::MIN_SUPPORTED_VERSION_SMALL_VLM, true, true}, false},
        TestParams {TestParams::ElfParams {KernelDB::MIN_SUPPORTED_VERSION_SMALL_VLM, false, false}, false},
        TestParams {TestParams::ElfParams {KernelDB::MIN_SUPPORTED_VERSION_SMALL_VLM - 1, true, true}, true},
        TestParams {TestParams::ElfParams {KernelDB::MIN_SUPPORTED_VERSION_SMALL_VLM - 1, true, false}, true}));
