#include "code_generator.h"
#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "passes/tensors_allocator.h"
#include "tensor.h"
#include "tensors_epoch_allocator.h"

#include "gtest/internal/gtest-param-util.h"

namespace
{
struct MultibufferTestPrametrized
: public GraphOptimizerTest
, public ::testing::WithParamInterface<std::tuple<uint32_t, bool>>
{
    static std::string genName(const ::testing::TestParamInfo<ParamType>& info)
    {
        const uint32_t bufferingLevel = std::get<0>(info.param);
        const bool     testSRAM       = std::get<1>(info.param);
        return fmt::format("{}_Level{}", (testSRAM ? "SRAM" : "DRAM"), bufferingLevel);
    }
};
}  // anonymous namespace

TEST_P(MultibufferTestPrametrized, testMultibufferAdressReuse)
{
    const uint32_t bufferingLevel = std::get<0>(GetParam());
    const bool     testSRAM       = std::get<1>(GetParam());

    setGlobalConfForTest(testSRAM ? GCFG_ENABLE_SRAM_MULTI_BUFFERING : GCFG_ENABLE_DRAM_MULTI_BUFFERING, "1");

    constexpr uint32_t channels          = 16;
    constexpr uint32_t width             = 16;
    constexpr uint32_t height            = 16;
    constexpr uint32_t batch             = 16;
    constexpr uint32_t tensorsInSameAddr = 3;

    constexpr std::array<TSize, 4> dims = {channels, width, height, batch};

    const uint32_t nodes = tensorsInSameAddr * bufferingLevel;

    GaudiGraph gaudiGraph;
    std::unique_ptr<CodeGenerator>& codeGenerator = gaudiGraph.getCodeGenerator();
    codeGenerator->getSramAllocator().Init(codeGenerator->getSramSize(), codeGenerator->getSramBaseAddr());
    codeGenerator->getWorkspaceAllocator().Init(codeGenerator->getDramSize(), codeGenerator->getDramBaseAddr());

    // Note that there's a certain assymetry with how the allocators work: DRAM
    // looks at output tensors from each node, while SRAM looks at inputs.
    // To avoid changing this or having a special case in the test,
    // we just add surrounding memcpy and only check tensors which are between
    // nodes...
    pTensor tensor_netowrk_in(new Tensor(dims.size(), dims.data(), syn_type_bf16));

    // We build a graph with tensorsInSameAddrXBufferinglevel memcopy nodes, from MB0 to MB1.
    TensorSet checkedTensors;
    for (uint32_t i = 0; i < nodes; i++)
    {
        pTensor tensorin(new Tensor(dims.size(), dims.data(), syn_type_bf16));
        pTensor tensorout(new Tensor(dims.size(), dims.data(), syn_type_bf16));
        pTensor tensor_network_out(new Tensor(dims.size(), dims.data(), syn_type_bf16));

        tensorout->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(0);
        tensorout->getTensorAnnotation().nonPersistentSectionInfo.bufferingLevel.set(bufferingLevel);
        checkedTensors.insert(tensorout);

        if (testSRAM)
        {
            tensorout->setTensorInSram();
        }
        else
        {
            tensorout->setTensorInWorkspace();
        }
        tensorin->setTensorInWorkspace();

        pNode memcopyNodeDummyIn  = NodeFactory::createNode({tensor_netowrk_in}, {tensorin}, nullptr, "memcpy", "");
        pNode memcopyNode         = NodeFactory::createNode({tensorin}, {tensorout}, nullptr, "memcpy", "");
        pNode memcopyNodeDummyOut = NodeFactory::createNode({tensorout}, {tensor_network_out}, nullptr, "memcpy", "");
        GraphEditor::addNode(gaudiGraph, memcopyNodeDummyIn);
        GraphEditor::addNode(gaudiGraph, memcopyNode);
        GraphEditor::addNode(gaudiGraph, memcopyNodeDummyOut);
    }
    gaudiGraph.setTensorsAlignment();
    setNonPersistentSectionInfo(gaudiGraph);
    allocateTensors(gaudiGraph);

    if (testSRAM)
    {
        // Verify that only the expected tensors are in SRAM
        for (const TensorPtr& t : gaudiGraph.getTensors())
        {
            if (t->inSram())
            {
                EXPECT_TRUE(checkedTensors.count(t))
                    << "tesnor \"" << t->getName() << "\" not part of checked tensors but is in SRAM";
            }
        }
    }
    else
    {
        // Verify that nothing is in SRAM
        for (const TensorPtr& t : gaudiGraph.getTensors())
        {
            EXPECT_FALSE(t->inSram());
        }
    }

    std::map<deviceAddrOffset, uint32_t> offset2Count;
    for (const TensorPtr& t : checkedTensors)
    {
        EXPECT_TRUE(testSRAM ? t->inSram() : t->inDram());
        ++offset2Count[testSRAM ? t->getSramOffset() : t->getDramOffset()];
    }
    for (const auto& v : offset2Count)
    {
        EXPECT_EQ(v.second, tensorsInSameAddr) << fmt::format("0x{:x} count was wrong", v.first);
    }
}

INSTANTIATE_TEST_SUITE_P(MultiBufferTest,
                         MultibufferTestPrametrized,
                         ::testing::Combine(::testing::Values(1, 2, 3, 4, 5), /* bufferingLevel */
                                            ::testing::Values(false, true)),  /* testSRAM */
                         MultibufferTestPrametrized::genName);