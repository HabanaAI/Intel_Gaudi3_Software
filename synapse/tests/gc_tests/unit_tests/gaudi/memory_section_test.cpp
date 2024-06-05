#include "code_generator.h"
#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "passes/tensors_allocator.h"
#include "tensor.h"
#include "tensors_epoch_allocator.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"

class MemorySectionTest : public GraphOptimizerTest
{
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    }
};

static unsigned int calcMemSectionSize(const TensorVector tensorsInSection)
{
    uint64_t allocationSize = 0;
    for (const auto& t : tensorsInSection)
    {
        uint64_t offset                = t->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.value();
        uint64_t currentAllocationSize = getWriteSpaceForTensor(t) + offset;
        allocationSize                 = std::max(allocationSize, currentAllocationSize);
    }
    return allocationSize;
}

TEST_F(MemorySectionTest, single_rmw_section)
{
    GaudiGraph gaudiGraph;
    std::unique_ptr<CodeGenerator>& codeGen = gaudiGraph.getCodeGenerator();
    codeGen->getSramAllocator().Init(codeGen->getSramSize(),  codeGen->getSramBaseAddr());

    std::vector<TSize> sizes = {5, 3, 1, 10};

    pTensor section1A(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    pTensor section1B(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    pTensor section1C(new Tensor(sizes.size(), sizes.data(), syn_type_float));

    std::vector<uint64_t> section1Offsets = {0, 5002, 13030};
    TensorVector          section1Tensors = {section1A, section1B, section1C};

    for (auto i = 0; i < section1Tensors.size(); ++i)
    {
        section1Tensors[i]->setTensorInSram();
        section1Tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(23);
        section1Tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(section1Offsets[i]);
        section1Tensors[i]->setTensorAlignment(gaudiGraph.getHALReader()->getCacheLineSizeInBytes());
    }

    unsigned int section1AllcationSize = calcMemSectionSize(section1Tensors);

    pNode memsetA     = NodeFactory::createNode({}, {section1A}, nullptr, "memset", "memsetA");
    pNode memcopyAtoB = NodeFactory::createNode({section1A}, {section1B}, nullptr, "memcpy", "memcopyAtoB");
    pNode memcopyBtoC = NodeFactory::createNode({section1B}, {section1C}, nullptr, "memcpy", "memcopyBtoC");

    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memsetA));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopyAtoB));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopyBtoC));

    setNonPersistentSectionInfo(gaudiGraph);
    allocateTensors(gaudiGraph);

    // Check that all the tensors are in SRAM
    for (const TensorPtr& t : gaudiGraph.getTensors())
    {
        ASSERT_TRUE(t->inSram());
    }

    // Check allocation sizes and offsets
    for (const auto& t : section1Tensors)
    {
        ASSERT_EQ(t->getTensorAnnotation().sizeToAllocate.value(), section1AllcationSize);
    }
    ASSERT_EQ(section1Tensors[1]->getSramOffset() - section1Tensors[0]->getSramOffset(), section1Offsets[1]);
    ASSERT_EQ(section1Tensors[2]->getSramOffset() - section1Tensors[0]->getSramOffset(), section1Offsets[2]);
}

TEST_F(MemorySectionTest, multiple_rmw_sections)
{
    GaudiGraph gaudiGraph;
    std::unique_ptr<CodeGenerator>& codeGen = gaudiGraph.getCodeGenerator();
    codeGen->getSramAllocator().Init(codeGen->getSramSize(),  codeGen->getSramBaseAddr());

    std::vector<TSize> aSizes = {5, 3, 1, 10};
    std::vector<TSize> bSizes = {2, 5, 7, 1};
    std::vector<TSize> cSizes = {13, 4, 7, 9};

    // Memory section 1:
    pTensor section1A(new Tensor(aSizes.size(), aSizes.data(), syn_type_float));
    pTensor section1B(new Tensor(bSizes.size(), bSizes.data(), syn_type_float));
    pTensor section1C(new Tensor(cSizes.size(), cSizes.data(), syn_type_float));

    std::vector<uint64_t> section1Offsets = {0, 234, 4000};
    TensorVector          section1Tensors = {section1A, section1B, section1C};

    for (auto i = 0; i < section1Tensors.size(); ++i)
    {
        section1Tensors[i]->setTensorInSram();
        section1Tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(1);
        section1Tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(section1Offsets[i]);
        section1Tensors[i]->setTensorAlignment(gaudiGraph.getHALReader()->getCacheLineSizeInBytes());
    }

    unsigned int section1AllcationSize = calcMemSectionSize(section1Tensors);

    // Memory section 2:
    pTensor section2A(new Tensor(aSizes.size(), aSizes.data(), syn_type_float));
    pTensor section2B(new Tensor(bSizes.size(), bSizes.data(), syn_type_float));
    pTensor section2C(new Tensor(cSizes.size(), cSizes.data(), syn_type_float));

    std::vector<uint64_t> section2Offsets = {0, 5400, 67};
    TensorVector          section2Tensors = {section2A, section2B, section2C};

    for (auto i = 0; i < section2Tensors.size(); ++i)
    {
        section2Tensors[i]->setTensorInSram();
        section2Tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(2);
        section2Tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(section2Offsets[i]);
        section2Tensors[i]->setTensorAlignment(gaudiGraph.getHALReader()->getCacheLineSizeInBytes());
    }

    unsigned int section2AllcationSize = calcMemSectionSize(section2Tensors);

    // Memory section 3:
    pTensor section3A(new Tensor(aSizes.size(), aSizes.data(), syn_type_float));
    pTensor section3B(new Tensor(bSizes.size(), bSizes.data(), syn_type_float));
    pTensor section3C(new Tensor(cSizes.size(), cSizes.data(), syn_type_float));

    std::vector<uint64_t> section3Offsets = {0, 800, 33333};
    TensorVector          section3Tensors = {section3A, section3B, section3C};

    for (auto i = 0; i < section3Tensors.size(); ++i)
    {
        section3Tensors[i]->setTensorInSram();
        section3Tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(3);
        section3Tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(section3Offsets[i]);
        section3Tensors[i]->setTensorAlignment(gaudiGraph.getHALReader()->getCacheLineSizeInBytes());
    }

    unsigned int section3AllcationSize = calcMemSectionSize(section3Tensors);

    pNode memsetSec1A = NodeFactory::createNode({}, {section1A}, nullptr, "memset", "memsetSec1A");
    pNode memcopySec1AtoB =
        NodeFactory::createNode({section1A}, {section1B}, nullptr, "debug", "memcopySec1AtoB");
    pNode memcopySec1BtoC =
        NodeFactory::createNode({section1B}, {section1C}, nullptr, "debug", "memcopySec1BtoC");

    pNode memsetSec2A = NodeFactory::createNode({}, {section2A}, nullptr, "memset", "memsetSec2A");
    pNode memcopySec2AtoB =
        NodeFactory::createNode({section2A}, {section2B}, nullptr, "debug", "memcopySec2AtoB");
    pNode memcopySec2BtoC =
        NodeFactory::createNode({section2B}, {section2C}, nullptr, "debug", "memcopySec2BtoC");

    pNode memsetSec3A = NodeFactory::createNode({}, {section3A}, nullptr, "memset", "memsetSec3A");
    pNode memcopySec3AtoB =
        NodeFactory::createNode({section3A}, {section3B}, nullptr, "debug", "memcopySec3AtoB");
    pNode memcopySec3BtoC =
        NodeFactory::createNode({section3B}, {section3C}, nullptr, "debug", "memcopySec3BtoC");

    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memsetSec1A));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopySec1AtoB));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopySec1BtoC));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memsetSec2A));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopySec2AtoB));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopySec2BtoC));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memsetSec3A));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopySec3AtoB));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopySec3BtoC));

    setNonPersistentSectionInfo(gaudiGraph);
    allocateTensors(gaudiGraph);

    // Check that all the tensors are in Sram
    for (const TensorPtr& t : gaudiGraph.getTensors())
    {
        ASSERT_TRUE(t->inSram());
    }

    // Section 1 - check allocation sizes and offsets
    for (const auto& t : section1Tensors)
    {
        ASSERT_EQ(t->getTensorAnnotation().sizeToAllocate.value(), section1AllcationSize);
    }
    ASSERT_EQ(section1Tensors[1]->getSramOffset() - section1Tensors[0]->getSramOffset(), section1Offsets[1]);
    ASSERT_EQ(section1Tensors[2]->getSramOffset() - section1Tensors[0]->getSramOffset(), section1Offsets[2]);

    // Section 2 - check allocation sizes and offsets
    for (const auto& t : section2Tensors)
    {
        ASSERT_EQ(t->getTensorAnnotation().sizeToAllocate.value(), section2AllcationSize);
    }
    ASSERT_EQ(section2Tensors[1]->getSramOffset() - section2Tensors[0]->getSramOffset(), section2Offsets[1]);
    ASSERT_EQ(section2Tensors[2]->getSramOffset() - section2Tensors[0]->getSramOffset(), section2Offsets[2]);

    // Section 3 - check allocation sizes and offsets
    for (const auto& t : section3Tensors)
    {
        ASSERT_EQ(t->getTensorAnnotation().sizeToAllocate.value(), section3AllcationSize);
    }
    ASSERT_EQ(section3Tensors[1]->getSramOffset() - section3Tensors[0]->getSramOffset(), section3Offsets[1]);
    ASSERT_EQ(section3Tensors[2]->getSramOffset() - section3Tensors[0]->getSramOffset(), section3Offsets[2]);
}

TEST_F(MemorySectionTest, rmw_section_and_multi_buffer)
{
    GaudiGraph gaudiGraph;
    std::unique_ptr<CodeGenerator>& codeGen = gaudiGraph.getCodeGenerator();
    codeGen->getSramAllocator().Init(codeGen->getSramSize(),  codeGen->getSramBaseAddr());

    std::vector<TSize> sizes = {15, 3, 1, 2};

    // RMW section:
    pTensor rmwSectionA(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    pTensor rmwSectionB(new Tensor(sizes.size(), sizes.data(), syn_type_float));

    std::vector<uint64_t> rmwSectionOffsets = {0, 4567};
    TensorVector          rmwSectionTensors = {rmwSectionA, rmwSectionB};

    for (auto i = 0; i < rmwSectionTensors.size(); ++i)
    {
        rmwSectionTensors[i]->setTensorInSram();
        rmwSectionTensors[i]->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(34);
        rmwSectionTensors[i]->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(rmwSectionOffsets[i]);
        rmwSectionTensors[i]->setTensorAlignment(gaudiGraph.getHALReader()->getCacheLineSizeInBytes());
    }

    unsigned int rmwSectionAllcationSize = calcMemSectionSize(rmwSectionTensors);

    // Multi buffer:
    pTensor multiBufferA(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    pTensor multiBufferB(new Tensor(sizes.size(), sizes.data(), syn_type_float));

    TensorVector multiBufferTensors = {multiBufferA, multiBufferB};

    for (auto i = 0; i < multiBufferTensors.size(); ++i)
    {
        multiBufferTensors[i]->setTensorInSram();
        multiBufferTensors[i]->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(3);
        multiBufferTensors[i]->getTensorAnnotation().nonPersistentSectionInfo.bufferingLevel.set(2);
        multiBufferTensors[i]->setTensorAlignment(gaudiGraph.getHALReader()->getCacheLineSizeInBytes());
    }

    pNode memsetRmwA = NodeFactory::createNode({}, {rmwSectionA}, nullptr, "memset", "memsetRmwA");
    pNode memcopyInsideRmwSection =
        NodeFactory::createNode({rmwSectionA}, {rmwSectionB}, nullptr, "memcpy", "memcopyInsideRmwSection");
    pNode memcopyRmwToMultiBuffer =
        NodeFactory::createNode({rmwSectionB}, {multiBufferB}, nullptr, "memcpy", "memcopyRmwToMultiBuffer");
    pNode memcopyInsideMultiBuffer =
        NodeFactory::createNode({multiBufferB}, {multiBufferA}, nullptr, "memcpy", "memcopyInsideMultiBuffer");

    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memsetRmwA));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopyInsideRmwSection));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopyRmwToMultiBuffer));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopyInsideMultiBuffer));

    gaudiGraph.addControlDependency({memsetRmwA}, {memcopyInsideRmwSection});

    setNonPersistentSectionInfo(gaudiGraph);
    allocateTensors(gaudiGraph);

    // Check that all the tensors are in Sram
    for (const TensorPtr& t : gaudiGraph.getTensors())
    {
        ASSERT_TRUE(t->inSram());
    }

    // RMW section - check allocation sizes and offsets
    for (const auto& t : rmwSectionTensors)
    {
        ASSERT_EQ(t->getTensorAnnotation().sizeToAllocate.value(), rmwSectionAllcationSize);
    }
    ASSERT_EQ(rmwSectionTensors[1]->getSramOffset() - rmwSectionTensors[0]->getSramOffset(), rmwSectionOffsets[1]);
}

TEST_F(MemorySectionTest, single_dram_section)
{
    GaudiGraph gaudiGraph;
    std::unique_ptr<CodeGenerator>& codeGenerator = gaudiGraph.getCodeGenerator();
    codeGenerator->getWorkspaceAllocator().Init(codeGenerator->getDramSize(), codeGenerator->getDramBaseAddr());
    codeGenerator->getSramAllocator().Init(codeGenerator->getSramSize(),  codeGenerator->getSramBaseAddr());

    std::vector<TSize> sizes = {5, 3, 1, 10};

    pTensor section1A(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    pTensor section1B(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    pTensor section1C(new Tensor(sizes.size(), sizes.data(), syn_type_float));

    std::vector<uint64_t> section1Offsets = {0, 5002, 13030};
    TensorVector          section1Tensors = {section1A, section1B, section1C};

    for (auto i = 0; i < section1Tensors.size(); ++i)
    {
        section1Tensors[i]->setTensorInWorkspace();
        section1Tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(23);
        section1Tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(section1Offsets[i]);
    }

    unsigned int section1AllcationSize = calcMemSectionSize(section1Tensors);

    pNode memsetA     = NodeFactory::createNode({}, {section1A}, nullptr, "memset", "memsetA");
    pNode memcopyAtoB = NodeFactory::createNode({section1A}, {section1B}, nullptr, "memcpy", "memcopyAtoB");
    pNode memcopyBtoC = NodeFactory::createNode({section1B}, {section1C}, nullptr, "memcpy", "memcopyBtoC");

    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memsetA));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopyAtoB));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopyBtoC));

    setNonPersistentSectionInfo(gaudiGraph);
    allocateTensors(gaudiGraph);

    // Check that all the tensors are in DRAM
    for (const TensorPtr& t : gaudiGraph.getTensors())
    {
        ASSERT_TRUE(t->inDram());
    }

    // Check allocation sizes and offsets
    for (const auto& t : section1Tensors)
    {
        ASSERT_EQ(t->getTensorAnnotation().sizeToAllocate.value(), section1AllcationSize);
    }
    ASSERT_EQ(section1Tensors[1]->getDramOffset() - section1Tensors[0]->getDramOffset(), section1Offsets[1]);
    ASSERT_EQ(section1Tensors[2]->getDramOffset() - section1Tensors[0]->getDramOffset(), section1Offsets[2]);
}
