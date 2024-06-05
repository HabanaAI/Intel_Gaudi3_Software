#include "gaudi_graph.h"
#include "tensor.h"
#include "passes/sram_management/bundle.h"
#include "passes/sram_management/tensor_section.h"
#include "passes/sram_management/tensor_slicer.h"
#include "passes/sram_management/bundle_slicer.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"

class BeSlicerTest : public GraphOptimizerTest
{
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    }
};

using SliceOperand  = Bundle::Solution::SlicedOperand;
using pSliceOperand = Bundle::Solution::pSlicedOperand;

TEST_F(BeSlicerTest, tensor_slicer)
{
    GaudiGraph            graph;
    static const TSize orig_C = 320;
    static const TSize orig_W = 34;
    static const TSize orig_H = 511;
    static const TSize orig_B = 10;

    static const TSize chunk_C = 256;
    static const TSize chunk_W = orig_W;
    static const TSize chunk_H = 256;
    static const TSize chunk_B = orig_B;

    TSize originalSize[] = {orig_C, orig_W, orig_H, orig_B};
    pTensor origTensor(new Tensor(4, originalSize, syn_type_bf16));

    pSliceOperand sliceOp(new SliceOperand(origTensor));
    SizeArray chunk = {chunk_C, chunk_W, chunk_H, chunk_B};
    sliceOp->chunkDimensions = chunk;

    TensorSlicer slicer(sliceOp);

    pTensor firstSlice  = slicer.getSlice(*graph.getHALReader(), {0}, false);
    pTensor secondSlice = slicer.getSlice(*graph.getHALReader(), {0}, false);
    pTensor thirdSlice  = slicer.getSlice(*graph.getHALReader(), {0}, true);
    pTensor fourthSlice = slicer.getSlice(*graph.getHALReader(), {1, 0, 1, 0, 0}, false);
    slicer.destroySlice({0});
    pTensor fifthSlice = slicer.getSlice(*graph.getHALReader(), {0});

    ASSERT_EQ(firstSlice, secondSlice);
    ASSERT_NE(firstSlice, thirdSlice);
    ASSERT_NE(firstSlice, fifthSlice);
    ASSERT_NE(thirdSlice, fifthSlice);

    ASSERT_EQ(fourthSlice->getSizeInElements(DIM_C), orig_C - chunk_C);
    ASSERT_EQ(fourthSlice->getSizeInElements(DIM_W), orig_W);
    ASSERT_EQ(fourthSlice->getSizeInElements(DIM_H), orig_H - chunk_H);
    ASSERT_EQ(fourthSlice->getSizeInElements(DIM_B), orig_B);
}

TEST_F(BeSlicerTest, tensor_section_producers)
{
    static const TSize orig_C = 320;
    static const TSize orig_W = 34;
    static const TSize orig_H = 511;
    static const TSize orig_B = 10;

    static const TSize chunk_C = 256;
    static const TSize chunk_W = orig_W;
    static const TSize chunk_H = 256;
    static const TSize chunk_B = orig_B;

    TSize originalSize[] = {orig_C, orig_W, orig_H, orig_B};
    pTensor origTensor(new Tensor(4, originalSize, syn_type_bf16));

    pSliceOperand sliceOp(new SliceOperand(origTensor));
    SizeArray chunk = {chunk_C, chunk_W, chunk_H, chunk_B};
    sliceOp->chunkDimensions = chunk;
    sliceOp->resideInSRAM = true;

    GaudiGraph           graph;
    TensorSection        section(*graph.getHALReader(), sliceOp, 0, BundleType::UNDEFINED, Settable<uint64_t>(), 0);
    uint32_t opIdx = 0;

    {
        // First Concat
        {
            // First reduction
            section.addProduceSlice({0}, opIdx++);
            section.addProduceSlice({0}, opIdx++);
            section.addProduceSlice({0}, opIdx++);
        }
        {
            // Second reduction
            section.addProduceSlice({1,0,0,0,0}, opIdx++);
            section.addProduceSlice({1,0,0,0,0}, opIdx++);
            section.addProduceSlice({1,0,0,0,0}, opIdx++);
        }
    }

    {
        // Second concat
        section.addProduceSlice({0,0,1,0,0}, opIdx++);
        section.addProduceSlice({1,0,1,0,0}, opIdx++);
    }

    section.generateGraphSection(graph, true);

    uint32_t nodeIdx = 0;
    for (auto node : graph.getExeSortedNodes())
    {
        switch(nodeIdx++)
        {
            case 0:
            case 2:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_REDUCTION);
                break;
            case 1:
            case 3:
            case 5:
            case 6:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_MEMCOPY);
                break;
            case 4:
            case 7:
            case 8:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_CONCAT);
                break;
        }
    }
    ASSERT_EQ(graph.getExeSortedNodes().size(), 9);
}

TEST_F(BeSlicerTest, tensor_section_consumers)
{
    static const TSize orig_C = 320;
    static const TSize orig_W = 34;
    static const TSize orig_H = 511;
    static const TSize orig_B = 10;

    static const TSize chunk_C = 256;
    static const TSize chunk_W = orig_W;
    static const TSize chunk_H = 256;
    static const TSize chunk_B = orig_B;

    TSize originalSize[] = {orig_C, orig_W, orig_H, orig_B};
    pTensor origTensor(new Tensor(4, originalSize, syn_type_bf16));

    pSliceOperand sliceOp(new SliceOperand(origTensor));
    SizeArray chunk = {chunk_C, chunk_W, chunk_H, chunk_B};
    sliceOp->chunkDimensions = chunk;
    sliceOp->resideInSRAM = true;

    GaudiGraph    graph;
    TensorSection section(*graph.getHALReader(), sliceOp, 0, BundleType::UNDEFINED, Settable<uint64_t>(), 0);
    uint32_t opIdx = 0;

    // First memcpy
    section.addConsumeSlice({0}, opIdx++);
    section.addConsumeSlice({0}, opIdx++);
    // Third memcpy
    section.addConsumeSlice({1,0,0,0,0}, opIdx++);
    // Fourth memcpy
    section.addConsumeSlice({0,0,1,0,0}, opIdx++);
    // Fifth memcpy
    section.addConsumeSlice({1,0,1,0,0}, opIdx++);

    section.generateGraphSection(graph, true);

    uint32_t nodeIdx = 0;
    for (auto node : graph.getExeSortedNodes())
    {
        switch (nodeIdx)
        {
            case 0:
            case 1:
            case 4:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_SPLIT);
                break;
            case 2:
            case 3:
            case 5:
            case 6:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_MEMCOPY);
                break;
        }
        ++nodeIdx;
    }
    ASSERT_EQ(graph.getExeSortedNodes().size(), 7);
}

TEST_F(BeSlicerTest, tensor_section_trivial_solution)
{
    static const TSize orig_C = 320;
    static const TSize orig_W = 34;
    static const TSize orig_H = 511;
    static const TSize orig_B = 10;

    TSize originalSize[] = {orig_C, orig_W, orig_H, orig_B};
    pTensor origTensor(new Tensor(4, originalSize, syn_type_bf16));

    pSliceOperand sliceOp(new SliceOperand(origTensor));
    SizeArray chunk = {orig_C, orig_W, orig_H, orig_B};
    sliceOp->chunkDimensions = chunk;
    sliceOp->resideInSRAM = true;

    GaudiGraph    graph;
    TensorSection section(*graph.getHALReader(), sliceOp, 0, BundleType::UNDEFINED, Settable<uint64_t>(), 0);
    uint32_t opIdx = 0;

    // Reduction
    section.addProduceSlice({0}, opIdx++);
    section.addProduceSlice({0}, opIdx++);
    // Memcpy up
    section.addConsumeSlice({0}, opIdx++);

    section.generateGraphSection(graph, true);

    uint32_t nodeIdx = 0;
    for (auto node : graph.getExeSortedNodes())
    {
        switch (nodeIdx)
        {
            case 0:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_REDUCTION);
                break;
            case 1:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_MEMCOPY);
                break;
        }
        ++nodeIdx;
    }
    ASSERT_EQ(graph.getExeSortedNodes().size(), 2);
}

TEST_F(BeSlicerTest, tensor_section_pass_through)
{
    static const TSize orig_C = 320;
    static const TSize orig_W = 34;
    static const TSize orig_H = 511;
    static const TSize orig_B = 10;

    static const TSize chunk_C = 256;
    static const TSize chunk_W = orig_W;
    static const TSize chunk_H = 256;
    static const TSize chunk_B = orig_B;

    TSize originalSize[] = {orig_C, orig_W, orig_H, orig_B};
    pTensor origTensor(new Tensor(4, originalSize, syn_type_bf16));

    pSliceOperand sliceOp(new SliceOperand(origTensor));
    SizeArray chunk = {chunk_C, chunk_W, chunk_H, chunk_B};
    sliceOp->chunkDimensions = chunk;
    sliceOp->resideInSRAM = true;

    GaudiGraph    graph;
    TensorSection section(*graph.getHALReader(), sliceOp, 0, BundleType::UNDEFINED, Settable<uint64_t>(), 0);
    uint32_t opIdx = 0;

    section.addProduceSlice({0}, opIdx++);
    section.addConsumeSlice({0}, opIdx++);

    section.addProduceSlice({1,0,0,0,0}, opIdx++);
    section.addConsumeSlice({1,0,0,0,0}, opIdx++);

    section.addProduceSlice({0,0,1,0,0}, opIdx++);
    section.addConsumeSlice({0,0,1,0,0}, opIdx++);

    section.addProduceSlice({1,0,1,0,0}, opIdx++);
    section.addConsumeSlice({1,0,1,0,0}, opIdx++);

    // Memcpy up and down
    section.addConsumeSlice({0}, opIdx++);

    section.generateGraphSection(graph, true);

    uint32_t nodeIdx = 0;
    for (auto node : graph.getExeSortedNodes())
    {
        switch (nodeIdx)
        {
            case 0:
            case 1:
            case 3:
            case 4:
            case 7:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_MEMCOPY);
                break;
            case 2:
            case 5:
            case 6:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_CONCAT);
                break;
        }
        ++nodeIdx;
    }
    ASSERT_EQ(graph.getExeSortedNodes().size(), 8);
}

static Bundle::Solution::Operation duplicateOperation(const Bundle::Solution::Operation& other)
{
    Bundle::Solution::Operation newOp(other.originalNode);
    for (auto pInput : other.inputs)
    {
        newOp.inputs.push_back(std::make_shared<Bundle::Solution::Operation::SliceReference>(*pInput));
    }
    for (auto pOutput : other.outputs)
    {
        newOp.outputs.push_back(std::make_shared<Bundle::Solution::Operation::SliceReference>(*pOutput));
    }
    return newOp;
}

TEST_F(BeSlicerTest, bundle_slicer)
{
    static const TSize orig_C = 320;
    static const TSize orig_W = 34;
    static const TSize orig_H = 511;
    static const TSize orig_B = 10;

    static const TSize chunk_C = 256;
    static const TSize chunk_W = orig_W;
    static const TSize chunk_H = 256;
    static const TSize chunk_B = orig_B;

    TSize originalSize[] = {orig_C, orig_W, orig_H, orig_B};
    pTensor startTensor(new Tensor(4, originalSize, syn_type_bf16));
    startTensor->setName("start_tensor");
    pTensor middleTensor(new Tensor(4, originalSize, syn_type_bf16));
    middleTensor->setName("middle_tensor");
    pTensor endTensor(new Tensor(4, originalSize, syn_type_bf16));
    endTensor->setName("end_tensor");

    //In this test, middle and end tensors are persistent and should be evicted.
    synMemoryDescriptor descriptor;
    descriptor.m_isPersistent = true;

    middleTensor->setMemoryDescriptor(descriptor);
    endTensor->setMemoryDescriptor(descriptor);

    SizeArray chunk = {chunk_C, chunk_W, chunk_H, chunk_B};
    pSliceOperand startSliceOp(new SliceOperand(startTensor));
    startSliceOp->numOfBuffers = 2;
    startSliceOp->chunkDimensions = chunk;
    startSliceOp->resideInSRAM = true;

    pSliceOperand middleSliceOp(new SliceOperand(middleTensor));
    middleSliceOp->numOfBuffers = 2;
    middleSliceOp->chunkDimensions = chunk;
    middleSliceOp->resideInSRAM = true;

    pSliceOperand endSliceOp(new SliceOperand(endTensor));
    endSliceOp->numOfBuffers = 2;
    endSliceOp->chunkDimensions = chunk;
    endSliceOp->resideInSRAM = true;

    pNode firstNode  = NodeFactory::createNode({startTensor}, {middleTensor}, nullptr, NOP_KERNEL_NAME, "first_node");
    pNode secondNode = NodeFactory::createNode({middleTensor}, {endTensor}, nullptr, NOP_KERNEL_NAME, "second_node");

    GaudiGraph graph;
    GraphEditor::addNode(graph, firstNode);
    GraphEditor::addNode(graph, secondNode);

    Bundle bundle;
    bundle.addNode(firstNode);
    bundle.addNode(secondNode);

    auto& solution = bundle.getSolution();

    // First Slice
    solution.operations.push_back(Bundle::Solution::Operation(firstNode));
    auto& op1 = solution.operations.back();
    op1.inputs.resize(1, std::make_shared<Bundle::Solution::Operation::SliceReference>(startSliceOp));
    op1.inputs.back()->coordinates = {0};

    op1.outputs.resize(1, std::make_shared<Bundle::Solution::Operation::SliceReference>(middleSliceOp));
    op1.outputs.back()->coordinates = {0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    // Second Slice
    solution.operations.push_back(duplicateOperation(op1));
    auto& op2 = solution.operations.back();
    op2.inputs.back()->coordinates = {1,0,0,0,0};
    op2.outputs.back()->coordinates = {1,0,0,0,0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    // Third Slice
    solution.operations.push_back(duplicateOperation(op1));
    auto& op3 = solution.operations.back();
    op3.inputs.back()->coordinates = {0,0,1,0,0};
    op3.outputs.back()->coordinates = {0,0,1,0,0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    // Fourth Slice
    solution.operations.push_back(duplicateOperation(op1));
    auto& op4 = solution.operations.back();
    op4.inputs.back()->coordinates = {1,0,1,0,0};
    op4.outputs.back()->coordinates = {1,0,1,0,0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    BundleSlicer::sliceBundle(bundle, graph);
    graphVisualizationPost(graph);

    uint32_t nodeIdx = 0;
    for (auto node : graph.getExeSortedNodes())
    {
        switch (nodeIdx)
        {
            case 0:
            case 1:
            case 2:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_SPLIT) << "Wrong node at: " << nodeIdx;
                break;
            case 3:
            case 4:
            case 7:
            case 8:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_MEMCOPY) << "Wrong node at: " << nodeIdx;
                ASSERT_TRUE(! node->getInput(0)->inSram() && node->getOutput(0)->inSram()) << "Wrong node at: " << nodeIdx << " Eviction instead of fill";
                break;
            case 11:
            case 12:
            case 16:
            case 17:
            case 21:
            case 22:
            case 23:
            case 26:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_MEMCOPY) << "Wrong node at: " << nodeIdx;
                ASSERT_TRUE(node->getInput(0)->inSram() && ! node->getOutput(0)->inSram()) << "Wrong node at: " << nodeIdx << " Fill instead of eviction";
                break;
            case 5:
            case 6:
            case 9:
            case 10:
            case 14:
            case 15:
            case 19:
            case 20:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_USER) << "Wrong node at: " << nodeIdx;
                break;
            case 13:
            case 18:
            case 24:
            case 25:
            case 27:
            case 28:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_CONCAT) << "Wrong node at: " << nodeIdx;
                break;
        }
        ++nodeIdx;
    }
    ASSERT_EQ(graph.getExeSortedNodes().size(), 29);
}

TEST_F(BeSlicerTest, bundle_slicer_evict_end)
{
    static const TSize orig_C = 320;
    static const TSize orig_W = 34;
    static const TSize orig_H = 511;
    static const TSize orig_B = 10;

    static const TSize chunk_C = 256;
    static const TSize chunk_W = orig_W;
    static const TSize chunk_H = 256;
    static const TSize chunk_B = orig_B;

    TSize originalSize[] = {orig_C, orig_W, orig_H, orig_B};
    pTensor startTensor(new Tensor(4, originalSize, syn_type_bf16));
    startTensor->setName("start_tensor");
    pTensor middleTensor(new Tensor(4, originalSize, syn_type_bf16));
    middleTensor->setName("middle_tensor");
    pTensor endTensor(new Tensor(4, originalSize, syn_type_bf16));
    endTensor->setName("end_tensor");

    //In this test, only end tensor is persistent and should be evicted.
    synMemoryDescriptor descriptor;
    descriptor.m_isPersistent = true;

    endTensor->setMemoryDescriptor(descriptor);

    SizeArray chunk = {chunk_C, chunk_W, chunk_H, chunk_B};
    pSliceOperand startSliceOp(new SliceOperand(startTensor));
    startSliceOp->chunkDimensions = chunk;
    startSliceOp->resideInSRAM = true;

    pSliceOperand middleSliceOp(new SliceOperand(middleTensor));
    middleSliceOp->chunkDimensions = chunk;
    middleSliceOp->resideInSRAM = true;

    pSliceOperand endSliceOp(new SliceOperand(endTensor));
    endSliceOp->chunkDimensions = chunk;
    endSliceOp->resideInSRAM = true;

    pNode firstNode  = NodeFactory::createNode({startTensor}, {middleTensor}, nullptr, NOP_KERNEL_NAME, "first_node");
    pNode secondNode = NodeFactory::createNode({middleTensor}, {endTensor}, nullptr, NOP_KERNEL_NAME, "second_node");

    GaudiGraph graph;
    GraphEditor::addNode(graph, firstNode);
    GraphEditor::addNode(graph, secondNode);

    Bundle bundle;
    bundle.addNode(firstNode);
    bundle.addNode(secondNode);

    auto& solution = bundle.getSolution();

    // First Slice
    solution.operations.push_back(Bundle::Solution::Operation(firstNode));
    auto& op1 = solution.operations.back();
    op1.inputs.resize(1, std::make_shared<Bundle::Solution::Operation::SliceReference>(startSliceOp));
    op1.inputs.back()->coordinates = {0};

    op1.outputs.resize(1, std::make_shared<Bundle::Solution::Operation::SliceReference>(middleSliceOp));
    op1.outputs.back()->coordinates = {0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    // Second Slice
    solution.operations.push_back(duplicateOperation(op1));
    auto& op2 = solution.operations.back();
    op2.inputs.back()->coordinates = {1,0,0,0,0};
    op2.outputs.back()->coordinates = {1,0,0,0,0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    // Third Slice
    solution.operations.push_back(duplicateOperation(op1));
    auto& op3 = solution.operations.back();
    op3.inputs.back()->coordinates = {0,0,1,0,0};
    op3.outputs.back()->coordinates = {0,0,1,0,0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    // Fourth Slice
    solution.operations.push_back(duplicateOperation(op1));
    auto& op4 = solution.operations.back();
    op4.inputs.back()->coordinates = {1,0,1,0,0};
    op4.outputs.back()->coordinates = {1,0,1,0,0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    BundleSlicer::sliceBundle(bundle, graph);
    graphVisualizationPost(graph);

    uint32_t nodeIdx = 0;
    for (auto node : graph.getExeSortedNodes())
    {
        switch (nodeIdx)
        {
            case 0:
            case 1:
            case 2:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_SPLIT) << "Wrong node at: " << nodeIdx;
                break;
            case 3:
            case 5:
            case 9:
            case 14:
            case 8:
            case 12:
            case 17:
            case 19:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_MEMCOPY) << "Wrong node at: " << nodeIdx;
                break;
            case 4:
            case 6:
            case 7:
            case 10:
            case 11:
            case 15:
            case 16:
            case 18:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_USER) << "Wrong node at: " << nodeIdx;
                break;
            case 13:
            case 20:
            case 21:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_CONCAT) << "Wrong node at: " << nodeIdx;
                break;
        }
        ++nodeIdx;
    }
    ASSERT_EQ(graph.getExeSortedNodes().size(), 22);
}

TEST_F(BeSlicerTest, intermediate_reduction)
{
    static const TSize orig_C = 512;
    static const TSize orig_W = 34;
    static const TSize orig_H = 511;
    static const TSize orig_B = 10;

    static const TSize chunk_C = 256;
    static const TSize chunk_W = orig_W;
    static const TSize chunk_H = 256;
    static const TSize chunk_B = orig_B;

    TSize startSize[] = {orig_C, orig_W, orig_H, orig_B};
    pTensor startTensor(new Tensor(4, startSize, syn_type_bf16));
    startTensor->setName("start_tensor");
    // After reduction on C
    TSize endSize[] = {chunk_C, orig_W, orig_H, orig_B};
    pTensor middleTensor(new Tensor(4, endSize, syn_type_bf16));
    middleTensor->setName("middle_tensor");
    pTensor endTensor(new Tensor(4, endSize, syn_type_bf16));
    endTensor->setName("end_tensor");

    //In this test end tensor is persistent and should be evicted.
    synMemoryDescriptor descriptor;
    descriptor.m_isPersistent = true;
    endTensor->setMemoryDescriptor(descriptor);

    SizeArray chunk = {chunk_C, chunk_W, chunk_H, chunk_B};
    pSliceOperand startSliceOp(new SliceOperand(startTensor));
    startSliceOp->chunkDimensions = chunk;
    startSliceOp->resideInSRAM = true;

    pSliceOperand middleSliceOp(new SliceOperand(middleTensor));
    middleSliceOp->numOfBuffers = 2;
    middleSliceOp->chunkDimensions = chunk;
    middleSliceOp->resideInSRAM = true;

    pSliceOperand endSliceOp(new SliceOperand(endTensor));
    endSliceOp->numOfBuffers = 2;
    endSliceOp->chunkDimensions = chunk;
    endSliceOp->resideInSRAM = true;

    pNode firstNode  = NodeFactory::createNode({startTensor}, {middleTensor}, nullptr, NOP_KERNEL_NAME, "first_node");
    pNode secondNode = NodeFactory::createNode({middleTensor}, {endTensor}, nullptr, NOP_KERNEL_NAME, "second_node");

    GaudiGraph graph;
    GraphEditor::addNode(graph, firstNode);
    GraphEditor::addNode(graph, secondNode);

    Bundle bundle;
    bundle.addNode(firstNode);
    bundle.addNode(secondNode);

    auto& solution = bundle.getSolution();

    // First Slice
    solution.operations.push_back(Bundle::Solution::Operation(firstNode));
    auto& op1 = solution.operations.back();
    op1.inputs.resize(1, std::make_shared<Bundle::Solution::Operation::SliceReference>(startSliceOp));
    op1.inputs.back()->coordinates = {0};

    op1.outputs.resize(1, std::make_shared<Bundle::Solution::Operation::SliceReference>(middleSliceOp));
    op1.outputs.back()->coordinates = {0};

    // Add reduction
    solution.operations.push_back(duplicateOperation(op1));
    auto& op2 = solution.operations.back();
    op2.inputs.back()->coordinates = {1,0,0,0,0};

    solution.operations.push_back(duplicateOperation(op1));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    // Second Slice
    solution.operations.push_back(duplicateOperation(op1));
    auto& op3 = solution.operations.back();
    op3.inputs.back()->coordinates = {0,0,1,0,0};
    op3.outputs.back()->coordinates = {0,0,1,0,0};

    // Add reduction
    solution.operations.push_back(duplicateOperation(op3));
    auto& op4 = solution.operations.back();
    op4.inputs.back()->coordinates = {1,0,1,0,0};

    solution.operations.push_back(duplicateOperation(op3));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    BundleSlicer::sliceBundle(bundle, graph);
    graphVisualizationPost(graph);

    uint32_t nodeIdx = 0;
    for (auto node : graph.getExeSortedNodes())
    {
        switch (nodeIdx)
        {
            case 0:
            case 1:
            case 2:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_SPLIT) << "Wrong node at: " << nodeIdx;
                break;
            case 3:
            case 5:
            case 8:
            case 10:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_MEMCOPY) << "Wrong node at: " << nodeIdx;
                ASSERT_TRUE(! node->getInput(0)->inSram() && node->getOutput(0)->inSram()) << "Wrong node at: " << nodeIdx << " Eviction instead of fill";
                break;
            case 15:
            case 16:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_MEMCOPY) << "Wrong node at: " << nodeIdx;
                ASSERT_TRUE(node->getInput(0)->inSram() && ! node->getOutput(0)->inSram()) << "Wrong node at: " << nodeIdx << " Fill instead of eviction";
                break;
            case 4:
            case 6:
            case 9:
            case 11:
            case 13:
            case 14:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_USER) << "Wrong node at: " << nodeIdx;
                break;
            case 7:
            case 12:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_REDUCTION) << "Wrong node at: " << nodeIdx;
                break;
            case 17:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_CONCAT) << "Wrong node at: " << nodeIdx;
                break;
        }
        ++nodeIdx;
    }
    ASSERT_EQ(graph.getExeSortedNodes().size(), 18);
}

TEST_F(BeSlicerTest, bundle_slicer_edges_in_sram)
{
    static const TSize orig_C = 320;
    static const TSize orig_W = 34;
    static const TSize orig_H = 511;
    static const TSize orig_B = 10;

    static const TSize chunk_C = 256;
    static const TSize chunk_W = orig_W;
    static const TSize chunk_H = 256;
    static const TSize chunk_B = orig_B;

    TSize originalSize[] = {orig_C, orig_W, orig_H, orig_B};
    pTensor startTensor(new Tensor(4, originalSize, syn_type_bf16));
    startTensor->setName("start_tensor");
    startTensor->setTensorInSram();
    pTensor middleTensor(new Tensor(4, originalSize, syn_type_bf16));
    middleTensor->setName("middle_tensor");
    pTensor endTensor(new Tensor(4, originalSize, syn_type_bf16));
    endTensor->setName("end_tensor");

    endTensor->setTensorInSram();

    SizeArray chunk = {chunk_C, chunk_W, chunk_H, chunk_B};
    pSliceOperand startSliceOp(new SliceOperand(startTensor));
    startSliceOp->chunkDimensions = chunk;
    startSliceOp->resideInSRAM = true;

    pSliceOperand middleSliceOp(new SliceOperand(middleTensor));
    middleSliceOp->chunkDimensions = chunk;
    middleSliceOp->resideInSRAM = true;

    pSliceOperand endSliceOp(new SliceOperand(endTensor));
    endSliceOp->chunkDimensions = chunk;
    endSliceOp->resideInSRAM = true;

    pNode firstNode  = NodeFactory::createNode({startTensor}, {middleTensor}, nullptr, NOP_KERNEL_NAME, "first_node");
    pNode secondNode = NodeFactory::createNode({middleTensor}, {endTensor}, nullptr, NOP_KERNEL_NAME, "second_node");

    // Create another node for graph, so the last tensor will have consumer outside the bundle
    pTensor outOfBundleTensor(new Tensor(4, originalSize, syn_type_bf16));
    outOfBundleTensor->setName("outOfBundleTensor");
    pNode outOfBundleNode =
        NodeFactory::createNode({endTensor}, {outOfBundleTensor}, nullptr, NOP_KERNEL_NAME, "outOfBundleNode");

    GaudiGraph graph;
    GraphEditor::addNode(graph, firstNode);
    GraphEditor::addNode(graph, secondNode);
    GraphEditor::addNode(graph, outOfBundleNode);

    Bundle bundle;
    bundle.addNode(firstNode);
    bundle.addNode(secondNode);

    auto& solution = bundle.getSolution();

    // First Slice
    solution.operations.push_back(Bundle::Solution::Operation(firstNode));
    auto& op1 = solution.operations.back();
    op1.inputs.resize(1, std::make_shared<Bundle::Solution::Operation::SliceReference>(startSliceOp));
    op1.inputs.back()->coordinates = {0};

    op1.outputs.resize(1, std::make_shared<Bundle::Solution::Operation::SliceReference>(middleSliceOp));
    op1.outputs.back()->coordinates = {0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    // Second Slice
    solution.operations.push_back(duplicateOperation(op1));
    auto& op2 = solution.operations.back();
    op2.inputs.back()->coordinates = {1,0,0,0,0};
    op2.outputs.back()->coordinates = {1,0,0,0,0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    // Third Slice
    solution.operations.push_back(duplicateOperation(op1));
    auto& op3 = solution.operations.back();
    op3.inputs.back()->coordinates = {0,0,1,0,0};
    op3.outputs.back()->coordinates = {0,0,1,0,0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    // Fourth Slice
    solution.operations.push_back(duplicateOperation(op1));
    auto& op4 = solution.operations.back();
    op4.inputs.back()->coordinates = {1,0,1,0,0};
    op4.outputs.back()->coordinates = {1,0,1,0,0};

    solution.operations.push_back(duplicateOperation(solution.operations.back()));
    solution.operations.back().originalNode = secondNode;
    for (auto& t : solution.operations.back().inputs)
    {
        t->operand = middleSliceOp;
    }
    for (auto& t : solution.operations.back().outputs)
    {
        t->operand = endSliceOp;
    }

    BundleSlicer::sliceBundle(bundle, graph);
    graphVisualizationPost(graph);

    uint32_t nodeIdx = 0;
    for (auto node : graph.getExeSortedNodes())
    {
        switch (nodeIdx)
        {
            case 0:
            case 1:
            case 7:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_SPLIT) << "Wrong node at: " << nodeIdx;
                break;
            case 2:
            case 3:
            case 4:
            case 5:
            case 8:
            case 9:
            case 10:
            case 11:
            case 14:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_USER) << "Wrong node at: " << nodeIdx;
                break;
            case 6:
            case 12:
            case 13:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_CONCAT) << "Wrong node at: " << nodeIdx;
                break;
        }
        ++nodeIdx;
    }
    ASSERT_EQ(graph.getExeSortedNodes().size(), 15);
}
