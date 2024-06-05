#include "passes/sram_management/bundle.h"
#include "passes/sram_management/sram_management.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "graph_compiler/compilation_hal_reader.h"

#include "gtest/gtest.h"
#include <graph_compiler/passes/sram_management/sliced_operand_traversal.h>
#include <graph_compiler/passes/sram_management/solution_generator.h>
#include <graph_compiler/passes/sram_management/reshape_aligner.h>
#include <graph_compiler/passes/sram_management/batch_slicing_solver.h>
#include "settable.h"
#include "sram_management_fe_test.h"
#include "perf_lib_layer_params.h"
#include "graph_compiler/passes/sram_management/slicing_utils.h"
#include "graph_compiler/passes/sram_management/spatial_slicing_solver.h"
#include "graph_compiler/passes/sram_management/common_dim_slicing_solver.h"
#include "graph_compiler/passes/sram_management/mme_dim_controller.h"
#include "graph_compiler/passes/sram_management/bundle_slicer.h"

#include "graph_compiler/passes/sram_management/bundle_expander.h"
#include <habana_global_conf.h>
#include "graph_compiler/passes/sram_management/flatten_mme.h"
#include "node_utils.h"
#include "platform/gaudi/graph_compiler/passes.h"

using namespace gaudi;

void SRAMManagementTest::SetUp()
{
    GraphOptimizerTest::SetUp();
    m_orgSramMaxCap = GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.getValueStr();
    m_orgKnobs = SlicingBrain::knobs;

    CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
}

void SRAMManagementTest::TearDown()
{
    SlicingBrain::knobs = m_orgKnobs;
    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setFromString(m_orgSramMaxCap);
    GraphOptimizerTest::TearDown();
}

void SRAMManagementTest::setGlobalConfForTest(hl_gcfg::GcfgItem& gConf, const std::string& stringValue)
{
    GraphOptimizerTest::setGlobalConfForTest(gConf, stringValue);
    MMESlicingBrain dummyBrain {getGraph()};  // Force re-init of knobs
    UNUSED(dummyBrain);
}

pTensor SRAMManagementTest::createTensor(const std::vector<TSize>& shape,
                                         synDataType               dataType,
                                         bool                      isPersistent /*= true*/,
                                         const std::vector<TSize>& minShape /*= std::vector<TSize>()*/,
                                         synTensorType             tensorType /*= DATA_TENSOR*/)
{
    synMemoryDescriptor memDesc(isPersistent);
    auto                tensor = std::make_shared<Tensor>(shape.size(), shape.data(), dataType);
    tensor->setMemoryDescriptor(memDesc);
    if (isPersistent)
    {
        tensor->setMemorySectionID(m_memorySectionId++);
    }
    tensor->setDramOffset(0x1000000);
    tensor->map();
    if (!minShape.empty())
    {
        tensor->setMinSize(minShape.data());
    }
    tensor->setTensorType(tensorType);
    return tensor;
}

pBundle SRAMManagementTest::createSingleMMENodeBundle(TensorVector       inputs,
                                                      TensorVector       outputs,
                                                      const std::string& guid,
                                                      void*              params,
                                                      unsigned           paramsSize)
{
    synConvolutionParams localParams{};
    if (!params)
    {
        params = &localParams;
        paramsSize = sizeof(localParams);
    }
    if  (guid == NodeFactory::deDwNodeTypeName)
    {
        localParams.kH = outputs[0]->getSizeInElements(WEIGHT_DIM_R);
        localParams.kW = outputs[0]->getSizeInElements(WEIGHT_DIM_S);
    }
    else
    {
        localParams.kH = inputs[1]->getSizeInElements(WEIGHT_DIM_R);
        localParams.kW = inputs[1]->getSizeInElements(WEIGHT_DIM_S);
    }
    pNode n = NodeFactory::createNode(inputs, outputs, params, paramsSize, guid.c_str(), guid);
    if (n->getNodeType() != Node::TYPE_FC) // Not supported yet in GaudiGraph
    {
        GraphEditor::addNode(getGraph(), n);
    }
    auto bundle = std::make_shared<Bundle>(BundleType::MME);
    bundle->addNode(n);
    return bundle;
}

const MMESlicingBrain& SRAMManagementTest::getMmeBrain() const
{
    return m_brains.m_mmeBrain;
}

const TPCSlicingBrain& SRAMManagementTest::getTpcBrain() const
{
    return m_brains.m_tpcBrain;
}

const AllBrains& SRAMManagementTest::getSlicingBrains() const
{
    return m_brains;
}

void SRAMManagementTest::checkSolutionSize(const Solution& solution, unsigned operandSize, unsigned operationSize) const
{
    ASSERT_EQ(operandSize, solution.operands.size()) << "Wrong number of operands in solution";
    ASSERT_EQ(operationSize, solution.operations.size()) << "Wrong number of operation in solution";
}

void SRAMManagementTest::checkChunkSize(const Solution& solution, const pTensor operand, const SizeArray& expChunkSize)
{
    Solution::pSlicedOperand slicedOp;
    for (auto op : solution.operands)
    {
        if (op->originalTensor == operand)
        {
            slicedOp = op;
        }
    }
    ASSERT_TRUE(slicedOp) << "Couldn't find sliced " << operand->getName();

    for (unsigned dim = 0; dim < operand->getDim(); dim++)
    {
        ASSERT_EQ(expChunkSize[dim], slicedOp->chunkDimensions[dim]) << "Wrong chunk size in dim " << dim;
    }
}

void SRAMManagementTest::ExecutionOrderChecker::checkWalkRightExecutionOrder(const Solution&  solution,
                                                                             unsigned         expRows,
                                                                             unsigned         expCols,
                                                                             const DimVector& outputSlicedSpatialDims)
{
    HB_ASSERT(outputSlicedSpatialDims.size() == 2, "output should be sliced on 2 dims only");
    if (SlicingBrain::knobs.snakeWalkingTraversal)
    {
        checkWalkRightSnakeExecutionOrder(solution, expRows, expCols, outputSlicedSpatialDims);
    }
    else
    {
        checkWalkRightSimpleExecutionOrder(solution, expRows, expCols, outputSlicedSpatialDims);
    }
}

void SRAMManagementTest::ExecutionOrderChecker::checkWalkRightSimpleExecutionOrder(
    const Solution&  solution,
    unsigned int     expRows,
    unsigned int     expCols,
    const DimVector& outputSlicedSpatialDims)
{
    auto it = solution.operations.begin();
    for (unsigned row = 0; row < expRows; row++)
    {
        for (unsigned col = 0; col < expCols; col++)
        {
            ASSERT_NE(it, solution.operations.end()) << "Operation not found for slice [" << row << "][" << col << "]";
            const auto& op = getNextOperation(it);
            ASSERT_EQ(op.inputs[0]->coordinates[m_opASlicingDim], row);
            ASSERT_EQ(op.outputs[0]->coordinates[outputSlicedSpatialDims.front()], row);
            ASSERT_EQ(op.inputs[1]->coordinates[m_opBSlicingDim], col);
            ASSERT_EQ(op.outputs[0]->coordinates[outputSlicedSpatialDims.back()], col);
        }
    }
}

void SRAMManagementTest::ExecutionOrderChecker::checkWalkRightSnakeExecutionOrder(
    const Solution&  solution,
    unsigned int     expRows,
    unsigned int     expCols,
    const DimVector& outputSlicedSpatialDims)
{
    auto it = solution.operations.begin();
    for (int row = 0; row < expRows; row++)
    {
        bool forward = (row % 2) == 0;
        int  kBegin  = forward ? 0 : expCols - 1;
        int  kEnd    = forward ? expCols: 0;

        for (int col = kBegin;
             forward ? col < kEnd : col >= kEnd ;
             forward ? col++      : col--)
        {
            ASSERT_NE(it, solution.operations.end()) << "Operation not found for slice [" << row << "][" << col << "]";
            const auto& op = getNextOperation(it);
            ASSERT_EQ(op.inputs[0]->coordinates[m_opASlicingDim], row);

            ASSERT_EQ(op.outputs[0]->coordinates[outputSlicedSpatialDims.front()], row);
            ASSERT_EQ(op.inputs[1]->coordinates[m_opBSlicingDim], col);
            ASSERT_EQ(op.outputs[0]->coordinates[outputSlicedSpatialDims.back()], col);
        }
    }
}

void SRAMManagementTest::ExecutionOrderChecker::checkWalkDownExecutionOrder(const Solution&  solution,
                                                                            unsigned         expRows,
                                                                            unsigned         expCols,
                                                                            const DimVector& outputSlicedSpatialDims)
{
    HB_ASSERT(outputSlicedSpatialDims.size() == 2, "output should be sliced on 2 dims only");
    if (SlicingBrain::knobs.snakeWalkingTraversal)
    {
        checkWalkDownSnakeExecutionOrder(solution, expRows, expCols, outputSlicedSpatialDims);
    }
    else
    {
        checkWalkDownSimpleExecutionOrder(solution, expRows, expCols, outputSlicedSpatialDims);
    }
}

void SRAMManagementTest::ExecutionOrderChecker::checkWalkDownSimpleExecutionOrder(
    const Solution&  solution,
    unsigned int     expRows,
    unsigned int     expCols,
    const DimVector& outputSlicedSpatialDims)
{
    auto it = solution.operations.begin();
    for (unsigned col = 0; col < expCols; col++)
    {
        for (unsigned row = 0; row < expRows; row++)
        {
            ASSERT_NE(it, solution.operations.end()) << "Operation not found for slice [" << row << "][" << col << "]";
            const auto& op = getNextOperation(it);
            ASSERT_EQ(op.inputs[0]->coordinates[m_opASlicingDim], row);
            ASSERT_EQ(op.outputs[0]->coordinates[outputSlicedSpatialDims.front()], row);

            ASSERT_EQ(op.inputs[1]->coordinates[m_opBSlicingDim], col);
            ASSERT_EQ(op.outputs[0]->coordinates[outputSlicedSpatialDims.back()], col);
        }
    }
}

void SRAMManagementTest::ExecutionOrderChecker::checkWalkDownSnakeExecutionOrder(
    const Solution&  solution,
    unsigned int     expRows,
    unsigned int     expCols,
    const DimVector& outputSlicedSpatialDims)
{
    auto it = solution.operations.begin();
    for (unsigned col = 0; col < expCols; col++)
    {
        bool forward = (col % 2) == 0;
        int  kBegin  = forward ? 0 : expRows - 1;
        int  kEnd    = forward ? expRows: 0;

        for (int row = kBegin;
             forward ? row < kEnd : row >= kEnd ;
             forward ? row++      : row--)
        {
            ASSERT_NE(it, solution.operations.end()) << "Operation not found for slice [" << row << "][" << col << "]";
            const auto& op = getNextOperation(it);
            ASSERT_EQ(op.inputs[0]->coordinates[m_opASlicingDim], row);
            ASSERT_EQ(op.outputs[0]->coordinates[outputSlicedSpatialDims.front()], row);

            ASSERT_EQ(op.inputs[1]->coordinates[m_opBSlicingDim], col);
            ASSERT_EQ(op.outputs[0]->coordinates[outputSlicedSpatialDims.back()], col);
        }
    }
}

void SRAMManagementTest::ExecutionOrderChecker::checkReductionOnlyExecutionOrder(const Solution& solution,
                                                                                 unsigned int expSlices)
{
    auto it = solution.operations.begin();
     for (unsigned i = 0; i < expSlices; i++)
    {
        ASSERT_NE(it, solution.operations.end()) << "Operation not found for slice [" << i << "]";
        const auto& op = getNextOperation(it);
        ASSERT_EQ(op.inputs[0]->coordinates[m_opASlicingDim], i);
        ASSERT_EQ(op.inputs[1]->coordinates[m_opBSlicingDim], i);
        ASSERT_EQ(op.outputs[0]->coordinates, CoordArray{0});
    }
}

void SRAMManagementTest::ExecutionOrderChecker::setOpSkipFactor(unsigned skip)
{
    m_opSkipFactor = skip;
}

const Solution::Operation& SRAMManagementTest::ExecutionOrderChecker::getNextOperation(std::list<Solution::Operation>::const_iterator& currentOperation)
{
    const Solution::Operation& ret = *currentOperation;
    for (int i = 0; i < m_opSkipFactor; i++)
    {
        currentOperation++;
    }
    return ret;
}

void SRAMManagementTest::solveBundleWithStrategy(pBundle bundle)
{
    auto strategies = getMmeBrain().getSolutionStrategies(bundle);
    ASSERT_NE(0, strategies.size()) << "brain did not return any strategy to solve the bundle";
    const auto& winning = findWinningStrategy(strategies, bundle, getGraph(), getSlicingBrains());
    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, winning).fillSolution());
}

bool sliceGraphToSRAMCapacity(HabanaGraph& g);

namespace
{
class TestGraph : public GaudiGraph
{
public:
    bool addValidatedNode(pNode node) override { return GaudiGraph::addValidatedNode(node); }
};
}  // namespace

TEST_F(SRAMManagementTest, testBundleReshapeElimination)
{
    setGlobalConfForTest(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT, "true");

    TestGraph      graph;
    ReshapeAligner reshapeAligner(graph);

    /* Creat the pattern:
     * --[t1]-->(BN2)--[t2]-->(Reshape)--[t3]-->(MME)--[t4]-->*/

    /* Create the mme node */
    synConvolutionParams convParams{};
    pTensor x = createTensor({16, 16}, syn_type_uint8);
    pTensor w = createTensor({16, 16}, syn_type_uint8);
    pTensor o = createTensor({16, 16}, syn_type_uint8);
    pNode fwd = NodeFactory::createNode({x, w}, {o}, &convParams, NodeFactory::convolutionNodeTypeName, "fwd");


    /* Create the reshape node*/
    /*                                   c  h  b  w */
    pTensor batchNormOfm = createTensor({1, 2, 8, 16}, syn_type_uint8);
    pNode   reshape      = NodeFactory::createNode({batchNormOfm},
                                                    {x},
                                                    nullptr,
                                                    NodeFactory::reshapeNodeTypeName,
                                                    "reshape");
    /* Create the TPC BN2 node*/
    /*                                              c  h  b  w */
    pTensor batchNormIfm            = createTensor({1, 2, 8, 16}, syn_type_uint8);
    pTensor tensorRunningMeanIn     = createTensor({1, 2},        syn_type_uint8);
    pTensor sigmaAndSigmaSquaredIn  = createTensor({1, 2},        syn_type_uint8);
    pTensor meanIn                  = createTensor({1, 1},        syn_type_uint8);
    pTensor runningMeanAndVarOut    = createTensor({1, 1},        syn_type_uint8);
    pTensor meanAndStdOut           = createTensor({1, 1},        syn_type_uint8);
    pNode   stage2Node = NodeFactory::createNode({batchNormIfm, tensorRunningMeanIn, sigmaAndSigmaSquaredIn, meanIn},
                                                 {batchNormOfm, runningMeanAndVarOut, meanAndStdOut},
                                                 nullptr,
                                                 "batch_norm_stage2_fwd_bf16",
                                                 "stage2");

    ASSERT_TRUE(graph.addValidatedNode(stage2Node));
    ASSERT_TRUE(graph.addValidatedNode(reshape));
    ASSERT_TRUE(graph.addValidatedNode(fwd));

    pBundleExpansion candidate = std::make_shared<BundleExpansion>();
    candidate->nodeToStitch = stage2Node;
    candidate->reshapeNode = reshape;
    candidate->stitchedOperand = pSlicedOperand(new SlicedOperand(x));
    ASSERT_TRUE(reshapeAligner.alignProducerReshape(candidate));

    graphVisualizationPost(graph);

    int nodeIdx = 0;
    SizeArray expectedShape = {16, 16, 1 ,1, 1};
    for (auto& node : graph.getExeSortedNodes())
    {
        switch(nodeIdx++)
        {
            case 0:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_RESHAPE);
                break;
            case 1:
                ASSERT_EQ(std::dynamic_pointer_cast<TPCNode>(node)->getGUIDWithoutDtype(), "batch_norm_stage2_fwd");
                ASSERT_EQ(std::dynamic_pointer_cast<TPCNode>(node)->getInput(0)->getAllSizesInElements(), expectedShape);
                break;
            case 2:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_CONVOLUTION);
                break;
            case 3:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_RESHAPE);
                break;
        }
    }
}

TEST_F(SRAMManagementTest, testBundleReshapeEliminationDynamicShapes)
{
    setGlobalConfForTest(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT, "true");

    TestGraph      graph;
    ReshapeAligner reshapeAligner(graph);

    /* Creat the pattern:
     * --[t1]-->(AND)--[t2]-->(Reshape)--[t3]-->(MME)--[t4]-->*/

    /* Create the mme node */
    synConvolutionParams convParams {};
    pTensor              x = createTensor({16, 16}, syn_type_uint8, false, {16, 10});
    pTensor              w = createTensor({16, 16}, syn_type_uint8);
    pTensor              o = createTensor({16, 16}, syn_type_uint8);
    pNode fwd = NodeFactory::createNode({x, w}, {o}, &convParams, NodeFactory::convolutionNodeTypeName, "fwd");

    /* Create the reshape node*/
    pTensor andOfm       = createTensor({1, 2, 8, 16}, syn_type_uint8, false, {1, 2, 8, 10});
    pTensor addOfm_shape = createTensor({1, 2, 8, 16}, syn_type_uint8, false, {1, 2, 8, 10}, SHAPE_TENSOR);
    pNode   reshape =
        NodeFactory::createNode({andOfm, addOfm_shape}, {x}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    /* Create the TPC and node*/
    pTensor and_1   = createTensor({1, 2, 8, 16}, syn_type_uint8, true, {1, 2, 8, 10});
    pTensor and_2   = createTensor({1, 2, 8, 16}, syn_type_uint8, true, {1, 2, 8, 10});
    pNode   andNode = NodeFactory::createNode({and_1, and_2}, {andOfm}, nullptr, "bitwise_and_fwd_u8", "and");

    ASSERT_TRUE(graph.addValidatedNode(andNode));
    ASSERT_TRUE(graph.addValidatedNode(reshape));
    ASSERT_TRUE(graph.addValidatedNode(fwd));

    pBundleExpansion candidate = std::make_shared<BundleExpansion>();
    candidate->nodeToStitch    = andNode;
    candidate->reshapeNode     = reshape;
    candidate->stitchedOperand = pSlicedOperand(new SlicedOperand(x));
    ASSERT_TRUE(reshapeAligner.alignProducerReshape(candidate));

    graphVisualizationPost(graph);

    int       nodeIdx       = 0;
    SizeArray expectedShape = {16, 16, 1, 1, 1};
    for (auto& node : graph.getExeSortedNodes())
    {
        switch (nodeIdx++)
        {
            case 0:
            case 1:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_RESHAPE);
                break;
            case 2:
                ASSERT_EQ(std::dynamic_pointer_cast<TPCNode>(node)->getGUIDWithoutDtype(), "bitwise_and_fwd");
                ASSERT_EQ(std::dynamic_pointer_cast<TPCNode>(node)->getInput(0)->getAllSizesInElements(),
                          expectedShape);
                break;
            case 3:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_CONVOLUTION);
                break;
            case 4:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_RESHAPE);
                break;
        }
    }
}

TEST_F(SRAMManagementTest, testBundleReshapeEliminationBF16)
{
    setGlobalConfForTest(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT, "true");

    /* The motivation behind this test:
     * For the following bundle: TPC-->reshape-->MME
     * As part of bundle reshape elimination.
     * The reshape node is being eliminated from the bundle, then we insert reshape node before the TPC node.
     * The graph now look like this: reshape-->TPC-->MME
     * TPC node inputs and outputs should be adjusted to the new shape.
     * There are cases that we have more than a single 4D output.
     * BN2_BF16 has two outputs with 4D shape. Both of them should be handled.*/
    TestGraph      graph;
    ReshapeAligner reshapeAligner(graph);

    /* Creat the pattern:
     * --[t1]-->(BN2)--[t2]-->(Reshape)--[t3]-->(MME)--[t4]-->*/

    /* Create the mme node */
    synConvolutionParams convParams{};
    pTensor x = createTensor({16, 16}, syn_type_uint8);
    pTensor w = createTensor({16, 16}, syn_type_uint8);
    pTensor o = createTensor({16, 16}, syn_type_uint8);
    pNode fwd = NodeFactory::createNode({x, w}, {o}, &convParams, NodeFactory::convolutionNodeTypeName, "fwd");


    /* Create the reshape node*/
    /*                                   c  h  b  w */
    pTensor batchNormOfm    = createTensor({1, 2, 8, 16}, syn_type_uint8);
    pTensor batchNormOfmCpy = createTensor({1, 2, 8, 16}, syn_type_uint8);
    pNode   reshape         = NodeFactory::createNode({batchNormOfm},
                                                    {x},
                                                    nullptr,
                                                    NodeFactory::reshapeNodeTypeName,
                                                    "reshape");
    /* Create the TPC BN2 node*/
    /*                                              c  h  b  w */
    pTensor batchNormIfm            = createTensor({1, 2, 8, 16}, syn_type_uint8);
    pTensor tensorRunningMeanIn     = createTensor({1, 2},        syn_type_uint8);
    pTensor sigmaAndSigmaSquaredIn  = createTensor({1, 2},        syn_type_uint8);
    pTensor meanIn                  = createTensor({1, 1},        syn_type_uint8);
    pTensor runningMeanAndVarOut    = createTensor({1, 1},        syn_type_uint8);
    pTensor meanAndStdOut           = createTensor({1, 1},        syn_type_uint8);
    pNode   stage2Node = NodeFactory::createNode({batchNormIfm, tensorRunningMeanIn, sigmaAndSigmaSquaredIn, meanIn},
                                                 {batchNormOfm, batchNormOfmCpy, runningMeanAndVarOut, meanAndStdOut},
                                                 nullptr,
                                                 "batch_norm_stage2_fwd_bf16",
                                                 "stage2");

    ASSERT_TRUE(graph.addValidatedNode(stage2Node));
    ASSERT_TRUE(graph.addValidatedNode(reshape));
    ASSERT_TRUE(graph.addValidatedNode(fwd));


    pBundleExpansion candidate = std::make_shared<BundleExpansion>();
    candidate->nodeToStitch = stage2Node;
    candidate->reshapeNode = reshape;
    candidate->stitchedOperand = pSlicedOperand(new SlicedOperand(x));
    ASSERT_TRUE(reshapeAligner.alignProducerReshape(candidate));

    graphVisualizationPost(graph);
    SizeArray    expected = {16, 16, 1, 1, 1};
    int nodeIdx = 0;
    for (auto& node : graph.getExeSortedNodes())
    {
        switch(nodeIdx++)
        {
            case 0:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_RESHAPE);
                break;
            case 1:
                ASSERT_EQ(node->getOutputs()[1]->getAllSizesInElements(), expected);
                break;
            case 2:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_CONVOLUTION);
                break;
            case 3:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_RESHAPE);
                break;
            case 4:
                ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_RESHAPE);
                break;
        }
    }
}

TEST_F(SRAMManagementTest, reshape_aligner_should_reshape_same_size_output)
{
    setGlobalConfForTest(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT, "true");

    // Given (MME) -> [ofm] -> (Reshape) -> [tIn] -> (TPC) -> [tOut]
    // with tOut the same shape as tIn
    pTensor ifm = createTensor({128, 256}, syn_type_bf16);
    pTensor wgh = createTensor({64,  128}, syn_type_bf16);
    pTensor ofm = createTensor({64,  256}, syn_type_bf16);
    synConvolutionParams params{};
    pNode fwd = NodeFactory::createNode({ifm, wgh}, {ofm}, &params, NodeFactory::convolutionNodeTypeName, "fwd");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), fwd));

    pTensor tIn  = createTensor({128,  128}, syn_type_bf16);
    pTensor tOut = createTensor({128,  128}, syn_type_bf16);
    pNode   tpc  = NodeFactory::createNode({tIn}, {tOut}, nullptr, NOP_KERNEL_NAME, "nop");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), tpc));

    pNode reshape = NodeFactory::createNode({ofm}, {tIn}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reshape));

    pBundleExpansion candidate = std::make_shared<BundleExpansion>();
    candidate->stitchedOperand  = std::make_shared<SlicedOperand>(ofm);
    candidate->nodeToStitch     = tpc;
    candidate->reshapeNode      = reshape;

    // When
    ReshapeAligner(getGraph()).alignConsumerReshape(candidate);

    // Then new graph would be:
    // (MME) -> [ofm] ---> (TPC) ---> [reshaped_tOut] -> (Reshape) -> [tOut]
    //               \-> (Reshape) -> [tIn]
    const NodeVector& exeSched = getGraph().getExeSortedNodes();
    ASSERT_EQ(4, exeSched.size());
    bool foundReshapeOfmTIn = false;
    bool foundReshapeTOut = false;
    for (const pNode& node : exeSched)
    {
        switch (node->getNodeType())
        {
        case Node::TYPE_CONVOLUTION:
            break; // No change
        case Node::TYPE_USER:
            ASSERT_EQ(node->getInput(0), ofm);
            ASSERT_NE(node->getOutput(0), tOut);
            break;
        case Node::TYPE_INTERNAL_RESHAPE:
            if (node->getInput(0) == ofm)
            {
                // [ofm] -> (Reshape) -> [tIn] found?
                ASSERT_EQ(node->getOutput(0), tIn);
                foundReshapeOfmTIn = true;
            }
            else
            {
                // [reshaped_tOut] -> (Reshape) -> [tOut] found?
                ASSERT_EQ(node->getOutput(0), tOut);
                ASSERT_EQ(getGraph().getTensorProducer(node->getInput(0)), tpc);
                foundReshapeTOut = true;
            }
            break;
        default:
            FAIL() << "Unexpected node in graph after reshape elimination";
        }
    }
    ASSERT_TRUE(foundReshapeOfmTIn);
    ASSERT_TRUE(foundReshapeTOut);
}

TEST_F(SRAMManagementTest, reshape_aligner_should_not_reshape_different_size_output)
{
    setGlobalConfForTest(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT, "true");

    // Given (MME) -> [ofm] -> (Reshape) -> [tIn] -> (TPC) -> [tOut]
    // with tOut with a different shape than tIn
    pTensor ifm = createTensor({128, 256}, syn_type_bf16);
    pTensor wgh = createTensor({64,  128}, syn_type_bf16);
    pTensor ofm = createTensor({64,  256}, syn_type_bf16);
    synConvolutionParams params{};
    pNode fwd = NodeFactory::createNode({ifm, wgh}, {ofm}, &params, NodeFactory::convolutionNodeTypeName, "fwd");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), fwd));

    pTensor tIn  = createTensor({128, 128}, syn_type_bf16);
    pTensor tOut = createTensor({128, 1}, syn_type_bf16);    // Simulate a BN sigma/dotp output
    pNode   tpc  = NodeFactory::createNode({tIn}, {tOut}, nullptr, NOP_KERNEL_NAME, "nop");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), tpc));

    pNode reshape = NodeFactory::createNode({ofm}, {tIn}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reshape));

    pBundleExpansion candidate = std::make_shared<BundleExpansion>();
    candidate->stitchedOperand  = std::make_shared<SlicedOperand>(ofm);
    candidate->nodeToStitch     = tpc;
    candidate->reshapeNode      = reshape;

    // When
    ReshapeAligner(getGraph()).alignConsumerReshape(candidate);

    // Then new graph would be:
    // (MME) -> [ofm] ---> (TPC) ---> [tOut]
    //               \-> (Reshape) -> [tIn]
    const NodeVector& exeSched = getGraph().getExeSortedNodes();
    ASSERT_EQ(3, exeSched.size());
    for (const pNode& node : exeSched)
    {
        switch (node->getNodeType())
        {
        case Node::TYPE_CONVOLUTION:
            break; // No change
        case Node::TYPE_USER:
            ASSERT_EQ(node->getInput(0), ofm);
            ASSERT_EQ(node->getOutput(0), tOut);
            break;
        case Node::TYPE_INTERNAL_RESHAPE:
            ASSERT_EQ(node->getInput(0), ofm);
            ASSERT_EQ(node->getOutput(0), tIn);
            break;
        default:
            FAIL() << "Unexpected node in graph after reshape elimination";
        }
    }
}

TEST_F(SRAMManagementTest, reshape_aligner_should_reshape_same_size_input)
{
    setGlobalConfForTest(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT, "true");

    // Given (MME) -> [ofm] -> (Reshape) -> [tIn1] -> (TPC) -> [tOut]
    //                                      [tIn2] /
    // with tIn1 the same shape as tIn2
    pTensor ifm = createTensor({128, 256}, syn_type_bf16);
    pTensor wgh = createTensor({64, 128}, syn_type_bf16);
    pTensor ofm = createTensor({64, 256}, syn_type_bf16);
    synConvolutionParams params{};
    pNode fwd = NodeFactory::createNode({ifm, wgh}, {ofm}, &params, NodeFactory::convolutionNodeTypeName, "fwd");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), fwd));

    pTensor tIn1 = createTensor({128, 128}, syn_type_bf16);
    pTensor tIn2 = createTensor({128, 128}, syn_type_bf16);
    pTensor tOut = createTensor({128, 1}, syn_type_bf16);
    pNode   tpc  = NodeFactory::createNode({tIn1, tIn2}, {tOut}, nullptr, NOP_KERNEL_NAME, "nop");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), tpc));

    pNode reshape = NodeFactory::createNode({ofm}, {tIn1}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reshape));

    pBundleExpansion candidate = std::make_shared<BundleExpansion>();
    candidate->stitchedOperand  = std::make_shared<SlicedOperand>(ofm);
    candidate->nodeToStitch    = tpc;
    candidate->reshapeNode     = reshape;

    // When
    ReshapeAligner(getGraph()).alignConsumerReshape(candidate);

    // Then new graph would be:
    //
    // [tIn2] -> (Reshape) -> [reshaped_tIn2]------+
    //                                             |
    //           (MME) -> [ofm] ---------------> (TPC) -> [tOut]
    //             |
    //             +-> (Reshape) -> [tIn1]
    //
    const NodeVector& exeSched = getGraph().getExeSortedNodes();
    ASSERT_EQ(4, exeSched.size());
    bool foundReshapeOfmTIn1 = false;
    bool foundReshapeTIn2   = false;
    for (const pNode& node : exeSched)
    {
        switch (node->getNodeType())
        {
        case Node::TYPE_CONVOLUTION:
            break; // No change
        case Node::TYPE_USER:
            ASSERT_EQ(node->getInput(0), ofm);
            ASSERT_NE(node->getInput(1), tIn2);
            ASSERT_EQ(node->getOutput(0), tOut);
            break;
        case Node::TYPE_INTERNAL_RESHAPE:
            if (node->getInput(0) == ofm)
            {
                // [ofm] -> (Reshape) -> [tIn1] found?
                ASSERT_EQ(node->getOutput(0), tIn1);
                foundReshapeOfmTIn1 = true;
            }
            else
            {
                // [tIn2] -> (Reshape) -> [reshaped_tIn2] found?
                ASSERT_EQ(node->getInput(0), tIn2);
                NodeList reshapeOutputConsumers = getGraph().getTensorConsumers(node->getOutput(0));
                ASSERT_EQ(reshapeOutputConsumers.size(), 1);
                ASSERT_EQ(reshapeOutputConsumers.front(), tpc);
                foundReshapeTIn2 = true;
            }
            break;
        default:
            FAIL() << "Unexpected node in graph after reshape elimination";
        }
    }
    ASSERT_TRUE(foundReshapeOfmTIn1);
    ASSERT_TRUE(foundReshapeTIn2);
}

TEST_F(SRAMManagementTest, reshape_aligner_should_not_reshape_different_size_inputs)
{
    setGlobalConfForTest(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT, "true");

    // Given (MME) -> [ofm] -> (Reshape) -> [tIn1] -> (TPC) -> [tOut]
    //                                      [tIn2] /
    // with tIn1 with a different shape than tIn2
    pTensor ifm = createTensor({128, 256}, syn_type_bf16);
    pTensor wgh = createTensor({64,  128}, syn_type_bf16);
    pTensor ofm = createTensor({64,  256}, syn_type_bf16);
    synConvolutionParams params{};
    pNode fwd = NodeFactory::createNode({ifm, wgh}, {ofm}, &params, NodeFactory::convolutionNodeTypeName, "fwd");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), fwd));

    pTensor tIn1  = createTensor({128, 128}, syn_type_bf16);
    pTensor tIn2  = createTensor({128, 1}, syn_type_bf16);   // Simulate a BN sigma/dotp output
    pTensor tOut = createTensor({128, 1}, syn_type_bf16);    // Simulate a BN sigma/dotp output
    pNode   tpc   = NodeFactory::createNode({tIn1, tIn2}, {tOut}, nullptr, NOP_KERNEL_NAME, "nop");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), tpc));

    pNode reshape = NodeFactory::createNode({ofm}, {tIn1}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reshape));

    pBundleExpansion candidate = std::make_shared<BundleExpansion>();
    candidate->stitchedOperand  = std::make_shared<SlicedOperand>(ofm);
    candidate->nodeToStitch     = tpc;
    candidate->reshapeNode      = reshape;

    // When
    ReshapeAligner(getGraph()).alignConsumerReshape(candidate);

    // Then new graph would be:
    //             /---> (Reshape) -> [tIn1]
    // (MME) -> [ofm] ---> (TPC) ---> [tOut]
    //           [tIn2] /
    const NodeVector& exeSched = getGraph().getExeSortedNodes();
    ASSERT_EQ(3, exeSched.size());
    for (const pNode& node : exeSched)
    {
        switch (node->getNodeType())
        {
        case Node::TYPE_CONVOLUTION:
            break; // No change
        case Node::TYPE_USER:
            ASSERT_EQ(node->getInput(0), ofm);
            ASSERT_EQ(node->getInput(1), tIn2);
            ASSERT_EQ(node->getOutput(0), tOut);
            break;
        case Node::TYPE_INTERNAL_RESHAPE:
            ASSERT_EQ(node->getInput(0), ofm);
            ASSERT_EQ(node->getOutput(0), tIn1);
            break;
        default:
            FAIL() << "Unexpected node in graph after reshape elimination";
        }
    }
}

// conv -> reshape -> tpc
// tpc node is separable, but there was a case where this reshape wasn't aligned because of wrong separability check
// This test makes sure the same problem wouldn't happen again :)
TEST_F(SRAMManagementTest, reshape_should_be_aligned_on_separable_tpc)
{
    setGlobalConfForTest(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT, "true");
    setGlobalConfForTest(GCFG_IGNORE_INDEX_SPACE_FOR_SLICING, "true");

    const TSize b1 = 1, h1 = 1, w1 = 6272, c1 = 512;
    const TSize r = 1, s = 1, k = 256;

    pTensor convIn = createTensor({c1, w1, h1, b1}, syn_type_bf16);
    pTensor convWgh = createTensor({k, c1, s, r}, syn_type_bf16);
    pTensor convOut = createTensor({k, w1 - s + 1, h1 - s + 1, b1}, syn_type_bf16);

    synConvolutionParams convParams;
    pNode convNode = NodeFactory::createNode({convIn, convWgh}, {convOut}, &convParams,
                                             NodeFactory::convolutionNodeTypeName, "conv");

    const TSize b2 = 32, h2 = 14, w2 = 14, c2 = 256; // c2 is the batch norm "depth"

    pTensor tIFM = createTensor({c2, w2, h2, b2}, syn_type_bf16);
    pTensor tBeta = createTensor({c2}, syn_type_float);
    pTensor tGamma = createTensor({c2}, syn_type_float);
    pTensor tMean = createTensor({c2}, syn_type_float);
    pTensor tVariance = createTensor({c2}, syn_type_float);
    pTensor tOFM = createTensor({c2, w2, h2, b2}, syn_type_bf16);

    pNode reshapeNode = NodeFactory::createNode({convOut}, {tIFM}, nullptr, NodeFactory::reshapeNodeTypeName,
                                                "reshape");

    ns_BatchNormKernel::Params tpcParams;
    tpcParams.momentum = 0;
    tpcParams.threshold.f = 0;
    tpcParams.epsilon = 1e-05;
    pNode batchNormInfNode = NodeFactory::createGenericTPCNode({tIFM, tBeta, tGamma, tMean, tVariance}, {tOFM},
                                                               &tpcParams, "batch_norm_inf_bf16", "batch_norm_inf");

    GraphEditor::addNode(getGraph(), convNode);
    Bundlizer bundlizer(getGraph());
    BundleList bundles = bundlizer.getMMEBundles();
    ASSERT_EQ(bundles.size(), 1);
    pBundle bundle = bundles.front();
    pMmeSlicingStrategy slicingStrategy = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(getMmeBrain().getSolutionStrategies(bundle), bundle, getGraph(), getSlicingBrains()));
    GraphEditor::addNode(getGraph(), reshapeNode);
    GraphEditor::addNode(getGraph(), batchNormInfNode);

    // batchNormInfNode is added to expansion candidate because it is eligible for
    // stitching, specifically being separable, among other conditions (inside findTpcConsumerExpansionCandidate())
    pBundleExpansion expCnd = bundlizer.findTpcConsumerExpansionCandidate(slicingStrategy);
    ASSERT_EQ(expCnd->nodeToStitch, batchNormInfNode);
    ASSERT_EQ(expCnd->reshapeNode, reshapeNode);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, convOut);

    ReshapeAligner reshapeAligner(getGraph());
    ASSERT_TRUE(reshapeAligner.alignConsumerReshape(expCnd));
}

TEST_F(SRAMManagementTest, mme_inputs_should_be_located_in_sram)
{
    GaudiGraph g;

    // Fwd
    synConvolutionParams convParams{};
    pTensor x = createTensor({16, 16}, syn_type_bf16);
    pTensor w = createTensor({16, 16}, syn_type_bf16);
    pTensor o = createTensor({16, 16}, syn_type_bf16);
    pNode fwd = NodeFactory::createNode({x, w}, {o}, &convParams, NodeFactory::convolutionNodeTypeName, "fwd");
    ASSERT_TRUE(GraphEditor::addNode(g, fwd));

    // Gemm
    synGEMMParams gemmParams{};
    pTensor a = createTensor({16, 16}, syn_type_float);
    pTensor b = createTensor({16, 16}, syn_type_float);
    pTensor gemmOut = createTensor({16,16}, syn_type_float);
    pNode gemm = NodeFactory::createNode({a, b}, {gemmOut}, &gemmParams, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(GraphEditor::addNode(g, gemm));

    // DeDx
    pTensor dy = createTensor({16, 16}, syn_type_bf16);
    pTensor dx = createTensor({16, 16}, syn_type_bf16);
    pNode dedx = NodeFactory::createNode({dy, w}, {dx}, &convParams, NodeFactory::deDxNodeTypeName, "dedx");
    ASSERT_TRUE(GraphEditor::addNode(g, dedx));

    // DeDw
    pTensor dw = createTensor({16, 16}, syn_type_bf16);
    pNode dedw = NodeFactory::createNode({dy, x}, {dw}, &convParams, NodeFactory::deDwNodeTypeName, "dedw");
    ASSERT_TRUE(GraphEditor::addNode(g, dedw));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(g));
    for (const NodePtr& n : g.getExeSortedNodes())
    {
        if (g.runsOnMME(n))
        {
            for (pTensor input : n->getInputs())
            {
                if (input)
                {
                    ASSERT_TRUE(input->inSram());
                }
            }
        }
    }
}

TEST_F(SRAMManagementTest, DISABLED_at_least_one_startegy)
{
    /* TODO: [SW-15430] Enable this test */

    /* This case was taken from mask RCNN
     * We wish to make sure that we do find a strategy for this bundle
     * node name :Convolution6517*/
    TSize c = 1024, w = 64, h = 64, b = 1;
    TSize k = 2048, s = 1, r = 1;
    TSize wOut = 32, hOut = 32;

    std::vector<TSize> sizesA   = {c, w, h, b};
    std::vector<TSize> sizesB   = {k, c, s, r};
    std::vector<TSize> sizesOut = {k, wOut, hOut,b};
    std::shared_ptr<Bundle> bundle = createSingleMMENodeBundle(
            {createTensor(sizesA, syn_type_float), createTensor(sizesB, syn_type_float)},
            {createTensor(sizesOut, syn_type_float)},
            NodeFactory::convolutionNodeTypeName);

    auto strategies = getMmeBrain().getSolutionStrategies(bundle);
    ASSERT_NE(0, strategies.size()) << "The brain didn't return any strategy to solve the bundle";
}

TEST_F(SRAMManagementTest, bundlizer_should_bundle_mme_nodes)
{
    // Given Graph: TPC -> MME1 -> MME2 -> TPC -> MME3 -> TPC
    GaudiGraph g;

    TensorVector tensors;
    for (int i = 0; i < 7; i++)
    {
        tensors.push_back(std::make_shared<Tensor>(syn_type_bf16));
    }

    synConvolutionParams params;
    pNode                nop1 = NodeFactory::createNode({}, {tensors[0]}, nullptr, NOP_KERNEL_NAME, "nop1");
    pNode dedx = NodeFactory::createNode(
        {tensors[1], tensors[6]}, {tensors[2]}, &params, NodeFactory::deDxNodeTypeName, "dedx");
    pNode dedw = NodeFactory::createNode(
        {tensors[2], tensors[6]}, {tensors[3]}, &params, NodeFactory::deDwNodeTypeName, "dedw");
    pNode nop2 = NodeFactory::createNode({tensors[3]}, {tensors[4]}, nullptr, NOP_KERNEL_NAME, "nop2");
    pNode conv = NodeFactory::createNode(
        {tensors[4], tensors[6], nullptr, nullptr}, {tensors[5]}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    pNode nop3 = NodeFactory::createNode({tensors[5]}, {}, nullptr, NOP_KERNEL_NAME, "nop3");

    ASSERT_TRUE(GraphEditor::addNode(g, nop1));
    ASSERT_TRUE(GraphEditor::addNode(g, dedx));
    ASSERT_TRUE(GraphEditor::addNode(g, dedw));
    ASSERT_TRUE(GraphEditor::addNode(g, nop2));
    ASSERT_TRUE(GraphEditor::addNode(g, conv));
    ASSERT_TRUE(GraphEditor::addNode(g, nop3));

    // When bundling the graph
    BundleList bundles = Bundlizer(g).getMMEBundles();

    // Expected bundles: {MME1}, {MME2}, {MME3}
    ASSERT_EQ(3, bundles.size()) << "Wrong number of bundles";

    pNode expNodes[] = {dedx, dedw, conv};
    unsigned bundleIdx = 0;
    for (const pBundle& b : bundles)
    {
        if (bundleIdx < 1) continue;
        ASSERT_EQ(1, b->getNodes().size()) << "Wrong number of nodes in bundle";
        ASSERT_EQ(expNodes[bundleIdx], b->getNodes()[0]) << "Wrong node added to bundle";
        ++bundleIdx;
    }
}

TEST_F(SRAMManagementTest, scalar_pipe_bundlizer_should_bundle_tpc_node)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "0");
    TSize dataDims[2] = {2, 2};
    TSize idxDims[1] = {2};
    TSize validCountDims[1] = {2};

    pTensor inputData = pTensor(new Tensor(2, dataDims, syn_type_float));
    pTensor inputIndices = pTensor(new Tensor(1, idxDims, syn_type_int32));
    pTensor validCount = pTensor(new Tensor(1, validCountDims, syn_type_int32));
    pTensor output = pTensor(new Tensor(2, dataDims, syn_type_float));

    pNode gatherNode = NodeFactory::createNode({inputData, inputIndices, validCount},{output},
                                               nullptr, "gather_with_valid_count_2d_f32", "gatherNode");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), gatherNode));

    BundleList bundles = Bundlizer(getGraph()).getTPCScalarPipeBundles();
    ASSERT_EQ(1, bundles.size()) << "Wrong number of bundles";

    pBundle& bundle = bundles.front();
    ASSERT_EQ(1, bundle->getNodes().size()) << "Wrong number of nodes in bundle";
    ASSERT_EQ(gatherNode, bundle->getNodes()[0]) << "Wrong node added to bundle";
}

TEST_F(SRAMManagementTest, scalar_pipe_bundlizer_should_not_bundle_below_threshold)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "128");
    TSize dataDims[2]       = {2, 2};
    TSize idxDims[1]        = {2};
    TSize validCountDims[1] = {2};

    pTensor inputData    = pTensor(new Tensor(2, dataDims, syn_type_float));
    pTensor inputIndices = pTensor(new Tensor(1, idxDims, syn_type_int32));
    pTensor validCount   = pTensor(new Tensor(1, validCountDims, syn_type_int32));
    pTensor output       = pTensor(new Tensor(2, dataDims, syn_type_float));

    pNode gatherNode = NodeFactory::createNode({inputData, inputIndices, validCount},
                                               {output},
                                               nullptr,
                                               "gather_with_valid_count_2d_f32",
                                               "gatherNode");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), gatherNode));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
    for (const auto& n : getGraph().getExeSortedNodes())
    {
        ASSERT_FALSE(n->getNodeAnnotation().bundleInfo.is_set());
        for (pTensor input : n->getInputs())
        {
            if (input)
            {
                ASSERT_FALSE(input->inSram());
            }
        }
    }
}

TEST_F(SRAMManagementTest, scalar_pipe_tpc_brain_should_return_trivial_strategy_for_small_tensors)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "0");
    TSize dataDims[2]       = {2, 2};
    TSize idxDims[1]        = {2};
    TSize validCountDims[1] = {2};

    constexpr unsigned idxTensorsSize = 2;  // in elements
    constexpr unsigned vcTensorsSize  = 2;  // in elements

    pTensor inputData    = pTensor(new Tensor(2, dataDims, syn_type_float));
    pTensor inputIndices = pTensor(new Tensor(1, idxDims, syn_type_int32));
    pTensor validCount   = pTensor(new Tensor(1, validCountDims, syn_type_int32));
    pTensor output       = pTensor(new Tensor(2, dataDims, syn_type_float));

    pNode gatherNode = NodeFactory::createNode({inputData, inputIndices, validCount},
                                               {output},
                                               nullptr,
                                               "gather_with_valid_count_2d_f32",
                                               "gatherNode");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), gatherNode));

    BundleList          bundles = Bundlizer(getGraph()).getTPCScalarPipeBundles();
    SlicingStrategyList lst     = getTpcBrain().getSolutionStrategies(bundles.front());

    ASSERT_EQ(1, lst.size());
    SlicingStrategyPtr s = findWinningStrategy(lst, bundles.front(), getGraph(), getSlicingBrains());
    ASSERT_EQ(s->getMetrics().SRAMCapacity, (idxTensorsSize + vcTensorsSize) * 4)
        << "SRAM capacity should be equal to the size of the tensors in bytes.";
}

TEST_F(SRAMManagementTest, scalar_pipe_tpc_brain_should_set_trivial_solution_for_small_tensors)
{
    setGlobalConfForTest(GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT, "0");
    TSize dataDims[2] = {2, 2};
    TSize idxDims[1] = {2};
    TSize validCountDims[1] = {2};

    pTensor inputData = pTensor(new Tensor(2, dataDims, syn_type_float));
    pTensor inputIndices = pTensor(new Tensor(1, idxDims, syn_type_int32));
    pTensor validCount = pTensor(new Tensor(1, validCountDims, syn_type_int32));
    pTensor output = pTensor(new Tensor(2, dataDims, syn_type_float));

    pNode gatherNode = NodeFactory::createNode({inputData, inputIndices, validCount},{output},
                                               nullptr, "gather_with_valid_count_2d_f32", "gatherNode");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), gatherNode));

    BundleList bundles = Bundlizer(getGraph()).getTPCScalarPipeBundles();
    pBundle bundle = bundles.front();

    auto strategies = getTpcBrain().getSolutionStrategies(bundle);
    ASSERT_EQ(1, strategies.size()) << "brain did not return any strategy to solve the bundle";
    const auto& winning = findWinningStrategy(strategies, bundle, getGraph(), getSlicingBrains());
    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, winning).fillSolution());

    Solution& solution = bundle->getSolution();
    checkSolutionSize(solution, 4, 1);

    auto operation = solution.operations.front();
    ASSERT_FALSE(operation.inputs[0]->operand->resideInSRAM);
    ASSERT_TRUE(operation.inputs[1]->operand->resideInSRAM);
    ASSERT_TRUE(operation.inputs[2]->operand->resideInSRAM);
    for (unsigned coord : operation.inputs[0]->coordinates)
    {
        ASSERT_EQ(0, coord);
    }
    for (unsigned coord : operation.inputs[1]->coordinates)
    {
        ASSERT_EQ(0, coord);
    }
    for (unsigned coord : operation.inputs[2]->coordinates)
    {
        ASSERT_EQ(0, coord);
    }

    ASSERT_FALSE(operation.outputs[0]->operand->resideInSRAM);
    for (unsigned coord : operation.outputs[0]->coordinates)
    {
        ASSERT_EQ(0, coord);
    }
}

TEST_F(SRAMManagementTest, scalar_pipe_tpc_brain_should_return_no_strategy_for_big_tensors)
{
    TSize dataDims[2] = {1024, 1024};
    TSize idxDims[1] = {1024 * 1024 * 6};
    TSize validCountDims[1] = {1024};

    pTensor inputData = pTensor(new Tensor(2, dataDims, syn_type_float));
    pTensor inputIndices = pTensor(new Tensor(1, idxDims, syn_type_int32));
    pTensor validCount = pTensor(new Tensor(1, validCountDims, syn_type_int32));
    pTensor output = pTensor(new Tensor(2, dataDims, syn_type_float));

    pNode gatherNode = NodeFactory::createNode({inputData, inputIndices, validCount},{output},
                                               nullptr, "gather_with_valid_count_2d_f32", "gatherNode");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), gatherNode));

    BundleList bundles = Bundlizer(getGraph()).getTPCScalarPipeBundles();

    ASSERT_EQ(0, bundles.size());
}

TEST_F(SRAMManagementTest, brain_should_return_trivial_strategy_for_small_tensors)
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_MULTIPLE_SOLVERS_ENABLED, "false");

    //Given
    std::vector<TSize> sizes = {128, 128};
    constexpr TSize totalTensorSize = 128*128; // in elements
    std::shared_ptr<Bundle> bundle = createSingleMMENodeBundle(
        {createTensor(sizes, syn_type_bf16), createTensor(sizes,syn_type_bf16)},
        {createTensor(sizes, syn_type_bf16)},
        NodeFactory::convolutionNodeTypeName);

    // When
    SlicingStrategyList lst = getMmeBrain().getSolutionStrategies(bundle);

    // Then
    ASSERT_EQ(1, lst.size());
    SlicingStrategyPtr s = findWinningStrategy(lst, bundle, getGraph(), getSlicingBrains());
    ASSERT_EQ(s->getMetrics().SRAMCapacity, (totalTensorSize+totalTensorSize)*2) << "SRAM capacity should be equal to the size of the tensors in bytes.";
    ASSERT_EQ(s->getMetrics().SBReuse, 0) << "There should not be SB reuse in small tensors";
    ASSERT_EQ(s->getMetrics().HBMBandwidth, SlicingBrain::knobs.hbmAvailableBWGBps) << "BW should be maximal for small tensors";
}

TEST_F(SRAMManagementTest, brain_should_set_trivial_solution_for_small_tensors)
{
    //Given
    std::vector<TSize> sizes = {4, 4};
    std::shared_ptr<Bundle> bundle = createSingleMMENodeBundle(
        {createTensor(sizes, syn_type_bf16), createTensor(sizes,syn_type_bf16)},
        {createTensor(sizes, syn_type_bf16)},
        NodeFactory::convolutionNodeTypeName);

    // When
    solveBundleWithStrategy(bundle);

    //Then
    Solution& solution = bundle->getSolution();
    checkSolutionSize(solution, 3, 1);

    auto operation = solution.operations.front();
    ASSERT_TRUE(operation.inputs[0]->operand->resideInSRAM);
    ASSERT_TRUE(operation.inputs[1]->operand->resideInSRAM);
    for (unsigned coord : operation.inputs[0]->coordinates)
    {
        ASSERT_EQ(0, coord);
    }
    for (unsigned coord : operation.inputs[1]->coordinates)
    {
        ASSERT_EQ(0, coord);
    }

    ASSERT_FALSE(operation.outputs[0]->operand->resideInSRAM);
    for (unsigned coord : operation.outputs[0]->coordinates)
    {
        ASSERT_EQ(0, coord);
    }
}

TEST_F(SRAMManagementTest, brain_should_offer_trivial_solution_for_strided_and_filtered_convs)
{
    // Given
    pBundle                bundleFiltered(new Bundle(BundleType::MME));
    pBundle                bundleStrided(new Bundle(BundleType::MME));
    const MMESlicingBrain& brain = getMmeBrain();

    const TSize sizes[] = {16, 16};
    pTensor x(new Tensor(2, sizes, syn_type_bf16));
    pTensor w(new Tensor(2, sizes, syn_type_bf16));
    pTensor o1(new Tensor(2, sizes, syn_type_bf16));
    pTensor o2(new Tensor(2, sizes, syn_type_bf16));

    synConvolutionParams params;
    params.kH = 7;
    params.kW = 7;
    pNode filteredConv = NodeFactory::createNode(
        {x, w, nullptr, nullptr}, {o1}, &params, NodeFactory::convolutionNodeTypeName, "conv");

    params.kH = 1;
    params.kW = 1;
    params.dilH = 2;
    params.dilH = 2;
    pNode stridedConv = NodeFactory::createNode(
        {x, w, nullptr, nullptr}, {o2}, &params, NodeFactory::convolutionNodeTypeName, "conv");

    // When
    bundleFiltered->addNode(filteredConv);
    GraphEditor::addNode(getGraph(), filteredConv);
    bundleStrided->addNode(stridedConv);
    GraphEditor::addNode(getGraph(), stridedConv);

    // Then
    SlicingStrategyList filteredStrategies = brain.getSolutionStrategies(bundleFiltered);
    ASSERT_TRUE(filteredStrategies.size());
    ASSERT_TRUE(
        SolutionGenerator(getGraph(),
                          bundleFiltered,
                          findWinningStrategy(filteredStrategies, bundleFiltered, getGraph(), getSlicingBrains()))
            .fillSolution());
    SlicingStrategyList stridedStrategies = brain.getSolutionStrategies(bundleStrided);
    ASSERT_TRUE(stridedStrategies.size());
    ASSERT_TRUE(SolutionGenerator(getGraph(),
                                  bundleStrided,
                                  findWinningStrategy(stridedStrategies, bundleStrided, getGraph(), getSlicingBrains()))
                    .fillSolution());

    for (pBundle bundle : std::vector<pBundle>{bundleFiltered, bundleStrided})
    {

        Solution& solution = bundle->getSolution();
        ASSERT_EQ(3, solution.operands.size()) << "Wrong number of operands in trivial solution";
        ASSERT_EQ(1, solution.operations.size()) << "Wrong number of operations in trivial solution";

        auto operation = solution.operations.front();
        ASSERT_TRUE(operation.inputs[0]->operand->resideInSRAM);
        ASSERT_TRUE(operation.inputs[1]->operand->resideInSRAM);
        for (unsigned coord : operation.inputs[0]->coordinates)
        {
            ASSERT_EQ(0, coord);
        }
        for (unsigned coord : operation.inputs[1]->coordinates)
        {
            ASSERT_EQ(0, coord);
        }

        ASSERT_FALSE(operation.outputs[0]->operand->resideInSRAM);
        for (unsigned coord : operation.outputs[0]->coordinates)
        {
            ASSERT_EQ(0, coord);
        }
    }
}

TEST_F(SRAMManagementTest, brain_should_slice_bhw_when_ifm_is_big)
{
    setGlobalConfForTest(GCFG_ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING, "true");

    // IFM size (64 * 5.5)x2 bfloats or (32*5.5)x2 floats,
    // WGH size 2x4 bfloats\floats,
    // SRAM can fit the WGH and 1 slice of 64\32 lines out of the IFM
    // Expects WGH to stay in SRAM while 64x1 (bfloat) or 32x1 (float) chunks from IFM are brought to a single buffer for compute

    // Given
    for (synDataType dtype : {syn_type_bf16, syn_type_float})
    {
        const TSize expSlices  = 6;
        TSize sliceSize = 64;
        float w = expSlices-0.5f;
        if (dtype == syn_type_float)
        {
            sliceSize /= 2;
        }
        TSize ifmsizes[] = {2, w * sliceSize, 1, 1};
        const TSize wsizes[]   = {4, 2};
        const TSize ofmsizes[] = {4, w * sliceSize, 1, 1};

        pTensor  ifm(new Tensor(4, ifmsizes, dtype));
        pTensor  wgh(new Tensor(2, wsizes, dtype));
        pTensor  ofm(new Tensor(4, ofmsizes, dtype));


        const uint64_t ifmLineSizeInBytes = 2 * ifm->getElementSizeInBytes();
        const uint64_t weightSize         = wgh->getDenseSizeInBytes();

        SlicingBrain::knobs.maxSRAMCapInBytes =
                weightSize + sliceSize * ifmLineSizeInBytes;

        // When
        pBundle bundle = createSingleMMENodeBundle({ifm, wgh}, {ofm}, NodeFactory::convolutionNodeTypeName);

        // Then
        solveBundleWithStrategy(bundle);
        Solution& solution = bundle->getSolution();
        checkSolutionSize(solution, 3, expSlices);
        checkChunkSize(solution, ifm, {ifm->getSizeInElements(0), sliceSize, 1, 1});
        checkChunkSize(solution, wgh, wgh->getAllSizesInElements());
        ExecutionOrderChecker(DIM_W, WEIGHT_DIM_K).checkWalkDownExecutionOrder(solution, expSlices, 1);
    }
}

TEST_F(SRAMManagementTest, brain_should_slice_k_when_wgh_is_big)
{
    // IFM size 12x32 bfloats\floats,
    // WGH size 32x(64 * 5.5) bfloats or 32x(32*5.5) floats,
    // SRAM can fit the IFM and 1 slice of 64 (bfloat) or 32 (float) columns out of the WGH
    // Expects IFM to stay in SRAM while 32x64 (bfloat) or 32x32 (float) chunks from IFM are brought to a single buffer for compute

    // Given
    for (synDataType dtype : {syn_type_bf16, syn_type_float})
    {
        const TSize expSlices  = 6;
        TSize sliceSize = 64;
        const TSize ifmsizes[] = {32, 12, 1, 1};
        float w = expSlices - 0.5f;
        if (dtype == syn_type_float)
        {
            sliceSize /= 2;
        }
        TSize wsizes[] = {sliceSize * w, 32};
        const TSize ofmsizes[] = {sliceSize * w,  12, 1, 1};

        pTensor ifm(new Tensor(4, ifmsizes, dtype));
        pTensor wgh(new Tensor(2, wsizes, dtype));
        pTensor ofm(new Tensor(4, ofmsizes, dtype));

        const uint64_t wghLineSizeInBytes = 32 * wgh->getElementSizeInBytes();
        const uint64_t ifmSize            = ifm->getDenseSizeInBytes();

        SlicingBrain::knobs.maxSRAMCapInBytes =
                ifmSize + sliceSize * wghLineSizeInBytes;

        // When
        pBundle bundle = createSingleMMENodeBundle({ifm, wgh}, {ofm}, NodeFactory::convolutionNodeTypeName);

        // Then
        solveBundleWithStrategy(bundle);
        Solution& solution = bundle->getSolution();
        checkSolutionSize(solution, 3, expSlices);
        checkChunkSize(solution, ifm, ifm->getAllSizesInElements());
        checkChunkSize(solution, wgh, {sliceSize, wgh->getSizeInElements(1)});
        ExecutionOrderChecker(DIM_W, WEIGHT_DIM_K).checkWalkRightExecutionOrder(solution, 1, expSlices);
    }
}

TEST_F(SRAMManagementTest, brain_should_slice_c_when_wgh_is_big_and_node_is_dedx)
{
    // IFM size 12x32 bfloats\floats,
    // WGH size 32x(64 * 5.5) bfloats or 32x(32 * 5.5) floats,
    // SRAM can fit the IFM and 1 slice of 64\32 columns out of the WGH
    // Expects IFM to stay in SRAM while 32x64 (bfloat) or 32x32 (float) chunks from IFM are brought to a single buffer for compute

    // Given
    for (auto& dtype : {syn_type_bf16, syn_type_float})
    {
        const TSize expSlices  = 6;
        TSize sliceSize = 64;
        float w = expSlices - 0.5f;
        if (dtype == syn_type_float)
        {
            sliceSize = sliceSize / 2;
        }
        const TSize ifmsizes[] = {32, 12, 1, 1};
        const TSize wsizes[]   = {32, sliceSize * w};
        const TSize ofmsizes[] = {sliceSize * w, 12, 1, 1};
        pTensor dedy(new Tensor(4, ifmsizes, dtype));
        pTensor wgh(new Tensor(2, wsizes, dtype));
        pTensor dedx(new Tensor(4, ofmsizes, dtype));

        const uint64_t wghLineSizeInBytes = 32 * wgh->getElementSizeInBytes();
        const uint64_t ifmSize            = dedy->getDenseSizeInBytes();

        SlicingBrain::knobs.maxSRAMCapInBytes = ifmSize + sliceSize * wghLineSizeInBytes;

        // When
        pBundle bundle = createSingleMMENodeBundle({dedy, wgh}, {dedx}, NodeFactory::deDxNodeTypeName);
        // Then
        solveBundleWithStrategy(bundle);
        Solution& solution = bundle->getSolution();
        checkSolutionSize(solution, 3, expSlices);
        checkChunkSize(solution, dedy, dedy->getAllSizesInElements());
        checkChunkSize(solution, wgh, {wgh->getSizeInElements(0), sliceSize});
        ExecutionOrderChecker(DIM_W, WEIGHT_DIM_C).checkWalkRightExecutionOrder(solution, 1, expSlices);
    }
}

TEST_F(SRAMManagementTest, brain_should_select_narrow_ifm_slice_size_according_to_sram_cap)
{
    // IFM size (4 * 512)x8 bfloats,
    // WGH size 8x4 bfloats,
    // Expects WGH to stay in SRAM while 512 chunks from IFM are brought to a double buffer for compute

    // Given
    const TSize bhwSize    = 4 * 512;
    TSize ifmsizes[] = {8, bhwSize, 1, 1};
    const TSize wsizes[]   = {4, 8};
    const TSize ofmsizes[] = {4, bhwSize, 1, 1};
    for (auto& dtype : {syn_type_bf16, syn_type_float})
    {
        pTensor ifm(new Tensor(4, ifmsizes, dtype));
        pTensor wgh(new Tensor(2, wsizes, dtype));
        pTensor ofm(new Tensor(4, ofmsizes, dtype));

        const uint64_t ifmLineSizeInBytes = 8 * ifm->getElementSizeInBytes();
        const uint64_t weightSize         = wgh->getDenseSizeInBytes();
        pBundle        bundle             = createSingleMMENodeBundle({ifm, wgh},
                                                                      {ofm},
                                                                      NodeFactory::convolutionNodeTypeName);

        // Chunk size should not be smaller than MME geomentry
        const auto mmeMaxGeometryElements =
            4 * getGraph().getHALReader()->getMmeVectorSize() / dataTypeSizeInBytes(dtype);
        for (unsigned expSlices = bhwSize / 512; expSlices <= bhwSize / mmeMaxGeometryElements; expSlices *= 2)
        {
            // When
            unsigned expChunkSize = bhwSize / expSlices;
            SlicingBrain::knobs.maxSRAMCapInBytes = 2 * (weightSize + expChunkSize * ifmLineSizeInBytes);

            // clear prev solution.
            bundle->getSolution().operations.clear();
            bundle->getSolution().operands.clear();
            // Then
            solveBundleWithStrategy(bundle);
            Solution& solution = bundle->getSolution();
            checkSolutionSize(solution, 3, expSlices);
            SizeArray expChunkDimArray = ifm->getAllSizesInElements();
            expChunkDimArray[1] = expChunkSize;
            checkChunkSize(solution, ifm, expChunkDimArray);
        }
    }
}

// Same position shared mme input, ifm (x) is the shared operand. Ifm (x) is sliced on a spatial dimension, so the slave
// non-shared operand or output slicing depends on the shared operand slicing. The only valid solver is NonCD4dSolver,
// which can't be expanded with shared input.
TEST_F(SRAMManagementTest, same_position_shared_mme_input_ifm_sliced_on_spatial_dim)
{
    const TSize cd = 16;
    const TSize w = 512, h = 512, b = 1;
    const TSize k = 30, s = 3, r = 3;
    const TSize ifmSizes[] = {cd, w, h, b};
    const TSize wSizes[]   = {k, cd, s, r};

    synConvolutionParams params;
    params.kW = s;
    params.kH = r;
    params.dH = r;
    params.dW = s;

    TSize       ofmH       = convOutputDimSize(h, params.kH, params.dH, params.padT + params.padB, params.dilH);
    TSize       ofmW       = convOutputDimSize(w, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const TSize ofmSizes[] = {k, ofmW, ofmH, b};

    pTensor ifm(new Tensor(4, ifmSizes, syn_type_bf16));
    pTensor wgh(new Tensor(4, wSizes, syn_type_bf16));
    pTensor wgh2(new Tensor(4, wSizes, syn_type_bf16));
    pTensor ofm(new Tensor(4, ofmSizes, syn_type_bf16));
    pTensor ofm2(new Tensor(4, ofmSizes, syn_type_bf16));

    const unsigned ifmSizeInBytes = ifm->getDenseSizeInBytes();

    // Set room for 1/10 of ifm tensor - so trivial solver won't find a strategy
    SlicingBrain::knobs.maxSRAMCapInBytes = ifmSizeInBytes / 10;
    setGlobalConfForTest(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES, std::to_string(SlicingBrain::knobs.maxSRAMCapInBytes));
    pNode conv1 = NodeFactory::createNode({ifm, wgh}, {ofm}, &params, NodeFactory::convolutionNodeTypeName, "conv1");
    GraphEditor::addNode(getGraph(), conv1);

    pNode conv2 = NodeFactory::createNode({ifm, wgh2}, {ofm2}, &params, NodeFactory::convolutionNodeTypeName, "conv2");
    GraphEditor::addNode(getGraph(), conv2);

    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
    // Make sure each node is bundled
    ASSERT_TRUE(conv1->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(conv2->getNodeAnnotation().bundleInfo.is_set());
    // Make sure the bundles are separate - no shared input expansion was done.
    ASSERT_NE(conv1->getNodeAnnotation().bundleInfo->bundleIndex, conv2->getNodeAnnotation().bundleInfo->bundleIndex);
}

// Same position shared mme input, w is the shared operand. When non-shared operand (x/y) slicing dimension is spatial,
// we keep non-shared operand trivially sliced.
TEST_F(SRAMManagementTest, same_position_shared_mme_input_wgh)
{
    const TSize cd = 4;
    const TSize w = 1280, h = 1280, b = 1;
    const TSize k = 30, s = 3, r = 3;
    const TSize ifmSizes[]      = {cd, w, h, b};
    const TSize slaveIfmSizes[] = {cd, w / 20, h / 20, b};
    const TSize wSizes[]        = {k, cd, s, r};

    synConvolutionParams params;
    params.kW = s;
    params.kH = r;
    params.dH = r;
    params.dW = s;

    unsigned    ofmH       = convOutputDimSize(h, params.kH, params.dH, params.padT + params.padB, params.dilH);
    unsigned    ofmW       = convOutputDimSize(w, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const TSize ofmSizes[] = {k, ofmW, ofmH, b};
    unsigned    slaveOfmH  = convOutputDimSize(h / 20, params.kH, params.dH, params.padT + params.padB, params.dilH);
    unsigned    slaveOfmW  = convOutputDimSize(w / 20, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const TSize slaveOfmSizes[] = {k, slaveOfmW, slaveOfmH, b};

    pTensor ifm(new Tensor(4, ifmSizes, syn_type_bf16));
    pTensor slaveIfm(new Tensor(4, slaveIfmSizes, syn_type_bf16));
    pTensor wgh(new Tensor(4, wSizes, syn_type_bf16));
    pTensor ofm(new Tensor(4, ofmSizes, syn_type_bf16));
    pTensor slaveOfm(new Tensor(4, slaveOfmSizes, syn_type_bf16));

    const unsigned ifmSizeInBytes      = ifm->getDenseSizeInBytes();
    const unsigned slaveIfmSizeInBytes = slaveIfm->getDenseSizeInBytes();
    // Set room for 1/10 of master ifm tensor with some room for the entire slave Ifm
    SlicingBrain::knobs.maxSRAMCapInBytes = ifmSizeInBytes / 10 + slaveIfmSizeInBytes;
    setGlobalConfForTest(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES, std::to_string(SlicingBrain::knobs.maxSRAMCapInBytes));

    pNode slaveMme =
        NodeFactory::createNode({slaveIfm, wgh}, {slaveOfm}, &params, NodeFactory::convolutionNodeTypeName, "conv1");
    GraphEditor::addNode(getGraph(), slaveMme);

    pNode masterMme =
        NodeFactory::createNode({ifm, wgh}, {ofm}, &params, NodeFactory::convolutionNodeTypeName, "conv2");
    GraphEditor::addNode(getGraph(), masterMme);

    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
    // Make sure each node is bundled
    ASSERT_TRUE(slaveMme->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(masterMme->getNodeAnnotation().bundleInfo.is_set());
    // Make sure the nodes are in the same bundle- since there is available sram cap for the slave non shared operand
    ASSERT_EQ(slaveMme->getNodeAnnotation().bundleInfo->bundleIndex,
              masterMme->getNodeAnnotation().bundleInfo->bundleIndex);
}

// Same position shared mme input, w is the shared operand. When the slave non-shared operand (x/y) slicing dimension is
// not a spatial dim, we can slice it however we want (bundle expansion will work)
TEST_F(SRAMManagementTest, same_position_shared_mme_input_wgh_ifm_sliced_on_b)
{
    const TSize cd = 512;
    const TSize w = 64, h = 64, b = 4;
    const TSize wSlave = w / 4;
    const TSize hSlave = h / 4;
    const TSize k = 30, s = 3, r = 3;
    const TSize ifmSizes[]      = {cd, w, h, b};
    const TSize slaveIfmSizes[] = {cd, wSlave, hSlave, b};
    const TSize wSizes[]        = {k, cd, s, r};

    synConvolutionParams params;
    params.kW = s;
    params.kH = r;
    params.dW = s;
    params.dH = r;

    unsigned    ofmH       = convOutputDimSize(h, params.kH, params.dH, params.padT + params.padB, params.dilH);
    unsigned    ofmW       = convOutputDimSize(w, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const TSize ofmSizes[] = {k, ofmW, ofmH, b};
    unsigned    slaveOfmH  = convOutputDimSize(hSlave, params.kH, params.dH, params.padT + params.padB, params.dilH);
    unsigned    slaveOfmW  = convOutputDimSize(wSlave, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const TSize slaveOfmSizes[] = {k, slaveOfmW, slaveOfmH, b};

    pTensor ifm(new Tensor(4, ifmSizes, syn_type_bf16));
    pTensor slaveIfm(new Tensor(4, slaveIfmSizes, syn_type_bf16));
    pTensor wgh(new Tensor(4, wSizes, syn_type_bf16));
    pTensor ofm(new Tensor(4, ofmSizes, syn_type_bf16));
    pTensor slaveOfm(new Tensor(4, slaveOfmSizes, syn_type_bf16));

    const unsigned ifmSizeInBytes = ifm->getDenseSizeInBytes();
    const unsigned slaveIfmSizeInBytes = slaveIfm->getDenseSizeInBytes();

    // Let 1 batch fit, and leave room for double buffer
    const unsigned sramForMasterIfm = 2 * ifmSizeInBytes / b;
    // Let the slave fit without slicing, and leave room for double buffer (doSamePositionStitching doesn't fine tune
    // enough the sram capacity checking for trivially sliced non-shared operand)
    const unsigned sramForSlaveIfm = 2 * slaveIfmSizeInBytes;
    // Not sliced as it's too narrow - no need for double buffer
    const unsigned sramForWgh = wgh->getDenseSizeInBytes();

    SlicingBrain::knobs.maxSRAMCapInBytes = sramForMasterIfm + sramForSlaveIfm + sramForWgh;
    setGlobalConfForTest(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES, std::to_string(SlicingBrain::knobs.maxSRAMCapInBytes));

    pNode slaveMme =
        NodeFactory::createNode({slaveIfm, wgh}, {slaveOfm}, &params, NodeFactory::convolutionNodeTypeName, "conv1");
    GraphEditor::addNode(getGraph(), slaveMme);

    pNode masterMme =
        NodeFactory::createNode({ifm, wgh}, {ofm}, &params, NodeFactory::convolutionNodeTypeName, "conv2");
    GraphEditor::addNode(getGraph(), masterMme);

    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
    ASSERT_TRUE(slaveMme->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(masterMme->getNodeAnnotation().bundleInfo.is_set());
    // Make sure the nodes are in the same bundle
    ASSERT_EQ(slaveMme->getNodeAnnotation().bundleInfo->bundleIndex,
              masterMme->getNodeAnnotation().bundleInfo->bundleIndex);
}

TEST_F(SRAMManagementTest, brain_should_select_narrow_wgh_slice_size_according_to_sram_cap)
{
    // IFM size 8x64 bfloats,
    // WGH size (16 * 256)x8 bfloats,
    // Expects IFM to stay in SRAM while n*256 chunks from WGH are brought to a double buffer for compute

    // Given
    const TSize kSize    = 16 * 256;
    const TSize ifmsizes[] = {8, 64, 1, 1};
    const TSize wsizes[]   = {kSize, 8};
    const TSize ofmsizes[] = {kSize, 64, 1, 1};

    pTensor ifm(new Tensor(4, ifmsizes, syn_type_bf16));
    pTensor wgh(new Tensor(2, wsizes, syn_type_bf16));
    pTensor ofm(new Tensor(4, ofmsizes, syn_type_bf16));

    const uint64_t wghLineSizeInBytes = 8 * wgh->getElementSizeInBytes();
    const uint64_t ifmSize            = ifm->getDenseSizeInBytes();

    pBundle bundle = createSingleMMENodeBundle({ifm, wgh}, {ofm}, NodeFactory::convolutionNodeTypeName);

    for (unsigned expSlices = kSize / (256*4); expSlices <= kSize / 256; expSlices *= 2)
    {
        // When
        unsigned expChunkSize = kSize / expSlices;
        SlicingBrain::knobs.maxSRAMCapInBytes = 2 * (ifmSize + expChunkSize * wghLineSizeInBytes);
        // clear prev solution
        bundle->getSolution().operands.clear();
        bundle->getSolution().operations.clear();

        // SRAM can fit the WGH and 2 slices of expChunkSize lines out of the IFM

        // Then
        solveBundleWithStrategy(bundle);
        Solution& solution = bundle->getSolution();
        checkSolutionSize(solution, 3, expSlices);
        SizeArray expChunkDimArray = wgh->getAllSizesInElements();
        expChunkDimArray[0] = expChunkSize;
        checkChunkSize(solution, wgh, expChunkDimArray);
    }
}

TEST_F(SRAMManagementTest, brain_should_supply_single_buffer_solution_if_no_other_is_available)
{
    // IFM size (5 * 64)x1024 bfloats,
    // WGH size 1024x32 bfloats,
    // SRAM can fit WGH and a single 64x1024 slice of IFM
    // Expects WGH to stay in SRAM while 64x1024 slices of IFM are SINGLE BUFFEREDly brought to SRAM

    // Given
    const TSize cd         = 1024;
    const TSize k          = 32;
    const TSize bhw        = 5 * 64;
    const TSize ifmsizes[] = {cd, bhw, 1, 1};
    const TSize wsizes[]   = {k, cd};
    const TSize ofmsizes[] = {k, bhw, 1, 1};

    pTensor ifm(new Tensor(4, ifmsizes, syn_type_bf16));
    pTensor wgh(new Tensor(2, wsizes, syn_type_bf16));
    pTensor ofm(new Tensor(4, ofmsizes, syn_type_bf16));

    SlicingBrain::knobs.maxSRAMCapInBytes = (k * cd + 64 * cd) * ifm->getElementSizeInBytes();

    // When
    pBundle bundle = createSingleMMENodeBundle({ifm, wgh}, {ofm}, NodeFactory::convolutionNodeTypeName);

    // Then
    solveBundleWithStrategy(bundle);
    Solution& solution = bundle->getSolution();
    checkSolutionSize(solution, 3, 5);
    checkChunkSize(solution, ifm, {ifm->getSizeInElements(0), 64, 1, 1});
}

TEST_F(SRAMManagementTest, brain_should_slice_both_operands_when_neither_fit_in_sram)
{
    setGlobalConfForTest(GCFG_ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING, "true");

    // SRAM size is sizeof(bfloat16) * CD * 2 * (256 + 64)
    // IFM is 19 chunks of 256xCD + 0.5 chunk
    // WGH is 10 chunks of 64xCD + 0.5 chunk
    // Expect left to right walking direction solution with the above chunk sizes

    // Given
    const TSize cd              = 16;
    const TSize kChunks         = 11;
    const TSize expWGHChunkSize = 64;
    const TSize k               = kChunks * expWGHChunkSize - (expWGHChunkSize / 2);
    const TSize ifmChunks       = 20;
    const TSize expBHWChunkSize = 256;
    const TSize bhw             = ifmChunks * expBHWChunkSize - (expBHWChunkSize / 2);
    const TSize ifmsizes[]      = {cd, bhw, 1, 1};
    const TSize wsizes[]        = {k, cd};
    const TSize ofmsizes[]      = {k, bhw, 1, 1};

    pTensor ifm(new Tensor(4, ifmsizes, syn_type_bf16));
    pTensor wgh(new Tensor(2, wsizes, syn_type_bf16));
    pTensor ofm(new Tensor(4, ofmsizes, syn_type_bf16));

    SlicingBrain::knobs.maxSRAMCapInBytes =
        ifm->getElementSizeInBytes() * 2 * cd * (expWGHChunkSize + expBHWChunkSize);

    // When
    pBundle bundle = createSingleMMENodeBundle({ifm, wgh}, {ofm}, NodeFactory::convolutionNodeTypeName);

    solveBundleWithStrategy(bundle);
    Solution& solution = bundle->getSolution();
    checkSolutionSize(solution, 3, kChunks*ifmChunks);
    checkChunkSize(solution, ofm, {expWGHChunkSize, expBHWChunkSize, 1, 1});
    ExecutionOrderChecker(DIM_W, WEIGHT_DIM_K).checkWalkRightExecutionOrder(solution, ifmChunks, kChunks);
}

TEST_F(SRAMManagementTest, brain_should_select_single_buffer_when_no_double_buffer_solution_is_available)
{
    setGlobalConfForTest(GCFG_ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING, "true");

    // SRAM size is sizeof(bfloat16) * CD * <capacity for 1 or 2 ifm chunks + capacity for 1 or 2 wgh chunks>
    // Expect the single/double buffer selection to enable a solution no matter what.

    // Given
    const TSize cd              = 16;
    const TSize kChunks         = 41;
    const TSize expWGHChunkSize = 64;
    const TSize k               = kChunks * expWGHChunkSize - (expWGHChunkSize / 4);
    const TSize ifmChunks       = 18;
    const TSize expBHWChunkSize = 256;
    const TSize bhw             = ifmChunks * expBHWChunkSize - (expBHWChunkSize / 4);
    const TSize ifmsizes[]      = {cd, bhw, 1, 1};
    const TSize wsizes[]        = {k, cd};
    const TSize ofmsizes[]      = {k, bhw, 1, 1};

    pTensor ifm(new Tensor(4, ifmsizes, syn_type_bf16));
    pTensor wgh(new Tensor(2, wsizes, syn_type_bf16));
    pTensor ofm(new Tensor(4, ofmsizes, syn_type_bf16));

    pBundle bundle = createSingleMMENodeBundle({ifm, wgh}, {ofm}, NodeFactory::convolutionNodeTypeName);

    // When
    for (int doubleBuffered = 1; doubleBuffered >= 0; doubleBuffered--)
    {
        SlicingBrain::knobs.maxSRAMCapInBytes =
                ifm->getElementSizeInBytes() * cd *
                ((1 + doubleBuffered) * (expWGHChunkSize + expBHWChunkSize));
        // clear prev solution
        bundle->getSolution().operations.clear();
        bundle->getSolution().operands.clear();

        solveBundleWithStrategy(bundle);
        Solution& solution = bundle->getSolution();
        checkSolutionSize(solution, 3, kChunks * ifmChunks);
        checkChunkSize(solution, ifm, {cd, expBHWChunkSize, 1, 1});
        checkChunkSize(solution, wgh, {expWGHChunkSize, cd, 1, 1});
    }
}

TEST_F(SRAMManagementTest, brain_should_slice_greater_than_1x1_conv_on_batch_dimension)
{
    // SRAM Capacity can fit 1 activation batches and the complete weight tensor.
    // Expects slicing the ifm to a double batch chunks (B = 1)

    const TSize cd = 16;
    const TSize w = 256, h = 256, b = 32;
    const TSize k = 30, s = 3, r = 3;
    const TSize ifmsizes[] = {cd, w, h, b};
    const TSize wsizes[]   = {k, cd, s, r};
    const TSize ofmsizes[] = {k, w - s + 1, h - r + 1, b};

    pTensor ifm(new Tensor(4, ifmsizes, syn_type_bf16));
    pTensor wgh(new Tensor(4, wsizes, syn_type_bf16));
    pTensor ofm(new Tensor(4, ofmsizes, syn_type_bf16));

    synConvolutionParams params;
    params.kW  = s;
    params.kH  = r;

    pBundle bundle = createSingleMMENodeBundle({ifm, wgh, nullptr, nullptr}, {ofm}, NodeFactory::convolutionNodeTypeName);

    unsigned doubleBufferFactor = 2;
    SlicingBrain::knobs.maxSRAMCapInBytes = doubleBufferFactor * ifm->getElementSizeInBytes() * (1 * h * w * cd + k * cd * s * r);
    SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon4D = h*w;

    // Then
    solveBundleWithStrategy(bundle);
    Solution& solution = bundle->getSolution();

    checkSolutionSize(solution, 3, b);

    pSlicedOperand slice = solution.operands[0];
    ASSERT_EQ(slice->chunkDimensions[0], 16);
    ASSERT_EQ(slice->chunkDimensions[1], 256);
    ASSERT_EQ(slice->chunkDimensions[2], 256);
    ASSERT_EQ(slice->chunkDimensions[3], 1);

    slice = solution.operands[1];
    ASSERT_EQ(slice->chunkDimensions[0], 30);
    ASSERT_EQ(slice->chunkDimensions[1], 16);
    ASSERT_EQ(slice->chunkDimensions[2], 3);
    ASSERT_EQ(slice->chunkDimensions[3], 3);

    slice = solution.operands[2];
    ASSERT_EQ(slice->chunkDimensions[0], 30);
    ASSERT_EQ(slice->chunkDimensions[1], 254);
    ASSERT_EQ(slice->chunkDimensions[2], 254);
    ASSERT_EQ(slice->chunkDimensions[3], 1);

}

TEST_F(SRAMManagementTest, brain_should_slice_greater_than_1x1_conv_on_b_and_k_dimensions)
{
    // SRAM Capacity can fit 2 activation batches and the 64 weight chunks.
    // Expects slicing the ifm to a double batch chunks (B = 2)

    const TSize cd = 16;
    const TSize w = 256, h = 256, b = 32;
    const TSize k = 1, s = 3, r = 3;
    const TSize ifmsizes[] = {cd, w, h, b};
    const TSize wsizes[]   = {k, cd, s, r};
    const TSize ofmsizes[] = {k, w - s + 1, h - r + 1, b};

    pTensor ifm(new Tensor(4, ifmsizes, syn_type_bf16));
    pTensor wgh(new Tensor(4, wsizes, syn_type_bf16));
    pTensor ofm(new Tensor(4, ofmsizes, syn_type_bf16));

    synConvolutionParams params;
    params.kW  = s;
    params.kH  = r;


    pBundle bundle = createSingleMMENodeBundle({ifm, wgh, nullptr, nullptr}, {ofm}, NodeFactory::convolutionNodeTypeName);

    unsigned doubleBufferfactor = 2;
    SlicingBrain::knobs.maxSRAMCapInBytes = doubleBufferfactor * ifm->getElementSizeInBytes() * (2 * h * w * cd + k * cd * s * r);
    SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon4D = 2*h*w;

    solveBundleWithStrategy(bundle);
    Solution& solution = bundle->getSolution();
    ASSERT_EQ((b / 2), solution.operations.size());
}

TEST_F(SRAMManagementTest, brain_should_slice_common_dimension_when_spatial_slicing_cant_help)
{
    // SRAM Capacity can fit 8 dedy batches, 8 activation batches and the dedw tensor.
    // Expects slicing the both inputs to single batch slices and leave the output in tact.

    // Given
    const TSize w = 64, h = 64, b = 255;
    const TSize k = 16, c = 16;
    const TSize r = 2;
    const TSize s = 2;
    const TSize ifmsizes[] = {c, w, h, b};
    const TSize dedysizes[]   = {k, w, h, b};
    const TSize dedwsizes[] = {c, k, r, s};

    const unsigned expBatchSliceSize = 8;

    pTensor ifm(new Tensor(4, ifmsizes, syn_type_float));
    pTensor dedy(new Tensor(4, dedysizes, syn_type_float));
    pTensor dedw(new Tensor(4, dedwsizes, syn_type_float));

    const TSize elemSize = dedw->getElementSizeInBytes();

    synConvolutionParams params;
    params.kH = r;
    params.kW = s;
    pNode conv = NodeFactory::createNode(
        {dedy, ifm}, {dedw}, &params, NodeFactory::deDwNodeTypeName, "conv");

    SlicingBrain::knobs.maxSRAMCapInBytes =
        elemSize * expBatchSliceSize * (h * w * k + h * w * c) + dedw->getDenseSizeInBytes();

    SlicingBrain::knobs.minCDSizeForPartials = h * w * expBatchSliceSize;

    // When
    pBundle bundle(new Bundle(BundleType::MME));
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), conv));
    bundle->addNode(conv);

    // Then
    solveBundleWithStrategy(bundle);
    Solution solution = bundle->getSolution();
    unsigned expNofOps = div_round_up(b ,expBatchSliceSize);
    checkSolutionSize(solution, 3, expNofOps);
    // output is not sliced.
    checkChunkSize(solution, dedw, dedw->getAllSizesInElements());
    // inputs sliced on batch
    checkChunkSize(solution, ifm, {c, w, h, expBatchSliceSize});
    checkChunkSize(solution, dedy, {k, w, h, expBatchSliceSize});
    ExecutionOrderChecker(DIM_B, DIM_B).checkReductionOnlyExecutionOrder(solution, expNofOps);
}

TEST_F(SRAMManagementTest, brain_should_slice_common_dim_w_on_flat_dedw)
{
    // SRAM Capacity can fit 8 dedy batches, 8 activation batches and the dedw tensor.
    // Expects slicing the both inputs to single batch slices and leave the output in tact.

    // Given
    const TSize w = 64*64*255, h = 1, b = 1;
    const TSize k = 16, c = 16;
    const TSize ifmsizes[] = {c, w, h, b};
    const TSize dedysizes[]   = {k, w, h, b};
    const TSize dedwsizes[] = {c, k, 1, 1};

    const unsigned expSliceSize = 64*64*1;

    pTensor ifm(new Tensor(4, ifmsizes, syn_type_float));
    pTensor dedy(new Tensor(4, dedysizes, syn_type_float));
    pTensor dedw(new Tensor(4, dedwsizes, syn_type_float));

    const TSize elemSize = dedw->getElementSizeInBytes();

    SlicingBrain::knobs.maxSRAMCapInBytes =
        elemSize * (c * expSliceSize + k * expSliceSize) + dedw->getDenseSizeInBytes();

    SlicingBrain::knobs.minCDSizeForPartials = expSliceSize;

    // When
    pBundle bundle = createSingleMMENodeBundle({dedy, ifm}, {dedw}, NodeFactory::deDwNodeTypeName);

    // Then
    solveBundleWithStrategy(bundle);
    Solution solution = bundle->getSolution();
    checkSolutionSize(solution, 3, div_round_up(w, expSliceSize));
    checkChunkSize(solution, dedy, {k, expSliceSize, 1, 1});
    checkChunkSize(solution, ifm, {c, expSliceSize, 1, 1});
    checkChunkSize(solution, dedw, dedw->getAllSizesInElements());
}

TEST_F(SRAMManagementTest, brain_should_take_the_correct_dimensions_for_dedx)
{
    // dy size  (16 * 64 - (64/3)) x 256
    // wgh size (16 * 64 - (64/6)) x 256
    // Common dimension is k (dim-0) - wgh is transposed internally.

    // SRAM has (2*64 + 2*64) * 256 * 4 bytes capacity.
    // Expects output to be sliced to 16*16 64x64 chunks.

    // Given
    TSize expChunkSize = 64;
    TSize expOperandChunks = 16;
    TSize k = 256;
    TSize w = expOperandChunks * expChunkSize - (expChunkSize / 3);
    TSize c = expOperandChunks * expChunkSize - (expChunkSize / 6);

    pTensor dy  = createTensor({k, w, 1, 1}, syn_type_float);
    pTensor wgh = createTensor({k, c, 1, 1}, syn_type_float);
    pTensor dx  = createTensor({c, w, 1, 1}, syn_type_float);

    SlicingBrain::knobs.maxSRAMCapInBytes = 256 * k * dy->getElementSizeInBytes();

    auto bundle = createSingleMMENodeBundle({dy, wgh}, {dx}, NodeFactory::deDxNodeTypeName);

    // When
    solveBundleWithStrategy(bundle);
    const Solution& solution = bundle->getSolution();

    // Then
    checkSolutionSize(solution, 3, expOperandChunks * expOperandChunks);
    checkChunkSize(solution, dy, {k, expChunkSize, 1, 1});
    checkChunkSize(solution, wgh, {k, expChunkSize, 1, 1});
    checkChunkSize(solution, dx, {expChunkSize, expChunkSize, 1, 1});

    // K is the common dimension. Slicing is performed on dimension W of dy and dimension C of wgh.
    ExecutionOrderChecker eoc(DIM_W, WEIGHT_DIM_C);
    eoc.checkWalkRightExecutionOrder(solution, expOperandChunks, expOperandChunks);
}

TEST_F(SRAMManagementTest, brain_should_take_the_correct_dimensions_for_gemm)
{
    // opA size (16 * 64 - (64/3)) x 256
    // opB size (12 * 64 - (64/6)) x 256
    // SRAM has (2*64 + 2*64) * 256 * 4 bytes capacity.
    // Expects output to be sliced to 16*12 64x64 chunks.

    // Given
    TSize expChunkSize = 64;
    TSize expOpAChunks = 16;
    TSize expOpBChunks = 12;
    TSize cdSize = 256;

    SlicingBrain::knobs.maxSRAMCapInBytes = 256 * cdSize * 4;

    for (bool opATpos : {false, true})
    {
        for (bool opBTpos : {false, true})
        {
            if (opATpos && opBTpos) continue; // Unsupported in Gaudi

            // And given
            pTensor opA = opATpos ? createTensor({expOpAChunks * expChunkSize, cdSize}, syn_type_float) :
                createTensor({cdSize, expOpAChunks * expChunkSize}, syn_type_float);
            unsigned opASlicedDim = opATpos ? DIM_C : DIM_W;
            pTensor opB = opBTpos ? createTensor({cdSize, expOpBChunks * expChunkSize}, syn_type_float) :
                createTensor({expOpBChunks * expChunkSize, cdSize}, syn_type_float);
            unsigned opBSlicedDim = opBTpos ? WEIGHT_DIM_C : WEIGHT_DIM_K;
            pTensor out = createTensor({expOpBChunks * expChunkSize, expOpAChunks * expChunkSize}, syn_type_float);

            synGEMMParams params(opATpos, opBTpos);
            auto          bundle =
                createSingleMMENodeBundle({opA, opB}, {out}, NodeFactory::gemmNodeTypeName, &params, sizeof(params));

            // When
            solveBundleWithStrategy(bundle);
            const Solution& solution = bundle->getSolution();

            // Then
            checkSolutionSize(solution, 3, expOpAChunks * expOpBChunks);
            opATpos ? checkChunkSize(solution, opA, {expChunkSize, cdSize}) :
                      checkChunkSize(solution, opA, {cdSize, expChunkSize});
            opBTpos ? checkChunkSize(solution, opB, {cdSize, expChunkSize}) :
                      checkChunkSize(solution, opB, {expChunkSize, cdSize});
            checkChunkSize(solution, out, {expChunkSize, expChunkSize});

            // K is the common dimension. Slicing is performed on dimension W of dy and dimension C of wgh.
            ExecutionOrderChecker eoc(opASlicedDim, opBSlicedDim);
            eoc.checkWalkRightExecutionOrder(solution, expOpAChunks, expOpBChunks);
        }
    }
}

TEST_F(SRAMManagementTest, batchnorm_should_identify_as_elementwise)
{
    std::unordered_set<std::string> elementwiseGUIDs={"batch_norm_stage2_fwd",
                                                      "batch_norm_stage2_relu_fwd",
                                                      "batch_norm_stage2_add_relu_fwd",
                                                      "batch_norm_stage1_bwd"};
    for (const auto& GUID : elementwiseGUIDs)
    {
        pTensor t1 = createTensor({128, 256}, syn_type_bf16);
        pTensor t2 = createTensor({128, 256}, syn_type_bf16);
        pNode n = NodeFactory::createGenericTPCNode({t1}, {t2}, nullptr, GUID.c_str(), "bn");
        //validate:
        auto tpcNode = std::dynamic_pointer_cast<TPCNode>(n);
        ASSERT_NE(tpcNode, nullptr);
        ASSERT_TRUE(tpcNode->isSeparable(tpc_lib_api::DEVICE_ID_GAUDI))
            << "GUID " << GUID << " expected to be considered as elementwise";
    }
}

TEST_F(SRAMManagementTest, relu_should_identify_as_elementwise)
{

    // Given
    pTensor t1 = createTensor({128, 256}, syn_type_bf16);
    pTensor t2 = createTensor({128, 256}, syn_type_bf16);
    pNode relu = NodeFactory::createGenericTPCNode({t1}, {t2}, nullptr, "relu_fwd_bf16", "relu");
    auto tpcNode = std::dynamic_pointer_cast<TPCNode>(relu);

    ASSERT_TRUE(tpcNode->isSeparable(tpc_lib_api::DEVICE_ID_GAUDI));
}

TEST_F(SRAMManagementTest, avgpool_shuld_identify_as_not_elementwise)
{
    // Given
    pTensor t1 = createTensor({1, 128, 256, 16}, syn_type_bf16);
    pTensor t2 = createTensor({1, 127, 255, 16}, syn_type_bf16);
    ns_AveragePooling::Params params{};
    std::memset(&params, 0, sizeof(params));
    params.kernel_h   = 2;
    params.kernel_w   = 2;
    params.stride_h   = 1;
    params.stride_w   = 1;
    params.dilation_h = 1;
    params.dilation_w = 1;
    params.pooling_convention = POOLING_CONVENTION_VALID;
    params.includePadding     = false;
    pNode avgpool = NodeFactory::createGenericTPCNode({t1}, {t2}, &params, "avg_pool_2d_fwd_bf16", "avgpool");
    auto tpcNode = std::dynamic_pointer_cast<TPCNode>(avgpool);

    ASSERT_FALSE(tpcNode->isSeparable(tpc_lib_api::DEVICE_ID_GAUDI));
}

// if a node is "separable", we should expect the same result in case we check on a dimension
// that doesn't exist in one of the input tensors
TEST_F(SRAMManagementTest, tpc_node_is_separable_per_dim)
{
    TSize   depth = 32;
    pTensor tIFM = createTensor({depth, 14, 14, 128}, syn_type_bf16);
    pTensor tBeta = createTensor({depth}, syn_type_float);
    pTensor tGamma = createTensor({depth}, syn_type_float);
    pTensor tMean = createTensor({depth}, syn_type_float);
    pTensor tVariance = createTensor({depth}, syn_type_float);
    pTensor tOFM = createTensor({depth, 14, 14, 128}, syn_type_bf16);

    ns_BatchNormKernel::Params params;
    params.momentum = 0;
    params.threshold.f = 0;
    params.epsilon = 1e-05;
    pNode batchNormInfNode = NodeFactory::createGenericTPCNode({tIFM, tBeta, tGamma, tMean, tVariance}, {tOFM}, &params,
                                                               "batch_norm_inf_bf16", "batch_norm_inference");
    auto tpcNode = std::dynamic_pointer_cast<TPCNode>(batchNormInfNode);
    ASSERT_TRUE(tpcNode->isSeparable(tpc_lib_api::DEVICE_ID_GAUDI));
    // even though we check for dims 1,2,3 for each of the input tensors (unlike the previous line)
    // the result of calling isSeparable() for those dims should still be true
    for (unsigned dim = 0; dim < tIFM->getDim(); dim++)
    {
        ASSERT_TRUE(tpcNode->isSeparable(tpc_lib_api::DEVICE_ID_GAUDI, dim))
            << "The node is separable, but not separable on"
               " dim: "
            << dim;
    }
}

TEST_F(SRAMManagementTest, bundlizer_should_add_tpc_producer_to_trivially_sliced_mme_consumer)
{
    // Given graph TPC -> MME(small operands)
    pTensor relu_in         = createTensor({2, 2}, syn_type_float);
    pTensor relu_out_mme_in = createTensor({2, 2}, syn_type_float);
    pTensor wgh             = createTensor({2, 2}, syn_type_float);
    pTensor mme_out         = createTensor({2, 2}, syn_type_float);

    synGEMMParams params{};
    pNode gemm = NodeFactory::createNode({relu_out_mme_in, wgh}, {mme_out}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    GraphEditor::addNode(getGraph(), gemm);

    pNode relu = NodeFactory::createGenericTPCNode({relu_in}, {relu_out_mme_in}, nullptr, "relu_fwd_f32", "relu");
    GraphEditor::addNode(getGraph(), relu);

    Bundlizer bundlizer(getGraph());
    pBundle bundle = bundlizer.getMMEBundles().front();

    // When the slicing strategy is trivial
    pMmeSlicingStrategy strategy =
        MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle->getNodes().front());
    pBundleExpansion expCnd   = bundlizer.findWideTpcProducerExpansionCandidate(strategy);
    bundlizer.addCandidateToBundle(bundle, expCnd);

    // Then the bundlizer adds the relu to the bundle
    ASSERT_EQ(expCnd->nodeToStitch, relu);
    ASSERT_EQ(expCnd->reshapeNode, nullptr);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, relu_out_mme_in);
    ASSERT_EQ(bundle->getNodes().size(), 2);
    const auto& bundleNodes = bundle->getNodes();
    ASSERT_NE(std::find(bundleNodes.begin(), bundleNodes.end(), relu), bundleNodes.end());
}

TEST_F(SRAMManagementTest, operand_slice_traveler_should_generate_left_to_right_slices_sequence)
{
    // Given
    const TSize    k = 18;
    const TSize    w = 10;
    const unsigned chunkDim = 4;
    const unsigned nofKSlices = std::ceil(static_cast<double>(k) / chunkDim);
    const unsigned nofWSlices = std::ceil(static_cast<double>(w) / chunkDim);
    pTensor t1 = createTensor({k, w}, syn_type_bf16);
    pSlicedOperand slicedOperand(new Solution::SlicedOperand(t1));
    slicedOperand->chunkDimensions[DIM_C] = chunkDim;
    slicedOperand->chunkDimensions[DIM_W] = chunkDim;

    // When
    SlicedOperandTraversalPattern walkingPattern(slicedOperand, {DIM_C, DIM_W});
    MultiOperandSliceIterator it = walkingPattern.begin();

    // Then
    for (unsigned wCoord = 0; wCoord < nofWSlices; wCoord++)
    {
        for (unsigned kCoord = 0; kCoord < nofKSlices; kCoord++)
        {
            ASSERT_NE(it, walkingPattern.end());
            const pSliceReference& next = (*it++).first;
            ASSERT_EQ(next->coordinates[DIM_W], wCoord);
            ASSERT_EQ(next->coordinates[DIM_C], kCoord);
            ASSERT_EQ(it.getCurrentIterator().inputSliceChanged(), kCoord == nofKSlices-1);
        }
    }
    ASSERT_EQ(it, walkingPattern.end());
}

TEST_F(SRAMManagementTest, operand_slice_traveler_4d_tensor_sliced_on_2_common_dims)
{
    synConvolutionParams convParams{};
    convParams.kH = convParams.kW = 3;
    convParams.dW = convParams.dH = 3;
    convParams.dilW = convParams.dilH = 1;
    convParams.padT = convParams.padL = convParams.padB = convParams.padR = 0;

    const TSize c = 64;
    const TSize k = 512;
    const TSize b = 3;
    std::array<TSize, 2> inputSizes = {144, 360};
    std::array<TSize, 2> outputSizes = {0};
    outputSizes[0] = convOutputDimSize(inputSizes[0], convParams.kW, convParams.dW, convParams.getPadL() + convParams.getPadR(), convParams.dilW);
    outputSizes[1] = convOutputDimSize(inputSizes[1], convParams.kH, convParams.dH, convParams.getPadT() + convParams.getPadB(), convParams.dilH);

    pTensor dy = createTensor({k, outputSizes[0], outputSizes[1], b}, syn_type_bf16);
    pTensor x = createTensor({c, inputSizes[0], inputSizes[1], b}, syn_type_bf16);
    pTensor dw = createTensor({k, c, convParams.kH, convParams.kW}, syn_type_bf16);
    pNode dedw = NodeFactory::createNode({dy, x}, {dw}, &convParams, NodeFactory::deDwNodeTypeName, "dedw");

    pSlicedOperand slicedDyIn = std::make_shared<SlicedOperand>(dedw->getInput(0));
    pSlicedOperand slicedXIn = std::make_shared<SlicedOperand>(dedw->getInput(1));
    pSlicedOperand slicedDwOut = std::make_shared<SlicedOperand>(dedw->getOutput(0));

    const unsigned chunkDimB = 1;
    const unsigned dyChunkDimH = 10;
    const unsigned xChunkDimH = 30;

    slicedDyIn->chunkDimensions[DIM_B] = chunkDimB;
    slicedDyIn->chunkDimensions[DIM_H] = dyChunkDimH;

    slicedXIn->chunkDimensions[DIM_B] = chunkDimB;
    slicedXIn->chunkDimensions[DIM_H] = xChunkDimH;

    const unsigned numOfBSlices = SlicedOperandUtils::nofSlices(slicedDyIn, DIM_B);
    const unsigned numOfHSlices = SlicedOperandUtils::nofSlices(slicedDyIn, DIM_H);
    unsigned numOfCommonDimSlicesDy = 1;
    for (unsigned i = 0; i < slicedDyIn->originalTensor->getDim(); ++i)
    {
        numOfCommonDimSlicesDy *= SlicedOperandUtils::nofSlices(slicedDyIn, i);
    }

    // Make sure X operand is sliced to the same number of slices
    ASSERT_EQ(numOfBSlices, SlicedOperandUtils::nofSlices(slicedXIn, DIM_B));
    ASSERT_EQ(numOfHSlices, SlicedOperandUtils::nofSlices(slicedXIn, DIM_H));
    unsigned numOfCommonDimSlicesX = 1;
    for (unsigned i = 0; i < slicedXIn->originalTensor->getDim(); ++i)
    {
        numOfCommonDimSlicesX *= SlicedOperandUtils::nofSlices(slicedXIn, i);
    }
    ASSERT_EQ(numOfCommonDimSlicesDy, numOfCommonDimSlicesX);

    pBackwardSliceMapping mapping = MMESliceMapper::mapOutputToInputs(dedw,
                                                                      slicedDyIn,
                                                                      slicedXIn,
                                                                      slicedDwOut,
                                                                      nullptr);
    ASSERT_NE(mapping, nullptr);

    SlicedOperandTraversalPattern walkingPattern(slicedDwOut,
                                                 SlicedOperandTraversalPattern::TOP_TO_BOTTOM_2D,
                                                 SlicingBrain::knobs.snakeWalkingTraversal,
                                                 numOfCommonDimSlicesDy);

    MultiOperandSliceIterator it = walkingPattern.begin();
    for (unsigned bCoord = 0; bCoord < numOfBSlices; bCoord++)
    {
        for (unsigned hCoord = 0; hCoord < numOfHSlices; hCoord++)
        {
            ASSERT_NE(it, walkingPattern.end());

            auto inputSlices = mapping->getInputs(*it);
            ASSERT_EQ(inputSlices.size(), 2);
            pSliceReference dySlice = inputSlices.front();
            pSliceReference xSlice = inputSlices.back();
            ASSERT_EQ(dySlice->coordinates[DIM_B], bCoord);
            ASSERT_EQ(xSlice->coordinates[DIM_B], bCoord);
            ASSERT_EQ(dySlice->coordinates[DIM_H], hCoord);
            ASSERT_EQ(xSlice->coordinates[DIM_H], hCoord);

            it++;
        }
    }
    ASSERT_EQ(it, walkingPattern.end());
}

TEST_F(SRAMManagementTest, snake_pattern_single_chunk)
{
    const TSize k          = 16;
    const TSize w          = 16;
    const TSize chunkDim   = 16;

    pTensor        t1 = createTensor({k, w}, syn_type_bf16);
    pSlicedOperand slicedOperand(new Solution::SlicedOperand(t1));

    slicedOperand->chunkDimensions[DIM_C] = chunkDim;
    slicedOperand->chunkDimensions[DIM_W] = chunkDim;


    SlicedOperandTraversalPattern walkingPattern(slicedOperand, {DIM_C, DIM_W}, true);
    MultiOperandSliceIterator it = walkingPattern.begin();

    const pSliceReference& next = (*it++).first;
    LOG_TRACE(GO_TEST, "[{}, {}]", next->coordinates[DIM_C], next->coordinates[DIM_W]);

    ASSERT_EQ(it, walkingPattern.end());
}

TEST_F(SRAMManagementTest, snake_pattern_3_chunks)
{
    const TSize    k          = 4;
    const TSize    w          = 320;
    const unsigned nofKSlices = std::ceil(static_cast<double>(k) / 4);
    const unsigned nofWSlices = std::ceil(static_cast<double>(w) / 128);

    pTensor        t1 = createTensor({k, w}, syn_type_bf16);
    pSlicedOperand slicedOperand(new Solution::SlicedOperand(t1));

    slicedOperand->chunkDimensions[DIM_C] = 4;
    slicedOperand->chunkDimensions[DIM_W] = 128;


    SlicedOperandTraversalPattern walkingPattern(slicedOperand, {DIM_C, DIM_W}, true);
    MultiOperandSliceIterator it = walkingPattern.begin();

    for (int wCoord = 0; wCoord < nofWSlices; wCoord++)
    {
        bool forward = (wCoord % 2) == 0 ? true : false;
        int  kBegin  = forward ? 0 : nofKSlices - 1;
        int  kEnd    = forward ? nofKSlices: 0;
        for (int kCoord = kBegin;
             forward ? kCoord < kEnd : kCoord >= kEnd ;
             forward ? kCoord++      : kCoord--)
        {
            ASSERT_NE(it, walkingPattern.end());
            const pSliceReference& next = (*it++).first;
            LOG_TRACE(GO_TEST, "[{}, {}]", next->coordinates[DIM_C], next->coordinates[DIM_W]);
            ASSERT_EQ(next->coordinates[DIM_W], wCoord);
            ASSERT_EQ(next->coordinates[DIM_C], kCoord);
            ASSERT_EQ(it.getCurrentIterator().inputSliceChanged(), (forward && kCoord == kEnd-1) || (!forward && kCoord == kEnd));
        }
    }

    ASSERT_EQ(it, walkingPattern.end());
}

TEST_F(SRAMManagementTest, snake_pattern_2d)
{
    const TSize    k          = 18;
    const TSize    w          = 16;
    const unsigned chunkDim   = 4;
    const unsigned nofKSlices = std::ceil(static_cast<double>(k) / chunkDim);
    const unsigned nofWSlices = std::ceil(static_cast<double>(w) / chunkDim);

    pTensor        t1 = createTensor({k, w}, syn_type_bf16);
    pSlicedOperand slicedOperand(new Solution::SlicedOperand(t1));

    slicedOperand->chunkDimensions[DIM_C] = chunkDim;
    slicedOperand->chunkDimensions[DIM_W] = chunkDim;


    SlicedOperandTraversalPattern walkingPattern(slicedOperand, {DIM_C, DIM_W}, true);
    MultiOperandSliceIterator     it = walkingPattern.begin();

    for (int wCoord = 0; wCoord < nofWSlices; wCoord++)
    {
        bool forward = (wCoord % 2) == 0 ? true : false;
        int  kBegin  = forward ? 0 : nofKSlices - 1;
        int  kEnd    = forward ? nofKSlices: 0;

        for (int kCoord = kBegin;
             forward ? kCoord < kEnd : kCoord >= kEnd ;
             forward ? kCoord++      : kCoord--)
        {
            ASSERT_NE(it, walkingPattern.end());
            const pSliceReference& next = (*it++).first;
            LOG_TRACE(GO_TEST, "[{}, {}]", next->coordinates[DIM_C], next->coordinates[DIM_W]);
            ASSERT_EQ(next->coordinates[DIM_W], wCoord);
            ASSERT_EQ(next->coordinates[DIM_C], kCoord);
            ASSERT_EQ(it.getCurrentIterator().inputSliceChanged(), (forward && kCoord == kEnd-1) || (!forward && kCoord == kEnd));
        }
    }
    ASSERT_EQ(it, walkingPattern.end());
}

TEST_F(SRAMManagementTest, snake_pattern_3d)
{
    const TSize k = 18;
    const TSize w = 16;
    const TSize h = 16;

    const unsigned chunkDim = 4;
    const unsigned nofKSlices = std::ceil(static_cast<double>(k) / chunkDim);
    const unsigned nofWSlices = std::ceil(static_cast<double>(w) / chunkDim);
    const unsigned nofHSlices = std::ceil(static_cast<double>(h) / chunkDim);

    pTensor t1 = createTensor({k, w, h}, syn_type_bf16);
    pSlicedOperand slicedOperand(new Solution::SlicedOperand(t1));
    slicedOperand->chunkDimensions[DIM_C] = chunkDim;
    slicedOperand->chunkDimensions[DIM_W] = chunkDim;
    slicedOperand->chunkDimensions[DIM_H] = chunkDim;

    SlicedOperandTraversalPattern walkingPattern(slicedOperand, {DIM_C, DIM_W, DIM_H}, true);
    MultiOperandSliceIterator it = walkingPattern.begin();

    for (int hCoord = 0; hCoord < nofHSlices; hCoord++)
    {
        for (int wCoord = 0; wCoord < nofWSlices; wCoord++)
        {
            bool forward = (wCoord % 2) == 0 ? true : false;
            int  kBegin  = forward ? 0 : nofKSlices - 1;
            int  kEnd    = forward ? nofKSlices: 0;

            for (int kCoord = kBegin;
                 forward ? kCoord < kEnd : kCoord >= kEnd ;
                 forward ? kCoord++      : kCoord--)
            {
                ASSERT_NE(it, walkingPattern.end());
                const pSliceReference& next = (*it++).first;
                LOG_TRACE(GO_TEST, "[{}, {}, {}]", next->coordinates[DIM_C], next->coordinates[DIM_W], next->coordinates[DIM_H]);
                ASSERT_EQ(next->coordinates[DIM_W], wCoord);
                ASSERT_EQ(next->coordinates[DIM_C], kCoord);
                ASSERT_EQ(next->coordinates[DIM_H], hCoord);
                ASSERT_EQ(it.getCurrentIterator().inputSliceChanged(), (forward && kCoord == kEnd-1) || (!forward && kCoord == kEnd));
            }
        }
    }
    ASSERT_EQ(it, walkingPattern.end());
}


TEST_F(SRAMManagementTest, operand_slice_traveler_should_generate_top_to_bottom_slices_sequence)
{
    // Given
    const TSize    k = 18;
    const TSize    w = 10;
    const unsigned chunkDim = 4;
    const unsigned nofKSlices = std::ceil(static_cast<double>(k) / chunkDim);
    const unsigned nofWSlices = std::ceil(static_cast<double>(w) / chunkDim);
    pTensor t1 = createTensor({k, w}, syn_type_bf16);
    pSlicedOperand slicedOperand(new Solution::SlicedOperand(t1));
    slicedOperand->chunkDimensions[DIM_C] = chunkDim;
    slicedOperand->chunkDimensions[DIM_W] = chunkDim;

    // When
    SlicedOperandTraversalPattern walkingPattern(slicedOperand, {DIM_W, DIM_C});
    MultiOperandSliceIterator it = walkingPattern.begin();

    // Then
    for (unsigned kCoord = 0; kCoord < nofKSlices; kCoord++)
    {
        for (unsigned wCoord = 0; wCoord < nofWSlices; wCoord++)
        {
            ASSERT_NE(it, walkingPattern.end());
            const pSliceReference& next = (*it++).first;
            ASSERT_EQ(next->coordinates[DIM_W], wCoord);
            ASSERT_EQ(next->coordinates[DIM_C], kCoord);
            ASSERT_EQ(it.getCurrentIterator().inputSliceChanged(), wCoord == nofWSlices-1);
        }
    }
    ASSERT_EQ(it, walkingPattern.end());
}

TEST_F(SRAMManagementTest, operand_slice_traveler_should_generate_k_then_batch_slices_sequence)
{
    // Given
    const TSize    k = 18;
    const TSize    w = 17;
    const TSize    h = 13;
    const TSize    b = 10;
    const unsigned chunkDim = 4;
    const unsigned nofKSlices = std::ceil(static_cast<double>(k) / chunkDim);
    const unsigned nofBSlices = std::ceil(static_cast<double>(b) / chunkDim);
    pTensor t1 = createTensor({k, w, h, b}, syn_type_bf16);
    pSlicedOperand slicedOperand(new Solution::SlicedOperand(t1));
    slicedOperand->chunkDimensions[DIM_C] = chunkDim;
    slicedOperand->chunkDimensions[DIM_B] = chunkDim;

    // When
    SlicedOperandTraversalPattern walkingPattern(slicedOperand, {DIM_C, DIM_B});
    MultiOperandSliceIterator it = walkingPattern.begin();

    // Then
    for (unsigned bCoord = 0; bCoord < nofBSlices; bCoord++)
    {
        for (unsigned kCoord = 0; kCoord < nofKSlices; kCoord++)
        {
            ASSERT_NE(it, walkingPattern.end());
            const pSliceReference& next = (*it++).first;
            ASSERT_EQ(next->coordinates[DIM_C], kCoord);
            ASSERT_EQ(next->coordinates[DIM_W], 0);
            ASSERT_EQ(next->coordinates[DIM_H], 0);
            ASSERT_EQ(next->coordinates[DIM_B], bCoord);
            ASSERT_EQ(it.getCurrentIterator().inputSliceChanged(), kCoord == nofKSlices - 1);
        }
    }
    ASSERT_EQ(it, walkingPattern.end());
}

TEST_F(SRAMManagementTest, operand_slice_traveler_should_generate_partials)
{
    // Given
    const TSize    k = 2;
    const TSize    w = 2;
    const unsigned chunkDim = 1;
    const unsigned numOfCommonDimSlices = 4;
    const unsigned nofKSlices = std::ceil(static_cast<double>(k) / chunkDim);
    const unsigned nofBSlices = std::ceil(static_cast<double>(w) / chunkDim);
    pTensor t1 = createTensor({k, w}, syn_type_bf16);
    pSlicedOperand slicedOperand(new Solution::SlicedOperand(t1));
    slicedOperand->chunkDimensions[DIM_C] = chunkDim;
    slicedOperand->chunkDimensions[DIM_W] = chunkDim;

    // When
    SlicedOperandTraversalPattern walkingPattern(slicedOperand, {DIM_C, DIM_W}, true, numOfCommonDimSlices);
    MultiOperandSliceIterator it = walkingPattern.begin();

    // Then
    for (unsigned wCoord = 0; wCoord < nofBSlices; wCoord++)
    {
        for (unsigned kCoord = 0; kCoord < nofKSlices; kCoord++)
        {
            ASSERT_NE(it, walkingPattern.end());
            pSliceReference next;
            for (unsigned commonDimCoord = 0; commonDimCoord < numOfCommonDimSlices; commonDimCoord++)
            {
                // discard first slice
                if (commonDimCoord != 0 || kCoord != 0 || wCoord != 0)
                {
                    ASSERT_TRUE(it.getCurrentIterator().inputSliceChanged());
                }
                unsigned expectedCommonDimCoord = kCoord % 2 ? numOfCommonDimSlices - 1 - commonDimCoord : commonDimCoord; // snake
                ASSERT_EQ((*it).second, expectedCommonDimCoord);
                next = (*it++).first;
                LOG_TRACE(GO_TEST,"coordinates -  {}, commonDimSlice - {}", toString(next->coordinates, ','), commonDimCoord);
                unsigned expectedKCoord = wCoord % 2 ? nofKSlices - 1 - kCoord : kCoord;
                ASSERT_EQ(next->coordinates[DIM_C], expectedKCoord);
                ASSERT_EQ(next->coordinates[DIM_W], wCoord);
            }
        }
    }
    ASSERT_EQ(it, walkingPattern.end());
}

TEST_F(SRAMManagementTest, multi_operand_iterator_should_alternate_master_slave_top_down)
{
    // Tensors to traverse on are of size - 1x10  sliced 1x5
    const TSize    masterK = 1, slaveK = 1;
    const TSize    masterW = 10, slaveW = 10;
    const unsigned chunkDim = 5;
    pTensor master = createTensor({masterK, masterW}, syn_type_bf16);
    master->setName("masterTensor");
    pTensor slave = createTensor({slaveK, slaveW}, syn_type_bf16);
    slave->setName("slaveTensor");
    pSlicedOperand masterOperand(new Solution::SlicedOperand(master));
    pSlicedOperand slaveOperand(new Solution::SlicedOperand(slave));
    masterOperand->chunkDimensions[DIM_W] = chunkDim;
    slaveOperand->chunkDimensions[DIM_W] = chunkDim;
    // Test 1 - top-down walking pattern
    SlicedOperandTraversalPattern walkingPatternMaster(masterOperand, {DIM_W, DIM_C});
    SlicedOperandTraversalPattern walkingPatternSlave(slaveOperand, {DIM_W, DIM_C});
    MultiOperandSliceIterator multiIterator(walkingPatternMaster.begin());
    multiIterator.addOperandIterator(walkingPatternSlave.begin());
    unsigned sliceCount = 0;
    while(multiIterator != multiIterator.getEndIterator())
    {
        pSliceReference ref = (*multiIterator).first;
        LOG_TRACE(GO_TEST, "Operand - {}, coordinate - {}", ref->operand->originalTensor->getName(),
                  toString(ref->coordinates, ','));
        // all even slices should be master slices, odd slices are slave.
        ASSERT_EQ(ref->operand == masterOperand, sliceCount % 2 == 0);
        ASSERT_EQ(ref->operand == slaveOperand, sliceCount % 2 != 0);
        multiIterator++;
        sliceCount++;
    }
    ASSERT_EQ(multiIterator, multiIterator.getEndIterator());
    ASSERT_EQ(sliceCount, 4);
}

TEST_F(SRAMManagementTest, multi_operand_iterator_should_alternate_master_slave_left_right)
{
    // Tensors to traverse on are of size - 1x10  sliced 1x5
    const TSize    masterK = 1, slaveK = 1;
    const TSize    masterW = 10, slaveW = 10;
    const unsigned chunkDim = 5;
    pTensor master = createTensor({masterK, masterW}, syn_type_bf16);
    master->setName("masterTensor");
    pTensor slave = createTensor({slaveK, slaveW}, syn_type_bf16);
    slave->setName("slaveTensor");
    pSlicedOperand masterOperand(new Solution::SlicedOperand(master));
    pSlicedOperand slaveOperand(new Solution::SlicedOperand(slave));
    masterOperand->chunkDimensions[DIM_W] = chunkDim;
    slaveOperand->chunkDimensions[DIM_W] = chunkDim;
    SlicedOperandTraversalPattern walkingPatternMaster(masterOperand, {DIM_C, DIM_W});
    SlicedOperandTraversalPattern walkingPatternSlave(slaveOperand, {DIM_C, DIM_W});
    MultiOperandSliceIterator multiIterator(walkingPatternMaster.begin());
    multiIterator.addOperandIterator(walkingPatternSlave.begin());
    unsigned sliceCount = 0;
    while(multiIterator != multiIterator.getEndIterator())
    {
        pSliceReference ref = (*multiIterator).first;
        LOG_TRACE(GO_TEST,"Operand - {}, coordinate - {}", ref->operand->originalTensor->getName(),
                                                      toString(ref->coordinates, ','));
        // all even slices should be master slices, odd slices are slave.
        ASSERT_EQ(ref->operand == masterOperand, sliceCount % 2 == 0);
        ASSERT_EQ(ref->operand == slaveOperand, sliceCount % 2 != 0);
        multiIterator++;
        sliceCount++;
    }
    ASSERT_EQ(multiIterator, multiIterator.getEndIterator());
    ASSERT_EQ(sliceCount, 4);
}

TEST_F(SRAMManagementTest, multi_operand_iterator_2d_snake_slave_larger_than_master)
{
    const TSize    masterK = 2, slaveK = 4;
    const TSize    masterW = 2, slaveW = 2;
    const unsigned chunkDim = 1;
    pTensor master = createTensor({masterK, masterW}, syn_type_bf16);
    master->setName("masterTensor");
    pTensor slave = createTensor({slaveK, slaveW}, syn_type_bf16);
    slave->setName("slaveTensor");
    pSlicedOperand masterOperand(new Solution::SlicedOperand(master));
    pSlicedOperand slaveOperand(new Solution::SlicedOperand(slave));
    masterOperand->chunkDimensions[DIM_W] = chunkDim;
    masterOperand->chunkDimensions[DIM_C] = chunkDim;
    slaveOperand->chunkDimensions[DIM_W] = chunkDim;
    slaveOperand->chunkDimensions[DIM_C] = chunkDim;
    SlicedOperandTraversalPattern walkingPatternMaster(masterOperand, {DIM_W, DIM_C}, true);
    SlicedOperandTraversalPattern walkingPatternSlave(slaveOperand, {DIM_W, DIM_C}, true);
    MultiOperandSliceIterator multiIterator(walkingPatternMaster.begin());
    multiIterator.addOperandIterator(walkingPatternSlave.begin());
    unsigned sliceCount = 0;
    while(multiIterator != multiIterator.getEndIterator())
    {
        pSliceReference ref = (*multiIterator).first;
        LOG_TRACE(GO_TEST, "Operand - {}, coordinate - {}", ref->operand->originalTensor->getName(),
                  toString(ref->coordinates, ','));
        if (sliceCount < 8)
        {
            // all even slices should be master slices, odd slices are slave.
            ASSERT_EQ(ref->operand == masterOperand, sliceCount % 2 == 0);
            ASSERT_EQ(ref->operand == slaveOperand, sliceCount % 2 != 0);
        }
        else
        {
            // all operands are slave
            ASSERT_EQ(ref->operand, slaveOperand);
        }
        multiIterator++;
        sliceCount++;
    }
    ASSERT_EQ(multiIterator, multiIterator.getEndIterator());
    ASSERT_EQ(sliceCount, masterK * masterW + slaveK * slaveW);
}

TEST_F(SRAMManagementTest, multi_operand_iterator_2d_snake_master_larger_than_slave_top_down)
{
    const TSize    masterK = 4, slaveK = 2;
    const TSize    masterW = 4, slaveW = 2;
    const unsigned chunkDim = 1;
    pTensor master = createTensor({masterK, masterW}, syn_type_bf16);
    master->setName("masterTensor");
    pTensor slave = createTensor({slaveK, slaveW}, syn_type_bf16);
    slave->setName("slaveTensor");
    pSlicedOperand masterOperand(new Solution::SlicedOperand(master));
    pSlicedOperand slaveOperand(new Solution::SlicedOperand(slave));
    masterOperand->chunkDimensions[DIM_W] = chunkDim;
    masterOperand->chunkDimensions[DIM_C] = chunkDim;
    slaveOperand->chunkDimensions[DIM_W] = chunkDim;
    slaveOperand->chunkDimensions[DIM_C] = chunkDim;
    SlicedOperandTraversalPattern walkingPatternMaster(masterOperand, {DIM_W, DIM_C}, true);
    SlicedOperandTraversalPattern walkingPatternSlave(slaveOperand, {DIM_W, DIM_C}, true);
    MultiOperandSliceIterator multiIterator(walkingPatternMaster.begin());
    multiIterator.addOperandIterator(walkingPatternSlave.begin());
    unsigned sliceCount = 0;
    while(multiIterator != multiIterator.getEndIterator())
    {
        pSliceReference ref = (*multiIterator).first;
        LOG_TRACE(GO_TEST, "Operand - {}, coordinate - {}", ref->operand->originalTensor->getName(),
                  toString(ref->coordinates, ','));
        if (sliceCount < 4 || (sliceCount >= 6 && sliceCount < 10))
        {
            // all even slices should be master slices, odd slices are slave.
            ASSERT_EQ(ref->operand == masterOperand, sliceCount % 2 == 0);
            ASSERT_EQ(ref->operand == slaveOperand, sliceCount % 2 != 0);
        }
        else
        {
            // all operands are master
            ASSERT_EQ(ref->operand, masterOperand);
        }
        multiIterator++;
        sliceCount++;
    }
    ASSERT_EQ(multiIterator, multiIterator.getEndIterator());
    ASSERT_EQ(sliceCount, masterK * masterW + slaveK * slaveW);
}

TEST_F(SRAMManagementTest, multi_operand_iterator_2d_snake_master_larger_than_slave_left_right)
{
    const TSize    masterK = 4, slaveK = 2;
    const TSize    masterW = 4, slaveW = 2;
    const unsigned chunkDim = 1;
    pTensor master = createTensor({masterK, masterW}, syn_type_bf16);
    master->setName("masterTensor");
    pTensor slave = createTensor({slaveK, slaveW}, syn_type_bf16);
    slave->setName("slaveTensor");
    pSlicedOperand masterOperand(new Solution::SlicedOperand(master));
    pSlicedOperand slaveOperand(new Solution::SlicedOperand(slave));
    masterOperand->chunkDimensions[DIM_W] = chunkDim;
    masterOperand->chunkDimensions[DIM_C] = chunkDim;
    slaveOperand->chunkDimensions[DIM_W] = chunkDim;
    slaveOperand->chunkDimensions[DIM_C] = chunkDim;
    SlicedOperandTraversalPattern walkingPatternMaster(masterOperand, {DIM_C, DIM_W}, true);
    SlicedOperandTraversalPattern walkingPatternSlave(slaveOperand, {DIM_C, DIM_W}, true);
    MultiOperandSliceIterator multiIterator(walkingPatternMaster.begin());
    multiIterator.addOperandIterator(walkingPatternSlave.begin());
    unsigned sliceCount = 0;
    while(multiIterator != multiIterator.getEndIterator())
    {
        pSliceReference ref = (*multiIterator).first;
        LOG_TRACE(GO_TEST, "Operand - {}, coordinate - {}", ref->operand->originalTensor->getName(),
                  toString(ref->coordinates, ','));
        if (sliceCount < 4 || (sliceCount >= 6 && sliceCount < 10))
        {
            // all even slices should be master slices, odd slices are slave.
            ASSERT_EQ(ref->operand == masterOperand, sliceCount % 2 == 0);
            ASSERT_EQ(ref->operand == slaveOperand, sliceCount % 2 != 0);
        }
        else
        {
            // all operands are master
            ASSERT_EQ(ref->operand, masterOperand);
        }
        multiIterator++;
        sliceCount++;
    }
    ASSERT_EQ(multiIterator, multiIterator.getEndIterator());
    ASSERT_EQ(sliceCount, masterK * masterW + slaveK * slaveW);
}

TEST_F(SRAMManagementTest, multi_operand_iterator_2d_snake_with_partials)
{
    const TSize    masterK = 4, slaveK = 1;
    const TSize    masterW = 4, slaveW = 1;
    const unsigned slaveCD = 4;
    const unsigned chunkDim = 1;
    pTensor master = createTensor({masterK, masterW}, syn_type_bf16);
    master->setName("masterTensor");
    pTensor slave = createTensor({slaveK, slaveW}, syn_type_bf16);
    slave->setName("slaveTensor");
    pSlicedOperand masterOperand(new Solution::SlicedOperand(master));
    pSlicedOperand slaveOperand(new Solution::SlicedOperand(slave));
    masterOperand->chunkDimensions[DIM_W] = chunkDim;
    masterOperand->chunkDimensions[DIM_C] = chunkDim;
    slaveOperand->chunkDimensions[DIM_W] = chunkDim;
    slaveOperand->chunkDimensions[DIM_C] = chunkDim;
    SlicedOperandTraversalPattern walkingPatternMaster(masterOperand, {DIM_C, DIM_W}, true);
    SlicedOperandTraversalPattern walkingPatternSlave(slaveOperand, {DIM_C, DIM_W}, true, slaveCD);
    MultiOperandSliceIterator multiIterator(walkingPatternMaster.begin());
    multiIterator.addOperandIterator(walkingPatternSlave.begin());
    unsigned sliceCount = 0;
    unsigned slaveSliceCount = 0;
    CoordArray zeroCoord = {0};
    while(multiIterator != multiIterator.getEndIterator())
    {
        pSliceReference ref = (*multiIterator).first;
        unsigned commonDimSlice = (*multiIterator).second;
        LOG_TRACE(GO_TEST, "Operand - {}, coordinate - {}, commonDimSlice = {}", ref->operand->originalTensor->getName(),
                  toString(ref->coordinates, ','), commonDimSlice);
        if (sliceCount == 1 || sliceCount == 6 || sliceCount == 11 || sliceCount == 16)
        {
            // slave operands
            ASSERT_EQ(ref->operand, slaveOperand);
            ASSERT_EQ(ref->coordinates, zeroCoord);
            ASSERT_EQ(commonDimSlice, slaveSliceCount);
            slaveSliceCount++;
        }
        else
        {
            // master operands
            ASSERT_EQ(ref->operand, masterOperand);
        }
        multiIterator++;
        sliceCount++;
    }
    ASSERT_EQ(multiIterator, multiIterator.getEndIterator());
    ASSERT_EQ(sliceCount, masterK * masterW + slaveCD);
    ASSERT_EQ(slaveSliceCount, slaveCD);
}

TEST_F(SRAMManagementTest, multi_operand_iterator_2d_snake_different_walking_pattern)
{
    const TSize masterK = 4, slaveK = 4;
    const TSize masterW = 4, slaveW = 2;
    const unsigned chunkDim = 1;
    pTensor master = createTensor({masterK, masterW}, syn_type_bf16);
    master->setName("masterTensor");
    pTensor slave = createTensor({slaveK, slaveW}, syn_type_bf16);
    slave->setName("slaveTensor");
    pSlicedOperand masterOperand(new Solution::SlicedOperand(master));
    pSlicedOperand slaveOperand(new Solution::SlicedOperand(slave));
    masterOperand->chunkDimensions[DIM_W] = chunkDim;
    masterOperand->chunkDimensions[DIM_C] = chunkDim;
    slaveOperand->chunkDimensions[DIM_W] = chunkDim;
    slaveOperand->chunkDimensions[DIM_C] = chunkDim;
    SlicedOperandTraversalPattern walkingPatternMaster(masterOperand, {DIM_C, DIM_W}, true);
    SlicedOperandTraversalPattern walkingPatternSlave(slaveOperand, {DIM_W, DIM_C}, true);
    MultiOperandSliceIterator multiIterator(walkingPatternMaster.begin());
    multiIterator.addOperandIterator(walkingPatternSlave.begin());
    unsigned sliceCount = 0;
    while(multiIterator != multiIterator.getEndIterator())
    {
        pSliceReference ref = (*multiIterator).first;
        unsigned commonDimSlice = (*multiIterator).second;
        LOG_TRACE(GO_TEST, "Operand - {}, coordinate - {}, commonDimSlice = {}", ref->operand->originalTensor->getName(),
                  toString(ref->coordinates, ','), commonDimSlice);
        if (sliceCount == 1 || sliceCount == 3 || sliceCount == 7 || sliceCount == 9 ||
            sliceCount == 13 || sliceCount == 15 || sliceCount == 19 || sliceCount == 21)
        {
            // slave operands
            ASSERT_EQ(ref->operand, slaveOperand);
        }
        else
        {
            // master operands
            ASSERT_EQ(ref->operand, masterOperand);
        }
        multiIterator++;
        sliceCount++;
    }
    ASSERT_EQ(multiIterator, multiIterator.getEndIterator());
    ASSERT_EQ(sliceCount, masterK * masterW + slaveK * slaveW);
}

TEST_F(SRAMManagementTest, solution_generator_should_generate_single_node_bundle_rtl_execution_order)
{
    // Given
    const TSize chunkWidth = 8;
    const TSize chunkDepth = 4;
    const TSize expRows = 2;
    const TSize expCols = 4;
    pTensor x = createTensor({32                                , expRows * chunkWidth       }, syn_type_bf16);
    pTensor w = createTensor({expCols * chunkDepth              , 32                         }, syn_type_bf16);
    pTensor y = createTensor({w->getSizeInElements(WEIGHT_DIM_K), x->getSizeInElements(DIM_W)}, syn_type_bf16);

    pBundle bundle = createSingleMMENodeBundle({x, w}, {y}, NodeFactory::convolutionNodeTypeName);

    pMmeSlicingStrategy strategy =
        MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle->getNodes().front());
    pSlicedOperand slicedX = strategy->getMmeSlicingData().bundleTensors[0];
    pSlicedOperand slicedW = strategy->getMmeSlicingData().bundleTensors[1];
    pSlicedOperand slicedY = strategy->getMmeSlicingData().masterOperand;
    slicedX->chunkDimensions[DIM_W]        = chunkWidth;
    slicedW->chunkDimensions[WEIGHT_DIM_K] = chunkDepth;
    slicedY->chunkDimensions[DIM_W]        = chunkWidth;
    slicedY->chunkDimensions[WEIGHT_DIM_K] = chunkDepth;

    // When
    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, strategy).fillSolution());

    // Then
    const Solution& solution = bundle->getSolution();
    checkSolutionSize(solution, 3, expRows * expCols);
    checkChunkSize(solution, x, slicedX->chunkDimensions);
    checkChunkSize(solution, w, slicedW->chunkDimensions);
    checkChunkSize(solution, y, slicedY->chunkDimensions);
    ExecutionOrderChecker(DIM_W, WEIGHT_DIM_K).checkWalkRightExecutionOrder(solution, expRows, expCols);
}

TEST_F(SRAMManagementTest, solution_generator_should_generate_single_node_bundle_ttb_execution_order)
{
    // Given
    const TSize chunkWidth = 8;
    const TSize chunkDepth = 4;
    const TSize expRows = 2;
    const TSize expCols = 4;
    pTensor x = createTensor({32                                , expRows * chunkWidth       }, syn_type_bf16);
    pTensor w = createTensor({expCols * chunkDepth              , 32                         }, syn_type_bf16);
    pTensor y = createTensor({w->getSizeInElements(WEIGHT_DIM_K), x->getSizeInElements(DIM_W)}, syn_type_bf16);

    pBundle bundle = createSingleMMENodeBundle({x, w}, {y}, NodeFactory::convolutionNodeTypeName);

    pMmeSlicingStrategy strategy =
        MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle->getNodes().front());
    strategy->setOutputTraversalPattern(SlicedOperandTraversalPattern::TOP_TO_BOTTOM_2D);
    pSlicedOperand slicedX = strategy->getMmeSlicingData().bundleTensors[0];
    pSlicedOperand slicedW = strategy->getMmeSlicingData().bundleTensors[1];
    pSlicedOperand slicedY = strategy->getMmeSlicingData().masterOperand;
    slicedX->chunkDimensions[DIM_W]        = chunkWidth;
    slicedW->chunkDimensions[WEIGHT_DIM_K] = chunkDepth;
    slicedY->chunkDimensions[DIM_W]        = chunkWidth;
    slicedY->chunkDimensions[WEIGHT_DIM_K] = chunkDepth;

    // When
    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, strategy).fillSolution());

    // Then
    const Solution& solution = bundle->getSolution();
    checkSolutionSize(solution, 3, expRows * expCols);
    checkChunkSize(solution, x, slicedX->chunkDimensions);
    checkChunkSize(solution, w, slicedW->chunkDimensions);
    checkChunkSize(solution, y, slicedY->chunkDimensions);
    ExecutionOrderChecker(DIM_W, WEIGHT_DIM_K).checkWalkDownExecutionOrder(solution, expRows, expCols);
}

TEST_F(SRAMManagementTest, solution_generator_should_generate_single_node_bundle_batch_k_execution_order)
{
    // Given
    const TSize chunkWidth = 8;
    const TSize chunkDepth = 4;
    const TSize expBatches = 2;
    const TSize expCols    = 4;
    pTensor x = createTensor({32, 16, 16, expBatches * chunkWidth}, syn_type_bf16);
    pTensor w = createTensor({expCols * chunkDepth, 32}, syn_type_bf16);
    pTensor y = createTensor({w->getSizeInElements(WEIGHT_DIM_K), x->getSizeInElements(DIM_W),
                              x->getSizeInElements(DIM_H), x->getSizeInElements(DIM_B)}, syn_type_bf16);

    pBundle bundle = createSingleMMENodeBundle({x, w}, {y}, NodeFactory::convolutionNodeTypeName);

    pMmeSlicingStrategy strategy =
        MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle->getNodes().front());
    strategy->setOutputTraversalPattern(SlicedOperandTraversalPattern::LEFT_TO_RIGHT_4D);
    pSlicedOperand slicedX = strategy->getMmeSlicingData().bundleTensors[0];
    pSlicedOperand slicedW = strategy->getMmeSlicingData().bundleTensors[1];
    pSlicedOperand slicedY = strategy->getMmeSlicingData().masterOperand;
    slicedX->chunkDimensions[DIM_B]        = chunkWidth;
    slicedW->chunkDimensions[WEIGHT_DIM_K] = chunkDepth;
    slicedY->chunkDimensions[DIM_B]        = chunkWidth;
    slicedY->chunkDimensions[WEIGHT_DIM_K] = chunkDepth;

    // When
    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, strategy).fillSolution());

    // Then
    const Solution& solution = bundle->getSolution();
    checkSolutionSize(solution, 3, expBatches * expCols);
    checkChunkSize(solution, x, slicedX->chunkDimensions);
    checkChunkSize(solution, w, slicedW->chunkDimensions);
    checkChunkSize(solution, y, slicedY->chunkDimensions);
    ExecutionOrderChecker(DIM_B, WEIGHT_DIM_K).checkWalkRightExecutionOrder(solution, expBatches, expCols, {DIM_B, WEIGHT_DIM_K});
}

TEST_F(SRAMManagementTest, solution_generator_should_add_fwd_mapped_operands_to_execution_order)
{
    pTensor x       = createTensor({128, 16 * 1024}, syn_type_bf16);
    pTensor w       = createTensor({128, 128}      , syn_type_bf16);
    pTensor y       = createTensor({128, 16 * 1024}, syn_type_bf16);
    pTensor reluOut = createTensor({128, 16 * 1024}, syn_type_bf16);

    synConvolutionParams params {};
    pNode conv = NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    GraphEditor::addNode(getGraph(), conv);
    pNode relu = NodeFactory::createGenericTPCNode({y}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    GraphEditor::addNode(getGraph(), relu);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    Bundlizer bundlizer {getGraph()};
    pBundle bundle = bundlizer.getMMEBundles().front();
    ASSERT_TRUE(bundle);

    pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(getMmeBrain().getSolutionStrategies(bundle), bundle, getGraph(), getSlicingBrains()));
    ASSERT_TRUE(strategy);
    strategy->getMmeSlicingData().bundleTensors[0]->chunkDimensions[DIM_W] = 1024;
    strategy->getMmeSlicingData().masterOperand->chunkDimensions[DIM_W] = 1024;

    pBundleExpansion expCnd = bundlizer.findTpcConsumerExpansionCandidate(strategy);
    ASSERT_TRUE(expCnd);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, y);
    ASSERT_EQ(expCnd->nodeToStitch, relu);

    TPCSlaveBrain tpcSlaveBrain {getGraph()};
    tpcSlaveBrain.addConsumerToStrategy(expCnd, strategy);
    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, strategy).fillSolution());

    const Solution& solution = bundle->getSolution();
    ASSERT_EQ(solution.operations.size(), 16 * 2) << "Expects MME -> TPC x 16";
    unsigned opIdx = 0;
    CoordArray lastCoord;
    for (const Solution::Operation& op : bundle->getSolution().operations)
    {
        if (opIdx % 2)
        {
            ASSERT_EQ(op.originalNode, relu) << "wrong node in operation " << opIdx;
            ASSERT_EQ(op.inputs.size(), 1) << "wrong number of inputs in operation " << opIdx;
            ASSERT_EQ(op.inputs.front()->coordinates, lastCoord) << "wrong input coords in operation " << opIdx;
            ASSERT_EQ(op.outputs.size(), 1) << "wrong number of outputs in operation " << opIdx;
            ASSERT_EQ(op.outputs.front()->coordinates, lastCoord) << "wrong output coords in operation " << opIdx;
        }
        else
        {
            ASSERT_EQ(op.originalNode, conv) << "wrong node in operation " << opIdx;
            lastCoord = op.outputs.front()->coordinates;
        }
        opIdx++;
    }
}

TEST_F(SRAMManagementTest, solution_generator_should_add_fwd_mapped_operand_chain_to_execution_order)
{
    pTensor x       = createTensor({128, 1024}, syn_type_bf16);
    pTensor w       = createTensor({128, 128}      , syn_type_bf16);
    pTensor y       = createTensor({128, 1024}, syn_type_bf16);
    pTensor reluIn  = createTensor({128, 32, 32, 1}, syn_type_bf16);
    pTensor reluOut = createTensor({128, 32, 32, 1}, syn_type_bf16);

    synConvolutionParams params {};
    pNode conv = NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    GraphEditor::addNode(getGraph(), conv);
    pNode reshape = NodeFactory::createNode({y}, {reluIn}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    GraphEditor::addNode(getGraph(), reshape);
    pNode relu = NodeFactory::createGenericTPCNode({reluIn}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    GraphEditor::addNode(getGraph(), relu);

    Bundlizer bundlizer {getGraph()};
    pBundle bundle = bundlizer.getMMEBundles().front();
    ASSERT_TRUE(bundle);

    TrivialSolver solver(*getGraph().getHALReader(), bundle);
    solver.createAllStrategies();
    pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(solver.getStrategies().front());
    ASSERT_TRUE(strategy);

    pBundleExpansion expCnd = bundlizer.findTpcConsumerExpansionCandidate(strategy);
    ASSERT_TRUE(expCnd);
    ASSERT_TRUE(expCnd->stitchedOperand);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, y);
    ASSERT_EQ(expCnd->nodeToStitch, relu);
    ASSERT_EQ(expCnd->reshapeNode, reshape);

    ReshapeSlicingBrain reshapeBrain {getGraph()};
    reshapeBrain.addConsumerToStrategy(expCnd, strategy);
    TPCSlaveBrain tpcSlaveBrain {getGraph()};
    tpcSlaveBrain.addConsumerToStrategy(expCnd, strategy);

    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, strategy).fillSolution());

    const Solution& solution = bundle->getSolution();
    ASSERT_EQ(solution.operations.size(), 3) << "Expects MME -> RESHAPE -> TPC";
    unsigned opIdx = 0;
    for (const Solution::Operation& op : bundle->getSolution().operations)
    {
        switch (opIdx)
        {
        case 0:
            ASSERT_EQ(op.originalNode, conv);
            break;
        case 1:
            ASSERT_EQ(op.originalNode, reshape);
            break;
        case 2:
            ASSERT_EQ(op.originalNode, relu);
            break;
        default:
            FAIL() << "Unexpected operation in solution";
        }
        opIdx++;
    }
}
TEST_F(SRAMManagementTest, tpc_slave_brain_should_add_tpc_elementwise_producer_to_solution)
{
    // Given conv bundle and relu node produces the conv input-A
    pTensor reluIn        = createTensor({128, 128}, syn_type_float);
    pTensor reluOutConvIn = createTensor({128, 128}, syn_type_float);
    pTensor convWgh       = createTensor({128, 128}, syn_type_float);
    pTensor convOut       = createTensor({128, 128}, syn_type_float);
    pBundle bundle = createSingleMMENodeBundle({reluOutConvIn, convWgh},
                                               {convOut},
                                               NodeFactory::convolutionNodeTypeName);
    pNode relu = NodeFactory::createGenericTPCNode({reluIn}, {reluOutConvIn}, nullptr, "relu_fwd_f32", "relu");

    pNode conv = bundle->getNodes().front();
    pMmeSlicingStrategy slicingStrategy = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(getMmeBrain().getSolutionStrategies(bundle), bundle, getGraph(), getSlicingBrains()));

    pBundleExpansion expCnd = std::make_shared<BundleExpansion>();
    expCnd->nodeToStitch = relu;

    slicingStrategy->getMmeSlicingData().masterOperand->chunkDimensions[DIM_W]        = 32;
    slicingStrategy->getMmeSlicingData().masterOperand->chunkDimensions[WEIGHT_DIM_K] = 64;
    auto& bundleSlicedTensors = slicingStrategy->getMmeSlicingData().bundleTensors;
    for (auto& slicedTensor : bundleSlicedTensors)
    {
        if (slicedTensor->originalTensor == reluOutConvIn)
        {
            slicedTensor->chunkDimensions[DIM_W] = 32;
            expCnd->stitchedOperand = slicedTensor;
            slicedTensor->numOfBuffers = 2;
        }
        if (slicedTensor->originalTensor == convWgh)
        {
            slicedTensor->chunkDimensions[WEIGHT_DIM_K] = 64;
        }
    }

    GraphEditor::addNode(getGraph(), relu);
    bundle->addNode(relu);

    // When
    TPCSlaveBrain tpcSlaveBrain(getGraph());
    tpcSlaveBrain.addProducerToStrategy(expCnd, slicingStrategy);
    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, slicingStrategy).fillSolution());

    // Then
    Solution& solution = bundle->getSolution();
    checkSolutionSize(solution, 4, 12);
    checkChunkSize(solution, reluIn, {128, 32});

    auto opIter = solution.operations.begin();

    // Initial TPC activation
    const auto& expTpcOpPre = *opIter++;
    ASSERT_EQ(expTpcOpPre.originalNode, relu);
    ASSERT_EQ(expTpcOpPre.inputs[0]->coordinates[DIM_W], 0);
    ASSERT_EQ(expTpcOpPre.outputs[0]->coordinates[DIM_W], 0);

    // Interleaved TPC, MME activations
    for (unsigned tpcOpIdx = 1; tpcOpIdx < 4; tpcOpIdx++)
    {
        const auto& expTpcOp = *opIter++;
        const auto& expMmeOp1 = *opIter++;
        const auto& expMmeOp2 = *opIter++;

        ASSERT_EQ(expTpcOp.originalNode, relu);
        ASSERT_EQ(expMmeOp1.originalNode, conv);
        ASSERT_EQ(expMmeOp2.originalNode, conv);

        ASSERT_EQ(expTpcOp.inputs[0]->coordinates[DIM_W], tpcOpIdx);
        ASSERT_EQ(expTpcOp.outputs[0]->coordinates[DIM_W], tpcOpIdx);
    }

    // Remainder MME activations
    const auto& expMmeOp1Post = *opIter++;
    const auto& expMmeOp2Post = *opIter++;

    ASSERT_EQ(expMmeOp1Post.originalNode, conv);
    ASSERT_EQ(expMmeOp2Post.originalNode, conv);

}

TEST_F(SRAMManagementTest, tpc_slave_brain_should_add_tpc_separable_consumer_to_solution)
{
    // Given conv -> relu
    pTensor convOutReluIn = createTensor({32, 28*28*64}, syn_type_float);
    pTensor reluOut       = createTensor({32, 28*28*64}, syn_type_float);
    pTensor convWgh       = createTensor({32,       16}, syn_type_float);
    pTensor convIn        = createTensor({16, 28*28*64}, syn_type_float);
    synConvolutionParams params {};
    pNode conv = NodeFactory::createNode({convIn, convWgh}, {convOutReluIn}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    pNode relu = NodeFactory::createGenericTPCNode({convOutReluIn}, {reluOut}, nullptr, "relu_fwd_f32", "relu");

    GraphEditor::addNode(getGraph(), conv);
    GraphEditor::addNode(getGraph(), relu);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    Bundlizer bundlizer {getGraph()};
    pBundle bundle = bundlizer.getMMEBundles().front();
    SlicingBrain::knobs.maxNarrowSliceSize = 1024;
    SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon2D = 1024;
    pMmeSlicingStrategy strategy                           = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(getMmeBrain().getSolutionStrategies(bundle), bundle, getGraph(), getSlicingBrains()));

    // When
    pBundleExpansion expCand = bundlizer.findTpcConsumerExpansionCandidate(strategy);

    // Then
    ASSERT_EQ(expCand->nodeToStitch, relu);
    ASSERT_EQ(expCand->stitchedOperand, strategy->getMmeSlicingData().masterOperand);
    ASSERT_EQ(expCand->stitchedOperand->originalTensor, convOutReluIn);

    // And when adding the consumer to the strategy
    TPCSlaveBrain(getGraph()).addConsumerToStrategy(expCand, strategy);

    // Then the relu output should be added to the strategy and sliced.
    // relu input should not be added to the strategy a second time.
    const auto& bundleTensors = strategy->getMmeSlicingData().bundleTensors;
    bool reluOutFound = false;
    for (const pSlicedOperand& slicedOp : bundleTensors)
    {
        if (slicedOp->originalTensor == reluOut)
        {
            reluOutFound = true;
            ASSERT_EQ(slicedOp->chunkDimensions, strategy->getMmeSlicingData().masterOperand->chunkDimensions);
        }
        ASSERT_NE(slicedOp->originalTensor, convOutReluIn) << "Expected convOutReluIn to only be referenced by the "
                                                           << "masterOperand of the strategy";
    }
    ASSERT_TRUE(reluOutFound);

    ASSERT_TRUE(strategy->getMmeSlicingData().masterOperand->resideInSRAM) << "masterOperand is expected to reside in"
                                                                        << " SRAM after adding TPC consumer";
}

TEST_F(SRAMManagementTest, tpc_slave_brain_should_add_bn2_producer_to_solution)
{
    // TODO
}

TEST_F(SRAMManagementTest, tpc_slave_brain_should_add_bn1_bwd_producer_to_solution)
{
    // TODO
}

TEST_F(SRAMManagementTest, tpc_slice_mapper_should_map_elementwise_output_slices_to_input_slices)
{
    // Given
    pNode relu = NodeFactory::createGenericTPCNode({createTensor({128, 256}, syn_type_bf16),  // "element-wise operand"
                                                    createTensor({128, 1}, syn_type_bf16)},   // "Bx operand"
                                                   {createTensor({128, 256}, syn_type_bf16)},
                                                   nullptr,
                                                   NOP_KERNEL_NAME);
    pSlicedOperand slicedEwIn = std::make_shared<SlicedOperand>(relu->getInput(0));
    pSlicedOperand slicedBxIn = std::make_shared<SlicedOperand>(relu->getInput(1));
    pSlicedOperand slicedOut = std::make_shared<SlicedOperand>(relu->getOutput(0));
    slicedOut->chunkDimensions[0] = slicedEwIn->chunkDimensions[0] = slicedBxIn->chunkDimensions[0] = 32;
    slicedOut->chunkDimensions[1] = slicedEwIn->chunkDimensions[1] = 128;

    // When
    pBackwardSliceMapping mapping = TPCSliceMapper::mapOutputToInputs(relu, {slicedEwIn, slicedBxIn}, slicedOut);

    // Then
    ASSERT_NE(mapping, nullptr);
    unsigned nofSlices = 0;
    for (auto outputSlice : SlicedOperandTraversalPattern(slicedOut, {0, 1}))
    {
        nofSlices++;
        auto inputSlices = mapping->getInputs(outputSlice);
        pSliceReference ewSlice = inputSlices.front();
        pSliceReference bxSlice = inputSlices.back();
        ASSERT_EQ(ewSlice->coordinates, outputSlice.first->coordinates);
        ASSERT_EQ(bxSlice->coordinates[0], outputSlice.first->coordinates[0]);
        ASSERT_EQ(bxSlice->coordinates[1], 0);
    }
    ASSERT_EQ(nofSlices, (128/32) * (256/128));
}

TEST_F(SRAMManagementTest, tpc_slice_mapper_should_map_tpc_element_wise_forward)
{
    pNode tpc = NodeFactory::createGenericTPCNode({createTensor({128, 256}, syn_type_bf16),  // "element-wise operand"
                                                   createTensor({128, 1}, syn_type_bf16)},   // "Bx operand"
                                                  {createTensor({128, 256}, syn_type_bf16)},
                                                  nullptr,
                                                  NOP_KERNEL_NAME);
    pSlicedOperand slicedEwIn = std::make_shared<SlicedOperand>(tpc->getInput(0));
    pSlicedOperand slicedBxIn = std::make_shared<SlicedOperand>(tpc->getInput(1));
    pSlicedOperand slicedOut = std::make_shared<SlicedOperand>(tpc->getOutput(0));
    slicedOut->chunkDimensions[0] = slicedEwIn->chunkDimensions[0] = slicedBxIn->chunkDimensions[0] = 32;
    slicedOut->chunkDimensions[1] = slicedEwIn->chunkDimensions[1] = 128;

    // When
    pForwardSliceMapping mapping = TrivialSliceMapper::mapSlicedOperandForward({slicedEwIn, slicedBxIn}, {slicedOut});

    // Then
    ASSERT_NE(mapping, nullptr);
    unsigned nofSlices = 0;
    for (auto inputSlice : SlicedOperandTraversalPattern(slicedEwIn, {0, 1}))
    {
        nofSlices++;
        SliceReferenceList inputs, outputs;
        std::tie(inputs, outputs) = mapping->getInputsAndOutputs(inputSlice.first).front();
        ASSERT_EQ(inputs.size(), 2);
        pSliceReference ewSlice = inputs.front();
        ASSERT_EQ(ewSlice, inputSlice.first);
        pSliceReference bxSlice = inputs.back();
        ASSERT_EQ(outputs.size(), 1);
        pSliceReference outputSlice = outputs.front();
        ASSERT_EQ(ewSlice->coordinates, outputSlice->coordinates);
        ASSERT_EQ(bxSlice->coordinates[0], outputSlice->coordinates[0]);
        ASSERT_EQ(bxSlice->coordinates[1], 0);
    }
    ASSERT_EQ(nofSlices, (128/32) * (256/128));
}

// relu(tpc)->reshape->conv(mme)
// trivial solution - no slicing
// reshape node should be a part of the bundle and marked in SRAM
TEST_F(SRAMManagementTest, reshape_brain_tpc_producer)
{
    pTensor reluIn           = createTensor({4, 16}, syn_type_float);
    pTensor reluOutReshapeIn = createTensor({4, 16}, syn_type_float);
    pTensor reshapeOutConvIn = createTensor({8, 8}, syn_type_float);
    pTensor convWgh          = createTensor({7, 7}, syn_type_float);
    pTensor convOut          = createTensor({2, 2}, syn_type_float);

    synConvolutionParams params;
    pNode convNode = NodeFactory::createNode({reshapeOutConvIn, convWgh}, {convOut}, &params,
                                             NodeFactory::convolutionNodeTypeName, "conv");
    GraphEditor::addNode(getGraph(), convNode);

    Bundlizer bundlizer(getGraph());
    BundleList bundles = bundlizer.getMMEBundles();
    ASSERT_EQ(bundles.size(), 1);
    pBundle bundle = bundles.front();
    TrivialSolver solver(*getGraph().getHALReader(), bundle);
    solver.createAllStrategies();
    pMmeSlicingStrategy slicingStrategy = std::static_pointer_cast<MmeSlicingStrategy>(solver.getStrategies().front());

    pNode reluNode = NodeFactory::createGenericTPCNode({reluIn}, {reluOutReshapeIn}, nullptr, "relu_fwd_f32", "relu");
    pNode reshapeNode = NodeFactory::createNode({reluOutReshapeIn}, {reshapeOutConvIn}, nullptr,
                                                NodeFactory::reshapeNodeTypeName, "reshape");
    GraphEditor::addNode(getGraph(), reshapeNode);
    GraphEditor::addNode(getGraph(), reluNode);

    pBundleExpansion expCnd = bundlizer.findWideTpcProducerExpansionCandidate(slicingStrategy);

    // assert expansion candidate
    ASSERT_EQ(expCnd->nodeToStitch, reluNode);
    ASSERT_EQ(expCnd->reshapeNode, reshapeNode);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, reshapeOutConvIn);

    bundlizer.addCandidateToBundle(bundle, expCnd);
    ASSERT_EQ(bundle->getNodes().size(), 3); // both relu and reshape nodes should be added to the conv bundle now

    ReshapeSlicingBrain reshapeBrain(getGraph());
    reshapeBrain.addProducerToStrategy(expCnd, slicingStrategy);
    // after the previous line expCnd->stitchedOperand should be changed to the tensor between the relu and reshape node
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, reluOutReshapeIn);

    TPCSlaveBrain tpcSlaveBrain(getGraph());
    tpcSlaveBrain.addProducerToStrategy(expCnd, slicingStrategy);

    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, slicingStrategy).fillSolution());

    Solution& solution = bundle->getSolution();

    // validate solution
    checkSolutionSize(solution, 5, 3);
    auto opIter = solution.operations.begin();
    const auto& expectedTpcOp = *opIter++;
    const auto& expectedReshapeOp = *opIter++;
    const auto& expectedMmeOp = *opIter;
    ASSERT_EQ(expectedTpcOp.originalNode, reluNode);
    ASSERT_EQ(expectedReshapeOp.originalNode, reshapeNode);
    ASSERT_EQ(expectedMmeOp.originalNode, convNode);

    // trivial slicing
    ASSERT_TRUE(SlicedOperandUtils::isTriviallySliced(expectedMmeOp.inputs[0]->operand));
    // trivial slicing  - only 1 slice
    ASSERT_EQ(expectedTpcOp.outputs.size(), 1);
    ASSERT_EQ(expectedReshapeOp.inputs.size(), 1);
    ASSERT_EQ(expectedReshapeOp.outputs.size(), 1);

    ASSERT_TRUE(expectedTpcOp.outputs[0]->operand->resideInSRAM);
    ASSERT_TRUE(expectedReshapeOp.inputs[0]->operand->resideInSRAM);
    ASSERT_TRUE(expectedReshapeOp.outputs[0]->operand->resideInSRAM);
    ASSERT_TRUE(expectedMmeOp.inputs[0]->operand->resideInSRAM);

    for (auto iterOperation = solution.operations.begin(); iterOperation != solution.operations.end(); ++iterOperation)
    {   // all nodes should have 0 coordinates = trivial slicing
        for (unsigned coord : iterOperation->inputs[0]->coordinates)
        {
            ASSERT_EQ(0, coord);
        }
        for (unsigned coord :iterOperation->outputs[0]->coordinates)
        {
            ASSERT_EQ(0, coord);
        }
    }
}

// conv(mme)->reshape->relu(tpc)
// trivial solution - no slicing
// reshape node should be a part of the bundle and marked in SRAM
TEST_F(SRAMManagementTest, reshape_brain_tpc_consumer)
{
    pTensor convIn           = createTensor({8, 8}, syn_type_float);
    pTensor convWgh          = createTensor({7, 7}, syn_type_float);
    pTensor convOutReshapeIn = createTensor({2, 2}, syn_type_float);
    pTensor reshapeOutReluIn = createTensor({4, 1}, syn_type_float);
    pTensor reluOut          = createTensor({4, 1}, syn_type_float);

    synConvolutionParams params;
    pNode convNode = NodeFactory::createNode({convIn, convWgh}, {convOutReshapeIn}, &params,
                                             NodeFactory::convolutionNodeTypeName, "conv");
    GraphEditor::addNode(getGraph(), convNode);

    Bundlizer bundlizer(getGraph());
    BundleList bundles = bundlizer.getMMEBundles();
    ASSERT_EQ(bundles.size(), 1);
    pBundle bundle = bundles.front();
    pMmeSlicingStrategy slicingStrategy = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(getMmeBrain().getSolutionStrategies(bundle), bundle, getGraph(), getSlicingBrains()));

    pNode reshapeNode = NodeFactory::createNode({convOutReshapeIn}, {reshapeOutReluIn}, nullptr,
                                                NodeFactory::reshapeNodeTypeName, "reshape");
    pNode reluNode = NodeFactory::createGenericTPCNode({reshapeOutReluIn}, {reluOut}, nullptr, "relu_fwd_f32", "relu");

    GraphEditor::addNode(getGraph(), reshapeNode);
    GraphEditor::addNode(getGraph(), reluNode);
    pBundleExpansion expCnd = bundlizer.findTpcConsumerExpansionCandidate(slicingStrategy);

    // assert expansion candidate
    ASSERT_EQ(expCnd->nodeToStitch, reluNode);
    ASSERT_EQ(expCnd->reshapeNode, reshapeNode);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, convOutReshapeIn);

    bundlizer.addCandidateToBundle(bundle, expCnd);
    ASSERT_EQ(bundle->getNodes().size(), 3); // both relu and reshape nodes should be added to the conv bundle now

    ReshapeSlicingBrain reshapeBrain(getGraph());
    reshapeBrain.addConsumerToStrategy(expCnd, slicingStrategy);
    // after the line expCnd->stitchedOperand should be changed to the tensor between the reshape node and relu
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, reshapeOutReluIn);

    TPCSlaveBrain tpcSlaveBrain(getGraph());
    tpcSlaveBrain.addConsumerToStrategy(expCnd, slicingStrategy);

    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, slicingStrategy).fillSolution());

    Solution& solution = bundle->getSolution();

    // validate solution
    checkSolutionSize(solution, 5, 3);
    auto opIter = solution.operations.begin();
    const auto& expectedMmeOp = *opIter++;
    const auto& expectedReshapeOp = *opIter++;
    const auto& expectedTpcOp = *opIter;

    ASSERT_EQ(expectedMmeOp.originalNode, convNode);
    ASSERT_EQ(expectedReshapeOp.originalNode, reshapeNode);
    ASSERT_EQ(expectedTpcOp.originalNode, reluNode);

    // trivial slicing
    ASSERT_TRUE(SlicedOperandUtils::isTriviallySliced(expectedMmeOp.outputs[0]->operand));
    // trivial slicing  - only 1 slice
    ASSERT_EQ(expectedReshapeOp.inputs.size(), 1);
    ASSERT_EQ(expectedReshapeOp.outputs.size(), 1);
    ASSERT_EQ(expectedTpcOp.inputs.size(), 1);

    ASSERT_TRUE(expectedMmeOp.outputs[0]->operand->resideInSRAM);
    ASSERT_TRUE(expectedReshapeOp.inputs[0]->operand->resideInSRAM);
    ASSERT_TRUE(expectedReshapeOp.outputs[0]->operand->resideInSRAM);
    ASSERT_TRUE(expectedTpcOp.inputs[0]->operand->resideInSRAM);

    for (auto iterOperation = solution.operations.begin(); iterOperation != solution.operations.end(); ++iterOperation)
    {   // all nodes should have 0 coordinates = trivial slicing
        for (unsigned coord : iterOperation->inputs[0]->coordinates)
        {
            ASSERT_EQ(0, coord);
        }
        for (unsigned coord :iterOperation->outputs[0]->coordinates)
        {
            ASSERT_EQ(0, coord);
        }
    }
}

TEST_F(SRAMManagementTest, noncommondim2dsolver_should_slice_only_big_input)
{
    // IFM size (64 * 5)x2 bfloats,
    // WGH size 2x4 bfloats,
    // SRAM can fit the WGH and a 256x2 slice out of the IFM
    // Expects WGH to stay in SRAM while 256x2 chunk + 64x2 chunk from IFM are brought to for compute (single-buffer)

    for (synDataType dtype : {syn_type_bf16, syn_type_float})
    {
        const TSize expSlices  = 3;
        TSize       ifmsizes[] = {2, expSlices * 128, 1, 1};
        const TSize wsizes[] = {4, 2};
        TSize outsizes[] = {4, expSlices * 128, 1, 1};

        pTensor ifm(new Tensor(4, ifmsizes, dtype));
        pTensor wgh(new Tensor(2, wsizes, dtype));
        pTensor ofm(new Tensor(4, outsizes, dtype));

        const uint64_t ifmLineSizeInBytes = 2 * ifm->getElementSizeInBytes();
        const uint64_t weightSize         = wgh->getDenseSizeInBytes();

        SlicingBrain::knobs.maxSRAMCapInBytes =
            2 * (weightSize + GaudiHalReader::instance(synDeviceGaudi)->getMmeVectorSize() * ifmLineSizeInBytes);

        pBundle bundle = createSingleMMENodeBundle({ifm, wgh, nullptr, nullptr}, {ofm}, NodeFactory::convolutionNodeTypeName);

        NonCD2DSolver solver(*getGraph().getHALReader(), bundle);
        solver.createAllStrategies();
        const SlicingStrategyPtr& winningStrategy =
            findWinningStrategy(solver.getStrategies(), bundle, getGraph(), getSlicingBrains());
        SLC_DEBUG("Winning Strategy - ");
        winningStrategy->printLog(1, synapse::LogManager::LogType::SRAM_SLICE);
        SolutionGenerator(getGraph(), bundle, winningStrategy).fillSolution();
        Solution& solution = bundle->getSolution();
        checkSolutionSize(solution, 3, expSlices);
        checkChunkSize(solution, ifm, {2, 128, 1, 1});
        checkChunkSize(solution, wgh, {4, 2});
    }
}

TEST_F(SRAMManagementTest, non_common_dim_4d_solver)
{
    /* Tests a simple test slicing of non common 4d solver */
    synDataType dtype = syn_type_float;

    const TSize b = 10, h = 5, w = 5, c = 5;
    const TSize r = 3, s = 3, k = 512;
    TSize       ifmsizes[] = {c, w, h, b};
    const TSize wsizes[]   = {k, c, s, r};
    TSize       ofmsizes[] = {k, w - s + 1, h - s + 1, b};

    const unsigned expBatch = 3;
    const unsigned expK = 32;

    pTensor ifm(new Tensor(4, ifmsizes, dtype));
    pTensor wgh(new Tensor(4, wsizes,   dtype));
    pTensor ofm(new Tensor(4, ofmsizes, dtype));

    const uint64_t weightSize = wgh->getDenseSizeInBytes();
    const uint64_t ifmSize    = ifm->getDenseSizeInBytes();

    SlicingBrain::knobs.maxSRAMCapInBytes =  2 * (expK * weightSize/k + expBatch * ifmSize/b);

    pBundle bundle = createSingleMMENodeBundle({ifm, wgh, nullptr, nullptr}, {ofm}, NodeFactory::convolutionNodeTypeName);

    NonCommon4DSolver solver(*getGraph().getHALReader(), bundle);
    ASSERT_TRUE(solver.effectiveForBundle());
    solver.createAllStrategies();

    const pMmeSlicingStrategy& winningStrategy = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(solver.getStrategies(), bundle, getGraph(), getSlicingBrains()));

    SLC_DEBUG("Winning Strategy - ");
    winningStrategy->printLog(1, synapse::LogManager::LogType::SRAM_SLICE);

    auto& data = winningStrategy->getMmeSlicingData();
    SolutionGenerator(getGraph(), bundle, winningStrategy).fillSolution();
    Solution& solution = bundle->getSolution();
    checkChunkSize(solution, wgh, {expK, c, s, r});
    checkChunkSize(solution, ifm, {c, w, h, expBatch});
    checkChunkSize(solution, ofm, {expK, ofmsizes[1], ofmsizes[2], expBatch});

    ASSERT_EQ(winningStrategy->getMetrics().SRAMCapacity, SlicingBrain::knobs.maxSRAMCapInBytes);
    ASSERT_EQ(data.MMEGeometryUsed, gaudi_geometry_1wx4h);
}

TEST_F(SRAMManagementTest, dim_controller_2d_tensor)
{
    TSize              ifmsizes[] = {2, 64};
    const TSize        wsizes[]   = {4, 2};
    constexpr synDataType dtype      = syn_type_bf16;

    pTensor ifm(new Tensor(2, ifmsizes, dtype));
    pTensor wgh(new Tensor(2, wsizes, dtype));
    pTensor ofm(new Tensor(2, ifmsizes, dtype));

    synConvolutionParams params;
    pNode node = NodeFactory::createNode({ifm, wgh},{ofm}, &params,
                                         NodeFactory::convolutionNodeTypeName,"conv");
    MmeDimController     controller(node);
    ASSERT_EQ(controller.nonCommonDimOperandA().size(), 1);
    ASSERT_EQ(controller.nonCommonDimOperandB().size(), 1);
    ASSERT_EQ(controller.nonCommonDimOperandA().front(), DIM_W);
    ASSERT_EQ(controller.nonCommonDimOperandB().front(), WEIGHT_DIM_K);
    ASSERT_EQ(controller.commonDimOperandA().front(), DIM_C);
    ASSERT_EQ(controller.commonDimOperandB().front(), WEIGHT_DIM_C);
}

TEST_F(SRAMManagementTest, graph_should_stay_the_same_when_no_slicing_is_done)
{
    TSize hChunks =3, kChunks = 3, inCD = 2;
    const TSize chunkSize = 32;
    const TSize OUT_H = chunkSize * hChunks;
    const TSize OUT_K = chunkSize * kChunks;
    // set sram cap to 1 to ensure nothing fits in sram
    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setValue(1);

    TSize opASizes[] = {1, 1, OUT_H, inCD};
    pTensor opA(new Tensor(4U, opASizes, syn_type_float));
    opA->setName("opA");
    TSize opBSizes[] = {inCD, OUT_K, 1, 1};
    pTensor opB(new Tensor(4U, opBSizes, syn_type_float));
    opB->setName("opB");
    TSize opOutSizes[] = {1, 1, OUT_H, OUT_K};
    pTensor opOut(new Tensor(4U, opOutSizes, syn_type_float));
    TensorVector         inputs {opA, opB};
    TensorVector         outputs {opOut};
    synConvolutionParams convParams;
    pNode node = NodeFactory::createNode(inputs, outputs, &convParams,
                                         NodeFactory::convolutionNodeTypeName, "conv");
    GraphEditor::addNode(getGraph(), node);
    unsigned numOfNodesBefore = getGraph().getNodes().size();
    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
    ASSERT_EQ(getGraph().getNodes().size(), numOfNodesBefore);
}

TEST_F(SRAMManagementTest, 2D_solver_entire_pass_test)
{
    TSize hChunks =3, kChunks = 3, inCD = 2;
    const TSize chunkSize = 64;
    const TSize OUT_H = chunkSize * hChunks;
    const TSize OUT_K = chunkSize * kChunks;
    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setValue(2 * 2 * chunkSize * 4 * inCD);
    TSize opASizes[] = {inCD, OUT_H};
    pTensor opA(new Tensor(2U, opASizes, syn_type_float));
    opA->setName("opA");
    TSize opBSizes[] = {OUT_K, inCD};
    pTensor opB(new Tensor(2U, opBSizes, syn_type_float));
    opB->setName("opB");
    TSize opOutSizes[] = {OUT_K, OUT_H};
    pTensor opOut(new Tensor(2U, opOutSizes, syn_type_float));
    TensorVector         inputs {opA, opB};
    TensorVector         outputs {opOut};
    synConvolutionParams convParams;
    pNode node = NodeFactory::createNode(inputs, outputs, &convParams,
                                         NodeFactory::convolutionNodeTypeName, "conv");
    GraphEditor::addNode(getGraph(), node);
    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
    // slicing should add nodes to graph!
    ASSERT_GT(getGraph().getNodes().size(), 3);
}

TEST_F(SRAMManagementTest, bundlizer_should_not_add_same_node_to_different_bundles)
{
    // Given graph TPC --+--> MME
    //                   |
    //                   +--> MME
    pTensor reluIn       = createTensor({2, 2}, syn_type_float);
    pTensor reluOutMmeIn = createTensor({2, 2}, syn_type_float);
    pTensor wgh          = createTensor({2, 2}, syn_type_float);
    pTensor mmeOut1      = createTensor({2, 2}, syn_type_float);
    pTensor mmeOut2      = createTensor({2, 2}, syn_type_float);

    synGEMMParams params{};
    pNode gemm1 = NodeFactory::createNode({reluOutMmeIn, wgh},
                                          {mmeOut1},
                                          &params,
                                          NodeFactory::gemmNodeTypeName,
                                          "gemm1");
    GraphEditor::addNode(getGraph(), gemm1);
    pNode gemm2 = NodeFactory::createNode({reluOutMmeIn, wgh},
                                          {mmeOut2},
                                          &params,
                                          NodeFactory::gemmNodeTypeName,
                                          "gemm2");
    GraphEditor::addNode(getGraph(), gemm2);

    pNode relu = NodeFactory::createGenericTPCNode({reluIn}, {reluOutMmeIn}, nullptr, "relu_fwd_f32", "relu");
    GraphEditor::addNode(getGraph(), relu);

    Bundlizer bundlizer(getGraph());
    BundleList bundles = bundlizer.getMMEBundles();
    pBundle bundle1 = bundles.front();
    pBundle bundle2 = bundles.back();

    // When the slicing strategy is trivial
    pMmeSlicingStrategy strategy1 =
        MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle1->getNodes().front());
    pMmeSlicingStrategy strategy2 =
        MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle2->getNodes().front());

    pBundleExpansion expCnd1 = bundlizer.findWideTpcProducerExpansionCandidate(strategy1);
    // Then the bundlizer adds the relu to the bundle
    ASSERT_EQ(expCnd1->nodeToStitch, relu);
    ASSERT_FALSE(expCnd1->reshapeNode);
    ASSERT_EQ(expCnd1->stitchedOperand->originalTensor, reluOutMmeIn);

    bundlizer.addCandidateToBundle(bundle1, expCnd1);

    pBundleExpansion expCnd2 = bundlizer.findWideTpcProducerExpansionCandidate(strategy2);
    ASSERT_FALSE(expCnd2->nodeToStitch);
}


TEST_F(SRAMManagementTest, bundlizer_should_not_jump_over_bundles_while_expanding)
{
    // Given graph TPC -> MME -> MME
    pTensor relu_in         = createTensor({2, 2}, syn_type_float);
    pTensor relu_out_mme_in = createTensor({2, 2}, syn_type_float);
    pTensor wgh             = createTensor({2, 2}, syn_type_float);
    pTensor mme_out_0       = createTensor({2, 2}, syn_type_float);
    pTensor mme_out_1       = createTensor({2, 2}, syn_type_float);

    synGEMMParams params{};
    pNode mme1 = NodeFactory::createNode({relu_out_mme_in, wgh},
                                         {mme_out_0},
                                         &params,
                                         NodeFactory::gemmNodeTypeName,
                                         "mme1");
    GraphEditor::addNode(getGraph(), mme1);

    pNode mme2 = NodeFactory::createNode({relu_out_mme_in, mme_out_0},
                                         {mme_out_1},
                                         &params,
                                         NodeFactory::gemmNodeTypeName,
                                         "mme2");
    GraphEditor::addNode(getGraph(), mme2);

    pNode relu = NodeFactory::createGenericTPCNode({relu_in}, {relu_out_mme_in}, nullptr, "relu_fwd_f32", "relu");
    GraphEditor::addNode(getGraph(), relu);

    Bundlizer bundlizer(getGraph());
    BundleList bundles = bundlizer.getMMEBundles();
    pBundle mme2Bundle = bundles.back();

    pMmeSlicingStrategy strategy =
        MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), mme2Bundle->getNodes().front());

    // When
    ASSERT_EQ(mme2Bundle->getNodes().front(), mme2);
    pBundleExpansion expCnd = bundlizer.findWideTpcProducerExpansionCandidate(strategy);

    // Then the bundlizer does not add the relu to the bundle
    ASSERT_FALSE(BundleExpander::validateCandidatePaths(getGraph(),
                                                        expCnd,
                                                        strategy->getMmeSlicingData().getStrategyNodes(mme2Bundle),
                                                        strategy->getMmeSlicingData().getStrategyProducers()));
}

TEST_F(SRAMManagementTest, bundlizer_should_not_add_non_separable_tpc_nodes_to_sliced_mme_bundle)
{
    // Given
    pTensor avg_pool_in         = createTensor({1, 256, 256, 1}, syn_type_float);
    pTensor avg_pool_out_mme_in = createTensor({1, 128, 128, 1}, syn_type_float);
    pTensor wgh                 = createTensor({1,   1,   1, 1}, syn_type_float);
    pTensor mme_out             = createTensor({1, 128, 128, 1}, syn_type_float);

    pBundle bundle = createSingleMMENodeBundle({avg_pool_out_mme_in, wgh},
                                               {mme_out}, NodeFactory::convolutionNodeTypeName);

    ns_AveragePooling::Params params{};
    std::memset(&params, 0, sizeof(params));
    params.kernel_h   = 2;
    params.kernel_w   = 2;
    params.stride_h   = 2;
    params.stride_w   = 2;
    params.dilation_h = 1;
    params.dilation_w = 1;
    params.pooling_convention = POOLING_CONVENTION_VALID;
    params.includePadding     = false;
    pNode avgpool = NodeFactory::createGenericTPCNode({avg_pool_in}, {avg_pool_out_mme_in}, &params,
                                                      "avg_pool_2d_fwd_f32", "avgpool");

    ASSERT_TRUE(GraphEditor::addNode(getGraph(), avgpool));

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    pMmeSlicingStrategy strategy =
        MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle->getNodes().front());

    strategy->getMmeSlicingData().bundleTensors[0]->chunkDimensions[DIM_W] =
        67;  // pick a number that is not a multiple of the granularity

    Bundlizer        bundlizer(getGraph());
    pBundleExpansion expCnd = bundlizer.findWideTpcProducerExpansionCandidate(strategy);
    ASSERT_FALSE(expCnd->nodeToStitch);

    // Check consumer isn't added either.
    pTensor mme_prod_in         = createTensor({1, 256, 256, 1}, syn_type_float);
    bundle = createSingleMMENodeBundle({mme_prod_in, wgh},
        {avg_pool_in}, NodeFactory::convolutionNodeTypeName);
    strategy = MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle->getNodes().front());
    strategy->getMmeSlicingData().masterOperand->chunkDimensions[DIM_W] =
        67;  // pick a number that is not a multiple of the granularity

    expCnd = bundlizer.findTpcConsumerExpansionCandidate(strategy);
    ASSERT_FALSE(expCnd->nodeToStitch);

}

TEST_F(SRAMManagementTest, slave_producer_consumer_candidates_detection)
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_SHARED_MME_INPUT_PRODUCER_EXPANSION_ENABLED, "true");
    setGlobalConfForTest(GCFG_SRAM_SLICER_SHARED_MME_INPUT_CONSUMER_EXPANSION_ENABLED, "true");

    GaudiGraph g;
    const TSize b1 = 1, h1 = 1, w1 = 6272, c1 = 512;
    const TSize r = 1, s = 1, k = 256;

    pTensor convIn    = createTensor({c1, w1, h1, b1}, syn_type_bf16);
    pTensor convWgh   = createTensor({k, c1, s, r}, syn_type_bf16);
    pTensor relu3In   = createTensor({c1, w1, h1, b1}, syn_type_bf16);
    pTensor convWgh2  = createTensor({k, c1, s, r}, syn_type_bf16);
    pTensor relu2In   = createTensor({k, c1, s, r}, syn_type_bf16);
    pTensor convOut   = createTensor({k, w1 - s + 1, h1 - s + 1, b1}, syn_type_bf16);
    pTensor convOut2  = createTensor({k, w1 - s + 1, h1 - s + 1, b1}, syn_type_bf16);
    pTensor consumer1 = createTensor({k, w1 - s + 1, h1 - s + 1, b1}, syn_type_bf16);
    pTensor consumer2 = createTensor({k, w1 - s + 1, h1 - s + 1, b1}, syn_type_bf16);


    pNode   relu3Node    = NodeFactory::createGenericTPCNode({relu3In}, {convIn}, nullptr, "relu_fwd_bf16", "relu3prod");
    ASSERT_TRUE(GraphEditor::addNode(g, relu3Node));

    synConvolutionParams convParams;
    pNode convNode  = NodeFactory::createNode({convIn, convWgh},  {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv1");
    ASSERT_TRUE(GraphEditor::addNode(g, convNode));

    pNode   relu2Node    = NodeFactory::createGenericTPCNode({relu2In}, {convWgh2}, nullptr, "relu_fwd_bf16", "relu2prod");
    ASSERT_TRUE(GraphEditor::addNode(g, relu2Node));

    pNode convNode2 = NodeFactory::createNode({convIn, convWgh2}, {convOut2}, &convParams, NodeFactory::convolutionNodeTypeName, "conv2");
    ASSERT_TRUE(GraphEditor::addNode(g, convNode2));

    pNode   relu1ConsumerNode    = NodeFactory::createGenericTPCNode({convOut}, {consumer1}, nullptr, "relu_fwd_bf16", "relu1cons");
    ASSERT_TRUE(GraphEditor::addNode(g, relu1ConsumerNode));

    pNode   relu2ConsumerNode    = NodeFactory::createGenericTPCNode({convOut2}, {consumer2}, nullptr, "relu_fwd_bf16", "relu2cons");
    ASSERT_TRUE(GraphEditor::addNode(g, relu2ConsumerNode));

    ASSERT_TRUE(loadTpcKernels(g));

    std::unordered_map<pBundle, BundleSolvingData> solvingDataPerBundle;
    Bundlizer  bundlizer(g);
    BundleList bundles = bundlizer.getMMEBundles();
    AllBrains  brains(g);

    for (pBundle& bundle : bundles)
    {
        solvingDataPerBundle[bundle].strategies = brains.m_mmeBrain.getSolutionStrategies(bundle);
    }

    graphVisualizationPre(g);

    BundleExpander expander(g, brains, bundlizer, solvingDataPerBundle);

    LOG_DEBUG(SRAM_SLICE, "Looking for candidates for Bundle: {}", bundles.back()->getNodes().back()->getNodeName());

    std::list<pBundleExpansion> candidates = expander.discoverExpansionCandidatesForBundle(bundles.back(),
                                                                                           {BundleExpansion::NarrowInputProducer,
                                                                                            BundleExpansion::OutputConsumer,
                                                                                            BundleExpansion::SharedInputConsumer});
    unsigned independentCandidates = candidates.size();
    unsigned dependentCandidates   = 0;

    /* InputProducer, OutputConsumer, SharedInputConsumer */
    ASSERT_EQ(3, independentCandidates) << "The expander didn't detect all independent candidates ";

    for (const pBundleExpansion& candidate: candidates)
    {
        dependentCandidates += candidate->dependentCandidates.size();
    }

    /* SlaveOutputConsumer */
    ASSERT_EQ(2, dependentCandidates) << "The expander didn't detect all dependent candidates ";
}


TEST_F(SRAMManagementTest, slave_producer_consumer_candidates_strategy_expension)
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_SHARED_MME_INPUT_PRODUCER_EXPANSION_ENABLED, "true");
    setGlobalConfForTest(GCFG_SRAM_SLICER_SHARED_MME_INPUT_CONSUMER_EXPANSION_ENABLED, "true");

    GaudiGraph g;
    const TSize b1 = 1, h1 = 1, w1 = 6272, c1 = 512;
    const TSize r = 1, s = 1, k = 256;
    synConvolutionParams convParams;
    convParams.kH = r;
    convParams.kW = s;

    TSize wOut = convOutputDimSize(w1, convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW);
    TSize hOut = convOutputDimSize(h1, convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH);

    pTensor relu1In   = createTensor({c1, w1,   h1,   b1}, syn_type_bf16);
    pTensor convIn    = createTensor({c1, w1,   h1,   b1}, syn_type_bf16);
    pTensor convWgh   = createTensor({k,  c1,   s,    r},  syn_type_bf16);
    pTensor relu3In   = createTensor({k,  c1,   s,    r},  syn_type_bf16);
    pTensor convWgh2  = createTensor({k,  c1,   s,    r},  syn_type_bf16);
    pTensor relu2In   = createTensor({k,  c1,   s,    r},  syn_type_bf16);
    pTensor convOut   = createTensor({k,  wOut, hOut, b1}, syn_type_bf16);
    pTensor convOut2  = createTensor({k,  wOut, hOut, b1}, syn_type_bf16);
    pTensor consumer1 = createTensor({k,  wOut, hOut, b1}, syn_type_bf16);
    pTensor consumer2 = createTensor({k,  wOut, hOut, b1}, syn_type_bf16);

    pNode   relu1Node    = NodeFactory::createGenericTPCNode({relu1In}, {convIn}, nullptr, "relu_fwd_bf16", "relu1prod");
    ASSERT_TRUE(GraphEditor::addNode(g, relu1Node));

    pNode   relu3Node    = NodeFactory::createGenericTPCNode({relu3In}, {convWgh}, nullptr, "relu_fwd_bf16", "relu3prod");
    ASSERT_TRUE(GraphEditor::addNode(g, relu3Node));

    pNode convNode  = NodeFactory::createNode({convIn, convWgh},  {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv1");
    ASSERT_TRUE(GraphEditor::addNode(g, convNode));

    pNode   relu2Node    = NodeFactory::createGenericTPCNode({relu2In}, {convWgh2}, nullptr, "relu_fwd_bf16", "relu2prod");
    ASSERT_TRUE(GraphEditor::addNode(g, relu2Node));

    pNode convNode2 = NodeFactory::createNode({convIn, convWgh2}, {convOut2}, &convParams, NodeFactory::convolutionNodeTypeName, "conv2");
    ASSERT_TRUE(GraphEditor::addNode(g, convNode2));

    pNode   relu1ConsumerNode    = NodeFactory::createGenericTPCNode({convOut}, {consumer1}, nullptr, "relu_fwd_bf16", "relu1cons");
    ASSERT_TRUE(GraphEditor::addNode(g, relu1ConsumerNode));

    pNode   relu2ConsumerNode    = NodeFactory::createGenericTPCNode({convOut2}, {consumer2}, nullptr, "relu_fwd_bf16", "relu2cons");
    ASSERT_TRUE(GraphEditor::addNode(g, relu2ConsumerNode));

    ASSERT_TRUE(loadTpcKernels(g));

    std::unordered_map<pBundle, BundleSolvingData> solvingDataPerBundle;
    Bundlizer  bundlizer(g);
    BundleList bundles = bundlizer.getMMEBundles();
    AllBrains  brains(g);

    for (pBundle& bundle : bundles)
    {
        solvingDataPerBundle[bundle].strategies = brains.m_mmeBrain.getSolutionStrategies(bundle);
    }

    graphVisualizationPre(g);

    BundleExpander expander(g, brains, bundlizer, solvingDataPerBundle);

    LOG_DEBUG(SRAM_SLICE, "Looking for candidates for Bundle: {}", bundles.back()->getNodes().back()->getNodeName());

    SlicingStrategyList expandedStrategies = expander.generateExpandedStrategies(bundles.back());

    for (SlicingStrategyPtr& s : expandedStrategies)
    {
        pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
        int validCandidatesCounter = 0;
        for (pBundleExpansion extension :strategy->getMmeSlicingData().getRoleCandidates())
        {
            if (extension) validCandidatesCounter++;
        }

        int numberOfCandidates = strategy->getMmeSlicingData().getInvalidCandidates().size() + validCandidatesCounter;

        /* slave, 2 master producers, 2 slave producers, 2 consumers */
        ASSERT_EQ(numberOfCandidates, 7);
    }
}

TEST_F(SRAMManagementTest, bundlizer_should_offer_separable_tpc_consumer_candidate)
{
    // Given sliced MME -> Separable TPC
    const TSize b = 64, h = 16, w = 16, c = 32;
    const TSize r = 1,  s = 1,  k = 32;

    pTensor ifm     = createTensor({c, w * h * b, 1, 1}, syn_type_bf16);
    pTensor wgh     = createTensor({k,         c, s, r}, syn_type_bf16);
    pTensor mmeOut  = createTensor({k, w * h * b, 1, 1}, syn_type_bf16);
    pTensor reluOut = createTensor({k, w * h * b, 1, 1}, syn_type_bf16);

    synConvolutionParams params{};
    pNode conv = NodeFactory::createNode({ifm, wgh}, {mmeOut}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    GraphEditor::addNode(getGraph(), conv);

    pNode relu      = NodeFactory::createNode({mmeOut}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    GraphEditor::addNode(getGraph(), relu);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    Bundlizer bundlizer{getGraph()};
    pBundle bundle = bundlizer.getMMEBundles().front();

    SlicingBrain::knobs.maxNarrowSliceSize = 1024;
    SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon2D = 1024;

    // When
    auto             strategies         = getMmeBrain().getSolutionStrategies(bundle);
    pBundleExpansion expansionCandidate =
        bundlizer.findTpcConsumerExpansionCandidate(std::static_pointer_cast<MmeSlicingStrategy>(
            findWinningStrategy(strategies, bundle, getGraph(), getSlicingBrains())));

    // Then
    ASSERT_EQ(expansionCandidate->nodeToStitch, relu);
    ASSERT_EQ(expansionCandidate->stitchedOperand->originalTensor, mmeOut);
}

TEST_F(SRAMManagementTest, bundlizer_should_offer_reshaped_separable_tpc_consumer_candidate)
{
    // Given sliced MME -> Reshape -> Separable TPC
    const TSize b = 64, h = 16, w = 16, c = 32;
    const TSize r = 1,   s = 1,  k = 32;

    pTensor ifm     = createTensor({c, w * h * b, 1, 1}, syn_type_bf16);
    pTensor wgh     = createTensor({k,         c, s, r}, syn_type_bf16);
    pTensor mmeOut  = createTensor({k, w * h * b, 1, 1}, syn_type_bf16);

    synConvolutionParams params{};
    pNode conv = NodeFactory::createNode({ifm, wgh}, {mmeOut}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    GraphEditor::addNode(getGraph(), conv);

    pTensor reluIn  = createTensor({k, w, h, b}, syn_type_bf16);
    pTensor reluOut = createTensor({k, w, h, b}, syn_type_bf16);

    pNode reshape   = NodeFactory::createNode({mmeOut}, {reluIn}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    pNode relu      = NodeFactory::createNode({reluIn}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");

    GraphEditor::addNode(getGraph(), reshape);
    GraphEditor::addNode(getGraph(), relu);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    Bundlizer bundlizer{getGraph()};
    pBundle bundle = bundlizer.getMMEBundles().front();

    SlicingBrain::knobs.maxNarrowSliceSize = 1024;
    SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon2D = 1024;

    // When
    auto             strategies         = getMmeBrain().getSolutionStrategies(bundle);
    pBundleExpansion expansionCandidate =
        bundlizer.findTpcConsumerExpansionCandidate(std::static_pointer_cast<MmeSlicingStrategy>(
            findWinningStrategy(strategies, bundle, getGraph(), getSlicingBrains())));

    // Then
    ASSERT_EQ(expansionCandidate->nodeToStitch, relu);
    ASSERT_EQ(expansionCandidate->stitchedOperand->originalTensor, mmeOut);
    ASSERT_EQ(expansionCandidate->reshapeNode, reshape);
}

TEST_F(SRAMManagementTest, bundlizer_should_not_offer_tpc_consumer_that_jump_over_another_bundle)
{
    // Given MME1 -> MME2 ----> (TPC
    //          \___________-->   ADD)

    pTensor mme1Ifm = createTensor({1024, 1024, 1, 1}, syn_type_float);
    pTensor wgh     = createTensor({1024, 1024, 1, 1}, syn_type_float);
    pTensor mme1Out = createTensor({1024, 1024, 1, 1}, syn_type_float);
    pTensor mme2Out = createTensor({1024, 1024, 1, 1}, syn_type_float);
    pTensor addOut  = createTensor({1024, 1024, 1, 1}, syn_type_float);

    synConvolutionParams params{};
    pNode mme1 = NodeFactory::createNode({mme1Ifm, wgh},
                                         {mme1Out},
                                         &params,
                                         NodeFactory::convolutionNodeTypeName,
                                         "mme1");
    GraphEditor::addNode(getGraph(), mme1);
    pNode mme2 = NodeFactory::createNode({mme1Out, wgh},
                                         {mme2Out},
                                         &params,
                                         NodeFactory::convolutionNodeTypeName,
                                         "mme2");
    GraphEditor::addNode(getGraph(), mme2);

    pNode tpcAdd = NodeFactory::createGenericTPCNode({mme1Out, mme2Out}, {addOut}, nullptr, "add_fwd_f32", "add");
    GraphEditor::addNode(getGraph(), tpcAdd);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    const MMESlicingBrain& mmeBrain = getMmeBrain();
    Bundlizer bundlizer{getGraph()};

    BundleList bundles = bundlizer.getMMEBundles();

    // When
    SlicingStrategyList strategies1 = mmeBrain.getSolutionStrategies(bundles.front());
    const auto&         strategy1   = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(strategies1, bundles.front(), getGraph(), getSlicingBrains()));
    pBundleExpansion    expCnd1     = bundlizer.findTpcConsumerExpansionCandidate(strategy1);
    SlicingStrategyList strategies2 = mmeBrain.getSolutionStrategies(bundles.back());
    const auto&         strategy2   = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(strategies2, bundles.back(), getGraph(), getSlicingBrains()));
    pBundleExpansion expCnd2 = bundlizer.findTpcConsumerExpansionCandidate(strategy2);

    // Then
    ASSERT_FALSE(
        BundleExpander::validateCandidatePaths(getGraph(),
                                               expCnd1,
                                               strategy1->getMmeSlicingData().getStrategyNodes(bundles.front()),
                                               strategy1->getMmeSlicingData().getStrategyProducers()));
    ASSERT_EQ(expCnd2->nodeToStitch, tpcAdd);
    ASSERT_EQ(expCnd2->stitchedOperand->originalTensor, mme2Out);
    ASSERT_TRUE(BundleExpander::validateCandidatePaths(getGraph(),
                                                       expCnd2,
                                                       strategy2->getMmeSlicingData().getStrategyNodes(bundles.back()),
                                                       strategy2->getMmeSlicingData().getStrategyProducers()));
}

TEST_F(SRAMManagementTest, bundlizer_should_not_offer_consumer_scheduled_after_another_tpc_node)
{
    // Given  MME -> [] -> MEMCPY -> [] -> RELU -> [] -> ADD
    //                |                                  ^
    //                |                                  |
    //                +----------------------------------+
    // (Memcpy is added so the ReLU can't be stitched)

    synConvolutionParams params{};
    pTensor mmeIfm = createTensor({1024, 1024, 1, 1}, syn_type_float);
    pTensor wgh = createTensor({1024, 1024, 1, 1}, syn_type_float);
    pTensor mmeOut = createTensor({1024, 1024, 1, 1}, syn_type_float);
    pNode mme = NodeFactory::createNode({mmeIfm, wgh}, {mmeOut}, &params, NodeFactory::convolutionNodeTypeName, "mme");

    pTensor mmeOutCopy = createTensor({1024, 1024, 1, 1}, syn_type_float);
    pNode memcpyNode = NodeFactory::createNode({mmeOut}, {mmeOutCopy}, nullptr, NodeFactory::memcpyNodeTypeName, "memcpy");

    pTensor reluOut = createTensor({1024, 1024, 1, 1}, syn_type_float);
    pNode relu = NodeFactory::createNode({mmeOutCopy}, {reluOut}, nullptr, "relu_fwd_f32", "relu");

    pTensor addOut = createTensor({1024, 1024, 1, 1}, syn_type_float);
    pNode tpcAdd = NodeFactory::createGenericTPCNode({mmeOut, reluOut}, {addOut}, nullptr, "add_fwd_f32", "add");

    GraphEditor::addNode(getGraph(), mme);
    GraphEditor::addNode(getGraph(), memcpyNode);
    GraphEditor::addNode(getGraph(), relu);
    GraphEditor::addNode(getGraph(), tpcAdd);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    Bundlizer bundlizer{getGraph()};

    BundleList bundles = bundlizer.getMMEBundles();

    // When
    SlicingStrategyList strategies = getMmeBrain().getSolutionStrategies(bundles.front());
    const auto&         strategy   = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(strategies, bundles.front(), getGraph(), getSlicingBrains()));
    pBundleExpansion expCnd = bundlizer.findTpcConsumerExpansionCandidate(strategy);

    // Then
    ASSERT_FALSE(BundleExpander::validateCandidatePaths(getGraph(),
                                                        expCnd,
                                                        strategy->getMmeSlicingData().getStrategyNodes(bundles.front()),
                                                        strategy->getMmeSlicingData().getStrategyProducers()));
}

TEST_F(SRAMManagementTest, NonCD2DSolver_with_initial_strategy)
{
    // IFM size (64 * 5)x10 bfloats,
    // WGH size 10x64 bfloats,
    // After forcing the winning strategy to slice the CD to 2, check that the solver generate all strategies
    // with this CD slice size.
    const TSize CD = 10;
    TSize b = 1, h =1, w = 512;
    TSize k = 256;
    TSize       ifmsizes[] = {CD, w, h, b};
    const TSize wsizes[]   = {k, CD};
    TSize       ofmsizes[] = {k, w, h, b};

    pTensor ifm(new Tensor(4, ifmsizes, syn_type_bf16));
    pTensor wgh(new Tensor(2, wsizes, syn_type_bf16));
    pTensor ofm(new Tensor(4, ofmsizes, syn_type_bf16));

    const unsigned ifmSizeInBytes = ifm->getDenseSizeInBytes();
    const unsigned weightSize     = wgh->getDenseSizeInBytes();

    // Set room for 1/10 of weights and ifm tensors
    SlicingBrain::knobs.maxSRAMCapInBytes = (weightSize + ifmSizeInBytes) / 10;

    pBundle bundle = createSingleMMENodeBundle({ifm, wgh, nullptr, nullptr},
                                               {ofm},
                                               NodeFactory::convolutionNodeTypeName);

    pMmeSlicingStrategy initialStrategy =
        MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle->getNodes().front());
    initialStrategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
    // mimic another solver - slice on the CD.
    initialStrategy->getMmeSlicingData().bundleTensors[0]->chunkDimensions[0] = 2;
    initialStrategy->getMmeSlicingData().bundleTensors[1]->chunkDimensions[1] = 2;

    NonCD2DSolver solver(*getGraph().getHALReader(), bundle, initialStrategy);
    solver.createAllStrategies();
    const SlicingStrategyPtr& winningStrategy =
        findWinningStrategy(solver.getStrategies(), bundle, getGraph(), getSlicingBrains());
    ASSERT_EQ(winningStrategy->getSlicingData().bundleTensors[0]->chunkDimensions[0], 2) << "IFM wasn't sliced on CD";
    ASSERT_LT(winningStrategy->getSlicingData().bundleTensors[0]->chunkDimensions[1], w) << "IFM wasn't sliced on W";
    ASSERT_LT(winningStrategy->getSlicingData().bundleTensors[1]->chunkDimensions[0], k) << "WGH wasn't sliced on K";
    ASSERT_EQ(winningStrategy->getSlicingData().bundleTensors[1]->chunkDimensions[1], 2) << "WGH wasn't sliced on CD";
    ASSERT_LE(winningStrategy->getMetrics().SRAMCapacity, SlicingBrain::knobs.maxSRAMCapInBytes);
}

TEST_F(SRAMManagementTest, slicing_only_bhw_when_k_dim_is_1)
{
    // IFM size (64 * 5)x10 bfloats,
    // WGH size 10x1 bfloats,
    // SRAM can fit the WGH and a 128x10 slice out of the IFM
    // Expects WGH to stay in SRAM while 128x2 chunk (single-buffer) will be fetch from HBM.

    const TSize expSlices  = 5;
    TSize b = 1, h = 1, w = expSlices * 64;
    TSize k = 1;
    const TSize CD = 10;
    TSize       ifmsizes[] = {CD, w, h, b};
    const TSize wsizes[]   = {k, CD};
    TSize       ofmsizes[] = {k, w, h, b};

    pTensor ifm(new Tensor(4, ifmsizes, syn_type_bf16));
    pTensor wgh(new Tensor(2, wsizes, syn_type_bf16));
    pTensor ofm(new Tensor(4, ofmsizes, syn_type_bf16));

    const uint64_t ifmLineSizeInBytes = CD * ifm->getElementSizeInBytes();
    const uint64_t weightSize         = wgh->getDenseSizeInBytes();

    SlicingBrain::knobs.maxSRAMCapInBytes =
        (weightSize + GaudiHalReader::instance(synDeviceGaudi)->getMmeVectorSize() * ifmLineSizeInBytes);

    pBundle bundle = createSingleMMENodeBundle({ifm, wgh, nullptr, nullptr},
                                               {ofm},
                                               NodeFactory::convolutionNodeTypeName);

    NonCD2DSolver solver(*getGraph().getHALReader(), bundle);
    solver.createAllStrategies();
    const pMmeSlicingStrategy& winningStrategy = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(solver.getStrategies(), bundle, getGraph(), getSlicingBrains()));
    SolutionGenerator(getGraph(), bundle, winningStrategy).fillSolution();
    Solution solution = bundle->getSolution();

    // Expectation: since MME utilization is next to zero and BW is very low,
    // the comparator should prefer double buffered solution which is received by slicing BHW to 64 elements.
    checkChunkSize(solution, ifm, {CD, 64, h, b});
    checkChunkSize(solution, wgh, wgh->getAllSizesInElements());
    ASSERT_EQ(winningStrategy->getMetrics().SRAMCapacity, 2580);
    ASSERT_EQ(winningStrategy->getMmeSlicingData().MMEGeometryUsed, gaudi_geometry_4wx1h);
}

//Todo- SW-17694 rewrite test after flatten moved to sram management
TEST_F(SRAMManagementTest, DISABLED_slicing_on_common_dim_and_non_common_dim_2d)
{
    // SRAM Capacity can fit 8 dedy batches & 8 activation batches in chunks of 128x8 and the dedw tensor sliced to 128x128.
    // Expects slicing the both inputs and output and use single-buffer.

    // Given
    const TSize w = 1, h = 1, b = 255;
    const TSize k = 255, c = 255;
    const TSize xSizes[] = {c, w, h, b};
    const TSize dedySizes[]   = {k, w, h, b};
    const TSize dedwSizes[] = {k, c, 1, 1};

    const TSize expBatchSliceSize   = 16;
    const TSize expSpatialSliceSize = 64;

    pTensor x(new Tensor(4, xSizes, syn_type_float));
    pTensor dedy(new Tensor(4, dedySizes, syn_type_float));
    pTensor dedw(new Tensor(4, dedwSizes, syn_type_float));

    const TSize elemSize = dedw->getElementSizeInBytes();
    pBundle bundle = createSingleMMENodeBundle({dedy, x}, {dedw}, NodeFactory::deDwNodeTypeName);

    SlicingBrain::knobs.maxSRAMCapInBytes =
            elemSize * expBatchSliceSize * expSpatialSliceSize * 2 + elemSize * expSpatialSliceSize * expSpatialSliceSize;

    SlicingBrain::knobs.minCDSizeForPartials = h * w * expBatchSliceSize;

    // Then
    solveBundleWithStrategy(bundle);
    Solution solution = bundle->getSolution();
    unsigned expRows = std::ceil((float)k/expSpatialSliceSize);
    unsigned expCol = std::ceil((float)c/expSpatialSliceSize);
    unsigned expNofOps = expRows * expCol * div_round_up(b, expBatchSliceSize);
    checkSolutionSize(solution, 3, expNofOps);
    checkChunkSize(solution, x, {expSpatialSliceSize, 1, 1 , expBatchSliceSize});
    checkChunkSize(solution, dedy, {expSpatialSliceSize, 1, 1 ,expBatchSliceSize});
    checkChunkSize(solution, dedw, {expSpatialSliceSize, expSpatialSliceSize, 1, 1});
    ExecutionOrderChecker execOrderChecker(WEIGHT_DIM_K, DIM_C);
    execOrderChecker.setOpSkipFactor(div_round_up(b, expBatchSliceSize));
    execOrderChecker.checkWalkRightExecutionOrder(solution, expRows, expCol, {WEIGHT_DIM_K, WEIGHT_DIM_C});
}

TEST_F(SRAMManagementTest, slicing_on_common_dim_and_non_common_dim_4d)
{
    // SRAM Capacity can fit 8 dedy batches & 8 activation batches in chunks of 128x8 and the dedw tensor sliced to 128x128.
    // Expects slicing the both inputs and output and use single-buffer.

    // Given
    const TSize wx = 30, hx = 30, b = 255;
    const TSize k = 255, c = 255, r = 3, s = 3;
    const TSize wy = wx-r+1, hy = hx-s+1;
    const TSize xSizes[] = {c, wx, hx, b};
    const TSize dedySizes[]   = {k, wy, hy, b};
    const TSize dedwSizes[] = {k, c, s, r};

    const TSize expBatchSliceSize   = 128;
    const TSize expSpatialSliceSize = 64;

    pTensor x(new Tensor(4, xSizes, syn_type_float));
    pTensor dedy(new Tensor(4, dedySizes, syn_type_float));
    pTensor dedw(new Tensor(4, dedwSizes, syn_type_float));

    const TSize elemSize = dedw->getElementSizeInBytes();
    synConvolutionParams params;
    params.kH = r;
    params.kW = s;
    pBundle bundle =
        createSingleMMENodeBundle({dedy, x}, {dedw}, NodeFactory::deDwNodeTypeName, &params, sizeof(params));

    SlicingBrain::knobs.maxSRAMCapInBytes =
            (elemSize * expBatchSliceSize * hx * wx * expSpatialSliceSize +
             elemSize * expBatchSliceSize * hy * wy * expSpatialSliceSize +
             elemSize * expSpatialSliceSize * expSpatialSliceSize * r * s);

    SlicingBrain::knobs.minCDSizeForPartials = hy * wy * expBatchSliceSize;

    // Then
    solveBundleWithStrategy(bundle);
    Solution solution = bundle->getSolution();
    unsigned expRows = std::ceil((float)k/expSpatialSliceSize);
    unsigned expCol = std::ceil((float)c/expSpatialSliceSize);
    unsigned expNofOps = expRows * expCol * std::ceil((float)b/expBatchSliceSize);
    checkSolutionSize(solution, 3, expNofOps);
    checkChunkSize(solution, x, {expSpatialSliceSize, wx, hx , expBatchSliceSize});
    checkChunkSize(solution, dedy, {expSpatialSliceSize, wy, hy ,expBatchSliceSize});
    checkChunkSize(solution, dedw, {expSpatialSliceSize, expSpatialSliceSize, s, r});
    ExecutionOrderChecker execOrderChecker(WEIGHT_DIM_K, DIM_C);
    execOrderChecker.setOpSkipFactor(div_round_up(b, expBatchSliceSize));
    execOrderChecker.checkWalkRightExecutionOrder(solution, expRows, expCol, {WEIGHT_DIM_K, WEIGHT_DIM_C});
}

TEST_F(SRAMManagementTest, big_tensor_metric_calculation)
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_COST_MODEL_ENABLED, "false");

    // Disable spatial slicing to reduce test time.
    setGlobalConfForTest(GCFG_SRAM_SLICER_4D_CONV_SPATIAL_SLICE_ENABLED, "false");

    // make sure metrics will not overflow or zero when there are many slices.
    // Given
    const TSize w = 4096*4096*4, h = 1, b = 1;
    const TSize k = 4096, c = 4096;
    const TSize xSizes[] = {c, w, h, b};
    const TSize wSizes[]   = {k, c, 1, 1};
    const TSize ySizes[] = {k, w, h, b};

    // make sure many slices will be created.
    SlicingBrain::knobs.maxNarrowSliceSize = 1024;

    pTensor x(new Tensor(4, xSizes, syn_type_bf16));
    pTensor weights(new Tensor(4, wSizes, syn_type_bf16));
    pTensor y(new Tensor(4, ySizes, syn_type_bf16));

    pBundle bundle = createSingleMMENodeBundle({x, weights}, {y}, NodeFactory::convolutionNodeTypeName);

    // Then
    auto strategies = getMmeBrain().getSolutionStrategies(bundle);
    ASSERT_NE(0, strategies.size()) << "brain did not return any strategy to solve the bundle";
    const MmeSlicingStrategy::Metrics& met =
        findWinningStrategy(strategies, bundle, getGraph(), getSlicingBrains())->getMetrics();

    ASSERT_TRUE(std::isnormal(met.MMEUtilization)) << "MME Utilization should not be inf or nan";
    ASSERT_TRUE(std::isnormal(met.HBMBandwidth)) << "HBM Bandwidth should not be inf or nan";
    ASSERT_TRUE(met.SRAMCapacity > 0) << " SRAM Capacity should not be 0";
}

// in case of a stitched tpc node with multiple outputs, the order of the outputs should stay the same after the
// sram management slicing
TEST_F(SRAMManagementTest, stitched_tpc_consumer_with_multiple_outputs)
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_MULTIPLE_SOLVERS_ENABLED, "false");

    GaudiGraph g;
    pTensor tIn1 = createTensor({32, 28, 28, 128}, syn_type_bf16);
    pTensor tOut1 = createTensor({32, 14, 14, 128}, syn_type_int16);
    pTensor tOut2ConvIn = createTensor({32, 14, 14, 128}, syn_type_bf16);
    ns_SpatialReduction::Params maxpoolParams;
    maxpoolParams.pooling_convention = POOLING_CONVENTION_VALID;
    maxpoolParams.kernel_w = 3;
    maxpoolParams.kernel_h = 3;
    maxpoolParams.stride_w = 2;
    maxpoolParams.stride_h = 2;
    maxpoolParams.dilation_w = 1;
    maxpoolParams.dilation_h = 1;
    maxpoolParams.pad_w_begin = 1;
    maxpoolParams.pad_w_end = 1;
    maxpoolParams.pad_h_begin = 1;
    maxpoolParams.pad_h_end = 1;
    pNode maxpoolNode = NodeFactory::createGenericTPCNode({tIn1}, {tOut1, tOut2ConvIn}, &maxpoolParams,
                                                          "maxpool_2d_fwd_bf16", "maxpool");
    auto tpcNode = std::dynamic_pointer_cast<TPCNode>(maxpoolNode);
    ASSERT_FALSE(tpcNode->isSeparable(tpc_lib_api::DEVICE_ID_GAUDI));
    ASSERT_TRUE(GraphEditor::addNode(g, maxpoolNode));

    pTensor convWgh = createTensor({1, 1, 2, 2}, syn_type_bf16);
    pTensor convOut = createTensor({32, 14, 14 ,128}, syn_type_float);
    synConvolutionParams convParams;
    convParams.kH = 2;
    convParams.kW = 2;
    pNode convNode = NodeFactory::createNode({tOut2ConvIn, convWgh}, {convOut}, &convParams,
                                             NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(g, convNode));

    ASSERT_TRUE(loadTpcKernels(g));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(g));

    int i = 0;
    unsigned bundleID = -1;
    for (const NodePtr& n : g.getExeSortedNodes())
    {
        // all nodes should be in a bundle, and in the same one (because of TPC stitching):
        ASSERT_TRUE(n->getNodeAnnotation().bundleInfo.is_set());
        if (i == 0)
        {
            bundleID = n->getNodeAnnotation().bundleInfo->bundleIndex;
        }
        else
        {
            ASSERT_EQ(n->getNodeAnnotation().bundleInfo->bundleIndex, bundleID);
        }

        if (HabanaGraph::runsOnTPC(n))
        {
            ASSERT_EQ(n->getInputs().size(), 1);
            ASSERT_EQ(n->getOutputs().size(), 2);
            // the outputs should be in the same order as the original graph
            ASSERT_EQ(n->getOutput(0)->getElementType(), syn_type_int16);
            ASSERT_EQ(n->getOutput(1)->getElementType(), syn_type_bf16);
            ASSERT_FALSE(n->getOutput(0)->inSram());
            ASSERT_TRUE(n->getOutput(1)->inSram());

        }
        else if (HabanaGraph::runsOnMME(n))
        {
            ASSERT_EQ(n->getInputs().size(), 2);
            ASSERT_TRUE(n->getInput(0)->inSram());
            ASSERT_TRUE(n->getInput(1)->inSram());
            ASSERT_EQ(n->getOutputs().size(), 1);
        }
        ++i;
    }
}

// in case of a stitched tpc node with multiple outputs, the order of the outputs should stay the same after the
// sram management slicing
TEST_F(SRAMManagementTest, ctrl_dep_dont_stitch_tpc_consumer_with_multiple_outputs)
{
    GaudiGraph g;
    pTensor tIn1 = createTensor({32, 28, 28, 128}, syn_type_bf16);
    pTensor tOut1 = createTensor({32, 14, 14, 128}, syn_type_int16);
    pTensor tOut2ConvIn = createTensor({32, 14, 14, 128}, syn_type_bf16);
    ns_SpatialReduction::Params maxpoolParams;
    maxpoolParams.pooling_convention = POOLING_CONVENTION_VALID;
    maxpoolParams.kernel_w = 3;
    maxpoolParams.kernel_h = 3;
    maxpoolParams.stride_w = 2;
    maxpoolParams.stride_h = 2;
    maxpoolParams.dilation_w = 1;
    maxpoolParams.dilation_h = 1;
    maxpoolParams.pad_w_begin = 1;
    maxpoolParams.pad_w_end = 1;
    maxpoolParams.pad_h_begin = 1;
    maxpoolParams.pad_h_end = 1;
    pNode maxpoolNode = NodeFactory::createGenericTPCNode({tIn1}, {tOut1, tOut2ConvIn}, &maxpoolParams,
                                                          "maxpool_2d_fwd_bf16", "maxpool");
    auto tpcNode = std::dynamic_pointer_cast<TPCNode>(maxpoolNode);
    ASSERT_FALSE(tpcNode->isSeparable(tpc_lib_api::DEVICE_ID_GAUDI));
    ASSERT_TRUE(GraphEditor::addNode(g, maxpoolNode));

    pTensor convWgh = createTensor({1, 1, 2, 2}, syn_type_bf16);
    pTensor convOut = createTensor({32, 14, 14 ,128}, syn_type_float);
    synConvolutionParams convParams;
    convParams.kH = 2;
    convParams.kW = 2;
    pNode convNode = NodeFactory::createNode({tOut2ConvIn, convWgh}, {convOut}, &convParams,
                                             NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(g, convNode));

    g.addControlDependency({maxpoolNode}, {convNode});

    ASSERT_TRUE(loadTpcKernels(g));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(g));

    int i = 0;
    Settable<unsigned> bundleIndex;
    for (const NodePtr& n : g.getExeSortedNodes())
    {
        // all nodes should be in a bundle, and in the same one (because of TPC stitching):
        if (i == 0)
        {
            ASSERT_TRUE(HabanaGraph::runsOnTPC(n)) << "Expecting TPC node on idx 0";
            ASSERT_FALSE(n->getNodeAnnotation().bundleInfo.is_set()) << "TPC Node should not belong to the bundle";
        }
        else
        {
            ASSERT_TRUE(n->getNodeAnnotation().bundleInfo.is_set()) << "node should belong to a bundle";
            if (!bundleIndex.is_set())
            {
                bundleIndex = n->getNodeAnnotation().bundleInfo->bundleIndex;
            }
            else
            {
                ASSERT_EQ(n->getNodeAnnotation().bundleInfo->bundleIndex, bundleIndex.value());
            }
        }

        ++i;
    }
}

// when the reduction output (such as the output of dedw node) is defined by the user as bf16,
// we will change the reduction output to be fp32 in order for the reduction to be accurate.
TEST_F(SRAMManagementTest, reduction_bf16_with_tpc_consumer)
{
    SlicingBrain::knobs.minCDSizeForPartials = 512;
    // since b*h*w=10*64*64 is bigger than SlicingBrain::knobs.minCDSizeForPartials = 512, the common dim solver is used
    const TSize b = 255, h = 64, w = 64;
    const TSize c = 16, k = 4;
    const TSize dedySizes[]    = {k, w, h, b};
    const TSize xSizes[]       = {c, w, h, b};
    const TSize dedwOutSizes[] = {c, k, 1, 1};

    pTensor x(std::make_shared<Tensor>(4, xSizes, syn_type_bf16));
    pTensor dedy(std::make_shared<Tensor>(4, dedySizes, syn_type_bf16));
    pTensor dedwOutReluIn(std::make_shared<Tensor>(4, dedwOutSizes, syn_type_bf16));
    synConvolutionParams params;
    pNode dedwNode = NodeFactory::createNode({dedy, x}, {dedwOutReluIn}, &params, NodeFactory::deDwNodeTypeName,
                                             "dedw");
    pTensor reluOut(std::make_shared<Tensor>(4, dedwOutSizes, syn_type_bf16));
    pNode reluNode = NodeFactory::createGenericTPCNode({dedwOutReluIn}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");

    ASSERT_TRUE(GraphEditor::addNode(getGraph(), dedwNode));
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reluNode));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
    pTensor reductionOutpuTensor = nullptr;
    bool reductionFound = false, tpcNodeFound = false, memcpyFp32ToBf16Found = false;
    for (const NodePtr& node : getGraph().getExeSortedNodes())
    {
        // validate reduction multiple inputs and single output are in SRAM and have float type
        if (node->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            ASSERT_GT(node->getNumInputs(), 1);
            ASSERT_EQ(node->getNumOutputs(), 1);
            for (auto tensor : node->getOperands())
            {
                ASSERT_TRUE(tensor->inSram());
                ASSERT_EQ(tensor->getElementType(), syn_type_float);
            }
            reductionOutpuTensor = node->getOutput(0);
            reductionFound = true;
        }
        else if (reductionOutpuTensor != nullptr)
        {   // after the reduction node there should be a memcpy from float to bf16
            // (later it will be translated to a cast node)
            if (node->getNodeType() == Node::TYPE_MEMCOPY)
            {
                ASSERT_EQ(node->getNumInputs(), 1);
                ASSERT_EQ(node->getNumOutputs(), 1);
                ASSERT_EQ(node->getInput(0), reductionOutpuTensor);
                ASSERT_EQ(node->getInput(0)->getElementType(), syn_type_float);
                ASSERT_EQ(node->getOutput(0)->getElementType(), syn_type_bf16);
                memcpyFp32ToBf16Found = true;
            }
            else if (HabanaGraph::runsOnTPC(node))
            {
                ASSERT_EQ(node->getNumInputs(), 1);
                ASSERT_EQ(node->getNumOutputs(), 1);
                ASSERT_EQ(node->getInput(0), dedwOutReluIn);
                ASSERT_EQ(node->getOutput(0), reluOut);
                tpcNodeFound = true;
            }
        }
    }
    ASSERT_TRUE(reductionFound);
    ASSERT_TRUE(tpcNodeFound);
    ASSERT_TRUE(memcpyFp32ToBf16Found);
}

// when the reduction output (such as the output of dedw node) is defined by the user as bf16,
// we will change the reduction output to be fp32 in order for the reduction to be accurate.
// For this reason we want to block the stitching of a tpc consumer in that case
TEST_F(SRAMManagementTest, reduction_bf16_with_tpc_consumer_no_stitching)
{
    SlicingBrain::knobs.minCDSizeForPartials = 512;
    // since b*h*w=10*64*64 is bigger than SlicingBrain::knobs.minCDSizeForPartials = 512, the common dim solver is used
    const TSize b = 255, h = 64, w = 64;
    const TSize c = 16, k = 4;
    const TSize dedySizes[]    = {k, w, h, b};
    const TSize xSizes[]       = {c, w, h, b};
    const TSize dedwOutSizes[] = {c, k, 1, 1};

    pTensor x(std::make_shared<Tensor>(4, xSizes, syn_type_bf16));
    pTensor dedy(std::make_shared<Tensor>(4, dedySizes, syn_type_bf16));
    pTensor dedwOutReluIn(std::make_shared<Tensor>(4, dedwOutSizes, syn_type_bf16));
    synConvolutionParams params;
    pNode dedwNode = NodeFactory::createNode({dedy, x}, {dedwOutReluIn}, &params, NodeFactory::deDwNodeTypeName,
                                             "dedw");
    pTensor reluOut(std::make_shared<Tensor>(4, dedwOutSizes, syn_type_bf16));
    pNode reluNode = NodeFactory::createGenericTPCNode({dedwOutReluIn}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");

    ASSERT_TRUE(GraphEditor::addNode(getGraph(), dedwNode));
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reluNode));

    Bundlizer bundlizer(getGraph());
    pBundle bundle = bundlizer.getMMEBundles().front();
    CommonDimSlicingSolver  solver(*getGraph().getHALReader(), bundle);
    ASSERT_TRUE(solver.effectiveForBundle());
    pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(getMmeBrain().getSolutionStrategies(bundle), bundle, getGraph(), getSlicingBrains()));

    // validate the tpc node is not stitched
    pBundleExpansion expCand = bundlizer.findTpcConsumerExpansionCandidate(strategy);
    ASSERT_EQ(expCand->nodeToStitch, nullptr);
    ASSERT_EQ(expCand->stitchedOperand, nullptr);
}

TEST_F(SRAMManagementTest, reduction_memset_should_be_added_to_bundle)
{
    const std::vector<TSize> dySizes = {64, 56, 56, 256};
    const std::vector<TSize> xSizes  = {64, 56, 56, 256};
    const std::vector<TSize> dwSizes = {64, 64, 1, 1};

    pTensor              x  = createTensor(xSizes, syn_type_bf16, true);
    pTensor              dy = createTensor(dySizes, syn_type_bf16, true);
    pTensor              dw = createTensor(dwSizes, syn_type_bf16, true);
    synConvolutionParams params(1, 1, 1, 1, 0, 0, 0, 0, 1, 1);

    pNode dedwNode = NodeFactory::createNode({dy, x}, {dw}, &params, NodeFactory::deDwNodeTypeName, "dedw");

    ASSERT_TRUE(GraphEditor::addNode(getGraph(), dedwNode));

    // The dedw is sliced on common-dim and will use SRAM reduction,
    // identifyMmeConcurrency will add memset to the reduction node.
    ASSERT_TRUE(getGraph().compile());

    bool               reductionFound = false;
    bool               memsetFound    = false;
    Settable<unsigned> expectedBundleIdx;

    for (const NodePtr& node : getGraph().getExeSortedNodes())
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            reductionFound = true;
            ASSERT_TRUE(node->getNodeAnnotation().bundleInfo.is_set());
            ASSERT_TRUE(expectedBundleIdx.is_set());
            ASSERT_EQ(node->getNodeAnnotation().bundleInfo->bundleIndex, expectedBundleIdx.value());
        }
        if (node->isMemset())
        {
            memsetFound = true;
            ASSERT_EQ(node->getNumOutputs(), 1);
            ASSERT_TRUE(node->getOutput(0)->inSram());
            // The memset node should get the same bundle index as the reduction node.
            ASSERT_TRUE(node->getNodeAnnotation().bundleInfo.is_set());
            expectedBundleIdx.set(node->getNodeAnnotation().bundleInfo->bundleIndex);
        }
    }

    ASSERT_TRUE(reductionFound);
    ASSERT_TRUE(memsetFound);
}

TEST_F(SRAMManagementTest, HBM_overflow_metric_penalty)
{
   /* Tensors with the sizes of the MME will cause util of 1 and BW of 1.5 Tb/s,
    * which will cause util penalty - so expected metrics will be - MMEUtil of 0.63 , and max BW
    */
    setGlobalConfForTest(GCFG_SRAM_SLICER_MULTIPLE_SOLVERS_ENABLED, "false");

    std::vector<TSize> sizes = {128,128};
    std::shared_ptr<Bundle> bundle = createSingleMMENodeBundle(
        {createTensor(sizes, syn_type_bf16), createTensor(sizes,syn_type_bf16)},
        {createTensor(sizes, syn_type_bf16)},
        NodeFactory::convolutionNodeTypeName);

    const double mmeMaxBwGBps = 6 * 256.0; // BW at maximal frequency of 2GHz = 6 ports at 256GBps
    double currFreqMaxBwGBps  = mmeMaxBwGBps * (GaudiHalReader::instance(synDeviceGaudi)->getClockFreqMHz() / 2000.);

    SlicingStrategyList lst = getMmeBrain().getSolutionStrategies(bundle);
    ASSERT_EQ(1,lst.size());
    SlicingStrategyPtr s = lst.front();
    ASSERT_EQ(s->getMetrics().HBMBandwidth, SlicingBrain::knobs.hbmAvailableBWGBps);
    ASSERT_EQ(s->getMetrics().MMEUtilization, (float)(SlicingBrain::knobs.hbmAvailableBWGBps/currFreqMaxBwGBps));
}

TEST_F(SRAMManagementTest, graphsizeoptimization_noncommondim2dsolver)
{
    // both tensors are big enough to be sliced (larger than mme geometry). sram capacity is limited so
    // slicing is expected. We expect multiple strategies from each geometry, walking pattern in the queue
    const TSize b = 1, h = 1, w = 50000;
    const TSize c = 1, k = 30000;
    synConvolutionParams params;

    const TSize ifmSizes[] = {c, w, h, b};
    const TSize wSizes[]   = {k, c, params.kW, params.kH};
    const TSize outSizes[] = {k,
                              convOutputDimSize(w, params.kW, params.dW, params.padL + params.padR, params.dilW),
                              convOutputDimSize(h, params.kH, params.dH, params.padT + params.padB, params.dilH),
                              b}; // {k,w,h,b}

    pTensor ifm(std::make_shared<Tensor>(ARRAY_SIZE(ifmSizes), ifmSizes, syn_type_float));
    pTensor wgh(std::make_shared<Tensor>(ARRAY_SIZE(wSizes), wSizes, syn_type_float));
    pTensor ofm(std::make_shared<Tensor>(ARRAY_SIZE(outSizes), outSizes, syn_type_float));

    pBundle bundle =
        createSingleMMENodeBundle({ifm, wgh}, {ofm}, NodeFactory::convolutionNodeTypeName, &params, sizeof(params));

    // Decrementing tensor sizes by 1 forces the slicing
    setGlobalConfForTest(GCFG_SRAM_SLICER_REUSE_LIMIT_FACTOR, "0.0"); // use static value from knob.
    SlicingBrain::knobs.maxSRAMCapInBytes = ((ifm->getDenseSizeInBytes() + wgh->getDenseSizeInBytes())) - 1;
    SlicingBrain::knobs.maxNarrowSliceSize  = 1024;
    SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon2D = 1000;
    SlicingBrain::knobs.graphSizeOptimizationMultiplicationFactor = 2;
    NonCD2DSolver solver(*getGraph().getHALReader(), bundle);
    ASSERT_TRUE(solver.effectiveForBundle());
    solver.createAllStrategies();
    solver.AddStrategiesForGraphSizeOptimization();
    SlicingStrategyList strategies = solver.getStrategies();

    // we're supposed to have more than one strategy per geometry and walking pattern given the given knobs
    // and the sram capacity we set
    for (const MmeGeometry& geometry : GAUDI_GEOMETRY)
    {
        for (const auto& walkingPattern : {SlicedOperandTraversalPattern::LEFT_TO_RIGHT_2D,
                                           SlicedOperandTraversalPattern::TOP_TO_BOTTOM_2D})
        {
            // 1 per each strategy, geometry should be as original algorithm
            long countOriginalAlgorithm = std::count_if(strategies.begin(), strategies.end(),
                                            [&](const SlicingStrategyPtr& s)
                                            {
                                                pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
                                                return (strategy->getMmeSlicingData().MMEGeometryUsed == geometry &&
                                                        strategy->getMmeSlicingData().traversalPattern == walkingPattern &&
                                                        !strategy->getGraphSizeOptimized());
                                            });
            ASSERT_EQ(countOriginalAlgorithm, 1);

            // more than one additional "graph size optimized" strategy per geometry and walking pattern
            long countGraphSizeOptimized = std::count_if(strategies.begin(), strategies.end(),
                                            [&](const SlicingStrategyPtr& s)
                                            {
                                                pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
                                                return (strategy->getMmeSlicingData().MMEGeometryUsed == geometry &&
                                                        strategy->getMmeSlicingData().traversalPattern == walkingPattern &&
                                                        strategy->getGraphSizeOptimized());
                                            });
            ASSERT_GT(countGraphSizeOptimized, 1);

            // the strategy with the least sram capacity (per geometry and walking pattern) should be the one created
            // by the original algorithm, and the strategy with the max sram capacity should be created by the graph
            // size optimization solver
            const auto& strategiesWithMinMaxSRAMCap = std::minmax_element(strategies.begin(), strategies.end(),
                                                 [&](const SlicingStrategyPtr& a, const SlicingStrategyPtr& b)
                                                 {
                                                     return a->getMetrics().SRAMCapacity < b->getMetrics().SRAMCapacity;
                                                 });
            ASSERT_FALSE((*strategiesWithMinMaxSRAMCap.first)->getGraphSizeOptimized());
            ASSERT_TRUE((*strategiesWithMinMaxSRAMCap.second)->getGraphSizeOptimized());
        }
    }

    for (const auto& strategy : strategies)
    {
        strategy->printLog(1, synapse::LogManager::LogType::SRAM_SLICE);
        // all strategies supposed to be limited by the sram capacity
        ASSERT_TRUE(strategy->getMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes);
        // both inputs are supposed to be sliced in all strategies
        ASSERT_TRUE(!SlicedOperandUtils::isTriviallySliced(strategy->getSlicingData().bundleTensors[0]));
        ASSERT_TRUE(!SlicedOperandUtils::isTriviallySliced(strategy->getSlicingData().bundleTensors[1]));
        ASSERT_TRUE(!SlicedOperandUtils::isTriviallySliced(strategy->getSlicingData().masterOperand));
    }
}

TEST_F(SRAMManagementTest, graphsizeoptimization_noncommondim4dsolver)
{
    // tensors are large enough so we can have multiple strategies.
    const TSize b = 300, h = 10, w = 20;
    const TSize c = 5, k = 5000;
    synConvolutionParams params;
    params.kH = 3, params.kW = 2;

    const TSize ifmSizes[] = {c, w, h, b};
    const TSize wSizes[]   = {k, c, params.kW, params.kH};
    const TSize outSizes[] = {k,
                              convOutputDimSize(w, params.kW, params.dW, params.padL + params.padR, params.dilW),
                              convOutputDimSize(h, params.kH, params.dH, params.padT + params.padB, params.dilH),
                              b}; // {k,w-s+1,h-r+1,b}

    pTensor ifm(std::make_shared<Tensor>(ARRAY_SIZE(ifmSizes), ifmSizes, syn_type_float));
    pTensor wgh(std::make_shared<Tensor>(ARRAY_SIZE(wSizes), wSizes, syn_type_float));
    pTensor ofm(std::make_shared<Tensor>(ARRAY_SIZE(outSizes), outSizes, syn_type_float));

    pBundle bundle =
        createSingleMMENodeBundle({ifm, wgh}, {ofm}, NodeFactory::convolutionNodeTypeName, &params, sizeof(params));

    SlicingBrain::knobs.maxSRAMCapInBytes = (ifm->getDenseSizeInBytes() + wgh->getDenseSizeInBytes())/2;
    SlicingBrain::knobs.maxNarrowSliceSize  = 1024;
    SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon4D = 1000;
    SlicingBrain::knobs.graphSizeOptimizationMultiplicationFactor = 2;
    NonCommon4DSolver solver(*getGraph().getHALReader(), bundle);
    ASSERT_TRUE(solver.effectiveForBundle());
    solver.createAllStrategies();
    solver.AddStrategiesForGraphSizeOptimization();
    SlicingStrategyList strategies = solver.getStrategies();

    for (const MmeGeometry& geometry : {gaudi_geometry_1wx4h, gaudi_geometry_2wx2h, gaudi_geometry_4wx1h})
    {
        // we should have numerous strategies as from the original algorithm
        long countOriginalAlgorithm = std::count_if(strategies.begin(), strategies.end(),
                                                   [&](const SlicingStrategyPtr& s)
                                                   {
                                                       pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
                                                       return (strategy->getMmeSlicingData().MMEGeometryUsed == geometry &&
                                                               !strategy->getGraphSizeOptimized());
                                                   });
        ASSERT_GT(countOriginalAlgorithm, 1);

        // since the tensors are big enough(above knobs), we should have multiple "graph size optimized" strategies
        long countGraphSizeOptimized = std::count_if(strategies.begin(), strategies.end(),
                                                   [&](const SlicingStrategyPtr& s)
                                                   {
                                                       pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
                                                       return (strategy->getMmeSlicingData().MMEGeometryUsed == geometry &&
                                                               strategy->getGraphSizeOptimized());
                                                   });
        ASSERT_GT(countGraphSizeOptimized, 1);

        // the strategy with the least sram capacity (per geometry) should be the created by the original algorithm,
        // and the strategy with the max sram capacity should be created by the graph size optimization solver
        const auto& strategiesWithMinMaxSRAMCap = std::minmax_element(strategies.begin(), strategies.end(),
                                                 [&](const SlicingStrategyPtr& a, const SlicingStrategyPtr& b)
                                                 {
                                                     return a->getMetrics().SRAMCapacity < b->getMetrics().SRAMCapacity;
                                                 });
        ASSERT_FALSE((*strategiesWithMinMaxSRAMCap.first)->getGraphSizeOptimized());
        ASSERT_TRUE((*strategiesWithMinMaxSRAMCap.second)->getGraphSizeOptimized());
    }

    for (const auto& strategy : strategies)
    {
        strategy->printLog(1, synapse::LogManager::LogType::SRAM_SLICE);
        // all strategies supposed to be limited by the sram capacity
        ASSERT_TRUE(strategy->getMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes);
        // both inputs are supposed to be sliced in all strategies
        ASSERT_TRUE(!SlicedOperandUtils::isTriviallySliced(strategy->getSlicingData().bundleTensors[0]));
        ASSERT_TRUE(!SlicedOperandUtils::isTriviallySliced(strategy->getSlicingData().bundleTensors[1]));
        ASSERT_TRUE(!SlicedOperandUtils::isTriviallySliced(strategy->getSlicingData().masterOperand));
    }
}

TEST_F(SRAMManagementTest, graphsizeoptimization_commondimsolver4d)
{
    // since b*h*w=255*64*64 is bigger than knobs.maxWideSliceSizeForLinearIncrement_common = 512,
    // the common dim solver is used. Since tensors are 4d the slicing is on the batch dimension
    const TSize b = 255, h = 64, w = 64;
    const TSize c = 16, k = 4;
    synConvolutionParams params;
    const TSize xSizes[]    = {c, w, h, b};
    const TSize dedwSizes[] = {k, c, params.kW, params.kH};
    const TSize dedySizes[] = {k,
                               convOutputDimSize(w, params.kW, params.dW, params.padL + params.padR, params.dilW),
                               convOutputDimSize(h, params.kH, params.dH, params.padT + params.padB, params.dilH),
                               b}; // {k,w,h,b}

    pTensor x(std::make_shared<Tensor>(ARRAY_SIZE(xSizes), xSizes, syn_type_float));
    pTensor dedy(std::make_shared<Tensor>(ARRAY_SIZE(dedySizes), dedySizes, syn_type_float));
    pTensor dedwOut(std::make_shared<Tensor>(ARRAY_SIZE(dedwSizes), dedwSizes, syn_type_float));

    pBundle bundle =
        createSingleMMENodeBundle({dedy, x}, {dedwOut}, NodeFactory::deDwNodeTypeName, &params, sizeof(params));
    SlicingBrain::knobs.maxSRAMCapInBytes = 18 * 1024 * 1024;
    SlicingBrain::knobs.minCDSizeForPartials = 512;
    SlicingBrain::knobs.graphSizeOptimizationMultiplicationFactor = 2;

    CommonDimSlicingSolver solver(*getGraph().getHALReader(), bundle);
    ASSERT_TRUE(solver.effectiveForBundle());
    solver.createAllStrategies();
    solver.AddStrategiesForGraphSizeOptimization();
    SlicingStrategyList strategies = solver.getStrategies();

    //we're supposed to have exactly one strategy as the original algorithm
    long countOriginalAlgorithm = std::count_if(strategies.begin(), strategies.end(),
                                                [&](const SlicingStrategyPtr& strategy)
                                                {
                                                    return (!strategy->getGraphSizeOptimized());
                                                });
    ASSERT_EQ(countOriginalAlgorithm, 1);

    //we're supposed to have multiple "graph size optimized" strategies
    long countGraphSizeOptimized = std::count_if(strategies.begin(), strategies.end(),
                                                  [&](const SlicingStrategyPtr& strategy)
                                                 {
                                                     return (strategy->getGraphSizeOptimized());
                                                 });
    ASSERT_GT(countGraphSizeOptimized, 1);

    // the strategy with the least sram capacity should be the one created by the original algorithm, and the strategy
    // with the max sram capacity should be created by the graph size optimization solver
    const auto& strategiesWithMinMaxSRAMCap = std::minmax_element(strategies.begin(), strategies.end(),
                                                 [&](const SlicingStrategyPtr& a, const SlicingStrategyPtr& b)
                                                 {
                                                     return a->getMetrics().SRAMCapacity < b->getMetrics().SRAMCapacity;
                                                 });
    ASSERT_FALSE((*strategiesWithMinMaxSRAMCap.first)->getGraphSizeOptimized());
    ASSERT_TRUE((*strategiesWithMinMaxSRAMCap.second)->getGraphSizeOptimized());


    for (const auto& strategy : strategies)
    {
        strategy->printLog(1, synapse::LogManager::LogType::SRAM_SLICE);
        // all strategies supposed to be limited by the sram capacity
        ASSERT_TRUE(strategy->getMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes);
        // both input tensors supposed to be sliced in all strategies
        ASSERT_TRUE(!SlicedOperandUtils::isTriviallySliced(strategy->getSlicingData().bundleTensors[0]));
        ASSERT_TRUE(!SlicedOperandUtils::isTriviallySliced(strategy->getSlicingData().bundleTensors[1]));
    }
}

TEST_F(SRAMManagementTest, graphsizeoptimization_commondimsolver2d)
{
    // since b*h*w=1*1*2000 is bigger than knobs.maxWideSliceSizeForLinearIncrement_common = 512,
    // the common dim solver is used. Since tensors are (de facto) 2d the slicing is on the W dimension
    const TSize b = 1, h = 1, w = 2000;
    const TSize c = 16, k = 4;
    synConvolutionParams params;

    const TSize xSizes[]    = {c, w, h, b};
    const TSize dedwSizes[] = {k, c, params.kW, params.kH};
    const TSize dedySizes[] = {k,
                               convOutputDimSize(w, params.kW, params.dW, params.padL + params.padR, params.dilW),
                               convOutputDimSize(h, params.kH, params.dH, params.padT + params.padB, params.dilH),
                               b}; // {k,w,h,b}

    pTensor x(std::make_shared<Tensor>(ARRAY_SIZE(xSizes), xSizes, syn_type_float));
    pTensor dedy(std::make_shared<Tensor>(ARRAY_SIZE(dedySizes), dedySizes, syn_type_float));
    pTensor dedwOut(std::make_shared<Tensor>(ARRAY_SIZE(dedwSizes), dedwSizes, syn_type_float));

    SlicingBrain::knobs.maxSRAMCapInBytes = 18 * 1024 * 1024;
    SlicingBrain::knobs.minCDSizeForPartials = 512;
    SlicingBrain::knobs.graphSizeOptimizationMultiplicationFactor = 2;
    pBundle bundle =
        createSingleMMENodeBundle({dedy, x}, {dedwOut}, NodeFactory::deDwNodeTypeName, &params, sizeof(params));

    CommonDimSlicingSolver solver(*getGraph().getHALReader(), bundle);
    ASSERT_TRUE(solver.effectiveForBundle());
    solver.createAllStrategies();
    solver.AddStrategiesForGraphSizeOptimization();
    SlicingStrategyList strategies = solver.getStrategies();

    //we're supposed to have exactly one strategy as the original algorithm
    long countOriginalAlgorithm = std::count_if(strategies.begin(), strategies.end(),
                                                [&](const SlicingStrategyPtr& strategy)
                                                {
                                                    return (!strategy->getGraphSizeOptimized());
                                                });
    ASSERT_EQ(countOriginalAlgorithm, 1);

    //we're supposed to have 1 strategy not as the original algorithm
    long countGraphSizeOptimized = std::count_if(strategies.begin(), strategies.end(),
                                                  [&](const SlicingStrategyPtr& strategy)
                                                 {
                                                     return (strategy->getGraphSizeOptimized());
                                                 });
    ASSERT_EQ(countGraphSizeOptimized, 1);

    // the strategy with the least sram capacity should be the one created by the original algorithm, and the strategy
    // with the max sram capacity should be the one created by the graph size optimization solver
    const auto& strategiesWithMinMaxSRAMCap = std::minmax_element(strategies.begin(), strategies.end(),
                                                 [&](const SlicingStrategyPtr& a, const SlicingStrategyPtr& b)
                                                 {
                                                     return a->getMetrics().SRAMCapacity < b->getMetrics().SRAMCapacity;
                                                 });
    ASSERT_FALSE((*strategiesWithMinMaxSRAMCap.first)->getGraphSizeOptimized());
    ASSERT_TRUE((*strategiesWithMinMaxSRAMCap.second)->getGraphSizeOptimized());

    for (const auto& strategy : strategies)
    {
        strategy->printLog(1, synapse::LogManager::LogType::SRAM_SLICE);
        // all strategies supposed to be limited by the sram capacity
        ASSERT_TRUE(strategy->getMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes);
    }
}

TEST_F(SRAMManagementTest, brain_should_return_sliced_strategy_even_when_trivial_solver_is_effective)
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_MULTIPLE_SOLVERS_ENABLED, "false");
    std::vector<TSize> sizes = {1200, 1200};
    std::shared_ptr<Bundle> bundle = createSingleMMENodeBundle(
            {createTensor(sizes, syn_type_bf16), createTensor(sizes,syn_type_bf16)},
            {createTensor(sizes, syn_type_bf16)},
            NodeFactory::convolutionNodeTypeName);

    SlicingBrain::knobs.maxNarrowSliceSize  = 1024;
    SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon2D = 1000;

    for (bool multipleSolversEnabled : {false, true})
    {
        GCFG_SRAM_SLICER_MULTIPLE_SOLVERS_ENABLED.setValue(multipleSolversEnabled);
        MMESlicingBrain brain{getGraph()};
        SlicingStrategyList strategies = brain.getSolutionStrategies(bundle);
        long countSliced = std::count_if(strategies.begin(), strategies.end(),
                                         [&](const SlicingStrategyPtr &strategy)
                                         {
                                             return !SlicedOperandUtils::isTriviallySliced(*strategy);
                                         });
        if (multipleSolversEnabled) // 2d solver should add strategies (in addition to the trivial solver)
        {
            ASSERT_GT(strategies.size(), 1);
            ASSERT_GT(countSliced, 0);
        }
        else // only trivial solution should be chosen
        {
            ASSERT_EQ(strategies.size(), 1);
            ASSERT_EQ(countSliced, 0);
        }
    }
}

TEST_F(SRAMManagementTest, tpc_producer_stitching_to_narrow_when_dedw_sliced_on_multiple_dims)
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_4D_DEDW_SPATIAL_SLICE_ENABLED, "false"); // TODO - SW-25560 support bundle expansion for spatial slicing

    const TSize b = 3, hX = 140, wX = 140;
    const TSize c = 150, k = 200;
    synConvolutionParams params;
    params.dW = 2, params.dH = 2;

    TSize xSizes[] = {c, wX, hX, b};
    TSize dedwSizes[] = {k, c, params.kW, params.kH};
    TSize dedySizes[] = {k,
                         convOutputDimSize(wX, params.kW, params.dW, params.padL + params.padR, params.dilW),
                         convOutputDimSize(hX, params.kH, params.dH, params.padT + params.padB, params.dilH),
                         b}; // {k,wX/2,hX/2,b}

    pTensor reluIn(std::make_shared<Tensor>(ARRAY_SIZE(xSizes), xSizes, syn_type_float));
    reluIn->setName("reluIn");
    pTensor dedy(std::make_shared<Tensor>(ARRAY_SIZE(dedySizes), dedySizes, syn_type_float));
    dedy->setName("dedy");
    pTensor x(std::make_shared<Tensor>(ARRAY_SIZE(xSizes), xSizes, syn_type_float));
    x->setName("x");
    pTensor dedwOut(std::make_shared<Tensor>(ARRAY_SIZE(dedwSizes), dedwSizes, syn_type_float));
    dedwOut->setName("dedwOut");

    pNode relu = NodeFactory::createNode({reluIn}, {x}, nullptr, "relu_fwd_f32", "relu");
    pNode dedw = NodeFactory::createNode({dedy, x}, {dedwOut}, &params, NodeFactory::deDwNodeTypeName, "dedw");
    GraphEditor::addNode(getGraph(), dedw);
    GraphEditor::addNode(getGraph(), relu);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    SlicingBrain::knobs.minCDSizeForPartials = 512;
    SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon2D = 32768;
    SlicingBrain::knobs.maxSRAMCapInBytes = 18 * 1024 * 1024;
    SlicingBrain::knobs.maxNarrowSliceSize = 32768;
    SlicingBrain::knobs.graphSizeOptimizationMultiplicationFactor = 2;

    Bundlizer bundlizer(getGraph());
    pBundle bundle = bundlizer.getMMEBundles().front();
    auto strategies = getMmeBrain().getSolutionStrategies(bundle);
    ASSERT_NE(0, strategies.size()) << "brain did not return any strategy to solve the bundle";
    pMmeSlicingStrategy winningStrategy = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(strategies, bundle, getGraph(), getSlicingBrains()));
    const SlicingData& slicingData = winningStrategy->getMmeSlicingData();

    // make sure both mme inputs are sliced on multiple dimensions - this is the scenario of this test:
    for (const pSlicedOperand& op : slicingData.bundleTensors)
    {
        ASSERT_TRUE(SlicedOperandUtils::getNumOfSlicedDims(op) > 1);
    }
    unsigned numOfSlicesCommonDim = SlicedOperandUtils::nofSlices(slicingData.getWide(), DIM_B);

    // make sure it's possible to add the tpc producer to this bundle (in other words, that tpc stitching is
    // allowed in this scenario), and add the tpc node:
    pBundleExpansion expCnd = bundlizer.findNarrowTpcProducerExpansionCandidate(winningStrategy);
    ASSERT_EQ(expCnd->nodeToStitch, relu);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, x);
    TPCSlaveBrain tpcBrain{getGraph()};
    ASSERT_TRUE(tpcBrain.addProducerToStrategy(expCnd, winningStrategy));
    bundle->addNode(relu);

    // check solution:
    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, winningStrategy).fillSolution());
    const Solution &solution = bundle->getSolution();
    for (const auto& tensor : {reluIn, dedy, x, dedwOut})
    {
        // all original tensors should have 1 appearance in the solution operand list
        long countOriginalTensor = std::count_if(solution.operands.begin(), solution.operands.end(),
                                                 [&](const pSlicedOperand &operand)
                                                 { return (operand->originalTensor == tensor); });
        ASSERT_EQ(countOriginalTensor, 1);
    }

    unsigned numOfSlicesMMEOutput = 0;
    unsigned numOfSlicesStitchedOperand = 0;
    for (const auto& operand : solution.operands)
    {   // all operands are supposed to be sliced on multiple dims
        ASSERT_TRUE(SlicedOperandUtils::getNumOfSlicedDims(operand) > 1);
        if (operand->originalTensor == dedwOut)
        {
            numOfSlicesMMEOutput = SlicedOperandUtils::nofSlices(operand);
        }
        else  if (operand->originalTensor == x)
        {
            numOfSlicesStitchedOperand = SlicedOperandUtils::nofSlices(operand);
        }
        else if (operand->originalTensor == reluIn)
        {
            // the tpc should have the same number of slices as the stitched operand
            ASSERT_TRUE(numOfSlicesStitchedOperand > 0); // because it's supposed to be already filled
            ASSERT_EQ(SlicedOperandUtils::nofSlices(operand), numOfSlicesStitchedOperand);
        }
    }
    // numOfSlicesCommonDim is the number of partials per output slice , so:
    unsigned totalExpectedOutputOperations = numOfSlicesMMEOutput * numOfSlicesCommonDim;
    unsigned totalExpectedTPCOperations = numOfSlicesStitchedOperand;
    unsigned totalExpectedOperations = totalExpectedOutputOperations + totalExpectedTPCOperations;
    ASSERT_EQ(solution.operations.size(), totalExpectedOperations) << "wrong number of operations in solution";

    // check operations order
    // when stitching to narrow it should be (because of double buffer):
    // tpc, tpc, mme, tpc, mme, tpc, mme, tpc, mme... mme, mme, mme...
    int firstOp = 1;
    for (auto iter = solution.operations.begin(); iter != solution.operations.end();)
    {
        if (totalExpectedTPCOperations > 0)
        {
            for (int j = 0; j < 1 + firstOp; ++j) // the code inside will run twice only in 1st iteration
            {
                ASSERT_TRUE(iter->originalNode == relu);
                ASSERT_TRUE(iter->outputs.front()->operand->resideInSRAM);
                ASSERT_FALSE(iter->inputs.front()->operand->resideInSRAM);
                --totalExpectedTPCOperations;
                ++iter;
            }
            firstOp = 0;
        }
        ASSERT_TRUE(iter->originalNode == dedw);
        ASSERT_TRUE(iter->outputs.front()->operand->resideInSRAM);
        for (const auto &input : iter->inputs)
        {
            ASSERT_TRUE(input->operand->resideInSRAM);
        }
        ++iter;
    }
}

TEST_F(SRAMManagementTest, tpc_producer_stitching_to_wide_when_dedw_sliced_on_multiple_dims)
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_4D_DEDW_SPATIAL_SLICE_ENABLED, "false"); // TODO - SW-25560 support bundle expansion for spatial slicing

    const TSize b = 3, hX = 140, wX = 140;
    const TSize c = 150, k = 200;
    synConvolutionParams params;
    params.dW = 2, params.dH = 2;

    TSize xSizes[] = {c, wX, hX, b};
    TSize dedwSizes[] = {k, c, params.kW, params.kH};
    TSize dedySizes[] = {k,
                         convOutputDimSize(wX, params.kW, params.dW, params.padL + params.padR, params.dilW),
                         convOutputDimSize(hX, params.kH, params.dH, params.padT + params.padB, params.dilH),
                         b}; // {k,wX/2,hX/2,b}

    pTensor reluIn(std::make_shared<Tensor>(ARRAY_SIZE(dedySizes), dedySizes, syn_type_float));
    reluIn->setName("reluIn");
    pTensor dedy(std::make_shared<Tensor>(ARRAY_SIZE(dedySizes), dedySizes, syn_type_float));
    dedy->setName("dedy");
    pTensor x(std::make_shared<Tensor>(ARRAY_SIZE(xSizes), xSizes, syn_type_float));
    x->setName("x");
    pTensor dedwOut(std::make_shared<Tensor>(ARRAY_SIZE(dedwSizes), dedwSizes, syn_type_float));
    dedwOut->setName("dedwOut");

    pNode relu = NodeFactory::createNode({reluIn}, {dedy}, nullptr, "relu_fwd_f32", "relu");
    pNode dedw = NodeFactory::createNode({dedy, x}, {dedwOut}, &params, NodeFactory::deDwNodeTypeName, "dedw");
    GraphEditor::addNode(getGraph(), dedw);
    GraphEditor::addNode(getGraph(), relu);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    SlicingBrain::knobs.minCDSizeForPartials = 512;
    SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon2D = 32768;
    SlicingBrain::knobs.maxSRAMCapInBytes = 18 * 1024 * 1024;
    SlicingBrain::knobs.maxNarrowSliceSize = 32768;
    SlicingBrain::knobs.graphSizeOptimizationMultiplicationFactor = 2;

    Bundlizer bundlizer(getGraph());
    pBundle bundle = bundlizer.getMMEBundles().front();
    auto strategies = getMmeBrain().getSolutionStrategies(bundle);
    ASSERT_NE(0, strategies.size()) << "brain did not return any strategy to solve the bundle";
    pMmeSlicingStrategy winningStrategy = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(strategies, bundle, getGraph(), getSlicingBrains()));
    const SlicingData &slicingData = winningStrategy->getMmeSlicingData();

    // make sure both mme inputs are sliced on multiple dimensions - this is the scenario of this test:
    for (const pSlicedOperand& op : slicingData.bundleTensors)
    {
        ASSERT_TRUE(SlicedOperandUtils::getNumOfSlicedDims(op) > 1);
    }
    unsigned numOfSlicesCommonDim = SlicedOperandUtils::nofSlices(slicingData.getWide(), DIM_B);

    // make sure it's possible to add the tpc producer to this bundle (in other words, that tpc stitching is
    // allowed in this scenario), and add the tpc node:
    pBundleExpansion expCnd = bundlizer.findWideTpcProducerExpansionCandidate(winningStrategy);
    ASSERT_EQ(expCnd->nodeToStitch, relu);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, dedy);
    TPCSlaveBrain tpcBrain{getGraph()};
    ASSERT_TRUE(tpcBrain.addProducerToStrategy(expCnd, winningStrategy));
    bundle->addNode(relu);

    // check solution:
    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, winningStrategy).fillSolution());
    const Solution &solution = bundle->getSolution();
    for (const auto& tensor : {reluIn, dedy, x, dedwOut})
    {
        // all original tensors should have 1 appearance in the solution operand list
        long countOriginalTensor = std::count_if(solution.operands.begin(), solution.operands.end(),
                                                 [&](const pSlicedOperand &operand)
                                                 { return (operand->originalTensor == tensor); });
        ASSERT_EQ(countOriginalTensor, 1);
    }

    unsigned numOfSlicesMMEOutput = 0;
    unsigned numOfSlicesStitchedOperand = 0;
    for (const auto& operand : solution.operands)
    {   // all operands are supposed to be sliced on multiple dims
        ASSERT_TRUE(SlicedOperandUtils::getNumOfSlicedDims(operand) > 1);
        if (operand->originalTensor == dedwOut)
        {
            numOfSlicesMMEOutput = SlicedOperandUtils::nofSlices(operand);
        }
        else if (operand->originalTensor == dedy)
        {
            numOfSlicesStitchedOperand = SlicedOperandUtils::nofSlices(operand);
        }
        else if (operand->originalTensor == reluIn)
        {
            // the tpc should have the same number of slices as the stitched operand
            ASSERT_TRUE(numOfSlicesStitchedOperand > 0); // because it's supposed to be already filled
            ASSERT_EQ(SlicedOperandUtils::nofSlices(operand), numOfSlicesStitchedOperand);
        }
    }
    // numOfSlicesCommonDim is the number of partials per output slice , so:
    unsigned totalExpectedOutputOperations = numOfSlicesMMEOutput * numOfSlicesCommonDim;
    unsigned totalExpectedTPCOperations = numOfSlicesStitchedOperand;
    unsigned totalExpectedOperations = totalExpectedOutputOperations + totalExpectedTPCOperations;
    ASSERT_EQ(solution.operations.size(), totalExpectedOperations) << "wrong number of operations in solution";

    // check operations order
    // when stitching to wide it should be (because of double buffer):
    // tpc, tpc, mme, tpc, mme, tpc, mme, tpc, mme... mme, mme, mme...  <-- this is a pattern
    // And then similarly (only the beginning is different - no 2 consecutive 2 tpcs):
    // tpc, mme, tpc, mme, tpc, mme, tpc, mme... mme, mme, mme...  <-- this is a similar pattern
    // And so on...
    // we can generalize this in the following way:
    // [firstTPC], multiple times * {[nonFirstTPC, mmeInsideTPCs, nonFirstTPC, mmeInsideTPCs,...], [mmeNotInsideTPCs, mmeNotInsideTPCs,...]}
    enum opState {initial, firstTPC, nonFirstTPC, mmeInsideTPCs, mmeNotInsideTPCs} state = initial;
    for (auto iter = solution.operations.begin(); iter != solution.operations.end(); ++iter)
    {
        ASSERT_TRUE(iter->originalNode == relu || iter->originalNode == dedw);
        if (iter->originalNode == relu)
        {
            ASSERT_TRUE(iter->outputs.front()->operand->resideInSRAM);
            ASSERT_FALSE(iter->inputs.front()->operand->resideInSRAM);
            --totalExpectedTPCOperations;
        }
        else if (iter->originalNode == dedw)
        {
            ASSERT_TRUE(iter->outputs.front()->operand->resideInSRAM);
            for (const auto &input : iter->inputs)
            {
                ASSERT_TRUE(input->operand->resideInSRAM);
            }
            --totalExpectedOutputOperations;
        }
        switch (state)
        {
            case initial:
            {
                if (iter->originalNode == relu)
                {
                    state = firstTPC;
                }
                else
                {
                    ASSERT_TRUE(false) << "wrong operation order - expected an initial tpc operation for double buffer";
                }
                break;
            }
            case firstTPC:
            {
                if (iter->originalNode == relu)
                {
                    state = nonFirstTPC;
                }
                else
                {
                    ASSERT_TRUE(false) << "wrong operation order - expected a second tpc operation for double buffer";
                }
                break;
            }
            case nonFirstTPC:
            {
                if (iter->originalNode == dedw)
                {
                    state = mmeInsideTPCs;
                }
                else
                {
                    ASSERT_TRUE(false) << "wrong operation order - expected an mme operation after tpc operation";
                }
                break;
            }
            case mmeInsideTPCs:
            {
                if (iter->originalNode == relu)
                {
                    state = nonFirstTPC;
                }
                else if (iter->originalNode == dedw)
                {
                    state = mmeNotInsideTPCs;
                }
                break;
            }
            case mmeNotInsideTPCs:
            {
                if (iter->originalNode == dedw)
                {
                    state = mmeNotInsideTPCs;
                }
                else if (iter->originalNode == relu)
                {
                    state = firstTPC;
                }
                break;
            }
        }
    }
    ASSERT_EQ(totalExpectedTPCOperations, 0) << "wrong number of TPC operations in final solution";
    ASSERT_EQ(totalExpectedOutputOperations, 0) << "wrong number of dedw operations in final solution";
}

TEST_F(SRAMManagementTest, dedx_dedw_bundle_should_have_interleaved_execution_when_dedx_walking_pattern_is_top_to_bottom)
{
    /*
     * Build dedx+dedw bundle where the dedx is the master, and it is sliced to 4 n W and 2 on C,
     * dedw is the slave and it is sliced on the common dimension W.
     * force the strategy walking pattern to be Top-To-Bottom, which means that the dedx and dedw traverse
     * the dedy operand slices together while the first dedx column is computed. The dedw is fully computed
     * after that, so the dedx continue alone for another pass on the dedy slices.
     */

    // Build the graph
    const TSize
        b = 1,
        h = 1,
        w = 2048,
        k = 256,
        c = 128,
        r = 1,
        s = 1;

    TSize dySizes[]  = {k, w, h, b};
    TSize xSizes[]   = {c, w, h, b};
    TSize wghSizes[] = {k, c, s, r};

    pTensor dy(std::make_shared<Tensor>(ARRAY_SIZE(dySizes), dySizes, syn_type_bf16));
    pTensor wgh(std::make_shared<Tensor>(ARRAY_SIZE(wghSizes), wghSizes, syn_type_bf16));
    pTensor dw(std::make_shared<Tensor>(ARRAY_SIZE(wghSizes), wghSizes, syn_type_float));
    pTensor x(std::make_shared<Tensor>(ARRAY_SIZE(xSizes), xSizes, syn_type_bf16));
    pTensor dx(std::make_shared<Tensor>(ARRAY_SIZE(xSizes), xSizes, syn_type_bf16));

    synConvolutionParams params{};
    pNode dedx = NodeFactory::createNode({dy, wgh}, {dx}, &params, NodeFactory::deDxNodeTypeName, "dedx");
    GraphEditor::addNode(getGraph(), dedx);
    pNode dedw = NodeFactory::createNode({dy, x}, {dw}, &params, NodeFactory::deDwNodeTypeName, "dedw");
    GraphEditor::addNode(getGraph(), dedw);

    // Use trivial solver to create a base strategy for dedx
    Bundlizer     bundlizer(getGraph());
    auto bundles = bundlizer.getMMEBundles();
    pBundle dxBundle = bundles.front();
    TrivialSolver solver(*getGraph().getHALReader(), dxBundle);
    solver.createAllStrategies();
    pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(solver.getStrategies().front());
    ASSERT_NE(nullptr, strategy);

    // Update strategy to get the desired slicing and walking pattern
    auto& slicingData = strategy->getMmeSlicingData();
    slicingData.masterOperand->chunkDimensions[DIM_W] =
            slicingData.bundleTensors[0]->chunkDimensions[DIM_W] = w / 4;
    slicingData.masterOperand->chunkDimensions[DIM_C] =
            slicingData.bundleTensors[1]->chunkDimensions[WEIGHT_DIM_C] = c / 2;
    strategy->setDoubleBuffer(true);
    slicingData.traversalPattern = SlicedOperandTraversalPattern::TOP_TO_BOTTOM_2D;

    // Expand the strategy with the dedw node
    MMESlaveBrain slaveBrain{getGraph()};
    SharedMMEInputCandidateHandler handler;
    std::list<pBundleExpansion> candidates = handler.findSharedMMEInputCandidate(strategy, getGraph());
    ASSERT_FALSE(candidates.empty());
    auto& candidate = candidates.front();
    ASSERT_EQ(candidate->nodeToStitch, dedw);
    auto adjustedCandidate = slaveBrain.adjustCandidateToStrategy(candidate, strategy);
    slicingData.addValidCandidate(adjustedCandidate);
    bundlizer.removeBundle(dedw);
    slaveBrain.addSharedOperandMme(adjustedCandidate, strategy);
    bundlizer.addCandidateToBundle(dxBundle, adjustedCandidate);

    // Generate solution and validate the execution order
    SolutionGenerator solGen(getGraph(), dxBundle, strategy);
    solGen.fillSolution();

    auto& solution = dxBundle->getSolution();
    auto iter = solution.operations.begin();
    for (int i = 0; i < 8; i++)
    {
        ASSERT_NE(iter, solution.operations.end());
        ASSERT_EQ(iter->originalNode, dedx) << "Expected dedx when i=" << i ;
        if (i < 4)
        {
            ASSERT_EQ(iter->outputs.front()->coordinates[DIM_C], 0)
                << "Expected first 4 slices to traverse the W dimension, but for i="
                << i << " dimension C was changed";

            // First four dedx should be interleaved with dedw
            iter++;
            ASSERT_EQ(iter->originalNode, dedw) << "Expected dedw when i=" << i;
        }
        else
        {
            ASSERT_EQ(iter->outputs.front()->coordinates[DIM_C], 1)
                << "Expected the last 4 slices to traverse the W dimension, but for i="
                << i << " dimension C was changed";
        }
        iter++;
    }
}

TEST_F(SRAMManagementTest, dedw_master_and_dedx_slave_bundle_with_interleaved_execution)
{
    // Build the graph
    const TSize
        b = 1,
        h = 1,
        w = 2048,
        k = 256,
        c = 128,
        r = 1,
        s = 1;

    TSize dySizes[]  = {k, w, h, b};
    TSize xSizes[]   = {c, w, h, b};
    TSize wghSizes[] = {k, c, s, r};

    pTensor dy(std::make_shared<Tensor>(ARRAY_SIZE(dySizes), dySizes, syn_type_bf16));
    pTensor wgh(std::make_shared<Tensor>(ARRAY_SIZE(wghSizes), wghSizes, syn_type_bf16));
    pTensor dw(std::make_shared<Tensor>(ARRAY_SIZE(wghSizes), wghSizes, syn_type_float));
    pTensor x(std::make_shared<Tensor>(ARRAY_SIZE(xSizes), xSizes, syn_type_bf16));
    pTensor dx(std::make_shared<Tensor>(ARRAY_SIZE(xSizes), xSizes, syn_type_bf16));

    synConvolutionParams params{};
    pNode dedx = NodeFactory::createNode({dy, wgh}, {dx}, &params, NodeFactory::deDxNodeTypeName, "dedx");
    GraphEditor::addNode(getGraph(), dedx);
    pNode dedw = NodeFactory::createNode({dy, x}, {dw}, &params, NodeFactory::deDwNodeTypeName, "dedw");
    GraphEditor::addNode(getGraph(), dedw);

    // Use trivial solver to create a base strategy for dedw
    Bundlizer     bundlizer(getGraph());
    auto bundles = bundlizer.getMMEBundles();
    pBundle dwBundle = bundles.back();
    TrivialSolver solver(*getGraph().getHALReader(), dwBundle);
    solver.createAllStrategies();
    pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(solver.getStrategies().front());
    ASSERT_NE(nullptr, strategy);

    // Update strategy to get the desired slicing and walking pattern
    auto& slicingData = strategy->getMmeSlicingData();
    slicingData.bundleTensors[0]->chunkDimensions[DIM_W] = w / 4;
    slicingData.bundleTensors[1]->chunkDimensions[DIM_W] = w / 4;
    slicingData.masterOperand->resideInSRAM = true;
    strategy->setDoubleBuffer(true);

    // Expand the strategy with the dedx node
    MMESlaveBrain slaveBrain{getGraph()};
    SharedMMEInputCandidateHandler handler;
    std::list<pBundleExpansion> candidates = handler.findSharedMMEInputCandidate(strategy, getGraph());
    ASSERT_FALSE(candidates.empty());
    auto& candidate = candidates.front();
    ASSERT_EQ(candidate->nodeToStitch, dedx);
    auto adjustedCandidate = slaveBrain.adjustCandidateToStrategy(candidate, strategy);
    slicingData.addValidCandidate(adjustedCandidate);
    bundlizer.removeBundle(dedx);
    slaveBrain.addSharedOperandMme(adjustedCandidate, strategy);
    bundlizer.addCandidateToBundle(dwBundle, adjustedCandidate);

    // Generate solution and validate the execution order
    SolutionGenerator solGen(getGraph(), dwBundle, strategy);
    solGen.fillSolution();

    auto& solution = dwBundle->getSolution();
    auto iter = solution.operations.begin();
    for (int i = 0; i < 4; i++)
    {
        ASSERT_NE(iter, solution.operations.end());
        ASSERT_EQ(iter->originalNode, dedw) << "Expected dedw when i=" << i ;
        ASSERT_EQ(iter->outputs.front()->coordinates[DIM_C], 0)
            << "Expected first 4 slices to traverse the W dimension, but for i="
            << i << " dimension C was changed";
        ASSERT_EQ(iter->inputs.front()->coordinates[DIM_W], i)
            << "Expected slice index i="
            << i << "of dimension W but received index " << iter->inputs.front()->coordinates[DIM_W];
        // First four dedx should be interleaved with dedw
        iter++;
        ASSERT_EQ(iter->originalNode, dedx) << "Expected dedx when i=" << i;
        ASSERT_EQ(iter->inputs.front()->coordinates[DIM_W], i)
            << "Expected slice index i="
            << i << "of dimension W but received index " << iter->inputs.front()->coordinates[DIM_W];
        iter++;
    }
    ASSERT_EQ(iter, solution.operations.end());
}

TEST_F(SRAMManagementTest, multi_tpc_producer_stitching)
{
    const TSize c = 512, k = 3072, h = 2048;

    TSize operandASizes[] = {c, h};
    TSize operandBSizes[] = {k, c};
    TSize outSize[] = {k, h};

    pTensor reluIn(std::make_shared<Tensor>(ARRAY_SIZE(operandASizes), operandASizes, syn_type_float));
    reluIn->setName("reluIn");
    pTensor operandA(std::make_shared<Tensor>(ARRAY_SIZE(operandASizes), operandASizes, syn_type_float));
    operandA->setName("operandA");
    pTensor secondReluIn(std::make_shared<Tensor>(ARRAY_SIZE(operandBSizes), operandBSizes, syn_type_float));
    secondReluIn->setName("secondReluIn");
    pTensor operandB(std::make_shared<Tensor>(ARRAY_SIZE(operandBSizes), operandBSizes, syn_type_float));
    operandB->setName("operandB");
    pTensor out(std::make_shared<Tensor>(ARRAY_SIZE(outSize), outSize, syn_type_float));
    out->setName("out");

    pNode relu = NodeFactory::createNode({reluIn}, {operandA}, nullptr, "relu_fwd_f32", "relu");
    pNode secondrelu = NodeFactory::createNode({secondReluIn}, {operandB}, nullptr, "relu_fwd_f32", "secondRelu");

    synGEMMParams params{};
    pNode gemm = NodeFactory::createNode({operandA, operandB}, {out}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    GraphEditor::addNode(getGraph(), relu);
    GraphEditor::addNode(getGraph(), secondrelu);
    GraphEditor::addNode(getGraph(), gemm);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    Bundlizer bundlizer(getGraph());
    pBundle bundle = bundlizer.getMMEBundles().front();
    TrivialSolver solver(*getGraph().getHALReader(), bundle);
    solver.createAllStrategies();
    pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(solver.getStrategies().front());
    ASSERT_NE(nullptr, strategy);

    // Update strategy to get the desired slicing and walking pattern
    auto& slicingData = strategy->getMmeSlicingData();
    slicingData.masterOperand->chunkDimensions[DIM_W] =
            slicingData.bundleTensors[0]->chunkDimensions[DIM_W] = h / 2;
    slicingData.masterOperand->chunkDimensions[DIM_C] =
            slicingData.bundleTensors[1]->chunkDimensions[WEIGHT_DIM_C] = k / 3;
    strategy->setDoubleBuffer(true);
    slicingData.traversalPattern = SlicedOperandTraversalPattern::LEFT_TO_RIGHT_2D;

    // Add wide TPC producer
    pBundleExpansion expCnd = bundlizer.findWideTpcProducerExpansionCandidate(strategy);
    ASSERT_EQ(expCnd->nodeToStitch, relu);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, operandA);
    TPCSlaveBrain tpcBrain{getGraph()};
    ASSERT_TRUE(tpcBrain.addProducerToStrategy(expCnd, strategy));
    bundle->addNode(relu);

    // Add narrow TPC producer
    expCnd = bundlizer.findNarrowTpcProducerExpansionCandidate(strategy);
    ASSERT_EQ(expCnd->nodeToStitch, secondrelu);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, operandB);
    ASSERT_TRUE(tpcBrain.addProducerToStrategy(expCnd, strategy));
    bundle->addNode(secondrelu);

    // check solution:
    ASSERT_TRUE(SolutionGenerator(getGraph(), bundle, strategy).fillSolution());
   const Solution &solution = bundle->getSolution();
   // We expect 4 TPC nodes (first and second relu double buffer)
   // And then MME TPC MME MME ...
   uint32_t index = 0;
   for (const auto& operation : solution.operations)
   {
       switch (index)
       {
       // 3 Narrow parts
       case 0:
       case 2:
       case 5:
           ASSERT_EQ(secondrelu, operation.originalNode) << "Wrong node at index: " << index;
           break;
       // 2 Wide Parts
       case 1:
       case 3:
           ASSERT_EQ(relu, operation.originalNode) << "Wrong node at index: " << index;
           break;
       case 4:
       case 6:
       case 7:
       case 8:
       case 9:
       case 10:
           ASSERT_EQ(gemm, operation.originalNode) << "Wrong node at index: " << index;
           break;
       }
       ++index;
   }
   // Expect 11 operations
   ASSERT_EQ(index, 11);
}

void validateStrategies(GaudiGraph& g, const std::set<pNode>& mustExclude)
{
    std::unordered_map<pBundle, BundleSolvingData> solvingDataPerBundle;
    Bundlizer  bundlizer(g);
    BundleList bundles = bundlizer.getMMEBundles();
    AllBrains  brains(g);

    for (pBundle& bundle : bundles)
    {
        solvingDataPerBundle[bundle].strategies = brains.m_mmeBrain.getSolutionStrategies(bundle);
    }

    for (pBundle& bundle : bundles)
    {
        BundleExpander expander(g, brains, bundlizer, solvingDataPerBundle);
        SlicingStrategyList expandedStrategies = expander.generateExpandedStrategies(bundle);

        for (auto& s : expandedStrategies)
        {
            pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
            std::set<pNode> nodes(bundle->getNodes().begin(), bundle->getNodes().end());

            for(const pBundleExpansion& candidate : strategy->getMmeSlicingData().getRoleCandidates())
            {
                if(candidate)
                {
                    nodes.insert(candidate->nodeToStitch);
                }
            }

            bool includesIllegalCombination = std::includes(nodes.begin(), nodes.end(),
                                                            mustExclude.begin(), mustExclude.end());

            auto getNodeName = [](const pNode& n){ return n->getNodeName(); };
            EXPECT_FALSE(includesIllegalCombination)
                << "illegal strategy combination, strategy nodes: [" << toString(nodes, ',', getNodeName) << "]"
                << ", illegal combination nodes: [" << toString(mustExclude, ',', getNodeName) << "]";
        }
    }
}

//                      t_1
//                       |
//               t_0   (TPC_A)   t_2
//                \    /   \     /
//                 \  /     \   /
//                (TPC_B)   (MME)
//                    \     /
//                     \   /
//                    (TPC_C)
//                       |
// TPC A and TPC C shouldn't be in the same strategy
TEST_F(SRAMManagementTest, bundle_expender_single_producers)
{
    GaudiGraph g;

    // TPC A
    pTensor tpc_a_in_0 = createTensor({1, 1}, syn_type_float);
    pTensor tpc_a_out_0 = createTensor({1, 1}, syn_type_bf16, false);
    pNode tpc_a = NodeFactory::createNode({tpc_a_in_0}, {tpc_a_out_0},
                                          nullptr, "cast_f32_to_bf16", "TPC-A");
    ASSERT_TRUE(GraphEditor::addNode(g, tpc_a));

    // TPC B
    pTensor tpc_b_in_0 = createTensor({1, 1}, syn_type_bf16);
    pTensor tpc_b_out_0 = createTensor({1, 1}, syn_type_bf16, false);
    pNode   tpc_b = NodeFactory::createNode({tpc_b_in_0, tpc_a_out_0}, {tpc_b_out_0}, nullptr, "add_f32", "TPC-B");
    ASSERT_TRUE(GraphEditor::addNode(g, tpc_b));

    // MME
    pTensor mme_in_0 = createTensor({1, 1}, syn_type_bf16);
    pTensor mme_out_0 = createTensor({1, 1}, syn_type_bf16, false);
    pNode mme = NodeFactory::createNode({tpc_a_out_0, mme_in_0}, {mme_out_0},
                                        nullptr, NodeFactory::gemmNodeTypeName, "MME");
    ASSERT_TRUE(GraphEditor::addNode(g, mme));

    // TPC C
    pTensor tpc_c_out_0 = createTensor({1, 1}, syn_type_bf16);
    pNode   tpc_c       = NodeFactory::createNode({tpc_b_out_0, mme_out_0}, {tpc_c_out_0}, nullptr, "add_f32", "TPC-C");
    ASSERT_TRUE(GraphEditor::addNode(g, tpc_c));

    std::set<pNode> mustExclude = { tpc_a, tpc_c };
    validateStrategies(g, mustExclude);
}

//                 t_0         t_1
//                  |           |
//               (TPC_A)     (TPC_B)
//                 \   \__     /
//                  \     \   /
//                   \    (MME)
//                    \     /
//                     \   /
//                    (TPC_C)
//                       |
// TPC A and TPC C shouldn't be in the same strategy
TEST_F(SRAMManagementTest, bundle_expender_two_producers)
{
    GaudiGraph g;

    // TPC A
    pTensor tpc_a_in_0 = createTensor({1, 1}, syn_type_float);
    pTensor tpc_a_out_0 = createTensor({1, 1}, syn_type_bf16, false);
    pNode tpc_a = NodeFactory::createNode({tpc_a_in_0}, {tpc_a_out_0},
                                          nullptr, "cast_f32_to_bf16", "TPC-A");
    ASSERT_TRUE(GraphEditor::addNode(g, tpc_a));

    // TPC B
    pTensor tpc_b_in_0 = createTensor({1, 1}, syn_type_float);
    pTensor tpc_b_out_0 = createTensor({1, 1}, syn_type_bf16, false);
    pNode tpc_b = NodeFactory::createNode({tpc_b_in_0}, {tpc_b_out_0},
                                          nullptr, "cast_f32_to_bf16", "TPC-B");
    ASSERT_TRUE(GraphEditor::addNode(g, tpc_b));

    // MME
    synGEMMParams gemmParams{};
    pTensor mme_out_0 = createTensor({1, 1}, syn_type_bf16, false);
    pNode mme = NodeFactory::createNode({tpc_a_out_0, tpc_b_out_0}, {mme_out_0},
                                        &gemmParams, NodeFactory::gemmNodeTypeName, "MME");
    ASSERT_TRUE(GraphEditor::addNode(g, mme));

    // TPC C
    pTensor tpc_c_out_0 = createTensor({1, 1}, syn_type_bf16);
    pNode   tpc_c       = NodeFactory::createNode({tpc_a_out_0, mme_out_0}, {tpc_c_out_0}, nullptr, "add_f32", "TPC-C");
    ASSERT_TRUE(GraphEditor::addNode(g, tpc_c));

    std::set<pNode> mustExclude = { tpc_a, tpc_c };

    validateStrategies(g, mustExclude);
}

//                 t_0       t_1
//                 |  \_____/ |
//                 |  /     \ |
//               (MME_A)   (MME_B)
//                    \     /
//                     \   /
//                     (TPC)
//                       |
// MME A, MME C and TPC shouldn't be in the same strategy
TEST_F(SRAMManagementTest, bundle_expender_two_mme)
{
    GaudiGraph g;

    synConvolutionParams convParams{};
    pTensor mme_in_0 = createTensor({1, 1}, syn_type_bf16);
    pTensor mme_in_1 = createTensor({1, 1}, syn_type_bf16);

    // MME A
    pTensor mme_a_out_0 = createTensor({1, 1}, syn_type_bf16);
    pNode mme_a = NodeFactory::createNode({mme_in_0, mme_in_1}, {mme_a_out_0},
                                          &convParams, NodeFactory::deDxNodeTypeName, "MME-A");
    ASSERT_TRUE(GraphEditor::addNode(g, mme_a));

    // MME B
    pTensor mme_b_out_0 = createTensor({1, 1}, syn_type_bf16);
    pNode mme_b = NodeFactory::createNode({mme_in_0, mme_in_1}, {mme_b_out_0},
                                          &convParams, NodeFactory::deDwNodeTypeName, "MME-B");
    ASSERT_TRUE(GraphEditor::addNode(g, mme_b));

    // TPC A
    pTensor tpc_out_0 = createTensor({1, 1}, syn_type_bf16);
    pNode   tpc       = NodeFactory::createNode({mme_a_out_0, mme_b_out_0}, {tpc_out_0}, nullptr, "add_f32", "TPC");
    ASSERT_TRUE(GraphEditor::addNode(g, tpc));

    ASSERT_TRUE(loadTpcKernels(g));

    std::set<pNode> mustExclude = { mme_a, mme_b, tpc };

    validateStrategies(g, mustExclude);
}

TEST_F(SRAMManagementTest, sram_management_should_flatten_mme_node_when_possible)
{
    setGlobalConfForTest(GCFG_ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING, "true");

    pTensor x = createTensor({64, 56, 56, 64}, syn_type_bf16);
    pTensor w = createTensor({1, 1, 64, 64}, syn_type_bf16);
    pTensor y = createTensor({64, 56, 56, 64}, syn_type_bf16);

    synConvolutionParams params{};
    pNode conv = NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    GraphEditor::addNode(getGraph(), conv);
    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));

    int reshapeNodesCounter = 0;
    //Since weights are 1X1 we expect 2 reshapes, on the input and on the output.
    for (const NodePtr& n : getGraph().getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_STATIC_RESHAPE)
        {
            //Flatten the input
            if (reshapeNodesCounter == 0)
            {
                ASSERT_TRUE(n->getOutput(0)->getAllSizesInElements() == MMENodeFlattener::getFlattenShape(x));
            }
                //Flatten the weights
            else if (reshapeNodesCounter == 1)
            {
                ASSERT_TRUE(n->getOutput(0)->getAllSizesInElements() == MMENodeFlattener::getFlattenShape(w));
            }
                //Flatten the output
            else if (reshapeNodesCounter == 2)
            {
                ASSERT_TRUE(n->getInput(0)->getAllSizesInElements() == MMENodeFlattener::getFlattenShape(y));
                ASSERT_TRUE(n->getOutput(0)->getAllSizesInElements() == y->getAllSizesInElements());
            }
            reshapeNodesCounter++;
        }
    }
    ASSERT_TRUE(reshapeNodesCounter == 3);
}

TEST_F(SRAMManagementTest, bundle_expander_should_create_strategies_with_all_possible_candidates_as_valid)
{
    // Create a network with 2 mmes sharing an input, with a tpc producer
    // to each mme input and a tpc consumer to each mme output

    std::vector<TSize> sizes = {128, 128};

    pTensor shared = createTensor(sizes, syn_type_bf16);

    pTensor gemm0Wgh = createTensor(sizes, syn_type_bf16);
    pTensor gemm0Out = createTensor(sizes, syn_type_bf16);

    pTensor gemm1Wgh = createTensor(sizes, syn_type_bf16);
    pTensor gemm1Out = createTensor(sizes, syn_type_bf16);

    pTensor reluInShared = createTensor(sizes, syn_type_bf16);
    pTensor reluInGemm0Wgh = createTensor(sizes, syn_type_bf16);
    pTensor reluInGemm1Wgh = createTensor(sizes, syn_type_bf16);

    pTensor reluOutGemm0 = createTensor(sizes, syn_type_bf16);
    pTensor reluOutGemm1 = createTensor(sizes, syn_type_bf16);

    synGEMMParams gemm0Params{};
    synGEMMParams gemm1Params{};

    pNode gemm0 = NodeFactory::createNode({shared, gemm0Wgh}, {gemm0Out}, &gemm0Params, NodeFactory::gemmNodeTypeName, "gemm0");
    GraphEditor::addNode(getGraph(), gemm0);
    pNode gemm1 = NodeFactory::createNode({shared, gemm1Wgh}, {gemm1Out}, &gemm1Params, NodeFactory::gemmNodeTypeName, "gemm1");
    GraphEditor::addNode(getGraph(), gemm1);

    unsigned i = 0;
    for (std::pair<pTensor, pTensor> reluOperands : std::vector<std::pair<pTensor, pTensor>>{
            {reluInShared, shared},
            {reluInGemm0Wgh, gemm0Wgh},
            {reluInGemm1Wgh, gemm1Wgh},
            {gemm0Out, reluOutGemm0},
            {gemm1Out, reluOutGemm1}
    })
    {
        std::string name{"relu"};
        pNode relu = NodeFactory::createNode({reluOperands.first}, {reluOperands.second}, nullptr, "relu_fwd_bf16", name + std::to_string(i++));
        GraphEditor::addNode(getGraph(), relu);
    }

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    // Create bundles and initial strategies
    AllBrains allBrains{getGraph()};
    SRAMSlicingManager sramManager{getGraph()};
    sramManager.generateInitialBundles();
    sramManager.generateInitialStrategies();
    auto bundlesSolvingData = sramManager.getBundlesSolvingData();

    // When expending a bundle (they are symmetric, so no matter which),
    BundleExpander bundleExpander{getGraph(), allBrains, sramManager.getBundlizer(), bundlesSolvingData};
    const pBundle& bundle = bundlesSolvingData.begin()->first;

    // If found a strategy with all possible candidates, done.
    const std::vector<BundleExpansion::Role> requiredRoles = {
            BundleExpansion::Role::WideInputProducer,
            BundleExpansion::Role::NarrowInputProducer,
            BundleExpansion::Role::OutputConsumer,
            BundleExpansion::Role::SharedInputConsumer,     // AKA: Slave
            BundleExpansion::Role::SlaveInputProducer,
            BundleExpansion::Role::SlaveOutputConsumer,
    };
    for (const auto& s : bundleExpander.generateExpandedStrategies(bundle))
    {
        pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
        const auto& validCandidates = strategy->getMmeSlicingData().getRoleCandidates();
        bool missingRole = false;
        for (auto role : requiredRoles)
        {
            if (!validCandidates[role] || !(validCandidates[role]->nodeToStitch))
            {
                missingRole = true;
                break;
            }
        }
        if (!missingRole)
        {
            // Found a strategy with everything!
            strategy->printLog(3, synapse::LogManager::LogType::SYN_TEST);
            return;
        }
    }
    FAIL() << "No strategy found with all required roles.";
}

TEST_F(SRAMManagementTest, bundle_tpc_nodes_simple)
{
    // Given (MME) -> (TPC) -> (TPC) -> (TPC) -> (MME)
    // Bundle the tpc's together
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph g;
    const unsigned dims = 2;
    TSize sizes[dims] = {16, 16};
    TensorVector tensors;
    for (int i = 0; i < 8; i++)
    {
        tensors.push_back(pTensor(new Tensor(dims, sizes, syn_type_float)));
    }

    synGEMMParams params{};

    pNode gemm1 = NodeFactory::createNode({tensors[0], tensors[1]}, {tensors[2]}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    pNode relu = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[3]}, nullptr, "relu_fwd_f32", "relu");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[4]}, nullptr, "relu_fwd_f32", "relu2");
    pNode relu3 = NodeFactory::createGenericTPCNode({tensors[4]}, {tensors[5]}, nullptr, "relu_fwd_f32", "relu3");
    pNode gemm2 = NodeFactory::createNode({tensors[5], tensors[6]}, {tensors[7]}, &params, NodeFactory::gemmNodeTypeName, "gemm2");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(GraphEditor::addNode(g, relu3));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm2));

    // When bundling the graph
    BundleList bundles = Bundlizer(g).getTPCBundles();

    // Expected bundle: relu, relu2, relu3};
    ASSERT_EQ(1, bundles.size()) << "Wrong number of bundles";
    ASSERT_EQ(3, bundles.front()->getNodes().size()) << "Wrong number of nodes";

    pNode expNodes[] = {relu, relu2, relu3};
    int bundleIdx = 0;
    for (const pNode& n : bundles.front()->getNodes())
    {
        ASSERT_EQ(expNodes[bundleIdx], n) << "Wrong node added to bundle";
        ++bundleIdx;
    }
}

TEST_F(SRAMManagementTest, bundle_parallel_tpc_nodes)
{
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");

    GaudiGraph     g;
    const unsigned dims        = 2;
    TSize          sizes[dims] = {16, 16};
    TensorVector   tensors;
    for (int i = 0; i < 15; i++)
    {
        tensors.push_back(pTensor(new Tensor(dims, sizes, syn_type_float)));
    }

    synGEMMParams params {};

    // Create the first MME node
    pNode gemm1 = NodeFactory::createNode({tensors[0], tensors[1]},
                                          {tensors[2]},
                                          &params,
                                          NodeFactory::gemmNodeTypeName,
                                          "gemm1");
    // Create a chain of 3 TPC nodes
    pNode relu1 = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[3]}, nullptr, "relu_fwd_f32", "relu1");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[4]}, nullptr, "relu_fwd_f32", "relu2");
    pNode relu3 = NodeFactory::createGenericTPCNode({tensors[4]}, {tensors[5]}, nullptr, "relu_fwd_f32", "relu3");
    // Create 3 MMEs consuming the output of the TPC chain
    pNode gemm2 = NodeFactory::createNode({tensors[5], tensors[6]},
                                          {tensors[7]},
                                          &params,
                                          NodeFactory::gemmNodeTypeName,
                                          "gemm2");
    pNode gemm3 = NodeFactory::createNode({tensors[5], tensors[8]},
                                          {tensors[9]},
                                          &params,
                                          NodeFactory::gemmNodeTypeName,
                                          "gemm3");
    pNode gemm4 = NodeFactory::createNode({tensors[5], tensors[10]},
                                          {tensors[11]},
                                          &params,
                                          NodeFactory::gemmNodeTypeName,
                                          "gemm4");
    // Add a second TPC producer for each one of the MME nodes
    pNode relu4 = NodeFactory::createGenericTPCNode({tensors[12]}, {tensors[6]}, nullptr, "relu_fwd_f32", "relu4");
    pNode relu5 = NodeFactory::createGenericTPCNode({tensors[13]}, {tensors[8]}, nullptr, "relu_fwd_f32", "relu5");
    pNode relu6 = NodeFactory::createGenericTPCNode({tensors[14]}, {tensors[10]}, nullptr, "relu_fwd_f32", "relu6");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(GraphEditor::addNode(g, relu3));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm2));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm3));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm4));
    ASSERT_TRUE(GraphEditor::addNode(g, relu4));
    ASSERT_TRUE(GraphEditor::addNode(g, relu5));
    ASSERT_TRUE(GraphEditor::addNode(g, relu6));

    BundleList bundles = Bundlizer(g).getTPCBundles();

    // Expected TPC bundle: relu1, relu2, relu3, relu4, relu5, relu6
    ASSERT_EQ(1, bundles.size());
    ASSERT_EQ(6, bundles.front()->getNodes().size());
}

TEST_F(SRAMManagementTest, dont_bundle_tpc_nodes_simple)
{
    // Given (MME) -> (TPC) -> (TPC) -> (MME)
    // Bundle the tpc's together
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph g;

    const unsigned dims        = 2;
    TSize          sizes[dims] = {16, 16};
    TensorVector tensors;
    for (int i = 0; i < 7; i++)
    {
        tensors.push_back(pTensor(new Tensor(dims, sizes, syn_type_float)));
    }

    synGEMMParams params{};

    pNode gemm1 = NodeFactory::createNode({tensors[0], tensors[1]}, {tensors[2]}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    pNode relu = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[3]}, nullptr, "relu_fwd_f32", "relu");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[4]}, nullptr, "relu_fwd_f32", "relu2");
    pNode gemm2 = NodeFactory::createNode({tensors[4], tensors[4]}, {tensors[6]}, &params, NodeFactory::gemmNodeTypeName, "gemm2");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm2));

    // When bundling the graph
    BundleList bundles = Bundlizer(g).getTPCBundles();

    // no bundles expected
    ASSERT_EQ(0, bundles.size()) << "Wrong number of bundles";
}

TEST_F(SRAMManagementTest, dont_bundle_tpc_nodes_two_chains)
{
    // dont bundle in this scenario:
    //           +-----+   +-----+
    //    +----> | TPC |+->| TPC |+-----+
    //    |      +-----+   +-----+      v
    // +--+--+                        +-----+
    // | MME |                        | MME |
    // +-----+                        +-----+
    //    +      +-----+   +-----+      ^
    //    +----->| TPC |+->| TPC |+-----+
    //           +-----+   +-----+
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph g;
    const unsigned dims        = 2;
    TSize          sizes[dims] = {16, 16};
    TensorVector tensors;
    for (int i = 0; i < 8; i++)
    {
        tensors.push_back(pTensor(new Tensor(dims, sizes, syn_type_float)));
    }

    synGEMMParams params{};

    pNode gemm1 = NodeFactory::createNode({tensors[0], tensors[1]}, {tensors[2]}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    pNode relu  = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[3]}, nullptr, "relu_fwd_f32", "relu");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[4]}, nullptr, "relu_fwd_f32", "relu2");
    pNode relu3 = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[5]}, nullptr, "relu_fwd_f32", "relu3");
    pNode relu4 = NodeFactory::createGenericTPCNode({tensors[5]}, {tensors[6]}, nullptr, "relu_fwd_f32", "relu4");
    pNode gemm2 = NodeFactory::createNode({tensors[4], tensors[6]}, {tensors[7]}, &params, NodeFactory::gemmNodeTypeName, "gemm2");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(GraphEditor::addNode(g, relu3));
    ASSERT_TRUE(GraphEditor::addNode(g, relu4));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm2));

    // When bundling the graph
    BundleList bundles = Bundlizer(g).getTPCBundles();

    // no bundles expected
    ASSERT_EQ(0, bundles.size()) << "Wrong number of bundles";
}

TEST_F(SRAMManagementTest, bundle_tpc_nodes_two_chains_one_valid)
{
    // bundle the lower chain in this scenario and take one paralel tpc producer:
    //           +-----+   +-----+
    //    +----> | TPC |+->| TPC |+------------+
    //    |      +-----+   +-----+             v
    // +--+--+                              +-----+
    // | MME |                              | MME |
    // +-----+                              +-----+
    //    +      +-----+   +-----+   +-----+   ^
    //    +----->| TPC |+->| TPC |+->| TPC |+--+
    //           +-----+   +-----+   +-----+
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph g;

    const unsigned dims        = 2;
    TSize          sizes[dims] = {16, 16};
    TensorVector tensors;
    for (int i = 0; i < 9; i++)
    {
        tensors.push_back(pTensor(new Tensor(dims, sizes, syn_type_float)));
    }

    synGEMMParams params{};

    pNode gemm1 = NodeFactory::createNode({tensors[0], tensors[1]}, {tensors[2]}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    pNode relu  = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[3]}, nullptr, "relu_fwd_f32", "relu");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[4]}, nullptr, "relu_fwd_f32", "relu2");
    pNode relu3 = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[5]}, nullptr, "relu_fwd_f32", "relu3");
    pNode relu4 = NodeFactory::createGenericTPCNode({tensors[5]}, {tensors[6]}, nullptr, "relu_fwd_f32", "relu4");
    pNode relu5 = NodeFactory::createGenericTPCNode({tensors[6]}, {tensors[7]}, nullptr, "relu_fwd_f32", "relu5");
    pNode gemm2 = NodeFactory::createNode({tensors[7], tensors[4]}, {tensors[8]}, &params, NodeFactory::gemmNodeTypeName, "gemm2");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(GraphEditor::addNode(g, relu3));
    ASSERT_TRUE(GraphEditor::addNode(g, relu4));
    ASSERT_TRUE(GraphEditor::addNode(g, relu5));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm2));

    // When bundling the graph
    BundleList bundles = Bundlizer(g).getTPCBundles();

    // Expected bundle: {relu3, relu4, relu5}
    ASSERT_EQ(1, bundles.size()) << "Wrong number of bundles";
    ASSERT_EQ(4, bundles.front()->getNodes().size()) << "Wrong number of bundles";

    pNode expNodes[] = {relu3, relu4, relu5, relu2};
    int bundleIdx = 0;
    for (const pNode& n : bundles.front()->getNodes())
    {
        ASSERT_EQ(expNodes[bundleIdx], n) << "Wrong node added to bundle";
        ++bundleIdx;
    }
}

TEST_F(SRAMManagementTest, bundle_tpc_nodes_two_valid_chains)
{
    //           +------+   +------+
    //    +----> | TPC1 |+->| TPC2 |+------------+
    //    |      +------+   +------+             v
    // +--+--+                              +------+     +-----+
    // | MME |                              | TPC6 |+--->| MME |
    // +-----+                              +------+     +-----+
    //    +      +------+   +------+   +------+   ^
    //    +----->| TPC3 |+->| TPC4 |+->| TPC5 |+--+
    //           +------+   +------+   +------+
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph g;

    const unsigned dims        = 2;
    TSize          sizes[dims] = {16, 16};
    TensorVector tensors;
    for (int i = 0; i < 11; i++)
    {
        tensors.push_back(pTensor(new Tensor(dims, sizes, syn_type_float)));
    }

    synGEMMParams params{};

    pNode gemm1 = NodeFactory::createNode({tensors[0], tensors[1]}, {tensors[2]}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    pNode relu  = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[3]}, nullptr, "relu_fwd_f32", "relu");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[4]}, nullptr, "relu_fwd_f32", "relu2");
    pNode relu3 = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[5]}, nullptr, "relu_fwd_f32", "relu3");
    pNode relu4 = NodeFactory::createGenericTPCNode({tensors[5]}, {tensors[6]}, nullptr, "relu_fwd_f32", "relu4");
    pNode relu5 = NodeFactory::createGenericTPCNode({tensors[6]}, {tensors[7]}, nullptr, "relu_fwd_f32", "relu5");
    pNode add = NodeFactory::createNode({tensors[4], tensors[7]}, {tensors[8]}, nullptr, "add_fwd_f32", "add");
    pNode gemm2 = NodeFactory::createNode({tensors[8], tensors[9]}, {tensors[10]}, &params, NodeFactory::gemmNodeTypeName, "gemm2");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(GraphEditor::addNode(g, relu3));
    ASSERT_TRUE(GraphEditor::addNode(g, relu4));
    ASSERT_TRUE(GraphEditor::addNode(g, relu5));
    ASSERT_TRUE(GraphEditor::addNode(g, add));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm2));

    // When bundling the graph
    BundleList bundles = Bundlizer(g).getTPCBundles();

    // Expected bundle: {relu3, relu4,  relu5, add}
    ASSERT_EQ(1, bundles.size()) << "Wrong number of bundles";
    ASSERT_EQ(4, bundles.front()->getNodes().size()) << "Wrong number of bundles";

    pNode expNodes[] = {relu3, relu4,  relu5, add};
    int bundleIdx = 0;
    for (const pNode& n : bundles.front()->getNodes())
    {
        ASSERT_EQ(expNodes[bundleIdx], n) << "Wrong node added to bundle";
        ++bundleIdx;
    }
}

TEST_F(SRAMManagementTest, dont_bundle_tpc_node_with_more_than_one_path2)
{
    // dont bundle tpc node with more than 1 graph connection
    //                                  +----------------------------------+
    //                                  |                                  v
    // +--+--+   +------+   +------+   +---+--+   +---+--+   +-----+    +------+
    // | MME |-->| TPC1 |+->| TPC2 |+->| TPC3 |-->| TPC4 |-->| MME |+-->| TPC5 |
    // +-----+   +------+   +------+   +------+   +------+   +-----+    +------+
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph g;

    const unsigned dims        = 2;
    TSize          sizes[dims] = {16, 16};
    TensorVector tensors;
    for (int i = 0; i < 11; i++)
    {
        tensors.push_back(pTensor(new Tensor(dims, sizes, syn_type_float)));
    }

    synGEMMParams params{};

    pNode gemm1 = NodeFactory::createNode({tensors[0], tensors[1]}, {tensors[2]}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    pNode relu  = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[3]}, nullptr, "relu_fwd_f32", "relu");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[4]}, nullptr, "relu_fwd_f32", "relu2");
    pNode relu3 = NodeFactory::createGenericTPCNode({tensors[4]}, {tensors[5]}, nullptr, "relu_fwd_f32", "relu3");
    pNode add = NodeFactory::createNode({tensors[5], tensors[9]}, {tensors[10]}, nullptr, "add_fwd_f32", "add");
    pNode relu4 = NodeFactory::createGenericTPCNode({tensors[5]}, {tensors[6]}, nullptr, "relu_fwd_f32", "relu4");
    pNode gemm2 = NodeFactory::createNode({tensors[6], tensors[7]}, {tensors[8]}, &params, NodeFactory::gemmNodeTypeName, "gemm2");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(GraphEditor::addNode(g, relu3));
    ASSERT_TRUE(GraphEditor::addNode(g, relu4));
    ASSERT_TRUE(GraphEditor::addNode(g, add));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm2));

    // When bundling the graph
    BundleList bundles = Bundlizer(g).getTPCBundles();

    // Expected bundle: {relu1, relu2, relu3, relu4}
    ASSERT_EQ(1, bundles.size()) << "Wrong number of bundles";
    ASSERT_EQ(4, bundles.front()->getNodes().size()) << "Wrong number of bundles";

    pNode expNodes[] = {relu, relu2, relu3, relu4};
    int bundleIdx = 0;
    for (const pNode& n : bundles.front()->getNodes())
    {
        ASSERT_EQ(expNodes[bundleIdx], n) << "Wrong node added to bundle";
        ++bundleIdx;
    }
}

TEST_F(SRAMManagementTest, bundle_tpc_nodes_with_backtracking)
{
    //   +-----+    +------+     +------+     +------+    +-----+
    //   | MME |+-->| TPC1 |+--->| TPC0 |+--->| TPC3 |+-->| MME |
    //   +-----+    +--+---+     +------+     +------+    +-----+
    //                 |
    //                 v
    //              +------+       +------+      +------+
    //              | TPC2 | +---> | TPC4 |+---> | TPC5 |
    //              +------+       +------+      +------+
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph g;

    const unsigned dims        = 2;
    TSize          sizes[dims] = {16, 16};
    TensorVector tensors;
    for (int i = 0; i < 11; i++)
    {
        tensors.push_back(pTensor(new Tensor(dims, sizes, syn_type_float)));
    }

    synGEMMParams params{};

    pNode gemm1 = NodeFactory::createNode({tensors[0], tensors[1]}, {tensors[2]}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    pNode relu1  = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[3]}, nullptr, "relu_fwd_f32", "relu1");
    pNode relu0 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[4]}, nullptr, "relu_fwd_f32", "relu0");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[8]}, nullptr, "relu_fwd_f32", "relu2");
    pNode relu3 = NodeFactory::createGenericTPCNode({tensors[4]}, {tensors[5]}, nullptr, "relu_fwd_f32", "relu3");
    pNode relu4 = NodeFactory::createGenericTPCNode({tensors[8]}, {tensors[9]}, nullptr, "relu_fwd_f32", "relu4");
    pNode relu5 = NodeFactory::createGenericTPCNode({tensors[9]}, {tensors[10]}, nullptr, "relu_fwd_f32", "relu5");
    pNode gemm2 = NodeFactory::createNode({tensors[5], tensors[6]}, {tensors[7]}, &params, NodeFactory::gemmNodeTypeName, "gemm2");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu0));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(GraphEditor::addNode(g, relu3));
    ASSERT_TRUE(GraphEditor::addNode(g, relu4));
    ASSERT_TRUE(GraphEditor::addNode(g, relu5));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm2));

    // When bundling the graph
    BundleList bundles = Bundlizer(g).getTPCBundles();

    // Expected bundle: {relu1, relu0, relu3}
    ASSERT_EQ(1, bundles.size()) << "Wrong number of bundles";
    ASSERT_EQ(3, bundles.front()->getNodes().size()) << "Wrong number of bundles";

    pNode expNodes[] = {relu1, relu0, relu3};
    int bundleIdx = 0;
    for (const pNode& n : bundles.front()->getNodes())
    {
        ASSERT_EQ(expNodes[bundleIdx], n) << "Wrong node added to bundle";
        ++bundleIdx;
    }
}

TEST_F(SRAMManagementTest, bundle_tpc_nodes_with_backtracking2)
{
    //   +-----+    +------+     +------+     +------+    +-----+
    //   | MME |+-->| TPC1 |+--->| TPC0 |+--->| TPC3 |+-->| MME |
    //   +-----+    +--+---+     +------+     +------+    +-----+
    //                 |
    //                 v
    //              +------+       +------+
    //              | TPC2 | +---> | TPC4 |
    //              +------+       +------+
    //                 |
    //                 v
    //              +------+
    //              | TPC5 |
    //              +------+
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph g;

    const unsigned dims        = 2;
    TSize          sizes[dims] = {16, 16};
    TensorVector tensors;
    for (int i = 0; i < 11; i++)
    {
        tensors.push_back(pTensor(new Tensor(dims, sizes, syn_type_float)));
    }

    synGEMMParams params{};

    pNode gemm1 = NodeFactory::createNode({tensors[0], tensors[1]}, {tensors[2]}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    pNode relu1  = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[3]}, nullptr, "relu_fwd_f32", "relu1");
    pNode relu0 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[4]}, nullptr, "relu_fwd_f32", "relu0");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[8]}, nullptr, "relu_fwd_f32", "relu2");
    pNode relu3 = NodeFactory::createGenericTPCNode({tensors[4]}, {tensors[5]}, nullptr, "relu_fwd_f32", "relu3");
    pNode relu4 = NodeFactory::createGenericTPCNode({tensors[8]}, {tensors[9]}, nullptr, "relu_fwd_f32", "relu4");
    pNode relu5 = NodeFactory::createGenericTPCNode({tensors[8]}, {tensors[10]}, nullptr, "relu_fwd_f32", "relu5");
    pNode gemm2 = NodeFactory::createNode({tensors[5], tensors[6]}, {tensors[7]}, &params, NodeFactory::gemmNodeTypeName, "gemm2");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu0));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(GraphEditor::addNode(g, relu3));
    ASSERT_TRUE(GraphEditor::addNode(g, relu4));
    ASSERT_TRUE(GraphEditor::addNode(g, relu5));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm2));

    // When bundling the graph
    BundleList bundles = Bundlizer(g).getTPCBundles();

    // Expected bundle: {relu1, relu0, relu3}
    ASSERT_EQ(1, bundles.size()) << "Wrong number of bundles";
    ASSERT_EQ(3, bundles.front()->getNodes().size()) << "Wrong number of bundles";

    pNode expNodes[] = {relu1, relu0, relu3};
    int bundleIdx = 0;
    for (const pNode& n : bundles.front()->getNodes())
    {
        ASSERT_EQ(expNodes[bundleIdx], n) << "Wrong node added to bundle";
        ++bundleIdx;
    }
}

TEST_F(SRAMManagementTest, dont_bundle_between_mme_and_logical_node)
{
    // +--+--+   +-----+   +-----+   +--+--+   +-----+
    // | MME |-->| TPC |+->| TPC |+->| TPC |-->| LOG |
    // +-----+   +-----+   +-----+   +-----+   +-----+
    GaudiGraph g;

    const unsigned dims        = 2;
    TSize          sizes[dims] = {16, 16};
    TensorVector tensors;
    for (int i = 0; i < 7; i++)
    {
        tensors.push_back(pTensor(new Tensor(dims, sizes, syn_type_float)));
    }

    synGEMMParams params{};

    pNode gemm1 = NodeFactory::createNode({tensors[0], tensors[1]}, {tensors[2]}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    pNode relu1  = NodeFactory::createGenericTPCNode({tensors[2]}, {tensors[3]}, nullptr, "relu_fwd_f32", "relu1");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[3]}, {tensors[4]}, nullptr, "relu_fwd_f32", "relu2");
    pNode relu3 = NodeFactory::createGenericTPCNode({tensors[4]}, {tensors[5]}, nullptr, "relu_fwd_f32", "relu3");
    pNode reshape = NodeFactory::createNode({tensors[5]}, {tensors[6]}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu1));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(GraphEditor::addNode(g, relu3));
    ASSERT_TRUE(GraphEditor::addNode(g, reshape));

    // When bundling the graph
    BundleList bundles = Bundlizer(g).getTPCBundles();

    ASSERT_EQ(0, bundles.size()) << "Wrong number of bundles";
}

TEST_F(SRAMManagementTest, dont_bundle_the_same_nodes_twice)
{
    //   +-----+
    //   | MME +--------+
    //   +-----+        |
    //               +--v---+    +------+    +------+    +-----+
    //               | TPC1 +--->| TPC2 +--->| TPC3 +--->| MME |
    //               +------+    +------+    +------+    +-----+
    //                  ^
    //   +-----+        |
    //   | MME +--------+
    //   +-----+
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph g;
    const unsigned dims        = 2;
    TSize          sizes[dims] = {16, 16};
    TensorVector tensors;
    for (int i = 0; i < 11; i++)
    {
        tensors.push_back(pTensor(new Tensor(dims, sizes, syn_type_float)));
    }

    synGEMMParams params{};

    pNode gemm1 = NodeFactory::createNode({tensors[0], tensors[1]}, {tensors[2]}, &params, NodeFactory::gemmNodeTypeName, "gemm1");
    pNode gemm2 = NodeFactory::createNode({tensors[3], tensors[4]}, {tensors[5]}, &params, NodeFactory::gemmNodeTypeName, "gemm2");
    pNode add = NodeFactory::createNode({tensors[2], tensors[5]}, {tensors[6]}, nullptr, "add_fwd_f32", "add");
    pNode relu2 = NodeFactory::createGenericTPCNode({tensors[6]}, {tensors[7]}, nullptr, "relu_fwd_f32", "relu2");
    pNode relu3 = NodeFactory::createGenericTPCNode({tensors[7]}, {tensors[8]}, nullptr, "relu_fwd_f32", "relu3");
    pNode gemm3 = NodeFactory::createNode({tensors[8], tensors[9]}, {tensors[10]}, &params, NodeFactory::gemmNodeTypeName, "gemm3");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm2));
    ASSERT_TRUE(GraphEditor::addNode(g, add));
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));
    ASSERT_TRUE(GraphEditor::addNode(g, relu3));
    ASSERT_TRUE(GraphEditor::addNode(g, gemm3));

    // When bundling the graph
    BundleList bundles = Bundlizer(g).getTPCBundles();

    ASSERT_EQ(1, bundles.size()) << "Wrong number of bundles";
    // Expected bundle: {add, relu2, relu3}
    ASSERT_EQ(3, bundles.front()->getNodes().size()) << "Wrong number of bundles";

    pNode expNodes[] = {add, relu2, relu3};
    int bundleIdx = 0;
    for (const pNode& n : bundles.front()->getNodes())
    {
        ASSERT_EQ(expNodes[bundleIdx], n) << "Wrong node added to bundle";
        ++bundleIdx;
    }
}

TEST_F(SRAMManagementTest, DISABLED_place_scalar_output_on_sram_for_reduction)
{
    //For reduction kernel which requires RMW memory place only required tensors in SRAM, do not exceed GCFG_MAX_RMW_TENSOR_BYTES
    // +--v---+
    // | TPC1 +
    // +------+

    setGlobalConfForTest(GCFG_MAX_RMW_TENSOR_BYTES, "1024");
    GaudiGraph g;

    auto gradIn           = createTensor({1024}, syn_type_float, true);
    auto inputFeatureMap1 = createTensor({1024}, syn_type_float, true);
    auto inputFeatureMap2 = createTensor({1}, syn_type_float, true);
    auto gradOut1         = createTensor({1024}, syn_type_float);
    auto gradOut2         = createTensor({1}, syn_type_float);

    pNode maxGrad = NodeFactory::createGenericTPCNode({gradIn, inputFeatureMap1, inputFeatureMap2},
                                                      {gradOut1, gradOut2},
                                                      nullptr,
                                                      "max_bwd_f32",
                                                      "max_grad");

    ASSERT_TRUE(GraphEditor::addNode(g, maxGrad));
    ASSERT_TRUE(g.compile());

    //output tensor 0 is in dram
    ASSERT_FALSE(maxGrad->getOutput(0)->inSram());
    //output tensor 1 (scalar) is in sram and marked for reduction
    ASSERT_TRUE(maxGrad->getOutput(1)->inSram());
    ASSERT_TRUE(maxGrad->getOutput(1)->isReductionEnabled(true));
}

TEST_F(SRAMManagementTest, snake_pattern_with_double_buffer)
{
    // This test verifies optimal allocation of the double buffer tensors.
    // In case a tensor can be allocated to more than one level - it should be allocated to the level that gets free
    // first.

    // Slicing Strategy - Left-to-Right , 2Wx2H, graph size optimized: false
    // Original Input [0] Tensor-0 : 128x512, Sliced : 128x256, Num of slices: 2, Buffers: 2, inSram: true,
    // alignedToCL:false
    // Original Input [1] Tensor-1 : 1024x128, Sliced : 128x128, Num of slices: 8, Buffers: 2, inSram:
    // true, alignedToCL:false
    // Original Output Tensor-2 : 1024x512, Sliced : 128x256, Num of slices: 16, Buffers: 1,
    // inSram: false, alignedToCL:false

    pTensor x = createTensor({128, 512}, syn_type_bf16);
    pTensor w = createTensor({1024, 128}, syn_type_bf16);
    pTensor y = createTensor({1024, 512}, syn_type_bf16);

    pBundle bundle = createSingleMMENodeBundle({x, w}, {y}, NodeFactory::convolutionNodeTypeName);

    pMmeSlicingStrategy strategy =
        MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle->getNodes().front());
    strategy->setOutputTraversalPattern(SlicedOperandTraversalPattern::LEFT_TO_RIGHT_2D);
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
    pSlicedOperand slicedX                 = strategy->getMmeSlicingData().bundleTensors[0];
    pSlicedOperand slicedW                 = strategy->getMmeSlicingData().bundleTensors[1];
    pSlicedOperand slicedY                 = strategy->getMmeSlicingData().masterOperand;
    slicedX->chunkDimensions[DIM_W]        = 256;
    slicedX->numOfBuffers                  = 2;
    slicedW->chunkDimensions[WEIGHT_DIM_K] = 128;
    slicedW->numOfBuffers                  = 2;
    slicedY->chunkDimensions[DIM_W]        = 256;
    slicedY->chunkDimensions[WEIGHT_DIM_K] = 128;

    SolutionGenerator generator(getGraph(), bundle, strategy);
    ASSERT_TRUE(generator.fillSolution());
    BundleSlicer::sliceBundle(*bundle, getGraph());

    setNonPersistentSectionInfo(getGraph());

    auto numOfWSlicesDimK = SlicedOperandUtils::nofSlices(slicedW, WEIGHT_DIM_K);

    uint32_t expectedLevelIdx = 0;
    uint32_t sliceIdx         = 1;
    for (const NodePtr& n : getGraph().getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_MEMCOPY)
        {
            const auto outTensorInfo = n->getOutput(0)->getTensorAnnotation().nonPersistentSectionInfo;
            ASSERT_TRUE(outTensorInfo.bufferingLevel.is_set());
            ASSERT_EQ(outTensorInfo.bufferingLevel.value(), 2);
            ASSERT_TRUE(outTensorInfo.sectionId.is_set());
            if (outTensorInfo.sectionId.value() == 1)  // belong to operand x double-buffer
            {
                // Expected allocation:
                // @MultiBuffer_1 Tensor "Tensor-1_slice_0_0_0_0_0__0__bundle_0" buffer index set to 0 (offset 0) and
                // sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_1_0_0_0_0__0__bundle_0" buffer index set to 1 (offset 32768)
                // and sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_2_0_0_0_0__0__bundle_0" buffer index set to 0 (offset 0) and
                // sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_3_0_0_0_0__0__bundle_0" buffer index set to 1 (offset 32768)
                // and sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_4_0_0_0_0__0__bundle_0" buffer index set to 0 (offset 0) and
                // sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_5_0_0_0_0__0__bundle_0" buffer index set to 1 (offset 32768)
                // and sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_6_0_0_0_0__0__bundle_0" buffer index set to 0 (offset 0) and
                // sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_7_0_0_0_0__0__bundle_0" buffer index set to 1 (offset 32768)
                // and sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_5_0_0_0_0__1__bundle_0" buffer index set to 1 (offset 32768)
                // and sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_4_0_0_0_0__1__bundle_0" buffer index set to 0 (offset 0) and
                // sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_3_0_0_0_0__1__bundle_0" buffer index set to 1 (offset 32768)
                // and sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_2_0_0_0_0__1__bundle_0" buffer index set to 0 (offset 0) and
                // sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_1_0_0_0_0__1__bundle_0" buffer index set to 1 (offset 32768)
                // and sizeToAllocate to 65536
                // @MultiBuffer_1 Tensor "Tensor-1_slice_0_0_0_0_0__1__bundle_0" buffer index set to 0 (offset 0) and
                // sizeToAllocate to 65536
                if (expectedLevelIdx == 0)
                {
                    ASSERT_TRUE(outTensorInfo.offsetFromBase.value() == 0);
                }
                else
                {
                    ASSERT_TRUE(outTensorInfo.offsetFromBase.value() > 0);
                }
                if (sliceIdx != numOfWSlicesDimK)
                {
                    expectedLevelIdx = (expectedLevelIdx + 1) % 2;
                }
                sliceIdx++;
            }
        }
    }
}

TEST_F(SRAMManagementTest, sram_slice_using_cost_model)
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_COST_MODEL_ENABLED, "true");

    // Create a graph with 2 MMEs sharing an input, with a TPC producer
    // to each MME input and a TPC consumer to each MME output.

    std::vector<TSize> sizes = {1024, 1024};

    pTensor shared         = createTensor(sizes, syn_type_bf16);
    pTensor gemm0Wgh       = createTensor(sizes, syn_type_bf16);
    pTensor gemm0Out       = createTensor(sizes, syn_type_bf16);
    pTensor gemm1Wgh       = createTensor(sizes, syn_type_bf16);
    pTensor gemm1Out       = createTensor(sizes, syn_type_bf16);
    pTensor reluInShared   = createTensor(sizes, syn_type_bf16);
    pTensor reluInGemm0Wgh = createTensor(sizes, syn_type_bf16);
    pTensor reluInGemm1Wgh = createTensor(sizes, syn_type_bf16);
    pTensor reluOutGemm0   = createTensor(sizes, syn_type_bf16);
    pTensor reluOutGemm1   = createTensor(sizes, syn_type_bf16);

    synGEMMParams gemm0Params {};
    synGEMMParams gemm1Params {};
    pNode         gemm0 =
        NodeFactory::createNode({shared, gemm0Wgh}, {gemm0Out}, &gemm0Params, NodeFactory::gemmNodeTypeName, "gemm0");
    GraphEditor::addNode(getGraph(), gemm0);
    pNode gemm1 =
        NodeFactory::createNode({shared, gemm1Wgh}, {gemm1Out}, &gemm1Params, NodeFactory::gemmNodeTypeName, "gemm1");
    GraphEditor::addNode(getGraph(), gemm1);

    unsigned i = 0;
    for (std::pair<pTensor, pTensor> reluOperands :
         std::vector<std::pair<pTensor, pTensor>> {{reluInShared, shared},
                                                   {reluInGemm0Wgh, gemm0Wgh},
                                                   {reluInGemm1Wgh, gemm1Wgh},
                                                   {gemm0Out, reluOutGemm0},
                                                   {gemm1Out, reluOutGemm1}})
    {
        std::string name {"relu"};
        pNode       relu = NodeFactory::createNode({reluOperands.first},
                                             {reluOperands.second},
                                             nullptr,
                                             "relu_fwd_bf16",
                                             name + std::to_string(i++));
        GraphEditor::addNode(getGraph(), relu);
    }

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
}

TEST_F(SRAMManagementTest, sram_slice_using_cost_model_with_reshapes)
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_COST_MODEL_ENABLED, "true");

    pTensor xProducerIn  = createTensor({512, 32, 64, 1}, syn_type_bf16, true);
    pTensor xProducerOut = createTensor({512, 32, 64, 1}, syn_type_bf16, false);
    pTensor x            = createTensor({512, 2048}, syn_type_bf16, false);
    pTensor wProducerIn  = createTensor({128, 2, 256}, syn_type_bf16, true);
    pTensor wProducerOut = createTensor({128, 2, 256}, syn_type_bf16, false);
    pTensor w            = createTensor({128, 512}, syn_type_bf16, false);
    pTensor y            = createTensor({128, 2048}, syn_type_bf16, false);
    pTensor consumerIn   = createTensor({128, 64, 32, 1}, syn_type_bf16, false);
    pTensor consumerOut  = createTensor({128, 64, 32, 1}, syn_type_bf16, true);

    pNode xProducer =
        NodeFactory::createGenericTPCNode({xProducerIn}, {xProducerOut}, nullptr, "relu_fwd_bf16", "xProducer");
    GraphEditor::addNode(getGraph(), xProducer);
    pNode xReshape =
        NodeFactory::createNode({xProducerOut}, {x}, nullptr, NodeFactory::reshapeNodeTypeName, "xReshape");
    GraphEditor::addNode(getGraph(), xReshape);
    pNode wProducer =
        NodeFactory::createGenericTPCNode({wProducerIn}, {wProducerOut}, nullptr, "relu_fwd_bf16", "wProducer");
    GraphEditor::addNode(getGraph(), wProducer);
    pNode wReshape =
        NodeFactory::createNode({wProducerOut}, {w}, nullptr, NodeFactory::reshapeNodeTypeName, "wReshape");
    GraphEditor::addNode(getGraph(), wReshape);
    synGEMMParams params {};
    pNode         gemm = NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::gemmNodeTypeName, "gemm");
    GraphEditor::addNode(getGraph(), gemm);
    pNode consumerReshape =
        NodeFactory::createNode({y}, {consumerIn}, nullptr, NodeFactory::reshapeNodeTypeName, "consumerReshape");
    GraphEditor::addNode(getGraph(), consumerReshape);
    pNode consumer =
        NodeFactory::createGenericTPCNode({consumerIn}, {consumerOut}, nullptr, "relu_fwd_bf16", "consumer");
    GraphEditor::addNode(getGraph(), consumer);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
}

TEST_F(SRAMManagementTest, non_shared_operand_cl_align)
{
    std::vector<TSize> sizes = {129, 129};

    pTensor shared = createTensor(sizes, syn_type_bf16);
    pTensor in1    = createTensor(sizes, syn_type_bf16);
    pTensor in2    = createTensor(sizes, syn_type_bf16);
    pTensor out    = createTensor(sizes, syn_type_bf16);
    pTensor out2   = createTensor(sizes, syn_type_bf16);

    synGEMMParams gemm0Params {};
    synGEMMParams gemm1Params {};
    pNode gemm0 = NodeFactory::createNode({in1, shared}, {out}, &gemm0Params, NodeFactory::gemmNodeTypeName, "gemm0");
    GraphEditor::addNode(getGraph(), gemm0);
    pNode gemm1 = NodeFactory::createNode({in2, shared}, {out2}, &gemm1Params, NodeFactory::gemmNodeTypeName, "gemm1");
    GraphEditor::addNode(getGraph(), gemm1);

    // Create bundles and initial strategies
    AllBrains          allBrains {getGraph()};
    SRAMSlicingManager sramManager {getGraph()};
    sramManager.generateInitialBundles();
    sramManager.generateInitialStrategies();
    auto bundlesSolvingData = sramManager.getBundlesSolvingData();

    BundleExpander bundleExpander {getGraph(), allBrains, sramManager.getBundlizer(), bundlesSolvingData};
    const pBundle& bundle = bundlesSolvingData.begin()->first;

    const auto& expandedStrategies = bundleExpander.generateExpandedStrategies(bundle);

    for (const auto& s : expandedStrategies)
    {
        pMmeSlicingStrategy mmeStrategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
        if (mmeStrategy->getMmeSlicingData().hasRole(BundleExpansion::SharedInputConsumer))
        {
            auto nonSharedOperand = mmeStrategy->getMmeSlicingData()
                                        .getRoleCandidates()[BundleExpansion::SharedInputConsumer]
                                        ->slaveOperands.getInput();
            ASSERT_TRUE(nonSharedOperand->alignWithCacheLine);
        }
    }
}

TEST_F(SRAMManagementTest, multiple_producers_for_tensor_in_same_section)
{
    // Two relu TPC nodes produce output in the same section. It is then consumed by GEMM.
    // Relu must not be stitched to gemm in this case.

    GaudiGraph g;

    auto                tensorFromUserA = createTensor({1024, 1024}, syn_type_float, true);
    synMemoryDescriptor descriptor;
    descriptor.m_isPersistent = true;
    tensorFromUserA->setMemoryDescriptor(descriptor);
    tensorFromUserA->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);

    auto tensorFromUserB = createTensor({1024, 1024}, syn_type_float, true);
    tensorFromUserB->setMemoryDescriptor(descriptor);
    tensorFromUserA->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    auto tensorFromUserARelu = createTensor({1024, 1024}, syn_type_float, true);
    tensorFromUserARelu->setMemoryDescriptor(descriptor);
    tensorFromUserARelu->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);

    auto tensorFromUserBRelu = createTensor({1024, 1024}, syn_type_float, true);
    tensorFromUserBRelu->setMemoryDescriptor(descriptor);
    tensorFromUserBRelu->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);

    pNode reluA = NodeFactory::createGenericTPCNode({tensorFromUserA},
                                                    {tensorFromUserARelu},
                                                    nullptr,
                                                    "relu_fwd_f32",
                                                    "producerA");

    pNode reluB = NodeFactory::createGenericTPCNode({tensorFromUserB},
                                                    {tensorFromUserBRelu},
                                                    nullptr,
                                                    "relu_fwd_f32",
                                                    "producerB");
    ASSERT_TRUE(GraphEditor::addNode(g, reluA));
    ASSERT_TRUE(GraphEditor::addNode(g, reluB));
    g.addControlDependency({reluB}, {reluA});

    auto operandB = createTensor({1024, 1024}, syn_type_float, true);
    operandB->setMemoryDescriptor(descriptor);
    operandB->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);

    auto gemmOut = createTensor({1024, 1024}, syn_type_float, true);
    gemmOut->setMemoryDescriptor(descriptor);
    gemmOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);

    synGEMMParams params;

    pNode gemm = NodeFactory::createNode({tensorFromUserARelu, operandB},
                                         {gemmOut},
                                         &params,
                                         NodeFactory::gemmNodeTypeName,
                                         "gemm");

    ASSERT_TRUE(GraphEditor::addNode(g, gemm));
    ASSERT_TRUE(g.compile());
    ASSERT_FALSE(reluB->getNodeAnnotation().bundleInfo.is_set());
}

TEST_F(SRAMManagementTest, validate_no_path_between_producers)
{
    std::vector<TSize   > sizes        = {128, 128};
    pTensor               relu1In      = createTensor(sizes, syn_type_float, true);
    pTensor               relu1Out     = createTensor(sizes, syn_type_float, false);
    pTensor               relu2Out     = createTensor(sizes, syn_type_float, false);
    pTensor               gemmSharedIn = createTensor(sizes, syn_type_float, true);
    pTensor               gemm1Out     = createTensor(sizes, syn_type_float, true);
    pTensor               gemm2Out     = createTensor(sizes, syn_type_float, true);

    pNode relu1 = NodeFactory::createGenericTPCNode({relu1In}, {relu1Out}, nullptr, "relu_fwd_f32", "relu1");
    GraphEditor::addNode(getGraph(), relu1);

    pNode relu2 = NodeFactory::createGenericTPCNode({relu1Out}, {relu2Out}, nullptr, "relu_fwd_f32", "relu2");
    GraphEditor::addNode(getGraph(), relu2);

    synGEMMParams gemm1Params {};
    pNode         gemm1 = NodeFactory::createNode({relu1Out, gemmSharedIn},
                                          {gemm1Out},
                                          &gemm1Params,
                                          NodeFactory::gemmNodeTypeName,
                                          "gemm1");
    GraphEditor::addNode(getGraph(), gemm1);

    synGEMMParams gemm2Params {};
    pNode         gemm2 = NodeFactory::createNode({relu2Out, gemmSharedIn},
                                          {gemm2Out},
                                          &gemm2Params,
                                          NodeFactory::gemmNodeTypeName,
                                          "gemm2");
    GraphEditor::addNode(getGraph(), gemm2);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));

    // Make sure both producers are not in the same bundle (there is a path relu1->relu2)
    if (relu1->getNodeAnnotation().bundleInfo.is_set() && relu2->getNodeAnnotation().bundleInfo.is_set())
    {
        ASSERT_NE(relu1->getNodeAnnotation().bundleInfo->bundleIndex,
                  relu2->getNodeAnnotation().bundleInfo->bundleIndex);
    }
}

TEST_F(SRAMManagementTest, flattenable_batch_gemm_expansion)
{
    std::vector<TSize> in1Sizes = {1024, 512, 8};
    std::vector<TSize> in2Sizes = {1024, 1024};
    std::vector<TSize> outSizes = {1024, 512, 8};

    pTensor shared     = createTensor(in1Sizes, syn_type_bf16);
    pTensor nonShared1 = createTensor(in2Sizes, syn_type_bf16);
    pTensor nonShared2 = createTensor(in2Sizes, syn_type_bf16);
    pTensor out1       = createTensor(outSizes, syn_type_bf16);
    pTensor out2       = createTensor(outSizes, syn_type_bf16);

    synGEMMParams bgemm1Params {};
    synGEMMParams bgemm2Params {};
    pNode         bgemm1 = NodeFactory::createNode({shared, nonShared1},
                                           {out1},
                                           &bgemm1Params,
                                           NodeFactory::batchGemmNodeTypeName,
                                           "bgemm1");
    GraphEditor::addNode(getGraph(), bgemm1);
    pNode bgemm2 = NodeFactory::createNode({shared, nonShared2},
                                           {out2},
                                           &bgemm2Params,
                                           NodeFactory::batchGemmNodeTypeName,
                                           "bgemm2");
    GraphEditor::addNode(getGraph(), bgemm2);

    // Create bundles and initial strategies
    std::unordered_map<pBundle, BundleSolvingData> solvingDataPerBundle;
    Bundlizer                                      bundlizer(getGraph());
    const BundleList&                              bundles = bundlizer.getMMEBundles();
    AllBrains                                      brains(getGraph());

    for (const pBundle& bundle : bundles)
    {
        // Add flattened strategy
        pMmeSlicingStrategy flattenedStrategy =
            MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle->getNodes().front());
        flattenedStrategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
        auto& flattenedSlicingData = flattenedStrategy->getMmeSlicingData();
        // Flatten operands
        ASSERT_TRUE(MMENodeFlattener::canFlattenMMENode(bundle->getNodes().front()));
        flattenedSlicingData.bundleTensors[0]->finalShape =
            MMENodeFlattener::getFlattenShape(flattenedSlicingData.bundleTensors[0]->originalTensor);
        flattenedSlicingData.bundleTensors[1]->finalShape =
            MMENodeFlattener::getFlattenShape(flattenedSlicingData.bundleTensors[1]->originalTensor);
        flattenedSlicingData.masterOperand->finalShape =
            MMENodeFlattener::getFlattenShape(flattenedSlicingData.masterOperand->originalTensor);
        flattenedSlicingData.bundleTensors[0]->chunkDimensions = flattenedSlicingData.bundleTensors[0]->finalShape;
        flattenedSlicingData.bundleTensors[1]->chunkDimensions = flattenedSlicingData.bundleTensors[1]->finalShape;
        flattenedSlicingData.masterOperand->chunkDimensions    = flattenedSlicingData.masterOperand->finalShape;
        // Slice operands
        flattenedSlicingData.bundleTensors[0]->chunkDimensions[DIM_W] =
            flattenedSlicingData.bundleTensors[0]->finalShape[DIM_W] / 8;
        flattenedSlicingData.bundleTensors[1]->chunkDimensions[WEIGHT_DIM_K] =
            flattenedSlicingData.bundleTensors[1]->finalShape[WEIGHT_DIM_K] / 2;
        flattenedSlicingData.masterOperand->chunkDimensions[DIM_W] =
            flattenedSlicingData.masterOperand->finalShape[DIM_W] / 8;
        flattenedSlicingData.masterOperand->chunkDimensions[WEIGHT_DIM_K] =
            flattenedSlicingData.masterOperand->finalShape[WEIGHT_DIM_K] / 2;
        solvingDataPerBundle[bundle].strategies.push_back(flattenedStrategy);

        // Add non flattened strategy
        pMmeSlicingStrategy nonFlattenedStrategy =
            MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), bundle->getNodes().front());
        nonFlattenedStrategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
        auto& nonFlattenedSlicingData = nonFlattenedStrategy->getMmeSlicingData();
        // Slice operands
        nonFlattenedSlicingData.bundleTensors[0]->chunkDimensions[DIM_GEMM_BATCH] =
            nonFlattenedSlicingData.bundleTensors[0]->finalShape[DIM_GEMM_BATCH] / 8;
        nonFlattenedSlicingData.bundleTensors[1]->chunkDimensions[WEIGHT_DIM_K] =
            nonFlattenedSlicingData.bundleTensors[1]->finalShape[WEIGHT_DIM_K] / 2;
        nonFlattenedSlicingData.masterOperand->chunkDimensions[DIM_GEMM_BATCH] =
            nonFlattenedSlicingData.masterOperand->finalShape[DIM_GEMM_BATCH] / 8;
        nonFlattenedSlicingData.masterOperand->chunkDimensions[WEIGHT_DIM_K] =
            nonFlattenedSlicingData.masterOperand->finalShape[WEIGHT_DIM_K] / 2;
        solvingDataPerBundle[bundle].strategies.push_back(nonFlattenedStrategy);
    }

    const pBundle& bundleMaster = solvingDataPerBundle.begin()->first;
    BundleExpander bundleExpander {getGraph(), brains, bundlizer, solvingDataPerBundle};

    // Expand the first bundle
    const auto& expandedStrategies = bundleExpander.generateExpandedStrategies(bundleMaster);

    for (const auto& s : expandedStrategies)
    {
        pMmeSlicingStrategy mmeStrategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
        if (mmeStrategy->getMmeSlicingData().hasRole(BundleExpansion::SharedInputConsumer))
        {
            // If the strategy has a slave MME expansion - make sure the slave is flattened only if needed (master is
            // flattened)
            auto masterOutput = mmeStrategy->getMmeSlicingData().masterOperand;
            auto slaveOutput  = mmeStrategy->getMmeSlicingData()
                                   .getRoleCandidates()[BundleExpansion::SharedInputConsumer]
                                   ->slaveOperands.getOutput();
            ASSERT_TRUE(SlicedOperandUtils::shouldOperandBeFlattened(masterOutput) ==
                        SlicedOperandUtils::shouldOperandBeFlattened(slaveOutput));
        }
    }
}

TEST_F(SRAMManagementTest, tpc_scalar_pipe_no_output_operand)
{
    auto& graph(getGraph());

    std::vector<TSize> sizes     = {1, 1};
    TensorPtr          rndSeedIn = createTensor(sizes, syn_type_uint32, true);
    NodePtr randomSeed = NodeFactory::createGenericTPCNode({rndSeedIn}, {}, nullptr, "random_seed_u32", "rs_u32");
    GraphEditor::addNode(graph, randomSeed);

    ASSERT_TRUE(graph.compile());
    ASSERT_FALSE(randomSeed->getNodeAnnotation().bundleInfo.is_set());
}

TEST_F(SRAMManagementTest, remove_opposite_cast_after_sram_reduction)
{
    const std::vector<TSize> dySizes = {512, 80, 40, 16};
    const std::vector<TSize> xSizes  = {512, 80, 40, 16};
    const std::vector<TSize> dwSizes = {512, 512, 3, 3};

    pTensor              x  = createTensor(xSizes, syn_type_bf16, true);
    pTensor              dy = createTensor(dySizes, syn_type_bf16, true);
    pTensor              dw = createTensor(dwSizes, syn_type_bf16, false);
    synConvolutionParams params(3, 3, 1, 1, 1, 1, 1, 1, 1, 1);

    // The dedw will be sliced on common-dim and will use SRAM reduction in FP32, the slicer will add cast back to bf16.
    pNode dedwNode = NodeFactory::createNode({dy, x}, {dw}, &params, NodeFactory::deDwNodeTypeName, "dedw");

    // This cast is expected to be optimized out - it is an opposite cast to the one created by the slicer.
    pTensor castOut  = createTensor(dwSizes, syn_type_single, true);
    pNode   castNode = NodeFactory::createNode({dw}, {castOut}, nullptr, "cast_bf16_to_f32", "cast");

    ASSERT_TRUE(GraphEditor::addNode(getGraph(), dedwNode));
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), castNode));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
    ASSERT_TRUE(selectMemcpyEngine(getGraph()));
    ASSERT_TRUE(removeContiguousCastNodes(getGraph()));

    bool               reductionFound = false;
    bool               castNodeFound  = false;
    bool               memcopyFound   = false;
    Settable<unsigned> expectedBundleIdx;

    for (const NodePtr& node : getGraph().getExeSortedNodes())
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            reductionFound = true;
            ASSERT_TRUE(node->getNodeAnnotation().bundleInfo.is_set());
            expectedBundleIdx.set(node->getNodeAnnotation().bundleInfo->bundleIndex);
        }
        else if (node->isCast())
        {
            // Both casts (slicer + user) should be eliminated
            castNodeFound = true;
        }
        else if (node->getNodeType() == Node::TYPE_MEMCOPY)
        {
            // Both casts are expected to be replaced with memcopy sram->hbm.
            memcopyFound = true;
            ASSERT_EQ(node->getNumInputs(), 1);
            ASSERT_EQ(node->getNumOutputs(), 1);
            ASSERT_EQ(node->getInput(0)->getElementType(), syn_type_float);
            ASSERT_EQ(node->getOutput(0)->getElementType(), syn_type_float);
            ASSERT_TRUE(node->getInput(0)->inSram());
            ASSERT_FALSE(node->getOutput(0)->inSram());
            // The memcopy node should get the same bundle info as the reduction node to make sure
            // the output is evicted at the end of the bundle.
            ASSERT_TRUE(node->getNodeAnnotation().bundleInfo.is_set());
            ASSERT_TRUE(expectedBundleIdx.is_set());
            ASSERT_EQ(node->getNodeAnnotation().bundleInfo->bundleIndex, expectedBundleIdx.value());
        }
    }

    ASSERT_TRUE(reductionFound);
    ASSERT_TRUE(memcopyFound);
    ASSERT_FALSE(castNodeFound);
}
