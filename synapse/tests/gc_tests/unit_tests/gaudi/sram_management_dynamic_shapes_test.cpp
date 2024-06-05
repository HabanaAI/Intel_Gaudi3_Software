#include "passes/sram_management/sram_management.h"

#include <graph_compiler/passes/sram_management/sliced_operand_traversal.h>
#include <graph_compiler/passes/sram_management/solution_generator.h>
#include <graph_compiler/passes/sram_management/reshape_aligner.h>
#include <graph_compiler/passes/sram_management/batch_slicing_solver.h>
#include "sram_management_fe_test.h"
#include "perf_lib_layer_params.h"
#include "graph_compiler/passes/sram_management/slicing_utils.h"
#include "graph_compiler/passes/sram_management/spatial_slicing_solver.h"
#include "graph_compiler/passes/sram_management/common_dim_slicing_solver.h"
#include "graph_compiler/passes/sram_management/mme_dim_controller.h"
#include <unordered_set>
#include <math.h>
#include "graph_compiler/passes/sram_management/bundle_expander.h"
#include <habana_global_conf.h>
#include "graph_compiler/passes/sram_management/flatten_mme.h"
#include <platform/gaudi/graph_compiler/passes.h>
#include <graph_compiler/passes/sram_management/bundle_slicer.h>

using namespace gaudi;

class SRAMManagementTestDsd : public SRAMManagementTest
{
public:
    SRAMManagementTestDsd()
    {
        m_convParams.kH   = 3;
        m_convParams.kW   = 3;
    }

    void SetUp() override
    {
        SRAMManagementTest::SetUp();
        createNodesAndTensors();
    }

    void TearDown() override
    {
        SRAMManagementTest::TearDown();
    }

protected:
    const TSize
        BATCH_MAX = 2,
        BATCH_MIN = 1,
        H = 128,
        W = 128,
        K = 16,
        C = 16;

    synConvolutionParams m_convParams;

    TensorPtr m_dyTensor;
    TensorPtr m_wghTensor;
    TensorPtr m_dwTensor;
    TensorPtr m_xTensor;
    TensorPtr m_dxTensor;
    TensorPtr m_shapeTensor;

    NodePtr m_dedxNode;
    pBundle m_bundleDedx;
    NodePtr m_dedwNode;
    pBundle m_bundleDedw;
    SlicingStrategyList m_slicingStrategyList;

    virtual void createNodesAndTensors() = 0;

    TensorPtr createTensor(unsigned dims, TSize* size, synDataType dataType, synTensorType tensorType = DATA_TENSOR,
                           bool isPersistent = false, std::string name = "")
    {
        return createTensor(dims, size, nullptr, dataType, tensorType, isPersistent, name);
    }

    TensorPtr createTensor(unsigned dims, TSize* maxSize, TSize* minSize, synDataType dataType,
                                  synTensorType tensorType = DATA_TENSOR, bool isPersistent = false,
                                  std::string name = "")
    {
        synMemoryDescriptor persistentDesc(isPersistent);
        auto ret = std::make_shared<Tensor>(dims, maxSize, dataType, nullptr, nullptr, false, false,
                                            INVALID_BATCH_POS, minSize, tensorType);
        ret->setName(name);
        ret->setMemoryDescriptor(persistentDesc);
        if (isPersistent)
        {
            ret->setMemorySectionID(m_memorySectionId++);
        }
        ret->setDramOffset(0x1000000);
        ret->map();
        return ret;
    }

    void createDedwTensors()
    {
        unsigned dims = 4;
        TSize xSizesMax[]   = {C, W, H, BATCH_MAX};
        TSize xSizesMin[]   = {C, W, H, BATCH_MIN};
        TSize wghSizes[] = {K, C, m_convParams.kW, m_convParams.kH};

        m_dwTensor = createTensor(dims, wghSizes, syn_type_bf16, DATA_TENSOR, true, "dw");
        m_xTensor = createTensor(dims, xSizesMax, xSizesMin, syn_type_bf16, DATA_TENSOR, true, "x");
    }

    void addDedwNode()
    {
        m_dedwNode = NodeFactory::createNode({m_dyTensor, m_xTensor}, {m_dwTensor},
                                             &m_convParams, NodeFactory::deDwNodeTypeName, "dedw");
        GraphEditor::addNode(getGraph(), m_dedwNode);
    }

    void createDedxTensors()
    {
        unsigned dims = 4;

        size_t yH = convOutputDimSize(H, m_convParams.kH, m_convParams.dH, m_convParams.padT + m_convParams.padB, m_convParams.dilH);
        size_t yW = convOutputDimSize(W, m_convParams.kW, m_convParams.dW, m_convParams.padL + m_convParams.padR, m_convParams.dilW);

        TSize dySizesMax[]  = {K, yW, yH, BATCH_MAX};
        TSize dySizesMin[]  = {K, yW, yH, BATCH_MIN};
        TSize xSizesMax[]   = {C, W, H, BATCH_MAX};
        TSize xSizesMin[]   = {C, W, H, BATCH_MIN};
        TSize wghSizes[]    = {K, C, m_convParams.kW, m_convParams.kH};

        m_wghTensor = createTensor(dims, wghSizes, syn_type_bf16, DATA_TENSOR, true, "w");
        m_dyTensor = createTensor(dims, dySizesMax, dySizesMin, syn_type_bf16, DATA_TENSOR, true, "dy");
        m_dxTensor = createTensor(dims, xSizesMax, xSizesMin, syn_type_bf16, DATA_TENSOR, true, "dx");
        m_shapeTensor = createTensor(dims, xSizesMax, xSizesMin, syn_type_bf16, SHAPE_TENSOR, false, "dx_shape");
    }

    void addDedxNode()
    {
        m_dedxNode = NodeFactory::createNode({m_dyTensor, m_wghTensor, m_shapeTensor},
                                             {m_dxTensor}, &m_convParams, NodeFactory::deDxNodeTypeName, "dedx");
        GraphEditor::addNode(getGraph(), m_dedxNode);
    }
};

class SRAMManagementTestDsdDedxAndDeDw : public SRAMManagementTestDsd
{
protected:
    virtual void createNodesAndTensors() override
    {
        createDedxTensors();
        addDedxNode();
        createDedwTensors();
        addDedwNode();
    }
};

class SRAMManagementTestDsdDedx : public SRAMManagementTestDsd
{
public:
    virtual void createNodesAndTensors() override
    {
        createDedxTensors();
        addDedxNode();
    }
};

class SRAMManagementTestDsdMasterDedx : public SRAMManagementTestDsd
{
public:
    virtual void createNodesAndTensors() override
    {
        createDedxTensors();
        addDedxNode();
        createStrategies();
    }

protected:
    void createStrategies()
    {
        Bundlizer bundlizer(getGraph());
        auto bundles = bundlizer.getMMEBundles();
        m_bundleDedx = bundles.back();
        MMESlicingBrain brain(getGraph());
        m_slicingStrategyList = brain.getSolutionStrategies(m_bundleDedx);
    }
};

class SRAMManagementTestDsdSlaveDedx : public SRAMManagementTestDsd
{
public:
    virtual void createNodesAndTensors() override
    {
        createDedwTensors();
        addDedwNode();
        createDedxTensors();
        addDedxNode();
        createStrategies();
    }

protected:
    void createStrategies()
    {
        Bundlizer bundlizer(getGraph());
        auto bundles = bundlizer.getMMEBundles();
        m_bundleDedx = bundles.back();
        MMESlicingBrain brain(getGraph());
        m_slicingStrategyList = brain.getSolutionStrategies(m_bundleDedx);
    }
};

// Strategy for dynamic dedx should include a sliced operand for the shape tensor.
TEST_F(SRAMManagementTestDsdMasterDedx, strategy_for_dynamic_dedx_should_include_a_sliced_operand_for_the_shape_tensor)
{
    SlicingStrategyPtr strategy = m_slicingStrategyList.front();
    ASSERT_NE(nullptr, strategy);

    auto& slicingData = strategy->getSlicingData();
    bool containsSlicedOperandForTheShapeTensor = false;
    const auto& slicedOperands = slicingData.getSlicedOperands();
    bool containsShapeTensor = false;

    for(const auto& so : slicedOperands)
    {
        if(so->originalTensor->isShapeTensor())
        {
            containsShapeTensor = true;
            containsSlicedOperandForTheShapeTensor = true;
        }
    }
    ASSERT_TRUE(containsShapeTensor);
    ASSERT_TRUE(containsSlicedOperandForTheShapeTensor);
}

// Strategy for dynamic dedx is generated with 1:1 mapping from output to the shape tensor input sliced operand
TEST_F(SRAMManagementTestDsdMasterDedx, strategy_has_1_to_1_mapping_from_output_to_the_shape_tensor_input_sliced_operand)
{
    SlicingStrategyPtr strategy = m_slicingStrategyList.front();
    ASSERT_NE(nullptr, strategy);

    auto& slicingData = strategy->getSlicingData();
    auto outputSlices = slicingData.getOutputSlices();
    bool containsShapeTensor = false;

    for(const auto& slice : outputSlices)
    {
        const auto& sliceInputs = slicingData.getInputsForSlice(slice);
        for(const auto& ref : sliceInputs)
        {
            if(ref->operand->originalTensor->isShapeTensor())
            {
                containsShapeTensor = true;
                EXPECT_EQ(ref->coordinates, slice.first->coordinates);
            }
        }
    }
    ASSERT_TRUE(containsShapeTensor);
}

// All strategies of a sliced dynamic dedx have the same slicing to the shape tensor sliced operand and the output
TEST_F(SRAMManagementTestDsdMasterDedx, all_strategies_have_same_slicing_to_the_shape_tensor_sliced_operand_and_the_output)
{
    for(SlicingStrategyPtr strategy : m_slicingStrategyList)
    {
        auto& slicingData = strategy->getSlicingData();
        auto outputSlices = slicingData.getOutputSlices();
        bool containsShapeTensor = false;

        for(const auto& slice : outputSlices)
        {
            const auto& sliceInputs = slicingData.getInputsForSlice(slice);
            for(const auto& ref : sliceInputs)
            {
                if(ref->operand->originalTensor->isShapeTensor())
                {
                    containsShapeTensor = true;
                    EXPECT_EQ(ref->operand->finalShape, slice.first->operand->finalShape);
                }
            }
        }
        ASSERT_TRUE(containsShapeTensor);
    }
}

// Solution generation for dedx slicing solution traverse the shape tensor together with the output slices
// (each operation has the same coordinates for the output and the shape tensor input)
TEST_F(SRAMManagementTestDsdMasterDedx, solution_generation_traverse_the_shape_tensor_together_with_the_output_slices)
{
    pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(m_slicingStrategyList.front());
    ASSERT_NE(nullptr, strategy);

    // Generate solution and validate the execution order
    SolutionGenerator solGen(getGraph(), m_bundleDedx, strategy);
    if (solGen.fillSolution())
    {
        BundleSlicer::sliceBundle(*m_bundleDedx, getGraph());
    }

    const auto& slicingData = strategy->getMmeSlicingData();
    bool containsShapeTensor = false;

    for(const auto& slice : slicingData.getOutputSlices())
    {
        const auto& sliceInputs = slicingData.getInputsForSlice(slice);
        for(const auto& ref : sliceInputs)
        {
            if(ref->operand->originalTensor->isShapeTensor())
            {
                containsShapeTensor = true;
                EXPECT_EQ(ref->operand->finalShape, slice.first->operand->finalShape);
            }
        }
    }
    ASSERT_TRUE(containsShapeTensor);
}

// Slicer test – check that several operations consuming different parts of a shape tensor
// lead to the shape tensor being split and the outputs of the split are also shape tensors.
TEST_F(SRAMManagementTestDsdDedx, slicer_test_verify_split_to_shape_tensors_and_different_consumers)
{
    SRAMSlicingManager sm(getGraph());
    sm.sliceGraph();
    graphVisualizationPost(getGraph());
    NodeList shapeTensorConsumers = getGraph().getTensorConsumers(m_shapeTensor);
    ASSERT_EQ(1, shapeTensorConsumers.size());
    NodePtr shapeTensorSplit = shapeTensorConsumers.front();
    ASSERT_EQ(Node::TYPE_SPLIT_SHAPE, shapeTensorSplit->getNodeType());
    TensorVector splitVectors = shapeTensorSplit->getOutputs();
    EXPECT_GT(splitVectors.size(), 1);

    std::set<unsigned> nodes;
    std::set<unsigned> tensors;

    for(const TensorPtr& t : splitVectors)
    {
        ASSERT_TRUE(t->isShapeTensor());
        NodeList dedxConsumers = getGraph().getTensorConsumers(t);
        ASSERT_EQ(1, dedxConsumers.size());
        NodePtr shapeTensorDedx = dedxConsumers.front();
        ASSERT_EQ(Node::TYPE_DEDX, shapeTensorDedx->getNodeType());
        nodes.insert(shapeTensorDedx->getId());
        tensors.insert(t->getId());
    }

    EXPECT_EQ(splitVectors.size(), nodes.size());
    EXPECT_EQ(splitVectors.size(), tensors.size());
}

// Full compilation test - After slicing dynamic dedx, all dedx nodes in the sliced graph
// should have shape tensors that are produced by a split from the original shape tensor created by the test.
// fails on internal validation (LogicalOpNode::getRealTensor()) when the shape tensor is splitted
// since there in no "real tensor" input on the split node
TEST_F(SRAMManagementTestDsdDedx, full_compilation_all_dedx_nodes_has_sliced_shape_tensors_inputs)
{
    getGraph().compile();
    NodeList shapeTensorConsumers = getGraph().getTensorConsumers(m_shapeTensor);
    ASSERT_EQ(1, shapeTensorConsumers.size());
    NodePtr shapeTensorSplit = shapeTensorConsumers.front();
    ASSERT_EQ(Node::TYPE_SPLIT_SHAPE, shapeTensorSplit->getNodeType());
    TensorVector splitVectors = shapeTensorSplit->getOutputs();
    EXPECT_GT(splitVectors.size(), 1);

    std::set<unsigned> nodes;
    std::set<unsigned> tensors;

    for(const TensorPtr& t : splitVectors)
    {
        ASSERT_TRUE(t->isShapeTensor());
        NodeList dedxConsumers = getGraph().getTensorConsumers(t);
        ASSERT_EQ(1, dedxConsumers.size());
        NodePtr shapeTensorDedx = dedxConsumers.front();
        ASSERT_EQ(Node::TYPE_DEDX, shapeTensorDedx->getNodeType());
        nodes.insert(shapeTensorDedx->getId());
        tensors.insert(t->getId());
    }

    EXPECT_EQ(splitVectors.size(), nodes.size());
    EXPECT_EQ(splitVectors.size(), tensors.size());
}

TEST_F(SRAMManagementTestDsdDedxAndDeDw, full_compilation_all_dedx_nodes_has_sliced_shape_tensors_inputs)
{
    getGraph().compile();
    graphVisualizationPost(getGraph());
    NodeList shapeTensorConsumers = getGraph().getTensorConsumers(m_shapeTensor);
    ASSERT_EQ(1, shapeTensorConsumers.size());
    NodePtr shapeTensorSplit = shapeTensorConsumers.front();
    ASSERT_EQ(Node::TYPE_SPLIT_SHAPE, shapeTensorSplit->getNodeType());
    TensorVector splitVectors = shapeTensorSplit->getOutputs();
    EXPECT_GT(splitVectors.size(), 1);

    std::set<unsigned> nodes;
    std::set<unsigned> tensors;

    for(const TensorPtr& t : splitVectors)
    {
        ASSERT_TRUE(t->isShapeTensor());
        NodeList dedxConsumers = getGraph().getTensorConsumers(t);
        ASSERT_EQ(1, dedxConsumers.size());
        NodePtr shapeTensorDedx = dedxConsumers.front();
        ASSERT_EQ(Node::TYPE_DEDX, shapeTensorDedx->getNodeType());
        nodes.insert(shapeTensorDedx->getId());
        tensors.insert(t->getId());
    }

    EXPECT_EQ(splitVectors.size(), nodes.size());
    EXPECT_EQ(splitVectors.size(), tensors.size());
}