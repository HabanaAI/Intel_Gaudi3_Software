#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "graph_optimizer_test_configuration.hpp"
#include "habana_pass.h"
#include <memory>
#include "node_factory.h"
#include "passes/ir_translation/ir_to_synapse_translator.hpp"
#include "passes/ir_translation/synapse_graph_wrapper.hpp"
#include "passes/synapse_mlir_optimizer.h"
#include "perf_lib_layer_params.h"

using namespace gc_protocol;

class TestSynapseMLIROptimizer : public GraphOptimizerTest
{
public:
    static inline bool s_shouldRunMLIRTests = false;
    static void        SetUpTestSuite()
    {
        GlobalConfManager::instance().init("");
        s_shouldRunMLIRTests = GCFG_RUN_GC_MLIR_TESTS.value();
    }

protected:
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        if (!s_shouldRunMLIRTests) GTEST_SKIP() << "Not running gc mlir tests";
        GCFG_SYNAPSE_MLIR_MODE.setValue(SynapseMLIROptimizerModeEnabledValidation);
    }
};

TEST_F(TestSynapseMLIROptimizer, test_synapse_mlir_optimizer_sanity)
{
    // currently just run the pass, test nothing
    GaudiGraph g;
    ASSERT_TRUE(synapseMLIROptimizer(g));
}

// below are auxiliary alias and struct for relating ir tensors to ir nodes
using IRNodePtr       = std::shared_ptr<ProtocolNode>;
using IRTensorPtr     = std::shared_ptr<ProtocolTensor>;
using IRNodeList      = std::list<IRNodePtr>;
using IRTensorsMap    = std::map<uint64_t, IRTensorPtr>;  // mapping tensors ids to IR tensors
using IRTensorsVector = std::vector<IRTensorPtr>;

struct NodeTensors
{
    IRTensorsVector inputs;
    IRTensorsVector outputs;
};

using IRNodesIdsToTensorsMap = std::map<uint64_t, NodeTensors>;

/*
 * A GraphInterface implementation for testing needs.
 * Iterates gc_protocol IR objects , and doesn't change them before invoking handlers,
 * since they are already in translator::protocol IR format.
 * (The source IR is gc_protocol IR)
 */
struct DefaultGraphWrapper : public ProtocolGraph
{
    DefaultGraphWrapper(IRNodeList& nodes, IRNodesIdsToTensorsMap& nodesIdsAndTensors)
    : m_wrapperIRNodes(nodes), m_wrapperIRNodesAndIRTensors(nodesIdsAndTensors) {};
    ~DefaultGraphWrapper() { m_wrapperIRNodes.clear(); }
    gcapi::DeviceId_t getDeviceId() const override { return gcapi::DeviceId_t::DEVICE_ID_GAUDI; }
    unsigned getMaxAvailableTpc() const override { return TPCNode::getMaxAvailableTpc(tpc_lib_api::DEVICE_ID_GAUDI); }
    unsigned getEagerMode() const override { return 0;}
    unsigned getNumNodes() const override { return m_wrapperIRNodes.size(); };
    bool     foreachNode(ProtocolNodeHandler& handler) const override
    {
        for (IRNodePtr irNode : m_wrapperIRNodes)
        {
            handler.handleNode(*irNode);
        }
        return true;
    }

    bool foreachInputTensor(uint64_t nodeId, ProtocolInputTensorHandler& handler) const override
    {
        NodeTensors nodeTensors = m_wrapperIRNodesAndIRTensors.find(nodeId)->second;
        for (auto tensor : nodeTensors.inputs)
        {
            handler.handleTensor(*tensor);
        }
        return true;
    }

    unsigned getNumInputTensors(uint64_t nodeId) const override { return 0; };

    bool foreachOutputTensor(uint64_t nodeId, ProtocolOutputTensorHandler& handler) const override
    {
        NodeTensors nodeTensors = m_wrapperIRNodesAndIRTensors.find(nodeId)->second;
        for (auto tensor : nodeTensors.outputs)
        {
            handler.handleTensor(*tensor);
        }
        return true;
    }

    unsigned getNumOutputTensors(uint64_t nodeId) const override { return 0; };

    // below data structs provide a graph representation
    IRNodeList             m_wrapperIRNodes;              // execution order sorted IR nodes
    IRNodesIdsToTensorsMap m_wrapperIRNodesAndIRTensors;  // maps between IR nodes to their IR Tensors
};

/*
 * A Translator for testing needs.
 * Just translates from gc_protocol IR to gc_protocol IR (the target IR is also gc_protocol IR)
 */
struct DefaultGraphTranslator
: public ProtocolNodeHandler
, public ProtocolInputTensorHandler
, public ProtocolOutputTensorHandler
{
    DefaultGraphTranslator(ProtocolGraph& graphProvider) : m_graphWrapper(graphProvider) {}

    ~DefaultGraphTranslator()
    {
        for (auto irTensor : m_tranlsatorIRTensors)
        {
            if (irTensor.second == nullptr) continue;
            // deallocate shape related arrays that were allocated at ConvertInput/OutputTensors
            clearIRTensorShapeArrays(*irTensor.second);
        }
        m_translatorIRNodes.clear();
        m_translatorIRNodesIdsAndIRTensors.clear();
        m_currentInputs.clear();
        m_currentOutputs.clear();
    }

    bool convertToDefault()
    {
        m_graphWrapper.foreachNode(*this);
        return true;
    };

    bool handleNode(const ProtocolNode& node) override
    {
        m_currentOutputs.clear();
        m_currentInputs.clear();

        IRNodePtr nodePtr = std::make_shared<ProtocolNode>(node);
        m_graphWrapper.foreachInputTensor(node.id, *this);
        m_graphWrapper.foreachOutputTensor(node.id, *this);

        NodeTensors nodeTensors {m_currentInputs, m_currentOutputs};
        m_translatorIRNodesIdsAndIRTensors.insert({node.id, nodeTensors});
        m_translatorIRNodes.push_back(nodePtr);

        return true;
    }
    bool handleInputTensor(const ProtocolTensor& tensor) override
    {
        if (auto tensorItr = m_tranlsatorIRTensors.find(tensor.id); tensorItr != m_tranlsatorIRTensors.end())
        {
            m_currentInputs.push_back(tensorItr->second);
        }
        else
        {
            IRTensorPtr irTensor = createIRTensorCopy(tensor);
            m_tranlsatorIRTensors.insert({irTensor->id, irTensor});
            m_currentInputs.push_back(irTensor);
        }
        return true;
    }
    bool handleOutputTensor(const ProtocolTensor& tensor) override
    {
        if (auto tensorItr = m_tranlsatorIRTensors.find(tensor.id); tensorItr != m_tranlsatorIRTensors.end())
        {
            m_currentOutputs.push_back(tensorItr->second);
        }
        else
        {
            IRTensorPtr tensorPtr = createIRTensorCopy(tensor);
            m_tranlsatorIRTensors.insert({tensorPtr->id, tensorPtr});
            m_currentOutputs.push_back(tensorPtr);
        }
        return true;
    }

    IRTensorPtr createIRTensorCopy(const ProtocolTensor& origIRTensor)
    {
        IRTensorPtr copyIRTensor     = std::make_shared<ProtocolTensor>();
        copyIRTensor->id             = origIRTensor.id;
        copyIRTensor->name           = origIRTensor.name;
        copyIRTensor->elementType    = origIRTensor.elementType;
        copyIRTensor->rank           = origIRTensor.rank;
        copyIRTensor->pData          = origIRTensor.pData;

        // allocate and copy shape related arrays
        // can't use the orig tensor array pointers, since they were copied by value and may be dangling at some point
        auto copySizes    = new uint64_t[Tensor::c_tensorMaxNDim];
        auto copyMinSizes = new uint64_t[Tensor::c_tensorMaxNDim];
        auto copyStrides  = new uint64_t[Tensor::c_numOfNStrides];
        for (unsigned i = 0; i < Tensor::c_tensorMaxNDim; i++)
        {
            copySizes[i]    = origIRTensor.maxSizes[i];
            copyMinSizes[i] = origIRTensor.minSizes[i];
            copyStrides[i]  = origIRTensor.strides[i];
        }
        copyStrides[Tensor::c_numOfNStrides - 1] = origIRTensor.strides[Tensor::c_numOfNStrides - 1];

        copyIRTensor->maxSizes = copySizes;
        copyIRTensor->minSizes = copyMinSizes;
        copyIRTensor->strides  = copyStrides;

        if (origIRTensor.attributes != nullptr)
        {
            copyIRTensor->attributes = new(ProtocolTensorAttributes);
            copyIRTensor->attributes->isGraphOutput  = origIRTensor.attributes->isGraphOutput;
            copyIRTensor->attributes->isInitialized  = origIRTensor.attributes->isInitialized;
            copyIRTensor->attributes->isNotNeeded    = origIRTensor.attributes->isNotNeeded;
            copyIRTensor->attributes->type           = origIRTensor.attributes->type;
            copyIRTensor->attributes->tensorDataType = origIRTensor.attributes->tensorDataType;

            if (origIRTensor.attributes->quantizationParams != nullptr)
            {
                copyIRTensor->attributes->quantizationParams            = new(ProtocolTensorQuantizationParams);
                copyIRTensor->attributes->quantizationParams->zeroPoint =
                    origIRTensor.attributes->quantizationParams->zeroPoint;
                copyIRTensor->attributes->quantizationParams->scale     =
                    origIRTensor.attributes->quantizationParams->scale;
            }
        }
        if (origIRTensor.tensorSection != nullptr)
        {
            copyIRTensor->tensorSection = new(ProtocolTensorSection_t);
            copyIRTensor->tensorSection->type   = origIRTensor.tensorSection->type;
            copyIRTensor->tensorSection->offset = origIRTensor.tensorSection->offset;
            copyIRTensor->tensorSection->id     = origIRTensor.tensorSection->id;
        }

        return copyIRTensor;
    }

    void clearIRTensorShapeArrays(ProtocolTensor& IRtensor)
    {
        delete[] IRtensor.maxSizes;
        delete[] IRtensor.minSizes;
        delete[] IRtensor.strides;
        if (IRtensor.attributes != nullptr)
        {   if (IRtensor.attributes->quantizationParams != nullptr)
            {
                delete[] IRtensor.attributes->quantizationParams;
            }
            delete[] IRtensor.attributes;
        }
        delete[] IRtensor.tensorSection;
    }

    ProtocolGraph& m_graphWrapper;

    IRNodeList             m_translatorIRNodes;                 // execution order sorted IR nodes that were converted
    IRTensorsMap           m_tranlsatorIRTensors;               // already created IR tensors
    IRNodesIdsToTensorsMap m_translatorIRNodesIdsAndIRTensors;  // maps between IR nodes to their IR Tensors
    // temporary vectors to store tensors that are related to the current node being converted
    IRTensorsVector m_currentInputs;
    IRTensorsVector m_currentOutputs;
};

class TestSynapseMLIROptimizerTranslation : public TestSynapseMLIROptimizer
{
public:
    void compareTensorsWithoutId(const TensorPtr origTensor, const TensorPtr newTensor)
    {
        EXPECT_NE(origTensor->getId(), newTensor->getId());
        EXPECT_NE(origTensor->getName(), newTensor->getName());
        // check that new tensor name starts with old tensor name
        EXPECT_TRUE(newTensor->getName().rfind(origTensor->getName()) == 0);
        EXPECT_EQ(origTensor->getElementType(), newTensor->getElementType());
        EXPECT_EQ(origTensor->getDim(), newTensor->getDim());
        EXPECT_EQ(origTensor->isDynamicShape(), newTensor->isDynamicShape());
        EXPECT_EQ(origTensor->getAllNSizesInElements(), newTensor->getAllNSizesInElements());
        EXPECT_EQ(origTensor->getNMinimalSizesInElements(), newTensor->getNMinimalSizesInElements());
        EXPECT_EQ(origTensor->getNStridesInElements(), newTensor->getNStridesInElements());
        EXPECT_TRUE(origTensor->compareGeometry(*newTensor));
        EXPECT_EQ(origTensor->getDenseSizeInBytes(), newTensor->getDenseSizeInBytes());
        EXPECT_EQ(origTensor->getBufferSizeInBytes(), newTensor->getBufferSizeInBytes());
        EXPECT_EQ(origTensor->getBufferDataType(), newTensor->getBufferDataType());
        EXPECT_EQ(origTensor->getBatchPos(), newTensor->getBatchPos());
        EXPECT_EQ(origTensor->getAddress(), newTensor->getAddress());
        EXPECT_EQ(origTensor->getMemorySectionID(), newTensor->getMemorySectionID());
        EXPECT_EQ(origTensor->getMemorySectionOffset(), newTensor->getMemorySectionOffset());
        EXPECT_EQ(origTensor->isPersistent(), newTensor->isPersistent());
        EXPECT_EQ(origTensor->isPartOfRMWSection(), newTensor->isPartOfRMWSection());
        EXPECT_EQ(origTensor->getScale(), newTensor->getScale());
        EXPECT_EQ(origTensor->getZeroPoint(), newTensor->getZeroPoint());
        EXPECT_EQ(origTensor->isStaticParam(), newTensor->isStaticParam());
    }

    void compareNodesWithoutId(const NodePtr origNode, const NodePtr newNode)
    {
        EXPECT_NE(origNode->getId(), newNode->getId());
        EXPECT_EQ(origNode->getGUID(), newNode->getGUID());
        EXPECT_EQ(origNode->getNumInputs(), newNode->getNumInputs());
        EXPECT_EQ(origNode->getNumOutputs(), newNode->getNumOutputs());
        EXPECT_EQ(origNode->getParamsRawData().size(), newNode->getParamsRawData().size());
        EXPECT_EQ(origNode->getParamsRawData(), newNode->getParamsRawData());
        //EXPECT_EQ(origNode->getNodePrecision(), newNode->getNodePrecision()); TODO - hadnle node precision

        auto newInputsIterator = newNode->getInputs().begin();
        for (auto originalInput : origNode->getInputs())
        {
            auto newInput = *newInputsIterator++;
            if (originalInput == nullptr)
            {
                EXPECT_EQ(newInput, nullptr);
                continue;
            }
            compareTensorsWithoutId(originalInput, newInput);
        }
        auto newOutputsIterator = newNode->getOutputs().begin();
        for (auto originalOutput : origNode->getOutputs())
        {
            auto newOutput = *newOutputsIterator++;
            if (originalOutput == nullptr)
            {
                ASSERT_EQ(newOutput, nullptr);
                continue;
            }
            compareTensorsWithoutId(originalOutput, newOutput);
        }
    }

    void compareGraphsWithoutId(HabanaGraph& origGraph, HabanaGraph& newGraph)
    {
        auto oldGraphNodes = origGraph.getExeSortedNodes();
        auto newGraphNodes = newGraph.getExeSortedNodes();
        ASSERT_EQ(oldGraphNodes.size(), newGraphNodes.size());
        ASSERT_EQ(origGraph.getTensors().size(), newGraph.getTensors().size());
        // compare new nodes to original nodes - should be same accept IDs
        auto newNodesIterator = newGraphNodes.begin();
        for (auto origNode : oldGraphNodes)
        {
            auto newNode = *newNodesIterator++;
            compareNodesWithoutId(origNode, newNode);
        }
        ASSERT_TRUE(newGraph.isomorphicTo(origGraph));  // verify graph "shape" is same
    }
};

TEST_F(TestSynapseMLIROptimizerTranslation, test_translation_flow_sanity_single_node)
{
    /*
     * Test Synapse translation flow for a single node.
     * Translate from Synapse to IR and from IR to Synapse.
     */

    // create simple node
    GaudiGraph            g;
    TSize                 sizes[2] = {5, 5};
    TensorPtr             input(new Tensor(2U, sizes, syn_type_float));
    TensorPtr             output(new Tensor(2U, sizes, syn_type_float));

    float data[25] = {0};
    input->setTensorBuffer((void*)data, input->getTotalSizeInBytes(), syn_type_float, true);
    input->setAsStaticParam();

    ns_ReluKernel::Params reluParams;
    reluParams.threshold.f = 0.75;
    NodePtr reluNode =
        NodeFactory::createNode({input}, {output}, &reluParams, sizeof(reluParams), "relu_fwd_f32", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, reluNode));
    // create synapse graph wrapper and default graph translator
    SynapseGraphWrapper    synapseGraphWrapper(g, false);
    DefaultGraphTranslator defaultGraphTranslator(synapseGraphWrapper);
    // convert synapse->IR->IR
    defaultGraphTranslator.convertToDefault();
    // create default graph wrapper and synapse graph translator
    DefaultGraphWrapper     defaultGraphWrapper(defaultGraphTranslator.m_translatorIRNodes,
                                            defaultGraphTranslator.m_translatorIRNodesIdsAndIRTensors);
    IRToSynapseDummyGraphTranslator mlirToSynapseTranslator(defaultGraphWrapper);
    // convert IR->IR->synapse
    mlirToSynapseTranslator.startTranslationToSynapse();
    // create new Synapse graph from converted nodes
    GaudiGraph newGraph;
    for (const auto& newSynNode : mlirToSynapseTranslator.getExecutionSortedNodes())
    {
        ASSERT_TRUE(GraphEditor::addNode(newGraph, newSynNode));
    }
    // verify that original and new graph are same, except nodes & tensors ids
    compareGraphsWithoutId(g, newGraph);
}

TEST_F(TestSynapseMLIROptimizerTranslation, test_translation_flow_sanity_several_tpc_nodes)
{
    /*
     * Test Synapse translation flow for a several tpc node.
     * Translate from Synapse to IR and from IR to Synapse.
     */

    GaudiGraph     g;
    synDataType    dtype              = syn_type_bf16;  // test with non-default dtype
    const TSize    inSizes[4]         = {3, 9, 9, 3};
    const TSize    bnormStatsSizes[1] = {3};

    // building graph from nodes that are supported in mlir: add(t1,t2)->relu(t3)->memcpy(t4)->batchnorm(t5)
    TensorPtr addInput1 = std::make_shared<Tensor>(4, inSizes, dtype);
    addInput1->setName("addInput1");
    TensorPtr addInput2 = std::make_shared<Tensor>(4, inSizes, dtype);
    addInput2->setName("addInput2");
    TensorPtr addOutput = std::make_shared<Tensor>(4, inSizes, dtype);
    addOutput->setName("addOutput");
    TensorPtr reluOutput = std::make_shared<Tensor>(4, inSizes, dtype);
    reluOutput->setName("reluOutput");
    TensorPtr memcopyOutput = std::make_shared<Tensor>(4, inSizes, dtype);
    memcopyOutput->setName("memcopyOutput");
    TensorPtr inBeta = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inBeta->setName("inBeta");
    TensorPtr inGamma = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inGamma->setName("inGamma");
    TensorPtr inRunningMean = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inRunningMean->setName("inRunningMean");
    TensorPtr inRunningVar = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inRunningVar->setName("inRunningVar");
    TensorPtr bnormOutput = std::make_shared<Tensor>(4, inSizes, dtype);
    bnormOutput->setName("bnormOutput");
    TensorPtr outMean = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    outMean->setName("outMean");
    TensorPtr outLtsd = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    outLtsd->setName("outLtsd");
    TensorPtr outRunningMean = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    outRunningMean->setName("outRunningMean");
    TensorPtr outRunningVar = std::make_shared<Tensor>(4, bnormStatsSizes, dtype);
    outRunningVar->setName("outRunningVar");

    ns_ReluKernel::Params reluParams;
    reluParams.threshold.f = 0.75;

    ns_BatchNormKernel::ParamsV2 params;
    params.momentum   = 0.1;
    params.epsilon    = 0.00001;
    params.isTraining = true;

    NodeVector nodes(5);

    NodePtr addNode = NodeFactory::createNode({addInput1, addInput2}, {addOutput}, nullptr, 0, "add_fwd_bf16", "add");
    ASSERT_TRUE(addNode != nullptr);

    NodePtr reluNode =
        NodeFactory::createNode({addOutput}, {reluOutput}, &reluParams, sizeof(reluParams), "relu_fwd_bf16", "relu");
    ASSERT_TRUE(reluNode != nullptr);
    NodePtr memcopyNode =
        NodeFactory::createNode({reluOutput}, {memcopyOutput}, nullptr, NodeFactory::memcpyNodeTypeName, "memcpy");
    ASSERT_TRUE(memcopyNode != nullptr);
    memcopyNode->setNodePrecision(syn_type_bf16);
    NodePtr batchNormNode = NodeFactory::createNode({memcopyOutput, inBeta, inGamma, inRunningMean, inRunningVar},
                                                    {bnormOutput, outMean, outLtsd, outRunningMean, outRunningVar},
                                                    &params,
                                                    "batch_norm_fwd_bf16",
                                                    "bn");
    ASSERT_TRUE(batchNormNode != nullptr);
    nodes.push_back(batchNormNode);

    ASSERT_TRUE(GraphEditor::addNode(g, addNode));
    ASSERT_TRUE(GraphEditor::addNode(g, reluNode));
    ASSERT_TRUE(GraphEditor::addNode(g, memcopyNode));
    ASSERT_TRUE(GraphEditor::addNode(g, batchNormNode));

    // create synapse graph wrapper and default graph translator
    SynapseGraphWrapper    synapseGraphWrapper(g, false);
    DefaultGraphTranslator defaultGraphTranslator(synapseGraphWrapper);
    // convert synapse->IR->IR
    defaultGraphTranslator.convertToDefault();
    // create default graph wrapper and synapse graph translator
    DefaultGraphWrapper     defaultGraphWrapper(defaultGraphTranslator.m_translatorIRNodes,
                                             defaultGraphTranslator.m_translatorIRNodesIdsAndIRTensors);
    IRToSynapseDummyGraphTranslator mlirToSynapseTranslator(defaultGraphWrapper);
    // convert IR->IR->synapse
    mlirToSynapseTranslator.startTranslationToSynapse();
    // create new Synapse graph from converted nodes
    GaudiGraph newGraph;
    for (const auto& newSynNode : mlirToSynapseTranslator.getExecutionSortedNodes())
    {
        ASSERT_TRUE(GraphEditor::addNode(newGraph, newSynNode));
    }
    // verify that original and new graph are same, except nodes & tensors ids
    compareGraphsWithoutId(g, newGraph);
}

TEST_F(TestSynapseMLIROptimizerTranslation, test_translation_flow_2_way_sanity_add_node)
{
    /*
     * Test Synapse translation flow for a single node.
     * Translate from Synapse to IR and from IR to Synapse.
     */

    // create simple node
    GaudiGraph g;
    TSize      sizes[2] = {5, 5};

    TensorPtr input1(new Tensor(2U, sizes, syn_type_float));
    TensorPtr input2(new Tensor(2U, sizes, syn_type_float));
    TensorPtr output(new Tensor(2U, sizes, syn_type_float));

    NodePtr addNode = NodeFactory::createNode({input1, input2}, {output}, nullptr, 0, "add_fwd_f32", "add");
    ASSERT_TRUE(GraphEditor::addNode(g, addNode));
    ASSERT_TRUE(synapseMLIROptimizer(g));
}

TEST_F(TestSynapseMLIROptimizerTranslation, test_translation_flow_2_way_sanity_several_nodes)
{
    /*
     * Test 2-way translation flow for a several tpc node.
     */

    GaudiGraph     g;
    synDataType    dtype              = syn_type_bf16;  // test with non-default dtype
    const TSize    inSizes[4]         = {3, 9, 9, 3};
    const TSize    bnormStatsSizes[1] = {3};

    // building graph from nodes that are supported in mlir: add(t1,t2)->relu(t3)->memcpy(t4)->batchnorm(t5)
    TensorPtr addInput1 = std::make_shared<Tensor>(4, inSizes, dtype);
    addInput1->setName("addInput1");
    TensorPtr addInput2 = std::make_shared<Tensor>(4, inSizes, dtype);
    addInput2->setName("addInput2");
    TensorPtr addOutput = std::make_shared<Tensor>(4, inSizes, dtype);
    addOutput->setName("addOutput");
    TensorPtr reluOutput = std::make_shared<Tensor>(4, inSizes, dtype);
    reluOutput->setName("reluOutput");
    TensorPtr memcopyOutput = std::make_shared<Tensor>(4, inSizes, dtype);
    memcopyOutput->setName("memcopyOutput");
    TensorPtr inBeta = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inBeta->setName("inBeta");
    TensorPtr inGamma = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inGamma->setName("inGamma");
    TensorPtr inRunningMean = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inRunningMean->setName("inRunningMean");
    TensorPtr inRunningVar = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inRunningVar->setName("inRunningVar");
    TensorPtr bnormOutput = std::make_shared<Tensor>(4, inSizes, dtype);
    bnormOutput->setName("bnormOutput");
    TensorPtr outMean = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    outMean->setName("outMean");
    TensorPtr outLtsd = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    outLtsd->setName("outLtsd");
    TensorPtr outRunningMean = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    outRunningMean->setName("outRunningMean");
    TensorPtr outRunningVar = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    outRunningVar->setName("outRunningVar");

    ns_ReluKernel::Params reluParams;
    reluParams.threshold.f = 0.75;

    ns_BatchNormKernel::ParamsV2 params;
    params.momentum    = 0.5;
    params.epsilon     = 0.6;
    params.threshold.f = 0.7;
    params.isTraining  = true;

    NodeVector nodes(5);

    NodePtr addNode = NodeFactory::createNode({addInput1, addInput2}, {addOutput}, nullptr, 0, "add_fwd_bf16", "add");
    ASSERT_TRUE(addNode != nullptr);

    NodePtr reluNode =
        NodeFactory::createNode({addOutput}, {reluOutput}, &reluParams, sizeof(reluParams), "relu_fwd_bf16", "relu");
    ASSERT_TRUE(reluNode != nullptr);
    NodePtr memcopyNode =
        NodeFactory::createNode({reluOutput}, {memcopyOutput}, nullptr, NodeFactory::memcpyNodeTypeName, "memcpy");
    ASSERT_TRUE(memcopyNode != nullptr);
    memcopyNode->setNodePrecision(syn_type_bf16);
    NodePtr batchNormNode = NodeFactory::createNode({memcopyOutput, inBeta, inGamma, inRunningMean, inRunningVar},
                                                    {bnormOutput, outMean, outLtsd, outRunningMean, outRunningVar},
                                                    &params,
                                                    "batch_norm_fwd_bf16",
                                                    "bn");
    ASSERT_TRUE(batchNormNode != nullptr);
    nodes.push_back(batchNormNode);

    addNode->getNodeAnnotation().inputPermutations.clear();
    addNode->getNodeAnnotation().inputPermutations.push_back(DimVector {0, 2, 1, 3});
    addNode->getNodeAnnotation().inputPermutations.push_back(DimVector {2, 1, 0, 3});

    batchNormNode->getNodeAnnotation().inputPermutations.clear();
    batchNormNode->getNodeAnnotation().inputPermutations.push_back(DimVector {3, 2, 0, 1});
    batchNormNode->getNodeAnnotation().inputPermutations.push_back(DimVector {});
    batchNormNode->getNodeAnnotation().inputPermutations.push_back(DimVector {});
    batchNormNode->getNodeAnnotation().inputPermutations.push_back(DimVector {});
    batchNormNode->getNodeAnnotation().inputPermutations.push_back(DimVector {});

    ASSERT_TRUE(GraphEditor::addNode(g, addNode));
    ASSERT_TRUE(GraphEditor::addNode(g, reluNode));
    ASSERT_TRUE(GraphEditor::addNode(g, memcopyNode));
    ASSERT_TRUE(GraphEditor::addNode(g, batchNormNode));

    ASSERT_TRUE(synapseMLIROptimizer(g));
}

TEST_F(TestSynapseMLIROptimizerTranslation, test_translation_flow_2_way_sanity_mme_node)
{
    /*
     * Test 2-way translation flow for a mme node.
     */

    const TSize C         = 2;
    const TSize W         = 5;
    const TSize H         = 5;
    const TSize N         = 1;
    const TSize K         = 2;
    const TSize R         = 2;
    const TSize S         = 2;
    const TSize inSizes[] = {C, W, H, N};
    const TSize weightsStride   = 1;
    const TSize weightsPadding  = 1;
    const TSize weightsSizes[] = {K, C, S, R};
    const TSize outW       = ((W - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outH       = ((H - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outC       = K;
    const TSize outSizes[] = {outC, outW, outH, N};

    synConvolutionParams params;
    params.dH   = weightsStride;
    params.dW   = weightsStride;
    params.kH   = S;
    params.kW   = R;
    params.padT = weightsPadding;
    params.padB = weightsPadding;
    params.padL = weightsPadding;

    GaudiGraph     g;
    synDataType    dtype = syn_type_bf16;  // test with non-default dtype

    TensorPtr in      = std::make_shared<Tensor>(4U, inSizes, dtype);
    TensorPtr weights = std::make_shared<Tensor>(4U, weightsSizes, dtype);
    TensorPtr out     = std::make_shared<Tensor>(4U, outSizes, dtype);

    Node::NodeProperties layouts;
    layouts.inputLayouts  = {gc::Layout("WHCN"), gc::Layout("CSKR")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr conv1         = NodeFactory::createNode({in, weights}, {out}, &params, "spatial_convolution", "conv1",
                                                     layouts);
    GraphEditor::addNode(g, conv1);

    ASSERT_TRUE(synapseMLIROptimizer(g));
}

TEST_F(TestSynapseMLIROptimizerTranslation, test_translation_flow_2_way_sanity_mme_node_with_static_weight)
{
    /*
     * Test 2-way translation flow for a mme node.
     */

    const TSize C         = 2;
    const TSize W         = 5;
    const TSize H         = 5;
    const TSize N         = 1;
    const TSize K         = 2;
    const TSize R         = 2;
    const TSize S         = 2;
    const TSize inSizes[] = {C, W, H, N};
    const TSize weightsStride  = 1;
    const TSize weightsPadding = 1;
    const TSize weightsSizes[] = {K, C, S, R};
    const TSize outW       = ((W - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outH       = ((H - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outC       = K;
    const TSize outSizes[] = {outC, outW, outH, N};

    synConvolutionParams params;
    params.dH   = weightsStride;
    params.dW   = weightsStride;
    params.kH   = S;
    params.kW   = R;
    params.padT = weightsPadding;
    params.padB = weightsPadding;
    params.padL = weightsPadding;

    GaudiGraph  g;
    synDataType dtype = syn_type_bf16;  // test with non-default dtype

    TensorPtr in      = std::make_shared<Tensor>(4U, inSizes, dtype);
    TensorPtr weights = std::make_shared<Tensor>(4U, weightsSizes, dtype);
    TensorPtr out     = std::make_shared<Tensor>(4U, outSizes, dtype);

    float data[16] = {0};
    weights->setTensorBuffer((void*)data, weights->getTotalSizeInBytes(), dtype, true);
    weights->setAsWeights();
    weights->setAsStaticParam();

    Node::NodeProperties layouts;
    layouts.inputLayouts  = {gc::Layout("WHCN"), gc::Layout("CSKR")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr conv1 = NodeFactory::createNode({in, weights}, {out}, &params, "spatial_convolution", "conv1", layouts);
    conv1->getNodeAnnotation().inputPermutations.clear();
    conv1->getNodeAnnotation().inputPermutations.push_back(DimVector {0, 1, 2, 3});
    conv1->getNodeAnnotation().inputPermutations.push_back(DimVector {2, 0, 3, 1});

    GraphEditor::addNode(g, conv1);

    ASSERT_TRUE(synapseMLIROptimizer(g));
}

TEST_F(TestSynapseMLIROptimizerTranslation, test_translation_flow_2_way_node_replacment_mme_add)
{
    /*
     * Test node replacement flow - extraction case:
     * Input graph - Conv2d node with bias
     * Expected output graph - Conv2d without bias and Add node
     */
    GCFG_SYNAPSE_MLIR_MODE.setValue(SynapseMLIROptimizerModeEnabled);

    const TSize C              = 2;
    const TSize W              = 5;
    const TSize H              = 5;
    const TSize N              = 1;
    const TSize K              = 2;
    const TSize R              = 2;
    const TSize S              = 2;
    const TSize inSizes[]      = {C, W, H, N};
    const TSize weightsStride  = 1;
    const TSize weightsPadding = 1;
    const TSize weightsSizes[] = {K, C, S, R};
    const TSize outW           = ((W - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outH           = ((H - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outC           = K;
    const TSize outSizes[]     = {outC, outW, outH, N};

    synConvolutionParams params;
    params.dH   = weightsStride;
    params.dW   = weightsStride;
    params.kH   = S;
    params.kW   = R;
    params.padT = weightsPadding;
    params.padB = weightsPadding;
    params.padL = weightsPadding;

    GaudiGraph  g;
    synDataType dtype = syn_type_bf16;

    TensorPtr in      = std::make_shared<Tensor>(4U, inSizes, dtype);
    TensorPtr weights = std::make_shared<Tensor>(4U, weightsSizes, dtype);
    TensorPtr bias    = std::make_shared<Tensor>(4U, outSizes, dtype);
    TensorPtr out     = std::make_shared<Tensor>(4U, outSizes, dtype);

    Node::NodeProperties layouts;
    layouts.inputLayouts  = {gc::Layout("WHCN"), gc::Layout("CSKR"), gc::Layout("WHCN")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr conv1 = NodeFactory::createNode({in, weights, bias}, {out}, &params, "spatial_convolution", "conv1", layouts);
    GraphEditor::addNode(g, conv1);

    ASSERT_TRUE(synapseMLIROptimizer(g));
    ASSERT_EQ(g.getNumNodes(), 2); // verify extraction occurred
    for (const auto& node : g.getExeSortedNodes())
    {
        // verify nodes are conv and add
        ASSERT_TRUE(node->getGUID().rfind(std::string("spatial_convolution"), 0) == 0 ||
                    node->getGUID().rfind(std::string("add_fwd"), 0) == 0);
        if (node->getGUID().rfind(std::string("spatial_convolution"), 0) == 0)
        {
            // verify no bias
            ASSERT_EQ(node->getNumInputs(), 2);
        }
    }
}

// currently, disabled since synaspe doesn't have kernel "batch_norm_relu" as defined in mlir side.
// the correct guid is probably "batch_norm_stage2_relu_fwd".
TEST_F(TestSynapseMLIROptimizerTranslation, DISABLED_test_translation_flow_2_way_node_replacment_fuse_bnorm)
{
    /* Test node replacement flow - fusion case:
     * Input graph - Batcnorm node followed by Relu node
     * Expected output graph - Fused Batchnorm-relu node
     */

    GCFG_SYNAPSE_MLIR_MODE.setValue(SynapseMLIROptimizerModeEnabled);

    GaudiGraph     g;
    synDataType    dtype              = syn_type_single;
    const TSize    inSizes[4]         = {3, 9, 9, 3};
    const TSize    bnormStatsSizes[1] = {3};


    TensorPtr bnormInput = std::make_shared<Tensor>(4, inSizes, dtype);
    bnormInput->setName("bnormInput");
    TensorPtr inBeta = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inBeta->setName("inBeta");
    TensorPtr inGamma = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inGamma->setName("inGamma");
    TensorPtr inRunningMean = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inRunningMean->setName("inRunningMean");
    TensorPtr inRunningVar = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    inRunningVar->setName("inRunningVar");
    TensorPtr bnormOutput = std::make_shared<Tensor>(4, inSizes, dtype);
    bnormOutput->setName("bnormOutput");
    TensorPtr outMean = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    outMean->setName("outMean");
    TensorPtr outLtsd = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    outLtsd->setName("outLtsd");
    TensorPtr outRunningMean = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    outRunningMean->setName("outRunningMean");
    TensorPtr outRunningVar = std::make_shared<Tensor>(1, bnormStatsSizes, dtype);
    outRunningVar->setName("outRunningVar");

    TensorPtr reluOutput = std::make_shared<Tensor>(4, inSizes, dtype);
    reluOutput->setName("reluOutput");

//    ns_ReluKernel::Params reluParams;
//    reluParams.threshold.f = 0;

    ns_BatchNormKernel::ParamsV2 params;
    params.momentum    = 0.5;
    params.epsilon     = 0.6;
    params.threshold.f = 0.7;

    NodePtr batchNormNode = NodeFactory::createNode({bnormInput, inBeta, inGamma, inRunningMean, inRunningVar},
                                                    {bnormOutput, outMean, outLtsd, outRunningMean, outRunningVar},
                                                    &params,
                                                    sizeof(params),
                                                    "batch_norm_fwd_f32",
                                                    "bn");
    ASSERT_TRUE(batchNormNode != nullptr);
    NodePtr reluNode =
        NodeFactory::createNode({bnormOutput}, {reluOutput}, nullptr, 0, "relu_fwd_f32", "relu");
    ASSERT_TRUE(reluNode != nullptr);

    ASSERT_TRUE(GraphEditor::addNode(g, reluNode));
    ASSERT_TRUE(GraphEditor::addNode(g, batchNormNode));
    ASSERT_TRUE(g.getNumNodes() == 2);
    ASSERT_TRUE(synapseMLIROptimizer(g));
    ASSERT_TRUE(g.getNumNodes() == 1);

}
