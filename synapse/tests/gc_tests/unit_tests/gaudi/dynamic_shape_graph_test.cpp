#include "compilation_hal_reader.h"
#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "smf/shape_func_registry.h"
#include "tensor.h"

class DynamicShapeGraph : public GraphOptimizerTest
{
public:
    GaudiGraph m_graph;

    TensorPtr createTensor(const std::vector<TSize>& maxSizes, const std::vector<TSize>& minSizes = {})
    {
        if (minSizes.empty())
        {
            return std::make_shared<Tensor>(maxSizes.size(), maxSizes.data(), syn_type_float);
        }
        else
        {
            return std::make_shared<Tensor>(maxSizes.size(), maxSizes.data(), syn_type_float,
                    nullptr, nullptr, false, false, INVALID_BATCH_POS, minSizes.data(), DATA_TENSOR_DYNAMIC);
        }
    }

    TensorPtr createPersistentTensor(const std::vector<TSize>& maxSizes, const std::vector<TSize>& minSizes = {})
    {
        auto tensor = createTensor(maxSizes, minSizes);
        synMemoryDescriptor memDesc(true /* persistent */);
        tensor->setMemoryDescriptor(memDesc);
        tensor->setDramOffset(0x1000000);
        tensor->map();
        return tensor;
    }

    TensorPtr createShapeTensor(const std::vector<TSize>& maxSizes, const std::vector<TSize>& minSizes)
    {
        return std::make_shared<Tensor>(maxSizes.size(),
                                        maxSizes.data(),
                                        syn_type_float,
                                        nullptr,
                                        nullptr,
                                        false,
                                        false,
                                        INVALID_BATCH_POS,
                                        minSizes.data(),
                                        SHAPE_TENSOR);
    }

    template<typename CONTAINER>
    void checkTensorMinSizes(const pTensor& tensor, CONTAINER sizes)
    {
        auto minSizeIter = sizes.begin();
        for (int dim = 0; dim < tensor->getDim(); dim++)
        {
            ASSERT_EQ(*minSizeIter++, tensor->getMinimalSizeInElements(dim))
                                    << "Tensor " << tensor->getName() << " unexpected min size in dimension " << dim;
        }
    }

protected:
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    }
};

TEST_F(DynamicShapeGraph, static_graph)
{
    ASSERT_FALSE(m_graph.isDynamicShape());
    const TSize sizes[] = {64, 64};

    std::shared_ptr<Tensor> A = std::make_shared<Tensor>(2U, sizes, syn_type_bf16);
    std::shared_ptr<Tensor> B = std::make_shared<Tensor>(2U, sizes, syn_type_bf16);
    std::shared_ptr<Tensor> C = std::make_shared<Tensor>(2U, sizes, syn_type_bf16);

    synGEMMParams params;
    pNode n  = NodeFactory::createNode({A, B, nullptr}, {C}, &params, NodeFactory::gemmNodeTypeName, "");

    ASSERT_TRUE(GraphEditor::addNode(m_graph, n)) << "node was not added to the graph";

    ASSERT_FALSE(m_graph.isDynamicShape());
}

TEST_F(DynamicShapeGraph, dynamic_graph)
{
    ASSERT_FALSE(m_graph.isDynamicShape());
    const TSize maxSizes[] = {128, 128};
    const TSize minSizes[] = {64, 64};

    std::shared_ptr<Tensor> A = std::make_shared<Tensor>(2U, maxSizes, syn_type_bf16, minSizes);
    std::shared_ptr<Tensor> B = std::make_shared<Tensor>(2U, maxSizes, syn_type_bf16, minSizes);
    std::shared_ptr<Tensor> C = std::make_shared<Tensor>(2U, maxSizes, syn_type_bf16, minSizes);

    synGEMMParams params;
    pNode n  = NodeFactory::createNode({A, B, nullptr}, {C}, &params, NodeFactory::gemmNodeTypeName, "");

    ASSERT_TRUE(GraphEditor::addNode(m_graph, n)) << "node was not added to the graph";
    ASSERT_TRUE(m_graph.isDynamicShape());

    GraphEditor::removeNode(m_graph, n);
    ASSERT_FALSE(m_graph.isDynamicShape());
}

TEST_F(DynamicShapeGraph, multi_node_dynamic)
{
    ASSERT_FALSE(m_graph.isDynamicShape());
    const TSize maxSizes[] = {128, 128};
    const TSize minSizes[] = {64, 64};

    std::shared_ptr<Tensor> A = std::make_shared<Tensor>(2U, maxSizes, syn_type_bf16);
    std::shared_ptr<Tensor> B = std::make_shared<Tensor>(2U, maxSizes, syn_type_bf16);
    std::shared_ptr<Tensor> C = std::make_shared<Tensor>(2U, maxSizes, syn_type_bf16);
    std::shared_ptr<Tensor> D = std::make_shared<Tensor>(2U, maxSizes, syn_type_bf16, minSizes);
    std::shared_ptr<Tensor> E = std::make_shared<Tensor>(2U, maxSizes, syn_type_bf16, minSizes);
    std::shared_ptr<Tensor> F = std::make_shared<Tensor>(2U, maxSizes, syn_type_bf16, minSizes);

    synGEMMParams params;
    pNode n1  = NodeFactory::createNode({A, B, nullptr}, {C}, &params, NodeFactory::gemmNodeTypeName, "");
    pNode n2  = NodeFactory::createNode({D, E, nullptr}, {F}, &params, NodeFactory::gemmNodeTypeName, "");

    ASSERT_TRUE(GraphEditor::addNode(m_graph, n1)) << "node was not added to the graph";
    ASSERT_FALSE(m_graph.isDynamicShape());

    ASSERT_TRUE(GraphEditor::addNode(m_graph, n2)) << "node was not added to the graph";
    ASSERT_TRUE(m_graph.isDynamicShape());

    GraphEditor::removeNode(m_graph, n1);
    ASSERT_TRUE(m_graph.isDynamicShape());

    GraphEditor::removeNode(m_graph, n2);
    ASSERT_FALSE(m_graph.isDynamicShape());
}

tpc_lib_api::GlueCodeReturn dummySIF(tpc_lib_api::DeviceId deviceId, const tpc_lib_api::ShapeInferenceParams *sifParams, tpc_lib_api::ShapeInferenceOutput *sifOutput)
{
    return tpc_lib_api::GLUE_SUCCESS;
}

TEST_F(GraphOptimizerTest, shape_func_repository)
{
    ShapeFuncRegistry& registry = ShapeFuncRegistry::instance();

    ASSERT_NE(registry.getSMF(SMF_DYNAMIC_EXE), nullptr);
    ASSERT_EQ(registry.getSMF(SHAPE_FUNC_MAX_ID), nullptr);

    sif_t pDummySIF = dummySIF;
    sm_function_id_t id;
    id.sm_func_index = 200;
    registry.registerSIF(id, pDummySIF, "dummySIF", GC_SIF_VERSION);
    ASSERT_EQ(registry.getSIF(id), pDummySIF);

    registry.registerSIF(SHAPE_FUNC_MAX_ID, pDummySIF, "dummySIF", GC_SIF_VERSION, LIB_ID_FIRST_GLUE_CODE_SIF);
    id.sm_tableid = LIB_ID_FIRST_GLUE_CODE_SIF;
    id.sm_funcid = SHAPE_FUNC_MAX_ID;
    ASSERT_EQ(registry.getSIF(id), pDummySIF);
    ASSERT_EQ(registry.getSIF(SHAPE_FUNC_MAX_ID, LIB_ID_FIRST_GLUE_CODE_SIF), pDummySIF);
    ASSERT_EQ(registry.getSIF(SHAPE_FUNC_MAX_ID), nullptr);
}

TEST_F(GraphOptimizerTest, shape_func_user_defeined_repository)
{
    ShapeFuncRegistry& registry = ShapeFuncRegistry::instance();

    sif_t            pDummySIF = dummySIF;
    sm_function_id_t id;
    id.sm_tableid = 18;
    id.sm_funcid  = 200;
    registry.registerSIF(id, pDummySIF, "dummySIF", GC_SIF_VERSION);
    ASSERT_EQ(registry.getSIF(id), pDummySIF);

    registry.registerSIF(SHAPE_FUNC_MAX_ID, pDummySIF, "dummySIF", GC_SIF_VERSION, LIB_ID_FIRST_GLUE_CODE_SIF);
    id.sm_tableid = LIB_ID_FIRST_GLUE_CODE_SIF;
    id.sm_funcid  = SHAPE_FUNC_MAX_ID;
    ASSERT_EQ(registry.getSIF(id), pDummySIF);
    ASSERT_EQ(registry.getSIF(SHAPE_FUNC_MAX_ID, LIB_ID_FIRST_GLUE_CODE_SIF), pDummySIF);
    ASSERT_EQ(registry.getSIF(SHAPE_FUNC_MAX_ID), nullptr);
}

TEST_F(GraphOptimizerTest, roi_dsd_type)
{
    {
        uint32_t dimCount = 4;
        TSize minSizes[] = {5,5,5,5};
        TOffset  roiOffset[] = {2, 1, 3, 4};
        TSize roiSize[]   = {3, 4, 2, 1};
        EXPECT_EQ(getRoiShapeType(dimCount, minSizes, roiOffset, roiSize), RoiShapeType::FIXED_ROI);
    }
    {
        // only one dim
        uint32_t dimCount = 4;
        TSize minSizes[] = {5,5,5,5};
        TOffset  roiOffset[] = {2, 1, 3, 5};
        TSize roiSize[]   = {3, 4, 2, 1};
        EXPECT_EQ(getRoiShapeType(dimCount, minSizes, roiOffset, roiSize), RoiShapeType::DYNAMIC_ROI);
    }
    {
        // all outside
        uint32_t dimCount = 4;
        TSize minSizes[] = {5,5,5,5};
        TOffset  roiOffset[] = {6, 3, 2, 1};
        TSize roiSize[]   = {2, 1, 2, 1};
        EXPECT_EQ(getRoiShapeType(dimCount, minSizes, roiOffset, roiSize), RoiShapeType::DYNAMIC_ROI);
    }
    {
        // all outside
        uint32_t dimCount = 4;
        TSize minSizes[] = {5,5,5,5};
        TOffset  roiOffset[] = {6, 8, 7, 6};
        TSize roiSize[]   = {5, 5, 2, 1};
        EXPECT_EQ(getRoiShapeType(dimCount, minSizes, roiOffset, roiSize), RoiShapeType::DYNAMIC_ROI);
    }
}

TEST_F(DynamicShapeGraph, tpc_infer_min_size_bn_fwd_f32)
{
    const size_t dim = 2;
    const TSize maxSizes[] = {128, 128};
    const TSize minSizes[] = {64, 64};

    pTensor data  = std::make_shared<Tensor>(2U, maxSizes, syn_type_float, minSizes);
    pTensor beta  = std::make_shared<Tensor>(1, maxSizes, syn_type_float, minSizes);
    pTensor gamma = std::make_shared<Tensor>(1, maxSizes, syn_type_float, minSizes);
    pTensor mean  = std::make_shared<Tensor>(1, maxSizes, syn_type_float, minSizes);
    pTensor var   = std::make_shared<Tensor>(1, maxSizes, syn_type_float, minSizes);

    pTensor out_data     = std::make_shared<Tensor>(2, maxSizes, syn_type_float);
    pTensor out_mean     = std::make_shared<Tensor>(1, maxSizes, syn_type_float);
    pTensor out_std      = std::make_shared<Tensor>(1, maxSizes, syn_type_float);
    pTensor out_run_mean = std::make_shared<Tensor>(1, maxSizes, syn_type_float);
    pTensor out_run_var  = std::make_shared<Tensor>(1, maxSizes, syn_type_float);

    pNode n = NodeFactory::createNode({data, beta, gamma, mean, var},
                                      {out_data, out_mean, out_std, out_run_mean, out_run_var},
                                      nullptr, "batch_norm_fwd_f32", "");

    ASSERT_FALSE(out_data->isDynamicShape());
    ASSERT_FALSE(out_mean->isDynamicShape());
    ASSERT_FALSE(out_std->isDynamicShape());
    ASSERT_FALSE(out_run_mean->isDynamicShape());
    ASSERT_FALSE(out_run_var->isDynamicShape());

    n->inferOutputsShape(synDeviceGaudi, /*inferMax*/ false);

    for (int i = 0; i < dim; i++)
    {
        ASSERT_EQ(out_data->getSizeInElements(i), maxSizes[i]);
        ASSERT_EQ(out_data->getMinimalSizeInElements(i), minSizes[i]);
    }
    ASSERT_TRUE(out_data->isDynamicShape());
    ASSERT_EQ(out_mean->getSizeInElements(0), maxSizes[0]);
    ASSERT_EQ(out_mean->getMinimalSizeInElements(0), minSizes[0]);
    ASSERT_TRUE(out_mean->isDynamicShape());
    ASSERT_EQ(out_std->getSizeInElements(0), maxSizes[0]);
    ASSERT_EQ(out_std->getMinimalSizeInElements(0), minSizes[0]);
    ASSERT_TRUE(out_std->isDynamicShape());
    ASSERT_EQ(out_run_mean->getSizeInElements(0), maxSizes[0]);
    ASSERT_EQ(out_run_mean->getMinimalSizeInElements(0), minSizes[0]);
    ASSERT_TRUE(out_run_mean->isDynamicShape());
    ASSERT_EQ(out_run_var->getSizeInElements(0), maxSizes[0]);
    ASSERT_EQ(out_run_var->getMinimalSizeInElements(0), minSizes[0]);
    ASSERT_TRUE(out_run_var->isDynamicShape());
}

TEST_F(DynamicShapeGraph, tpc_infer_min_size_bn_bwd_f32)
{
    const size_t dim = 2;
    const TSize maxSizes[] = {128, 128};
    const TSize minSizes[] = {64, 64};

    pTensor data    = std::make_shared<Tensor>(2U, maxSizes, syn_type_float, minSizes);
    pTensor grad_in = std::make_shared<Tensor>(2U, maxSizes, syn_type_float, minSizes);
    pTensor mean    = std::make_shared<Tensor>(1, maxSizes, syn_type_float, minSizes);
    pTensor lstd    = std::make_shared<Tensor>(1, maxSizes, syn_type_float, minSizes);
    pTensor gamma   = std::make_shared<Tensor>(1, maxSizes, syn_type_float, minSizes);

    pTensor out_grad  = std::make_shared<Tensor>(2, maxSizes, syn_type_float);
    pTensor out_beta  = std::make_shared<Tensor>(1, maxSizes, syn_type_float);
    pTensor out_gamma = std::make_shared<Tensor>(1, maxSizes, syn_type_float);

    pNode n = NodeFactory::createNode({data, grad_in, mean, lstd, gamma},
                                      {out_grad, out_beta, out_gamma},
                                      nullptr, "batch_norm_bwd_f32", "");

    ASSERT_FALSE(out_grad->isDynamicShape());
    ASSERT_FALSE(out_beta->isDynamicShape());
    ASSERT_FALSE(out_gamma->isDynamicShape());

    n->inferOutputsShape(synDeviceGaudi, /*inferMax*/ false);

    for (int i = 0; i < dim; i++)
    {
        ASSERT_EQ(out_grad->getSizeInElements(i), maxSizes[i]);
        ASSERT_EQ(out_grad->getMinimalSizeInElements(i), minSizes[i]);
    }
    ASSERT_TRUE(out_grad->isDynamicShape());

    ASSERT_EQ(out_beta->getSizeInElements(0), maxSizes[0]);
    ASSERT_EQ(out_beta->getMinimalSizeInElements(0), minSizes[0]);
    ASSERT_TRUE(out_beta->isDynamicShape());
    ASSERT_EQ(out_gamma->getSizeInElements(0), maxSizes[0]);
    ASSERT_EQ(out_gamma->getMinimalSizeInElements(0), minSizes[0]);
    ASSERT_TRUE(out_gamma->isDynamicShape());
}

class InternalTensorShapeInferenceTest : public DynamicShapeGraph
{
public:

    // Using BN fwd because it has a SIF ready while this test is written
    // If the ofm is not supplied, adding a non-persistent tensor with maxSize from IFM (no min size)
    // Returns the ofm if supplied or the generated new tensor mentioned above.
    pTensor addBNFwd(pTensor ifm, pTensor ofm = nullptr)
    {

        std::vector<TSize> channelsMaxSize = {ifm->getSizeInElements(0)};
        std::vector<TSize> channelsMinSize = {ifm->getMinimalSizeInElements(0)};

        if (ofm == nullptr)
        {
            const auto& ifmSizes = ifm->getAllSizesInElements();
            std::vector<TSize> ofmMaxSizes(ifmSizes.begin(), std::next(ifmSizes.begin(), ifm->getDim()));
            ofm = createTensor(ofmMaxSizes);
        }

        TensorVector inputs;
        inputs.reserve(5);
        inputs.push_back(ifm);
        for (int i = 0; i < 4; i++)
        {
            inputs.push_back(createPersistentTensor(channelsMaxSize, channelsMinSize));
        }

        TensorVector outputs;
        outputs.reserve(5);
        outputs.push_back(ofm);
        for (int i = 0; i < 4; i++)
        {
            outputs.push_back(createTensor(channelsMaxSize));
        }

        ns_BatchNormKernel::Params params{};
        pNode bnFwd = NodeFactory::createNode(inputs, outputs, &params, "batch_norm_fwd_f32", "BN");
        GraphEditor::addNode(m_graph, bnFwd);

        return ofm;
    }

};

TEST_F(InternalTensorShapeInferenceTest, internal_tensors_pass_should_infer_tensor_min_size)
{
    std::vector<TSize> inMaxSizes  = {128, 128};
    std::vector<TSize> inMinSizes  = { 64,  64};

    // Create the following nwk:
    //
    // t1 (Persistent) -> BN -> t2 (WS) -> BN -> t3 (WS)
    //                              '----> BN -> t4 (Persistent) -> BN -> t5 (Persistent)
    pTensor t1 = createPersistentTensor(inMaxSizes, inMinSizes);
    pTensor t2 = addBNFwd(t1);
    pTensor t3 = addBNFwd(t2);
    pTensor t4 = createTensor(inMaxSizes);
    addBNFwd(t2, t4);
    pTensor t5 = createTensor(inMaxSizes);
    addBNFwd(t4, t5);

    ASSERT_TRUE(t1->isDynamicShape());
    ASSERT_FALSE(t2->isDynamicShape());
    ASSERT_FALSE(t3->isDynamicShape());
    ASSERT_FALSE(t4->isDynamicShape());
    ASSERT_FALSE(t5->isDynamicShape());

    ASSERT_TRUE(internalTensorsDynamicShape(m_graph));

    ASSERT_TRUE(t1->isDynamicShape());
    checkTensorMinSizes(t1, inMinSizes);
    ASSERT_TRUE(t2->isDynamicShape());
    checkTensorMinSizes(t2, inMinSizes);
    ASSERT_TRUE(t3->isDynamicShape());
    checkTensorMinSizes(t3, inMinSizes);
    ASSERT_TRUE(t4->isDynamicShape());
    checkTensorMinSizes(t4, inMinSizes);
    ASSERT_TRUE(t5->isDynamicShape());
    checkTensorMinSizes(t5, inMinSizes);
}

TEST_F(DynamicShapeGraph, node_should_be_able_to_update_other_nodes_output_shape)
{
    // Given the graph:
    // [ShapeTensor]---> (Memset0) ---> [Zeros0]
    //
    //                   (Memset1) ---> [Zeros1]
    //
    //                   (Memset2) ---> [Zeros2]
    //
    // with Zeros1/2 set to be updated by Memset0,
    // When running the internal SIF pass, it is expected that the min size of Zeros1/2 will be set to the same min
    // size as Zeros0

    const std::vector<TSize> maxSizes = {256, 32, 32, 64};
    const std::vector<TSize> minSizes = {256, 21, 21, 33};
    TensorPtr shapeTensor = createShapeTensor(maxSizes, minSizes);
    TensorPtr zeros0 = createTensor(maxSizes);
    TensorPtr zeros1 = createTensor(maxSizes);
    TensorPtr zeros2 = createTensor(maxSizes);

    NodePtr memset0 = NodeFactory::createNode({shapeTensor}, {zeros0},  nullptr, NodeFactory::dmaMemsetNodeTypeName, "memset0");
    NodePtr memset1 = NodeFactory::createNode({}, {zeros1},  nullptr, NodeFactory::dmaMemsetNodeTypeName, "memset1");
    NodePtr memset2 = NodeFactory::createNode({}, {zeros2},  nullptr, NodeFactory::dmaMemsetNodeTypeName, "memset2");

    GraphEditor::addNode(m_graph, memset0);
    GraphEditor::addNode(m_graph, memset1);
    GraphEditor::addNode(m_graph, memset2);

    checkTensorMinSizes(zeros0, maxSizes);
    checkTensorMinSizes(zeros1, maxSizes);
    checkTensorMinSizes(zeros2, maxSizes);

    memset0->getShapeNode()->addPostSifUpdate(zeros0, zeros1);
    memset0->getShapeNode()->addPostSifUpdate(zeros0, zeros2);

    internalTensorsDynamicShape(m_graph);

    checkTensorMinSizes(zeros0, minSizes);
    checkTensorMinSizes(zeros1, minSizes);
    checkTensorMinSizes(zeros2, minSizes);

}

TEST_F(DynamicShapeGraph, reduction_memset_shapes_should_be_linked_to_non_internal_memset_node)
{
    // Given the following graph:
    //
    // [t1] -----> (memcpy ) --> [t2] ------+
    //                                      v
    //             (memset1) --> [t3] ->(Reduction) -> [t5]
    //                                      ^
    // [shape] --> (memset2) --> [t4] ------+
    //
    // t3 shape is expected to be linked to memset2 or memcopy and the shape inference should propogate the min size
    // throughout the graph.

    std::vector<TSize> maxSizes = {64, 256};
    std::vector<TSize> minSizes = {64, 128};

    TensorPtr t1    = createPersistentTensor(maxSizes, minSizes);
    TensorPtr shape = createShapeTensor(maxSizes, minSizes);

    TensorPtr t2    = createTensor(maxSizes);
    t2->setTensorInSram();
    TensorPtr t3 = createTensor(maxSizes);
    t3->setTensorInSram();
    TensorPtr t4 = createTensor(maxSizes);
    t4->setTensorInSram();
    TensorPtr t5 = createTensor(maxSizes);
    t5->setTensorInSram();
    TensorPtr      t6          = createPersistentTensor(minSizes);
    TensorPtr      t7          = createTensor(minSizes);
    synSplitParams splitParams = {1};

    NodePtr memcpy    = NodeFactory::createNode({t1}, {t2}, nullptr, NodeFactory::memcpyNodeTypeName, "memcpy");
    NodePtr memset1   = NodeFactory::createNode({}, {t3}, nullptr, NodeFactory::memsetNodeTypeName, "memset1");
    NodePtr memset2   = NodeFactory::createNode({shape}, {t4}, nullptr, NodeFactory::memsetNodeTypeName, "memset2");
    NodePtr reduction = NodeFactory::createNode({t2, t3, t4}, {t5}, nullptr, NodeFactory::reductionNodeTypeName, "reduction");
    NodePtr split     = NodeFactory::createNode({t5}, {t6, t7}, &splitParams, "split", "split");

    ASSERT_TRUE(GraphEditor::addNode(m_graph, memcpy));
    ASSERT_TRUE(GraphEditor::addNode(m_graph, memset1));
    ASSERT_TRUE(GraphEditor::addNode(m_graph, memset2));
    ASSERT_TRUE(GraphEditor::addNode(m_graph, reduction));
    ASSERT_TRUE(GraphEditor::addNode(m_graph, split));

    ASSERT_TRUE(m_graph.compile()) << "Compilation failed";

    TensorVector linkedOutputs;
    NodeVector   nodesUpdatingLinkedOutput;
    for (const auto& node : m_graph.getNodes())
    {
        if (node->getOutput(0)->getDim() == minSizes.size())
        {
            checkTensorMinSizes(node->getOutput(0), minSizes);
        }
        for (const auto& srcDstTensors : node->getShapeNode()->getPostSifUpdates())
        {
            linkedOutputs.push_back(srcDstTensors.second);
            nodesUpdatingLinkedOutput.push_back(node);
        }
    }

    ASSERT_EQ(linkedOutputs.size(), 1);
    ASSERT_EQ(linkedOutputs.front(), t3);

    // The output need to be linked to either the memset with shape tensor or the memcpy.
    // Since the nodes may be replaced, it's not possible to check 'memset2' or 'memcpy' directly,
    // but since they are the only nodes in the graph with a single input, checking that should be enough.
    ASSERT_EQ(nodesUpdatingLinkedOutput.size(), 1);
    ASSERT_TRUE(nodesUpdatingLinkedOutput.front()->getNumInputs() == 1);
}

TEST_F(DynamicShapeGraph, remove_zero_sized_tensor_emptyGraphCase)
{
    TSize g_0_output_dynamic_max_sizes[] = {0};
    TSize g_0_output_dynamic_min_sizes[] = {0};

    pTensor input =
        std::make_shared<Tensor>(1, g_0_output_dynamic_max_sizes, syn_type_float, g_0_output_dynamic_min_sizes);
    pTensor output =
        std::make_shared<Tensor>(1, g_0_output_dynamic_max_sizes, syn_type_float, g_0_output_dynamic_min_sizes);
    input->setAsStaticParam();
    pNode identity  = NodeFactory::createNode({input}, {output}, nullptr, NodeFactory::identityNodeTypeName, "identity");
    ASSERT_TRUE(GraphEditor::addNode(m_graph, identity));
    m_graph.compile();
    ASSERT_TRUE(m_graph.getExeSortedNodes().size() == 0);
}