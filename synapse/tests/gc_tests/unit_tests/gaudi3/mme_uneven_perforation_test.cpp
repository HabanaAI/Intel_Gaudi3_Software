#include "gaudi3_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "hal_reader/hal_reader.h"
#include "slicer/node_dcore_rois_setter.h"
#include "synapse_common_types.h"
#include "node_factory.h"
#include "synapse_common_types.hpp"
#include "types.h"
#include "platform/gaudi3/graph_compiler/passes.h"
#include "compilation_hal_reader.h"

using namespace gc::layered_brain;
using namespace gc::access_pattern;

class MmeUnevenPerforationTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<std::tuple<unsigned,  // num dcores
                                                  TSize,     // perforation dim size
                                                  TSize,     // perforation dim granularity
                                                  bool>>     // conv / bgemm
{
protected:
    MmeUnevenPerforationTest() : m_halSetter(&m_graph)
    {
        std::tie(m_numDcores, m_perforationDimSize, m_granularity, m_isConv) = GetParam();
    }

    TensorPtr createTensor(const std::vector<TSize>& shape) const
    {
        return std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_bf16);
    }

    void setNodeAnnotations()
    {
        m_node->getNodeAnnotation().sliceROI = m_node->getNodeAccessPattern()->getNodeResolution();

        CacheMetaData cacheMetaData;
        cacheMetaData.cacheDirective = CacheDirective::HomeAllocate;
        m_node->getNodeAnnotation().inputsCacheMetaData.resize(2);
        m_node->getNodeAnnotation().outputsCacheMetaData.resize(1);
        m_node->getNodeAnnotation().inputsCacheMetaData[0]  = cacheMetaData;
        m_node->getNodeAnnotation().inputsCacheMetaData[1]  = cacheMetaData;
        m_node->getNodeAnnotation().outputsCacheMetaData[0] = cacheMetaData;
    }

    void createNode()
    {
        if (m_isConv)
        {
            createConvNode();
        }
        else
        {
            createBgemmNode();
        }

        setNodeAnnotations();
        ASSERT_TRUE(GraphEditor::addNode(m_graph, m_node));
    }

    void createConvNode()
    {
        synConvolutionParams params {};
        std::vector<TSize>   aSizes   = {256, 56, 56, m_perforationDimSize};
        std::vector<TSize>   bSizes   = {128, 256, 1, 1};
        std::vector<TSize>   outSizes = {128, 56, 56, m_perforationDimSize};
        TensorPtr            a        = createTensor(aSizes);
        TensorPtr            b        = createTensor(bSizes);
        TensorPtr            out      = createTensor(outSizes);
        m_node = NodeFactory::createNode({a, b}, {out}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    }

    void createBgemmNode()
    {
        synGEMMParams      params {};
        std::vector<TSize> aSizes   = {512, 512, m_perforationDimSize};
        std::vector<TSize> bSizes   = {512, 512, m_perforationDimSize};
        std::vector<TSize> outSizes = {512, 512, m_perforationDimSize};
        TensorPtr          a        = createTensor(aSizes);
        TensorPtr          b        = createTensor(bSizes);
        TensorPtr          out      = createTensor(outSizes);
        m_node = NodeFactory::createNode({a, b}, {out}, &params, NodeFactory::batchGemmNodeTypeName, "bgemm");
    }

    void test()
    {
        createNode();

        auto perforationDim =  // Perforate on batch dim
            m_node->getNodeAccessPattern()->getIndexSpaceDim(m_node->getOutput(0), m_node->getOutput(0)->getDim() - 1);
        NodeDcoreROIsSetter(m_node, m_numDcores).splitToDcoreROIs(perforationDim, m_granularity, 0);
        ASSERT_TRUE(m_node->getNodeAnnotation().perforationDim.has_value());

        ASSERT_TRUE(validateNodesLayout(m_graph));
        ASSERT_TRUE(generateROIs(m_graph));
        ASSERT_TRUE(generateMmeDescriptors(m_graph));

        const MmeDescriptorGenerator& descGen    = m_graph.getMmeNodeDescriptorGenerator(m_node);
        Gaudi3SignalingInfo           info;
        auto                          expectedNumDescs = descGen.getMmeNr();
        for (const auto& activation : descGen.getMmeActivations())
        {
            ASSERT_FALSE(activation.descriptors.empty());
            ASSERT_EQ(activation.descriptors.size(), expectedNumDescs);
            auto expectedNumSignals = info.countSignals(&activation.descriptors.front());
            for (const auto& desc : activation.descriptors)
            {
                ASSERT_EQ(info.countSignals(&desc), expectedNumSignals);
            }
        }
    }

    Gaudi3Graph                m_graph;
    CompilationHalReaderSetter m_halSetter;
    unsigned                   m_numDcores;
    TSize                      m_perforationDimSize;
    TSize                      m_granularity;
    bool                       m_isConv;
    NodePtr                    m_node;
};

TEST_P(MmeUnevenPerforationTest, test_uneven_perforation)
{
    test();
}

INSTANTIATE_TEST_SUITE_P(split_to_dcore_rois_no_granularity_constraints,
                         MmeUnevenPerforationTest,
                         ::testing::Combine(::testing::Values(2, 3, 4),                          // num dcores
                                            ::testing::Values(1923, 59, 697, 7, 3407, 349, 13),  // perforation dim size
                                            ::testing::Values(1),                                // granularity
                                            ::testing::Values(true, false)));                    // conv/bgemm

INSTANTIATE_TEST_SUITE_P(
    split_to_dcore_rois_with_granularity_constraints,
    MmeUnevenPerforationTest,
    ::testing::Combine(::testing::Values(2, 3, 4),                             // num dcores
                       ::testing::Values(2048, 3999, 3000, 1217, 1245, 1001),  // perforation dim size
                       ::testing::Values(199, 320, 15, 299, 30, 110, 7),       // granularity
                       ::testing::Values(true, false)));                       // conv/bgemm

INSTANTIATE_TEST_SUITE_P(split_to_dcore_rois_zero_work_dcore,
                         MmeUnevenPerforationTest,
                         ::testing::Combine(::testing::Values(4),                      // num dcores
                                            ::testing::Values(2048),                   // perforation dim size
                                            ::testing::Values(2048, 1024, 1200, 720),  // granularity
                                            ::testing::Values(true, false)));          // conv/bgemm