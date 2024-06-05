#include "graph_optimizer_test.h"
#include "slicer/node_dcore_rois_setter.h"
#include "synapse_common_types.h"
#include "node_factory.h"
#include "types.h"

using namespace gc::layered_brain;
using namespace gc::access_pattern;

class NodeDcoreROIsSetterTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<std::tuple<unsigned,  // num dcores
                                                  unsigned,  // perforation dim
                                                  TSize,     // perforation dim size
                                                  TSize,     // perforation dim granularity
                                                  bool>>     // is perforation expected
{
protected:
    NodeDcoreROIsSetterTest()
    {
        std::tie(m_numDcores, m_perforationDim, m_perforationDimSize, m_granularity, m_isPerforationExpected) =
            GetParam();
    }

    void createNode()
    {
        m_node = NodeFactory::createNode({std::make_shared<Tensor>(syn_type_float)},
                                         {std::make_shared<Tensor>(syn_type_float)},
                                         nullptr,
                                         TPCNode::NOP_KERNEL_NAME,
                                         "TPC");
        ASSERT_TRUE(m_node);
        NodeTile nodeROI(NodeTile::Geometry {10, 20, 30, 40}, NodeTile::Offset {40, 30, 20, 10});
        ASSERT_LT(m_perforationDim, nodeROI.geometry.size());
        nodeROI.geometry[m_perforationDim]   = m_perforationDimSize;
        m_node->getNodeAnnotation().sliceROI = nodeROI;
    }

    std::vector<TSize> splitToDcores(TSize size, TSize granularity) const
    {
        std::vector<TSize> sizes(m_numDcores, 0);
        while (size >= granularity)
        {
            for (auto dcore = 0; dcore < m_numDcores; dcore++)
            {
                sizes[dcore] += granularity;
                size -= granularity;
                if (size < granularity) break;
            }
        }
        sizes[m_numDcores - 1] += size;  // The last DCORE gets the remainder
        return sizes;
    }

    void validateSplit() const
    {
        const auto& dcoreROIs = m_node->getNodeAnnotation().m_dcoreROIs;
        if (m_isPerforationExpected)
        {
            const auto& nodeRoi = m_node->getNodeAnnotation().sliceROI;
            ASSERT_TRUE(nodeRoi.has_value());
            const auto& expectedDcoreSplit = splitToDcores(nodeRoi->geometry.at(m_perforationDim), m_granularity);
            ASSERT_EQ(std::accumulate(expectedDcoreSplit.begin(), expectedDcoreSplit.end(), 0UL),
                      nodeRoi->geometry.at(m_perforationDim));

            TSize totalPerforatedDimSize = 0;

            ASSERT_EQ(dcoreROIs.size(), m_numDcores);
            for (auto dim = 0; dim < nodeRoi->geometry.size(); dim++)
            {
                for (auto dcore = 0; dcore < m_numDcores; dcore++)
                {
                    if (dim == m_perforationDim)
                    {
                        ASSERT_EQ(dcoreROIs.at(dcore).size[dim], expectedDcoreSplit[dcore]);
                        ASSERT_EQ(dcoreROIs.at(dcore).baseOffset[dim],
                                  nodeRoi->offset.at(dim) + totalPerforatedDimSize);
                        totalPerforatedDimSize += dcoreROIs.at(dcore).size[dim];
                    }
                    else
                    {
                        ASSERT_EQ(dcoreROIs.at(dcore).size[dim], nodeRoi->geometry.at(dim));
                        ASSERT_EQ(dcoreROIs.at(dcore).baseOffset[dim], nodeRoi->offset.at(dim));
                    }
                }
            }
            ASSERT_EQ(totalPerforatedDimSize, nodeRoi->geometry.at(m_perforationDim));
        }
        else
        {
            ASSERT_TRUE(dcoreROIs.empty());
        }
    }

    void test()
    {
        createNode();

        NodeDcoreROIsSetter(m_node, m_numDcores).splitToDcoreROIs(m_perforationDim, m_granularity, 0);

        validateSplit();
    }

    unsigned m_numDcores;
    unsigned m_perforationDim;
    TSize    m_perforationDimSize;
    TSize    m_granularity;
    bool     m_isPerforationExpected;
    NodePtr  m_node;
};

TEST_P(NodeDcoreROIsSetterTest, test_dcore_roi_split)
{
    test();
}

INSTANTIATE_TEST_SUITE_P(
    split_to_dcore_rois_last_dcores_no_work,
    NodeDcoreROIsSetterTest,  // num dcores, perforation dim, perforation dim size, granularity, is perforation expected
    ::testing::Values(std::make_tuple(4, 3, 3, 1, true),
                      std::make_tuple(4, 1, 128, 64, true),
                      std::make_tuple(4, 0, 128, 96, true),
                      std::make_tuple(99, 0, 100, 2, true),
                      std::make_tuple(7, 1, 30, 5, true)));

INSTANTIATE_TEST_SUITE_P(
    split_to_dcore_rois_last_roi_smaller_than_granularity,
    NodeDcoreROIsSetterTest,  // num dcores, perforation dim, perforation dim size, granularity, is perforation expected
    ::testing::Values(std::make_tuple(4, 2, 40, 11, true),  // size per DCORE: 11,11,11,7
                      std::make_tuple(7, 1, 33, 5, true),   // size per DCORE: 5,5,5,5,5,5,3
                      std::make_tuple(2, 0, 5, 4, true)));  // size per DCORE: 4,1

INSTANTIATE_TEST_SUITE_P(split_to_dcore_rois_no_granularity_constraints,
                         NodeDcoreROIsSetterTest,
                         ::testing::Combine(::testing::Values(2, 3, 4),                       // num dcores
                                            ::testing::Values(0, 2),                          // perforation dim
                                            ::testing::Values(4, 128, 256, 6, 348, 349, 13),  // perforation dim size
                                            ::testing::Values(1),                             // granularity
                                            ::testing::Values(true)));                        // is perforation expected

INSTANTIATE_TEST_SUITE_P(split_to_dcore_rois_with_granularity_constraints,
                         NodeDcoreROIsSetterTest,
                         ::testing::Combine(::testing::Values(2, 3, 4),                     // num dcores
                                            ::testing::Values(1, 3),                        // perforation dim
                                            ::testing::Values(128, 256, 66, 348, 349, 33),  // perforation dim size
                                            ::testing::Values(4, 7, 3),                     // granularity
                                            ::testing::Values(true)));                      // is perforation expected