#include "tensor.h"
#include "node.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include "graph_optimizer_test.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "types.h"
#include "graph_compiler/passes/generate_work_distribution.h"

namespace gaudi3
{
class WDTest : public GraphOptimizerTest
{
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(Gaudi3HalReader::instance());
    }
};

TEST_F(WDTest, monolithic_test)
{
    const unsigned tensor_dim = 1;
    const TSize    size       = 1;
    Gaudi3Graph    g;
    TensorPtr      i = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr      o = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorVector   inputTensors;
    TensorVector   outputTensors;

    inputTensors.push_back(i);
    outputTensors.push_back(o);
    TPCNode             tpcNode(inputTensors, outputTensors, "test node");
    NodeROI*            tpcNodeRoi = new NodeROI();
    std::list<NodeROI>* nodeRois   = new std::list<NodeROI>();
    for (unsigned dim = 0; dim < MAX_DIMENSIONS_NUM; dim++)
    {
        tpcNodeRoi->baseOffset[dim] = 0 + dim * 0x1000;
        tpcNodeRoi->size[dim]       = 8;
        tpcNode.getNodeAnnotation().tpcSplitDims.push_back(MAX_DIMENSIONS_NUM - dim - 1);
    }

    nodeRois->push_back(*tpcNodeRoi);
    std::array<unsigned, MAX_NUM_DCORES> tpcShuffleIndex {};
    bool                                 previousTpcNodeLocalityMode = false;
    workDistributionManager::tpcWorkDistribution(tpcNode,
                                                 *nodeRois,
                                                 g.getNumTpcEng(),
                                                 tpcShuffleIndex,
                                                 previousTpcNodeLocalityMode,
                                                 false,
                                                 false,
                                                 MAX_NUM_DCORES);

    NodeROI   resultNodeRoi                     = nodeRois->front();
    TSize     resultBoxSize[MAX_DIMENSIONS_NUM] = {8, 8, 8, 1, 1};
    TpcWdCtx& tpcWdCtx                          = resultNodeRoi.tpcWdCtx.front();
    for (unsigned dim = 0; dim < MAX_DIMENSIONS_NUM; dim++)
    {
        ASSERT_EQ(tpcWdCtx.baseCord[dim], 0 + dim * 0x1000);
        ASSERT_EQ(tpcWdCtx.gridSize[dim], 8);
        ASSERT_EQ(tpcWdCtx.boxSize[dim], resultBoxSize[dim]);
    }

    delete tpcNodeRoi;
    delete nodeRois;
}

TEST_F(WDTest, locality_basic_test)
{
    const unsigned tensor_dim = 1;
    const TSize    size       = 1;
    Gaudi3Graph    g;
    TensorPtr      i = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr      o = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorVector   inputTensors;
    TensorVector   outputTensors;
    TSize          resultGridSize[MAX_DIMENSIONS_NUM] = {4, 8, 8, 4, 12};

    inputTensors.push_back(i);
    outputTensors.push_back(o);
    TPCNode             tpcNode(inputTensors, outputTensors, "test node");
    tpcNode.getNodeAnnotation().perforation = LitePerforationHints {0, 1, true};
    NodeROI*            tpcNodeRoi = new NodeROI();
    std::list<NodeROI>* nodeRois   = new std::list<NodeROI>();
    for (unsigned dcore = 0; dcore < MAX_NUM_DCORES; dcore++)
    {
        DcoreROI dcoreRoi;
        for (unsigned dim = 0; dim < MAX_DIMENSIONS_NUM; dim++)
        {
            dcoreRoi.baseOffset[dim] = 0 + dcore * 0x1000 + dim * (0x1000 / MAX_NUM_DCORES);
            // split to Dcore Rois on SCD
            dcoreRoi.size[dim] =
                (dim == MAX_DIMENSIONS_NUM - 1) ? resultGridSize[dim] / MAX_NUM_DCORES : resultGridSize[dim];
            if (dcore == 0)  // fill once only
            {
                tpcNodeRoi->baseOffset[dim] = 0 + dim * 0x1000;
                tpcNode.getNodeAnnotation().tpcSplitDims.push_back(MAX_DIMENSIONS_NUM - dim - 1);
                tpcNodeRoi->size[dim] = resultGridSize[dim];
            }
        }
        tpcNodeRoi->dcoreROIs.push_back(dcoreRoi);
    }

    nodeRois->push_back(*tpcNodeRoi);
    std::array<unsigned, MAX_NUM_DCORES> tpcShuffleIndex {};
    bool                                 previousTpcNodeLocalityMode = false;
    workDistributionManager::tpcWorkDistribution(tpcNode,
                                                 *nodeRois,
                                                 g.getNumTpcEng(),
                                                 tpcShuffleIndex,
                                                 previousTpcNodeLocalityMode,
                                                 false,
                                                 false,
                                                 MAX_NUM_DCORES);

    NodeROI resultNodeRoi                     = nodeRois->front();
    TSize   resultBoxSize[MAX_DIMENSIONS_NUM] = {4, 8, 2, 1, 3};
    auto    tpcWdCtxs                         = resultNodeRoi.tpcWdCtx;
    for (unsigned dcore = 0; dcore < MAX_NUM_DCORES; dcore++)
    {
        for (unsigned dim = 0; dim < MAX_DIMENSIONS_NUM; dim++)
        {
            ASSERT_EQ(tpcWdCtxs[dcore].baseCord[dim], 0 + dcore * 0x1000 + dim * (0x1000 / MAX_NUM_DCORES));
            ASSERT_EQ(tpcWdCtxs[dcore].boxSize[dim], resultBoxSize[dim]);
            if (dim == MAX_DIMENSIONS_NUM - 1)
            {
                ASSERT_EQ(tpcWdCtxs[dcore].gridSize[dim], resultGridSize[dim] / MAX_NUM_DCORES);
            }
            else
            {
                ASSERT_EQ(tpcWdCtxs[dcore].gridSize[dim], resultGridSize[dim]);
            }
        }
    }

    delete tpcNodeRoi;
    delete nodeRois;
}

TEST_F(WDTest, locality_empty_dcore_test)
{
    const unsigned tensor_dim = 1;
    const TSize    size       = 1;
    Gaudi3Graph    g;
    TensorPtr      i = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr      o = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorVector   inputTensors;
    TensorVector   outputTensors;
    TSize          resultGridSize[MAX_DIMENSIONS_NUM] = {4, 8, 8, 4, 12};
    TSize          RoiSize[MAX_DIMENSIONS_NUM]        = {4, 8, 8, 4, 6};  // only 2 docres full
    inputTensors.push_back(i);
    outputTensors.push_back(o);
    TPCNode             tpcNode(inputTensors, outputTensors, "test node");
    tpcNode.getNodeAnnotation().perforation = LitePerforationHints {0, 1, true};
    NodeROI*            tpcNodeRoi = new NodeROI();
    std::list<NodeROI>* nodeRois   = new std::list<NodeROI>();
    for (unsigned dcore = 0; dcore < MAX_NUM_DCORES; dcore++)
    {
        DcoreROI dcoreRoi;
        for (unsigned dim = 0; dim < MAX_DIMENSIONS_NUM; dim++)
        {
            if (dcore == 0 || dcore == 2)
            {
                dcoreRoi.baseOffset[dim] = 0;
                dcoreRoi.size[dim]       = 0;
            }
            else
            {
                dcoreRoi.baseOffset[dim] = 0 + dcore * 0x1000 + dim * (0x1000 / MAX_NUM_DCORES);
                // split to Dcore Rois on SCD
                dcoreRoi.size[dim] =
                    (dim == MAX_DIMENSIONS_NUM - 1) ? resultGridSize[dim] / MAX_NUM_DCORES : resultGridSize[dim];
            }
            if (dcore == 0)  // fill once only
            {
                tpcNodeRoi->size[dim] = RoiSize[dim];
                tpcNodeRoi->baseOffset[dim] = 0 + dim * 0x1000;
                tpcNode.getNodeAnnotation().tpcSplitDims.push_back(MAX_DIMENSIONS_NUM - dim - 1);
            }
        }
        tpcNodeRoi->dcoreROIs.push_back(dcoreRoi);
    }

    nodeRois->push_back(*tpcNodeRoi);
    std::array<unsigned, MAX_NUM_DCORES> tpcShuffleIndex {};
    bool                                 previousTpcNodeLocalityMode = false;
    workDistributionManager::tpcWorkDistribution(tpcNode,
                                                 *nodeRois,
                                                 g.getNumTpcEng(),
                                                 tpcShuffleIndex,
                                                 previousTpcNodeLocalityMode,
                                                 false,
                                                 false,
                                                 MAX_NUM_DCORES);

    NodeROI resultNodeRoi                     = nodeRois->front();
    TSize   resultBoxSize[MAX_DIMENSIONS_NUM] = {4, 8, 2, 1, 3};
    auto    tpcWdCtxs                         = resultNodeRoi.tpcWdCtx;
    for (unsigned dcore = 0; dcore < MAX_NUM_DCORES; dcore++)
    {
        for (unsigned dim = 0; dim < MAX_DIMENSIONS_NUM; dim++)
        {
            if (dcore == 0 || dcore == 2)
            {
                ASSERT_EQ(tpcWdCtxs[dcore].baseCord[dim], 0);
                ASSERT_EQ(tpcWdCtxs[dcore].boxSize[dim], 0);
                ASSERT_EQ(tpcWdCtxs[dcore].gridSize[dim], 0);
            }
            else
            {
                ASSERT_EQ(tpcWdCtxs[dcore].baseCord[dim], 0 + dcore * 0x1000 + dim * (0x1000 / MAX_NUM_DCORES));
                ASSERT_EQ(tpcWdCtxs[dcore].boxSize[dim], resultBoxSize[dim]);
                if (dim == MAX_DIMENSIONS_NUM - 1)
                {
                    ASSERT_EQ(tpcWdCtxs[dcore].gridSize[dim], resultGridSize[dim] / MAX_NUM_DCORES);
                }
                else
                {
                    ASSERT_EQ(tpcWdCtxs[dcore].gridSize[dim], resultGridSize[dim]);
                }
            }
        }
    }

    delete tpcNodeRoi;
    delete nodeRois;
}

TEST_F(WDTest, locality_one_dcore_test)
{
    // small work is sent to one dcore only
    const unsigned tensor_dim = 1;
    const TSize    size       = 1;
    Gaudi3Graph    g;
    TensorPtr      i = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr      o = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorVector   inputTensors;
    TensorVector   outputTensors;
    TSize          resultGridSize[MAX_DIMENSIONS_NUM] = {1, 1, 1, 1, 1};

    inputTensors.push_back(i);
    outputTensors.push_back(o);
    TPCNode             tpcNode(inputTensors, outputTensors, "test node");
    tpcNode.getNodeAnnotation().perforation = LitePerforationHints {0, 1, true};
    NodeROI*            tpcNodeRoi = new NodeROI();
    std::list<NodeROI>* nodeRois   = new std::list<NodeROI>();
    for (unsigned dcore = 0; dcore < MAX_NUM_DCORES; dcore++)
    {
        DcoreROI dcoreRoi;
        for (unsigned dim = 0; dim < MAX_DIMENSIONS_NUM; dim++)
        {
            if (dcore == 0)
            {
                // fill NodeRoi
                tpcNodeRoi->baseOffset[dim] = 0x100;
                tpcNodeRoi->size[dim]       = resultGridSize[dim];
                tpcNode.getNodeAnnotation().tpcSplitDims.push_back(MAX_DIMENSIONS_NUM - dim - 1);
                // fill DcoreRoi
                dcoreRoi.baseOffset[dim] = 0x100;
                // split to Dcore Rois on SCD
                dcoreRoi.size[dim] = resultGridSize[dim];
            }
            else
            {
                dcoreRoi.baseOffset[dim] = 0;
                dcoreRoi.size[dim]       = 0;
            }
        }
        tpcNodeRoi->dcoreROIs.push_back(dcoreRoi);
    }

    nodeRois->push_back(*tpcNodeRoi);
    std::array<unsigned, MAX_NUM_DCORES> tpcShuffleIndex {};
    bool                                 previousTpcNodeLocalityMode = false;
    workDistributionManager::tpcWorkDistribution(tpcNode,
                                                 *nodeRois,
                                                 g.getNumTpcEng(),
                                                 tpcShuffleIndex,
                                                 previousTpcNodeLocalityMode,
                                                 false,
                                                 false,
                                                 MAX_NUM_DCORES);

    NodeROI resultNodeRoi                     = nodeRois->front();
    TSize   resultBoxSize[MAX_DIMENSIONS_NUM] = {1, 1, 1, 1, 1};
    auto    tpcWdCtxs                         = resultNodeRoi.tpcWdCtx;
    for (unsigned dcore = 0; dcore < MAX_NUM_DCORES; dcore++)
    {
        for (unsigned dim = 0; dim < MAX_DIMENSIONS_NUM; dim++)
        {
            if (dcore == 0)
            {
                ASSERT_EQ(tpcWdCtxs[dcore].baseCord[dim], 0x100);
                ASSERT_EQ(tpcWdCtxs[dcore].boxSize[dim], resultBoxSize[dim]);
                ASSERT_EQ(tpcWdCtxs[dcore].gridSize[dim], resultGridSize[dim]);
            }
            else
            {
                ASSERT_EQ(tpcWdCtxs[dcore].baseCord[dim], 0);
                ASSERT_EQ(tpcWdCtxs[dcore].boxSize[dim], 0);
                ASSERT_EQ(tpcWdCtxs[dcore].gridSize[dim], 0);
            }
        }
    }

    delete tpcNodeRoi;
    delete nodeRois;
}

TEST_F(WDTest, locality_two_size_dcore_test)
{
    const unsigned tensor_dim = 1;
    const TSize    size       = 1;
    Gaudi3Graph    g;
    TensorPtr      i = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr      o = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorVector   inputTensors;
    TensorVector   outputTensors;
    TSize          resultDcoreSize[MAX_DIMENSIONS_NUM]     = {4, 8, 8, 4, 3};
    TSize          resultlastDcoreSize[MAX_DIMENSIONS_NUM] = {4, 8, 8, 4, 1};
    TSize          RoiSize[MAX_DIMENSIONS_NUM]             = {4, 8, 8, 4, 10};

    inputTensors.push_back(i);
    outputTensors.push_back(o);
    TPCNode             tpcNode(inputTensors, outputTensors, "test node");
    tpcNode.getNodeAnnotation().perforation = LitePerforationHints {0, 1, true};
    NodeROI*            tpcNodeRoi = new NodeROI();
    std::list<NodeROI>* nodeRois   = new std::list<NodeROI>();
    for (unsigned dcore = 0; dcore < MAX_NUM_DCORES; dcore++)
    {
        DcoreROI dcoreRoi;
        for (unsigned dim = 0; dim < MAX_DIMENSIONS_NUM; dim++)
        {
            dcoreRoi.baseOffset[dim] = 0 + dcore * 0x1000 + dim * (0x1000 / MAX_NUM_DCORES);
            // split to Dcore Rois on SCD
            if (dcore != 3)
            {
                dcoreRoi.size[dim] = resultDcoreSize[dim];
            }
            else
            {
                dcoreRoi.size[dim] = resultlastDcoreSize[dim];
            }

            if (dcore == 0)  // fill once only
            {
                tpcNodeRoi->baseOffset[dim] = 0 + dim * 0x1000;
                tpcNodeRoi->size[dim]       = RoiSize[dim];
                tpcNode.getNodeAnnotation().tpcSplitDims.push_back(MAX_DIMENSIONS_NUM - dim - 1);
            }
        }
        tpcNodeRoi->dcoreROIs.push_back(dcoreRoi);
    }

    nodeRois->push_back(*tpcNodeRoi);
    std::array<unsigned, MAX_NUM_DCORES> tpcShuffleIndex {};
    bool                                 previousTpcNodeLocalityMode = false;
    workDistributionManager::tpcWorkDistribution(tpcNode,
                                                 *nodeRois,
                                                 g.getNumTpcEng(),
                                                 tpcShuffleIndex,
                                                 previousTpcNodeLocalityMode,
                                                 false,
                                                 false,
                                                 MAX_NUM_DCORES);

    NodeROI resultNodeRoi                              = nodeRois->front();
    TSize   resultBoxSize[MAX_DIMENSIONS_NUM]          = {4, 8, 2, 1, 3};
    TSize   resultlastDcoreBoxSize[MAX_DIMENSIONS_NUM] = {4, 8, 2, 1, 1};
    auto    tpcWdCtxs                                  = resultNodeRoi.tpcWdCtx;
    for (unsigned dcore = 0; dcore < MAX_NUM_DCORES; dcore++)
    {
        for (unsigned dim = 0; dim < MAX_DIMENSIONS_NUM; dim++)
        {
            ASSERT_EQ(tpcWdCtxs[dcore].baseCord[dim], 0 + dcore * 0x1000 + dim * (0x1000 / MAX_NUM_DCORES));
            ASSERT_EQ(tpcWdCtxs[dcore].boxSize[dim], (dcore == 3) ? resultlastDcoreBoxSize[dim] : resultBoxSize[dim]);
            ASSERT_EQ(tpcWdCtxs[dcore].gridSize[dim], (dcore == 3) ? resultlastDcoreSize[dim] : resultDcoreSize[dim]);
        }
    }

    delete tpcNodeRoi;
    delete nodeRois;
}
}  // namespace gaudi3