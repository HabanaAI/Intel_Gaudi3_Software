#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "scoped_configuration_change.h"

class SplitToPhysicalRoisTest : public GraphOptimizerTest
{
    virtual void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        // optimize tpc may change the node shape and change the number of pipeline levels a little.
        // disabling this for test verification
        setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "false");
    }
};


TEST_F(SplitToPhysicalRoisTest, even_distribution_per_tpc_engine)
{
    GaudiGraph g;
    TSize sizes[] = {84,129,11,32};
    unsigned int dim = ARRAY_SIZE(sizes);
    pTensor tReluBwdIn1(new Tensor(dim, sizes, syn_type_float, nullptr, nullptr, false, true));
    pTensor tReluBwdIn2(new Tensor(dim, sizes, syn_type_float, nullptr, nullptr, false, true));
    pTensor tReluBwdOut(new Tensor(dim, sizes, syn_type_float, nullptr, nullptr, true));
    pNode nodeReluBwd = NodeFactory::createGenericTPCNode({tReluBwdIn1, tReluBwdIn2}, {tReluBwdOut}, nullptr,
                                                          "relu_bwd_f32", "node_relu_bwd");
    GraphEditor::addNode(g, nodeReluBwd);

    // Set graph's input/output tensors as persistent
    synMemoryDescriptor softamx_in1_memDesc(true);
    tReluBwdIn1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    tReluBwdIn1->setMemoryDescriptor(softamx_in1_memDesc);
    synMemoryDescriptor softamx_in2_memDesc(true);
    tReluBwdIn2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    tReluBwdIn2->setMemoryDescriptor(softamx_in2_memDesc);
    // Set graph's output tensor as persistent
    synMemoryDescriptor softamx_out_memDesc(true);
    tReluBwdOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    tReluBwdOut->setMemoryDescriptor(softamx_out_memDesc);


    ASSERT_TRUE(g.compile()) << "compilation failed";
    const NodeVector& nodes = g.getExeSortedNodes();
    for (const NodePtr& node : nodes)
    {
        if (node->getNodeType() == Node::TYPE_USER)
        {
            std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
            ASSERT_NE(tpcNode, nullptr);
            if (tpcNode->getGUID() == "relu_bwd_f32")
            {
                const std::list<NodeROI> *rois = node->getPhysicalRois();
                ASSERT_NE(rois, nullptr);
                uint64_t totalRoisSize = 0;
                std::vector<std::vector<uint64_t>> roiSizesPerEngine;
                roiSizesPerEngine.resize(g.getNumTpcEng());

                for (const auto& roi: *rois)
                {
                    uint64_t roiSize = multiplyElements(roi.size, roi.size + dim);
                    roiSizesPerEngine[roi.engineIndex].push_back(roiSize);
                    totalRoisSize += roiSize;
                }
                uint64_t expectedTotalRoiPerEngineSize = totalRoisSize / g.getNumTpcEng();

                const std::vector<uint64_t> roiSizesInFirstEngine = roiSizesPerEngine[0];
                bool engineWithDifferentRoisThan1stExists = false;
                for (const std::vector<uint64_t>& roiSizesInCurEngine: roiSizesPerEngine)
                {
                    uint64_t totalRoiSizeInCurEngine = std::accumulate(roiSizesInCurEngine.begin(), roiSizesInCurEngine.end(), 0);
                    // check that the sum of all roi sizes in this engine is equal to
                    // expected value = total size of all rois / number of tpc engines.
                    // it also means that each engine has the same total work as other engines
                    ASSERT_EQ(totalRoiSizeInCurEngine, expectedTotalRoiPerEngineSize);
                    if (roiSizesInCurEngine != roiSizesInFirstEngine)
                    {
                        engineWithDifferentRoisThan1stExists = true;
                    }
                }
                // check that there is at least one engine ("x") with work which is different than the work of the 1st
                // engine. this fact, together with the fact they both have the same total work, means that in one
                // logical roi the bigger work was scheduled to the 1st engine, while in another logical roi the bigger
                // work was scheduled to engine x.
                ASSERT_TRUE(engineWithDifferentRoisThan1stExists);
            }
        }
    }
}

// test for split physical ROIs distribution when when roi size < num of engines, so the rois are split on multiple dims
TEST_F(SplitToPhysicalRoisTest, balanced_distribution_per_tpc_engine_split_multiple_dims)
{
    GaudiGraph g;
    ScopedConfigurationChange MaxAvailableTpcMode("TPC_ENGINES_ENABLED_MASK", "0x7f");  // use 7 TPCs
    TSize sizes[] = {128,128,1,8};
    unsigned int dims = ARRAY_SIZE(sizes);
    pTensor tReluBwdIn1(new Tensor(dims, sizes, syn_type_float, nullptr, nullptr, false, true));
    pTensor tReluBwdIn2(new Tensor(dims, sizes, syn_type_float, nullptr, nullptr, false, true));
    pTensor tReluBwdOut(new Tensor(dims, sizes, syn_type_float, nullptr, nullptr, true));
    pNode nodeReluBwd = NodeFactory::createGenericTPCNode({tReluBwdIn1, tReluBwdIn2}, {tReluBwdOut}, nullptr,
                                                          "relu_bwd_f32", "node_relu_bwd");
    GraphEditor::addNode(g, nodeReluBwd);

    // Set graph's input/output tensors as persistent
    synMemoryDescriptor in1MemDesc(true);
    tReluBwdIn1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    tReluBwdIn1->setMemoryDescriptor(in1MemDesc);
    synMemoryDescriptor in2MemDesc(true);
    tReluBwdIn2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    tReluBwdIn2->setMemoryDescriptor(in2MemDesc);
    // Set graph's output tensor as persistent
    synMemoryDescriptor outMemDesc(true);
    tReluBwdOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    tReluBwdOut->setMemoryDescriptor(outMemDesc);
    ASSERT_TRUE(g.compile()) << "compilation failed";
    const NodeVector& nodes        = g.getExeSortedNodes();
    unsigned numOfEngines = g.getNumTpcEng();
    EXPECT_GT(numOfEngines, 1);
    for (const NodePtr& node : nodes)
    {
        if (node->getNodeType() == Node::TYPE_USER)
        {
            std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
            ASSERT_NE(tpcNode, nullptr);
            if (tpcNode->getGUID() == "relu_bwd_f32")
            {
                const std::list<NodeROI>* logicalRois = g.GetNodeROIs(node);
                ASSERT_NE(logicalRois, nullptr);
                // Expect at least 3 logical ROIs
                EXPECT_GE(logicalRois->size(), 3);
                const std::list<NodeROI>* physicalRois = node->getPhysicalRois();
                ASSERT_NE(physicalRois, nullptr);
                ASSERT_EQ(logicalRois->size(), std::ceil(physicalRois->size() / numOfEngines));
                // just prepare data so we can iterate each logical ROI:
                std::vector<std::list<NodeROI>> physicalRoisPerLogicalRoi;
                physicalRoisPerLogicalRoi.resize(logicalRois->size());
                int i = 1, logicalRoiIndex = 0;
                for (const auto& roi: *physicalRois)
                {
                    physicalRoisPerLogicalRoi[logicalRoiIndex].push_back(roi);
                    if (i == numOfEngines)
                    {
                        ++logicalRoiIndex;
                        i = 1;
                    }
                    else
                    {
                        ++i;
                    }
                }
                auto iterLogical = logicalRois->begin();
                std::unordered_map<unsigned, unsigned> engineIndicesToBiggerRoisCount;
                for (auto& physicalRois : physicalRoisPerLogicalRoi)
                {
                    auto curLogicalRoi = iterLogical->size;
                    // for each logical roi, find maximum roi size (and make sure not all rois have the same size so
                    // we'll have an interesting case to check), then fill the set with all indices of engines that
                    // have this max size. We don't expect to have the same indices in different logical rois (or if it
                    // happens, it should be because all of the indices already exist in the set)
                    auto minMaxRoi = std::minmax_element(physicalRois.begin(), physicalRois.end(),
                            [&] (NodeROI& a, NodeROI& b) {
                                uint64_t aRoiSize = multiplyElements(a.size, a.size + dims);
                                uint64_t bRoiSize = multiplyElements(b.size, b.size + dims);
                                return aRoiSize < bRoiSize;
                    });
                    uint64_t minRoiSize =  multiplyElements(minMaxRoi.first->size, minMaxRoi.first->size + dims);
                    uint64_t maxRoiSize =  multiplyElements(minMaxRoi.second->size, minMaxRoi.second->size + dims);
                    ASSERT_GT(maxRoiSize, minRoiSize);

                    for (const auto& physicalRoi : physicalRois)
                    {
                        // validate the logical roi is split on multiple dims
                        unsigned splitDims = 0;
                        for (unsigned dim = 0; dim < Tensor::c_tensorMaxDim; ++dim)
                        {
                            if (curLogicalRoi[dim] != physicalRoi.size[dim])
                            {
                                ++splitDims;
                            };
                        }
                        ASSERT_GT(splitDims,1);

                        //keep track of engine indices which have the max roi (in each logical roi)
                        uint64_t roiSize = multiplyElements(physicalRoi.size, physicalRoi.size + dims);
                        if (roiSize == maxRoiSize)
                        {
                            engineIndicesToBiggerRoisCount[physicalRoi.engineIndex]++;

                        }
                    }
                    ++iterLogical;
                }

                auto minMaxCountBigRoisPerEngine = std::minmax_element(engineIndicesToBiggerRoisCount.begin(), engineIndicesToBiggerRoisCount.end(),
                                                     [&] (const std::pair<unsigned,unsigned>& a, const std::pair<unsigned,unsigned>& b) {
                                                         return a.second < b.second;
                                                     });
                unsigned minNumBigRoisInEngine =  minMaxCountBigRoisPerEngine.first->second;
                unsigned maxNumBigRoisInEngine =  minMaxCountBigRoisPerEngine.second->second;
                // we allow the gap between the engine with the biggest number of logical rois where his physical roi
                // was the biggest to the engine with the smallest number of such occurrences to be no more than 1.
                // since we have at least 3 logical rois and for each of them there exists physical rois in different
                // sizes, it means some kind of balancing was done here
                ASSERT_TRUE(maxNumBigRoisInEngine - minNumBigRoisInEngine <= 1);
            }
        }
    }
}
