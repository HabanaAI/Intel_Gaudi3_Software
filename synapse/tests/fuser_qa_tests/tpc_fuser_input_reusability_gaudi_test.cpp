#include "gaudi_tests/gaudi_test_infra.h"
#include "synapse_test.hpp"
#include "node_factory.h"
#include "syn_singleton.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

class SynGaudiTPCFuserInputReusabilityTest : public SynGaudiTestInfra
{
protected:
    virtual void SetUpTest() override
    {
        SynGaudiTestInfra::SetUpTest();
        prev_CFG_RUN_TPC_FUSER     = GCFG_RUN_TPC_FUSER.value();
        prev_ENABLE_INTERNAL_NODES = GCFG_ENABLE_INTERNAL_NODES.value();
        GCFG_RUN_TPC_FUSER.setValue(true);
        GCFG_ENABLE_INTERNAL_NODES.setValue(true);
    }

    virtual void TearDownTest() override
    {
        GCFG_RUN_TPC_FUSER.setValue(prev_CFG_RUN_TPC_FUSER);
        GCFG_ENABLE_INTERNAL_NODES.setValue(prev_ENABLE_INTERNAL_NODES);
        SynGaudiTestInfra::TearDownTest();
    };

    unsigned createTensorWithGraphIdx(TensorUsage  usage,
                                      MemInitType  initSelect,
                                      const float* initializer,
                                      unsigned*    sizes,
                                      unsigned     dims,
                                      synDataType  dataType,
                                      unsigned*    strides,
                                      unsigned*    minSizes,
                                      unsigned     graphIdx)
    {
        TensorIndices index = createTensors(1,
                                            usage,
                                            false,
                                            nullptr,
                                            initSelect,
                                            initializer,
                                            sizes,
                                            dims,
                                            dataType,
                                            strides,
                                            graphIdx,
                                            0,
                                            nullptr,
                                            false,
                                            minSizes);
        return index[0];
    }

    void compareTensors(std::vector<unsigned>& prevIter, std::vector<unsigned>& currIter)
    {
        for (unsigned outPersitentTensorIdx = 0; outPersitentTensorIdx < currIter.size(); outPersitentTensorIdx++)
        {
            float* output_x = (float*)m_hostBuffers[prevIter[outPersitentTensorIdx]];
            float* output_y = (float*)m_hostBuffers[currIter[outPersitentTensorIdx]];

            for (uint64_t idx = 0; idx < getDefaultNumberOfElements(); idx++)
            {
                if (std::isnan(*output_x) && std::isnan(*output_y)) continue;
                ASSERT_EQ(*output_x, *output_y) << "OUTPUT: Mismatch for at index " << outPersitentTensorIdx
                                                << " |Expected:" << *output_x << " |Result: " << *output_y;
                output_x++;
                output_y++;
            }
        }
        return;
    }

    virtual void CheckReuseUsingDramOffset(const NodePtr& n, std::uint16_t refReusability, unsigned outIdx)
    {
        // Identify the InIndexes to check the DramOffset Value
        std::vector<unsigned> pos(16, 0);
        unsigned              inIdx = 0;
        while (refReusability)
        {
            if (refReusability & 1) pos[inIdx] = 1;
            inIdx += 1;
            refReusability >>= 1;
        }
        // Matching DramOffsets
        unsigned matching_dramoffset_count = 0;
        LOG_DEBUG(SYN_TEST, "OUT_Tensor DramOffset: 0x{:x}", n->getOutput(outIdx)->getDramOffset());
        for (int inIdx = 0; inIdx < pos.size() && inIdx < n->getNumInputs(); inIdx++)
        {
            LOG_DEBUG(SYN_TEST, "IN_Tensor {} DramOffset: 0x{:x}", inIdx, n->getInput(inIdx)->getDramOffset());
            if (pos[inIdx] == 1 && n->getInput(inIdx)->getDramOffset() == n->getOutput(outIdx)->getDramOffset())
            {
                matching_dramoffset_count += 1;
                LOG_DEBUG(SYN_TEST, "IN_Tensor: {} is Reused by GC for OUT_tensor: {}", inIdx, outIdx);
            }
        }
        LOG_DEBUG(SYN_TEST, "DramOffset Match:", matching_dramoffset_count);
    }

    virtual void NodeFusionCheck(HabanaGraph* graph, std::unordered_map<std::string, int> nodeCount)
    {
        for (const NodePtr& n : graph->getNodes())
        {
            std::string nodeName = n->getNodeName();
            nodeName.erase(remove_if(nodeName.begin(), nodeName.end(), [](char c) { return !isalpha(c); }),
                           nodeName.end());  // get op-name without indexing
            if (nodeCount.find(nodeName) != nodeCount.end())
            {
                --nodeCount[nodeName];
            }
            else
            {
                ASSERT_TRUE(false) << "Node Found in reference map::" << nodeName << std::endl;
            }
        }
        // Check if all count match with expected
        for (auto it = nodeCount.begin(); it != nodeCount.end(); it++)
        {
            EXPECT_EQ(it->second, 0) << "All the nodes are not fused as expected."
                                     << "Issue observed with::" << it->first;
        }
    }

    virtual void FuserRecommendationCheck(HabanaGraph*                         graph,
                                          std::map<unsigned, unsigned>         refReusability,
                                          std::unordered_map<std::string, int> nodeCount)
    {
        NodeFusionCheck(graph, nodeCount);
        for (const NodePtr& n : graph->getNodes())
        {
            LOG_DEBUG(SYN_TEST, "DebugNodeData: {}", n->getNodeName());
            LOG_DEBUG(SYN_TEST, "Search for Memcpy in {}: {}", n->getNodeName(), n->getNodeName().rfind("Memcpy", 0));
            if (!n->getNodeName().rfind("fused", 0))
            {
                TPCNodePtr tpcNode    = std::dynamic_pointer_cast<TPCNode>(n);
                unsigned   numOutputs = tpcNode->getOutputs().size();
                for (unsigned outIdx = 0; outIdx < numOutputs; ++outIdx)
                {
                    auto     ref = refReusability.find(outIdx);
                    unsigned out = tpcNode->getInstance().outputTensorAccessPattern[outIdx].inputsReusability;

                    EXPECT_EQ(ref->second, out) << "Input-Reusablity Mismatch for output tensor: " << outIdx
                                                << " |Expected:" << ref->second << " |Result: " << out;
                    LOG_DEBUG(SYN_TEST, "DebugNodeData: for output {} inputReusability - 0x{:x}", outIdx, out);
                    CheckReuseUsingDramOffset(n, (uint16_t)ref->second, outIdx);
                }
            }
        }
    }

    virtual void EvaluateInputReusability(unsigned                             index,
                                          std::map<unsigned, unsigned>         refReusability,
                                          std::unordered_map<std::string, int> nodeCount = {})
    {
        GraphData&   graphData = getGraph(index);
        HabanaGraph* graph     = synSingleton::getInstanceInternal()->getGraph(graphData.graphHandle);

        FuserRecommendationCheck(graph, refReusability, nodeCount);
    }

private:
    bool prev_CFG_RUN_TPC_FUSER;
    bool prev_ENABLE_INTERNAL_NODES;
};

/*
The input_reusability_validation  tests are 9 unique graph pattern found in fused kernel subgraph captured from
real WL execution. Detailed description is shared in PPT:
https://habanalabs-my.sharepoint.com/:p:/g/personal/nchaudhari_habana_ai/EShw_5ZUbClLmuBGokC8s0oBJewGuIpEFfYlGsWzuuzHuw?e=uf5hav
. Visualization of graph is also captured in PPT. These patterns differs from each other in-term of number of available
tensor for re-use, multiple output tensor suggesting same inputs for re-use , inputs with mutiple successors etc.

reusability suggestion reference values are decided manually and those are mentioned in the beginning of each test .i.e
(refReusability.insert(std::pair<unsigned, unsigned>(0, 0x1)); After the graph execution these values will be used to
check against actual reusability suggestion.
*/

TEST_F_GC(SynGaudiTPCFuserInputReusabilityTest, input_reusability_validation_G79)
{
    unsigned int gIndex      = 0;
    unsigned int dims        = 1;
    unsigned int size1[dims] = {512};
    unsigned int size2[dims] = {1};

    // Ref input re-usablity <outputTensorID, bitMap>
    std::map<unsigned, unsigned> refReusability;
    refReusability.insert(std::pair<unsigned, unsigned>(0, 0x1));
    refReusability.insert(std::pair<unsigned, unsigned>(1, 0x1));

    auto createGraph79 = [&](unsigned graphIdx) {
        std::vector<unsigned> tensorBuffergraph;
        unsigned              temp_in1_1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in3_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in4_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);

        unsigned in1_1 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out1  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in2_1 = connectOutputTensorToInputTensor(out1);
        unsigned in2_2 = connectOutputTensorToInputTensor(out1);
        unsigned out2  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in3_1 = connectOutputTensorToInputTensor(out2);
        unsigned in3_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out3  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1,
                                            dims,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            graphIdx);

        unsigned in4_1 = connectOutputTensorToInputTensor(out3);
        unsigned in4_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out4  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1,
                                            dims,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            graphIdx);
        addNodeToGraph("memcpy", {temp_in1_1}, {in1_1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in3_2}, {in3_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in4_2}, {in4_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("reciprocal_fwd_f32", {in1_1}, {out1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_f32", {in2_1, in2_2}, {out2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("sub_fwd_f32", {in3_1, in3_2}, {out3}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_f32", {in4_1, in4_2}, {out4}, nullptr, 0, nullptr, graphIdx);

        tensorBuffergraph.push_back(out3);
        tensorBuffergraph.push_back(out4);
        return tensorBuffergraph;
    };

    std::vector<unsigned> outputPersistentTensorGraph0 = createGraph79(0);
    compileAndRun();
    std::unordered_map<std::string, int> nodeCount = {{"Memcpy", 3}, {"fusedTPCNode", 1}};
    EvaluateInputReusability(gIndex, refReusability, nodeCount);

    createGraph();
    std::vector<unsigned> outputPersistentTensorGraph1 = createGraph79(1);

    GCFG_RUN_TPC_FUSER.setValue(false);
    compileTopology("reference_gen", 1);
    runTopology(1);
    compareTensors(outputPersistentTensorGraph0, outputPersistentTensorGraph1);
    GCFG_RUN_TPC_FUSER.setValue(true);
}

TEST_F_GC(SynGaudiTPCFuserInputReusabilityTest, input_reusability_validation_G183)
{
    unsigned int gIndex      = 0;
    unsigned int dims        = 4;
    unsigned int size1[dims] = {256, 1024, 1, 1};
    unsigned int size2[dims] = {1, 1, 1, 1};

    // Ref input re-usablity <outputTensorID, bitMap>
    std::map<unsigned, unsigned> refReusability;
    refReusability.insert(std::pair<unsigned, unsigned>(0, 0xc));
    refReusability.insert(std::pair<unsigned, unsigned>(1, 0x0));

    auto createGraph183 = [&](unsigned graphIdx) {
        std::vector<unsigned> tensorBuffergraph;
        uint64_t              memSize    = getMemorySize(size1, syn_type_float, dims);
        unsigned              sectionIdx = createSection(3 * memSize, graphIdx);

        unsigned temp_in2_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned temp_in3_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned temp_in4_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);

        unsigned in1_1 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             size1,
                                             dims,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             graphIdx,
                                             0,
                                             &sectionIdx);
        unsigned out1  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1,
                                            dims,
                                            syn_type_bf16,
                                            nullptr,
                                            nullptr,
                                            graphIdx,
                                            memSize,
                                            &sectionIdx);

        unsigned in2_1 = connectOutputTensorToInputTensor(out1);
        unsigned in2_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out2  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in3_1 = connectOutputTensorToInputTensor(out2);
        unsigned in3_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out3  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in4_1 = connectOutputTensorToInputTensor(out3);
        unsigned in4_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out4  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1,
                                            dims,
                                            syn_type_bf16,
                                            nullptr,
                                            nullptr,
                                            graphIdx,
                                            2 * memSize,
                                            &sectionIdx);
        addNodeToGraph("memcpy", {temp_in2_2}, {in2_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in3_2}, {in3_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in4_2}, {in4_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("cast_f32_to_bf16", {in1_1}, {out1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in2_1, in2_2}, {out2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in3_1, in3_2}, {out3}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("add_fwd_bf16", {in4_1, in4_2}, {out4}, nullptr, 0, nullptr, graphIdx);

        tensorBuffergraph.push_back(out1);
        tensorBuffergraph.push_back(out4);
        return tensorBuffergraph;
    };

    std::vector<unsigned> outputPersistentTensorGraph0 = createGraph183(0);
    compileAndRun();
    std::unordered_map<std::string, int> nodeCount = {{"Memcpy", 3}, {"fusedTPCNode", 1}};
    EvaluateInputReusability(gIndex, refReusability, nodeCount);

    createGraph();
    std::vector<unsigned> outputPersistentTensorGraph1 = createGraph183(1);

    GCFG_RUN_TPC_FUSER.setValue(false);
    compileTopology("reference_gen", 1);
    runTopology(1);
    compareTensors(outputPersistentTensorGraph0, outputPersistentTensorGraph1);
    GCFG_RUN_TPC_FUSER.setValue(true);
}

TEST_F_GC(SynGaudiTPCFuserInputReusabilityTest, input_reusability_validation_G403)
{
    unsigned int gIndex      = 0;
    unsigned int dims        = 1;
    unsigned int size1[dims] = {1024};
    unsigned int size2[dims] = {1};

    // Ref input re-usablity <outputTensorID, bitMap>
    std::map<unsigned, unsigned> refReusability;
    refReusability.insert(std::pair<unsigned, unsigned>(0, 0x1));
    refReusability.insert(std::pair<unsigned, unsigned>(1, 0x20));
    refReusability.insert(std::pair<unsigned, unsigned>(2, 0x21));

    auto createGraph403 = [&](unsigned graphIdx) {
        std::vector<unsigned> tensorBuffergraph;
        uint64_t              memSize    = getMemorySize(size1, syn_type_float, dims);
        unsigned              sectionIdx = createSection(4 * memSize, graphIdx);

        unsigned temp_in1_2  = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned temp_in2_1  = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned temp_in2_2  = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned temp_in4_2  = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned temp_in5_2  = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned temp_in7_2  = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned temp_in9_2  = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned temp_in11_2 = createPersistTensor(INPUT_TENSOR,
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   size2,
                                                   dims,
                                                   syn_type_float,
                                                   nullptr,
                                                   nullptr,
                                                   graphIdx);

        unsigned in1_1 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             size1,
                                             dims,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             graphIdx,
                                             0,
                                             &sectionIdx);
        unsigned in1_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out1  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in2_1 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned in2_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out2  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in3_1 = connectOutputTensorToInputTensor(out1);
        unsigned in3_2 = connectOutputTensorToInputTensor(out2);
        unsigned out3  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1,
                                            dims,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            graphIdx,
                                            memSize,
                                            &sectionIdx);

        unsigned in4_1 = connectOutputTensorToInputTensor(out3);
        unsigned in4_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out4  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in5_1 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             size1,
                                             dims,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             graphIdx,
                                             2 * memSize,
                                             &sectionIdx);
        unsigned in5_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out5  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in6_1 = in2_2;
        unsigned in6_2 = in2_2;
        unsigned out6  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in7_1 = connectOutputTensorToInputTensor(out6);
        unsigned in7_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out7  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in8_1 = connectOutputTensorToInputTensor(out5);
        unsigned in8_2 = connectOutputTensorToInputTensor(out7);
        unsigned out8  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1,
                                            dims,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            graphIdx,
                                            3 * memSize,
                                            &sectionIdx);

        unsigned in9_1 = connectOutputTensorToInputTensor(out8);
        unsigned in9_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out9  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in10_1 = connectOutputTensorToInputTensor(out9);
        unsigned out10  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);

        unsigned in11_1 = connectOutputTensorToInputTensor(out10);
        unsigned in11_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   size2,
                                                   dims,
                                                   syn_type_float,
                                                   nullptr,
                                                   0,
                                                   graphIdx);
        unsigned out11  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);

        unsigned in12_1 = connectOutputTensorToInputTensor(out4);
        unsigned in12_2 = connectOutputTensorToInputTensor(out11);
        unsigned out12  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);

        addNodeToGraph("memcpy", {temp_in1_2}, {in1_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in2_1}, {in2_1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in2_2}, {in2_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in4_2}, {in4_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in5_2}, {in5_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in7_2}, {in7_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in9_2}, {in9_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in11_2}, {in11_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_f32", {in1_1, in1_2}, {out1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_f32", {in2_1, in2_2}, {out2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("add_fwd_f32", {in3_1, in3_2}, {out3}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("div_fwd_f32", {in4_1, in4_2}, {out4}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_f32", {in5_1, in5_2}, {out5}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_f32", {in6_1, in6_2}, {out6}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_f32", {in7_1, in7_2}, {out7}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("add_fwd_f32", {in8_1, in8_2}, {out8}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("div_fwd_f32", {in9_1, in9_2}, {out9}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("sqrt_fwd_f32", {in10_1}, {out10}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("add_fwd_f32", {in11_1, in11_2}, {out11}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("div_fwd_f32", {in12_1, in12_2}, {out12}, nullptr, 0, nullptr, graphIdx);

        tensorBuffergraph.push_back(out3);
        tensorBuffergraph.push_back(out8);

        return tensorBuffergraph;
    };

    std::vector<unsigned> outputPersistentTensorGraph0 = createGraph403(0);
    compileAndRun();
    std::unordered_map<std::string, int> nodeCount = {{"Memcpy", 8}, {"fusedTPCNode", 1}};
    EvaluateInputReusability(gIndex, refReusability, nodeCount);

    createGraph();
    std::vector<unsigned> outputPersistentTensorGraph1 = createGraph403(1);

    GCFG_RUN_TPC_FUSER.setValue(false);
    compileTopology("reference_gen", 1);
    runTopology(1);
    compareTensors(outputPersistentTensorGraph0, outputPersistentTensorGraph1);
    GCFG_RUN_TPC_FUSER.setValue(true);
}

TEST_F_GC(SynGaudiTPCFuserInputReusabilityTest, input_reusability_validation_G513)
{
    unsigned int gIndex      = 0;
    unsigned int dims        = 3;
    unsigned int size1[dims] = {768, 54, 16};
    unsigned int size2[dims] = {768, 1, 1};

    // Ref input re-usablity <outputTensorID, bitMap>
    std::map<unsigned, unsigned> refReusability;
    refReusability.insert(std::pair<unsigned, unsigned>(0, 0x7));
    refReusability.insert(std::pair<unsigned, unsigned>(1, 0x7));

    auto createGraph513 = [&](unsigned graphIdx) {
        std::vector<unsigned> tensorBuffergraph;
        unsigned              temp_in1_1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in1_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in2_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in3_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);

        unsigned in1_1 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned in1_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out1  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in2_1 = connectOutputTensorToInputTensor(out1);
        unsigned in2_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out2  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1,
                                            dims,
                                            syn_type_bf16,
                                            nullptr,
                                            nullptr,
                                            graphIdx);

        unsigned in3_1 = connectOutputTensorToInputTensor(out2);
        unsigned in3_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out3  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        addNodeToGraph("memcpy", {temp_in1_1}, {in1_1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in1_2}, {in1_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in2_2}, {in2_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in3_2}, {in3_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("add_fwd_bf16", {in1_1, in1_2}, {out1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("add_fwd_bf16", {in2_1, in2_2}, {out2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in3_1, in3_2}, {out3}, nullptr, 0, nullptr, graphIdx);

        tensorBuffergraph.push_back(out2);
        return tensorBuffergraph;
    };

    std::vector<unsigned> outputPersistentTensorGraph0 = createGraph513(0);
    compileAndRun();
    std::unordered_map<std::string, int> nodeCount = {{"Memcpy", 4}, {"fusedTPCNode", 1}};
    EvaluateInputReusability(gIndex, refReusability, nodeCount);

    createGraph();
    std::vector<unsigned> outputPersistentTensorGraph1 = createGraph513(1);

    GCFG_RUN_TPC_FUSER.setValue(false);
    compileTopology("reference_gen", 1);
    runTopology(1);
    compareTensors(outputPersistentTensorGraph0, outputPersistentTensorGraph1);
    GCFG_RUN_TPC_FUSER.setValue(true);
}

TEST_F_GC(SynGaudiTPCFuserInputReusabilityTest, input_reusability_validation_G532)
{
    unsigned int gIndex      = 0;
    unsigned int dims        = 3;
    unsigned int size1[dims] = {768, 250, 16};
    unsigned int size2[dims] = {768, 1, 1};
    unsigned int size3[dims] = {1, 250, 16};

    // Ref input re-usablity <outputTensorID, bitMap>
    std::map<unsigned, unsigned> refReusability;
    refReusability.insert(std::pair<unsigned, unsigned>(0, 0x1));
    refReusability.insert(std::pair<unsigned, unsigned>(1, 0x5));

    auto createGraph532 = [&](unsigned graphIdx) {
        std::vector<unsigned> tensorBuffergraph;
        unsigned              temp_in1_1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in1_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size3,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in2_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in3_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);

        unsigned in1_1 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned in1_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size3,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out1  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1,
                                            dims,
                                            syn_type_bf16,
                                            nullptr,
                                            nullptr,
                                            graphIdx);

        unsigned in2_1 = connectOutputTensorToInputTensor(out1);
        unsigned in2_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out2  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in3_1 = connectOutputTensorToInputTensor(out2);
        unsigned in3_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out3  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        addNodeToGraph("memcpy", {temp_in1_1}, {in1_1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in1_2}, {in1_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in2_2}, {in2_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in3_2}, {in3_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in1_1, in1_2}, {out1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in2_1, in2_2}, {out2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in3_1, in3_2}, {out3}, nullptr, 0, nullptr, graphIdx);

        tensorBuffergraph.push_back(out1);
        return tensorBuffergraph;
    };

    std::vector<unsigned> outputPersistentTensorGraph0 = createGraph532(0);
    compileAndRun();
    std::unordered_map<std::string, int> nodeCount = {{"Memcpy", 4}, {"fusedTPCNode", 1}};
    EvaluateInputReusability(gIndex, refReusability, nodeCount);

    createGraph();
    std::vector<unsigned> outputPersistentTensorGraph1 = createGraph532(1);

    GCFG_RUN_TPC_FUSER.setValue(false);
    compileTopology("reference_gen", 1);
    runTopology(1);
    compareTensors(outputPersistentTensorGraph0, outputPersistentTensorGraph1);
    GCFG_RUN_TPC_FUSER.setValue(true);
}

TEST_F_GC(SynGaudiTPCFuserInputReusabilityTest, input_reusability_validation_G517)
{
    unsigned int gIndex      = 0;
    unsigned int dims        = 3;
    unsigned int size1[dims] = {768, 54, 16};
    unsigned int size2[dims] = {1, 54, 16};
    unsigned int size3[dims] = {768, 1, 1};

    // Ref input re-usablity <outputTensorID, bitMap>
    std::map<unsigned, unsigned> refReusability;
    refReusability.insert(std::pair<unsigned, unsigned>(0, 0x2));
    refReusability.insert(std::pair<unsigned, unsigned>(1, 0x2));

    auto createGraph517 = [&](unsigned graphIdx) {
        std::vector<unsigned> tensorBuffergraph;
        unsigned              temp_in1_1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in1_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in2_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size3,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);

        unsigned in1_1 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned in1_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out1  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1,
                                            dims,
                                            syn_type_bf16,
                                            nullptr,
                                            nullptr,
                                            graphIdx);

        unsigned in2_1 = connectOutputTensorToInputTensor(out1);
        unsigned in2_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size3,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out2  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        addNodeToGraph("memcpy", {temp_in1_1}, {in1_1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in1_2}, {in1_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in2_2}, {in2_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in1_1, in1_2}, {out1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in2_1, in2_2}, {out2}, nullptr, 0, nullptr, graphIdx);

        tensorBuffergraph.push_back(out1);
        return tensorBuffergraph;
    };

    std::vector<unsigned> outputPersistentTensorGraph0 = createGraph517(0);
    compileAndRun();
    std::unordered_map<std::string, int> nodeCount = {{"Memcpy", 3}, {"fusedTPCNode", 1}};
    EvaluateInputReusability(gIndex, refReusability, nodeCount);

    createGraph();
    std::vector<unsigned> outputPersistentTensorGraph1 = createGraph517(1);

    GCFG_RUN_TPC_FUSER.setValue(false);
    compileTopology("reference_gen", 1);
    runTopology(1);
    compareTensors(outputPersistentTensorGraph0, outputPersistentTensorGraph1);
    GCFG_RUN_TPC_FUSER.setValue(true);
}

TEST_F_GC(SynGaudiTPCFuserInputReusabilityTest, input_reusability_validation_G540)
{
    unsigned int gIndex      = 0;
    unsigned int dims        = 3;
    unsigned int size1[dims] = {768, 54, 16};
    unsigned int size2[dims] = {1, 1, 1};
    unsigned int size3[dims] = {768, 1, 1};

    // Ref input re-usablity <outputTensorID, bitMap>
    std::map<unsigned, unsigned> refReusability;
    refReusability.insert(std::pair<unsigned, unsigned>(0, 0x7));
    refReusability.insert(std::pair<unsigned, unsigned>(1, 0x7));

    auto createGraph540 = [&](unsigned graphIdx) {
        std::vector<unsigned> tensorBuffergraph;
        unsigned              temp_in1_1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in1_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in2_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in3_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size3,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);

        unsigned in1_1 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned in1_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out1  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in2_1 = connectOutputTensorToInputTensor(out1);
        unsigned in2_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out2  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1,
                                            dims,
                                            syn_type_bf16,
                                            nullptr,
                                            nullptr,
                                            graphIdx);

        unsigned in3_1 = connectOutputTensorToInputTensor(out2);
        unsigned in3_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size3,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out3  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        addNodeToGraph("memcpy", {temp_in1_1}, {in1_1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in1_2}, {in1_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in2_2}, {in2_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in3_2}, {in3_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in1_1, in1_2}, {out1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in2_1, in2_2}, {out2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in3_1, in3_2}, {out3}, nullptr, 0, nullptr, graphIdx);
        tensorBuffergraph.push_back(out2);
        return tensorBuffergraph;
    };

    std::vector<unsigned> outputPersistentTensorGraph0 = createGraph540(0);
    compileAndRun();
    std::unordered_map<std::string, int> nodeCount = {{"Memcpy", 4}, {"fusedTPCNode", 1}};
    EvaluateInputReusability(gIndex, refReusability, nodeCount);

    createGraph();
    std::vector<unsigned> outputPersistentTensorGraph1 = createGraph540(1);
    GCFG_RUN_TPC_FUSER.setValue(false);
    compileTopology("reference_gen", 1);
    runTopology(1);
    compareTensors(outputPersistentTensorGraph0, outputPersistentTensorGraph1);
    GCFG_RUN_TPC_FUSER.setValue(true);
}

TEST_F_GC(SynGaudiTPCFuserInputReusabilityTest, input_reusability_validation_G601)
{
    unsigned int gIndex      = 0;
    unsigned int dims        = 4;
    unsigned int size1[dims] = {512, 256, 3, 3};
    unsigned int size2[dims] = {1, 1, 1, 1};

    // Ref input re-usablity <outputTensorID, bitMap>
    std::map<unsigned, unsigned> refReusability;
    refReusability.insert(std::pair<unsigned, unsigned>(0, 0x1));
    refReusability.insert(std::pair<unsigned, unsigned>(1, 0x0));

    auto createGraph601 = [&](unsigned graphIdx) {
        std::vector<unsigned> tensorBuffergraph;

        uint64_t memSize    = getMemorySize(size1, syn_type_float, dims);
        unsigned sectionIdx = createSection(2 * memSize, graphIdx);
        unsigned temp_in2_1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned temp_in3_1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);

        unsigned in1_1 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             size1,
                                             dims,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             graphIdx,
                                             0,
                                             &sectionIdx);
        unsigned out1  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1,
                                            dims,
                                            syn_type_bf16,
                                            nullptr,
                                            nullptr,
                                            graphIdx,
                                            memSize,
                                            &sectionIdx);

        unsigned in2_1 = connectOutputTensorToInputTensor(out1);
        unsigned in2_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size2,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out2  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in3_1 = connectOutputTensorToInputTensor(out2);
        unsigned in3_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out3  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in4_1 = connectOutputTensorToInputTensor(out3);
        unsigned out4  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        addNodeToGraph("memcpy", {temp_in2_1}, {in2_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in3_1}, {in3_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("cast_f32_to_bf16", {in1_1}, {out1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in2_1, in2_2}, {out2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_bf16", {in3_1, in3_2}, {out3}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("cast_bf16_to_f32", {in4_1}, {out4}, nullptr, 0, nullptr, graphIdx);

        tensorBuffergraph.push_back(out1);  // 8 persistent
        return tensorBuffergraph;
    };

    std::vector<unsigned> outputPersistentTensorGraph0 = createGraph601(0);
    compileAndRun();
    std::unordered_map<std::string, int> nodeCount = {{"Memcpy", 2}, {"fusedTPCNode", 1}};
    EvaluateInputReusability(gIndex, refReusability, nodeCount);

    createGraph();
    std::vector<unsigned> outputPersistentTensorGraph1 = createGraph601(1);

    GCFG_RUN_TPC_FUSER.setValue(false);
    compileTopology("reference_gen", 1);
    runTopology(1);
    compareTensors(outputPersistentTensorGraph0, outputPersistentTensorGraph1);
    GCFG_RUN_TPC_FUSER.setValue(true);
}

TEST_F_GC(SynGaudiTPCFuserInputReusabilityTest, input_reusability_validation_G718)
{
    unsigned int gIndex       = 0;
    unsigned int dims         = 1;
    unsigned int size1[dims]  = {1};
    unsigned int size1k[dims] = {1024};

    // Ref input re-usablity <outputTensorID, bitMap>
    std::map<unsigned, unsigned> refReusability;
    refReusability.insert(std::pair<unsigned, unsigned>(0, 0x6));
    refReusability.insert(std::pair<unsigned, unsigned>(1, 0x58));
    refReusability.insert(std::pair<unsigned, unsigned>(2, 0x25E));

    auto createGraph718 = [&](unsigned graphIdx) {
        std::vector<unsigned> outputPersistentTensorBuffer;
        unsigned              temp_in1_1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in4_1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1k,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in4_2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1k,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in5_1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);
        unsigned              temp_in9_1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  nullptr,
                                                  graphIdx);

        unsigned in1_1 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out1  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in2_1 = connectOutputTensorToInputTensor(out1);
        unsigned in2_2 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             size1k,
                                             dims,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             graphIdx);
        unsigned out2  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1k,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in3_1 = connectOutputTensorToInputTensor(out2);
        unsigned in3_2 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             size1k,
                                             dims,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             graphIdx);
        unsigned out3  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1k,
                                            dims,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            graphIdx);

        unsigned in4_1 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1k,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned in4_2 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1k,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out4  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1k,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in5_1 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out5  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in6_1 = connectOutputTensorToInputTensor(out4);
        unsigned in6_2 = connectOutputTensorToInputTensor(out5);
        unsigned out6  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1k,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in7_1 = connectOutputTensorToInputTensor(out6);
        unsigned in7_2 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             size1k,
                                             dims,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             graphIdx);
        unsigned out7  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            size1k,
                                            dims,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            graphIdx);

        unsigned in8_1 = connectOutputTensorToInputTensor(out7);
        unsigned out8  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1k,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in9_1 = createTensorWithGraphIdx(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  size1,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);
        unsigned out9  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 size1,
                                                 dims,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 graphIdx);

        unsigned in10_1 = connectOutputTensorToInputTensor(out8);
        unsigned in10_2 = connectOutputTensorToInputTensor(out9);
        unsigned out10  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  size1k,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);

        unsigned in11_1 = connectOutputTensorToInputTensor(out3);
        unsigned in11_2 = connectOutputTensorToInputTensor(out10);
        unsigned out11  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  size1k,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);

        unsigned in12_1 = connectOutputTensorToInputTensor(out11);
        unsigned in12_2 = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              size1,
                                              dims,
                                              syn_type_float,
                                              nullptr,
                                              nullptr,
                                              graphIdx);
        unsigned out12  = createTensorWithGraphIdx(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  size1k,
                                                  dims,
                                                  syn_type_float,
                                                  nullptr,
                                                  0,
                                                  graphIdx);

        unsigned in13_1 = connectOutputTensorToInputTensor(out12);
        unsigned in13_2 = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              size1k,
                                              dims,
                                              syn_type_float,
                                              nullptr,
                                              nullptr,
                                              graphIdx);
        unsigned out13  = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             size1k,
                                             dims,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             graphIdx);

        addNodeToGraph("memcpy", {temp_in1_1}, {in1_1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in4_1}, {in4_1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in4_2}, {in4_2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in5_1}, {in5_1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("memcpy", {temp_in9_1}, {in9_1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("constant_f32", {in1_1}, {out1}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_f32", {in2_1, in2_2}, {out2}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("add_fwd_f32", {in3_1, in3_2}, {out3}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_f32", {in4_1, in4_2}, {out4}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("constant_f32", {in5_1}, {out5}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_f32", {in6_1, in6_2}, {out6}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("add_fwd_f32", {in7_1, in7_2}, {out7}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("sqrt_fwd_f32", {in8_1}, {out8}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("constant_f32", {in9_1}, {out9}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("add_fwd_f32", {in10_1, in10_2}, {out10}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("div_fwd_f32", {in11_1, in11_2}, {out11}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("mult_fwd_f32", {in12_1, in12_2}, {out12}, nullptr, 0, nullptr, graphIdx);
        addNodeToGraph("add_fwd_f32", {in13_1, in13_2}, {out13}, nullptr, 0, nullptr, graphIdx);

        outputPersistentTensorBuffer.push_back(out3);   // 0
        outputPersistentTensorBuffer.push_back(out7);   // 1
        outputPersistentTensorBuffer.push_back(out13);  // 2
        return outputPersistentTensorBuffer;
    };
    std::vector<unsigned> outputPersistentTensorGraph0 = createGraph718(0);
    compileAndRun();
    std::unordered_map<std::string, int> nodeCount = {{"Memcpy", 5}, {"TPC", 3}, {"fusedTPCNode", 1}};
    EvaluateInputReusability(gIndex, refReusability, nodeCount);

    createGraph();
    std::vector<unsigned> outputPersistentTensorGraph1 = createGraph718(1);
    GCFG_RUN_TPC_FUSER.setValue(false);
    compileTopology("reference_gen", 1);
    runTopology(1);
    compareTensors(outputPersistentTensorGraph0, outputPersistentTensorGraph1);
    GCFG_RUN_TPC_FUSER.setValue(true);
}
