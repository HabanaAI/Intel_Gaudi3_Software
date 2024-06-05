#include <stdint.h>
#include <memory>
#include <platform/gaudi2/graph_compiler/gaudi2_graph.h>
#include <recipe_allocator.h>

#include "node_factory.h"
#include "gc_gaudi_test_infra.h"
#include "syn_singleton.hpp"
#include <gtest/gtest.h>

class SynGaudiSFGTests : public SynGaudiTestInfra
{
public:
    SynGaudiSFGTests() { ReleaseDevice(); }

    void
    createSFGGraph(unsigned numNodes, unsigned& reluIn, unsigned& dmaOut, unsigned dmaFreq = 1, unsigned tpcFreq = 1)
    {
        unsigned rollingIdxIn  = 0;
        unsigned rollingIdxOut = 0;

        // Graph will have this structure: [relu_fwd]->[memcpy]->[relu_fwd]->[memcpy]->[relu_fwd]-> ... ->[memcpy]
        reluIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE);
        rollingIdxOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

        Tensor* tpcTensor = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut));
        tpcTensor->setTensorAsExternal(true);

        unsigned count = 1;

        addNodeToGraph("relu_fwd_f32", {reluIn}, {rollingIdxOut});
        for (unsigned i = 0; i < numNodes; i++)
        {
            rollingIdxIn  = connectOutputTensorToInputTensor(rollingIdxOut);
            rollingIdxOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

            if (count % dmaFreq == 0)
            {
                Tensor* dmaTensor = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut));
                dmaTensor->setTensorAsExternal(true);
            }
            addNodeToGraph("memcpy", {rollingIdxIn}, {rollingIdxOut});

            rollingIdxIn  = connectOutputTensorToInputTensor(rollingIdxOut);
            rollingIdxOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

            if (count % tpcFreq == 0)
            {
                tpcTensor = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut));
                tpcTensor->setTensorAsExternal(true);
            }

            addNodeToGraph("relu_fwd_f32", {rollingIdxIn}, {rollingIdxOut});

            count++;
        }
        rollingIdxIn = connectOutputTensorToInputTensor(rollingIdxOut);
        dmaOut       = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);
        addNodeToGraph("memcpy", {rollingIdxIn}, {dmaOut});
    }

    void createParallelSFGGraph(unsigned  numNodes,
                                unsigned& reluIn1,
                                unsigned& dmaOut1,
                                unsigned& reluIn2,
                                unsigned& dmaOut2,
                                unsigned& reluIn3,
                                unsigned& dmaOut3,
                                bool      randomizeExtTensor = false,
                                unsigned  dmaFreq            = 1,
                                unsigned  tpcFreq            = 1)
    {
        unsigned rollingIdxIn1  = 0;
        unsigned rollingIdxOut1 = 0;

        unsigned rollingIdxIn2  = 0;
        unsigned rollingIdxOut2 = 0;

        unsigned rollingIdxIn3  = 0;
        unsigned rollingIdxOut3 = 0;

        // Graph will have this structure: [relu_fwd]->[memcpy]->[relu_fwd]->[memcpy]->[relu_fwd]-> ... ->[memcpy]
        // We can either set all tensors to be external tensors of randomize it (randomizeExtTensor=true)
        reluIn1        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE);
        rollingIdxOut1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

        reluIn2        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE);
        rollingIdxOut2 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

        reluIn3        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE);
        rollingIdxOut3 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

        Tensor* tpcTensor1 = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut1));
        tpcTensor1->setTensorAsExternal(true);

        Tensor* tpcTensor2 = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut2));
        tpcTensor2->setTensorAsExternal(true);

        Tensor* tpcTensor3 = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut3));
        tpcTensor3->setTensorAsExternal(true);

        unsigned count = 1;

        addNodeToGraph("relu_fwd_f32", {reluIn1}, {rollingIdxOut1});
        addNodeToGraph("relu_fwd_f32", {reluIn2}, {rollingIdxOut2});
        addNodeToGraph("relu_fwd_f32", {reluIn3}, {rollingIdxOut3});

        for (unsigned i = 0; i < numNodes; i++)
        {
            rollingIdxIn1  = connectOutputTensorToInputTensor(rollingIdxOut1);
            rollingIdxOut1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

            rollingIdxIn2  = connectOutputTensorToInputTensor(rollingIdxOut2);
            rollingIdxOut2 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

            rollingIdxIn3  = connectOutputTensorToInputTensor(rollingIdxOut3);
            rollingIdxOut3 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

            if (count % dmaFreq == 0)
            {
                if (!randomizeExtTensor || (randomizeExtTensor && rand() % 4 == 2))
                {
                    Tensor* dmaTensor1 = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut1));
                    dmaTensor1->setTensorAsExternal(true);

                    Tensor* dmaTensor2 = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut2));
                    dmaTensor2->setTensorAsExternal(true);

                    Tensor* dmaTensor3 = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut3));
                    dmaTensor3->setTensorAsExternal(true);
                }
            }
            addNodeToGraph("memcpy", {rollingIdxIn1}, {rollingIdxOut1});
            addNodeToGraph("memcpy", {rollingIdxIn2}, {rollingIdxOut2});
            addNodeToGraph("memcpy", {rollingIdxIn3}, {rollingIdxOut3});

            rollingIdxIn1  = connectOutputTensorToInputTensor(rollingIdxOut1);
            rollingIdxOut1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

            rollingIdxIn2  = connectOutputTensorToInputTensor(rollingIdxOut2);
            rollingIdxOut2 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

            rollingIdxIn3  = connectOutputTensorToInputTensor(rollingIdxOut3);
            rollingIdxOut3 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

            if (count % tpcFreq == 0)
            {
                if (!randomizeExtTensor || (randomizeExtTensor && rand() % 4 == 2))
                {
                    tpcTensor1 = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut1));
                    tpcTensor1->setTensorAsExternal(true);

                    tpcTensor2 = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut2));
                    tpcTensor2->setTensorAsExternal(true);

                    tpcTensor3 = reinterpret_cast<Tensor*>(getTensorByIndex(rollingIdxOut3));
                    tpcTensor3->setTensorAsExternal(true);
                }
            }

            addNodeToGraph("relu_fwd_f32", {rollingIdxIn1}, {rollingIdxOut1});
            addNodeToGraph("relu_fwd_f32", {rollingIdxIn2}, {rollingIdxOut2});
            addNodeToGraph("relu_fwd_f32", {rollingIdxIn3}, {rollingIdxOut3});

            count++;
        }
        rollingIdxIn1 = connectOutputTensorToInputTensor(rollingIdxOut1);
        dmaOut1       = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

        rollingIdxIn2 = connectOutputTensorToInputTensor(rollingIdxOut2);
        dmaOut2       = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

        rollingIdxIn3 = connectOutputTensorToInputTensor(rollingIdxOut3);
        dmaOut3       = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

        addNodeToGraph("memcpy", {rollingIdxIn1}, {dmaOut1});
        addNodeToGraph("memcpy", {rollingIdxIn2}, {dmaOut2});
        addNodeToGraph("memcpy", {rollingIdxIn3}, {dmaOut3});
    }

    void validateBucketsCompilation(GraphData& graphData,
                                    unsigned   numNodes,
                                    unsigned   dmaFreq       = 1,
                                    unsigned   tpcFreq       = 1,
                                    bool       usingBuckets  = false,
                                    unsigned   rightMonArm   = 1,
                                    unsigned   rightMonSetup = 1,
                                    unsigned   leftMonSetup  = 1)
    {
        unsigned numOfSignalingTensors;
        unsigned numOfEngineTypes;

        HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(graphData.graphHandle);

        graph->getSignalOutInfo(numOfSignalingTensors, numOfEngineTypes);

        ASSERT_EQ(numOfEngineTypes, 2);

        ASSERT_EQ(numOfSignalingTensors, (numNodes / dmaFreq + numNodes / tpcFreq) + 1);

        auto     nodes = graph->getExeSortedNodes();
        unsigned i     = 0;
        const std::shared_ptr<SyncObjectManager>& syncObjectManager = graph->getCodeGenerator()->getSyncObjectManager();
        for (auto& node : nodes)
        {
            if (graph->runsOnTPC(node) || node->isDma())
            {
                auto preSyncs = node->getNodeAnnotation().syncScheme[0].preSyncsAndMon;
                if (preSyncs.size() == 1)
                {
                    EXPECT_EQ(preSyncs.front().type, SyncOrMonitor::MONITOR_OBJ);
                    EXPECT_EQ(preSyncs.front().monitor.shouldInc, true);
                    EXPECT_EQ(preSyncs.front().monitor.mask, (1 << node->getNodeAnnotation().syncScheme.size()) - 1);
                    if (usingBuckets)
                    {
                        EXPECT_GE(preSyncs.front().monitor.setupValue, leftMonSetup);
                    }
                    if (graph->runsOnTPC(node))
                    {
                        EXPECT_EQ(preSyncs.front().monitor.signalSyncId,
                                  syncObjectManager->getSyncConventions().getSignalOutGroup() + 1);
                    }
                    else  // DMA
                    {
                        EXPECT_EQ(preSyncs.front().monitor.signalSyncId,
                                  syncObjectManager->getSyncConventions().getSignalOutGroup());
                    }
                }

                if (!node->getNodeAnnotation().syncScheme.front().patchableMonitors.empty())
                {
                    EXPECT_EQ(node->getNodeAnnotation().syncScheme.front().patchableMonitors.size(), 1);
                    auto monitor = node->getNodeAnnotation().syncScheme.front().patchableMonitors[0];

                    EXPECT_EQ(monitor.monObject.shouldInc, true);
                    EXPECT_EQ(monitor.monObject.syncId,
                              syncObjectManager->getSyncConventions().getSignalOutGroup() /
                                  syncObjectManager->getSyncConventions().getGroupSize());
                    if (usingBuckets)
                    {
                        EXPECT_GE(monitor.monObject.setupValue, rightMonSetup);  // bucket size
                        EXPECT_GE(monitor.monObject.armValue, rightMonArm);
                    }
                    else
                    {
                        EXPECT_EQ(monitor.monObject.setupValue, 1);
                        EXPECT_EQ(monitor.monObject.armValue, ++i);
                    }
                }
            }
        }
    }

    void validateCompilation(GraphData& graphData,
                             unsigned   numNodes,
                             unsigned   dmaFreq            = 1,
                             unsigned   tpcFreq            = 1,
                             bool       usingBuckets       = false,
                             bool       randomizeExtTensor = false,
                             unsigned   parallelExe        = 1)
    {
        // relevant only for gaudi device type
        if (m_deviceType != synDeviceGaudi) return;

        unsigned numOfSignalingTensors;
        unsigned numOfEngineTypes;

        HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(graphData.graphHandle);

        graph->getSignalOutInfo(numOfSignalingTensors, numOfEngineTypes);

        ASSERT_EQ(numOfEngineTypes, 2);
        if (!randomizeExtTensor)
        {
            ASSERT_EQ(numOfSignalingTensors, (numNodes / dmaFreq + numNodes / tpcFreq) * parallelExe + parallelExe);
        }

        auto     nodes = graph->getExeSortedNodes();
        unsigned i     = 0;
        const SyncConventions& syncConventions = graph->getCodeGenerator()->getSyncObjectManager()->getSyncConventions();

        for (auto& node : nodes)
        {
            if (graph->runsOnTPC(node) || node->isDma())
            {
                auto preSyncs = node->getNodeAnnotation().syncScheme[0].preSyncsAndMon;
                if (preSyncs.size() == 1)
                {
                    EXPECT_EQ(preSyncs.front().type, SyncOrMonitor::MONITOR_OBJ);
                    EXPECT_EQ(preSyncs.front().monitor.shouldInc, true);
                    EXPECT_EQ(preSyncs.front().monitor.mask, (1 << node->getNodeAnnotation().syncScheme.size()) - 1);
                    if (graph->runsOnTPC(node))
                    {
                        EXPECT_EQ(preSyncs.front().monitor.signalSyncId, syncConventions.getSignalOutGroup() + 1);
                    }
                    else  // DMA
                    {
                        EXPECT_EQ(preSyncs.front().monitor.signalSyncId, syncConventions.getSignalOutGroup());
                    }
                }

                if (!node->getNodeAnnotation().syncScheme.front().patchableMonitors.empty())
                {
                    EXPECT_EQ(node->getNodeAnnotation().syncScheme.front().patchableMonitors.size(), 1);
                    auto monitor = node->getNodeAnnotation().syncScheme.front().patchableMonitors[0];

                    EXPECT_EQ(monitor.monObject.shouldInc, true);
                    EXPECT_EQ(monitor.monObject.syncId, syncConventions.getSignalOutGroup() / syncConventions.getGroupSize());
                    if (usingBuckets)
                    {
                        EXPECT_GE(monitor.monObject.setupValue, 1);  // bucket size
                    }
                    else
                    {
                        EXPECT_EQ(monitor.monObject.setupValue, 1);
                        EXPECT_EQ(monitor.monObject.armValue, ++i);
                    }
                }
            }
        }
    }

    void validateNodeAndGraphAnnotation(GraphData& graphData,
                                        unsigned   numNodes,
                                        unsigned   dmaFreq            = 1,
                                        unsigned   tpcFreq            = 1,
                                        bool       randomizeExtTensor = false,
                                        unsigned   parallelExe        = 1)
    {
        // relevant only for gaudi2 device type
        if (m_deviceType != synDeviceGaudi2) return;

        unsigned numOfSignalingTensors;
        unsigned numOfEngineTypes;

        HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(graphData.graphHandle);

        graph->getSignalOutInfo(numOfSignalingTensors, numOfEngineTypes);

        ASSERT_EQ(numOfEngineTypes, 2);
        if (!randomizeExtTensor)
        {
            ASSERT_EQ(numOfSignalingTensors, (numNodes / dmaFreq + numNodes / tpcFreq) * parallelExe + parallelExe);
        }

        HB_ASSERT(graph->getCodeGenerator()->getDeviceSfgInitValue().size() > 1,
                  "Expecting at least 2 entries in GraphAnnotation deviceSfgInitValue map");

        for (auto& sfgInitVal : graph->getCodeGenerator()->getDeviceSfgInitValue())
        {
            HB_ASSERT(sfgInitVal.second > 0, "Invalid SFG init value");
        }

        auto nodes                      = graph->getExeSortedNodes();
        bool bNodeAnnotationContainsSFG = false;

        for (auto& node : nodes)
        {
            if (graph->runsOnTPC(node) || node->isDma())
            {
                if (node->getNodeAnnotation().sfgSyncObjValue.is_set())
                {
                    bNodeAnnotationContainsSFG = true;
                    HB_ASSERT(node->getNodeAnnotation().sfgSyncObjValue.value() > 0, "Invalid node SFG value");
                }
            }
        }

        HB_ASSERT(bNodeAnnotationContainsSFG, "Failed to fin SFG value in nodes");
    }

    void validateECBList(GraphData& graphData)
    {
        // relevant only for gaudi2 device type
        if (m_deviceType != synDeviceGaudi2) return;

        HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(graphData.graphHandle);

        RecipeAllocator recipeAlloc;
        recipe_t*       recipe = graph->serializeDataPlane(&recipeAlloc);

        // verify ECB SFG command are produced
        ASSERT_EQ(recipe->arc_jobs_nr, 4);

        char*                                   pChar       = nullptr;
        struct eng_arc_cmd_list_size_t*         sizeCmd     = nullptr;
        struct eng_arc_cmd_nop_t*               nopCmd      = nullptr;
        struct eng_arc_cmd_sfg_t*               sfgCmd      = nullptr;
        struct eng_arc_cmd_sched_dma_t*         schedDmaCmd = nullptr;
        struct eng_arc_cmd_wd_fence_and_exec_t* wdCmd       = nullptr;

        for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
        {
            // dynamic ecb list should start with 1 ListSize
            pChar   = (char*)recipe->arc_jobs[i].dynamic_ecb.cmds;
            sizeCmd = (struct eng_arc_cmd_list_size_t*)pChar;
            ASSERT_EQ(sizeCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_LIST_SIZE);
            ASSERT_EQ(sizeCmd->yield, 0);
            pChar += sizeof(struct eng_arc_cmd_list_size_t);

            if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::TPC)
            {
                // no SFG init cmd for TPC. We expect to find the first SignalOutEngArcCommand after the 3rd WD command

                // Nop command
                nopCmd = (struct eng_arc_cmd_nop_t*)pChar;
                ASSERT_EQ(nopCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_NOP);
                ASSERT_EQ(nopCmd->yield, 0);
                pChar += sizeof(struct eng_arc_cmd_nop_t);

                // skip the first 8 leading sched_dma commands
                for (unsigned j = 0; j < 8; j++)
                {
                    schedDmaCmd = (struct eng_arc_cmd_sched_dma_t*)pChar;
                    ASSERT_EQ(schedDmaCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_SCHED_DMA);
                    ASSERT_EQ(schedDmaCmd->yield, 1);
                    pChar += sizeof(struct eng_arc_cmd_sched_dma_t);
                }
                // Now we expect WD->SCHED_DMA->WD->SCHED_DMA->WD->SFG
                wdCmd = (struct eng_arc_cmd_wd_fence_and_exec_t*)pChar;
                ASSERT_EQ(wdCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_WD_FENCE_AND_EXE);
                ASSERT_EQ(wdCmd->yield, 0);
                pChar += sizeof(struct eng_arc_cmd_wd_fence_and_exec_t);

                schedDmaCmd = (struct eng_arc_cmd_sched_dma_t*)pChar;
                ASSERT_EQ(schedDmaCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_SCHED_DMA);
                ASSERT_EQ(schedDmaCmd->yield, 1);
                pChar += sizeof(struct eng_arc_cmd_sched_dma_t);

                wdCmd = (struct eng_arc_cmd_wd_fence_and_exec_t*)pChar;
                ASSERT_EQ(wdCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_WD_FENCE_AND_EXE);
                ASSERT_EQ(wdCmd->yield, 0);
                pChar += sizeof(struct eng_arc_cmd_wd_fence_and_exec_t);

                schedDmaCmd = (struct eng_arc_cmd_sched_dma_t*)pChar;
                ASSERT_EQ(schedDmaCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_SCHED_DMA);
                ASSERT_EQ(schedDmaCmd->yield, 1);
                pChar += sizeof(struct eng_arc_cmd_sched_dma_t);

                wdCmd = (struct eng_arc_cmd_wd_fence_and_exec_t*)pChar;
                ASSERT_EQ(wdCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_WD_FENCE_AND_EXE);
                ASSERT_EQ(wdCmd->yield, 0);
                pChar += sizeof(struct eng_arc_cmd_wd_fence_and_exec_t);

                sfgCmd = (struct eng_arc_cmd_sfg_t*)pChar;
                ASSERT_EQ(sfgCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_SFG);
                ASSERT_EQ(sfgCmd->yield, 0);
                ASSERT_EQ(sfgCmd->sob_inc_value, 1);
            }
            else  // MME/DMA/ROT
            {
                // expecting 1 InitSfg and 1 NOP command in dynamic ECB list
                sfgCmd = (struct eng_arc_cmd_sfg_t*)pChar;
                ASSERT_EQ(sfgCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_SFG);
                ASSERT_EQ(sfgCmd->yield, 0);
                if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::DMA)
                {
                    ASSERT_EQ(sfgCmd->sob_inc_value, 6);
                }
                else
                {
                    ASSERT_EQ(sfgCmd->sob_inc_value, 12);
                }
                pChar += sizeof(struct eng_arc_cmd_sfg_t);

                nopCmd = (struct eng_arc_cmd_nop_t*)pChar;
                ASSERT_EQ(nopCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_NOP);
                ASSERT_EQ(nopCmd->yield, 0);
            }
        }
    }
};

// Basic test to validate GC logic when using SFG feature:
// 1. Create long graph: tpcNode->DMANode->tpcNode-> ...->DMANode
// 2. Validate graph SignalOutInfo: numOfSignalingTensors & numOfEngineTypes
// 3. Validate monitors setup (including patchable monitors) - gaudi
// 4. Validate all monitors are satisffied (execution completes) - gaudi
// 5. Validate SFG info on graph & node annotation - gaudi2
TEST_F_GC(SynGaudiSFGTests, signaling_from_graph_with_tpc_and_dma_using_buckets, {synDeviceGaudi2})
{
    static const unsigned numNodes = 300;
    unsigned              reluIn   = 0;
    unsigned              dmaOut   = 0;
    unsigned              dmaFreq  = 3;
    unsigned              tpcFreq  = 1;

    createSFGGraph(numNodes, reluIn, dmaOut, dmaFreq, tpcFreq);

    compileTopology();

    GraphData&   graphData = getGraph(0);

    validateCompilation(graphData, numNodes, dmaFreq, tpcFreq, true);

    validateNodeAndGraphAnnotation(graphData, numNodes, dmaFreq, tpcFreq);

    runTopology();

    auto* input  = m_hostBuffers[reluIn];
    auto* output = m_hostBuffers[dmaOut];

    ASSERT_EQ(memcmp(input, output, getDefaultNumberOfElements() * sizeof(float)), 0);
}

TEST_F_GC(SynGaudiSFGTests, signaling_from_graph_with_tpc_and_single_dma, {synDeviceGaudi2})
{
    static const unsigned numNodes = 10;
    unsigned              reluIn   = 0;
    unsigned              dmaOut   = 0;
    unsigned              dmaFreq  = 6;
    unsigned              tpcFreq  = 1;

    createSFGGraph(numNodes, reluIn, dmaOut, dmaFreq, tpcFreq);

    compileTopology();

    GraphData& graphData = getGraph(0);

    validateCompilation(graphData, numNodes, dmaFreq, tpcFreq);

    validateNodeAndGraphAnnotation(graphData, numNodes, dmaFreq, tpcFreq);

    validateECBList(graphData);

    runTopology();

    auto* input  = m_hostBuffers[reluIn];
    auto* output = m_hostBuffers[dmaOut];

    ASSERT_EQ(memcmp(input, output, getDefaultNumberOfElements() * sizeof(float)), 0);
}

TEST_F_GC(SynGaudiSFGTests, signaling_from_graph_with_buckets, {synDeviceGaudi2})
{
    static const unsigned numNodes = 400;
    unsigned              reluIn   = 0;
    unsigned              dmaOut   = 0;
    unsigned              dmaFreq  = 1;
    unsigned              tpcFreq  = 1;

    createSFGGraph(numNodes, reluIn, dmaOut, dmaFreq, tpcFreq);

    compileTopology();

    GraphData& graphData = getGraph(0);

    validateCompilation(graphData, numNodes, dmaFreq, tpcFreq, true);

    validateNodeAndGraphAnnotation(graphData, numNodes, dmaFreq, tpcFreq);

    runTopology();

    auto* input  = m_hostBuffers[reluIn];
    auto* output = m_hostBuffers[dmaOut];

    ASSERT_EQ(memcmp(input, output, getDefaultNumberOfElements() * sizeof(float)), 0);
}

TEST_F_GC(SynGaudiSFGTests, signaling_from_graph_parallel_exe_with_buckets, {synDeviceGaudi2})
{
    static const unsigned numNodes = 100;
    unsigned              reluIn1  = 0;
    unsigned              dmaOut1  = 0;
    unsigned              reluIn2  = 0;
    unsigned              dmaOut2  = 0;
    unsigned              reluIn3  = 0;
    unsigned              dmaOut3  = 0;
    unsigned              dmaFreq  = 1;
    unsigned              tpcFreq  = 1;

    createParallelSFGGraph(numNodes, reluIn1, dmaOut1, reluIn2, dmaOut2, reluIn3, dmaOut3);

    compileTopology();

    GraphData& graphData = getGraph(0);

    validateCompilation(graphData, numNodes, dmaFreq, tpcFreq, true, false, 3);

    validateNodeAndGraphAnnotation(graphData, numNodes, dmaFreq, tpcFreq, false, 3);

    runTopology();

    auto* input1  = m_hostBuffers[reluIn1];
    auto* output1 = m_hostBuffers[dmaOut1];

    ASSERT_EQ(memcmp(input1, output1, getDefaultNumberOfElements() * sizeof(float)), 0);

    auto* input2  = m_hostBuffers[reluIn2];
    auto* output2 = m_hostBuffers[dmaOut2];

    ASSERT_EQ(memcmp(input2, output2, getDefaultNumberOfElements() * sizeof(float)), 0);

    auto* input3  = m_hostBuffers[reluIn3];
    auto* output3 = m_hostBuffers[dmaOut3];

    ASSERT_EQ(memcmp(input3, output3, getDefaultNumberOfElements() * sizeof(float)), 0);
}

TEST_F_GC(SynGaudiSFGTests,
          signaling_from_graph_parallel_exe_random_graph_with_buckets_ASIC_CI,
          {synDeviceGaudi2})
{
    static const unsigned numNodes = 300;
    unsigned              reluIn1  = 0;
    unsigned              dmaOut1  = 0;
    unsigned              reluIn2  = 0;
    unsigned              dmaOut2  = 0;
    unsigned              reluIn3  = 0;
    unsigned              dmaOut3  = 0;
    unsigned              dmaFreq  = 1;
    unsigned              tpcFreq  = 1;

    createParallelSFGGraph(numNodes, reluIn1, dmaOut1, reluIn2, dmaOut2, reluIn3, dmaOut3, true);

    compileTopology();

    GraphData& graphData = getGraph(0);

    validateCompilation(graphData, numNodes, dmaFreq, tpcFreq, true, true, 3);

    validateNodeAndGraphAnnotation(graphData, numNodes, dmaFreq, tpcFreq, true, 3);

    runTopology();

    auto* input1  = m_hostBuffers[reluIn1];
    auto* output1 = m_hostBuffers[dmaOut1];

    ASSERT_EQ(memcmp(input1, output1, getDefaultNumberOfElements() * sizeof(float)), 0);

    auto* input2  = m_hostBuffers[reluIn2];
    auto* output2 = m_hostBuffers[dmaOut2];

    ASSERT_EQ(memcmp(input2, output2, getDefaultNumberOfElements() * sizeof(float)), 0);

    auto* input3  = m_hostBuffers[reluIn3];
    auto* output3 = m_hostBuffers[dmaOut3];

    ASSERT_EQ(memcmp(input3, output3, getDefaultNumberOfElements() * sizeof(float)), 0);
}

class SynGaudiSFGWithBucketsTests
: public SynGaudiSFGTests
, public testing::WithParamInterface<std::tuple<bool, int, int, int, int>>
{
};

TEST_P_GC(SynGaudiSFGWithBucketsTests, DISABLED_signaling_from_graph_with_bucketing, {synDeviceGaudi})
{
    static const unsigned numNodes = 20;
    unsigned              reluIn   = 0;
    unsigned              dmaOut   = 0;
    unsigned              dmaFreq  = 4;
    unsigned              tpcFreq  = 3;

    bool     usingBuckets;
    unsigned numOfMonitors;
    unsigned rightMonArm;
    unsigned rightMonSetup;
    unsigned leftMonSetup;

    std::tie(usingBuckets, numOfMonitors, rightMonArm, rightMonSetup, leftMonSetup) = GetParam();

    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    GCFG_SFG_MAX_NUM_OF_MONITORS.setValue(numOfMonitors);

    createSFGGraph(numNodes, reluIn, dmaOut, dmaFreq, tpcFreq);

    compileTopology();

    GraphData& graphData = getGraph(0);

    validateBucketsCompilation(graphData,
                               numNodes,
                               dmaFreq,
                               tpcFreq,
                               usingBuckets,
                               rightMonArm,
                               rightMonSetup,
                               leftMonSetup);

    runTopology();

    auto* input  = m_hostBuffers[reluIn];
    auto* output = m_hostBuffers[dmaOut];

    ASSERT_EQ(memcmp(input, output, getDefaultNumberOfElements() * sizeof(float)), 0);
}

INSTANTIATE_TEST_SUITE_P(,
                         SynGaudiSFGWithBucketsTests,
                         testing::Values(std::make_tuple(false, 6, 1, 1, 1),
                                         std::make_tuple(true, 8, 6, 2, 3),
                                         std::make_tuple(true, 10, 4, 2, 3),
                                         std::make_tuple(false, 24, 1, 1, 1),
                                         std::make_tuple(false, 400, 1, 1, 1)));
