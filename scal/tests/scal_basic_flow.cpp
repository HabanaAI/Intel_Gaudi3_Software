#include "scal_internal/pkt_macros.hpp"
#include "scal_basic_test.h"
#include "scal.h"
#include "hlthunk.h"
#include "logger.h"
#include "scal_test_utils.h"
#include "gtest/gtest.h"
#include <string>
#include <limits>
#include "scal_test_pqm_pkt_utils.h"

const unsigned cCommandBufferMinSize = 64*1024;

struct TestConfig
{
    const char *scheduler_name;
    const char *completion_group_name;
    const char *cluster_name;
    const char *stream_name;
    uint32_t    scal_group;
};

class SCALFlowTest : public SCALTestDevice,
                     public testing::WithParamInterface<TestConfig>
{
    protected:
        void scheduler_nop_test_internal(unsigned mode);
};

// Special identifier for this speicific test's use-case
#define TPC_AND_MME_COMPUTE_GROUPS  (std::numeric_limits<std::uint32_t>::max())

INSTANTIATE_TEST_SUITE_P(, SCALFlowTest, testing::Values(
    // we can't send alloc barrier and dispatch to pdma streams so when the stream is compute we must use compute cq
    TestConfig({"compute_media_scheduler", "compute_completion_queue0", "",             "compute0", std::numeric_limits<std::uint8_t>::max()}), // scheduler level NOP
    TestConfig({"compute_media_scheduler", "compute_completion_queue0", "mme",          "compute0", SCAL_MME_COMPUTE_GROUP}),
    TestConfig({"compute_media_scheduler", "compute_completion_queue0", "compute_tpc",  "compute0", SCAL_TPC_COMPUTE_GROUP}),
    TestConfig({"compute_media_scheduler", "compute_completion_queue0", "compute_edma", "compute0", SCAL_EDMA_COMPUTE_GROUP}),
    TestConfig({"compute_media_scheduler", "compute_completion_queue0", "",             "compute0", TPC_AND_MME_COMPUTE_GROUPS}),
    TestConfig({"compute_media_scheduler", "compute_completion_queue0", "rotator",      "compute0", SCAL_RTR_COMPUTE_GROUP}),
    TestConfig({"compute_media_scheduler", "compute_completion_queue0", "cme",          "compute0", SCAL_CME_GROUP}),
    TestConfig({"compute_media_scheduler", "pdma_rx_completion_queue0", "",             "pdma_rx0", SCAL_PDMA_RX_GROUP}),     // Direct mode PDMA NOP
    TestConfig({"compute_media_scheduler", "pdma_tx_completion_queue0", "",             "pdma_tx0", SCAL_PDMA_TX_DATA_GROUP}) // Direct mode PDMA NOP
    // TestConfig({"scaleup_receive",         "network_scaleup_init_completion_queue0", "", "scaleup_receive0", std::numeric_limits<std::uint8_t>::max()})
)
);



TEST_P_CHKDEV(SCALFlowTest, gaudi3_scheduler_nop_test)
{
    TestConfig testConfig = GetParam();
    bool isCmeOnGaudi2 = testConfig.cluster_name == std::string("cme") && getScalDeviceType() == dtGaudi2;
    if (testConfig.cluster_name == std::string("compute_edma") || isCmeOnGaudi2)
    {
        LOG_INFO(SCAL,"gaudi3_scheduler_nop_test skipped. {} not supported", testConfig.cluster_name);
        GTEST_SKIP() << "gaudi3_scheduler_nop_test skipped " << testConfig.cluster_name << " not supported";
    }
    scheduler_nop_test_internal(0);
}

TEST_P_CHKDEV(SCALFlowTest, gaudi3_scheduler_nop_test_wrap_buffer,{GAUDI3})
{
    TestConfig testConfig = GetParam();
    if (testConfig.cluster_name == std::string("compute_edma"))
    {
        LOG_INFO(SCAL,"gaudi3_scheduler_nop_test_wrap_buffer skipped. {} not supported", testConfig.cluster_name);
        GTEST_SKIP() << "gaudi3_scheduler_nop_test_wrap_buffer skipped " << testConfig.cluster_name << " not supported";
    }
    scheduler_nop_test_internal(4);
}

TEST_P_CHKDEV(SCALFlowTest, scheduler_nop_test,{GAUDI2})
{
    TestConfig testConfig = GetParam();
    if (testConfig.cluster_name == std::string("cme"))
    {
        LOG_INFO(SCAL,"gaudi2 scheduler_nop_test skipped. {} not supported", testConfig.cluster_name);
        GTEST_SKIP() << "gaudi2 scheduler_nop_test skipped " << testConfig.cluster_name << " not supported";
    }
    scheduler_nop_test_internal(0);
}

TEST_P_CHKDEV(SCALFlowTest, scheduler_nop_test_wrap_buffer,{GAUDI2})
{
    TestConfig testConfig = GetParam();
    if (testConfig.cluster_name == std::string("cme"))
    {
        LOG_INFO(SCAL,"gaudi2 scheduler_nop_test_wrap_buffer skipped. {} not supported", testConfig.cluster_name);
        GTEST_SKIP() << "gaudi2 scheduler_nop_test_wrap_buffer skipped " << testConfig.cluster_name << " not supported";
    }
    scheduler_nop_test_internal(4);
}

TEST_P_CHKDEV(SCALFlowTest, scal_stub_test,{GAUDI2})
{
    TestConfig testConfig = GetParam();
    if (testConfig.cluster_name == std::string("cme"))
    {
        LOG_INFO(SCAL,"gaudi2 scal_stub_test skipped. {} not supported", testConfig.cluster_name);
        GTEST_SKIP() << "gaudi2 scal_stub_test skipped " << testConfig.cluster_name << " not supported";
    }
    setenv("ENABLE_SCAL_STUB","true",1); // does overwrite
    scheduler_nop_test_internal(1);
    setenv("ENABLE_SCAL_STUB","false",1); // does overwrite
}

static unsigned CheckCmdWrap(unsigned prevPI, unsigned newPI, unsigned streamDccmBufSize)
{
    //  scheduler commands cannot wrap on the scheduler internal dccm buffer
    //    the dccm size is defined in the config (json) and can be obtained by:
    //   rc = scal_stream_get_info(m_computeStreamHandle, &m_streamInfo);
    //   stream_cmd_buffer_dccm_size = m_streamInfo.command_alignment
    //
    //   so if our host cmd buffer is 2K and dccm buffer is 512 bytes
    //     our command cannot cross  512,1024,1536,2048
    unsigned currQuarter = prevPI / streamDccmBufSize;
    unsigned newQuarter = (newPI-1) / streamDccmBufSize;

    if(newQuarter != currQuarter)
        return (streamDccmBufSize-(prevPI%streamDccmBufSize)); // how much padding needed
    return 0;// OK
}

void SCALFlowTest::scheduler_nop_test_internal(unsigned mode)
{
    int rc;
    scal_handle_t scalHandle;
    scal_pool_handle_t memPool;
    scal_buffer_handle_t ctrlBuff;
    //scal_core_handle_t schedHandle;
    scal_stream_handle_t stream;

    scalDeviceType deviceType = getScalDeviceType();

    uint32_t hostCyclicBufferSize = cCommandBufferMinSize;
    TestConfig testConfig = GetParam();
    int scalFd = m_fd;
    if ((int)getScalDeviceType() != getScalDeviceTypeEnv())
    {
        LOG_ERR(SCAL,"{}: actual device {} != environment SCAL_DEVICE_TYPE {}",__FUNCTION__,  (int)getScalDeviceType(), getScalDeviceTypeEnv());
        ASSERT_EQ(1, 0);
        return;
    }
    std::string confFileStr = getConfigFilePath(":/default.json");
    const char* confFile    = confFileStr.c_str();
    printf("Loading scal with config=%s\n",confFile);

    rc = scal_init(scalFd, confFile, &scalHandle, nullptr);
    ASSERT_EQ(rc, 0);

    rc = scal_get_pool_handle_by_name(scalHandle, "host_shared", &memPool);
    ASSERT_EQ(rc, 0);

    rc = scal_allocate_buffer(memPool, hostCyclicBufferSize, &ctrlBuff);
    ASSERT_EQ(rc, 0);

    rc = scal_get_stream_handle_by_name(scalHandle, testConfig.stream_name, &stream);
    ASSERT_EQ(rc, 0);

    rc = scal_stream_set_priority(stream, 1);
    ASSERT_EQ(rc, 0);

    rc = scal_stream_set_commands_buffer(stream, ctrlBuff);
    ASSERT_EQ(rc, 0);

    // index is from stream info
    scal_stream_info_t streamInfo;
    rc = scal_stream_get_info(stream, &streamInfo);
    ASSERT_EQ(rc, 0);

    bool isStreamDirectMode = streamInfo.isDirectMode;

    scal_buffer_info_t bufferInfo;
    rc = scal_buffer_get_info(ctrlBuff, &bufferInfo);
    ASSERT_EQ(rc, 0);

    // generate command buffer
    char* currCommandBufferLoc = (char*)bufferInfo.host_address;

    scal_comp_group_handle_t cgHandle;
    rc = scal_get_completion_group_handle_by_name(scalHandle, testConfig.completion_group_name, &cgHandle);
    ASSERT_EQ(rc, 0);

    scal_completion_group_infoV2_t cgInfo;
    rc = scal_completion_group_get_infoV2(cgHandle, &cgInfo);
    ASSERT_EQ(rc, 0);

    scal_cluster_handle_t cluster;
    scal_cluster_info_t   info;

    uint8_t engine_groups[4] = {testConfig.scal_group, QMAN_ENGINE_GROUP_TYPE_COUNT,
                                QMAN_ENGINE_GROUP_TYPE_COUNT, QMAN_ENGINE_GROUP_TYPE_COUNT};
    uint8_t* engine_groups_ptr = nullptr;
    unsigned num_engine_groups = 0;
    uint32_t numCompletions = cgInfo.force_order;

    if (testConfig.scal_group == TPC_AND_MME_COMPUTE_GROUPS)
    {
        engine_groups_ptr = &engine_groups[0];
        engine_groups[0] = SCAL_MME_COMPUTE_GROUP;
        engine_groups[1] = SCAL_TPC_COMPUTE_GROUP;
        num_engine_groups = 2;
        rc = scal_get_cluster_handle_by_name(scalHandle, "compute_tpc", &cluster);
        ASSERT_EQ(rc, 0);
        rc = scal_cluster_get_info(cluster, &info);
        ASSERT_EQ(rc, 0);
        numCompletions += info.numCompletions;
        rc = scal_get_cluster_handle_by_name(scalHandle, "mme", &cluster);
        ASSERT_EQ(rc, 0);
        rc = scal_cluster_get_info(cluster, &info);
        ASSERT_EQ(rc, 0);
        numCompletions += info.numCompletions;
        printf("using both TPC & MME.  numCompletions=%d\n", numCompletions);
    }
    else if (testConfig.cluster_name != std::string(""))
    {
        rc = scal_get_cluster_handle_by_name(scalHandle, testConfig.cluster_name, &cluster);
        ASSERT_EQ(rc, 0);
        rc = scal_cluster_get_info(cluster, &info);
        ASSERT_EQ(rc, 0);
        numCompletions += info.numCompletions;
        engine_groups_ptr = &engine_groups[0];
        num_engine_groups = 1;
        printf("using cluster %s  numCompletions=%d\n", testConfig.cluster_name, numCompletions);
    }
    // each loop raises pi by 20
    unsigned loop_size_in_bytes = isStreamDirectMode ? 32 : 20;
    uint32_t num_loops = 0;
    uint64_t target = 1;
    uint32_t num_wraps = 5; // full buffer wrap arounds
    uint64_t pi3 = 0;
    unsigned pi = 0;
    auto buildPkt = createBuildPkts(getScalDeviceType());

    while(1)
    {
        if (isStreamDirectMode)
        {
            PqmPktUtils::buildPqmFenceCmd((uint8_t*)currCommandBufferLoc, 0/*fence_id*/, 0, 0);
            currCommandBufferLoc += PqmPktUtils::getFenceCmdSize();
        }
        else if (deviceType != dtGaudi2)
        {
            currCommandBufferLoc += fillPkt<AcpFenceWaitPkt>(buildPkt, currCommandBufferLoc, 0 /*fenceId*/, 0 /*target*/);
        }
        else
        {
            currCommandBufferLoc += fillPkt<FenceWaitPkt>(buildPkt, currCommandBufferLoc, 0, 0);
        }

        if ((testConfig.scal_group == SCAL_PDMA_TX_CMD_GROUP)  ||
            (testConfig.scal_group == SCAL_PDMA_TX_DATA_GROUP) ||
            (testConfig.scal_group == SCAL_PDMA_RX_GROUP))
        {
            PqmPktUtils::sendPdmaCommand(
                isStreamDirectMode, buildPkt, currCommandBufferLoc, 0, 0, 0, testConfig.scal_group,
                -1/*workloadType*/, 0/*ctxId*/, 0/*payload*/, 0/*payloadAddr*/, 0/*bMemset*/, 1/*signal_to_cg*/, false/*wr_comp*/, cgInfo.index_in_scheduler/*completionGroupIndex*/,
                0, 0);

            currCommandBufferLoc += PqmPktUtils::getPdmaCmdSize(isStreamDirectMode, buildPkt, false, 1);
        }
        else
        {
            const EngineGroupArrayType engineGroupType {};
            currCommandBufferLoc += fillPkt<AllocBarrierV2bPkt>(buildPkt, currCommandBufferLoc, cgInfo.index_in_scheduler, numCompletions, false, false, 0, engineGroupType, 0, 0);
            currCommandBufferLoc += fillPkt<DispatchBarrierPkt>(buildPkt, currCommandBufferLoc, num_engine_groups, engine_groups_ptr, 0);
        }

        //rc = scal_get_core_handle_by_name(scalHandle, testConfig.scheduler_name, &schedHandle);

        unsigned newPi = currCommandBufferLoc - (char *)bufferInfo.host_address;
        pi3 += (newPi - pi);
        pi = newPi;
        if (mode == 1)
            rc = scal_stream_submit(stream, pi, 113);// this should fail unless ENABLE_SCAL_STUB was defined
        else
        {
            rc = scal_stream_submit(stream, (unsigned int)pi3, streamInfo.submission_alignment);
        }
        ASSERT_EQ(rc, 0);

        rc = scal_completion_group_wait(cgHandle, target, SCAL_FOREVER);
        ASSERT_EQ(rc, 0);
        target++;
        if ((pi <= 60) || (pi > hostCyclicBufferSize-60))
        {
           printf("num_loops=%d pi=%d pi3=%d subalign=%d\n", num_loops, pi,(unsigned)pi3,streamInfo.submission_alignment);
        }
        if (pi + loop_size_in_bytes >= hostCyclicBufferSize)
        {
            // pad (see SchedCmd::PadReset())
            unsigned left = hostCyclicBufferSize - pi;
            pi3 += left;
            unsigned paddingDwords = (left - (sizeof(struct g2fw::sched_arc_cmd_nop_t)))/4;
            printf("hostCyclicBufferSize=%d left=%d paddingDwords=%d\n", hostCyclicBufferSize, left, paddingDwords);
            if (isStreamDirectMode)
            {
                for (int i = 0 ; i < 4; i++)
                {
                    PqmPktUtils::buildNopCmd(currCommandBufferLoc);
                    currCommandBufferLoc += PqmPktUtils::getNopCmdSize();
                }
            } else
            {
                fillPktNoSize<NopCmdPkt>(buildPkt, currCommandBufferLoc, left);
            }
            // wrap around
            currCommandBufferLoc = (char*)bufferInfo.host_address;
            printf("Wrap Around\n");
            //pi = getchar();
            pi = 0;
            num_wraps--;
            if (num_wraps == 0)
                break;
        }
        unsigned padding_needed = CheckCmdWrap(pi, pi+loop_size_in_bytes, 256);// dccm buf size
        if(padding_needed != 0)
        {
            fillPktNoSize<NopCmdPkt>(buildPkt, currCommandBufferLoc, padding_needed);
            currCommandBufferLoc += padding_needed;
        }
        num_loops++;
        if (mode != 4)
            break;
    }
#ifdef PAUSE
    printf("press a key\n\n");
    fflush(stdout);
    getchar();
#endif
    // tear down
    rc = scal_free_buffer(ctrlBuff);
    ASSERT_EQ(rc, 0);

    rc = scal_get_pool_handle_by_name(scalHandle, "global_hbm", &memPool);
    ASSERT_EQ(rc, 0);
    scal_memory_pool_infoV2 poolInfo;
    rc = scal_pool_get_infoV2(memPool, &poolInfo);
    ASSERT_EQ(rc, 0);

    scal_destroy(scalHandle);

    LOG_INFO(SCAL,"completed basic test flow");
}
