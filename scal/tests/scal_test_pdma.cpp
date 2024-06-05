#include "scal_basic_test.h"
#include "scal.h"
#include "hlthunk.h"
#include "logger.h"
#include "scal_test_utils.h"
#include "scal_test_pqm_pkt_utils.h"

#include "scal_gaudi3_sync_monitor.h"

class SCALInitTest : public SCALTest
{
    public:
        void acquireDevice();
        void releaseDevice();

        void init();

        int                       scalFd     = -1;
        scal_handle_t             scalHandle = nullptr;
        struct hlthunk_hw_ip_info hw_ip;
};

void SCALInitTest::acquireDevice()
{
    scalFd = hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    ASSERT_GE(scalFd, 0) << "hlthunk_open failed. return errno " << errno;;

    int ret = hlthunk_get_hw_ip_info(scalFd, &hw_ip);
    ASSERT_EQ(ret, 0);

    scalHandle = nullptr;
}

void SCALInitTest::releaseDevice()
{
    if (scalHandle != nullptr)
    {
        scal_destroy(scalHandle);
    }

    int rc = hlthunk_close(scalFd);
    ASSERT_EQ(rc, 0) << "hlthunk_close failed. return errno " << errno;
    printf("Clean up Finished\n");
}

void SCALInitTest::init()
{
    int rc;

    const char configFilePath[] = ":/default.json";

    std::string confFileStr = getConfigFilePath(configFilePath);
    const char* confFile    = confFileStr.c_str();
    printf("Loading scal with config=%s\n",confFile);
    rc = scal_init(scalFd, confFile, &scalHandle, nullptr);
    ASSERT_EQ(rc, 0);
}

TEST_F_CHKDEV(SCALInitTest, scal_init_test)
{
    acquireDevice();

    const char* envVarName  = "SCAL_CFG_OVERRIDE_PATH";
    const char* envVarValue = getenv(envVarName);

    init();

    if (envVarValue != nullptr)
    {
        setenv(envVarName, envVarValue, 1); // does overwrite
    }
    else
    {
        unsetenv(envVarName);
    }

    releaseDevice();
}

TEST_F(SCALTest, scal_sm_map_test)
{
    int scalFd = hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    ASSERT_GE(scalFd, 0);

    int rc;
    scal_handle_t scalHandle;
    const char configFilePath[] = ":/default.json";
    std::string confFileStr = getConfigFilePath(configFilePath);
    const char* confFile    = confFileStr.c_str();
    printf("Loading scal with config=%s\n",confFile);
    rc = scal_init(scalFd, confFile, &scalHandle, nullptr);
    ASSERT_EQ(rc, 0);

    printf("Read SM0\n");
    scal_sm_info_t smInfo;
    rc = scal_get_sm_info(scalHandle, 0, &smInfo);
    ASSERT_EQ(rc, 0);

    printf("Check SM0\n");
    ASSERT_EQ(smInfo.idx, 0u);
    ASSERT_EQ(smInfo.objs, nullptr);
    ASSERT_EQ(smInfo.glbl, nullptr);

    printf("Read SM2\n");
    rc = scal_get_sm_info(scalHandle, 2, &smInfo);
    ASSERT_EQ(rc, 0);

    printf("Check SM2\n");
    ASSERT_EQ(smInfo.idx, 2u);
    ASSERT_TRUE(smInfo.objs);
    ASSERT_TRUE(smInfo.glbl);

    printf("Write to Sync Object #216\n");

    scal_write_mapped_reg(&smInfo.objs[216], 0x00000003);
    scal_write_mapped_reg(&smInfo.objs[216], 0x80000004);
    ASSERT_EQ(scal_read_mapped_reg(&smInfo.objs[216]), 7u);

    scal_write_mapped_reg(&smInfo.objs[216], 0x00000002);
    ASSERT_EQ(scal_read_mapped_reg(&smInfo.objs[216]), 2u);

    scal_destroy(scalHandle);
    hlthunk_close(scalFd);
    printf("Clean up Finished \n");
}

INSTANTIATE_TEST_SUITE_P(, SCALTestDmPdma, testing::Values(
    TestParams({"pdma_tx0", "pdma_rx0", "pdma_tx_completion_queue0", "pdma_rx_completion_queue0", SCAL_PDMA_TX_DATA_GROUP, SCAL_PDMA_RX_GROUP, true}),
    TestParams({"pdma_tx0", "compute0", "pdma_tx_completion_queue0", "compute_completion_queue0", SCAL_PDMA_TX_DATA_GROUP, SCAL_PDMA_RX_DEBUG_GROUP, false})
)
);
// need to add engine group for test params of rx_debug and need to use compute stream

void SCALTestDmPdma::pdma_host_sync_internal()
{
    TestParams testConfig = GetParam();
    if ((getScalDeviceType() == dtGaudi2) && (!testConfig.is_supported_by_gaudi2))
    {
        GTEST_SKIP() << "This test is not suitable Gaudi2";
    }

    int rc;
    scal_handle_t scalHandle;
    scal_pool_handle_t memHostPoolHandle;
    scal_buffer_handle_t ctrlBuffHandle = nullptr;
    scal_buffer_handle_t ctrlBuffHandle2 = nullptr;
    scal_buffer_handle_t host2devBuffHandle = nullptr;
    scal_buffer_handle_t dev2hostBuffHandle = nullptr;
    scal_stream_handle_t H2DStreamHandle  = nullptr;
    scal_stream_handle_t D2HStreamHandle  = nullptr;
    scal_buffer_handle_t deviceDataBuffHandle = nullptr;
    uint8_t *hostBuffer = nullptr;
    uint8_t *hostBuffer2 = nullptr;
    unsigned pi = 0;
    uint64_t target = 0;

    const char     configFilePath[]      = ":/default.json";
    const unsigned cCommandBufferMinSize = 64*1024;

    const unsigned testDmaBufferSize = 4096 * 10; // 1MB

    uint32_t hostCyclicBufferSize = cCommandBufferMinSize;

    int scalFd = m_fd; //hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    ASSERT_GE(scalFd, 0);
    std::string confFileStr = getConfigFilePath(configFilePath);
    const char* confFile    = confFileStr.c_str();
    printf("Loading scal with config=%s\n",confFile);
    rc = scal_init(scalFd, confFile, &scalHandle, nullptr);
    ASSERT_EQ(rc, 0);

    // check that init succeedded
    scal_comp_group_handle_t cgHandleRx;
    scal_comp_group_handle_t cgHandleTx;
    scal_stream_info_t streamInfo;
    scal_stream_info_t streamInfo2;
    scal_pool_handle_t deviceMemPoolHandle;
    scal_buffer_info_t cmdBufferInfo;
    scal_buffer_info_t host2devBufferInfo;
    scal_buffer_info_t deviceDataBuffInfo;
    scal_buffer_info_t dev2hostBufferInfo;
    scal_buffer_info_t cmdBufferInfo2;
    scal_completion_group_infoV2_t cgInfoRx;
    scal_completion_group_infoV2_t cgInfoTx;
    char *currCommandBufferLoc = nullptr;
    char *currCommandBufferLoc2 = nullptr;

    rc = scal_get_completion_group_handle_by_name(scalHandle, testConfig.first_completion_group_name, &cgHandleTx);
    ASSERT_EQ(rc, 0);

    rc = scal_get_completion_group_handle_by_name(scalHandle, testConfig.second_completion_group_name, &cgHandleRx);
    ASSERT_EQ(rc, 0);

    //  Get Host Memory Pool handle
    rc = scal_get_pool_handle_by_name(scalHandle, "host_shared", &memHostPoolHandle);
    ASSERT_EQ(rc, 0);

    //  Get Device Memory Pool handle
    rc = scal_get_pool_handle_by_name(scalHandle, "hbm_shared", &deviceMemPoolHandle);
    ASSERT_EQ(rc, 0);

    // Allocate Buffer on the Host for commands
    rc = scal_allocate_buffer(memHostPoolHandle, hostCyclicBufferSize, &ctrlBuffHandle);
    ASSERT_EQ(rc, 0);

    // Allocate Buffer on the Host for DMA
    rc = scal_allocate_buffer(memHostPoolHandle, testDmaBufferSize, &host2devBuffHandle);
    ASSERT_EQ(rc, 0);

    // get device data buffer
    rc = scal_allocate_buffer(deviceMemPoolHandle, testDmaBufferSize, &deviceDataBuffHandle);
    ASSERT_EQ(rc, 0);

    // get the PDMA1 stream handle (H2D  host 2 device)
    rc = scal_get_stream_handle_by_name(scalHandle, testConfig.first_stream_name, &H2DStreamHandle);
    ASSERT_EQ(rc, 0);

    // assign ctrlBuff to be the command buffer of H2DStreamHandle
    rc = scal_stream_set_commands_buffer(H2DStreamHandle, ctrlBuffHandle);
    ASSERT_EQ(rc, 0);

    // index is from stream info
    rc = scal_stream_get_info(H2DStreamHandle, &streamInfo);
    ASSERT_EQ(rc, 0);

    // command buffer
    rc = scal_buffer_get_info(ctrlBuffHandle, &cmdBufferInfo);
    ASSERT_EQ(rc, 0);

    // Host DMA buffer
    rc = scal_buffer_get_info(host2devBuffHandle, &host2devBufferInfo);
    ASSERT_EQ(rc, 0);

    // HBM DMA buffer
    rc = scal_buffer_get_info(deviceDataBuffHandle, &deviceDataBuffInfo);
    ASSERT_EQ(rc, 0);

    hostBuffer = (uint8_t *)host2devBufferInfo.host_address;
    memset(hostBuffer, 0xCC, testDmaBufferSize); // just some value

    // generate command buffer
    currCommandBufferLoc = (char *)cmdBufferInfo.host_address;
    // test the sched_arc_cmd_pdma_batch_transfer_t command

    auto buildPkt = createBuildPkts(getScalDeviceType());

    bool isTxDirectMode = streamInfo.isDirectMode;

    rc = scal_completion_group_get_infoV2(cgHandleTx, &cgInfoTx);
    ASSERT_EQ(rc, 0);

    PqmPktUtils::sendPdmaCommand(
        isTxDirectMode, buildPkt, currCommandBufferLoc, (uint64_t) host2devBufferInfo.device_address, (uint64_t) deviceDataBuffInfo.device_address,
        testDmaBufferSize, testConfig.first_engine_group,
        -1/*workloadType*/, 0/*ctxId*/, 0/*payload*/, 0/*payloadAddr*/, 0/*bMemset*/, 0/*signal_to_cg*/, false/*wr_comp*/, 0/*completionGroupIndex*/,
        0, 0);

    currCommandBufferLoc += PqmPktUtils::getPdmaCmdSize(isTxDirectMode, buildPkt, false/*wr_comp*/, 1);

    PqmPktUtils::sendPdmaCommand(
        isTxDirectMode, buildPkt, currCommandBufferLoc,  0/*src*/, 0/*dst*/, /*size*/0, testConfig.first_engine_group,
        -1/*workloadType*/, 0/*ctxId*/, 0/*payload*/, 0/*payloadAddr*/, 0/*bMemset*/, 1/*signal_to_cg*/, false/*wr_comp*/, cgInfoTx.index_in_scheduler/*completionGroupIndex*/,
        0, 0);

    currCommandBufferLoc += PqmPktUtils::getPdmaCmdSize(isTxDirectMode, buildPkt, false/*wr_comp*/, 1);

    // submit the command buffer on H2DStreamHandle
    pi = currCommandBufferLoc - (char *)cmdBufferInfo.host_address;
    rc = scal_stream_submit(H2DStreamHandle, pi, streamInfo.submission_alignment);
    ASSERT_EQ(rc, 0);

    target = 1; // tbd!
    // wait for completion
    printf("Waiting for stream 1 completion\n");
    rc = scal_completion_group_wait(cgHandleTx, target, SCAL_FOREVER);
    ASSERT_EQ(rc, 0);
    printf("stream 1 completed\n");

    //
    // Now - copy the data back ( from device to host,  and compare)
    //

    // Allocate Buffer on the Host for commands
    rc = scal_allocate_buffer(memHostPoolHandle, hostCyclicBufferSize, &ctrlBuffHandle2);
    ASSERT_EQ(rc, 0);

    // Allocate Buffer on the Host for DMA
    rc = scal_allocate_buffer(memHostPoolHandle, testDmaBufferSize, &dev2hostBuffHandle);
    ASSERT_EQ(rc, 0);

    // get the PDMA0 stream handle
    rc = scal_get_stream_handle_by_name(scalHandle, testConfig.second_stream_name, &D2HStreamHandle);
    ASSERT_EQ(rc, 0);

    // assign ctrlBuff2 to be the command buffer of D2HStreamHandle
    rc = scal_stream_set_commands_buffer(D2HStreamHandle, ctrlBuffHandle2);
    ASSERT_EQ(rc, 0);

    // index is from stream info
    rc = scal_stream_get_info(D2HStreamHandle, &streamInfo2);
    ASSERT_EQ(rc, 0);

    // command buffer
    rc = scal_buffer_get_info(ctrlBuffHandle2, &cmdBufferInfo2);
    ASSERT_EQ(rc, 0);

    rc = scal_buffer_get_info(dev2hostBuffHandle, &dev2hostBufferInfo);
    ASSERT_EQ(rc, 0);

    // generate command buffer
    // use pdma0 as Device To Host stream
    currCommandBufferLoc2 = (char *)cmdBufferInfo2.host_address;
    hostBuffer2 = (uint8_t *)dev2hostBufferInfo.host_address;
    memset(hostBuffer2, 0x0, testDmaBufferSize);

    //
    // copy from device back to host
    //

    rc = scal_completion_group_get_infoV2(cgHandleRx, &cgInfoRx);
    ASSERT_EQ(rc, 0);

    bool isStreamDirectMode = streamInfo2.isDirectMode;
    PqmPktUtils::sendPdmaCommand(
        isStreamDirectMode, buildPkt, currCommandBufferLoc2, (uint64_t) deviceDataBuffInfo.device_address, (uint64_t) dev2hostBufferInfo.device_address,
        testDmaBufferSize, testConfig.second_engine_group,
        -1/*workloadType*/, 0/*ctxId*/, 0/*payload*/, 0/*payloadAddr*/, 0/*bMemset*/, 0/*signal_to_cg*/, false/*wr_comp*/, 0/*completionGroupIndex*/,
        0, 0);

    currCommandBufferLoc2 += PqmPktUtils::getPdmaCmdSize(isStreamDirectMode, buildPkt, false/*wr_comp*/, 1);

    // Use zero-size PDMA with signal-to-cg as a PDMA-barrier
    PqmPktUtils::sendPdmaCommand(
        isStreamDirectMode, buildPkt, currCommandBufferLoc2,  0/*src*/, 0/*dst*/, /*size*/0, testConfig.second_engine_group/*engineGroupType*/,
        -1/*workloadType*/, 0/*ctxId*/, 0/*payload*/, 0/*payloadAddr*/, 0/*bMemset*/, 1/*signal_to_cg*/, false/*wr_comp*/, cgInfoRx.index_in_scheduler,
        cgInfoRx.long_so_sm, cgInfoRx.long_so_index);

    currCommandBufferLoc2 += PqmPktUtils::getPdmaCmdSize(isStreamDirectMode, buildPkt, false/*wr_comp*/, 1);

    // submit the command buffer on D2HStreamHandle stream
    pi = currCommandBufferLoc2 - (char *)cmdBufferInfo2.host_address;
    rc = scal_stream_submit(D2HStreamHandle, pi, streamInfo2.submission_alignment);

    // wait for completion
    printf("Waiting for stream 2 completion\n");
    rc = scal_completion_group_wait(cgHandleRx, target, SCAL_FOREVER);
    ASSERT_EQ(rc, 0);
    printf("stream 2 completed\n");

    // compare
    for(uint32_t i = 0; i < testDmaBufferSize; i++) {
        ASSERT_EQ(hostBuffer[i], hostBuffer2[i]);
    }

    printf("Clean up\n");
    // tear down
    if(ctrlBuffHandle)
    {
        rc = scal_free_buffer(ctrlBuffHandle);
        ASSERT_EQ(rc, 0);
    }
    if(ctrlBuffHandle2)
    {
        rc = scal_free_buffer(ctrlBuffHandle2);
        ASSERT_EQ(rc, 0);
    }
    if(deviceDataBuffHandle)
    {
        rc = scal_free_buffer(deviceDataBuffHandle);
        ASSERT_EQ(rc, 0);
    }
    if(host2devBuffHandle)
    {
        rc = scal_free_buffer(host2devBuffHandle);
        ASSERT_EQ(rc, 0);
    }
    if(dev2hostBuffHandle)
    {
        rc = scal_free_buffer(dev2hostBuffHandle);
        ASSERT_EQ(rc, 0);
    }

    scal_destroy(scalHandle);
    //hlthunk_close(scalFd);
    printf("Clean up Finished \n");
}

TEST_P(SCALTestDmPdma, pdma_test_host_sync)
{
    pdma_host_sync_internal();
}

#define MAX_NUM_OF_VALUES_TO_WRITE_G3 7
TEST_F_CHKDEV(SCALTestDevice, pdma_test_device_sync, {GAUDI3})
{
    int rc;
    scal_handle_t scalHandle;
    scal_pool_handle_t memHostPoolHandle;
    scal_buffer_handle_t ctrlBuffHandle = nullptr;
    scal_buffer_handle_t ctrlBuffHandle2 = nullptr;
    scal_buffer_handle_t host2devBuffHandle = nullptr;
    scal_buffer_handle_t dev2hostBuffHandle = nullptr;
    scal_stream_handle_t H2DStreamHandle  = nullptr;
    scal_stream_handle_t D2HStreamHandle  = nullptr;
    scal_buffer_handle_t deviceDataBuffHandle = nullptr;
    uint8_t *hostBuffer = nullptr;
    uint8_t *hostBuffer2 = nullptr;
    unsigned pi = 0;
    uint64_t target = 0;

    const char     configFilePath[]      = ":/default.json";
    const unsigned cCommandBufferMinSize = 64*1024;

    const unsigned testDmaBufferSize = 4096 * 10; // 1MB

    uint32_t hostCyclicBufferSize = cCommandBufferMinSize;

    int scalFd = m_fd; //hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    ASSERT_GE(scalFd, 0);
    std::string confFileStr = getConfigFilePath(configFilePath);
    const char* confFile    = confFileStr.c_str();
    printf("Loading scal with config=%s\n",confFile);
    rc = scal_init(scalFd, confFile, &scalHandle, nullptr);
    ASSERT_EQ(rc, 0);

    // check that init succeedded
    scal_comp_group_handle_t cgHandleRx;
    scal_comp_group_handle_t cgHandleTx;
    scal_stream_info_t streamInfo;
    scal_stream_info_t streamInfo2;
    scal_pool_handle_t deviceMemPoolHandle;
    scal_buffer_info_t cmdBufferInfo;
    scal_buffer_info_t host2devBufferInfo;
    scal_buffer_info_t deviceDataBuffInfo;
    scal_buffer_info_t dev2hostBufferInfo;
    scal_buffer_info_t cmdBufferInfo2;
    scal_completion_group_infoV2_t cgInfoRx;
    scal_completion_group_infoV2_t cgInfoTx;
    char *currCommandBufferLoc = nullptr;
    char *currCommandBufferLoc2 = nullptr;

    rc = scal_get_completion_group_handle_by_name(scalHandle, "pdma_rx_completion_queue0", &cgHandleRx);
    ASSERT_EQ(rc, 0);

    rc = scal_get_completion_group_handle_by_name(scalHandle, "pdma_tx_completion_queue0", &cgHandleTx);
    ASSERT_EQ(rc, 0);

    //  Get Host Memory Pool handle
    rc = scal_get_pool_handle_by_name(scalHandle, "host_shared", &memHostPoolHandle);
    ASSERT_EQ(rc, 0);

    //  Get Device Memory Pool handle
    rc = scal_get_pool_handle_by_name(scalHandle, "hbm_shared", &deviceMemPoolHandle);
    ASSERT_EQ(rc, 0);

    // Allocate Buffer on the Host for commands
    rc = scal_allocate_buffer(memHostPoolHandle, hostCyclicBufferSize, &ctrlBuffHandle);
    ASSERT_EQ(rc, 0);

    // Allocate Buffer on the Host for DMA
    rc = scal_allocate_buffer(memHostPoolHandle, testDmaBufferSize, &host2devBuffHandle);
    ASSERT_EQ(rc, 0);

    // get device data buffer
    rc = scal_allocate_buffer(deviceMemPoolHandle, testDmaBufferSize, &deviceDataBuffHandle);
    ASSERT_EQ(rc, 0);

    // get the PDMA1 stream handle (H2D  host 2 device)
    rc = scal_get_stream_handle_by_name(scalHandle, "pdma_tx0", &H2DStreamHandle);
    ASSERT_EQ(rc, 0);

    // assign ctrlBuff to be the command buffer of H2DStreamHandle
    rc = scal_stream_set_commands_buffer(H2DStreamHandle, ctrlBuffHandle);
    ASSERT_EQ(rc, 0);

    // index is from stream info
    rc = scal_stream_get_info(H2DStreamHandle, &streamInfo);
    ASSERT_EQ(rc, 0);

    // command buffer
    rc = scal_buffer_get_info(ctrlBuffHandle, &cmdBufferInfo);
    ASSERT_EQ(rc, 0);

    // Host DMA buffer
    rc = scal_buffer_get_info(host2devBuffHandle, &host2devBufferInfo);
    ASSERT_EQ(rc, 0);

    // HBM DMA buffer
    rc = scal_buffer_get_info(deviceDataBuffHandle, &deviceDataBuffInfo);
    ASSERT_EQ(rc, 0);

    hostBuffer = (uint8_t *)host2devBufferInfo.host_address;
    memset(hostBuffer, 0xCC, testDmaBufferSize); // just some value

    // generate command buffer
    currCommandBufferLoc = (char *)cmdBufferInfo.host_address;
    // test the sched_arc_cmd_pdma_batch_transfer_t command

    auto buildPkt = createBuildPkts(getScalDeviceType());

    bool isTxDirectMode = streamInfo.isDirectMode;

    rc = scal_completion_group_get_infoV2(cgHandleTx, &cgInfoTx);
    ASSERT_EQ(rc, 0);

    rc = scal_completion_group_get_infoV2(cgHandleRx, &cgInfoRx);
    ASSERT_EQ(rc, 0);

    target = 1; // tbd!

    // get data on pdma_rx stream////
    rc = scal_allocate_buffer(memHostPoolHandle, hostCyclicBufferSize, &ctrlBuffHandle2);
    ASSERT_EQ(rc, 0);

    // Allocate Buffer on the Host for DMA
    rc = scal_allocate_buffer(memHostPoolHandle, testDmaBufferSize, &dev2hostBuffHandle);
    ASSERT_EQ(rc, 0);

    // get the PDMA0 stream handle
    rc = scal_get_stream_handle_by_name(scalHandle, "pdma_rx0", &D2HStreamHandle);
    ASSERT_EQ(rc, 0);

    // assign ctrlBuff2 to be the command buffer of D2HStreamHandle
    rc = scal_stream_set_commands_buffer(D2HStreamHandle, ctrlBuffHandle2);
    ASSERT_EQ(rc, 0);

    // index is from stream info
    rc = scal_stream_get_info(D2HStreamHandle, &streamInfo2);
    ASSERT_EQ(rc, 0);

    if (!isTxDirectMode || !(streamInfo2.isDirectMode))
    {
        GTEST_SKIP() << "this test is suitable only for direct mode pdma channels";
    }

    // command buffer
    rc = scal_buffer_get_info(ctrlBuffHandle2, &cmdBufferInfo2);
    ASSERT_EQ(rc, 0);

    rc = scal_buffer_get_info(dev2hostBuffHandle, &dev2hostBufferInfo);
    ASSERT_EQ(rc, 0);

    // generate command buffer
    // use pdma0 as Device To Host stream
    currCommandBufferLoc2 = (char *)cmdBufferInfo2.host_address;
    hostBuffer2 = (uint8_t *)dev2hostBufferInfo.host_address;
    memset(hostBuffer2, 0x0, testDmaBufferSize);

    // config & arm monitor to listen on long so of pdma_tx and update the fence
    scal_monitor_pool_handle_t monPoolHandle;
    scal_monitor_pool_info     monPoolInfo;
    uint64_t longSOtargetValue = 1;

    rc = scal_get_so_monitor_handle_by_name(scalHandle, "compute_completion_queue_monitors", &monPoolHandle);
    assert(rc == 0);
    rc |= scal_monitor_pool_get_info(monPoolHandle, &monPoolInfo);
    assert(rc == 0);
    if (rc != 0) // needed for release build
    {
        LOG_ERR(SCAL, "{}: error getting monitor pool info", __FUNCTION__);
    }

    // RT should handle the allocation of monitors from monPoolInfo.baseIdx to monPoolInfo.baseIdx + monPoolInfo.size
    // for the case when multiple streams request monitors, in different threads.
    // here we just take the 1st one (index 0)
    unsigned monitorID = monPoolInfo.baseIdx;
    if (monitorID % 8 != 0)// in Gaudi3, we can relax this to (monitorID % 4 != 0)
    {
        LOG_DEBUG(SCAL, "{}: configure long monitor needs monitorID ({})to be 32 Bytes aligned, e.g monitor index diviseable by 8. adjusting", __FUNCTION__, monitorID);
        monitorID = monitorID + (8-(monitorID % 8));
    }

    uint64_t addr[MAX_NUM_OF_VALUES_TO_WRITE_G3];
    uint32_t value[MAX_NUM_OF_VALUES_TO_WRITE_G3];
    rc = gaudi3_configMonitorForLongSO(cgInfoTx.long_so_sm_base_addr,
                                       0/* no need in core handle*/,
                                       monitorID,
                                       cgInfoTx.long_so_index,
                                       false/*compareEQ*/,
                                       streamInfo2.fenceCounterAddress,
                                       PqmPktUtils::getPayloadDataFenceInc(0/*fence_id*/),
                                       addr,
                                       value);
    uint32_t numPayloads = MAX_NUM_OF_VALUES_TO_WRITE_G3;
    assert(rc == 0);

     // write the config + payload data
    for (unsigned i = 0; i < numPayloads; i++)
    {
        PqmPktUtils::buildPqmMsgLong((char*) currCommandBufferLoc2, value[i], addr[i]);
        currCommandBufferLoc2 += PqmPktUtils::getMsgLongCmdSize();
    }

    uint64_t prevLongSOtargetValue = longSOtargetValue;
    //  must send smBase of sm 5 which is where the longSO and Monitor belong
    rc = gaudi3_armMonitorForLongSO(cgInfoTx.long_so_sm_base_addr, monitorID, cgInfoTx.long_so_index,
                                    longSOtargetValue, prevLongSOtargetValue, false/*compareEQ*/, addr, value, numPayloads);
     if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SCAL, "{}: error arming longSo monitor", __FUNCTION__);
        assert(0);
    }
    // write only the arm data
    for (unsigned i = 0; i < numPayloads; i++)
    {
        PqmPktUtils::buildPqmMsgLong((char*) currCommandBufferLoc2, value[i], addr[i]);
        currCommandBufferLoc2 += PqmPktUtils::getMsgLongCmdSize();
    }

    // send fence packet to pdma_rx
    PqmPktUtils::buildPqmFenceCmd((uint8_t*)currCommandBufferLoc2, 0/*fence_id*/, 1, target);
    currCommandBufferLoc2 += PqmPktUtils::getFenceCmdSize();

    //
    // Now - copy the data back ( from device to host,  and compare)
    //

    // Allocate Buffer on the Host for commands

    //
    // copy from device back to host
    //

    bool isStreamDirectMode = streamInfo2.isDirectMode;
    PqmPktUtils::sendPdmaCommand(
        isStreamDirectMode, buildPkt, currCommandBufferLoc2, (uint64_t) deviceDataBuffInfo.device_address, (uint64_t) dev2hostBufferInfo.device_address,
        testDmaBufferSize, SCAL_PDMA_RX_GROUP,
        -1/*workloadType*/, 0/*ctxId*/, 0/*payload*/, 0/*payloadAddr*/, 0/*bMemset*/, 0/*signal_to_cg*/, false/*wr_comp*/, 0/*completionGroupIndex*/,
        0, 0);

    currCommandBufferLoc2 += PqmPktUtils::getPdmaCmdSize(isStreamDirectMode, buildPkt, false/*wr_comp*/, 1);

    // Use zero-size PDMA with signal-to-cg as a PDMA-barrier
    PqmPktUtils::sendPdmaCommand(
        isStreamDirectMode, buildPkt, currCommandBufferLoc2,  0/*src*/, 0/*dst*/, /*size*/0, SCAL_PDMA_RX_GROUP/*engineGroupType*/,
        -1/*workloadType*/, 0/*ctxId*/, 0/*payload*/, 0/*payloadAddr*/, 0/*bMemset*/, 1/*signal_to_cg*/, false/*wr_comp*/, cgInfoRx.index_in_scheduler,
        0, 0);

    currCommandBufferLoc2 += PqmPktUtils::getPdmaCmdSize(isStreamDirectMode, buildPkt, false/*wr_comp*/, 1);

    // submit the command buffer on D2HStreamHandle stream
    pi = currCommandBufferLoc2 - (char *)cmdBufferInfo2.host_address;
    rc = scal_stream_submit(D2HStreamHandle, pi, streamInfo2.submission_alignment);

    // copy data and at the end wr_comp to long so of pdma_tx
    PqmPktUtils::sendPdmaCommand(
        isTxDirectMode, buildPkt, currCommandBufferLoc, (uint64_t) host2devBufferInfo.device_address, (uint64_t) deviceDataBuffInfo.device_address,
        testDmaBufferSize, SCAL_PDMA_TX_DATA_GROUP,
        -1/*workloadType*/, 0/*ctxId*/, 0/*payload*/, 0/*payloadAddr*/, 0/*bMemset*/, 0/*signal_to_cg*/,true/*wr_comp*/, 0/*completionGroupIndex*/,
        cgInfoTx.long_so_sm, cgInfoTx.long_so_index);

    currCommandBufferLoc += PqmPktUtils::getPdmaCmdSize(isTxDirectMode, buildPkt, true/*wr_comp*/, 1);

    // submit the command buffer on H2DStreamHandle
    pi = currCommandBufferLoc - (char *)cmdBufferInfo.host_address;
    rc = scal_stream_submit(H2DStreamHandle, pi, streamInfo.submission_alignment);
    ASSERT_EQ(rc, 0);

    // wait for completion
    printf("Waiting for stream 2 completion\n");
    rc = scal_completion_group_wait(cgHandleRx, target, SCAL_FOREVER);
    ASSERT_EQ(rc, 0);
    printf("stream 2 completed\n");

    // compare
    for(uint32_t i = 0; i < testDmaBufferSize; i++) {
        ASSERT_EQ(hostBuffer[i], hostBuffer2[i]);
    }

    printf("Clean up\n");
    // tear down
    if(ctrlBuffHandle)
    {
        rc = scal_free_buffer(ctrlBuffHandle);
        ASSERT_EQ(rc, 0);
    }
    if(ctrlBuffHandle2)
    {
        rc = scal_free_buffer(ctrlBuffHandle2);
        ASSERT_EQ(rc, 0);
    }
    if(deviceDataBuffHandle)
    {
        rc = scal_free_buffer(deviceDataBuffHandle);
        ASSERT_EQ(rc, 0);
    }
    if(host2devBuffHandle)
    {
        rc = scal_free_buffer(host2devBuffHandle);
        ASSERT_EQ(rc, 0);
    }
    if(dev2hostBuffHandle)
    {
        rc = scal_free_buffer(dev2hostBuffHandle);
        ASSERT_EQ(rc, 0);
    }

    scal_destroy(scalHandle);
    //hlthunk_close(scalFd);
    printf("Clean up Finished \n");
}


TEST_F_CHKDEV(SCALTestDevice, test_device_sync_between_pdma_and_compute, {GAUDI3})
{
    int rc;
    scal_handle_t scalHandle;
    scal_pool_handle_t memHostPoolHandle;
    scal_buffer_handle_t ctrlBuffHandle = nullptr;
    scal_buffer_handle_t ctrlComputeBuffHandle = nullptr;
    scal_buffer_handle_t host2devBuffHandle = nullptr;
    scal_stream_handle_t H2DStreamHandle  = nullptr;
    scal_stream_handle_t CmptStreamHandle  = nullptr;
    scal_buffer_handle_t deviceDataBuffHandle = nullptr;
    uint8_t *hostBuffer = nullptr;
    unsigned pi = 0;
    uint64_t target = 0;

    const char     configFilePath[]      = ":/default.json";
    const unsigned cCommandBufferMinSize = 64*1024;

    const unsigned testDmaBufferSize = 4096 * 10; // 1MB

    uint32_t hostCyclicBufferSize = cCommandBufferMinSize;

    int scalFd = m_fd; //hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    ASSERT_GE(scalFd, 0);
    std::string confFileStr = getConfigFilePath(configFilePath);
    const char* confFile    = confFileStr.c_str();
    printf("Loading scal with config=%s\n",confFile);
    rc = scal_init(scalFd, confFile, &scalHandle, nullptr);
    ASSERT_EQ(rc, 0);

    // check that init succeedded
    scal_comp_group_handle_t cgHandleTx;
    scal_comp_group_handle_t cgHandleCompute;
    scal_stream_info_t streamInfo;
    scal_stream_info_t streamInfoCompute;
    scal_pool_handle_t deviceMemPoolHandle;
    scal_buffer_info_t cmdBufferInfo;
    scal_buffer_info_t host2devBufferInfo;
    scal_buffer_info_t deviceDataBuffInfo;
    scal_buffer_info_t cmdBufferComputeInfo;
    scal_completion_group_infoV2_t cgInfoTx;
    scal_completion_group_infoV2_t cgInfoCompute;
    char* currCommandBufferLoc = nullptr;
    char* cmdBuffLocCompute    = nullptr;

    rc = scal_get_completion_group_handle_by_name(scalHandle, "pdma_tx_completion_queue0", &cgHandleTx);
    ASSERT_EQ(rc, 0);

    rc = scal_get_completion_group_handle_by_name(scalHandle, "compute_completion_queue0", &cgHandleCompute);
    ASSERT_EQ(rc, 0);

    //  Get Host Memory Pool handle
    rc = scal_get_pool_handle_by_name(scalHandle, "host_shared", &memHostPoolHandle);
    ASSERT_EQ(rc, 0);

    //  Get Device Memory Pool handle
    rc = scal_get_pool_handle_by_name(scalHandle, "hbm_shared", &deviceMemPoolHandle);
    ASSERT_EQ(rc, 0);

    // Allocate Buffer on the Host for commands
    rc = scal_allocate_buffer(memHostPoolHandle, hostCyclicBufferSize, &ctrlBuffHandle);
    ASSERT_EQ(rc, 0);

    // Allocate Buffer on the Host for DMA
    rc = scal_allocate_buffer(memHostPoolHandle, testDmaBufferSize, &host2devBuffHandle);
    ASSERT_EQ(rc, 0);

    // Allocate Buffer on the compute stream
    rc = scal_allocate_buffer(memHostPoolHandle, hostCyclicBufferSize, &ctrlComputeBuffHandle);
    ASSERT_EQ(rc, 0);

    // get device data buffer
    rc = scal_allocate_buffer(deviceMemPoolHandle, testDmaBufferSize, &deviceDataBuffHandle);
    ASSERT_EQ(rc, 0);

    // get the PDMA1 stream handle (H2D  host 2 device)
    rc = scal_get_stream_handle_by_name(scalHandle, "pdma_tx0", &H2DStreamHandle);
    ASSERT_EQ(rc, 0);

    // assign ctrlBuff to be the command buffer of H2DStreamHandle
    rc = scal_stream_set_commands_buffer(H2DStreamHandle, ctrlBuffHandle);
    ASSERT_EQ(rc, 0);

    // index is from stream info
    rc = scal_stream_get_info(H2DStreamHandle, &streamInfo);
    ASSERT_EQ(rc, 0);

    //get compute stream handle
    rc = scal_get_stream_handle_by_name(scalHandle, "compute0", &CmptStreamHandle);
    ASSERT_EQ(rc, 0);

    //assign the cmd buffer to compute stream
    rc = scal_stream_set_commands_buffer(CmptStreamHandle, ctrlComputeBuffHandle);
    ASSERT_EQ(rc, 0);

    rc = scal_stream_get_info(CmptStreamHandle, &streamInfoCompute);
    ASSERT_EQ(rc, 0);

    // command buffer
    rc = scal_buffer_get_info(ctrlBuffHandle, &cmdBufferInfo);
    ASSERT_EQ(rc, 0);

    // Host DMA buffer
    rc = scal_buffer_get_info(host2devBuffHandle, &host2devBufferInfo);
    ASSERT_EQ(rc, 0);

    // HBM DMA buffer
    rc = scal_buffer_get_info(deviceDataBuffHandle, &deviceDataBuffInfo);
    ASSERT_EQ(rc, 0);

    hostBuffer = (uint8_t *)host2devBufferInfo.host_address;
    memset(hostBuffer, 0xCC, testDmaBufferSize); // just some value

    // generate command buffer
    currCommandBufferLoc = (char *)cmdBufferInfo.host_address;
    // test the sched_arc_cmd_pdma_batch_transfer_t command

    auto buildPkt = createBuildPkts(getScalDeviceType());

    bool isTxDirectMode = streamInfo.isDirectMode;

    rc = scal_completion_group_get_infoV2(cgHandleTx, &cgInfoTx);
    ASSERT_EQ(rc, 0);

    rc = scal_completion_group_get_infoV2(cgHandleCompute, &cgInfoCompute);
    ASSERT_EQ(rc, 0);

    target = 1; // tbd!

    // get info on compute stream buffer
    rc = scal_buffer_get_info(ctrlComputeBuffHandle, &cmdBufferComputeInfo);
    ASSERT_EQ(rc, 0);

    cmdBuffLocCompute = (char*) cmdBufferComputeInfo.host_address;

    // config & arm monitor to listen on long so of pdma_tx and update the fence
    scal_monitor_pool_handle_t monPoolHandle;
    scal_monitor_pool_info     monPoolInfo;
    uint64_t longSOtargetValue = 1;

    rc = scal_get_so_monitor_handle_by_name(scalHandle, "compute_completion_queue_monitors", &monPoolHandle);
    assert(rc == 0);
    rc |= scal_monitor_pool_get_info(monPoolHandle, &monPoolInfo);
    assert(rc == 0);
    if (rc != 0) // needed for release build
    {
        LOG_ERR(SCAL, "{}: error getting monitor pool info", __FUNCTION__);
    }

    // RT should handle the allocation of monitors from monPoolInfo.baseIdx to monPoolInfo.baseIdx + monPoolInfo.size
    // for the case when multiple streams request monitors, in different threads.
    // here we just take the 1st one (index 0)
    unsigned monitorID = monPoolInfo.baseIdx;
    if (monitorID % 8 != 0)// in Gaudi3, we can relax this to (monitorID % 4 != 0)
    {
        LOG_DEBUG(SCAL, "{}: configure long monitor needs monitorID ({})to be 32 Bytes aligned, e.g monitor index diviseable by 8. adjusting", __FUNCTION__, monitorID);
        monitorID = monitorID + (8-(monitorID % 8));
    }

    scal_control_core_infoV2_t coreInfo;
    scal_control_core_get_infoV2(cgInfoCompute.scheduler_handle, &coreInfo);

    uint64_t addr[MAX_NUM_OF_VALUES_TO_WRITE_G3];
    uint32_t value[MAX_NUM_OF_VALUES_TO_WRITE_G3];
    rc = gaudi3_configMonitorForLongSO(cgInfoTx.long_so_sm_base_addr,
                                       cgInfoCompute.scheduler_handle,
                                       monitorID,
                                       cgInfoTx.long_so_index,
                                       false/*compareEQ*/,
                                       coreInfo.dccm_message_queue_address,
                                       gaudi3_createPayload(0/* fenceID */),
                                       addr,
                                       value);
    uint32_t numPayloads = MAX_NUM_OF_VALUES_TO_WRITE_G3;
    assert(rc == 0);

     // write the config + payload data
    for (unsigned i = 0; i < numPayloads; i++)
    {
        fillPktNoSize<LbwWritePkt>(buildPkt, cmdBuffLocCompute, (uint32_t)addr[i], value[i], false);
        cmdBuffLocCompute += getPktSize<LbwWritePkt>(buildPkt);
    }

    uint64_t prevLongSOtargetValue = longSOtargetValue;
    //  must send smBase of sm 5 which is where the longSO and Monitor belong
    rc = gaudi3_armMonitorForLongSO(cgInfoTx.long_so_sm_base_addr, monitorID, cgInfoTx.long_so_index,
                                    longSOtargetValue, prevLongSOtargetValue, false/*compareEQ*/, addr, value, numPayloads);
     if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SCAL, "{}: error arming longSo monitor", __FUNCTION__);
        assert(0);
    }
    // write only the arm data
    for (unsigned i = 0; i < numPayloads; i++)
    {
        fillPktNoSize<LbwWritePkt>(buildPkt,  cmdBuffLocCompute, (uint32_t)addr[i], value[i], false);
        cmdBuffLocCompute += getPktSize<LbwWritePkt>(buildPkt);
    }

    // send fence packet to compute stream
    fillPktNoSize<FenceWaitPkt>(buildPkt, cmdBuffLocCompute, 0/*fence id*/, target);
    cmdBuffLocCompute += getPktSize<FenceWaitPkt>(buildPkt);

    pi = cmdBuffLocCompute - (char *)cmdBufferComputeInfo.host_address;
    rc = scal_stream_submit(CmptStreamHandle, pi, streamInfoCompute.submission_alignment);
    ASSERT_EQ(rc, 0);

    // compute stream - alloc barries & dispatch
    scal_cluster_handle_t mmeCluster;
    scal_cluster_info_t   mmeClusterInfo;
    rc = scal_get_cluster_handle_by_name(scalHandle, "mme", &mmeCluster);
    ASSERT_EQ(rc, 0);
    rc = scal_cluster_get_info(mmeCluster, &mmeClusterInfo);
    ASSERT_EQ(rc, 0);

    const EngineGroupArrayType engineGroupType {SCAL_MME_COMPUTE_GROUP, 0, 0, 0};
    fillPkt<AllocBarrierV2bPkt>(buildPkt, cmdBuffLocCompute, cgInfoCompute.index_in_scheduler, mmeClusterInfo.numCompletions, false, false/*rel_so_set*/, 1, engineGroupType, 0, 0);
    cmdBuffLocCompute += getPktSize<AllocBarrierV2bPkt>(buildPkt);

    uint8_t engine_groups[4] = {SCAL_MME_COMPUTE_GROUP, 0, 0, 0};
    fillPkt<DispatchBarrierPkt>(buildPkt, cmdBuffLocCompute, 1, engine_groups, 0);
    cmdBuffLocCompute += getPktSize<DispatchBarrierPkt>(buildPkt);

    // submit the command buffer on compute stream
    pi = cmdBuffLocCompute - (char *)cmdBufferComputeInfo.host_address;
    rc = scal_stream_submit(CmptStreamHandle, pi, streamInfoCompute.submission_alignment);
    ASSERT_EQ(rc, 0);

    // copy data and at the end wr_comp to long so of pdma_tx
    uint32_t signalToCg = 0;
    if (!isTxDirectMode)
    { // if the non compute stream is not direct mode we need to increase the long so
        signalToCg = 1;
    }
    PqmPktUtils::sendPdmaCommand(
        isTxDirectMode, buildPkt, currCommandBufferLoc, (uint64_t) host2devBufferInfo.device_address, (uint64_t) deviceDataBuffInfo.device_address,
        testDmaBufferSize, SCAL_PDMA_TX_DATA_GROUP,
        -1/*workloadType*/, 0/*ctxId*/, 0/*payload*/, 0/*payloadAddr*/, 0/*bMemset*/, signalToCg, true/*wr_comp*/, 0/*completionGroupIndex*/,
        cgInfoTx.long_so_sm, cgInfoTx.long_so_index);

    currCommandBufferLoc += PqmPktUtils::getPdmaCmdSize(isTxDirectMode, buildPkt, true/*wr_comp*/, 1);

    // submit the command buffer on H2DStreamHandle
    pi = currCommandBufferLoc - (char *)cmdBufferInfo.host_address;
    rc = scal_stream_submit(H2DStreamHandle, pi, streamInfo.submission_alignment);
    ASSERT_EQ(rc, 0);

    // add waiting on host to compute cq_counter to be incremented
    printf("Waiting for compute stream completion\n");
    rc = scal_completion_group_wait(cgHandleCompute, target, SCAL_FOREVER);
    ASSERT_EQ(rc, 0);
    printf("compute stream completed\n");

    printf("Clean up\n");
    // tear down
    if(ctrlBuffHandle)
    {
        rc = scal_free_buffer(ctrlBuffHandle);
        ASSERT_EQ(rc, 0);
    }

    if (ctrlComputeBuffHandle)
    {
        rc = scal_free_buffer(ctrlComputeBuffHandle);
        ASSERT_EQ(rc, 0);
    }

    if(deviceDataBuffHandle)
    {
        rc = scal_free_buffer(deviceDataBuffHandle);
        ASSERT_EQ(rc, 0);
    }
    if(host2devBuffHandle)
    {
        rc = scal_free_buffer(host2devBuffHandle);
        ASSERT_EQ(rc, 0);
    }

    scal_destroy(scalHandle);
    //hlthunk_close(scalFd);
    printf("Clean up Finished \n");
}