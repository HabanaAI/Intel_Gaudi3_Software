#include <iostream>
#include <algorithm>

#include "scal_basic_test.h"
#include "scal.h"
#include "hlthunk.h"
#include "logger.h"
#include "scal_test_utils.h"

#include "gaudi2/asic_reg_structs/qman_arc_aux_regs.h"
#include "gaudi2/asic_reg/gaudi2_blocks.h"

#include "gaudi3/asic_reg_structs/arc_acp_eng_regs.h" // block_arc_acp_eng

#include "gaudi2_arc_common_packets.h"
#include "gaudi2_arc_sched_packets.h"
#include "gaudi2_arc_common_packets.h"
#include "scal_test_pqm_pkt_utils.h"

#include "wscal.h"

#define varoffsetof(t, m) ((size_t)(&(((t*)0)->m)))

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum TestType {
    SANITY,
    COMPUTE_MEDIA_BASIC,
    COMPUTE_MEDIA_ADVANCED,
    FULL
};

struct test_params
{
    unsigned engineGroup;
    int32_t workload;
    uint32_t loopCount;
};

class SCAL_Fence_Test : public SCALTest, public testing::WithParamInterface<bool>
{
    public:
        int run_fence_test(struct test_params *params, bool forceFirmwareFence);
        int run_fence_test2(struct test_params* params, bool forceFirmwareFence);
        int run_barrier_test(struct test_params *params, bool forceFirmwareFence);
    protected:
        uint64_t m_target = 0;
        SchedCmd m_cmd;// handles scheduler commands
        struct test_params *m_params = nullptr;
};
INSTANTIATE_TEST_SUITE_P(, SCAL_Fence_Test, testing::Values(false, true) /* forceFirmwareFence */);

int SCAL_Fence_Test::run_fence_test(struct test_params *params, bool forceFirmwareFence)
{
    int rc   = 0;
    m_params = params;

    std::string configFilePath = getConfigFilePath(":/default.json");
    //
    //  Init scal , streams and completion groups
    //
    //  WScal will bind each column of stream + completion group + cluster
    //   e.g for column 0 (stream 0) below,
    //           "pdma0" will use "compute_completion_queue0" and "pdma_tx"
    //           column 1 (stream 1) "pdma1" + "compute_completion_queue1" + "pdma_rx"
    // and from the user point of view
    //     it will hide all cyclic buffer allocation and binding
    //     all completion group index and numCompletions usage in alloc barrier
    // user flow is:
    //     stream0 = wscal.getStreamX(0);
    //     stream0->h_cmd.  add some commands
    //     stream0->AllocAndDispatchBarrier
    //     stream0->stream_submit()
    //     stream0->completion_group_wait(target);
    //   where wscal will do all stream<->completion group <-> cluster bindings
    //   behind the scene.

    WScal wscal(configFilePath.c_str(),
                {"pdma_tx0", "pdma_rx0"},                                   // streams
                {"compute_completion_queue0", "compute_completion_queue1"}, // completion groups
                {"pdma_tx", "pdma_rx"},                                     // clusters
                true);                                                      // skip in case of direct mode stream
    rc = wscal.getStatus();
    assert(rc == 0 || rc == SCAL_UNSUPPORTED_TEST_CONFIG);
    if (rc)
    {
        return rc;
    }

    //
    // test the fence_wait command
    // this will suspend the pdma0 stream
    // ensure fence increment uses the same fence IDs
    //
    // stream 0
    //
    streamBundle* stream0 = wscal.getStreamX(0);// pdma0 stream

    uint32_t fence_id = 1;
    uint32_t target = 1;
    stream0->h_cmd.FenceCmd(fence_id, target, forceFirmwareFence, false /* isDirectMode */);// wait on fence id 1
    stream0->WaitOnHostOverPdmaStream(SCAL_PDMA_RX_GROUP, false /* isDirectMode */);
    rc |= stream0->stream_submit();
    assert(rc == 0);

    //
    // stream 1
    //

    streamBundle* stream1 = wscal.getStreamX(1);// pdma1 stream
    uint8_t arr_fence_id[1] = {fence_id};
    // NOTE!
    //    This solution works ONLY if both streams run on the
    //    same scheduler.
    stream1->h_cmd.FenceIncImmediate(1, arr_fence_id, forceFirmwareFence);
    stream1->WaitOnHostOverPdmaStream(SCAL_PDMA_TX_DATA_GROUP, false /* isDirectMode */);
    rc |= stream1->stream_submit();
    assert(rc == 0);

    // wait for completion of stream 1
    rc |= stream1->completion_group_wait(target);
    assert(rc == 0);

    // wait for completion of stream 0
    rc |= stream0->completion_group_wait(target);
    assert(rc == 0);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// here we make the stream2 inc_immediate even BEFORE stream1 enters the wait.
//
int SCAL_Fence_Test::run_fence_test2(struct test_params* params, bool forceFirmwareFence)
{
    int rc;
    m_params = params;

    std::string configFilePath = getConfigFilePath(":/default.json");
    WScal       wscal(configFilePath.c_str(), {"pdma_tx0", "pdma_rx0"},           // streams
                {"compute_completion_queue0", "compute_completion_queue1"}, // completion groups
                {"pdma_tx", "pdma_rx"},                                     // clusters
                true);                                                      // skip if direct mode
    rc = wscal.getStatus();
    assert(rc == 0 || rc == SCAL_UNSUPPORTED_TEST_CONFIG);
    if (rc)
    {
        return rc;
    }

    //
    // test the fence_wait command
    // this will suspend the pdma0 stream
    // ensure fence increment uses the same fence IDs

    uint32_t fence_id = 1;
    uint32_t target   = 1;

    // NOTE that here we send the FenceIncImmediate BEFORE the other stream even begins the wait
    //
    // stream 1
    //

    streamBundle* stream1         = wscal.getStreamX(1); // pdma1 stream
    uint8_t      arr_fence_id[1] = {fence_id};
    // NOTE!
    //    This solution works ONLY if both streams run on the
    //    same scheduler.
    stream1->h_cmd.FenceIncImmediate(1, arr_fence_id, forceFirmwareFence);
    stream1->WaitOnHostOverPdmaStream(SCAL_PDMA_TX_DATA_GROUP, false /* isDirectMode */);
    rc = stream1->stream_submit();
    assert(rc == 0);

    // wait for completion of stream 1
    rc = stream1->completion_group_wait(target);
    assert(rc == 0);

    //
    // stream 0
    //
    streamBundle* stream0 = wscal.getStreamX(0); // pdma0 stream

    stream0->h_cmd.FenceCmd(fence_id, target, forceFirmwareFence, false /* isDirectMode */); // wait on fence id 1
    stream0->WaitOnHostOverPdmaStream(SCAL_PDMA_RX_GROUP, false /* isDirectMode */);
    rc = stream0->stream_submit();
    assert(rc == 0);

    // wait for completion of stream 0
    rc = stream0->completion_group_wait(target);
    assert(rc == 0);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_Fence_Test::run_barrier_test(struct test_params *params, bool forceFirmwareFence)
{
    int rc                           = 0;
    m_params                         = params;
    const unsigned testDmaBufferSize = 4096 * 16; // 1MB

    std::string configFilePath = getConfigFilePath(":/default.json");
    //
    //  Init scal , streams and completion groups
    //
    WScal wscal(configFilePath.c_str(),
                {"pdma_tx0", "pdma_rx0"},                                    // streams
                {"compute_completion_queue0", "compute_completion_queue1"},  // completion groups
                {"pdma_tx", "pdma_rx"},                                      // clusters
                false,                                                       // (don't) skip in case of direct mode stream
                {"pdma_tx_completion_queue0", "pdma_rx_completion_queue0"}); //direct mode cgs

    rc = wscal.getStatus();
    assert(rc == 0);
    if (rc)
    {
        return rc;
    }

    // Host DMA buffer - our src buffer
    bufferBundle_t hostDMAbuf;
    rc |= wscal.getBufferX(WScal::HostSharedPool, testDmaBufferSize, &hostDMAbuf); // get both buffer handle and info
    assert(rc == 0);
    // HBM DMA buffer - our dest buffer
    bufferBundle_t deviceDMAbuf;
    rc |= wscal.getBufferX(WScal::devSharedPool, testDmaBufferSize, &deviceDMAbuf); // get both buffer handle and info
    assert(rc == 0);

    //
    // payload is used for synchronization between the streams
    //
    streamBundle* stream0 = wscal.getStreamX(0); // pdma0 stream
    streamBundle* stream1 = wscal.getStreamX(1); // pdma1 stream

    scal_control_core_infoV2_t coreInfo;
    scal_control_core_get_infoV2(stream0->h_cgInfo.scheduler_handle, &coreInfo);

    const uint32_t fence_id = 0;
    uint32_t       target   = 1;

    scalDeviceType deviceType     = wscal.getScalDeviceType();
    uint32_t       payloadData    = 0;
    uint64_t       payloadAddress = 0;
    switch (deviceType)
    {
        case dtGaudi2:
        {
            union g2fw::sched_mon_exp_msg_t pay_data;
            memset(&pay_data, 0 , sizeof(g2fw::sched_mon_exp_msg_t));
            pay_data.fence.opcode   = g2fw::MON_EXP_FENCE_UPDATE;
            pay_data.fence.fence_id = fence_id;
            payloadData             = pay_data.raw;

            // pay_addr = address of DCCM queue 0
            assert(coreInfo.dccm_message_queue_address);
            payloadAddress = coreInfo.dccm_message_queue_address;
        }
        break;

        case dtGaudi3:
        {
            if (stream0->h_streamInfo.isDirectMode)
            {
                payloadData    = PqmPktUtils::getPayloadDataFenceInc(fence_id);
                payloadAddress = stream0->h_streamInfo.fenceCounterAddress;
            }
            else if (forceFirmwareFence)
            {
                union g3fw::sched_mon_exp_msg_t pay_data;
                memset(&pay_data, 0, sizeof(g3fw::sched_mon_exp_msg_t));
                pay_data.fence.opcode   = g3fw::MON_EXP_FENCE_UPDATE;
                pay_data.fence.fence_id = fence_id;
                payloadData             = pay_data.raw;

                // pay_addr = address of DCCM queue 0
                assert(coreInfo.dccm_message_queue_address);
                payloadAddress = coreInfo.dccm_message_queue_address;
            }
            else
            {
                gaudi3::arc_acp_eng::reg_qsel_mask_counter maskCounter;
                maskCounter._raw  = 0;
                maskCounter.op    = 1 /* Operation is Add*/;
                maskCounter.value = 1;
                payloadData = maskCounter._raw;

                payloadAddress = SchedCmd::getArcAcpEngBaseAddr(coreInfo.hdCore) +
                                 varoffsetof(gaudi3::block_arc_acp_eng, qsel_mask_counter[fence_id]);
            }
        }
        break;

        default:
            assert(false);
    }

    // stream 0 will wait on fence id 0
    // stream 1 will do DMA and send payload to fence id 0
    // The payload addr contains address of DCCM queue and payload
    // data contains opcode to update corresponding fence
    for (unsigned loopIdx = 0; loopIdx < m_params->loopCount; loopIdx++)
    {
        //
        // stream 0
        //
        stream0->h_cmd.FenceCmd(fence_id, 1, forceFirmwareFence, stream0->h_streamInfo.isDirectMode); // wait on fence id 0
        stream0->WaitOnHostOverPdmaStream(SCAL_PDMA_RX_GROUP, stream0->h_streamInfo.isDirectMode);
        rc |= stream0->stream_submit();
        assert(rc == 0);

        //
        // stream 1
        //
        stream1->h_cmd.PdmaTransferCmd(stream1->h_streamInfo.isDirectMode,
                                       deviceDMAbuf.h_BufferInfo.device_address, // dest
                                       hostDMAbuf.h_BufferInfo.device_address,   // src
                                       testDmaBufferSize,                        // size
                                       m_params->engineGroup,                    // engine group type
                                       m_params->workload,                       // user or cmd
                                       payloadData,
                                       payloadAddress,
                                       0, 0, //signal to cg and compGrpIdx
                                       true /*wr_comp*/);
        if ((m_params->engineGroup == SCAL_PDMA_TX_CMD_GROUP)  ||
            (m_params->engineGroup == SCAL_PDMA_TX_DATA_GROUP) ||
            (m_params->engineGroup == SCAL_PDMA_RX_GROUP))
        {
            stream1->WaitOnHostOverPdmaStream(m_params->engineGroup, stream1->h_streamInfo.isDirectMode);
        }
        else
        {
            uint8_t engine_groups1[4] = {m_params->engineGroup, 0, 0, 0};
            stream1->AllocAndDispatchBarrier(1, engine_groups1);
        }
        rc |= stream1->stream_submit();
        assert(rc == 0);
        //
        // wait for completion of stream 1
        //
        rc |= stream1->completion_group_wait(target);
        assert(rc == 0);
        //
        // wait for completion of stream 0
        //
        rc |= stream0->completion_group_wait(target);
        assert(rc == 0);

        target++;
    }
    // releaseBuffers
    rc |= scal_free_buffer(hostDMAbuf.h_Buffer);
    assert(rc == 0);
    rc |= scal_free_buffer(deviceDMAbuf.h_Buffer);
    assert(rc == 0);
    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_Fence_Test, fence_test, {ALL})
{
    printf("running fence_test\n");
    struct test_params params;
    memset(&params,0,sizeof(params));
    int rc = run_fence_test(&params, GetParam());
    if (rc == SCAL_UNSUPPORTED_TEST_CONFIG)
    {
        GTEST_SKIP() << "this test does not support pdma direct mode";
    }
    ASSERT_EQ(rc, 0);
}

TEST_P_CHKDEV(SCAL_Fence_Test, multi_sched_fence_test2, {ALL})
{
    printf("running fence_test2\n");
    struct test_params params;
    memset(&params, 0, sizeof(params));
    int rc = run_fence_test2(&params, GetParam());
    if (rc == SCAL_UNSUPPORTED_TEST_CONFIG)
    {
        GTEST_SKIP() << "this test does not support pdma direct mode";
    }
    ASSERT_EQ(rc, 0);
}

TEST_P_CHKDEV(SCAL_Fence_Test, barrier_test0, {ALL})
{
    printf("running barrier_test0\n");
    struct test_params params;
    memset(&params,0,sizeof(params));
    params.engineGroup = SCAL_PDMA_RX_GROUP;
    params.workload = -1;// user data
    params.loopCount = 1;
    int rc = run_barrier_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}

TEST_P_CHKDEV(SCAL_Fence_Test, barrier_test2, {ALL})
{
    printf("running barrier_test2\n");
    struct test_params params;
    memset(&params,0,sizeof(params));
    params.engineGroup = SCAL_PDMA_TX_DATA_GROUP;
    params.workload = -1;// user data
    params.loopCount = 1;
    int rc = run_barrier_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
TEST_P_CHKDEV(SCAL_Fence_Test, barrier_test3, {ALL})
{
    printf("running barrier_test3\n");
    struct test_params params;
    memset(&params,0,sizeof(params));
    params.engineGroup = SCAL_PDMA_TX_CMD_GROUP;
    params.workload = -1;// cmd data
    params.loopCount = 1;
    int rc = run_barrier_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
TEST_P_CHKDEV(SCAL_Fence_Test, barrier_test4, {ALL})
{
    printf("running barrier_test4\n");
    struct test_params params;
    memset(&params,0,sizeof(params));
    params.engineGroup = SCAL_PDMA_TX_DATA_GROUP;
    params.workload = -1;// user data
    params.loopCount = 20;
    int rc = run_barrier_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
TEST_P_CHKDEV(SCAL_Fence_Test, barrier_test5, {ALL})
{
    printf("running barrier_test5\n");
    struct test_params params;
    memset(&params,0,sizeof(params));
    params.engineGroup = SCAL_PDMA_TX_CMD_GROUP;
    params.workload = -1;// cmd data
    params.loopCount = 20;
    int rc = run_barrier_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
