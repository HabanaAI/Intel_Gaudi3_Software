#include "scal_basic_test.h"
#include "scal.h"
#include "hlthunk.h"
#include "logger.h"
#include "wscal.h"
#include "scal_test_utils.h"
#include "scal_tests_helper.h"


// scal related
static const unsigned cCommandBufferMinSize = 64*1024;// 64K. This is the minimum size
static const unsigned hostCyclicBufferSize = cCommandBufferMinSize;

// test related
const unsigned testDmaBufferSize =  1024 * 1024; // 1 MB
// each loop includes 4 lpdma commands (24 bytes), so we'll have 48 bytes each loop
// 65536 / 96 = 682, so MAX_LOOPS needs to be greater than that
#define MAX_LOOPS 700

class SCAL_PDMA_Test : public SCALTestDevice
{
    public:
        enum PDMADirection { Host2Device, Device2Host};

    protected:
        int getScalPoolNStreamHandles();
        int getTestRelatedHandles();
        int submitPDMAcommand(uint64_t srcDevAddr,uint64_t destDevAddr, uint32_t size, PDMADirection direction, bool waitForCompletion = true);
        int releaseBuffers();
        void pdma_test(unsigned numLoops);
        void pdma_testOverflow(unsigned numLoops, unsigned initialCI);
        void pdma_loop(unsigned numLoops, uint8_t* hostBuffer, uint8_t* hostBuffer2, bool waitForCompletion = true);
        //
        //  pool,stream related handles
        //
        scal_handle_t m_scalHandle = nullptr;                        // main scal handle
        scal_pool_handle_t m_hostMemPoolHandle = nullptr;                 // handle for the "host shared" pool for the ctrl cmd buffer (memory is shared between host and arcs)
        scal_pool_handle_t m_deviceMemPoolHandle = nullptr;               // handle for the "global_hbm" pool for the user data on the device
        scal_buffer_handle_t m_ctrlCmdBuffHandle = nullptr;                 // our ctrl cmd buffer handle in the "host shared" pool
        scal_stream_handle_t m_computeStreamHandle  = nullptr;              // stream handle
        // sync related handles
        scal_comp_group_handle_t m_completionGroupHandle = nullptr;             // handle of the completion group we use with our stream
        // stream , cmd and cg buffer info
        scal_stream_info_t m_streamInfo;
        scal_buffer_info_t m_cmdBufferInfo;
        scal_completion_group_infoV2_t m_completionGroupInfo;



        // test related handles
        scal_buffer_handle_t m_deviceDataBuffHandle = nullptr;
        // test related buffers info
        scal_buffer_info_t m_deviceDataBuffInfo;
        uint8_t* m_hostBuf1 = nullptr;
        uint8_t* m_hostBuf2 = nullptr;
        uint64_t m_hostBufferDeviceAddr1 = 0;
        uint64_t m_hostBufferDeviceAddr2 = 0;
        //
        uint64_t m_target = 0;
        SchedCmd m_cmd;// handles scheduler commands
};

///////////////////////////////////////////////////////////////////////////////////////////
int SCAL_PDMA_Test::getScalPoolNStreamHandles()
{
    int rc = scal_get_completion_group_handle_by_name(m_scalHandle, "pdma_tx_completion_queue0", &m_completionGroupHandle);
    assert(rc==0);

    //  Get Shared Host Memory Pool handle for scheduler command buffers
    rc = scal_get_pool_handle_by_name(m_scalHandle, "host_shared", &m_hostMemPoolHandle);
    assert(rc==0);

    //  Get Device Global HBM Memory Pool handle
    rc = scal_get_pool_handle_by_name(m_scalHandle, "global_hbm", &m_deviceMemPoolHandle);
    assert(rc==0);

    // Allocate Buffer on the Host shared pool for the commands cyclic buffer for our stream
    rc = scal_allocate_buffer(m_hostMemPoolHandle, hostCyclicBufferSize, &m_ctrlCmdBuffHandle);
    assert(rc==0);

    // get the compute stream handle
    rc = scal_get_stream_handle_by_name(m_scalHandle, "pdma_tx0",  &m_computeStreamHandle);
    assert(rc==0);

    // set compute stream priority
    rc = scal_stream_set_priority(m_computeStreamHandle, SCAL_LOW_PRIORITY_STREAM); // e.g user data stream
    assert(rc==0);

    // assign ctrlCmdBuffHandle to be the scheduler command buffer of our compute stream
    rc = scal_stream_set_commands_buffer(m_computeStreamHandle, m_ctrlCmdBuffHandle);
    assert(rc==0);

    // stream info that we need for submission
    rc = scal_stream_get_info(m_computeStreamHandle, &m_streamInfo);
    assert(rc==0);

    rc = scal_completion_group_get_infoV2(m_completionGroupHandle, &m_completionGroupInfo);
    assert(rc==0);

    // scheduler ctrl command buffer info.  used to get the host/device address
    rc = scal_buffer_get_info(m_ctrlCmdBuffHandle, &m_cmdBufferInfo);
    assert(rc==0);

    return rc;
}

int SCAL_PDMA_Test::getTestRelatedHandles()
{
    int rc = 0;
    // we need 2 buffers, host and device, to copy "user" data

    // Allocate DEST Buffer on the Device for DMA
    //     it should be on the "global hbm" pool
    rc = scal_allocate_buffer(m_deviceMemPoolHandle, testDmaBufferSize, &m_deviceDataBuffHandle);
    assert(rc==0);

    // Dest (HBM) buffer  info
    rc = scal_buffer_get_info(m_deviceDataBuffHandle, &m_deviceDataBuffInfo);
    assert(rc==0);

    // Allocate buffers on the Host
    m_hostBuf1 = new uint8_t[testDmaBufferSize];
    m_hostBuf2 = new uint8_t[testDmaBufferSize];
    if (!m_hostBuf1 || !m_hostBuf2)
    {
        LOG_ERR(SCAL,"{}:error allocating host buffers size={}" ,__FUNCTION__,  testDmaBufferSize );
        return SCAL_FAILURE;
    }
    // map the buffers to have a device address too
    m_hostBufferDeviceAddr1 = hlthunk_host_memory_map(scal_get_fd(m_scalHandle), m_hostBuf1,0, testDmaBufferSize);
    assert(m_hostBufferDeviceAddr1 != 0);
    m_hostBufferDeviceAddr2 = hlthunk_host_memory_map(scal_get_fd(m_scalHandle), m_hostBuf2,0, testDmaBufferSize);
    assert(m_hostBufferDeviceAddr2 != 0);
    return rc;
}

int SCAL_PDMA_Test::submitPDMAcommand(uint64_t srcDevAddr,uint64_t destDevAddr, uint32_t size, PDMADirection direction, bool waitForCompletion)
{
    int rc = 0;

    unsigned group = SCAL_PDMA_TX_DATA_GROUP;
    if (direction == SCAL_PDMA_Test::Device2Host) group = SCAL_PDMA_RX_GROUP;

    bool isDirectMode = m_streamInfo.isDirectMode;

    // Use sched_arc_cmd_pdma_batch_transfer_t command
    //     note that we used device addresses since this would be executed on the device
    m_cmd.PdmaTransferCmd(isDirectMode, destDevAddr, srcDevAddr, size, group);

    // Use zero-size PDMA with signal-to-cg as a PDMA-barrier
    scal_completion_group_infoV2_t cgInfo;
    rc = scal_completion_group_get_infoV2(m_completionGroupHandle, &cgInfo);
    assert(rc==0);
    m_cmd.PdmaTransferCmd(isDirectMode,
                          0 /* dst */,
                          0 /* src */,
                          0 /* size */,
                          group,
                          -1 /* workload_type */,
                          0 /* payload */,
                          0 /* pay_addr */,
                          1 /* signal_to_cg */,
                          cgInfo.index_in_scheduler);

    // submit the command buffer on our stream
    rc = scal_stream_submit(m_computeStreamHandle, m_cmd.getPi3(), m_streamInfo.submission_alignment);
    assert(rc==0);
    m_target++;
    if (waitForCompletion)
    {
        // wait for completion
        LOG_DEBUG(SCAL, "Waiting for stream completion. target={}", m_target);
        rc = scal_completion_group_wait(m_completionGroupHandle, m_target, SCAL_FOREVER);
        assert(rc==0);

        LOG_DEBUG(SCAL, "stream completed. size = {}", size);
    }

    return rc;
}

int SCAL_PDMA_Test::releaseBuffers()
{
    int rc = 0;
    // tear down
    LOG_DEBUG(SCAL, "Clean up");
    if (m_ctrlCmdBuffHandle)
    {
        rc = scal_free_buffer(m_ctrlCmdBuffHandle);
        assert(rc == 0);
    }
    if (m_deviceDataBuffHandle)
    {
        rc = scal_free_buffer(m_deviceDataBuffHandle);
        assert(rc == 0);
    }
    if (m_hostBufferDeviceAddr1)
    {
        rc = hlthunk_memory_unmap(scal_get_fd(m_scalHandle), m_hostBufferDeviceAddr1);
        assert(rc == 0);
    }
    if (m_hostBufferDeviceAddr2)
    {
        rc = hlthunk_memory_unmap(scal_get_fd(m_scalHandle), m_hostBufferDeviceAddr2);
        assert(rc == 0);
    }
    delete[] m_hostBuf1;
    delete[] m_hostBuf2;
    return rc;
}

void SCAL_PDMA_Test::pdma_loop(unsigned numLoops, uint8_t* hostBuffer, uint8_t* hostBuffer2, bool waitForCompletion)
{
    for (unsigned i=0;i<numLoops;i++)
    {
        if (!waitForCompletion && (i == numLoops-10))
        {
            // when waitForCompletion is false it creates a backlog of requests
            // but we'd like to also check for correctness, so
            // on the last 10 loops, DO wait and check results
            waitForCompletion = true;
        }
        //
        // Now - copy the data from host to device
        //
        submitPDMAcommand(m_hostBufferDeviceAddr1, (uint64_t)m_deviceDataBuffInfo.device_address, testDmaBufferSize, SCAL_PDMA_Test::Host2Device, waitForCompletion);

        if (!waitForCompletion)
            continue;
        //
        // Now - copy the data back ( from device to host)
        //
        memset(hostBuffer2, 0x0, testDmaBufferSize);
        submitPDMAcommand((uint64_t)m_deviceDataBuffInfo.device_address, m_hostBufferDeviceAddr2, testDmaBufferSize, SCAL_PDMA_Test::Device2Host, waitForCompletion);
        //
        // compare
        //
        for(uint32_t i = 0; i < testDmaBufferSize; i++) {
            ASSERT_EQ(hostBuffer[i], hostBuffer2[i]) << "failed buffer compare hostBuffer[" << i << "]=" << (uint32_t)hostBuffer[i] << " hostBuffer2[" << i << "]=" << (uint32_t)hostBuffer2[i];
        }
    }
    printf("pdma loop done. (numLoops=%d)\n",numLoops);
}

void SCAL_PDMA_Test::pdma_test(unsigned numLoops)
{
    int rc;

    const char     configFilePath[]  = ":/default.json";

    //
    //   get fd from hlthunk and init_scal
    //
    int scalFd = m_fd; //hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    ASSERT_GE(scalFd, 0);
    std::string confFileStr = getConfigFilePath(configFilePath);
    const char* confFile    = confFileStr.c_str();

    rc = scal_init(scalFd, confFile, &m_scalHandle, nullptr);
    ASSERT_EQ(rc, 0);

    // get scal handles for pools, streams
    rc = getScalPoolNStreamHandles();
    ASSERT_EQ(rc, 0);

    // get test related buffer handles
    rc = getTestRelatedHandles();
    ASSERT_EQ(rc, 0);

    //  init our SRC (host) buffer data
    uint8_t* hostBuffer = m_hostBuf1;
    for(unsigned i=0;i<testDmaBufferSize;i++)
        *hostBuffer++ = (i % 0xFF); // just some values

    // start putting scheduler commands into the scheduler command buffer of our stream
    // and submit them. Then wait for completion
    m_cmd.Init((char *)m_cmdBufferInfo.host_address, hostCyclicBufferSize, m_streamInfo.command_alignment, getScalDeviceType());
    m_cmd.AllowBufferReset(); // since we do wait for completion before adding new commands, assume you can start from the beginning of the buffer
    hostBuffer = m_hostBuf1;
    uint8_t* hostBuffer2 = m_hostBuf2;

    pdma_loop(numLoops, hostBuffer, hostBuffer2, true);

    releaseBuffers();

    scal_destroy(m_scalHandle);
}

void SCAL_PDMA_Test::pdma_testOverflow(unsigned numLoops, unsigned initialCI)
{
    int rc;

    const char     configFilePath[]  = ":/default.json";

    //
    //   get fd from hlthunk and init_scal
    //
    int scalFd = m_fd; //hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    ASSERT_GE(scalFd, 0);
    std::string confFileStr = getConfigFilePath(configFilePath);
    const char* confFile    = confFileStr.c_str();

    rc = scal_init(scalFd, confFile, &m_scalHandle, nullptr);
    ASSERT_EQ(rc, 0);

    // get scal handles for pools, streams
    rc = getScalPoolNStreamHandles();
    ASSERT_EQ(rc, 0);

    // get test related buffer handles
    rc = getTestRelatedHandles();
    ASSERT_EQ(rc, 0);

    //  init our SRC (host) buffer data
    uint8_t* hostBuffer = m_hostBuf1;
    for(unsigned i=0;i<testDmaBufferSize;i++)
        *hostBuffer++ = (i % 0xFF); // just some values

    // start putting scheduler commands into the scheduler command buffer of our stream
    // and submit them. Then wait for completion
    m_cmd.Init((char *)m_cmdBufferInfo.host_address, hostCyclicBufferSize, m_streamInfo.command_alignment, getScalDeviceType());
    m_cmd.AllowBufferReset(); // since we do wait for completion before adding new commands, assume you can start from the beginning of the buffer
    hostBuffer = m_hostBuf1;
    uint8_t* hostBuffer2 = m_hostBuf2;

    PDMA_internals_helper pdma_internals;
    if (initialCI != 0)
    {
        // to test PI 32 bit overflow scenarios (HW cannot handle PI < CI)

        unsigned qid = pdma_internals.getQidFromChannelId(m_streamInfo.index);
        printf("set ci and pi to 0x%x on channel %d name %s\n", initialCI, qid, m_streamInfo.name);
        pdma_internals.init_baseA_block(scalFd, qid);
        pdma_internals.set_CI(initialCI);
        m_cmd.setPi(initialCI);
    }
    bool waitForCompletion = (initialCI == 0);// e.g. don't wait for completion when testing for overflow, on the contrary, we'd like to create a backlog of requests

    // due to the pressure of requests, it should use the ASYNC thread handling mechanism
    pdma_loop(numLoops, hostBuffer, hostBuffer2, waitForCompletion);

    if (initialCI != 0)
    {
        // test PI 32 bit overflow scenarios (HW cannot handle PI < CI)

        // Do a 2nd overflow, this time, it will use the BUSYWAIT overflow  handling

        unsigned qid = pdma_internals.getQidFromChannelId(m_streamInfo.index);
        uint64_t curPi = m_cmd.getPi3();
        // set pi to a larger value that will still correspond to the same place in the buffer
        uint32_t bufferSize = cCommandBufferMinSize;
        // what hw does is uint32_t hwPiRemainder = m_hwPi & (m_buffSize - 1);
        uint32_t newCi = curPi | ~(bufferSize - 1);
        printf("set ci and pi to 0x%x on channel %d\n", newCi, qid);
        pdma_internals.set_CI(newCi);
        m_cmd.setPi(newCi - bufferSize);
        pdma_loop(MAX_LOOPS, hostBuffer, hostBuffer2, true);
    }
    releaseBuffers();

    scal_destroy(m_scalHandle);
}

TEST_F_CHKDEV(SCAL_PDMA_Test, new_pdma_test_loop10,{ALL})
{
    pdma_test(10);
}

TEST_F_CHKDEV(SCAL_PDMA_Test, gaudi3_new_pdma_test_wrap_buff,{ALL})
{
    pdma_test(MAX_LOOPS);// MAX_LOOPS;
}

TEST_F_CHKDEV(SCAL_PDMA_Test, gaudi3_new_pdma_test_32bit_ovf,{GAUDI3})
{
    pdma_testOverflow(MAX_LOOPS*3, 0xFFFF0000);// cause a 32 bit PI overflow
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct TestParams2
{
    unsigned engineGroup;
    std::vector<std::string> streams;
    std::vector<std::string> cgs;
    std::vector<std::string> clusters;
    uint32_t loopCount;
    bool use_signal_to_cg;
};

class SCAL_PDMA2_Test : public SCALTest, public testing::WithParamInterface<TestParams2>
{
    public:
        int run_pdma_test(struct TestParams2 *params);
    protected:
        uint64_t m_target = 0;
        SchedCmd m_cmd;// handles scheduler commands
        struct TestParams2 *m_params = nullptr;
};

INSTANTIATE_TEST_SUITE_P(, SCAL_PDMA2_Test, testing::Values(
    TestParams2({.engineGroup = SCAL_PDMA_RX_DEBUG_GROUP, .streams = {"compute0"},            .cgs = {"compute_completion_queue0"},            .clusters = {"pdma_rx_debug"},      .loopCount = 100, .use_signal_to_cg = false}),
    TestParams2({.engineGroup = SCAL_PDMA_RX_DEBUG_GROUP, .streams = {"compute0"},            .cgs = {"compute_completion_queue0"},            .clusters = {"pdma_rx_debug"},      .loopCount = 100, .use_signal_to_cg = true }),
    TestParams2({.engineGroup = SCAL_PDMA_RX_DEBUG_GROUP, .streams = {"pdma_tx0"},            .cgs = {"pdma_tx_completion_queue0"},            .clusters = {"pdma_tx"},            .loopCount = 100, .use_signal_to_cg = false}),
    TestParams2({.engineGroup = SCAL_PDMA_RX_DEBUG_GROUP, .streams = {"pdma_tx0"},            .cgs = {"pdma_tx_completion_queue0"},            .clusters = {"pdma_tx"},            .loopCount = 100, .use_signal_to_cg = true }),
    TestParams2({.engineGroup = SCAL_PDMA_TX_CMD_GROUP,   .streams = {"pdma_tx0"},            .cgs = {"pdma_tx_completion_queue0"},            .clusters = {"pdma_tx"},            .loopCount = 100, .use_signal_to_cg = false}),
    TestParams2({.engineGroup = SCAL_PDMA_TX_CMD_GROUP,   .streams = {"pdma_tx0"},            .cgs = {"pdma_tx_completion_queue0"},            .clusters = {"pdma_tx"},            .loopCount = 100, .use_signal_to_cg = true }),
    TestParams2({.engineGroup = SCAL_PDMA_RX_DEBUG_GROUP, .streams = {"pdma_rx_debug0"},      .cgs = {"pdma_rx_debug_completion_queue0"},      .clusters = {"pdma_rx_debug"},      .loopCount = 1,   .use_signal_to_cg = false}),
    TestParams2({.engineGroup = SCAL_PDMA_RX_DEBUG_GROUP, .streams = {"pdma_tx_debug0"},      .cgs = {"pdma_tx_debug_completion_queue0"},      .clusters = {"pdma_dev2dev_debug"}, .loopCount = 1,   .use_signal_to_cg = false}),
    TestParams2({.engineGroup = SCAL_PDMA_RX_DEBUG_GROUP, .streams = {"pdma_dev2dev_debug0"}, .cgs = {"pdma_dev2dev_debug_completion_queue0"}, .clusters = {"pdma_dev2dev_debug"}, .loopCount = 1,   .use_signal_to_cg = false})

)
);

int SCAL_PDMA2_Test::run_pdma_test(struct TestParams2 *params)
{
    int rc                           = 0;
    m_params                         = params;
    const unsigned testDmaBufferSize = 4096 * 16; // 1MB

    std::string configFilePath = getConfigFilePath(":/default.json");
    //
    //  Init scal , streams and completion groups
    //
    WScal wscal(configFilePath.c_str(), m_params->streams,  // streams
                m_params->cgs,         // completion groups
                m_params->clusters);   // clusters

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

    streamBundle* stream0      = wscal.getStreamX(0); // pdma0 stream
    uint32_t      target       = 1;
    uint32_t      pay_addr     = 0;
    uint32_t      signal_to_cg = (uint32_t)(m_params->use_signal_to_cg);

    for (unsigned loopIdx = 0; loopIdx < m_params->loopCount; loopIdx++)
    {
        //
        // stream 0
        //
        stream0->h_cmd.PdmaTransferCmd(stream0->h_cgInfo.isDirectMode,
                                       deviceDMAbuf.h_BufferInfo.device_address, // dest
                                       hostDMAbuf.h_BufferInfo.device_address,   // src
                                       testDmaBufferSize,                        // size
                                       m_params->engineGroup,                    // engine group type
                                       -1,                       // user or cmd
                                       0, pay_addr,
                                       signal_to_cg,
                                       stream0->h_cgInfo.index_in_scheduler,
                                       true,                                    //wr_comp
                                       &stream0->h_cgInfo                       //scal_completion_group_infoV2_t*
                                       );
        if(!signal_to_cg)
        {
            // If PDMA engine group
            if ((m_params->engineGroup == SCAL_PDMA_TX_CMD_GROUP)       ||
                (m_params->engineGroup == SCAL_PDMA_TX_DATA_GROUP)      ||
                (m_params->engineGroup == SCAL_PDMA_RX_GROUP)           ||
                (m_params->engineGroup == SCAL_PDMA_RX_DEBUG_GROUP)     ||
                (m_params->engineGroup == SCAL_PDMA_DEV2DEV_DEBUG_GROUP))
            {
                stream0->WaitOnHostOverPdmaStream(m_params->engineGroup, stream0->h_cgInfo.isDirectMode);
            }
            else
            {
                uint8_t engine_groups1[4] = {m_params->engineGroup, 0, 0, 0};
                stream0->AllocAndDispatchBarrier(1, engine_groups1);
            }
        }
        rc |= stream0->stream_submit();
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
    return rc;
}
TEST_P_CHKDEV(SCAL_PDMA2_Test, pdma_sched_without_signal_cg, {GAUDI3})
{
    TestParams2 params = GetParam();
    int rc = run_pdma_test(&params);
    ASSERT_EQ(rc, 0);
}