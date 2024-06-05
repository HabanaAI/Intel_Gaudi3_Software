#include "scal_basic_test.h"
#include "scal.h"
#include "hlthunk.h"
#include "logger.h"
#include "scal_test_utils.h"

// scal related
static const unsigned cCommandBufferMinSize = 64*1024;
static const unsigned hostCyclicBufferSize = cCommandBufferMinSize;
static const unsigned DynamicEcbListSize = 256;
static const unsigned StaticEcbListSize = 20*1024;
static const unsigned ecbListBufferSize = 256*16;
static const unsigned ecbBufferSize = 4*1024;



////////////////////////////////////////////////////////////////////////////
#include "gaudi2/gaudi2_packets.h"

enum QMAN_EB {
        EB_FALSE = 0,
        EB_TRUE
};

enum QMAN_MB {
        MB_FALSE = 0,
        MB_TRUE
};

static uint32_t gaudi2_qman_add_nop_pkt(void *buffer, uint32_t buf_off, enum QMAN_EB eb, enum QMAN_MB mb)
{
        struct packet_nop packet;

        memset(&packet, 0, sizeof(packet));
        packet.opcode = PACKET_NOP;
        packet.eng_barrier = eb;
        packet.msg_barrier = mb;

        packet.ctl = htole32(packet.ctl);
        memcpy((uint8_t *) buffer + buf_off, &packet, sizeof(packet));

        return buf_off + sizeof(packet);
}
////////////////////////////////////////////////////////////////////////////

enum TestType {
    SANITY,
    COMPUTE_MEDIA_BASIC,
    COMPUTE_MEDIA_ADVANCED,
    FULL
};

enum ECBListTestType {
    StaticEcbList,
    ComputeEcbList
};

struct ecblist_test_params
{
    ECBListTestType whichTest;
    unsigned    engineGroup;
    unsigned    engine_cpu_index;
    const char* streamName;
    const char* clusterName;
    unsigned    streamPriority;
    bool        copyToHBM;
};

class SCAL_Compute_ECBList_Test : public SCALTestDevice
{
    public:
        enum PDMADirection { Host2Device, Device2Host};
        int  run_ecblist_test(struct ecblist_test_params *params);
        int  submitPDMAcommand(uint64_t srcDevAddr,uint64_t destDevAddr, uint32_t size, PDMADirection direction);
    protected:
        int getScalPoolNStreamHandles();
        int releaseBuffers();
        int buildAndSendEcbs( uint32_t engine_group_id,
            uint32_t target_value, uint32_t eng_group_count, uint32_t cpu_index, bool copy_To_HBM);

        //
        //  pool,stream related handles
        //
        scal_handle_t m_scalHandle = nullptr;                               // main scal handle
        scal_pool_handle_t m_hostMemPoolHandle = nullptr;                   // handle for the "host shared" pool for the ctrl cmd buffer (memory is shared between host and arcs)
        scal_pool_handle_t m_deviceSharedMemPoolHandle = nullptr;           // handle for the "hbm_shared" pool for the user data on the device
        scal_buffer_handle_t m_ctrlCmdBuffHandle = nullptr;                 // our ctrl cmd buffer handle in the "host shared" pool
        // ecb buf & ecb_list buf
        scal_buffer_handle_t m_ecbListBuffHandleOnDevice = nullptr;         // our ecb list buffer handle in the device pool
        scal_buffer_handle_t m_ecbListBuffHandle = nullptr;                 // our ecb list buffer handle in the "host_shared" pool
        scal_buffer_handle_t m_ecbBuffHandle = nullptr;                     // our ecb buffer for QMAN commands (e.g. recipe) handle in the "host_shared" pool
        //
        scal_cluster_handle_t m_clusterHandle = nullptr;

        scal_stream_handle_t m_streamHandle  = nullptr;                     // stream handle
        // sync related handles
        scal_comp_group_handle_t m_completionGroupHandle = nullptr;             // handle of the completion group we use with our stream
        // stream & cmd buffer info
        scal_stream_info_t m_streamInfo;
        scal_buffer_info_t m_cmdBufferInfo;
        scal_buffer_info_t m_ecbListBufferInfo;
        scal_buffer_info_t m_ecbListBufferOnDeviceInfo;
        scal_buffer_info_t m_ecbBufferInfo;
        scal_cluster_info_t m_clusterInfo;
        scal_completion_group_infoV2_t m_completionGroupInfo;

        uint64_t m_target = 0;
        SchedCmd m_cmd;// handles scheduler commands
        struct ecblist_test_params *m_params = nullptr;
};

int SCAL_Compute_ECBList_Test::getScalPoolNStreamHandles()
{
    int rc = 0;
    // since we have 4 instances of "compute_completion_queue", they are named compute_completion_queue0..3
    rc = scal_get_completion_group_handle_by_name(m_scalHandle, "compute_completion_queue0", &m_completionGroupHandle);
    assert(rc==0);

    //  Get Shared Host Memory Pool handle for scheduler command buffers
    rc = scal_get_pool_handle_by_name(m_scalHandle, "host_shared", &m_hostMemPoolHandle);
    assert(rc==0);

    //  Get Device Shared HBM Memory Pool handle
    rc = scal_get_pool_handle_by_name(m_scalHandle, "hbm_shared", &m_deviceSharedMemPoolHandle);
    assert(rc==0);

    // Allocate Buffer on the Host shared pool for the commands cyclic buffer for our stream
    rc = scal_allocate_buffer(m_hostMemPoolHandle, hostCyclicBufferSize, &m_ctrlCmdBuffHandle);
    assert(rc==0);

    // Allocate Buffer on the device HBM shared pool for the ecb list
    // ecb list buf
    rc = scal_allocate_buffer(m_hostMemPoolHandle, ecbListBufferSize, &m_ecbListBuffHandle);
    assert(rc==0);
    // ecb list buf on device for copy the list and run from device
    rc = scal_allocate_buffer(m_deviceSharedMemPoolHandle, ecbListBufferSize, &m_ecbListBuffHandleOnDevice);
    assert(rc==0);
    // ecb buf
    rc = scal_allocate_buffer(m_hostMemPoolHandle, ecbBufferSize, &m_ecbBuffHandle);
    assert(rc==0);

    // get the compute stream handle
    rc = scal_get_stream_handle_by_name(m_scalHandle, m_params->streamName,  &m_streamHandle);
    assert(rc==0);

    // set compute stream priority
    rc = scal_stream_set_priority(m_streamHandle, m_params->streamPriority );// e.g user data stream
    assert(rc==0);

    // assign ctrlCmdBuffHandle to be the scheduler command buffer of our compute stream
    rc = scal_stream_set_commands_buffer(m_streamHandle, m_ctrlCmdBuffHandle);
    assert(rc==0);

    // stream info that we need for submission
    rc = scal_stream_get_info(m_streamHandle, &m_streamInfo);
    assert(rc==0);

    rc = scal_completion_group_get_infoV2(m_completionGroupHandle, &m_completionGroupInfo);
    assert(rc==0);


    // scheduler ctrl command buffer info.  used to get the host/device address
    rc = scal_buffer_get_info(m_ctrlCmdBuffHandle, &m_cmdBufferInfo);
    assert(rc==0);

    // ecb list buffer info.  used to get the host/device address
    rc = scal_buffer_get_info(m_ecbListBuffHandle, &m_ecbListBufferInfo);
    // ecb list on device buffer info.  used to get the host/device address
    rc = scal_buffer_get_info(m_ecbListBuffHandleOnDevice, &m_ecbListBufferOnDeviceInfo);
    assert(rc==0);
    // ecb buffer info.  used to get the host/device address
    rc = scal_buffer_get_info(m_ecbBuffHandle, &m_ecbBufferInfo);
    assert(rc==0);

    rc = scal_get_cluster_handle_by_name(m_scalHandle, m_params->clusterName, &m_clusterHandle);
    assert(rc==0);

    rc = scal_cluster_get_info(m_clusterHandle, &m_clusterInfo);
    assert(rc==0);

    return rc;
}

int SCAL_Compute_ECBList_Test::releaseBuffers()
{
    int rc = 0;
    printf("Clean up\n");
    // tear down
    if(m_ctrlCmdBuffHandle)
    {
        rc = scal_free_buffer(m_ctrlCmdBuffHandle);
        assert(rc==0);
    }
    if(m_ecbListBuffHandle)
    {
        rc = scal_free_buffer(m_ecbListBuffHandle);
        assert(rc==0);
    }
    if (m_ecbBuffHandle)
    {
        rc = scal_free_buffer(m_ecbBuffHandle);
        assert(rc==0);
    }
    return rc;
}


int SCAL_Compute_ECBList_Test::submitPDMAcommand(uint64_t srcDevAddr,uint64_t destDevAddr, uint32_t size, PDMADirection direction)
{
    int rc = 0;
    unsigned group = SCAL_PDMA_TX_DATA_GROUP;
    if (direction == PDMADirection::Device2Host) group = SCAL_PDMA_RX_GROUP;

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
    rc = scal_stream_submit(m_streamHandle, m_cmd.getPi3(), m_streamInfo.submission_alignment);

    assert(rc==0);
    m_target++;
    // wait for completion
    LOG_DEBUG(SCAL, "Waiting for stream completion. target={}", m_target);
    rc = scal_completion_group_wait(m_completionGroupHandle, m_target, SCAL_FOREVER);
    assert(rc==0);

    LOG_DEBUG(SCAL, "stream completed. size = {}", size);

    return rc;
}


int  SCAL_Compute_ECBList_Test::buildAndSendEcbs( uint32_t engine_group_id,
        uint32_t target_value, uint32_t eng_group_count,
        uint32_t cpu_index, bool copy_To_HBM)
{
    uint32_t ecb_static_list_size = 256;
    uint32_t addr_index = 0;  //use first buffer out of 6 buffers available
    uint32_t addr_offset = 0; //offset from the starting of the buffer
    uint32_t switch_cq = 1;   // tells FW to switch between handling dynamic commands to static command, should come at the end of each buffer

    uint8_t engine_groups2[4] = {m_params->engineGroup, 0, 0, 0};

    //
    //
    //      Dynamic ECB List   (listsize + nop)
    //
    //
    //fill engine ecb commands to the ecb list hbm buffer
    //fill dynamic list first since it is in the beginning of buffer

    // fill scheduler cmd buff

    EcbListCmd ecbList((char*)m_ecbListBufferInfo.host_address,ecbListBufferSize, getScalDeviceType());
    ecbList.ArcListSizeCmd(DynamicEcbListSize);
    ecbList.Pad(DynamicEcbListSize, switch_cq);

    //
    //
    //      Static ECB List   (lissize + ECB StaticDesc + NOP)
    //
    //
    //fill static list next

    ecbList.ArcListSizeCmd(ecb_static_list_size);
    ecbList.staticDescCmd(m_params->engine_cpu_index, ecbBufferSize, addr_offset, addr_index);
    ecbList.Pad(DynamicEcbListSize + ecb_static_list_size, switch_cq);

    //
    //   QMan Commands on the ECB Buf. This is our "recipe"
    //
    //fill engine qman commands
    uint32_t curr_buff_off = 0;
    uint8_t* ecb_buff = (uint8_t *)m_ecbBufferInfo.host_address;
    while (curr_buff_off  < ecbBufferSize) {
        curr_buff_off = gaudi2_qman_add_nop_pkt(ecb_buff, curr_buff_off, EB_FALSE, MB_FALSE);
    }

    //
    //  fill scheduler commands                       - use update_recipe_base_cmd to point to the ECB buf
    //
    m_cmd.Init((char *)m_cmdBufferInfo.host_address, hostCyclicBufferSize, m_streamInfo.command_alignment, getScalDeviceType());
    m_cmd.AllowBufferReset(); // since we do wait for completion before adding new commands, assume you can start from the beginning of the buffer
    if (copy_To_HBM) // if needed, copy ecbList to HBM
    {
        submitPDMAcommand((uint64_t)m_ecbListBufferInfo.device_address, (uint64_t)m_ecbListBufferOnDeviceInfo.device_address, ecbListBufferSize, PDMADirection::Host2Device);
    }
    uint16_t recipe_base_index = 0;
    uint64_t recipeBaseAddr = m_ecbBufferInfo.device_address;
    m_cmd.UpdateRecipeBase(&recipe_base_index, eng_group_count, engine_groups2, &recipeBaseAddr, 1);
    bool single_static_chunk = true;
    bool single_dynamic_chunk = true;
    uint32_t engine_group_type = m_params->engineGroup;
    uint32_t static_ecb_list_offset = DynamicEcbListSize; //one chunk away
    uint32_t dynamic_ecb_list_addr;
    if (copy_To_HBM)
    {
        dynamic_ecb_list_addr = m_ecbListBufferOnDeviceInfo.core_address; // read the ecbList from HBM
    }
    else
    {
        dynamic_ecb_list_addr = m_ecbListBufferInfo.core_address; // read the ecbList from host
    }
    m_cmd.DispatchComputeEcbList(engine_group_type,
                        single_static_chunk, single_dynamic_chunk,
                        static_ecb_list_offset, dynamic_ecb_list_addr);
    m_cmd.NopCmd(0);

    m_cmd.AllocBarrier(m_completionGroupInfo.index_in_scheduler, m_clusterInfo.numCompletions, 0);
    m_cmd.DispatchBarrier(eng_group_count, engine_groups2);
    // submit the command buffer on our stream
    int rc = scal_stream_submit(m_streamHandle, m_cmd.getPi3(), m_streamInfo.submission_alignment);
    assert(rc==0);

    // TBD - release the so set by using AllocBarrier with release_so = 1 (currently crashing ...)
    m_target++;
    // wait for completion
    LOG_DEBUG(SCAL, "Waiting for stream completion. target={}", m_target);
    rc = scal_completion_group_wait(m_completionGroupHandle, m_target, SCAL_FOREVER);
    assert(rc==0);
    return rc;
}

int SCAL_Compute_ECBList_Test::run_ecblist_test(struct ecblist_test_params *params)
{
    int rc;
    m_params = params;

    const char configFilePath[] = ":/default.json";

    //
    //   get fd from hlthunk and init_scal
    //
    int scalFd = m_fd;//hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    assert(scalFd>=0);
    std::string confFileStr = getConfigFilePath(configFilePath);
    const char* confFile    = confFileStr.c_str();
    printf("Loading scal with config=%s\n",confFile);

    rc = scal_init(scalFd, confFile, &m_scalHandle, nullptr);
    assert(rc==0);

    // get scal handles for pools, streams
    rc = getScalPoolNStreamHandles();
    assert(rc==0);

    //
    //   build the ecb & ecb list buffers + scheduler command buffer
    //   submit work and wait for completion
    //
    buildAndSendEcbs(m_params->engineGroup, m_clusterInfo.numCompletions, 1, m_params->engine_cpu_index, m_params->copyToHBM);

    releaseBuffers();

    scal_destroy(m_scalHandle);
    printf("Test Finished\n");
    return rc;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F_CHKDEV(SCAL_Compute_ECBList_Test, compute_ecblist_test_tpc_from_host, {ALL})
{
    printf("running TPC ComputeEcbList from host\n");
    struct ecblist_test_params params;
    params.whichTest = ComputeEcbList;
    params.engineGroup    = SCAL_TPC_COMPUTE_GROUP;
    params.engine_cpu_index =  0;
    params.streamName = "compute0";
    params.streamPriority = SCAL_HIGH_PRIORITY_STREAM;
    params.clusterName = "compute_tpc";
    params.copyToHBM = false;
    int rc = run_ecblist_test(&params);
    ASSERT_EQ(rc, 0);
}

TEST_F_CHKDEV(SCAL_Compute_ECBList_Test, compute_ecblist_test_tpc_from_device, {ALL})
{
    printf("running TPC ComputeEcbList from device\n");
    struct ecblist_test_params params;
    params.whichTest = ComputeEcbList;
    params.engineGroup    = SCAL_TPC_COMPUTE_GROUP;
    params.engine_cpu_index =  0;
    params.streamName = "compute0";
    params.streamPriority = SCAL_HIGH_PRIORITY_STREAM;
    params.clusterName = "compute_tpc";
    params.copyToHBM = true;
    int rc = run_ecblist_test(&params);
    ASSERT_EQ(rc, 0);
}

TEST_F_CHKDEV(SCAL_Compute_ECBList_Test, compute_ecblist_test_mme_from_host, {ALL})
{
    printf("running MME ComputeEcbList from host\n");
    struct ecblist_test_params params;
    params.whichTest = ComputeEcbList;
    params.engineGroup    = SCAL_MME_COMPUTE_GROUP;
    params.engine_cpu_index =  0;
    params.streamName = "compute0";
    params.streamPriority = SCAL_HIGH_PRIORITY_STREAM;
    params.clusterName = "mme";
    params.copyToHBM = false;
    int rc = run_ecblist_test(&params);
    ASSERT_EQ(rc, 0);
}

TEST_F_CHKDEV(SCAL_Compute_ECBList_Test, compute_ecblist_test_mme_from_device, {ALL})
{
    printf("running MME ComputeEcbList from device\n");
    struct ecblist_test_params params;
    params.whichTest = ComputeEcbList;
    params.engineGroup    = SCAL_MME_COMPUTE_GROUP;
    params.engine_cpu_index =  0;
    params.streamName = "compute0";
    params.streamPriority = SCAL_HIGH_PRIORITY_STREAM;
    params.clusterName = "mme";
    params.copyToHBM = true;
    int rc = run_ecblist_test(&params);
    ASSERT_EQ(rc, 0);
}

TEST_F_CHKDEV(SCAL_Compute_ECBList_Test, compute_ecblist_test_edma_from_device, {GAUDI2})
{
    printf("running EDMA ComputeEcbList from device\n");
    struct ecblist_test_params params;
    params.whichTest = ComputeEcbList;
    params.engineGroup    = SCAL_EDMA_COMPUTE_GROUP;
    params.engine_cpu_index =  0;
    params.streamName = "compute0";
    params.streamPriority = SCAL_HIGH_PRIORITY_STREAM;
    params.clusterName = "compute_edma";
    params.copyToHBM = true;
    int rc = run_ecblist_test(&params);
    ASSERT_EQ(rc, 0);
}