#include <iostream>
#include <algorithm>
#include "scal_basic_test.h"
#include "scal.h"
#include "hlthunk.h"
#include "logger.h"
#include "scal_test_utils.h"
#include "common/pci_ids.h"
#include "gaudi3/asic_reg_structs/sob_objs_regs.h"
#include "scal_test_pqm_pkt_utils.h"

static const unsigned DynamicEcbListSize = 256;
static const unsigned ecbListBufferSize  = 256 * 16;
static const unsigned ecbBufferSize      = 4 * 1024;


#include "scal_gaudi2_sync_monitor.h"
#include "scal_gaudi3_sync_monitor.h"
#include "wscal.h"
#include "infra/sob.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum TestType
{
    SANITY,
    COMPUTE_MEDIA_BASIC,
    COMPUTE_MEDIA_ADVANCED,
    FULL
};

struct test_params
{
    unsigned engineGroup;
    unsigned engineGroup1;
    unsigned engine_cpu_index;
    unsigned use_HBM_buffers;
    unsigned WaitingSchedulerDcoreID;
    const char* jsonConfig;
    unsigned    num_loops;
    const char* streamName;
    const char* cqName;
    const char* clusterName;
    uint64_t    initialLongSOValue;
};

class SCAL_StreamSync_Test : public SCALTestDevice, public testing::WithParamInterface<bool>
{
public:
    int run_ecb_test(struct test_params* params, bool forceFirmwareFence);
    int run_lbw_write_test(struct test_params* params);
    int run_lbw_write_multi_streams(struct test_params* params);
    int run_blockStreamUnlockWrite_test(struct test_params* params);
    int run_blockStreamUnlockMonitor_test(struct test_params* params, bool forceFirmwareFence);
    int run_blockStreamUnlockMonitor_test_direct_and_non_direct_pdma(struct test_params* params, bool forceFirmwareFence);
    int run_blockStreamUnlockMonitor_mult_test(struct test_params* params, bool forceFirmwareFence);
    int run_blockStreamUnlockMonitor_mult_test_direct_and_non_direct_pdma(struct test_params* params, bool forceFirmwareFence);
    int run_blockStreamUnlockMonitor_multsched_test(struct test_params* params, bool forceFirmwareFence);
    int run_fence_inc_immediate_multsched_test(struct test_params* params, bool forceFirmwareFence);
    int run_pdma_with_payload_multsched_test(struct test_params* params, bool forceFirmwareFence);
    int run_blockStreamUnlockWrite_multsched_test(struct test_params* params);
    int run_just_use_other_scheduler_test(struct test_params* params);

protected:
    int buildAndSendEcbs(WScal& wscal, bool forceFirmwareFence);
    int lbwReadWriteTest(WScal& wscal);
    int lbwReadWriteTest_multi_streams(WScal& wscal);
    int BlockStreamUnlockWriteTest(WScal& wscal);
    int BlockStreamUnlockMonitor(WScal& wscal, bool forceFirmwareFence);
    int BlockStreamUnlockMonitorMult(WScal& wscal, bool forceFirmwareFence);
    int BlockStreamUnlockMonitor2(WScal& wscal, bool forceFirmwareFence);
    uint64_t getLbwScratchAddr(WScal& wscal);

    //
    int waitForOtherStream(WScal&   wscal,
                           unsigned waitingStreamIndex,
                           unsigned otherStreamIndex,
                           uint64_t longSOtargetValue,
                           uint64_t prevLongSOtargetValue,
                           unsigned fenceID,
                           bool     compareEQ,
                           bool     forceFirmwareFence);

    uint64_t            m_target = 0;
    struct test_params* m_params = nullptr;
    bool                m_firstTime = true;
};

INSTANTIATE_TEST_SUITE_P(, SCAL_StreamSync_Test, testing::Values(false, true) /* forceFirmwareFence */);

/*
stream B does some work
         Then uses dispatchBarrier to inc the long SO Y associated with its completion group

stream A -
            1 - arm monitor to watch long so Y (of the other stream)
                    When monitor fires - it will send a message to inc fence counter X
                    this message goes to dccm Q address so when ARC is handling it
                    it will unfreeze whoever is waiting on this fence (which will be stream A itself)
            2 - uses fence to wait on fence counter X (wait for the monitor to wake us)

more details:

    long SO Y :  is the long SO associated with stream B's completion group
                 e.g.  scal_completion_group_get_info(&info)
                    info.long_so_index


function   ConfigMonitorForLongSO(monitorID, longSoID, fenceID)
{
        // compute address of monitor

            scal config defines ranges of long monitors, so we have the index
            and can compute the exact address of the specific monitor (see similar code in scal)
            check scal_monitor_pool_get_info()

                    monitor comes from "compute_gp_monitors" group
                            scal_monitor_pool_info info;
                            scal_handle_t mon_pool;

                            scal_get_so_monitor_handle_by_name(scal, "compute_gp_monitors",&mon_pool)
                            scal_monitor_pool_get_info(mon_pool,&info)
                            // now use info->baseIdx + monitor id
        // configure monitor to watch long SO				see scal_init.cpp:542  configureMonitors()
             set monitor
                              Long_sob =1
                              LONG_HIGH_GROUP depends  on the long so index
                              Sid_msb =  which “quarter” the long SO belongs to


        // configure monitor where to fire and what (to fenceID)
                        Payload data : MON_EXP_FENCE_UPDATE ( write the correct fence id)
                        Payload addr – the address of the dccm queue in the scheduler
                           since we must send the "update fence" through the arc handling routines
                           and not do it directly in order to "resume whoever waits on this fence"
}

*/

// in this test, we use the same monitor over and over
// so we don't have to re-configure it every time
// just change the arm value.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MAX_NUM_OF_VALUES_TO_WRITE_G2 10
#define MAX_NUM_OF_VALUES_TO_WRITE_G3 7
int SCAL_StreamSync_Test::waitForOtherStream(WScal&   wscal,
                                             unsigned waitingStreamIndex,
                                             unsigned otherStreamIndex,
                                             uint64_t longSOtargetValue,
                                             uint64_t prevLongSOtargetValue,
                                             unsigned fenceID,
                                             bool     compareEQ,
                                             bool     forceFirmwareFence)
{
    streamBundle*              waitingStream = wscal.getStreamX(waitingStreamIndex);
    streamBundle*              otherStream   = wscal.getStreamX(otherStreamIndex);
    scal_monitor_pool_handle_t monPoolHandle;
    scal_monitor_pool_info     monPoolInfo;
    uint64_t                   addr[MAX_NUM_OF_VALUES_TO_WRITE_G2]; // mon_config + mon_pay_addrh + mon_pay_addrl + mon_pay_data + 3 mon_arm to 0
    uint32_t                   value[MAX_NUM_OF_VALUES_TO_WRITE_G2];
    bool                       isWaitingStreamDM = waitingStream->h_streamInfo.isDirectMode;

    // Monitor ID
    //   NOTE that the monitor and the longSO must belong to the same sm
    int rc = scal_get_so_monitor_handle_by_name(wscal.getScalHandle(), "compute_gp_monitors", &monPoolHandle);
    assert(rc == 0);
    rc |= scal_monitor_pool_get_info(monPoolHandle, &monPoolInfo);
    assert(rc == 0);
    if (rc != 0) // needed for release build
    {
        LOG_ERR(SCAL, "{}: error getting monitor pool info", __FUNCTION__);
    }

    // for gaudi2 long so monitor requires 2 payloads (HW bug)
    const bool dummyPayloadRequired =  getScalDeviceType() == dtGaudi2;


    // RT should handle the allocation of monitors from monPoolInfo.baseIdx to monPoolInfo.baseIdx + monPoolInfo.size
    // for the case when multiple streams request monitors, in different threads.
    // here we just take the 1st one (index 0)
    unsigned monitorID = monPoolInfo.baseIdx;
    if (monitorID % 8 != 0)// in Gaudi3, we can relax this to (monitorID % 4 != 0)
    {
        LOG_DEBUG(SCAL, "{}: configure long monitor needs monitorID ({})to be 32 Bytes aligned, e.g monitor index diviseable by 8. adjusting", __FUNCTION__, monitorID);
        monitorID = monitorID + (8-(monitorID % 8));
    }
    // SO ID - this is the long SO that other stream is using and we should monitor
    unsigned longSoID = otherStream->h_cgInfo.long_so_index; // otherStream->h_cgInfo.sos_base;
    //
    //   Here we write the data using scheduler lbw_write_cmd commands, but it can be QMAN msgLong or any other valid way
    //
    unsigned syncObjIdForDummyPayload = std::numeric_limits<unsigned>::max();
    uint64_t smBaseForDummyPayload = 0;
    unsigned numPayloads = 0;
    if (m_firstTime)
    {
        // Fence ID & target ID are sent by the user
        // the configuring stage returns MAX_NUM_OF_VALUES_TO_WRITE (7) pairs of address/value
        if (dummyPayloadRequired)
        {
            scal_so_pool_handle_t dummySoPoolHandle;
            scal_so_pool_info     dummySoPoolInfo;
            rc = scal_get_so_pool_handle_by_name(wscal.getScalHandle(), "sos_pool_long_monitor_wa", &dummySoPoolHandle);
            assert(rc == 0);
            rc |= scal_so_pool_get_info(dummySoPoolHandle, &dummySoPoolInfo);
            assert(rc == 0);
            if (rc != 0) // needed for release build
            {
                LOG_ERR(SCAL, "{}: error getting monitor pool info", __FUNCTION__);
            }
            syncObjIdForDummyPayload = dummySoPoolInfo.baseIdx;
            smBaseForDummyPayload = dummySoPoolInfo.smBaseAddr;
        }
        if (getScalDeviceType() == dtGaudi3)
        {
            uint64_t pay_addr = 0;
            uint32_t pay_data = 0;

            if (isWaitingStreamDM)
            {
                pay_addr = waitingStream->h_streamInfo.fenceCounterAddress;
                pay_data = PqmPktUtils::getPayloadDataFenceInc(fenceID);
            }
            else if (!forceFirmwareFence)
            {
                scal_control_core_infoV2_t coreInfo;
                scal_control_core_get_infoV2(waitingStream->h_cgInfo.scheduler_handle, &coreInfo);
                gaudi3_getAcpPayload(pay_addr, pay_data, fenceID, SchedCmd::getArcAcpEngBaseAddr(coreInfo.hdCore));
            }
            else
            {
                scal_control_core_infoV2_t coreInfo;
                scal_control_core_get_infoV2(waitingStream->h_cgInfo.scheduler_handle, &coreInfo);
                assert(coreInfo.dccm_message_queue_address);

                pay_addr = coreInfo.dccm_message_queue_address;
                pay_data = gaudi3_createPayload(fenceID);
            }

            //  must send smBase of sm 5 which is where the longSO and Monitor belong
            rc = gaudi3_configMonitorForLongSO(otherStream->h_cgInfo.long_so_sm_base_addr,
                                               waitingStream->h_cgInfo.scheduler_handle,
                                               monitorID,
                                               longSoID,
                                               compareEQ,
                                               pay_addr,
                                               pay_data,
                                               addr,
                                               value);
            numPayloads = MAX_NUM_OF_VALUES_TO_WRITE_G3;
        }
        else
        {
            rc = gaudi2_configMonitorForLongSO(otherStream->h_cgInfo.long_so_sm_base_addr, waitingStream->h_cgInfo.scheduler_handle, monitorID, longSoID, fenceID, compareEQ,
                                               syncObjIdForDummyPayload, smBaseForDummyPayload, addr, value);
            numPayloads = MAX_NUM_OF_VALUES_TO_WRITE_G2;
        }
        assert(rc == 0);
        // write the config + payload data
        for (unsigned i = 0; i < numPayloads; i++)
        {
            waitingStream->h_cmd.lbwWrite(addr[i], value[i], false, isWaitingStreamDM);
        }
    }

    if (getScalDeviceType() == dtGaudi3)
    {
        //  must send smBase of sm 5 which is where the longSO and Monitor belong
        rc = gaudi3_armMonitorForLongSO(otherStream->h_cgInfo.long_so_sm_base_addr, monitorID, longSoID, longSOtargetValue, prevLongSOtargetValue, compareEQ, addr, value, numPayloads);
    }
    else
    {
        rc = gaudi2_armMonitorForLongSO(otherStream->h_cgInfo.long_so_sm_base_addr, monitorID, longSoID, longSOtargetValue, prevLongSOtargetValue, compareEQ, addr, value, numPayloads);
    }
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SCAL, "{}: error arming longSo monitor", __FUNCTION__);
        assert(0);
    }
    // write only the arm data
    for (unsigned i = 0; i < numPayloads; i++)
    {
        waitingStream->h_cmd.lbwWrite(addr[i], value[i], false, isWaitingStreamDM);
    }

    if (!isWaitingStreamDM)
    {
        waitingStream->h_cmd.memFence(true, false, true); // wait for memory ops to finish
    }
    m_firstTime = false;
    return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::buildAndSendEcbs(WScal& wscal, bool forceFirmwareFence)
{
    uint32_t ecb_static_list_size = 256;
    uint32_t addr_index           = 0; // use first buffer out of 6 buffers available
    uint32_t addr_offset          = 0; // offset from the starting of the buffer
    uint32_t switch_cq = 1; // tells FW to switch between handling dynamic commands to static command, should come at
                            // the end of each buffer
    int rc = 0;

    uint8_t  engine_groups0[4] = {m_params->engineGroup, 0, 0, 0};
    uint32_t eng_group_count   = 1;
    // if we use ecb buffers on host shared, our life will be easier
    // but ALL THE ARCS will access the host via the PCIE <--------  THIS IS IMPORTANT!
    // so we don't want that. (think of 24 TPCs trying to read using the PCIE)
    // We can allocate the ECB buffers on the HBM and just copy the data (via PDMA)
    bool useHBMBuffers = m_params->use_HBM_buffers;

    //
    // stream 0
    //
    streamBundle* stream0 = wscal.getStreamX(0); // pdma0 stream

    //
    //
    //      Dynamic ECB List   (listsize + nop)
    //
    //
    // fill engine ecb commands to the ecb list hbm buffer
    // fill dynamic list first since it is in the beginning of buffer

    // allocate ecbList buffer
    bufferBundle_t ecbListbufHost;
    rc |= wscal.getBufferX(WScal::HostSharedPool, ecbListBufferSize, &ecbListbufHost); // get both buffer handle and info
    assert(rc == 0);
    uint8_t* ecbListHostAddr = (uint8_t*)ecbListbufHost.h_BufferInfo.host_address;
    // if we decide to use HBM buffers so we need to copy to HBM
    bufferBundle_t ecbListbufHBM;
    if (useHBMBuffers)
    {
        // allocate the ecb list buffer on the device HBM
        //  note if we use  GlobalHBMPool instead of devSharedPool we'll get a crash in FS_ArcCore::addrExtendCalc()
        //  since the ARC has no access to globalHBM memory
        rc |= wscal.getBufferX(WScal::devSharedPool, ecbListBufferSize, &ecbListbufHBM);
        assert(rc == 0);
    }
    // attach ecb list command handler to this buffer
    EcbListCmd ecbList((char*)ecbListHostAddr, ecbListBufferSize,  getScalDeviceType());
    ecbList.ArcListSizeCmd(DynamicEcbListSize);
    ecbList.Pad(DynamicEcbListSize, switch_cq);

    //
    //
    //      Static ECB List   (lissize + ECB StaticDesc + NOP)
    //
    //
    // fill static list next

    ecbList.ArcListSizeCmd(ecb_static_list_size);
    ecbList.staticDescCmd(m_params->engine_cpu_index, ecbBufferSize, addr_offset, addr_index);
    ecbList.Pad(DynamicEcbListSize + ecb_static_list_size, switch_cq);

    //
    //   QMan Commands on the ECB Buf. This is our "recipe"
    //
    // fill engine qman commands into ecb buff
    bufferBundle_t ecbBuf;
    rc |= wscal.getBufferX(WScal::HostSharedPool, ecbBufferSize, &ecbBuf);
    assert(rc == 0);
    uint8_t* ecbHostAddr = (uint8_t*)ecbBuf.h_BufferInfo.host_address;
    // if we decide to use HBM buffers so we need to copy to HBM
    bufferBundle_t ecbBufHBM;
    if (useHBMBuffers)
    {
        // allocate the ecb list buffer on the device HBM
        rc |= wscal.getBufferX(WScal::devSharedPool, ecbListBufferSize, &ecbBufHBM);
        assert(rc == 0);
    }
    uint32_t curr_buff_off = 0;
    uint8_t* ecb_buff      = ecbHostAddr;

    // set the correct packet builders according to the actual device
    uint32_t (*qman_add_nop_pkt)(void* , uint32_t , enum QMAN_EB, enum QMAN_MB);
    uint32_t (*createPayload)(uint32_t);
    if (getScalDeviceType() == dtGaudi3)
    {
        qman_add_nop_pkt = gaudi3_qman_add_nop_pkt;
        createPayload    = gaudi3_createPayload;
    }
    else
    {
        qman_add_nop_pkt = gaudi2_qman_add_nop_pkt;
        createPayload    = gaudi2_createPayload;
    }

    while (curr_buff_off < ecbBufferSize)
    {
        curr_buff_off = (*qman_add_nop_pkt)(ecb_buff, curr_buff_off, EB_FALSE, MB_FALSE);
    }

    //
    //  fill scheduler commands - use update_recipe_base_cmd to point to the ECB buf
    //

    const uint32_t fence_id = 0;
    // pay_addr = address of DCCM queue 0 of scheduler 0
    //  luckily, that's the one we use ...
    scal_control_core_infoV2_t coreInfo;
    scal_control_core_get_infoV2(stream0->h_cgInfo.scheduler_handle, &coreInfo);
    assert(coreInfo.dccm_message_queue_address);

    //
    // payload is used for synchronization
    //
    uint64_t pay_addr = 0;
    uint32_t pay_data = 0;
    if ((getScalDeviceType() != dtGaudi2) && (!forceFirmwareFence))
    {
        gaudi3_getAcpPayload(pay_addr, pay_data, fence_id, SchedCmd::getArcAcpEngBaseAddr(coreInfo.hdCore));
    }
    else
    {
        pay_addr = coreInfo.dccm_message_queue_address;
        pay_data = createPayload(fence_id);
    }

    if (useHBMBuffers)
    {
        // check if pdma streams are in scheduler mode
        scal_streamset_handle_t streamset_tx;
        scal_streamset_info_t   tx_info;

        scal_get_streamset_handle_by_name(wscal.getScalHandle(), "pdma_tx", &streamset_tx);
        scal_streamset_get_info(streamset_tx, &tx_info);

        if (tx_info.isDirectMode)
        {
            rc |= scal_free_buffer(ecbListbufHost.h_Buffer);
            assert(rc == 0);
            rc |= scal_free_buffer(ecbBuf.h_Buffer);
            assert(rc == 0);
            rc |= scal_free_buffer(ecbListbufHBM.h_Buffer);
            assert(rc == 0);
            rc |= scal_free_buffer(ecbBufHBM.h_Buffer);
            assert(rc == 0);
            return SCAL_UNSUPPORTED_TEST_CONFIG;
        }
        // need to copy the ecb & ecb list buffers to the HBM

        uint32_t target = 1;


        // pdma the ecb list from host to device
        stream0->h_cmd.PdmaTransferCmd(false, ecbListbufHBM.h_BufferInfo.device_address,  // dest
                                       ecbListbufHost.h_BufferInfo.device_address, // src
                                       ecbListBufferSize,                          // size
                                       SCAL_PDMA_TX_DATA_GROUP,                    // using SCAL_TPC_COMPUTE_GROUP here will crash
                                       -1,                                          // user data
                                       pay_data, pay_addr);
        // PdmaTransferCmd is asynchronic, so wait on fence id 0 for it to release us
        stream0->h_cmd.FenceCmd(fence_id, target, forceFirmwareFence, false /* isDirectMode */);
        // now copy the ecb buffer
        stream0->h_cmd.PdmaTransferCmd(false, ecbBufHBM.h_BufferInfo.device_address, // dest
                                       ecbBuf.h_BufferInfo.device_address,    // src
                                       ecbBufferSize,                         // size
                                       SCAL_PDMA_TX_DATA_GROUP,               // engine group type
                                       -1,                                     // user data
                                       pay_data, pay_addr);
        // and wait again
        stream0->h_cmd.FenceCmd(fence_id, target, forceFirmwareFence, false /* isDirectMode */);
    }
    uint16_t recipe_base_index = 0;
    uint64_t recipeBaseAddr    = 0;
    if (!useHBMBuffers)
        recipeBaseAddr = ecbBuf.h_BufferInfo.device_address; // <-------------- use the HOST SHARED ecb buff (recipe buffer) device address
    else
        recipeBaseAddr = ecbBufHBM.h_BufferInfo.device_address; // <-------------- use the HBM ecb buff (recipe buffer) address
    stream0->h_cmd.UpdateRecipeBase(&recipe_base_index, eng_group_count, engine_groups0, &recipeBaseAddr, 1);
    bool     single_static_chunk    = true;
    bool     single_dynamic_chunk   = true;
    uint32_t engine_group_type      = m_params->engineGroup;
    uint32_t static_ecb_list_offset = DynamicEcbListSize; // one chunk away
    uint32_t dynamic_ecb_list_addr  = 0;
    if (!useHBMBuffers)
        dynamic_ecb_list_addr = ecbListbufHost.h_BufferInfo.device_address; //  <-------------  use the HOST SHARED ecb list buf device address
    else
        dynamic_ecb_list_addr = ecbListbufHBM.h_BufferInfo.device_address; //  <-------------  use the HBM ecb list buf address
    stream0->h_cmd.DispatchComputeEcbList(engine_group_type, single_static_chunk, single_dynamic_chunk,
                                          static_ecb_list_offset, dynamic_ecb_list_addr);
    stream0->h_cmd.NopCmd(0);
    stream0->AllocAndDispatchBarrier(eng_group_count, engine_groups0);
    // submit the command buffer on our stream
    rc |= stream0->stream_submit();
    assert(rc == 0);

    // TBD - release the so set by using AllocBarrier with release_so = 1 (currently crashing ...)
    m_target++;
    // wait for completion
    LOG_DEBUG(SCAL, "Waiting for stream completion. target {}", m_target);

    rc |= stream0->completion_group_wait(m_target);
    assert(rc == 0);

    // releaseBuffers
    rc |= scal_free_buffer(ecbListbufHost.h_Buffer);
    assert(rc == 0);
    rc |= scal_free_buffer(ecbBuf.h_Buffer);
    assert(rc == 0);
    if (useHBMBuffers)
    {
        rc |= scal_free_buffer(ecbListbufHBM.h_Buffer);
        assert(rc == 0);
        rc |= scal_free_buffer(ecbBufHBM.h_Buffer);
        assert(rc == 0);
    }
    return rc;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool isPalladium(int fd)
{
    uint32_t deviceId = hlthunk_get_device_id_from_fd(fd);

    switch (deviceId)
    {
    case PCI_IDS_GAUDI3:
    case PCI_IDS_GAUDI2_FPGA:
    case PCI_IDS_GAUDI3_DIE1:
    case PCI_IDS_GAUDI3_SINGLE_DIE:
        return true;
    default:
        break;
    }
    return false;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
uint64_t SCAL_StreamSync_Test::getLbwScratchAddr(WScal& wscal)
{
    uint64_t      Addr2Write = 0;
    scal_handle_t scalHandle = wscal.getScalHandle();
    // scratch lbw area is not mapped to Gaudi3 PLDM to save space
    // so use something that is not needed for this test
    scal_so_pool_handle_t so_pool;
    int rc = scal_get_so_pool_handle_by_name(scalHandle,"network_gp_sos_1", &so_pool);
    if(rc)
    {
        LOG_ERR(SCAL, "scal_get_so_pool_handle_by_name() failed");
        assert(0);
    }
    scal_so_pool_info info;
    rc = scal_so_pool_get_info(so_pool,&info);
    if(rc)
    {
        LOG_ERR(SCAL, "scal_so_pool_get_info() failed");
        assert(0);
    }

    if (getScalDeviceType() == dtGaudi3)
    {
        Addr2Write = SobG3::getAddr(info.smBaseAddr, info.baseIdx);
    }
    else
    {
        Addr2Write = SobG2::getAddr(info.smBaseAddr, info.baseIdx);
    }

    return Addr2Write;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::lbwReadWriteTest(WScal& wscal)
{
    //
    // stream 0
    //
    streamBundle* stream0 = wscal.getStreamX(0); // pdma0 stream

    bufferBundle_t readBuf;
    int rc = wscal.getBufferX(WScal::HostSharedPool, sizeof(uint32_t), &readBuf); // get both buffer handle and info
    assert(rc == 0);

    uint32_t                 readAddr    = readBuf.h_BufferInfo.core_address;
    volatile uint32_t* const hostBufAddr = (uint32_t*)readBuf.h_BufferInfo.host_address;
    uint64_t Addr2Write = getLbwScratchAddr(wscal);

    for (unsigned i = 1; i <= 100; i++)
    {
        memset((void*)hostBufAddr, 0, sizeof(uint32_t));
        stream0->h_cmd.NopCmd(0);
        stream0->h_cmd.lbwWrite(Addr2Write, i, false);
        stream0->h_cmd.memFence(true, false, true);
        stream0->AllocAndDispatchBarrier(0, NULL, 0); // target value = 0
        m_target++;
        stream0->h_cmd.lbwRead(readAddr, Addr2Write, sizeof(uint32_t));
        stream0->h_cmd.memFence(true, false, true);
        stream0->AllocAndDispatchBarrier(0, NULL, 0); // target value = 0
        m_target++;
        rc |= stream0->stream_submit();
        assert(rc == 0);
        // wait for completion
        LOG_DEBUG(SCAL, "Waiting for stream completion. target {}", m_target);

        rc |= stream0->completion_group_wait(m_target);
        assert(rc == 0);
        // Compare written value with read value
        if (i != (*hostBufAddr))
        {
            LOG_ERR(SCAL, "compare error in lbwReadWriteTest. i={} *hostBufAddr={}", i, *hostBufAddr);
            rc |= -1;
            assert(0);
        }
    }
    int rc1 = scal_free_buffer(readBuf.h_Buffer);
    assert(rc1 == 0);
    return rc | rc1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <random>
int SCAL_StreamSync_Test::lbwReadWriteTest_multi_streams(WScal& wscal)
{
    const unsigned NUM_STREAMS = wscal.getNumStreams();
    bufferBundle_t readBuf;
    int rc = wscal.getBufferX(WScal::HostSharedPool, NUM_STREAMS * sizeof(uint32_t), &readBuf); // get both buffer handle and info
    assert(rc == 0);
    uint32_t readAddr = readBuf.h_BufferInfo.core_address;
    volatile uint32_t* const hostBufAddr = (uint32_t*)readBuf.h_BufferInfo.host_address;
    std::vector<unsigned> streams_ids(NUM_STREAMS) ; // vector of NUM_STREAMS size
    std::iota(std::begin(streams_ids), std::end(streams_ids), 0); // Fill with 0, 1, ...,
    std::random_device r;
    auto rng = std::default_random_engine {r()};
    std::shuffle(std::begin(streams_ids), std::end(streams_ids), rng); // Shuffle the stream ids
    uint64_t Addr2Write = getLbwScratchAddr(wscal);

    for(unsigned j=0 ; j < NUM_STREAMS ; j++)
    {
        unsigned stream_no = streams_ids[j];// random stream index
        streamBundle* stream0 = wscal.getStreamX(stream_no);

        unsigned target = 0;
        for (unsigned i = 1; i <= 10; i++)
        {
            memset((void*)&hostBufAddr[stream_no], 0, sizeof(uint32_t));
            LOG_INFO(SCAL, "sending data for stream {} ", stream0->h_streamInfo.name);
            stream0->h_cmd.NopCmd(0);
            // write value i into this core address (it has nothing to do with the address being of a monitor, just a place in memory that seems unused)
            for (unsigned k = 1; k <= 100; k++)
            {
                // send this command many times to fill up the buffer. It exposed some bugs ...
                stream0->h_cmd.lbwWrite(Addr2Write, i, false);
            }
            stream0->h_cmd.memFence(true, false, true);
            stream0->AllocAndDispatchBarrier(0, NULL, 0); // target value = 0
            target++;
            // now read that value from the same core address
            stream0->h_cmd.lbwRead(readAddr+stream_no*sizeof(uint32_t), Addr2Write, sizeof(uint32_t));
            stream0->h_cmd.memFence(true, false, true);
            stream0->AllocAndDispatchBarrier(0, NULL, 0); // target value = 0
            target++;
            rc |= stream0->stream_submit();
            assert(rc == 0);
            // wait for completion
            LOG_INFO(SCAL, "Waiting for stream {} #{} completion. target {}", stream0->h_streamInfo.name, stream0->h_streamInfo.index, target);

            rc |= stream0->completion_group_wait(target);
            assert(rc == 0);
            // Compare written value with read value
            if (i != (hostBufAddr[stream_no]))
            {
                LOG_ERR(SCAL, "compare error in lbwReadWriteTest_multi_streams. i={} *hostBufAddr={}", i, *hostBufAddr);
                rc |= -1;
                assert(0);
                // in release this will continue, which is by design
            }
        }
    }
    int rc1 = scal_free_buffer(readBuf.h_Buffer);
    assert(rc1 == 0);
    return rc | rc1;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  stream 0 - does pdma (host to device) and then AllocAndDispatchBarrier
//
//  stream 1 - configures a monitor to watch stream 0 longSO and unblock itself once longSO reaches the target value
//             blocks (using Fence) (waiting for the monitor above to unblock it)
//             copies the data (device to host) that stream 0 copied before
//
int SCAL_StreamSync_Test::BlockStreamUnlockMonitor(WScal& wscal,
                                                   bool   forceFirmwareFence)
{
    unsigned waitingStreamIndex = 1;
    unsigned otherStreamIndex   = 0;
    uint64_t longSOtargetValue  = 1ULL;
    unsigned fenceID            = 0;
    int32_t workload = -1; // user - user data can be of very large size and should be fragmented into chunks of size
                           // PDMA_USER_DATA_CHUNK_SIZE
    int rc = 0;

    uint64_t      prevLongSOtargetValue = longSOtargetValue;
    streamBundle* stream0 = wscal.getStreamX(0); // pdma0 stream
    streamBundle* stream1 = wscal.getStreamX(1); // pdma1 stream

    const unsigned testDmaBufferSize = 4096 * 16; // 1MB
    // Host DMA buffer - our src buffer
    bufferBundle_t hostDMAbuf;
    rc |= wscal.getBufferX(WScal::HostSharedPool, testDmaBufferSize, &hostDMAbuf); // get both buffer handle and info
    assert(rc == 0);
    // set source host buffer to some values so we can detect copy errors
    uint8_t* pHost1 = (uint8_t*)hostDMAbuf.h_BufferInfo.host_address;
    for (unsigned j = 0; j < testDmaBufferSize; j++)
        pHost1[j] = (j % 0xff);
    // HBM DMA buffer - our dest buffer
    bufferBundle_t deviceDMAbuf;
    rc |= wscal.getBufferX(WScal::devSharedPool, testDmaBufferSize, &deviceDMAbuf); // get both buffer handle and info
    assert(rc == 0);
    // Host DMA buffer - our compare buffer
    bufferBundle_t hostDMAbuf2;
    rc |= wscal.getBufferX(WScal::HostSharedPool, testDmaBufferSize, &hostDMAbuf2); // get both buffer handle and info
    assert(rc == 0);
    uint8_t* pHost2 = (uint8_t*)hostDMAbuf2.h_BufferInfo.host_address;

    bool isStream0DirectMode = stream0->h_streamInfo.isDirectMode;
    bool isStream1DirectMode = stream1->h_streamInfo.isDirectMode;

    for (unsigned i = 1; i <= 100 && rc == 0; i++)
    {
        LOG_TRACE(SCAL, "Starting loop {}", i);
        memset(pHost2, 0, testDmaBufferSize);

        //
        // stream 0 - copy buffer from host to device then notify (using AllocAndDispatchBarrier)
        //            This will increase the long so associated with the completion group of this stream
        //            In direct mode in order to increase the long so we need to use wr_comp
        //
        stream0->h_cmd.PdmaTransferCmd(isStream0DirectMode,
                                       deviceDMAbuf.h_BufferInfo.device_address, // dest
                                       hostDMAbuf.h_BufferInfo.device_address,   // src
                                       testDmaBufferSize,                        // size
                                       m_params->engineGroup,                    // engine group type. host2device (will split user data to small chunks)
                                       workload,                                 // user or cmd
                                       0, 0,                                     // payload addr , payload data - a great way to sync but not used in this test
                                       0/*signal to cg*/, 0/*compGrpidx*/, true, /*wr_comp*/
                                       &(stream0->h_cgInfo));                    // cg for wr_comp, increase long SO

        // we could have used the payload to unfreeze the other stream (as done in SCAL_Fence_Test::run_barrier_test)
        // but here we want to demonstrate how to configure monitors to watch barrier signals inside the device
        stream0->WaitOnHostOverPdmaStream(m_params->engineGroup, isStream0DirectMode);

        LOG_DEBUG(SCAL, "submitting stream 0");
        rc |= stream0->stream_submit();
        assert(rc == 0);
        //
        // stream 1 - wait for other stream (for its barrier long so to be incremented), then copy the buffer back to host
        //
        //  configure monitor to check other stream longSO to reach longSOtargetValue, when fired, unfreeze stream that
        //  waits on fence id
        //
        m_target++;
        longSOtargetValue = m_target;
        waitForOtherStream(wscal,
                           waitingStreamIndex,
                           otherStreamIndex,
                           longSOtargetValue,
                           prevLongSOtargetValue,
                           fenceID,
                           true,
                           forceFirmwareFence); // use compareEQUAL, e.g. ==
        prevLongSOtargetValue = longSOtargetValue;

        // wait on fence id
        // use target=1 because waitForOtherStream() configured the monitor payload to increase the fence by 1
        // (Fence wait set it to -1 (fenceCounter -= target), they up it to 0 (fenceCounter++), and it unfreezes when it
        // >=0)
        stream1->h_cmd.FenceCmd(fenceID, 1, forceFirmwareFence, isStream1DirectMode);

        // now copy the data back to the host
        stream1->h_cmd.PdmaTransferCmd(isStream1DirectMode,
                                       hostDMAbuf2.h_BufferInfo.device_address,  // dest
                                       deviceDMAbuf.h_BufferInfo.device_address, // src
                                       testDmaBufferSize,                        // size
                                       m_params->engineGroup1,                   // engine group type
                                       workload,                                 // user or cmd
                                       0, 0);                                    // payload addr , payload data

        stream1->WaitOnHostOverPdmaStream(m_params->engineGroup1, isStream1DirectMode);
        LOG_DEBUG(SCAL, "submitting stream 1");

        rc |= stream1->stream_submit();
        assert(rc == 0);
        //
        // wait for completion of stream 0
        //
        rc |= stream0->completion_group_wait(m_target);
        assert(rc == 0);
        //
        // wait for completion of stream 1
        //
        rc |= stream1->completion_group_wait(m_target);
        assert(rc == 0);

        // compare the buffers
        for (unsigned j = 0; j < testDmaBufferSize; j++)
        {
            if (pHost1[j] != pHost2[j])
            {
                LOG_ERR(SCAL, "compare error in BlockStreamUnlockMonitor. j={} pHost1[j]={:#x} pHost2[j]={:#x}", j, pHost1[j], pHost2[j]);
                rc |= -1;
                assert(0);
                break;
            }
        }
        if (rc != 0) break;
    }
    // releaseBuffers
    rc |= scal_free_buffer(hostDMAbuf.h_Buffer);
    assert(rc == 0);
    rc |= scal_free_buffer(hostDMAbuf2.h_Buffer);
    assert(rc == 0);
    rc |= scal_free_buffer(deviceDMAbuf.h_Buffer);
    assert(rc == 0);
    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Test in which stream A submits N jobs and simultaneously stream B waits on each of them and submit workloads after
   each of them expires
*/
int SCAL_StreamSync_Test::BlockStreamUnlockMonitorMult(WScal& wscal,
                                                       bool   forceFirmwareFence)
{
    unsigned waitingStreamIndex = 1;
    unsigned otherStreamIndex   = 0;
    uint64_t longSOtargetValue     = 1;
    uint64_t prevLongSOtargetValue = longSOtargetValue;
    unsigned fenceID            = 0;
    unsigned N                  = 10;
    int32_t workload = -1; // user - user data can be of very large size and should be fragmented into chunks of size
                           // PDMA_USER_DATA_CHUNK_SIZE
    int rc = 0;

    streamBundle* stream0 = wscal.getStreamX(0); // pdma0 stream
    streamBundle* stream1 = wscal.getStreamX(1); // pdma1 stream

    const unsigned testDmaBufferSize = 4096 * 16; // 1MB
    // Host DMA buffer - our src buffer
    bufferBundle_t hostDMAbuf[N];
    // HBM DMA buffer - our dest buffer
    bufferBundle_t deviceDMAbuf[N];
    // Host DMA buffer - our compare buffer
    bufferBundle_t hostDMAbuf2[N];
    for (unsigned j = 0; j < N; j++)
    {
        rc |= wscal.getBufferX(WScal::HostSharedPool, testDmaBufferSize,
                               &hostDMAbuf[j]); // get both buffer handle and info
        assert(rc == 0);
        // set source host buffer to some values so we can detect copy errors
        uint8_t* pHost1 = (uint8_t*)hostDMAbuf[j].h_BufferInfo.host_address;
        for (unsigned k = 0; k < testDmaBufferSize; k++)
            pHost1[k] = (j * 20) + (k % 20);
        rc |= wscal.getBufferX(WScal::devSharedPool, testDmaBufferSize, &deviceDMAbuf[j]);
        assert(rc == 0);
        rc |= wscal.getBufferX(WScal::HostSharedPool, testDmaBufferSize, &hostDMAbuf2[j]);
        assert(rc == 0);
    }

    bool isStream0DirectMode = stream0->h_streamInfo.isDirectMode;
    bool isStream1DirectMode = stream1->h_streamInfo.isDirectMode;

    for (unsigned i = 1; i <= 100 && rc == 0; i++)
    {
        LOG_TRACE(SCAL, "Starting loop {}", i);
        for (unsigned j = 0; j < N; j++)
        {
            uint8_t* pHost2 = (uint8_t*)hostDMAbuf2[j].h_BufferInfo.host_address;
            memset(pHost2, 0, testDmaBufferSize);
        }

        //
        // stream 0 - copy buffer from host to device then notify (using PDMA Barrier)
        //            This will increase the long so associated with the completion group of this stream
        //
        //    we send N * (PDMA commands + PDMA barrier) at once
        for (unsigned j = 0; j < N; j++)
        {
            stream0->h_cmd.PdmaTransferCmd(isStream0DirectMode, deviceDMAbuf[j].h_BufferInfo.device_address, // dest
                                           hostDMAbuf[j].h_BufferInfo.device_address,   // src
                                           testDmaBufferSize,                           // size
                                           m_params->engineGroup,                       // engine group type. host2device (will split user data to small chunks)
                                           workload,                                    // user or cmd
                                           0, 0,                                        // payload addr , payload data
                                           0/*signal to cg*/, 0/*compGrpidx*/, true/*wr_comp*/,
                                           &(stream0->h_cgInfo));                       // cg for wr_comp, increase long SO
            stream0->WaitOnHostOverPdmaStream(m_params->engineGroup, isStream0DirectMode);
        }
        //
        // stream 1 - wait for other stream (for its barrier long so to increment), then copy the buffer back to host
        //
        //  configure monitor to check other stream longSO to reach longSOtargetValue, when fired, unfreeze stream that
        //  waits on fence id
        //
        //      we wait for each of the steps of stream 0 and once done copy the buffer back to host and wait for the
        //      next step
        for (unsigned j = 0; j < N; j++)
        {
            m_target++;
            longSOtargetValue = m_target;
            waitForOtherStream(wscal,
                               waitingStreamIndex,
                               otherStreamIndex,
                               longSOtargetValue,
                               prevLongSOtargetValue,
                               fenceID,
                               false,
                               forceFirmwareFence); // use compare >=
            prevLongSOtargetValue = longSOtargetValue;
            // wait on fence id
            // use target=1 because waitForOtherStream() configured the monitor payload to increase the fence by 1
            // (Fence wait set it to -1 (fenceCounter -= target), they up it to 0 (fenceCounter++), and it unfreezes
            // when it
            // >=0)
            stream1->h_cmd.FenceCmd(fenceID, 1, forceFirmwareFence, isStream1DirectMode);
            // now copy the data back to the host
            stream1->h_cmd.PdmaTransferCmd(isStream1DirectMode,
                                           hostDMAbuf2[j].h_BufferInfo.device_address,  // dest
                                           deviceDMAbuf[j].h_BufferInfo.device_address, // src
                                           testDmaBufferSize,                           // size
                                           m_params->engineGroup1,                      // engine group type
                                           workload,                                    // user or cmd
                                           0, 0);                                       // payload addr , payload data

            stream1->WaitOnHostOverPdmaStream(m_params->engineGroup1, isStream1DirectMode);
        }
        // submit work on stream 1 BEFORE the work on stream 0  (so stream 1 will have to wait for stream 0)
        rc |= stream1->stream_submit();
        assert(rc == 0);
        rc |= stream0->stream_submit();
        assert(rc == 0);

#ifdef NOT_NEEDED_AS_WE_CAN_WAIT_FOR_STREAM_1_ONLY
        //
        // wait for completion of stream 0
        //
        rc |= stream0->completion_group_wait(m_target); // note that m_target is raised by N every loop
        assert(rc == 0);
#endif
        //
        // wait for completion of stream 1
        //
        rc |= stream1->completion_group_wait(m_target);
        assert(rc == 0);

        // compare the buffers
        for (unsigned j = 0; j < N; j++)
        {
            uint8_t* pHost1 = (uint8_t*)hostDMAbuf[j].h_BufferInfo.host_address;
            uint8_t* pHost2 = (uint8_t*)hostDMAbuf2[j].h_BufferInfo.host_address;
            for (unsigned k = 0; k < testDmaBufferSize; k++)
            {
                if (pHost1[k] != pHost2[k])
                {
                    LOG_ERR(SCAL, "compare error in BlockStreamUnlockMonitor. j={} k={} pHost1[k]={:#x} pHost2[k]={:#x}", j, k, pHost1[k], pHost2[k]);
                    rc |= -1;
                    assert(0);
                    break;
                }
            }
            if (rc != 0) break;
        }
        if (rc != 0) break;
    }
    // releaseBuffers
    for (unsigned j = 0; j < N; j++)
    {
        rc |= scal_free_buffer(hostDMAbuf[j].h_Buffer);
        assert(rc == 0);
        rc |= scal_free_buffer(hostDMAbuf2[j].h_Buffer);
        assert(rc == 0);
        rc |= scal_free_buffer(deviceDMAbuf[j].h_Buffer);
        assert(rc == 0);
    }
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
        stream 0 does lbw write with block = true
        stream 1 does direct write into the lcp mask of stream 0 to unblock it

*/
int SCAL_StreamSync_Test::BlockStreamUnlockWriteTest(WScal& wscal)
{
    streamBundle* stream0 = wscal.getStreamX(0);
    streamBundle* stream1 = wscal.getStreamX(1);

    bufferBundle_t readBuf;
    const unsigned dataSize = 4 * sizeof(uint32_t);
    int            rc = wscal.getBufferX(WScal::HostSharedPool, dataSize, &readBuf); // get both buffer handle and info
    assert(rc == 0);

    uint32_t                 readAddr       = readBuf.h_BufferInfo.core_address;
    volatile uint32_t* const hostBufAddr    = (uint32_t*)readBuf.h_BufferInfo.host_address;
    uint32_t                 knownDwords[4] = {0xABCDEFAB, 0xABCDEFBC, 0xABCDEFCD, 0xABCDEFDE};
    uint32_t                 acpMaskRegAddr = 0;
    uint64_t                 Addr2Write = getLbwScratchAddr(wscal);

    if (getScalDeviceType() == dtGaudi3)
    {
        acpMaskRegAddr = gaudi3_getAcpRegAddr(m_params->WaitingSchedulerDcoreID);
    }
    else
    {
        acpMaskRegAddr = gaudi2_getAcpRegAddr(m_params->WaitingSchedulerDcoreID);
    }

    for (unsigned i = 1; i <= 10 && rc == 0; i++)
    {
        memset((void*)hostBufAddr, 0, dataSize);
        //
        // stream 0
        //
        stream0->h_cmd.NopCmd(1);
        stream0->h_cmd.lbwBurstWrite(Addr2Write, &knownDwords[0],
                                     true); // blocking is true
        stream0->h_cmd.lbwRead(readAddr, Addr2Write, dataSize);
        stream0->h_cmd.memFence(true, false, true);
        stream0->AllocAndDispatchBarrier(0, NULL, 0); // target value = 0
        m_target++;
        rc |= stream0->stream_submit();
        assert(rc == 0);
        //
        // stream 1
        //

        // there is no guarantee that stream 0 will actually run before stream 1,
        //   sleep to make sure that stream 0 runs first otherwise it's a deadlock
        sleep(1);

        stream1->h_cmd.NopCmd(0);
        uint32_t lbwBaseAddr = ARC_LBU_ADDR(acpMaskRegAddr);
        stream1->h_cmd.lbwWrite(lbwBaseAddr + (stream0->h_streamInfo.index * 4), 0,
                                false); // Unmask pdma0 stream by writing to lbu_acp_mask register
        stream1->h_cmd.memFence(true, false, true);
        stream1->AllocAndDispatchBarrier(0, NULL, 0); // target value = 0
        rc |= stream1->stream_submit();
        assert(rc == 0);

        // wait for completion
        LOG_DEBUG(SCAL, "Waiting for stream completion. target {}", m_target);
        rc |= stream0->completion_group_wait(m_target);
        assert(rc == 0);
        rc |= stream1->completion_group_wait(m_target);
        assert(rc == 0);
        // Compare written value with read value
        for (unsigned j = 0; j < 4; j++)
        {
            if (knownDwords[j] != hostBufAddr[j])
            {
                LOG_ERR(SCAL, "compare error in BlockStreamUnlockMonitor. j={} knownDwords[j]={:#x} hostBufAddr[j]={:#x}", j, knownDwords[j], hostBufAddr[j]);
                rc |= -1;
                assert(0);
            }
        }
    }
    int rc1 = scal_free_buffer(readBuf.h_Buffer);
    assert(rc1 == 0);
    return rc | rc1;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::BlockStreamUnlockMonitor2(WScal& wscal,
                                                    bool   forceFirmwareFence)
{
    streamBundle* stream0 = wscal.getStreamX(0);
    streamBundle* stream1 = wscal.getStreamX(1);

    bufferBundle_t readBuf;
    const unsigned dataSize = 4 * sizeof(uint32_t);
    int            rc       = wscal.getBufferX(WScal::HostSharedPool, dataSize, &readBuf); // get both buffer handle and info
    assert(rc == 0);

    uint32_t                 readAddr           = readBuf.h_BufferInfo.core_address;
    volatile uint32_t* const hostBufAddr        = (uint32_t*)readBuf.h_BufferInfo.host_address;
    uint32_t                 knownDwords[4]     = {0xABCDEFAB, 0xABCDEFBC, 0xABCDEFCD, 0xABCDEFDE};
    unsigned                 waitingStreamIndex = 1;
    unsigned                 otherStreamIndex   = 0;
    uint64_t                 longSOtargetValue     = 1;
    uint64_t                 prevLongSOtargetValue = longSOtargetValue;
    unsigned                 fenceID            = 0;
    if (m_params->num_loops == 0)
        m_params->num_loops = 100;
    u_int64_t targetStream0 = 0;
    u_int64_t targetStream1 = 0;
    if (m_params->initialLongSOValue != 0)
    {
        // test corner cases - how the monitor works when the longSO has specific values
        if (m_params->initialLongSOValue > 0x10000)
        {
            // This method is only for VERY BIG numbers, where looping (like in the else block)
            // is not practical.  This methond has the disadvantage that the driver is not aware that
            // the value of the longSO has changed and we need to cheat a little with our target value

            // we preset stream 0 longSO to be the given value X
            // so we'll wait for X+1

            targetStream0 = m_params->initialLongSOValue;

            uint64_t smBase = stream0->h_cgInfo.sm_base_addr;
            // test corner cases - how the monitor works when the longSO has specific values
            uint64_t value = m_params->initialLongSOValue;
            // stream 1 will be monitor stream 0 longSO
            unsigned longSoID = stream0->h_cgInfo.long_so_index;
            // write 4 x  15 bit values
            for (unsigned i = 0; i < 4; i++)
            {
                uint64_t addr = 0;
                if (getScalDeviceType() == dtGaudi3)
                {
                    addr = SobG3::getAddr(smBase, longSoID + i);
                }
                else
                {
                    addr = SobG2::getAddr(smBase, longSoID + i);
                }

                uint32_t data =  value & 0x7FFF;
                stream0->h_cmd.lbwWrite((uint32_t)addr, data, false);
                value = value >> 15;
            }
            stream0->h_cmd.memFence(true, false, true);   // wait for memory operation to finish

            // also note that the driver limits the target value when calling hlthunk_wait_for_interrupt to 32 bit

        }
        else
        {
            // since direct writing into the longSO does not work
            // (the driver does not reflect the changed value, I guess the driver only gets ++ interupts, not full value)
            // we need to actually loop to raise the longSO value using AllocAndDispatchBarrier

            uint64_t timeout =  30 * 1000000ULL;// in microsecond, 1/MILION of second
            for (uint64_t i = 0; i <  m_params->initialLongSOValue; i++)
            {
                stream0->h_cmd.NopCmd(0);
                stream0->AllocAndDispatchBarrier(0, NULL, 0); // target value = 0
                targetStream0++;
                rc |= stream0->stream_submit();
                assert(rc == 0);
                if (targetStream0 % 4096 == 0)
                {
                        LOG_DEBUG(SCAL, "Waiting for stream 0 completion. target {:#x}", targetStream0);
                        rc |= stream0->completion_group_wait(targetStream0, timeout);
                        assert(rc == 0);
                }
            }
        }
    }
    uint64_t Addr2Write = getLbwScratchAddr(wscal);
    for (unsigned i = 1; i <=  m_params->num_loops && rc == 0; i++)
    {
        memset((void*)hostBufAddr, 0, dataSize);
        //
        // stream 0  - do lbwBurstWrite (without blocking) and dispatch barrier
        //
        stream0->h_cmd.NopCmd(1);
        stream0->h_cmd.lbwBurstWrite(Addr2Write, &knownDwords[0],
                                     false);          // blocking is false
        stream0->h_cmd.memFence(true, false, true);   // wait for memory operation to finish
        stream0->AllocAndDispatchBarrier(0, NULL, 0); // target value = 0
        targetStream0++;
        //
        // stream 1 - configure monitor to watch stream 0, and wait, upon release lbwRead value and compare
        //
        longSOtargetValue = targetStream0;
        // configure monitor to watch stream 0
        waitForOtherStream(wscal,
                           waitingStreamIndex,
                           otherStreamIndex,
                           longSOtargetValue,
                           prevLongSOtargetValue,
                           fenceID,
                           true,
                           forceFirmwareFence); // use compareEQUAL, e.g. ==
        prevLongSOtargetValue = longSOtargetValue;
        // wait on fence id
        // use target=1 because waitForOtherStream() configured the monitor payload to increase the fence by 1
        // (Fence wait set it to -1 (fenceCounter -= target), they up it to 0 (fenceCounter++), and it unfreezes when it
        // >=0)
        stream1->h_cmd.FenceCmd(fenceID, 1, forceFirmwareFence, false /* isDirectMode */); // wait
        // after stream1 finished waiting, read the data back
        stream1->h_cmd.lbwRead(readAddr, Addr2Write, dataSize);

        // NOTE! due to force-order initializing so's to 1, user should ++ the target value if force-order is enabled.
        // here it's done inside AllocAndDispatchBarrier
        stream1->AllocAndDispatchBarrier(0, NULL, 0); // target value = 0 because lbwRead/lbwWrite do not increase
        targetStream1++;
        // submit the work
        LOG_DEBUG(SCAL, "submitting stream 1");
        rc |= stream1->stream_submit();
        assert(rc == 0);
        LOG_DEBUG(SCAL, "submitting stream 0");
        rc |= stream0->stream_submit();
        assert(rc == 0);

        // wait for completion
        uint64_t timeout =  30 * 1000000ULL;// 30 seconds, in microsecond, 1/MILION of second
        if (m_params->initialLongSOValue <= 0x10000)
        {
            // the reason we don't enter here when initialLongSOValue > 0x10000
            // is that we use a hack to set the longSO value, and the driver is not aware of this
            // (because Scal::configureMonitors() sets the CQ message (monitor 2) to ++counter)
            // so completion_group_wait would not work.
            LOG_DEBUG(SCAL, "Waiting for stream 0 completion. target {:#x}", targetStream0);
            rc |= stream0->completion_group_wait(targetStream0, timeout);
            assert(rc == 0);
        }
        LOG_DEBUG(SCAL, "Waiting for stream 1 completion. target {:#x}", targetStream1);
        rc |= stream1->completion_group_wait(targetStream1, timeout);
        assert(rc == 0);
        // Compare written value with read value
        for (unsigned j = 0; j < 4; j++)
        {
            if (knownDwords[j] != hostBufAddr[j])
            {
                LOG_ERR(SCAL, "compare error in BlockStreamUnlockWriteTest. j={} knownDwords[j]={:#x} hostBufAddr[j]={:#x}", j, knownDwords[j], hostBufAddr[j]);
                rc |= -1;
                assert(0);
            }
        }
    }
    int rc1 = scal_free_buffer(readBuf.h_Buffer);
    assert(rc1 == 0);
    return rc | rc1;
    /*
        this was debugged with FW printouts to check that indeed every stream did its work:

                cmd_process_lbw_burst_write() on stream 6 scheduler 0 dcore 0
                stream 0 is suspended on scheduler 1 dcore 1
                stream 0 is resumed on scheduler 1 dcore 1
                cmd_process_lbw_read() on stream 0 scheduler 1 dcore 1

    */
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::run_ecb_test(struct test_params* params, bool forceFirmwareFence)
{
    int rc   = 0;
    m_params = params;

    std::string configFilePath = getConfigFilePath(":/default.json");
    //
    //  Init scal , streams and completion groups
    //
    //  WScal will bind each column of stream + completion group + cluster
    //  from the user point of view
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

    WScal wscal(configFilePath.c_str(), {"compute0"}, // streams
                {"compute_completion_queue0"},        // completion groups
                {"compute_tpc"},                      // clusters
                false/*skip direct mode*/, {}, m_fd);

    rc = wscal.getStatus();
    assert(rc == 0);
    if (rc)
    {
        return rc;
    }

    rc |= buildAndSendEcbs(wscal, forceFirmwareFence);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::run_lbw_write_test(struct test_params* params)
{
    int rc   = 0;
    m_params = params;

    std::string configFilePath = getConfigFilePath(":/default.json");
    //
    //  Init scal , streams and completion groups
    //
    //  WScal will bind each column of stream + completion group + cluster
    //  from the user point of view
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

    WScal wscal(configFilePath.c_str(), {"pdma_rx0", "pdma_tx0"},           // streams
                {"compute_completion_queue0", "compute_completion_queue1"}, // completion groups
                {"pdma_rx", "pdma_tx"},                                     // clusters
                false /*skip direct mode*/, {}, m_fd);

    rc = wscal.getStatus();
    assert(rc == 0);
    if (rc)
    {
        return rc;
    }
    rc |= lbwReadWriteTest(wscal);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    motivation for test:
       once we started to work with Auto fetcher of Gaudi3, we've found that scal tests worked great
       but Synapse tests crashed the simulator.
       The difference was that Synapse initialized many streams before starting its work and sent some work on them (monitor setups)
       so this test mimics that by sending work for many streams
*/
int SCAL_StreamSync_Test::run_lbw_write_multi_streams(struct test_params* params)
{
    int rc   = 0;
    m_params = params;

    std::string configFilePath = getConfigFilePath(":/default.json");
    //
    //  Init scal , streams and completion groups
    //
    //  WScal will bind each column of stream + completion group + cluster
    //  from the user point of view
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
                { // streams
                 "compute0", "compute1", "compute2",
                 "pdma_rx0", "pdma_rx1", "pdma_rx2",
                 "pdma_tx0", "pdma_tx1", "pdma_tx2",
                 "pdma_tx_commands0", "pdma_tx_commands1", "pdma_tx_commands2",
                },
                { // completion groups
                 "compute_completion_queue0", "compute_completion_queue1", "compute_completion_queue2",
                 "pdma_rx_completion_queue0", "pdma_rx_completion_queue1", "pdma_rx_completion_queue2",
                 "pdma_tx_completion_queue0", "pdma_tx_completion_queue1", "pdma_tx_completion_queue2",
                 "pdma_tx_commands_completion_queue0", "pdma_tx_commands_completion_queue1", "pdma_tx_commands_completion_queue2",
                },
                { // clusters
                 "compute_tpc", "compute_tpc", "compute_tpc",
                 "pdma_rx", "pdma_rx", "pdma_rx",
                 "pdma_tx", "pdma_tx", "pdma_tx",
                 "pdma_tx", "pdma_tx", "pdma_tx",
                },
                true /*skip direct mode*/, {}, m_fd);

    rc = wscal.getStatus();
    if (rc)
    {
        return rc;
    }

    rc |= lbwReadWriteTest_multi_streams(wscal);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



int SCAL_StreamSync_Test::run_blockStreamUnlockWrite_test(struct test_params* params)
{
    int rc   = 0;
    m_params = params;

    std::string configFilePath = getConfigFilePath(":/default.json");
    // std::string configFilePath = getConfigFilePath("multi_sched.json");

    WScal wscal(configFilePath.c_str(), {"pdma_rx0", "pdma_tx0"},           // streams
                {"compute_completion_queue0", "compute_completion_queue1"}, // completion groups
                {"pdma_rx", "pdma_tx"},                                     // clusters
                false /*skip direct mode*/, {}, m_fd);

    rc = wscal.getStatus();
    assert(rc == 0);
    if (rc)
    {
        return rc;
    }
    rc |= BlockStreamUnlockWriteTest(wscal);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::run_blockStreamUnlockMonitor_test(struct test_params* params,
                                                            bool                forceFirmwareFence)
{
    int rc   = 0;
    m_params = params;

    std::string configFilePath = getConfigFilePath(":/default.json");

    WScal wscal(configFilePath.c_str(), {"pdma_tx0", "pdma_rx0"},           // streams
                {"compute_completion_queue0", "compute_completion_queue1"}, // completion groups
                {"pdma_tx", "pdma_rx"},                                     // clusters
                false /*skip direct mode*/,
                {"pdma_tx_completion_queue0", "pdma_rx_completion_queue0"}, // directMode Completion groups
                m_fd);

    rc = wscal.getStatus();
    assert(rc == 0);
    if (rc)
    {
        return rc;
    }

    rc |= BlockStreamUnlockMonitor(wscal, forceFirmwareFence);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::run_blockStreamUnlockMonitor_test_direct_and_non_direct_pdma(struct test_params* params, bool forceFirmwareFence)
{
    int rc   = 0;
    m_params = params;

    std::string configFilePath = getConfigFilePath(":/default.json");

    WScal wscal(configFilePath.c_str(),
                {"pdma_tx0", "compute0"},                                   // streams
                {"compute_completion_queue0", "compute_completion_queue1"}, // completion groups
                {"pdma_tx", "compute_tpc"},                                 // clusters
                false /*skip direct mode*/,
                {"pdma_tx_completion_queue0", "compute_completion_queue0"}, // directMode Completion groups
                m_fd);

    rc = wscal.getStatus();
    assert(rc == 0);
    if (rc)
    {
        return rc;
    }

    rc |= BlockStreamUnlockMonitor(wscal, forceFirmwareFence);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::run_blockStreamUnlockMonitor_mult_test(struct test_params* params, bool forceFirmwareFence)
{
    int rc   = 0;
    m_params = params;

    std::string configFilePath = getConfigFilePath(":/default.json");

    WScal wscal(configFilePath.c_str(),
                {"pdma_tx0", "pdma_rx0"},                                   // streams
                {"compute_completion_queue0", "compute_completion_queue1"}, // completion groups
                {"pdma_tx", "pdma_rx"},                                     // clusters
                false /*skip direct mode*/,
                {"pdma_tx_completion_queue0", "pdma_rx_completion_queue0"}, // direct mode cgs
                m_fd);

    rc = wscal.getStatus();
    assert(rc == 0);
    if (rc)
    {
        return rc;
    }

    rc |= BlockStreamUnlockMonitorMult(wscal, forceFirmwareFence);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::run_blockStreamUnlockMonitor_mult_test_direct_and_non_direct_pdma(struct test_params* params, bool forceFirmwareFence)
{
    int rc   = 0;
    m_params = params;

    std::string configFilePath = getConfigFilePath(":/default.json");

    WScal wscal(configFilePath.c_str(),
                {"pdma_tx0", "compute0"},                                   // streams
                {"compute_completion_queue0", "compute_completion_queue1"}, // completion groups
                {"pdma_tx", "compute_tpc"},                                 // clusters
                false /*skip direct mode*/,
                {"pdma_tx_completion_queue0", "compute_completion_queue0"}, // direct mode cgs
                m_fd);

    rc = wscal.getStatus();
    assert(rc == 0);
    if (rc)
    {
        return rc;
    }

    rc |= BlockStreamUnlockMonitorMult(wscal, forceFirmwareFence);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::run_blockStreamUnlockMonitor_multsched_test(struct test_params* params, bool forceFirmwareFence)
{
    int rc   = 0;
    m_params = params;

    // multi_sched.json was built such that "network_reduction0" stream
    // handles the MME engine cluster. to be used just for this test.
    // it has also dummy monitor and so sets (e.g. there taken from the leftover monitors and sos)
    // e.g. this is not an official config, it was created just for this demo
    std::string configFilePath;

    if (!m_params->jsonConfig)
        configFilePath = getConfigFilePath("multi_sched.json");
    else
        configFilePath = getConfigFilePath(m_params->jsonConfig);

    /*
        here we test 2 streams synching between different schedulers and different dcores
        the sync method is by using a monitor that checks the other stream long so
    */
    WScal wscal(configFilePath.c_str(),
                {"pdma_tx0", "network_reduction0"},           // streams
                {"compute_completion_queue0", "network_reduction_completion_queue0"}, // completion groups
                {"pdma_tx", "mme"},                                                   // clusters
                false /*skip direct mode*/, {}, m_fd);

    rc = wscal.getStatus();
    assert(rc == 0);
    if (rc)
    {
        return rc;
    }

    rc |= BlockStreamUnlockMonitor2(wscal, forceFirmwareFence);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::run_fence_inc_immediate_multsched_test(struct test_params* params, bool forceFirmwareFence)
{
    int rc;
    m_params = params;

    //    D O E S   N O T    W O R K ! !

    // multi_sched.json was built such that "network_reduction0" stream
    // handles the MME engine cluster. to be used just for this test.
    // it has also dummy monitor and so sets (e.g. there taken from the leftover monitors and sos)
    // e.g. this is not an official config, it was created just for this demo
    std::string configFilePath = getConfigFilePath("multi_sched.json");

    /*
        here we test 2 streams synching between different schedulers and different dcores
    */
    WScal wscal(configFilePath.c_str(),
                {"pdma_tx0", "network_reduction0"},           // streams
                {"compute_completion_queue0", "network_reduction_completion_queue0"}, // completion groups
                {"pdma_tx", "mme"},                                                   // clusters
                false /*skip direct mode*/, {}, m_fd);
    rc = wscal.getStatus();
    assert(rc == 0);
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
    streamBundle* stream0 = wscal.getStreamX(0);

    uint32_t fence_id = 1;
    uint32_t target   = 1;
    stream0->h_cmd.FenceCmd(fence_id, target, forceFirmwareFence, false /* isDirectMode */); // wait on fence id 1
    uint8_t engine_groups[4] = {SCAL_PDMA_TX_CMD_GROUP, 0, 0, 0};
    stream0->AllocAndDispatchBarrier(1, engine_groups);
    rc = stream0->stream_submit();
    assert(rc == 0);

    //
    // stream 1
    //

    streamBundle* stream1         = wscal.getStreamX(1);
    uint8_t      arr_fence_id[1] = {fence_id};
    // NOTE!
    //    This solution works ONLY if both streams run on the
    //    same scheduler.
    stream1->h_cmd.FenceIncImmediate(1, arr_fence_id, forceFirmwareFence);
    uint8_t engine_groups1[4] = {SCAL_MME_COMPUTE_GROUP, 0, 0, 0};
    stream1->AllocAndDispatchBarrier(1, engine_groups1);
    rc = stream1->stream_submit();
    assert(rc == 0);

    // wait for completion of stream 1
    rc = stream1->completion_group_wait(target);
    assert(rc == 0);

    // wait for completion of stream 0
    rc = stream0->completion_group_wait(target);
    assert(rc == 0);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::run_pdma_with_payload_multsched_test(struct test_params* params, bool forceFirmwareFence)
{
    int rc;
    m_params                         = params;
    const unsigned testDmaBufferSize = 4096 * 16; // 1MB

    std::string configFilePath = getConfigFilePath("multi_sched.json");

    /*
        here we test 2 streams synching between different schedulers and different dcores
    */
    WScal wscal(configFilePath.c_str(), {"network_reduction0", "pdma_tx0"},           // streams
                {"network_reduction_completion_queue0", "compute_completion_queue0"}, // completion groups
                {"mme", "pdma_tx"},                                                   // clusters
                false /*skip direct mode*/, {}, m_fd);

    rc = wscal.getStatus();
    assert(rc == 0);
    if (rc)
    {
        return rc;
    }

    // Host DMA buffer - our src buffer
    bufferBundle_t hostDMAbuf;
    rc = wscal.getBufferX(WScal::HostSharedPool, testDmaBufferSize, &hostDMAbuf); // get both buffer handle and info
    assert(rc == 0);
    // HBM DMA buffer - our dest buffer
    bufferBundle_t deviceDMAbuf;
    rc = wscal.getBufferX(WScal::devSharedPool, testDmaBufferSize, &deviceDMAbuf); // get both buffer handle and info
    assert(rc == 0);

    //
    // payload is used for synchronization between the streams
    //
    uint32_t (*createPayload)(uint32_t);
    if (getScalDeviceType() == dtGaudi3)
    {
        createPayload = gaudi3_createPayload;

    }
    else
    {
        createPayload = gaudi2_createPayload;
    }
    uint32_t pay_data = createPayload(0);

    // pay_addr = address of DCCM queue 0
    //   --------->   since we're using 2 different schedulers we need to know the exact address of the other scheduler

    streamBundle* stream0 = wscal.getStreamX(0);
    streamBundle* stream1 = wscal.getStreamX(1); // pdma1 stream
    scal_control_core_infoV2_t coreInfo;
    scal_control_core_get_infoV2(stream0->h_cgInfo.scheduler_handle, &coreInfo);
    assert(coreInfo.dccm_message_queue_address);
    uint32_t pay_addr = coreInfo.dccm_message_queue_address;
    uint32_t fence_id = 0;
    uint32_t target   = 1;
    unsigned workload = -1; // user data

    // stream 0 will wait on fence id 0
    // stream 1 will do DMA and send payload to fence id 0
    // The payload addr contains address of DCCM queue and payload
    // data contains opcode to update corresponding fence
    for (unsigned loopIdx = 0; loopIdx < 100; loopIdx++)
    {
        //
        // stream 0
        //
        stream0->h_cmd.FenceCmd(fence_id, 1, forceFirmwareFence, false /* isDirectMode */); // wait on fence id 0
        uint8_t engine_groups[4] = {m_params->engineGroup, 0, 0, 0};
        stream0->AllocAndDispatchBarrier(1, engine_groups);
        rc = stream0->stream_submit();
        assert(rc == 0);

        //
        // stream 1
        //
        stream1->h_cmd.PdmaTransferCmd(false, deviceDMAbuf.h_BufferInfo.device_address, // dest
                                       hostDMAbuf.h_BufferInfo.device_address,   // src
                                       testDmaBufferSize,                        // size
                                       m_params->engineGroup1,                   // engine group type
                                       workload,
                                       pay_data,
                                       pay_addr);
        uint8_t engine_groups1[4] = {SCAL_PDMA_TX_DATA_GROUP, 0, 0, 0};
        stream1->AllocAndDispatchBarrier(1, engine_groups1);

        rc = stream1->stream_submit();
        assert(rc == 0);
        //
        // wait for completion of stream 1
        //
        rc = stream1->completion_group_wait(target);
        assert(rc == 0);
        //
        // wait for completion of stream 0
        //
        rc = stream0->completion_group_wait(target);
        assert(rc == 0);

        target++;
    }
    // releaseBuffers
    rc = scal_free_buffer(hostDMAbuf.h_Buffer);
    assert(rc == 0);
    rc = scal_free_buffer(deviceDMAbuf.h_Buffer);
    assert(rc == 0);
    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SCAL_StreamSync_Test::run_blockStreamUnlockWrite_multsched_test(struct test_params* params)
{
    int rc;
    m_params = params;

    std::string configFilePath = getConfigFilePath("multi_sched.json");

    /*
        here we test 2 streams synching between different schedulers and different dcores
        sync method:  lbw write into other stream acp register (e.g it's blocked (1), so let's free it by writing 0 ...)
    */
    WScal wscal(configFilePath.c_str(), {"network_reduction0", "pdma_tx0"},           // streams
                {"network_reduction_completion_queue0", "compute_completion_queue0"}, // completion groups
                {"mme", "pdma_tx"},                                                   // clusters
                false /*skip direct mode*/, {}, m_fd);

    rc = wscal.getStatus();
    assert(rc == 0);
    if (rc)
    {
        return rc;
    }

    //   D O E S   N O T  W O R K -  TBD
    m_params->WaitingSchedulerDcoreID = 1; // stream 0 does the waiting, it's network reduction on dcore 1
    m_params->engineGroup             = SCAL_MME_COMPUTE_GROUP;
    m_params->engineGroup1            = SCAL_PDMA_TX_DATA_GROUP;

    rc = BlockStreamUnlockWriteTest(wscal);

    printf("Test Finished\n");
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// simple test (no sync between streams) that just uses the other scheduler (e.g. not compute stream ...)
int SCAL_StreamSync_Test::run_just_use_other_scheduler_test(struct test_params* params)
{
    int rc = 0;
    m_params      = params;

    std::string configFilePath;

    if (!m_params->jsonConfig)
        configFilePath = getConfigFilePath("multi_sched.json");
    else
        configFilePath = getConfigFilePath(m_params->jsonConfig);

    /*
        here we test 2 streams synching between different schedulers and different dcores
        sync method:  lbw write into other stream acp register (e.g it's blocked (1), so let's free it by writing 0 ...)
    */

    WScal* wscal = nullptr;
    if (m_params->streamName && m_params->cqName && m_params->clusterName)
    {
        wscal = new WScal(configFilePath.c_str(), {m_params->streamName}, // streams
                          {m_params->cqName},        // completion groups
                          {m_params->clusterName},   // clusters
                          m_fd);
    }
    else
        wscal = new WScal(configFilePath.c_str(), {"network_reduction0"}, // streams
                          {"network_reduction_completion_queue0"},        // completion groups
                          {"mme"},                                        // clusters
                          m_fd);

    streamBundle* stream0 = wscal->getStreamX(0);
    m_target              = 0;
    uint64_t timeout =  30 * 1000000ULL;// in microsecond, 1/MILION of second
    for (unsigned i = 0; i < params->num_loops; i++)
    {
        // just do NOP
        stream0->h_cmd.NopCmd(0);
        uint8_t engine_groups1[4] = {m_params->engineGroup, 0, 0, 0};
        stream0->AllocAndDispatchBarrier(1, engine_groups1);
        // submit the command buffer on our stream
        rc = stream0->stream_submit();
        assert(rc == 0);
        m_target++;
        LOG_DEBUG(SCAL, "Waiting for stream completion. target {}", m_target);
        rc = stream0->completion_group_wait(m_target, timeout);
        assert(rc == 0);
    }
    delete wscal;
    return rc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, wscal_ecb_test1, {ALL})
{
    struct test_params params;
    params.engineGroup     = SCAL_TPC_COMPUTE_GROUP; // this will do something and release
    params.engineGroup1    = SCAL_MME_COMPUTE_GROUP; // this will wait
    params.engine_cpu_index   = 0;
    params.use_HBM_buffers = 0; // use host shared buffers
    int rc                 = run_ecb_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, wscal_ecb_test2, {ALL})
{
    struct test_params params;
    params.engineGroup   = SCAL_TPC_COMPUTE_GROUP; // this will do something and release
    params.engineGroup1  = SCAL_MME_COMPUTE_GROUP; // this will wait
    params.engine_cpu_index = 0;
    params.use_HBM_buffers = 1; // use HBM buffers
    int rc               = run_ecb_test(&params, GetParam());
    if (rc == SCAL_UNSUPPORTED_TEST_CONFIG)
    {
        GTEST_SKIP() << "this test is not supported with direct mode pdma";
    }
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_lbw_write_test, {ALL})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    int rc = run_lbw_write_test(&params);
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F_CHKDEV(SCAL_StreamSync_Test, wscal_lbw_write_multi_streams, {ALL})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    int rc = run_lbw_write_multi_streams(&params);
    if (rc == SCAL_UNSUPPORTED_TEST_CONFIG)
    {
        GTEST_SKIP() << "not supported in pdma direct mode";
    }
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// this test is using specific Gaudi2 addresses
TEST_F_CHKDEV(SCAL_StreamSync_Test, DISABLED_wscal_blockStream_write_test, {ALL})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    int rc = run_blockStreamUnlockWrite_test(&params);
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, wscal_blockStream_monitor_test , {ALL})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_PDMA_RX_GROUP;
    int rc = run_blockStreamUnlockMonitor_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, wscal_blockStream_monitor_test_direct_and_non_direct_pdma, {GAUDI3})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_PDMA_RX_DEBUG_GROUP;
    int rc = run_blockStreamUnlockMonitor_test_direct_and_non_direct_pdma(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, wscal_blockStream_monitor_mult_test,  {ALL})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_PDMA_RX_GROUP;
    int rc = run_blockStreamUnlockMonitor_mult_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, wscal_blockStream_monitor_mult_test_direct_and_non_direct, {GAUDI3})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_PDMA_RX_DEBUG_GROUP;
    int rc = run_blockStreamUnlockMonitor_mult_test_direct_and_non_direct_pdma(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_sync_streams_on_different_schedulers, {GAUDI2})
{
    // using multi_sched.json
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_MME_COMPUTE_GROUP;
    params.num_loops = 1;
    int rc              = run_blockStreamUnlockMonitor_multsched_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_sync_streams_on_different_schedulers100, {GAUDI2})
{
    // using multi_sched.json
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_MME_COMPUTE_GROUP;
    params.num_loops = 100;
    int rc              = run_blockStreamUnlockMonitor_multsched_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_sync_streams_on_different_schedulers_setLongSO, {GAUDI2})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_MME_COMPUTE_GROUP;
    params.num_loops = 100;
    params.initialLongSOValue = 0x7FFD; // 15 bit
    int rc              = run_blockStreamUnlockMonitor_multsched_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_sync_streams_on_different_schedulers_setLongSO2, {GAUDI2})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_MME_COMPUTE_GROUP;
    params.num_loops = 100;
    params.initialLongSOValue = 0x3FFFFFFD; // 30 bit
    int rc              = run_blockStreamUnlockMonitor_multsched_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_sync_streams_fence_inc_immediate_on_different_schedulers, {GAUDI2})
{
    //  doesn't work, stream 0 just keeps waiting in the fence.
    //  probably that's the way it works. Asked Rakesh and will post his answer here.
    //  Rakesh:
    //  No, it is not expected to work.
    //  Each scheduler has its own Fence counters, so if one scheduler tries to increment other scheduler’s Fence counter it should use
    //  “Monitor/SOB” pair to enqueue a Fence update message into other scheduler.

    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_MME_COMPUTE_GROUP;
    int rc              = run_fence_inc_immediate_multsched_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_sync_streams_pdma_payload_on_different_schedulers, {GAUDI2})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_MME_COMPUTE_GROUP;
    params.engineGroup1 = SCAL_PDMA_TX_DATA_GROUP;
    int rc              = run_pdma_with_payload_multsched_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_sync_streams_lbwWrite_on_different_schedulers, {GAUDI2})
{
    // TBD why it doesn't work
    struct test_params params;
    memset(&params, 0, sizeof(params));
    int rc = run_blockStreamUnlockWrite_multsched_test(&params);
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_sync_streams_on_different_schedulers_forceOrder, {GAUDI2})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_MME_COMPUTE_GROUP;
    params.jsonConfig   = "multi_sched_forceOrder.json";
    int rc              = run_blockStreamUnlockMonitor_multsched_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_sync_streams_on_different_schedulers_slaveSchedulers, {GAUDI2})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_MME_COMPUTE_GROUP;
    params.jsonConfig   = "multi_sched_slaveSched.json";
    int rc              = run_blockStreamUnlockMonitor_multsched_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_P_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_sync_streams_on_different_schedulers_forceOrder_plus_slaveSchedulers, {GAUDI2})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.engineGroup  = SCAL_PDMA_TX_DATA_GROUP;
    params.engineGroup1 = SCAL_MME_COMPUTE_GROUP;
    params.jsonConfig   = "multi_sched_forceOrder+slaveSched.json";
    int rc              = run_blockStreamUnlockMonitor_multsched_test(&params, GetParam());
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_just_use_other_scheduler, {GAUDI2})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.num_loops = 10;
    params.engineGroup = SCAL_MME_COMPUTE_GROUP;
    int rc = run_just_use_other_scheduler_test(&params);
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_just_use_other_scheduler_force_order, {GAUDI2})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.jsonConfig = "multi_sched_forceOrder.json";
    params.num_loops  = 10;
    params.engineGroup = SCAL_MME_COMPUTE_GROUP;
    int rc            = run_just_use_other_scheduler_test(&params);
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_just_use_other_scheduler_slaveSchedulers, {GAUDI2})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.jsonConfig = "multi_sched_slaveSched.json";
    params.num_loops  = 10;
    params.engineGroup = SCAL_MME_COMPUTE_GROUP;
    int rc            = run_just_use_other_scheduler_test(&params);
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F_CHKDEV(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_just_use_other_scheduler_forceOrder_plus_slaveSchedulers, {GAUDI2})
{
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.jsonConfig = "multi_sched_forceOrder+slaveSched.json";
    params.num_loops  = 10;
    params.engineGroup = SCAL_MME_COMPUTE_GROUP;
    int rc            = run_just_use_other_scheduler_test(&params);
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(SCAL_StreamSync_Test, DISABLED_multi_sched_wscal_just_use_other_scheduler2)
{
    /* hard to remember, but HCL schedulers do  not support alloc/dispatch barriers
       so the following setup does not work.
       I will update this when I understand better how they supposed to work
    */
    struct test_params params;
    memset(&params, 0, sizeof(params));
    params.jsonConfig = ":/default.json";
    params.num_loops = 10;
    params.streamName = "scaleup_receive0";
    params.cqName = "network_completion_queue_internal_00";
    params.clusterName = "nic_scaleup";
    params.engineGroup = SCAL_NIC_RECEIVE_SCALE_UP_GROUP;
    int rc = run_just_use_other_scheduler_test(&params);
    ASSERT_EQ(rc, 0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
