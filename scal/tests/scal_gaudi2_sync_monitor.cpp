#include <iostream>
#include <algorithm>
#include "scal_basic_test.h"
#include "scal.h"
#include "hlthunk.h"
#include "logger.h"
#include "scal_test_utils.h"


#include "gaudi2_arc_sched_packets.h"
#include "gaudi2/asic_reg/gaudi2_blocks.h"

#include "gaudi2/asic_reg_structs/qman_arc_aux_regs.h"

#include "gaudi2/asic_reg/arc_farm_arc0_acp_eng_regs.h"
#include "gaudi2/asic_reg/arc_farm_arc1_acp_eng_regs.h"
#include "gaudi2/asic_reg/arc_farm_arc2_acp_eng_regs.h"
#include "gaudi2/asic_reg/arc_farm_arc3_acp_eng_regs.h"

#include "scal_gaudi2_sync_monitor.h"
#include "infra/monitor.hpp"
#include "infra/sob.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#define LOCAL_ACP_OFFSET                 0xF000

using CqEn      = Monitor::CqEn;
using LongSobEn = Monitor::LongSobEn;
using LbwEn     = Monitor::LbwEn;
using CompType  = Monitor::CompType;

//#pragma GCC diagnostic pop

// clang-format off


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  configure monitor (monitorID)
//  to watch long SO  (longSoID) reach target (longSOtargetValue)
//  and when it does, unblock fence (fenceID)
//
// the output is a list of  (address, value) pairs (uint64_t,uint32_t) which the caller should write
// either with scheduler lbw_write_cmd command   OR with  QMAN msgLong
int gaudi2_configMonitorForLongSO(uint64_t smBase, scal_core_handle_t coreHandle, unsigned monitorID, unsigned longSoID, unsigned fenceID, bool compareEQ, unsigned syncObjIdForDummyPayload, uint64_t smBaseForDummyPayload,
                                  uint64_t addr[10], uint32_t value[10])
{
    // This function should not arm the monitor and should be called only once.

    /*
        To support 60 bit value in longSOB, we need to use 4 monitors, chained to a "long monitor" by the longSOB = 1 bit
        This implies that
        * each of the 4 monitor SOD field (when arming) holds the respective 15 bit value
        * it only fires 1 msg
        * monitor 0 index must be 32 byte aligned --> since each entry is 4 bytes, index should be diviseable by 8
        * arming mon 1..3 must come BEFORE arm0
        * we need only to config the 1st monitor
        * we must arm only the monitors that are expected to change
        * e.g. we need to compare the prev val with the new one and reconfigure val15_29, val30_44 and val45_59 only if they change.
          val_0_15 should be armed anyway.
    */
    if ((monitorID % 4 != 0) || (longSoID % 8 != 0))
    {
        LOG_ERR(SCAL, "{}: monitorID {} should be x4, longSoID {} should be x8", __FUNCTION__, monitorID, longSoID);
        assert(0);
        return SCAL_FAILURE;
    }
    LOG_INFO(SCAL, "{}: smBase {} monitorID {} longSoID {} fenceID {} compareEQ {}", __FUNCTION__, smBase, monitorID, longSoID, fenceID, compareEQ);
    unsigned index  = 0;
    unsigned soGroupIdx = longSoID >> 3; // every monitor guards a group of 8 sos

    scal_control_core_infoV2_t coreInfo;
    scal_control_core_get_infoV2(coreHandle, &coreInfo);
    assert(coreInfo.dccm_message_queue_address);
    const bool useDummyPayload = syncObjIdForDummyPayload != std::numeric_limits<decltype(syncObjIdForDummyPayload)>::max();
    int wrNum = useDummyPayload ? 1 : 0;
    //LOG_DEBUG(SCAL, "{}: configure monitor {} to watch longSO {}  (group {}) to release fenceId={} of scheduler {} {}", __FUNCTION__, monitorID, longSoID, soGroupIdx, fenceID, coreInfo.idx, coreInfo.name);

    // monitor config   (mc)
    uint32_t confVal = MonitorG2::buildConfVal(longSoID, wrNum, CqEn::off, LongSobEn::on, LbwEn::off);
    assert((longSoID % 8) == 0);

    MonitorG2 monitor(smBase, monitorID);

    addr[index]    = monitor.getRegsAddr().config;
    value[index++] = confVal;

    // The 64 bit address to write the completion message to in case CQ_EN=0.
    // set payload addr Hi & Low
    addr[index]    = monitor.getRegsAddr().payAddrH;
    value[index++] = upper_32_bits(coreInfo.dccm_message_queue_address);
    addr[index]    = monitor.getRegsAddr().payAddrL;
    value[index++] = lower_32_bits(coreInfo.dccm_message_queue_address);

    // configure the monitor payload data to send a fence update msg to the dccm Q of the scheduler
    struct g2fw::sched_mon_exp_fence_t pd;
    pd.opcode   = g2fw::MON_EXP_FENCE_UPDATE;
    pd.fence_id = fenceID;
    pd.reserved = 0;
    addr[index] = monitor.getRegsAddr().payData;
    value[index++] = pd.raw;

    if (useDummyPayload)
    {
        // 2nd dummy message to DCORE0_SYNC_MNGR_OBJS SOB_OBJ_8184 (as w/a for SM bug in H6 - SW-67146)
        uint64_t sosAddress = SobG2::getAddr(smBaseForDummyPayload, syncObjIdForDummyPayload);
        //
        // set payload addr Hi & Low
        MonitorG2 monitorP1(smBase, monitorID + 1);

        addr[index]    = monitorP1.getRegsAddr().payAddrH;
        value[index++] = upper_32_bits(sosAddress);
        addr[index]    = monitorP1.getRegsAddr().payAddrL;
        value[index++] = lower_32_bits(sosAddress);
        //
        // configure the monitor payload data to send a dummy data to to SOB_OBJ_8184
        addr[index]    = monitorP1.getRegsAddr().payData;
        value[index++] = 0;
    }
    // set monArm[1..3] to 0.
    for (unsigned i = 1; i <= 3; i++)
    {
        MonitorG2 monitorI(smBase, monitorID + i);
        addr[index]    = monitorI.getRegsAddr().arm;
        value[index++] = 0;
    }
    return 0;
}

int gaudi2_armMonitorForLongSO(uint64_t smBase, unsigned monitorID, unsigned longSoID, uint64_t longSOtargetValue, uint64_t prevLongSOtargetValue, bool compareEQ, uint64_t addr[7],
                                              uint32_t value[7], unsigned &numPayloads)
{
    /*
        To support 60 bit value in longSOB, we need to use 4 monitors, chained to a "long monitor" by the longSOB = 1 bit
        This implies that
        * each of the 4 monitor SOD field (when arming) holds the respective 15 bit value
        * it only fires 1 msg
        * monitor 0 index must be 32 byte aligned --> since each entry is 4 bytes, index should be diviseable by 8
        * arming mon 1..3 must come BEFORE arm0
        * we need only to config the 1st monitor
        * we must arm only the monitors that are expected to change
        * e.g. we need to compare the prev val with the new one and reconfigure val15_29, val30_44 and val45_59 only if they change.
          val_0_15 should be armed anyway.
    */
    numPayloads = 0;

    LOG_INFO(SCAL, "{}: smBase {} monitorID {} longSoID {} longSOtargetValue {} prevLongSOtargetValue {} compareEQ {}", __FUNCTION__,
        smBase, monitorID, longSoID, longSOtargetValue, prevLongSOtargetValue, compareEQ);
    // arm the monitors
    uint64_t prevT = prevLongSOtargetValue;
    uint64_t newT  = longSOtargetValue;
    for (unsigned i = 1; i <= 3; i++)
    {
        prevT = prevT >> 15;
        newT  = newT >> 15;
        // compare the prev val with the new one and reconfigure val15_29, val30_44 and val45_59 only if they change.
        if ((prevT & 0x7FFF) != (newT & 0x7FFF))
        {
            CompType compType = compareEQ ? CompType::EQUAL : CompType::BIG_EQUAL;
            uint32_t ma       = MonitorG2::buildArmVal(longSoID, newT & 0x7FFF, compType);

            //LOG_DEBUG(SCAL, "{}: arming extra mon {} for value {:#x}", __FUNCTION__, monitorID + i, ma.sod);
            addr[numPayloads]    = MonitorG2(smBase, monitorID + i).getRegsAddr().arm;
            value[numPayloads++] = ma;
        }
    }
    // keep this last
    // must be done EVERY time
    CompType compType = compareEQ ? CompType::EQUAL : CompType::BIG_EQUAL;
    uint32_t ma       = MonitorG2::buildArmVal(longSoID, longSOtargetValue, compType);

    addr[numPayloads]    = MonitorG2(smBase, monitorID).getRegsAddr().arm;
    value[numPayloads++] = ma;

    return SCAL_SUCCESS;
}

/*

   stream synchronization - several ways
   =========================================
       stream A is the the stream that waits
       stream B will unfreeze stream A

        -  1 SCAL_Fence_Test.fence_test     (in scal_test_fence.cpp)
                                     stream A -  fill_fence_wait_cmd (fence id = 1, target = 1)

                                     stream B -  fill_fence_inc_immediate_cmd  (array,1)  where array of size 1 has {1} e.g. fence id=1


        - 2  SCAL_Fence_Test.barrier_test* (in scal_test_fence.cpp)
                                     stream A -  fill_fence_wait_cmd (fence id = 0, target 1)

                                     stream B -  fill_pdma_transfer_cmd (src,dest,size
                                                    + payload data is a msg to inc fence id 0,
                                                    + payload address is address of the scheduler DCCM queue 0


        - 3  wscal_blockStream_write_test
                                     stream A - // pdma0 - write data and block=true
                                               fill_lbw_write_cmd(someaddr,somedata, block=true) (or fill_lbw_burst_write_cmd for 4 dwords)

                                     stream B -  // pdma1 - Unmask other stream by writing to lbu_acp_mask register
                                               fill_lbw_write_cmd(ARC_LBU_ADDR(acpMaskRegAddr) + (streamAInfo.index*4), 0, false);)
                                               fill_mem_fence_cmd(1, 0, 1)  (wait for memory ops)

         - 4 wscal_blockStream_monitor_test
                                     stream A -  config and arm a monitor to watch the longSO of stream B
                                                      when fired, unfreeze fence 0
                                                 FenceWait on fence 0

                                     stream B -  do PDMA from host to device
                                                 fill_alloc_barrier_cmd
                                                 fill_dispatch_barrier_cmd // this will, eventually, inc the longSO


*/
// clang-format on
////////////////////////////////////////////////////////////////////////////
#include "gaudi2/gaudi2_packets.h"

static int32_t getAcpOffsetVal(uint32_t offset)
{
    uint32_t i, retVal = 0x0;
    for (i = 0; i < (sizeof(gaudi2::block_qman_arc_aux_defaults) / sizeof(gaudi2::block_qman_arc_aux_defaults[0])); i++)
    {
        if (gaudi2::block_qman_arc_aux_defaults[i].offset == offset)
        {
            retVal = gaudi2::block_qman_arc_aux_defaults[i].val;
            break;
        }
    }
    return retVal;
}

uint32_t gaudi2_qman_add_nop_pkt(void* buffer, uint32_t buf_off, enum QMAN_EB eb, enum QMAN_MB mb)
{
    struct packet_nop packet;

    memset(&packet, 0, sizeof(packet));
    packet.opcode      = PACKET_NOP;
    packet.eng_barrier = eb;
    packet.msg_barrier = mb;

    packet.ctl = htole32(packet.ctl);
    memcpy((uint8_t*)buffer + buf_off, &packet, sizeof(packet));

    return buf_off + sizeof(packet);
}

uint32_t gaudi2_getAcpRegAddr(unsigned dcoreID)
{

    uint32_t                 acpBaseAddr    = getAcpOffsetVal(ARC_ACC_ENGS_VIRTUAL_ADDR_OFFSET) + LOCAL_ACP_OFFSET;
    uint32_t                 acpMaskRegAddr = 0;

    // just a note , when using & 0xFFF  all these are the same ....  (0x300)
    switch (dcoreID)
    {
    case 0:
        acpMaskRegAddr = acpBaseAddr + (mmARC_FARM_ARC0_ACP_ENG_ACP_MK_REG_0 & 0xFFF);
        break;
    case 1:
        acpMaskRegAddr = acpBaseAddr + (mmARC_FARM_ARC1_ACP_ENG_ACP_MK_REG_0 & 0xFFF);
        break;
    case 2:
        acpMaskRegAddr = acpBaseAddr + (mmARC_FARM_ARC2_ACP_ENG_ACP_MK_REG_0 & 0xFFF);
        break;
    case 3:
        acpMaskRegAddr = acpBaseAddr + (mmARC_FARM_ARC3_ACP_ENG_ACP_MK_REG_0 & 0xFFF);
        break;
    default:
        assert(0);
        break;
    };
    return acpMaskRegAddr;
}

uint32_t gaudi2_createPayload(uint32_t fence_id)
{
    union g2fw::sched_mon_exp_msg_t pay_data;
    memset(&pay_data, 0, sizeof(g2fw::sched_mon_exp_msg_t));
    pay_data.fence.opcode   = g2fw::MON_EXP_FENCE_UPDATE;
    pay_data.fence.fence_id = fence_id;
    return pay_data.raw;
}
