#include <iostream>
#include <algorithm>
#include <infra/monitor.hpp>
#include "infra/monitor.hpp"

#include "scal.h"
#include "hlthunk.h"
#include "logger.h"

#include "gaudi3_arc_sched_packets.h"
#include "gaudi3/asic_reg/arc_acp_eng_regs.h"
#include "gaudi3/asic_reg_structs/arc_acp_eng_regs.h"

#define varoffsetof(t, m) ((size_t)(&(((t*)0)->m)))

#include "scal_basic_test.h"
#include "scal_gaudi3_sync_monitor.h"


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#define LOCAL_ACP_OFFSET                 0xD000  // note, different from gaudi2

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
// if we are in direct mode we will increment a SOB == fence counter of the waiting stream
// address payload - used only in direct mode
// data payload - sed only in direct mode
int gaudi3_configMonitorForLongSO(uint64_t           smBase,
                                  scal_core_handle_t coreHandle,
                                  unsigned           monitorID,
                                  unsigned           longSoID,
                                  bool               compareEQ,
                                  uint64_t           payloadAddr,
                                  uint32_t           payloadData,
                                  uint64_t           addr[7],
                                  uint32_t           value[7])
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

    /*

            instead of using if (monid < 1024)  use mon_config_0/mon_arm_0 etc
                             else               use mon_config_1/mon_arm_1 etc

            Scal sets smBase like in:

                if (smId & 0x1)
                {
                    smBase += offsetof(gaudi3::block_sob_objs, sob_obj_1);
                }

            and ASSUMES that the whole layout of 0 & 1 is the same  (which is the case , as of now)
               e.g always use mon_config_0  (but if mondId>=1024) scal changes the offset of smBase such that it ACTUALLY falls in the area of mon_config_1 etc.
               This trick is also used in Synapse

            SO
               since we let scal set our input smBase
               (and scal works in sm terms, sm0 points to the 1st half of struct block_sob_objs of HDCORE 0
                                        and sm1 points to the 2nd half of struct block_sob_objs
                                        sm3,4 is the same for HDCORE 1  etc.
               we must assume monitorID < 1024 and longSoID < 8192 - 4


    */
    // actually since here we use monitorID+3, we limit monitorID to 1020

    if ((monitorID % 4 != 0) || (longSoID % 4 != 0) || (longSoID > 8188) || (monitorID > 1020))
    {
        LOG_ERR(SCAL, "{}: monitorID {} should be x4, longSoID {} should be x4, monitorID < 1021 (since we use Scal smBase) and longSoID <= 8188", __FUNCTION__, monitorID, longSoID);
        assert(0);
        return SCAL_FAILURE;
    }
    unsigned index  = 0;
    unsigned soGroupIdx = longSoID >> 3; // every monitor guards a group of 8 sos

    //LOG_DEBUG(SCAL, "{}: configure monitor {} to watch longSO {}  (group {}) to release fenceId={} of scheduler {} {}", __FUNCTION__, monitorID, longSoID, soGroupIdx, fenceID, coreInfo.idx, coreInfo.name);

    // monitor config   (mc)

    //  (monitorID < 1024) !!! (since we use Scal smBase - see remark above)
    MonitorG3 monitor(smBase, monitorID, 0);
    uint32_t mc = MonitorG3::buildConfVal(longSoID, 0, CqEn::off, LongSobEn::on, LbwEn::off);

    addr[index]    = monitor.getRegsAddr().config;
    value[index++] = mc;

    // The 64 bit address to write the completion message to in case CQ_EN=0.
    // set payload addr Hi & Low
    addr[index]    = monitor.getRegsAddr().payAddrH;
    value[index++] = upper_32_bits(payloadAddr);
    addr[index]    = monitor.getRegsAddr().payAddrL;
    value[index++] = lower_32_bits(payloadAddr);

    // configure the monitor payload data to send a fence update msg to the dccm Q of the scheduler
    addr[index]    = monitor.getRegsAddr().payData;
    value[index++] = payloadData;

    // set monArm[1..3] to 0.
    for (unsigned i = 1; i <= 3; i++)
    {
        MonitorG3 monitorI(smBase, monitorID + i, 0);
        addr[index]    = monitorI.getRegsAddr().arm;
        value[index++] = 0;
    }

    return 0;
}

int gaudi3_armMonitorForLongSO(uint64_t smBase, unsigned monitorID, unsigned longSoID, uint64_t longSOtargetValue, uint64_t prevLongSOtargetValue, bool compareEQ, uint64_t addr[7],
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

    //  (monitorID < 1024) !!! (since we use Scal smBase - see remark in gaudi3_configMonitorForLongSO() above)

    // actually since here we use monitorID+3, we limit monitorID to 1020

    if ((monitorID % 4 != 0) || (longSoID % 4 != 0) || (longSoID > 8188) || (monitorID > 1020))
    {
        LOG_ERR(SCAL, "{}: monitorID {} should be x4, longSoID {} should be x4, monitorID < 1021 (since we use Scal smBase) and longSoID <= 8188", __FUNCTION__, monitorID, longSoID);
        assert(0);
        return SCAL_FAILURE;
    }
    numPayloads = 0;

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
            uint32_t ma = MonitorG3::buildArmVal(longSoID, newT & 0x7FFF, compType);

            //LOG_DEBUG(SCAL, "{}: arming extra mon {} for value {:#x}", __FUNCTION__, monitorID + i, ma.sod);
            addr[numPayloads]    = MonitorG3(smBase, monitorID + 1).getRegsAddr().arm;
            value[numPayloads++] = ma;
        }
    }
    // keep this last
    // must be done EVERY time
    CompType compType = compareEQ ? CompType::EQUAL : CompType::BIG_EQUAL;
    uint32_t ma = MonitorG3::buildArmVal(longSoID, longSOtargetValue, compType);

    addr[numPayloads]    = MonitorG3(smBase, monitorID).getRegsAddr().arm;
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
                                                      when fired, unfreeze AcpFence 0
                                                 AcpFenceWait on AcpFence 0

                                     stream B -  do PDMA from host to device
                                                 fill_alloc_barrier_cmd
                                                 fill_dispatch_barrier_cmd // this will, eventually, inc the longSO


*/
// clang-format on
////////////////////////////////////////////////////////////////////////////
#include "gaudi3/gaudi3_packets.h"

uint32_t gaudi3_qman_add_nop_pkt(void* buffer, uint32_t buf_off, enum QMAN_EB eb, enum QMAN_MB mb)
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

uint32_t gaudi3_getAcpRegAddr(unsigned halfDcoreID /* unused */)
{
    uint32_t acpBaseAddr = 0xC500000;
    return (acpBaseAddr + LOCAL_ACP_OFFSET + mmARC_ACP_ENG_QSEL_MASK_0);
}


uint32_t gaudi3_createPayload(uint32_t fence_id)
{
        union sched_mon_exp_msg_t pay_data;
        memset(&pay_data, 0, sizeof(sched_mon_exp_msg_t));
        pay_data.fence.opcode   = MON_EXP_FENCE_UPDATE;
        pay_data.fence.fence_id = fence_id;
        return pay_data.raw;
}

void gaudi3_getAcpPayload(uint64_t& payloadAddress, uint32_t& payloadData, uint32_t fence_id, uint64_t acpEngBaseAddr)
{
    gaudi3::arc_acp_eng::reg_qsel_mask_counter maskCounter;
    maskCounter._raw  = 0;
    maskCounter.op    = 1 /* Operation is Add*/;
    maskCounter.value = 1;
    payloadData = maskCounter._raw;

    payloadAddress = acpEngBaseAddr + varoffsetof(gaudi3::block_arc_acp_eng, qsel_mask_counter[fence_id]);
}
