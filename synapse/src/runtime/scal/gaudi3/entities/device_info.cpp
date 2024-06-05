#include "device_info.hpp"

#include "defs.h"
#include "utils.h"

#include "runtime/scal/common/entities/scal_monitor.hpp"

// engine-arc
#include "gaudi3_arc_common_packets.h"  // QMAN_ENGINE_GROUP_TYPE_COUNT
#include "gaudi3_arc_eng_packets.h"     // XXX_COMPUTE_ECB_LIST_BUFF_SIZE
#include "gaudi3_arc_sched_packets.h"   // sched_mon_exp_fence_t, SCHED_PDMA_MAX_BATCH_TRANSFER_COUNT

// specs
#include "gaudi3/asic_reg_structs/arc_acp_eng_regs.h" // block_arc_acp_eng
#include "gaudi3/asic_reg_structs/sob_objs_regs.h"    // block_sob_objs
#include "gaudi3/asic_reg/gaudi3_blocks.h"            // SMs' Address
#include "gaudi3/gaudi3.h"

// TODO - utils infra
#define upper_32_bits(n)  ((uint32_t)(((n) >> 16) >> 16))
#define lower_32_bits(n)  ((uint32_t)(n))

using namespace gaudi3;

uint64_t DeviceInfo::getSosAddr(unsigned dcore, unsigned sos) const
{
    return getSmBase(dcore) + varoffsetof(gaudi3::block_sob_objs, sob_obj_0[sos]);
}

uint32_t DeviceInfo::getSosValue(unsigned value, bool isLongSo, bool isTraceEvict, bool isIncrement) const
{
    gaudi3::sob_objs::reg_sob_obj_0 data {};

    data.val         = value;
    data.long_sob    = isLongSo;
    data.trace_evict = isTraceEvict;
    data.inc         = isIncrement;

    return data._raw;
}

uint32_t DeviceInfo::getMonitorConfigRegisterValue(bool     isLongSob,
                                                   bool     isUpperLongSoBits,
                                                   bool     isCqEnabled,
                                                   unsigned writesNum,
                                                   unsigned sobjGroup) const
{
    gaudi3::sob_objs::reg_mon_config_0 mc;

    mc._raw = 0;

    // Monitoring a long sob
    mc.long_sob = isLongSob;

    // Defines which SOB’s would be used for 60xbit count.   “0”: Lower 4xSOB’s   “1”: Upper 4xSOB’s
    mc.long_high_group = isUpperLongSoBits;

    // Indicates the monitor is associated with a completion queue  (if you set to 1 you get the
    // error "Monitor was armed unsecured. It can't write to a secured CQ (completion queue))""
    mc.cq_en = isCqEnabled;

    // 0 ==> writes 1 time , e.g does 1 thing, 1 ==> writes 2 times
    mc.wr_num = writesNum;

    mc.msb_sid = sobjGroup;
    HB_ASSERT(sobjGroup == 0, "Assuming all msb_sid are equal to 0 for all cg_long_sob");

    return mc._raw;
}

uint32_t DeviceInfo::getMonitorArmRegisterValue(unsigned longSobTargetValue,
                                                bool     isEqualOperation,
                                                bool     maskValue,
                                                unsigned sobjId) const
{
    gaudi3::sob_objs::reg_mon_arm_0 ma;

    ma._raw = 0;
    ma.sod  = longSobTargetValue;

    // Operation to perform: >= (SOP=0) or == (SOP=1)
    ma.sop = isEqualOperation;

    // Indicates which of the 8 sync objects should be masked from monitoring
    // "If the nth mask bit is set, the nth sync object in the group should NOT be monitored"
    ma.mask = maskValue;

    ma.sid = sobjId;

    return ma._raw;
}

uint32_t DeviceInfo::getSchedMonExpFenceCommand(FenceIdType fenceId) const
{
    sched_mon_exp_fence_t monExpFenceCommand;

    monExpFenceCommand.opcode   = MON_EXP_FENCE_UPDATE;
    monExpFenceCommand.fence_id = fenceId;
    monExpFenceCommand.reserved = 0;

    return monExpFenceCommand.raw;
}

uint64_t DeviceInfo::getQueueSelectorMaskCounterAddress(unsigned smIndex, FenceIdType fenceId) const
{
    return getArcAcpEng(smIndex) + varoffsetof(gaudi3::block_arc_acp_eng, qsel_mask_counter[fenceId]);
}

uint64_t DeviceInfo::getQueueSelectorMaskCounterValue(common::QueueSelectorOperation operation, unsigned value) const
{
    gaudi3::arc_acp_eng::reg_qsel_mask_counter maskCounter;

    maskCounter._raw  = 0;
    maskCounter.op    = (operation == common::QueueSelectorOperation::ADD) ? 1 : 0;
    maskCounter.value = value;

    return maskCounter._raw;
}

uint64_t DeviceInfo::getMsixAddrress() const
{
    return mmD0_PCIE_MSIX_BASE;
}

uint32_t DeviceInfo::getMsixUnexpectedInterruptValue() const
{
    return RESERVED_MSIX_UNEXPECTED_USER_ERROR_INTERRUPT;
}

uint32_t DeviceInfo::getStaticComputeEcbListBufferSize() const
{
    return STATIC_COMPUTE_ECB_LIST_BUFF_SIZE;
}

uint32_t DeviceInfo::getDynamicComputeEcbListBufferSize() const
{
    return DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;
}

uint32_t DeviceInfo::getQmanEngineGroupTypeCount() const
{
    return QMAN_ENGINE_GROUP_TYPE_COUNT;
}

uint32_t DeviceInfo::getSchedPdmaCommandsTransferMaxParamCount() const
{
    return SCHED_PDMA_MAX_BATCH_TRANSFER_COUNT;
}

uint64_t DeviceInfo::getSmBase(unsigned smId) const
{
    uint64_t smBase = 0;
    switch (smId / 2)
    {
        case 0:
            smBase = mmHD0_SYNC_MNGR_OBJS_BASE;
            break;
        case 1:
            smBase = mmHD1_SYNC_MNGR_OBJS_BASE;
            break;
        case 2:
            smBase = mmHD2_SYNC_MNGR_OBJS_BASE;
            break;
        case 3:
            smBase = mmHD3_SYNC_MNGR_OBJS_BASE;
            break;
        case 4:
            smBase = mmHD4_SYNC_MNGR_OBJS_BASE;
            break;
        case 5:
            smBase = mmHD5_SYNC_MNGR_OBJS_BASE;
            break;
        case 6:
            smBase = mmHD6_SYNC_MNGR_OBJS_BASE;
            break;
        case 7:
            smBase = mmHD7_SYNC_MNGR_OBJS_BASE;
            break;
        default:
            HB_ASSERT(0, "invalid dcoreId");
    }

    if (smId & 0x1)
    {
        smBase += offsetof(gaudi3::block_sob_objs, sob_obj_1);
    }

    return smBase;
}

uint64_t DeviceInfo::getMonitorPayloadAddrHighRegisterSmOffset(MonitorIdType monitorId) const
{
    return varoffsetof(gaudi3::block_sob_objs, mon_pay_addrh_0[monitorId]);
}

uint64_t DeviceInfo::getMonitorPayloadAddrLowRegisterSmOffset(MonitorIdType monitorId) const
{
    return varoffsetof(gaudi3::block_sob_objs, mon_pay_addrl_0[monitorId]);
}

uint64_t DeviceInfo::getMonitorPayloadDataRegisterSmOffset(MonitorIdType monitorId) const
{
    return varoffsetof(gaudi3::block_sob_objs, mon_pay_data_0[monitorId]);
}

uint64_t DeviceInfo::getMonitorConfigRegisterSmOffset(MonitorIdType monitorId) const
{
    return varoffsetof(gaudi3::block_sob_objs, mon_config_0[monitorId]);
}

uint64_t DeviceInfo::getMonitorArmRegisterSmOffset(MonitorIdType monitorId) const
{
    return varoffsetof(gaudi3::block_sob_objs, mon_arm_0[monitorId]);
}

uint64_t DeviceInfo::getArcAcpEng(unsigned smIndex) const
{
    const unsigned smAmount = 16;

    HB_ASSERT_DEBUG_ONLY((smIndex < smAmount), "invalid smIndex");

    uint64_t smBase = 0;

    switch (smIndex)
    {
        case 0:
            smBase = mmHD0_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 1:
            smBase = mmHD0_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 2:
            smBase = mmHD1_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 3:
            smBase = mmHD1_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 4:
            smBase = mmHD2_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 5:
            smBase = mmHD2_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 6:
            smBase = mmHD3_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 7:
            smBase = mmHD3_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 8:
            smBase = mmHD4_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 9:
            smBase = mmHD4_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 10:
            smBase = mmHD5_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 11:
            smBase = mmHD5_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 12:
            smBase = mmHD6_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 13:
            smBase = mmHD6_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 14:
            smBase = mmHD7_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 15:
            smBase = mmHD7_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        default:
            HB_ASSERT(0, "invalid dcoreId");
    }

    return smBase;
}

uint16_t DeviceInfo::getNumArcCpus() const
{
    return CPU_ID_MAX;
}
