#include "device_info.hpp"

#include "defs.h"
#include "utils.h"

#include "runtime/scal/common/entities/scal_monitor.hpp"

// engine-arc
#include "gaudi2_arc_common_packets.h"  // QMAN_ENGINE_GROUP_TYPE_COUNT
#include "gaudi2_arc_eng_packets.h"     // XXX_COMPUTE_ECB_LIST_BUFF_SIZE
#include "gaudi2_arc_sched_packets.h"   // sched_mon_exp_fence_t, SCHED_PDMA_MAX_BATCH_TRANSFER_COUNT

// specs
#include "gaudi2/asic_reg_structs/sob_objs_regs.h"  // block_sob_objs
#include "gaudi2/asic_reg/gaudi2_blocks.h"          // SMs' Address
#include "gaudi2/gaudi2.h"

// TODO - utils infra
#define upper_32_bits(n)  ((uint32_t)(((n) >> 16) >> 16))
#define lower_32_bits(n)  ((uint32_t)(n))

using namespace gaudi2;

uint64_t DeviceInfo::getSosAddr(unsigned dcore, unsigned sos) const
{
    return getSmBase(dcore) + varoffsetof(gaudi2::block_sob_objs, sob_obj[sos]);
}

uint32_t DeviceInfo::getSosValue(unsigned value, bool isLongSo, bool isTraceEvict, bool isIncrement) const
{
    gaudi2::sob_objs::reg_sob_obj data {};

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
    gaudi2::sob_objs::reg_mon_config mc;

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
    gaudi2::sob_objs::reg_mon_arm ma;

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
    HB_ASSERT(false, "Not supported for Gaudi2");
    return std::numeric_limits<uint64_t>::max();
}

uint64_t DeviceInfo::getQueueSelectorMaskCounterValue(common::QueueSelectorOperation operation, unsigned value) const
{
    HB_ASSERT(false, "Not supported for Gaudi2");
    return std::numeric_limits<uint64_t>::max();
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

uint64_t DeviceInfo::getSmBase(unsigned dcoreId) const
{
    uint64_t smBase = 0;
    switch (dcoreId)
    {
        case 0:
            smBase = mmDCORE0_SYNC_MNGR_OBJS_BASE;
            break;
        case 1:
            smBase = mmDCORE1_SYNC_MNGR_OBJS_BASE;
            break;
        case 2:
            smBase = mmDCORE2_SYNC_MNGR_OBJS_BASE;
            break;
        case 3:
            smBase = mmDCORE3_SYNC_MNGR_OBJS_BASE;
            break;
        default:
            HB_ASSERT(0, "invalid dcoreId");
    }
    return smBase;
}

uint64_t DeviceInfo::getMsixAddrress() const
{
    return RESERVED_VA_FOR_VIRTUAL_MSIX_DOORBELL_START;
}

uint32_t DeviceInfo::getMsixUnexpectedInterruptValue() const
{
    return RESERVED_MSIX_UNEXPECTED_USER_ERROR_INTERRUPT;
}

uint64_t DeviceInfo::getMonitorPayloadAddrHighRegisterSmOffset(MonitorIdType monitorId) const
{
    return varoffsetof(gaudi2::block_sob_objs, mon_pay_addrh[monitorId]);
}

uint64_t DeviceInfo::getMonitorPayloadAddrLowRegisterSmOffset(MonitorIdType monitorId) const
{
    return varoffsetof(gaudi2::block_sob_objs, mon_pay_addrl[monitorId]);
}

uint64_t DeviceInfo::getMonitorPayloadDataRegisterSmOffset(MonitorIdType monitorId) const
{
    return varoffsetof(gaudi2::block_sob_objs, mon_pay_data[monitorId]);
}

uint64_t DeviceInfo::getMonitorConfigRegisterSmOffset(MonitorIdType monitorId) const
{
    return varoffsetof(gaudi2::block_sob_objs, mon_config[monitorId]);
}

uint64_t DeviceInfo::getMonitorArmRegisterSmOffset(MonitorIdType monitorId) const
{
    return varoffsetof(gaudi2::block_sob_objs, mon_arm[monitorId]);
}

uint16_t DeviceInfo::getNumArcCpus() const
{
    return CPU_ID_MAX;
}
