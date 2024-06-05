#include "scal_monitor_base.hpp"

#include "defs.h"
#include "synapse_common_types.h"
#include "types.h"

#include "runtime/scal/common/entities/scal_memory_pool.hpp"

#include "log_manager.h"

#include "runtime/scal/common/entities/device_info_interface.hpp"

#include "scal.h"

// TODO - utils infra
#define upper_32_bits(n)  ((uint32_t)(((n) >> 16) >> 16))
#define lower_32_bits(n)  ((uint32_t)(n))

ScalMonitorBase::ScalMonitorBase(MonitorIdType                      id,
                                 const common::DeviceInfoInterface& deviceInfoInterface,
                                 scal_handle_t                      deviceHandle)
: m_deviceInfoInterface(deviceInfoInterface), m_monitorId(id), m_deviceHandle(deviceHandle)
{
    HB_ASSERT(m_monitorId % 4 == 0, "monitorID {} should be x4", m_monitorId);
}

synStatus ScalMonitorBase::init(const scal_completion_group_infoV2_t& cgInfo, scal_stream_info_t* pStreamInfo)
{
    m_syncMgrId = cgInfo.long_so_sm;

    // all longSO should have the same GroupIndex
    // and every monitor guards a group of 8 sos
    m_soGroupIdx = cgInfo.long_so_index >> 3;

    m_lastLongSOtargetValue = 0;

    return synSuccess;
}

bool ScalMonitorBase::getConfigRegsForLongSO(FenceIdType           fenceId,
                                             uint8_t&              numRegs,
                                             MonitorAddressesType& addr,
                                             MonitorValuesType&    value) const
{
    uint64_t payloadAddress = 0;
    uint32_t payloadData    = 0;
    if (!getPayloadInfo(payloadAddress, payloadData, fenceId))
    {
        return false;
    }

    LOG_TRACE(SYN_DEVICE,
                   "configure monitor {} (longSOGroup {}) to release fenceId={}",
                   m_monitorId,
                   m_soGroupIdx,
                   fenceId);

    unsigned index = 0;

    bool isDummyMessageRequired = m_deviceInfoInterface.isDummyMessageRequiredForMonitorForLongSO();
    // monitor config   (mc)
    // 0 ==> writes 1 time , e.g does 1 thing, 1 ==> writes 2 times
    const uint32_t numOfWrites = isDummyMessageRequired ? 1 : 0;
    addr[index]                = m_deviceInfoInterface.getMonitorConfigRegisterAddress(m_syncMgrId, m_monitorId);
    value[index++] =
        m_deviceInfoInterface.getMonitorConfigRegisterValue(true, false, false, numOfWrites, (m_soGroupIdx >> 8));

    // The 64 bit address to write the completion message to in case CQ_EN=0.
    // set payload addr Hi & Low
    addr[index]    = m_deviceInfoInterface.getMonitorPayloadAddrHighRegisterAddress(m_syncMgrId, m_monitorId);
    value[index++] = upper_32_bits(payloadAddress);

    addr[index]    = m_deviceInfoInterface.getMonitorPayloadAddrLowRegisterAddress(m_syncMgrId, m_monitorId);
    value[index++] = lower_32_bits(payloadAddress);

    // configure the monitor payload data to send a fence update msg to the dccm Q of the scheduler
    addr[index]    = m_deviceInfoInterface.getMonitorPayloadDataRegisterAddress(m_syncMgrId, m_monitorId);
    value[index++] = payloadData;

    if (isDummyMessageRequired)
    {
        // 2nd dummy message to DCORE0_SYNC_MNGR_OBJS SOB_OBJ_8184 (as w/a for SM bug in H6 - SW-67146)
        scal_so_pool_handle_t dummySoPoolHandle;
        int rc = scal_get_so_pool_handle_by_name(m_deviceHandle, "sos_pool_long_monitor_wa", &dummySoPoolHandle);
        if (rc != 0)
        {
            LOG_ERR(SYN_DEVICE, "{}: error scal_get_so_monitor_handle_by_name {}", HLLOG_FUNC, rc);
            return false;
        }
        scal_so_pool_info dummySoPoolInfo;
        rc |= scal_so_pool_get_info(dummySoPoolHandle, &dummySoPoolInfo);
        assert(rc == 0);
        if (rc != 0)  // needed for release build
        {
            LOG_ERR(SYN_DEVICE, "{}: error scal_monitor_pool_get_info {}", HLLOG_FUNC, rc);
            return false;
        }

        // In case this will differ between platforms (it will not), we will retrieve it from the I/F
        const uint32_t dummySosIndex = dummySoPoolInfo.baseIdx;
        uint64_t       sosAddress    = m_deviceInfoInterface.getSosAddr(dummySoPoolInfo.dcoreIndex, dummySosIndex);
        //
        // set payload addr Hi & Low
        addr[index]    = m_deviceInfoInterface.getMonitorPayloadAddrHighRegisterAddress(m_syncMgrId, m_monitorId + 1);
        value[index++] = upper_32_bits(sosAddress);
        addr[index]    = m_deviceInfoInterface.getMonitorPayloadAddrLowRegisterAddress(m_syncMgrId, m_monitorId + 1);
        value[index++] = lower_32_bits(sosAddress);
        //
        // configure the monitor payload data to send a dummy data to to SOB_OBJ_8184
        addr[index]    = m_deviceInfoInterface.getMonitorPayloadDataRegisterAddress(m_syncMgrId, m_monitorId + 1);
        value[index++] = 0;
    }

    // set monArm[1..3] to 0 (don't arm them)
    for (unsigned i = 1; i <= 3; i++)
    {
        addr[index]    = m_deviceInfoInterface.getMonitorArmRegisterAddress(m_syncMgrId, m_monitorId + i);
        value[index++] = 0;
    }

    numRegs = index;

    return true;
}

void ScalMonitorBase::getArmRegsForLongSO(const ScalLongSyncObject& rLongSo,
                                          const FenceIdType         fenceId,
                                          uint8_t&                  numRegs,
                                          MonitorAddressesType&     addr,
                                          MonitorValuesType&        value)
{
    unsigned longSoId          = rLongSo.m_index;
    uint64_t longSOtargetValue = rLongSo.m_targetValue;
    bool     compareEQ         = false;

    HB_ASSERT(longSoId % 8 == 0, "longSoId {} should be x8", longSoId);

    uint64_t prevLongSOtargetValue = m_lastLongSOtargetValue;

    /*
     * To support 60 bit value in longSOB, we need to use 4 monitors, chained to a "long monitor" by the longSOB = 1
     * bit This implies that
     * each of the 4 monitor SOD field (when arming) holds the respective 15 bit value
     * it only fires 1 msg
     * monitor 0 index must be 32 byte aligned --> since each entry is 4 bytes, index should be diviseable by 4
     * arming mon 1..3 must come BEFORE arm0
     * we need only to config the 1st monitor
     * we must arm only the monitors that are expected to change
     * e.g. we need to compare the prev val with the new one and reconfigure val15_29, val30_44 and val45_59 only
     * if they change. val_0_15 should be armed anyway.
     */
    unsigned index      = 0;
    unsigned soGroupIdx = longSoId >> 3;  // every monitor guards a group of 8 sos
    HB_ASSERT((m_soGroupIdx >> 8) == ((longSoId >> 3) >> 8), "longSO group index should be identical for all");

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
            // LOG_DEBUG(SCAL, "{}: arming extra mon {} for value {:x}", HLLOG_FUNC, monitorID + i, ma.sod);
            addr[index]    = m_deviceInfoInterface.getMonitorArmRegisterAddress(m_syncMgrId, m_monitorId + i);
            value[index++] = m_deviceInfoInterface.getMonitorArmRegisterValue((newT & 0x7FFF),
                                                                              compareEQ,
                                                                              (~(1 << ((longSoId) % 8))),
                                                                              (((longSoId) >> 3) & 0xFF));
        }
    }

    // keep this last
    // must be done EVERY time
    addr[index]    = m_deviceInfoInterface.getMonitorArmRegisterAddress(m_syncMgrId, m_monitorId);
    value[index++] = m_deviceInfoInterface.getMonitorArmRegisterValue(longSOtargetValue,
                                                                      compareEQ,
                                                                      (~(1 << (longSoId % 8))),
                                                                      (soGroupIdx & 0xFF));

    numRegs = index;

    m_lastLongSOtargetValue = longSOtargetValue;
}
