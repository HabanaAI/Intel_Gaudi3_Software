#include "device_gaudi2scal.hpp"
#include "runtime/scal/gaudi2/coeff_table_configuration_manager.hpp"
#include "gaudi2/asic_reg/gaudi2_blocks.h"
#include "global_statistics.hpp"

const std::vector<uint64_t> DeviceGaudi2scal::s_tpcAddrVector
{
    mmDCORE0_TPC0_CFG_KERNEL_BASE,
    mmDCORE0_TPC1_CFG_KERNEL_BASE,
    mmDCORE0_TPC2_CFG_KERNEL_BASE,
    mmDCORE0_TPC3_CFG_KERNEL_BASE,
    mmDCORE0_TPC4_CFG_KERNEL_BASE,
    mmDCORE0_TPC5_CFG_KERNEL_BASE,
    // mmDCORE0_TPC6_CFG_KERNEL_BASE,
    mmDCORE1_TPC0_CFG_KERNEL_BASE,
    mmDCORE1_TPC1_CFG_KERNEL_BASE,
    mmDCORE1_TPC2_CFG_KERNEL_BASE,
    mmDCORE1_TPC3_CFG_KERNEL_BASE,
    mmDCORE1_TPC4_CFG_KERNEL_BASE,
    mmDCORE1_TPC5_CFG_KERNEL_BASE,
    mmDCORE2_TPC0_CFG_KERNEL_BASE,
    mmDCORE2_TPC1_CFG_KERNEL_BASE,
    mmDCORE2_TPC2_CFG_KERNEL_BASE,
    mmDCORE2_TPC3_CFG_KERNEL_BASE,
    mmDCORE2_TPC4_CFG_KERNEL_BASE,
    mmDCORE2_TPC5_CFG_KERNEL_BASE,
    mmDCORE3_TPC0_CFG_KERNEL_BASE,
    mmDCORE3_TPC1_CFG_KERNEL_BASE,
    mmDCORE3_TPC2_CFG_KERNEL_BASE,
    mmDCORE3_TPC3_CFG_KERNEL_BASE,
    mmDCORE3_TPC4_CFG_KERNEL_BASE,
    mmDCORE3_TPC5_CFG_KERNEL_BASE
};

DeviceGaudi2scal::DeviceGaudi2scal(const DeviceConstructInfo& deviceConstructInfo)
: DeviceScal(synDeviceGaudi2, deviceConstructInfo)
{
}

synStatus DeviceGaudi2scal::acquire(const uint16_t numSyncObj)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::unique_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    gaudi2::CoeffTableConf coeffTableConf(this);
    return _acquire(numSyncObj, coeffTableConf);
}