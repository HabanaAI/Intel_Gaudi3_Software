#include "device_gaudi3scal.hpp"
#include "runtime/scal/gaudi3/coeff_table_configuration_manager.hpp"
#include "gaudi3/asic_reg/gaudi3_blocks.h"
#include "global_statistics.hpp"

const std::vector<uint64_t> DeviceGaudi3scal::s_tpcAddrVector
{
    mmHD0_TPC0_CFG_KERNEL_BASE,
    mmHD0_TPC1_CFG_KERNEL_BASE,
    mmHD0_TPC2_CFG_KERNEL_BASE,
    mmHD0_TPC3_CFG_KERNEL_BASE,
    mmHD0_TPC4_CFG_KERNEL_BASE,
    mmHD0_TPC5_CFG_KERNEL_BASE,
    mmHD0_TPC6_CFG_KERNEL_BASE,
    mmHD0_TPC7_CFG_KERNEL_BASE,
    // mmHD0_TPC8_CFG_KERNEL_BASE,
    mmHD1_TPC0_CFG_KERNEL_BASE,
    mmHD1_TPC1_CFG_KERNEL_BASE,
    mmHD1_TPC2_CFG_KERNEL_BASE,
    mmHD1_TPC3_CFG_KERNEL_BASE,
    mmHD1_TPC4_CFG_KERNEL_BASE,
    mmHD1_TPC5_CFG_KERNEL_BASE,
    mmHD1_TPC6_CFG_KERNEL_BASE,
    mmHD1_TPC7_CFG_KERNEL_BASE,
    mmHD2_TPC0_CFG_KERNEL_BASE,
    mmHD2_TPC1_CFG_KERNEL_BASE,
    mmHD2_TPC2_CFG_KERNEL_BASE,
    mmHD2_TPC3_CFG_KERNEL_BASE,
    mmHD2_TPC4_CFG_KERNEL_BASE,
    mmHD2_TPC5_CFG_KERNEL_BASE,
    mmHD2_TPC6_CFG_KERNEL_BASE,
    mmHD2_TPC7_CFG_KERNEL_BASE,
    // mmHD2_TPC8_CFG_KERNEL_BASE,
    mmHD3_TPC0_CFG_KERNEL_BASE,
    mmHD3_TPC1_CFG_KERNEL_BASE,
    mmHD3_TPC2_CFG_KERNEL_BASE,
    mmHD3_TPC3_CFG_KERNEL_BASE,
    mmHD3_TPC4_CFG_KERNEL_BASE,
    mmHD3_TPC5_CFG_KERNEL_BASE,
    mmHD3_TPC6_CFG_KERNEL_BASE,
    mmHD3_TPC7_CFG_KERNEL_BASE,
    mmHD4_TPC0_CFG_KERNEL_BASE,
    mmHD4_TPC1_CFG_KERNEL_BASE,
    mmHD4_TPC2_CFG_KERNEL_BASE,
    mmHD4_TPC3_CFG_KERNEL_BASE,
    mmHD4_TPC4_CFG_KERNEL_BASE,
    mmHD4_TPC5_CFG_KERNEL_BASE,
    mmHD4_TPC6_CFG_KERNEL_BASE,
    mmHD4_TPC7_CFG_KERNEL_BASE,
    mmHD5_TPC0_CFG_KERNEL_BASE,
    mmHD5_TPC1_CFG_KERNEL_BASE,
    mmHD5_TPC2_CFG_KERNEL_BASE,
    mmHD5_TPC3_CFG_KERNEL_BASE,
    mmHD5_TPC4_CFG_KERNEL_BASE,
    mmHD5_TPC5_CFG_KERNEL_BASE,
    mmHD5_TPC6_CFG_KERNEL_BASE,
    mmHD5_TPC7_CFG_KERNEL_BASE,
    // mmHD5_TPC8_CFG_KERNEL_BASE,
    mmHD6_TPC0_CFG_KERNEL_BASE,
    mmHD6_TPC1_CFG_KERNEL_BASE,
    mmHD6_TPC2_CFG_KERNEL_BASE,
    mmHD6_TPC3_CFG_KERNEL_BASE,
    mmHD6_TPC4_CFG_KERNEL_BASE,
    mmHD6_TPC5_CFG_KERNEL_BASE,
    mmHD6_TPC6_CFG_KERNEL_BASE,
    mmHD6_TPC7_CFG_KERNEL_BASE,
    mmHD7_TPC0_CFG_KERNEL_BASE,
    mmHD7_TPC1_CFG_KERNEL_BASE,
    mmHD7_TPC2_CFG_KERNEL_BASE,
    mmHD7_TPC3_CFG_KERNEL_BASE,
    mmHD7_TPC4_CFG_KERNEL_BASE,
    mmHD7_TPC5_CFG_KERNEL_BASE,
    mmHD7_TPC6_CFG_KERNEL_BASE,
    mmHD7_TPC7_CFG_KERNEL_BASE,
    // mmHD7_TPC8_CFG_KERNEL_BASE
};

DeviceGaudi3scal::DeviceGaudi3scal(const DeviceConstructInfo& deviceConstructInfo)
: DeviceScal(synDeviceGaudi3, deviceConstructInfo)
{
}

synStatus DeviceGaudi3scal::acquire(const uint16_t numSyncObj)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::unique_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    gaudi3::CoeffTableConf coeffTableConf(this);
    return _acquire(numSyncObj, coeffTableConf);
}