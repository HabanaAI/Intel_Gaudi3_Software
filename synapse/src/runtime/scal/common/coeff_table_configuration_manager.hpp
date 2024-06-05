#pragma once

#include "runtime/scal/common/stream_base_scal.hpp"

namespace common
{
class DeviceScal;
}

#define COMPUTE_TPC_CLUSTER_NAME "compute_tpc"

namespace common
{
class CoeffTableConf
{
public:
    CoeffTableConf(common::DeviceScal* device);
    virtual ~CoeffTableConf() = default;

    synStatus submitCoeffTableConfiguration();

private:
    synStatus generateAndSendCoeffTableConfiguration(uint64_t coeffTableAddr, QueueBaseScalCommon* pStream);

    // Amount of regs requires to be configured per single entry
    virtual uint32_t getNumOfRegsPerTpcCoeffTable() = 0;
    // Amount table's entries
    virtual uint32_t getNumOfTpcCoeffTables() = 0;
    // Base address of a given TPC
    virtual uint64_t getTpcCfgBaseAddress(unsigned engineIndex) = 0;
    // TPC COEFF Base-Address table's first-entry (in order to iterate)
    virtual const uint64_t* getTpcCoeffHbmAddressOffsetTable() = 0;
    // TPC COEFF Regs-Offset table's first-entry (in order to iterate)
    virtual const uint32_t* getTpcCoeffAddrRegOffsetTable() = 0;

    virtual uint64_t getSpecialFuncCoeffTableSize() = 0;
    virtual void*    getSpecialFuncCoeffTableData() = 0;

    common::DeviceScal* m_device;
};
}  // namespace common