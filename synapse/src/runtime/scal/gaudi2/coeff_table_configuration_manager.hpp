#pragma once

#include "runtime/scal/common/coeff_table_configuration_manager.hpp"

namespace common
{
class DeviceScal;
}

namespace gaudi2
{
class CoeffTableConf : public common::CoeffTableConf
{
public:
    CoeffTableConf(common::DeviceScal* device) : common::CoeffTableConf(device) {};

    virtual ~CoeffTableConf() = default;

private:
    // Amount of regs requires to be configured per single entry
    virtual uint32_t getNumOfRegsPerTpcCoeffTable() override;
    // Amount table's entries
    virtual uint32_t getNumOfTpcCoeffTables() override;
    // Base address of a given TPC
    virtual uint64_t getTpcCfgBaseAddress(unsigned engineIndex) override;
    // TPC COEFF Base-Address table's first-entry (in order to iterate)
    virtual const uint64_t* getTpcCoeffHbmAddressOffsetTable() override;
    // TPC COEFF Regs-Offset table's first-entry (in order to iterate)
    virtual const uint32_t* getTpcCoeffAddrRegOffsetTable() override;

    virtual uint64_t getSpecialFuncCoeffTableSize() override;
    virtual void*    getSpecialFuncCoeffTableData() override;
};
}  // namespace gaudi2