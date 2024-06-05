#pragma once

#include "master_qmans_definition.hpp"

#include "runtime/qman/common/coeff_table_configuration_manager.hpp"
#include "runtime/common/device/device_common.hpp"

#include "platform/gaudi/graph_compiler/special_functions_coefficients_table.h"

#include "gaudi/asic_reg_structs/tpc_regs.h"

#include <cstddef>

using namespace gaudi;

extern HalReaderPtr instantiateGaudiHalReader();

namespace gaudi
{
#define GAUDI_COEFF_TABLE_ALLOCATED_SIZE 1024 * 1024

static const uint32_t TPC_COEFF_TABLES_NUM          = 4;
static const uint32_t TPC_COEFF_TABLE_ADDR_REGS_NUM = TPC_COEFF_TABLES_NUM * 2;  // base_addr_lo and base_addr_hi

const uint64_t TPC_COEFF_TABLE_BASE_ADDRESS[TPC_COEFF_TABLES_NUM] = {SPECIAL_FUNCS_INTERVAL256_BASE_ADDR,
                                                                     SPECIAL_FUNCS_INTERVAL128_BASE_ADDR,
                                                                     SPECIAL_FUNCS_INTERVAL64_BASE_ADDR,
                                                                     SPECIAL_FUNCS_INTERVAL32_BASE_ADDR};

const uint64_t TPC_COEFF_TABLE_BASE_ADDRESS_REGS[TPC_COEFF_TABLE_ADDR_REGS_NUM] = {
    offsetof(block_tpc, lut_func256_base_addr_lo),
    offsetof(block_tpc, lut_func256_base_addr_hi),
    offsetof(block_tpc, lut_func128_base_addr_lo),
    offsetof(block_tpc, lut_func128_base_addr_hi),
    offsetof(block_tpc, lut_func64_base_addr_lo),
    offsetof(block_tpc, lut_func64_base_addr_hi),
    offsetof(block_tpc, lut_func32_base_addr_lo),
    offsetof(block_tpc, lut_func32_base_addr_hi)};

class CoeffTableConfManager : public ::CoeffTableConfManager
{
public:
    CoeffTableConfManager(DeviceGaudi* device) : ::CoeffTableConfManager(device)
    {
        m_cmdBuffPktGenerator        = CommandBufferPktGenerator::getInstance();
        m_halReader                  = instantiateGaudiHalReader();
        m_qmanDefs                   = QmansDefinition::getInstance();
        m_isSyncWithExternalRequired = true;
        m_isConfigOnInternal         = true;
    };
    virtual ~CoeffTableConfManager() = default;

protected:
    // TODO: This is a temporary WA for SW-70851. We want to allocate full 1MB for coeff table.
    virtual uint64_t getSpecialFuncCoeffTableAllocatedSize() override { return GAUDI_COEFF_TABLE_ALLOCATED_SIZE; }
    virtual uint64_t getSpecialFuncCoeffTableSize() override { return sizeof(coefficientsTableH3); }
    virtual void*    getSpecialFuncCoeffTableData() override { return coefficientsTableH3; }
};

std::unique_ptr<CoeffTableConfManager> createCoeffTableConfManager(DeviceGaudi* device)
{
    return std::make_unique<CoeffTableConfManager>(device);
}
}  // namespace gaudi
