#include "coeff_table_configuration_manager.hpp"

#include "platform/gaudi2/graph_compiler/special_functions_coefficients_table.h"

// engine-arc
#include "gaudi2_arc_common_packets.h"  // CPU_ID_TPC_QMAN_ARC0

// specs
#include "gaudi2/asic_reg/gaudi2_blocks.h"
#include "gaudi2/asic_reg_structs/tpc_regs.h"

using namespace gaudi2;

#define COMPUTE_TPC_CLUSTER_NAME "compute_tpc"

static const uint32_t TPC_COEFF_TABLES_NUM                                     = 4;
static const uint64_t TPC_COEFF_HBM_ADDRESS_OFFSET_TABLE[TPC_COEFF_TABLES_NUM] = {SPECIAL_FUNCS_INTERVAL256_BASE_ADDR,
                                                                                  SPECIAL_FUNCS_INTERVAL128_BASE_ADDR,
                                                                                  SPECIAL_FUNCS_INTERVAL64_BASE_ADDR,
                                                                                  SPECIAL_FUNCS_INTERVAL32_BASE_ADDR};

static const uint32_t TPC_COEFF_TABLES_NUM_OF_CONFIG_REGS = 2;
static const uint32_t TPC_COEFF_TABLE_ADDR_REGS_NUM       = TPC_COEFF_TABLES_NUM * TPC_COEFF_TABLES_NUM_OF_CONFIG_REGS;

static const uint64_t INVALID_TPC_CFG_BASE_ADDRESS = std::numeric_limits<uint64_t>::max();

const uint32_t TPC_COEFF_ADDRESS_REGISTER_OFFSET_TABLE[TPC_COEFF_TABLE_ADDR_REGS_NUM] = {
    offsetof(block_tpc, lut_func256_base_addr_lo),
    offsetof(block_tpc, lut_func256_base_addr_hi),
    offsetof(block_tpc, lut_func128_base_addr_lo),
    offsetof(block_tpc, lut_func128_base_addr_hi),
    offsetof(block_tpc, lut_func64_base_addr_lo),
    offsetof(block_tpc, lut_func64_base_addr_hi),
    offsetof(block_tpc, lut_func32_base_addr_lo),
    offsetof(block_tpc, lut_func32_base_addr_hi)};

// For configuring the coeff table for TPCs
static const uint64_t TPCS_CFG_BASE_ADDRESS[] = {mmDCORE0_TPC0_CFG_BASE,
                                                 mmDCORE0_TPC1_CFG_BASE,
                                                 mmDCORE0_TPC2_CFG_BASE,
                                                 mmDCORE0_TPC3_CFG_BASE,
                                                 mmDCORE0_TPC4_CFG_BASE,
                                                 mmDCORE0_TPC5_CFG_BASE,
                                                 // mmDCORE0_TPC6_CFG_BASE,
                                                 mmDCORE1_TPC0_CFG_BASE,
                                                 mmDCORE1_TPC1_CFG_BASE,
                                                 mmDCORE1_TPC2_CFG_BASE,
                                                 mmDCORE1_TPC3_CFG_BASE,
                                                 mmDCORE1_TPC4_CFG_BASE,
                                                 mmDCORE1_TPC5_CFG_BASE,
                                                 mmDCORE2_TPC0_CFG_BASE,
                                                 mmDCORE2_TPC1_CFG_BASE,
                                                 mmDCORE2_TPC2_CFG_BASE,
                                                 mmDCORE2_TPC3_CFG_BASE,
                                                 mmDCORE2_TPC4_CFG_BASE,
                                                 mmDCORE2_TPC5_CFG_BASE,
                                                 mmDCORE3_TPC0_CFG_BASE,
                                                 mmDCORE3_TPC1_CFG_BASE,
                                                 mmDCORE3_TPC2_CFG_BASE,
                                                 mmDCORE3_TPC3_CFG_BASE,
                                                 mmDCORE3_TPC4_CFG_BASE,
                                                 mmDCORE3_TPC5_CFG_BASE,
                                                 INVALID_TPC_CFG_BASE_ADDRESS};

uint32_t CoeffTableConf::getNumOfRegsPerTpcCoeffTable()
{
    return TPC_COEFF_TABLES_NUM_OF_CONFIG_REGS;
}

uint32_t CoeffTableConf::getNumOfTpcCoeffTables()
{
    return TPC_COEFF_TABLES_NUM;
}

uint64_t CoeffTableConf::getTpcCfgBaseAddress(unsigned engineIndex)
{
    return TPCS_CFG_BASE_ADDRESS[engineIndex - CPU_ID_TPC_QMAN_ARC0];
}

const uint64_t* CoeffTableConf::getTpcCoeffHbmAddressOffsetTable()
{
    return TPC_COEFF_HBM_ADDRESS_OFFSET_TABLE;
}

const uint32_t* CoeffTableConf::getTpcCoeffAddrRegOffsetTable()
{
    return TPC_COEFF_ADDRESS_REGISTER_OFFSET_TABLE;
}

uint64_t CoeffTableConf::getSpecialFuncCoeffTableSize()
{
    return sizeof(coefficientsTableH6);
}

void* CoeffTableConf::getSpecialFuncCoeffTableData()
{
    return coefficientsTableH6;
}