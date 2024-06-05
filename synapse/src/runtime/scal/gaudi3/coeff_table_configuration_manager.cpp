#include "coeff_table_configuration_manager.hpp"

#include "platform/gaudi3/graph_compiler/special_functions_coefficients_table.h"

// engine-arc
#include "gaudi3_arc_common_packets.h"  // CPU_ID_TPC_QMAN_ARC0

// specs
#include "gaudi3/asic_reg/gaudi3_blocks.h"
#include "gaudi3/asic_reg_structs/tpc_regs.h"
using namespace gaudi3;

#define COMPUTE_TPC_CLUSTER_NAME "compute_tpc"

static const uint32_t TPC_COEFF_TABLES_NUM                                     = 4;
static const uint64_t TPC_COEFF_HBM_ADDRESS_OFFSET_TABLE[TPC_COEFF_TABLES_NUM] = {SPECIAL_FUNCS_INTERVAL256_BASE_ADDR,
                                                                                  SPECIAL_FUNCS_INTERVAL128_BASE_ADDR,
                                                                                  SPECIAL_FUNCS_INTERVAL64_BASE_ADDR,
                                                                                  SPECIAL_FUNCS_INTERVAL32_BASE_ADDR};

static const uint32_t TPC_COEFF_TABLES_NUM_OF_CONFIG_REGS = 2;
static const uint32_t TPC_COEFF_TABLE_ADDR_REGS_NUM       = TPC_COEFF_TABLES_NUM * TPC_COEFF_TABLES_NUM_OF_CONFIG_REGS;

static const uint32_t TPC_COEFF_ADDRESS_REGISTER_OFFSET_TABLE[TPC_COEFF_TABLE_ADDR_REGS_NUM] = {
    offsetof(block_tpc, lut_func256_base_addr_lo),
    offsetof(block_tpc, lut_func256_base_addr_hi),
    offsetof(block_tpc, lut_func128_base_addr_lo),
    offsetof(block_tpc, lut_func128_base_addr_hi),
    offsetof(block_tpc, lut_func64_base_addr_lo),
    offsetof(block_tpc, lut_func64_base_addr_hi),
    offsetof(block_tpc, lut_func32_base_addr_lo),
    offsetof(block_tpc, lut_func32_base_addr_hi)};

// For configuring the coeff table for TPCs
static const uint64_t TPCS_CFG_BASE_ADDRESS[] = {
    mmHD0_TPC0_CFG_BASE, mmHD0_TPC1_CFG_BASE, mmHD0_TPC2_CFG_BASE, mmHD0_TPC3_CFG_BASE, mmHD0_TPC4_CFG_BASE,
    mmHD0_TPC5_CFG_BASE, mmHD0_TPC6_CFG_BASE, mmHD0_TPC7_CFG_BASE,

    mmHD1_TPC0_CFG_BASE, mmHD1_TPC1_CFG_BASE, mmHD1_TPC2_CFG_BASE, mmHD1_TPC3_CFG_BASE, mmHD1_TPC4_CFG_BASE,
    mmHD1_TPC5_CFG_BASE, mmHD1_TPC6_CFG_BASE, mmHD1_TPC7_CFG_BASE,

    mmHD2_TPC0_CFG_BASE, mmHD2_TPC1_CFG_BASE, mmHD2_TPC2_CFG_BASE, mmHD2_TPC3_CFG_BASE, mmHD2_TPC4_CFG_BASE,
    mmHD2_TPC5_CFG_BASE, mmHD2_TPC6_CFG_BASE, mmHD2_TPC7_CFG_BASE,

    mmHD3_TPC0_CFG_BASE, mmHD3_TPC1_CFG_BASE, mmHD3_TPC2_CFG_BASE, mmHD3_TPC3_CFG_BASE, mmHD3_TPC4_CFG_BASE,
    mmHD3_TPC5_CFG_BASE, mmHD3_TPC6_CFG_BASE, mmHD3_TPC7_CFG_BASE,

    mmHD4_TPC0_CFG_BASE, mmHD4_TPC1_CFG_BASE, mmHD4_TPC2_CFG_BASE, mmHD4_TPC3_CFG_BASE, mmHD4_TPC4_CFG_BASE,
    mmHD4_TPC5_CFG_BASE, mmHD4_TPC6_CFG_BASE, mmHD4_TPC7_CFG_BASE,

    mmHD5_TPC0_CFG_BASE, mmHD5_TPC1_CFG_BASE, mmHD5_TPC2_CFG_BASE, mmHD5_TPC3_CFG_BASE, mmHD5_TPC4_CFG_BASE,
    mmHD5_TPC5_CFG_BASE, mmHD5_TPC6_CFG_BASE, mmHD5_TPC7_CFG_BASE,

    mmHD6_TPC0_CFG_BASE, mmHD6_TPC1_CFG_BASE, mmHD6_TPC2_CFG_BASE, mmHD6_TPC3_CFG_BASE, mmHD6_TPC4_CFG_BASE,
    mmHD6_TPC5_CFG_BASE, mmHD6_TPC6_CFG_BASE, mmHD6_TPC7_CFG_BASE,

    mmHD7_TPC0_CFG_BASE, mmHD7_TPC1_CFG_BASE, mmHD7_TPC2_CFG_BASE, mmHD7_TPC3_CFG_BASE, mmHD7_TPC4_CFG_BASE,
    mmHD7_TPC5_CFG_BASE, mmHD7_TPC6_CFG_BASE, mmHD7_TPC7_CFG_BASE

    // Binned
    // mmHD0_TPC8_CFG_BASE,
    // mmHD2_TPC8_CFG_BASE,
    // mmHD5_TPC8_CFG_BASE,
    // mmHD7_TPC8_CFG_BASE,
};

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
    return sizeof(coefficientsTableH9);
}

void* CoeffTableConf::getSpecialFuncCoeffTableData()
{
    return coefficientsTableH9;
}