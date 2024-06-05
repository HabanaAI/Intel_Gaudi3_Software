#pragma once

#include "scal_macros.h"
#include "scal_data.h"
#include "scal_gaudi2.h"
#include "gaudi2/asic_reg_structs/arc_dup_eng_regs.h"
#include "gaudi2_arc_common_packets.h"

#include "gaudi2/asic_reg/arc_farm_arc0_dup_eng_regs.h"
#include "gaudi2/asic_reg/arc_farm_arc1_dup_eng_regs.h"
#include "gaudi2/asic_reg/arc_farm_arc2_dup_eng_regs.h"
#include "gaudi2/asic_reg/arc_farm_arc3_dup_eng_regs.h"

#include "gaudi2/asic_reg/dcore0_mme_qm_arc_dup_eng_regs.h"
#include "gaudi2/asic_reg/dcore1_mme_qm_arc_dup_eng_regs.h"
#include "gaudi2/asic_reg/dcore2_mme_qm_arc_dup_eng_regs.h"
#include "gaudi2/asic_reg/dcore3_mme_qm_arc_dup_eng_regs.h"

#include "gaudi2/asic_reg/arc_farm_arc0_acp_eng_regs.h"
#include "gaudi2/asic_reg/arc_farm_arc1_acp_eng_regs.h"
#include "gaudi2/asic_reg/arc_farm_arc2_acp_eng_regs.h"
#include "gaudi2/asic_reg/arc_farm_arc3_acp_eng_regs.h"

struct EngineInfoG2
{
    const char * name;
    enum gaudi2_queue_id queueId;
    uint64_t dccmAddr;
    CoreType coreType;
    unsigned cpuId;
};

static const std::array<EngineInfoG2, CPU_ID_MAX> c_engine_info_arr{{
    // schedulers
    {"ARCFARM_0",     GAUDI2_QUEUE_ID_SIZE,             mmARC_FARM_ARC0_DCCM0_BASE,    SCHEDULER, CPU_ID_SCHED_ARC0},
    {"ARCFARM_1",     GAUDI2_QUEUE_ID_SIZE,             mmARC_FARM_ARC1_DCCM0_BASE,    SCHEDULER, CPU_ID_SCHED_ARC1},
    {"ARCFARM_2",     GAUDI2_QUEUE_ID_SIZE,             mmARC_FARM_ARC2_DCCM0_BASE,    SCHEDULER, CPU_ID_SCHED_ARC2},
    {"ARCFARM_3",     GAUDI2_QUEUE_ID_SIZE,             mmARC_FARM_ARC3_DCCM0_BASE,    SCHEDULER, CPU_ID_SCHED_ARC3},
    {"DCORE1_MME_0",  GAUDI2_QUEUE_ID_DCORE1_MME_0_0,   mmDCORE1_MME_QM_ARC_DCCM_BASE, SCHEDULER, CPU_ID_SCHED_ARC4},
    {"DCORE3_MME_0",  GAUDI2_QUEUE_ID_DCORE3_MME_0_0,   mmDCORE3_MME_QM_ARC_DCCM_BASE, SCHEDULER, CPU_ID_SCHED_ARC5},
    // engines
    {"DCORE0_TPC_0",  GAUDI2_QUEUE_ID_DCORE0_TPC_0_0,   mmDCORE0_TPC0_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC0},
    {"DCORE0_TPC_1",  GAUDI2_QUEUE_ID_DCORE0_TPC_1_0,   mmDCORE0_TPC1_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC1},
    {"DCORE0_TPC_2",  GAUDI2_QUEUE_ID_DCORE0_TPC_2_0,   mmDCORE0_TPC2_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC2},
    {"DCORE0_TPC_3",  GAUDI2_QUEUE_ID_DCORE0_TPC_3_0,   mmDCORE0_TPC3_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC3},
    {"DCORE0_TPC_4",  GAUDI2_QUEUE_ID_DCORE0_TPC_4_0,   mmDCORE0_TPC4_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC4},
    {"DCORE0_TPC_5",  GAUDI2_QUEUE_ID_DCORE0_TPC_5_0,   mmDCORE0_TPC5_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC5},
    {"DCORE1_TPC_0",  GAUDI2_QUEUE_ID_DCORE1_TPC_0_0,   mmDCORE1_TPC0_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC6},
    {"DCORE1_TPC_1",  GAUDI2_QUEUE_ID_DCORE1_TPC_1_0,   mmDCORE1_TPC1_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC7},
    {"DCORE1_TPC_2",  GAUDI2_QUEUE_ID_DCORE1_TPC_2_0,   mmDCORE1_TPC2_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC8},
    {"DCORE1_TPC_3",  GAUDI2_QUEUE_ID_DCORE1_TPC_3_0,   mmDCORE1_TPC3_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC9},
    {"DCORE1_TPC_4",  GAUDI2_QUEUE_ID_DCORE1_TPC_4_0,   mmDCORE1_TPC4_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC10},
    {"DCORE1_TPC_5",  GAUDI2_QUEUE_ID_DCORE1_TPC_5_0,   mmDCORE1_TPC5_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC11},
    {"DCORE2_TPC_0",  GAUDI2_QUEUE_ID_DCORE2_TPC_0_0,   mmDCORE2_TPC0_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC12},
    {"DCORE2_TPC_1",  GAUDI2_QUEUE_ID_DCORE2_TPC_1_0,   mmDCORE2_TPC1_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC13},
    {"DCORE2_TPC_2",  GAUDI2_QUEUE_ID_DCORE2_TPC_2_0,   mmDCORE2_TPC2_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC14},
    {"DCORE2_TPC_3",  GAUDI2_QUEUE_ID_DCORE2_TPC_3_0,   mmDCORE2_TPC3_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC15},
    {"DCORE2_TPC_4",  GAUDI2_QUEUE_ID_DCORE2_TPC_4_0,   mmDCORE2_TPC4_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC16},
    {"DCORE2_TPC_5",  GAUDI2_QUEUE_ID_DCORE2_TPC_5_0,   mmDCORE2_TPC5_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC17},
    {"DCORE3_TPC_0",  GAUDI2_QUEUE_ID_DCORE3_TPC_0_0,   mmDCORE3_TPC0_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC18},
    {"DCORE3_TPC_1",  GAUDI2_QUEUE_ID_DCORE3_TPC_1_0,   mmDCORE3_TPC1_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC19},
    {"DCORE3_TPC_2",  GAUDI2_QUEUE_ID_DCORE3_TPC_2_0,   mmDCORE3_TPC2_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC20},
    {"DCORE3_TPC_3",  GAUDI2_QUEUE_ID_DCORE3_TPC_3_0,   mmDCORE3_TPC3_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC21},
    {"DCORE3_TPC_4",  GAUDI2_QUEUE_ID_DCORE3_TPC_4_0,   mmDCORE3_TPC4_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC22},
    {"DCORE3_TPC_5",  GAUDI2_QUEUE_ID_DCORE3_TPC_5_0,   mmDCORE3_TPC5_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC23},
    // DCORE0_TPC_6 - Never present
    {"DCORE0_TPC_6",  GAUDI2_QUEUE_ID_DCORE0_TPC_6_0,   mmDCORE0_TPC6_QM_DCCM_BASE,    TPC,       CPU_ID_TPC_QMAN_ARC24},


    {"DCORE0_MME_0",  GAUDI2_QUEUE_ID_DCORE0_MME_0_0,   mmDCORE0_MME_QM_ARC_DCCM_BASE, MME,       CPU_ID_MME_QMAN_ARC0},
    {"DCORE2_MME_0",  GAUDI2_QUEUE_ID_DCORE2_MME_0_0,   mmDCORE2_MME_QM_ARC_DCCM_BASE, MME,       CPU_ID_MME_QMAN_ARC1},
    // EDMA QMAN cannot access the host memory, will be setting GAUDI2_QUEUE_ID_SIZE to prevents using it during init
    {"DCORE0_EDMA_0", GAUDI2_QUEUE_ID_DCORE0_EDMA_0_0,  mmDCORE0_EDMA0_QM_DCCM_BASE,   EDMA,      CPU_ID_EDMA_QMAN_ARC0},
    {"DCORE0_EDMA_1", GAUDI2_QUEUE_ID_DCORE0_EDMA_1_0,  mmDCORE0_EDMA1_QM_DCCM_BASE,   EDMA,      CPU_ID_EDMA_QMAN_ARC1},
    {"DCORE1_EDMA_0", GAUDI2_QUEUE_ID_DCORE1_EDMA_0_0,  mmDCORE1_EDMA0_QM_DCCM_BASE,   EDMA,      CPU_ID_EDMA_QMAN_ARC2},
    {"DCORE1_EDMA_1", GAUDI2_QUEUE_ID_DCORE1_EDMA_1_0,  mmDCORE1_EDMA1_QM_DCCM_BASE,   EDMA,      CPU_ID_EDMA_QMAN_ARC3},
    {"DCORE2_EDMA_0", GAUDI2_QUEUE_ID_DCORE2_EDMA_0_0,  mmDCORE2_EDMA0_QM_DCCM_BASE,   EDMA,      CPU_ID_EDMA_QMAN_ARC4},
    {"DCORE2_EDMA_1", GAUDI2_QUEUE_ID_DCORE2_EDMA_1_0,  mmDCORE2_EDMA1_QM_DCCM_BASE,   EDMA,      CPU_ID_EDMA_QMAN_ARC5},
    {"DCORE3_EDMA_0", GAUDI2_QUEUE_ID_DCORE3_EDMA_0_0,  mmDCORE3_EDMA0_QM_DCCM_BASE,   EDMA,      CPU_ID_EDMA_QMAN_ARC6},
    {"DCORE3_EDMA_1", GAUDI2_QUEUE_ID_DCORE3_EDMA_1_0,  mmDCORE3_EDMA1_QM_DCCM_BASE,   EDMA,      CPU_ID_EDMA_QMAN_ARC7},
    {"PDMA_0",        GAUDI2_QUEUE_ID_PDMA_0_0,         mmPDMA0_QM_ARC_DCCM_BASE,      PDMA,      CPU_ID_PDMA_QMAN_ARC0},
    {"PDMA_1",        GAUDI2_QUEUE_ID_PDMA_1_0,         mmPDMA1_QM_ARC_DCCM_BASE,      PDMA,      CPU_ID_PDMA_QMAN_ARC1},
    {"NIC_0",         GAUDI2_QUEUE_ID_NIC_0_0,          mmNIC0_QM_DCCM0_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC0},
    {"NIC_1",         GAUDI2_QUEUE_ID_NIC_1_0,          mmNIC0_QM_DCCM1_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC1},
    {"NIC_2",         GAUDI2_QUEUE_ID_NIC_2_0,          mmNIC1_QM_DCCM0_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC2},
    {"NIC_3",         GAUDI2_QUEUE_ID_NIC_3_0,          mmNIC1_QM_DCCM1_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC3},
    {"NIC_4",         GAUDI2_QUEUE_ID_NIC_4_0,          mmNIC2_QM_DCCM0_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC4},
    {"NIC_5",         GAUDI2_QUEUE_ID_NIC_5_0,          mmNIC2_QM_DCCM1_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC5},
    {"NIC_6",         GAUDI2_QUEUE_ID_NIC_6_0,          mmNIC3_QM_DCCM0_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC6},
    {"NIC_7",         GAUDI2_QUEUE_ID_NIC_7_0,          mmNIC3_QM_DCCM1_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC7},
    {"NIC_8",         GAUDI2_QUEUE_ID_NIC_8_0,          mmNIC4_QM_DCCM0_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC8},
    {"NIC_9",         GAUDI2_QUEUE_ID_NIC_9_0,          mmNIC4_QM_DCCM1_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC9},
    {"NIC_10",        GAUDI2_QUEUE_ID_NIC_10_0,         mmNIC5_QM_DCCM0_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC10},
    {"NIC_11",        GAUDI2_QUEUE_ID_NIC_11_0,         mmNIC5_QM_DCCM1_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC11},
    {"NIC_12",        GAUDI2_QUEUE_ID_NIC_12_0,         mmNIC6_QM_DCCM0_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC12},
    {"NIC_13",        GAUDI2_QUEUE_ID_NIC_13_0,         mmNIC6_QM_DCCM1_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC13},
    {"NIC_14",        GAUDI2_QUEUE_ID_NIC_14_0,         mmNIC7_QM_DCCM0_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC14},
    {"NIC_15",        GAUDI2_QUEUE_ID_NIC_15_0,         mmNIC7_QM_DCCM1_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC15},
    {"NIC_16",        GAUDI2_QUEUE_ID_NIC_16_0,         mmNIC8_QM_DCCM0_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC16},
    {"NIC_17",        GAUDI2_QUEUE_ID_NIC_17_0,         mmNIC8_QM_DCCM1_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC17},
    {"NIC_18",        GAUDI2_QUEUE_ID_NIC_18_0,         mmNIC9_QM_DCCM0_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC18},
    {"NIC_19",        GAUDI2_QUEUE_ID_NIC_19_0,         mmNIC9_QM_DCCM1_BASE,          NIC,       CPU_ID_NIC_QMAN_ARC19},
    {"NIC_20",        GAUDI2_QUEUE_ID_NIC_20_0,         mmNIC10_QM_DCCM0_BASE,         NIC,       CPU_ID_NIC_QMAN_ARC20},
    {"NIC_21",        GAUDI2_QUEUE_ID_NIC_21_0,         mmNIC10_QM_DCCM1_BASE,         NIC,       CPU_ID_NIC_QMAN_ARC21},
    {"NIC_22",        GAUDI2_QUEUE_ID_NIC_22_0,         mmNIC11_QM_DCCM0_BASE,         NIC,       CPU_ID_NIC_QMAN_ARC22},
    {"NIC_23",        GAUDI2_QUEUE_ID_NIC_23_0,         mmNIC11_QM_DCCM1_BASE,         NIC,       CPU_ID_NIC_QMAN_ARC23},
    {"ROT_0",         GAUDI2_QUEUE_ID_ROT_0_0,          mmROT0_QM_ARC_DCCM_BASE,       ROT,       CPU_ID_ROT_QMAN_ARC0},
    {"ROT_1",         GAUDI2_QUEUE_ID_ROT_1_0,          mmROT1_QM_ARC_DCCM_BASE,       ROT,       CPU_ID_ROT_QMAN_ARC1},
    }};

static inline bool qmanId2DccmAddr(const unsigned qid, uint64_t& dccmAddr)
{
    return qmanId2DccmAddr(qid, dccmAddr, c_engine_info_arr);
}

static inline bool arcName2QueueId(const std::string & arcName, unsigned &qid)
{
    return arcName2QueueId(arcName, qid, c_engine_info_arr);
}

static inline bool arcName2DccmAddr(const std::string & arcName, uint64_t &dccmAddr)
{
    return arcName2DccmAddr(arcName, dccmAddr, c_engine_info_arr);
}

static inline bool arcName2ArcType(const std::string & arcName, bool &isArcFarm, bool &isMmeSlave, bool &isEngine)
{
    for (const EngineInfoG2 & info : c_engine_info_arr)
    {
        if (arcName == info.name)
        {
            isArcFarm = (info.dccmAddr >= mmARC_FARM_ARC0_DCCM0_BASE) && (info.dccmAddr <= mmARC_FARM_ARC3_DCCM0_BASE);
            isMmeSlave = ((info.queueId == GAUDI2_QUEUE_ID_DCORE1_MME_0_0 || info.queueId == GAUDI2_QUEUE_ID_DCORE3_MME_0_0) && !isArcFarm);
            isEngine = !isArcFarm && !isMmeSlave;
            return true;
        }
    }
    return false;
}

static inline bool arcName2CoreType(const std::string & arcName, CoreType &coreType)
{
    return arcName2CoreType(arcName, coreType, c_engine_info_arr);
}

static inline bool arcName2CpuId(const std::string & arcName, unsigned &cpuId)
{
    return arcName2CpuId(arcName, cpuId, c_engine_info_arr);
}


/*
    We are using NIC DUP related registers in the “Compute and Media” scheduler instance.
    For rest of the engines only the Queue base address offset change is required.
    DUP programming for “Compute and Media” scheduler instance.

    or in English : the code below is a hack to save us some code in the FW
                    when we want to use pdma1 to send from device to host
                    we need a different DUP for pdma1 mask than pdma0
                    so instead of FW writing the mask each time according to the engine group
                    (which is 10 for pdma0 and 11 form pdma1)
                    we use nic0 dup instead and "tie" him to the pdma1 destination
                    and nic1 to tpc23 and nic2 to tpc24
                    (much like changing the "wiring")

                    in asic_reg_structs/arc_dup_eng_regs.h  block_arc_dup_eng_defaults array
                        { 0xf8  , 0x4c98              , 1 }, // dup_pdma_eng_addr_1
                        ...
                        { 0x5c  , 0x4648              , 1 }, // dup_tpc_eng_addr_23
                        { 0x60  , 0x4658              , 1 }, // dup_tpc_eng_addr_24

                    we will use the nics dup engines to handle all of the second groups types (*_GROUP1)
                    this way we can benifit from the flexibility, while maintaining high performance
*/
// the dup_nic_eng_addr and dup_nic_eng_mask used by the second group of each type (*_GROUP1) must have a predefined
// order since they share the same bitmask. e.g in fillSchedulerConfigs() we enumerate the engines in the (*_GROUP1)
//     so we can use a mask that just tells us which engines in this group to use
//     the idea is to repurpose the nic dups (which we don't use in the compute scheduler)
// for example, if:
//   MME_1 is not used that it wont affect the nics bitmask
//   TPC_MEDIA uses 2 engines from dupTriggerNicPri1_I then: dup_nic_eng_mask[1] = 011b  (e.g. the 1st two engines we
//   enumerated) PDMA_RX   uses 1 engine  from dupTriggerNicPri0_E then: dup_nic_eng_mask[4] = 100b  (e.g. the 3rd
//   engine  we enumerated) etc.
// ** we use the bit_mask_offset of each queue to allocate the bits of the NIC dup mask **

struct dup_trigger_info
{
    Scal_Gaudi2::DupTrigger         dup_trigger;
    uint64_t                         dup_engines_address_base_entry;
    uint64_t                         cluster_mask;
    unsigned                         engines;
    unsigned                         dup_trans_data_queues;
};

static const dup_trigger_info c_dup_trigger_info[] =
{
    {Scal_Gaudi2::dupTriggerTPC,       offsetof(gaudi2::block_arc_dup_eng, dup_tpc_eng_addr_0) / 4,  offsetof(gaudi2::block_arc_dup_eng, dup_tpc_eng_mask),       25, 4},
    {Scal_Gaudi2::dupTriggerMME,       offsetof(gaudi2::block_arc_dup_eng, dup_mme_eng_addr_0) / 4,  offsetof(gaudi2::block_arc_dup_eng, dup_mme_eng_mask),       4,  4},
    {Scal_Gaudi2::dupTriggerEDMA,      offsetof(gaudi2::block_arc_dup_eng, dup_edma_eng_addr_0) / 4, offsetof(gaudi2::block_arc_dup_eng, dup_edma_eng_mask),      8,  4},
    {Scal_Gaudi2::dupTriggerPDMA,      offsetof(gaudi2::block_arc_dup_eng, dup_pdma_eng_addr_0) / 4, offsetof(gaudi2::block_arc_dup_eng, dup_pdma_eng_mask),      2,  4},
    {Scal_Gaudi2::dupTriggerROT,       offsetof(gaudi2::block_arc_dup_eng, dup_rot_eng_addr_0) / 4,  offsetof(gaudi2::block_arc_dup_eng, dup_rot_eng_mask),       2,  4},
    {Scal_Gaudi2::dupTriggerRSRVD,     offsetof(gaudi2::block_arc_dup_eng, dup_rsvd_eng_addr_0) / 4, offsetof(gaudi2::block_arc_dup_eng, dup_rsvd_eng_mask),      16, 4},
    {Scal_Gaudi2::dupTriggerNicPri0_I, offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_addr_0) / 4,  offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_mask[0]), 24, 1},
    {Scal_Gaudi2::dupTriggerNicPri1_I, offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_addr_0) / 4,  offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_mask[1]), 24, 1},
    {Scal_Gaudi2::dupTriggerNicPri2_I, offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_addr_0) / 4,  offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_mask[2]), 24, 1},
    {Scal_Gaudi2::dupTriggerNicPri3_I, offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_addr_0) / 4,  offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_mask[3]), 24, 1},
    {Scal_Gaudi2::dupTriggerNicPri0_E, offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_addr_0) / 4,  offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_mask[4]), 24, 1},
    {Scal_Gaudi2::dupTriggerNicPri1_E, offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_addr_0) / 4,  offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_mask[5]), 24, 1},
    {Scal_Gaudi2::dupTriggerNicPri2_E, offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_addr_0) / 4,  offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_mask[6]), 24, 1},
    {Scal_Gaudi2::dupTriggerNicPri3_E, offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_addr_0) / 4,  offsetof(gaudi2::block_arc_dup_eng, dup_nic_eng_mask[7]), 24, 1}
};

struct dup_trigger_name_to_enum
{
    std::string                      name;
    Scal_Gaudi2::DupTrigger dupTrigger;
    unsigned                         dup_trans_data_q_index;
};

// the user can use 2 kinds of dup trigger naming conventions
//    1. aligned with the dup engine specs, the dup trigger includes the engine type it was designed to serve.
//    2. the dup triggers numbered from 0 to 6
// in both cases the postfix of the name indicates the dup_data_trans_q associated with the dup trigger (NICs has one
// queue, the rest has 4)
static const dup_trigger_name_to_enum c_dup_trigger_name_to_enum[] =
//  | dup_trigger_name      | dup_trigger
//  |                       |
{
    {"DUP_TRIGGER_TPC_0", Scal_Gaudi2::DupTrigger::dupTriggerTPC, 0},
    {"DUP_TRIGGER_TPC_1", Scal_Gaudi2::DupTrigger::dupTriggerTPC, 1},
    {"DUP_TRIGGER_TPC_2", Scal_Gaudi2::DupTrigger::dupTriggerTPC, 2},
    {"DUP_TRIGGER_TPC_3", Scal_Gaudi2::DupTrigger::dupTriggerTPC, 3},
    {"DUP_TRIGGER_MME_0", Scal_Gaudi2::DupTrigger::dupTriggerMME, 0},
    {"DUP_TRIGGER_MME_1", Scal_Gaudi2::DupTrigger::dupTriggerMME, 1},
    {"DUP_TRIGGER_MME_2", Scal_Gaudi2::DupTrigger::dupTriggerMME, 2},
    {"DUP_TRIGGER_MME_3", Scal_Gaudi2::DupTrigger::dupTriggerMME, 3},
    {"DUP_TRIGGER_EDMA_0", Scal_Gaudi2::DupTrigger::dupTriggerEDMA, 0},
    {"DUP_TRIGGER_EDMA_1", Scal_Gaudi2::DupTrigger::dupTriggerEDMA, 1},
    {"DUP_TRIGGER_EDMA_2", Scal_Gaudi2::DupTrigger::dupTriggerEDMA, 2},
    {"DUP_TRIGGER_EDMA_3", Scal_Gaudi2::DupTrigger::dupTriggerEDMA, 3},
    {"DUP_TRIGGER_PDMA_0", Scal_Gaudi2::DupTrigger::dupTriggerPDMA, 0},
    {"DUP_TRIGGER_PDMA_1", Scal_Gaudi2::DupTrigger::dupTriggerPDMA, 1},
    {"DUP_TRIGGER_PDMA_2", Scal_Gaudi2::DupTrigger::dupTriggerPDMA, 2},
    {"DUP_TRIGGER_PDMA_3", Scal_Gaudi2::DupTrigger::dupTriggerPDMA, 3},
    {"DUP_TRIGGER_ROT_0", Scal_Gaudi2::DupTrigger::dupTriggerROT, 0},
    {"DUP_TRIGGER_ROT_1", Scal_Gaudi2::DupTrigger::dupTriggerROT, 1},
    {"DUP_TRIGGER_ROT_2", Scal_Gaudi2::DupTrigger::dupTriggerROT, 2},
    {"DUP_TRIGGER_ROT_3", Scal_Gaudi2::DupTrigger::dupTriggerROT, 3},
    {"DUP_TRIGGER_RSRVD_0", Scal_Gaudi2::DupTrigger::dupTriggerRSRVD, 0},
    {"DUP_TRIGGER_RSRVD_1", Scal_Gaudi2::DupTrigger::dupTriggerRSRVD, 1},
    {"DUP_TRIGGER_RSRVD_2", Scal_Gaudi2::DupTrigger::dupTriggerRSRVD, 2},
    {"DUP_TRIGGER_RSRVD_3", Scal_Gaudi2::DupTrigger::dupTriggerRSRVD, 3},
    {"DUP_TRIGGER_NIC0_I", Scal_Gaudi2::DupTrigger::dupTriggerNicPri0_I, 0},
    {"DUP_TRIGGER_NIC1_I", Scal_Gaudi2::DupTrigger::dupTriggerNicPri1_I, 0},
    {"DUP_TRIGGER_NIC2_I", Scal_Gaudi2::DupTrigger::dupTriggerNicPri2_I, 0},
    {"DUP_TRIGGER_NIC3_I", Scal_Gaudi2::DupTrigger::dupTriggerNicPri3_I, 0},
    {"DUP_TRIGGER_NIC0_E", Scal_Gaudi2::DupTrigger::dupTriggerNicPri0_E, 0},
    {"DUP_TRIGGER_NIC1_E", Scal_Gaudi2::DupTrigger::dupTriggerNicPri1_E, 0},
    {"DUP_TRIGGER_NIC2_E", Scal_Gaudi2::DupTrigger::dupTriggerNicPri2_E, 0},
    {"DUP_TRIGGER_NIC3_E", Scal_Gaudi2::DupTrigger::dupTriggerNicPri3_E, 0},
    {"DATAQ_0_0", Scal_Gaudi2::DupTrigger::dupTriggerTPC, 0},
    {"DATAQ_0_1", Scal_Gaudi2::DupTrigger::dupTriggerTPC, 1},
    {"DATAQ_0_2", Scal_Gaudi2::DupTrigger::dupTriggerTPC, 2},
    {"DATAQ_0_3", Scal_Gaudi2::DupTrigger::dupTriggerTPC, 3},
    {"DATAQ_1_0", Scal_Gaudi2::DupTrigger::dupTriggerMME, 0},
    {"DATAQ_1_1", Scal_Gaudi2::DupTrigger::dupTriggerMME, 1},
    {"DATAQ_1_2", Scal_Gaudi2::DupTrigger::dupTriggerMME, 2},
    {"DATAQ_1_3", Scal_Gaudi2::DupTrigger::dupTriggerMME, 3},
    {"DATAQ_2_0", Scal_Gaudi2::DupTrigger::dupTriggerEDMA, 0},
    {"DATAQ_2_1", Scal_Gaudi2::DupTrigger::dupTriggerEDMA, 1},
    {"DATAQ_2_2", Scal_Gaudi2::DupTrigger::dupTriggerEDMA, 2},
    {"DATAQ_2_3", Scal_Gaudi2::DupTrigger::dupTriggerEDMA, 3},
    {"DATAQ_3_0", Scal_Gaudi2::DupTrigger::dupTriggerPDMA, 0},
    {"DATAQ_3_1", Scal_Gaudi2::DupTrigger::dupTriggerPDMA, 1},
    {"DATAQ_3_2", Scal_Gaudi2::DupTrigger::dupTriggerPDMA, 2},
    {"DATAQ_3_3", Scal_Gaudi2::DupTrigger::dupTriggerPDMA, 3},
    {"DATAQ_4_0", Scal_Gaudi2::DupTrigger::dupTriggerROT, 0},
    {"DATAQ_4_1", Scal_Gaudi2::DupTrigger::dupTriggerROT, 1},
    {"DATAQ_4_2", Scal_Gaudi2::DupTrigger::dupTriggerROT, 2},
    {"DATAQ_4_3", Scal_Gaudi2::DupTrigger::dupTriggerROT, 3},
    {"DATAQ_5_0", Scal_Gaudi2::DupTrigger::dupTriggerRSRVD, 0},
    {"DATAQ_5_1", Scal_Gaudi2::DupTrigger::dupTriggerRSRVD, 1},
    {"DATAQ_5_2", Scal_Gaudi2::DupTrigger::dupTriggerRSRVD, 2},
    {"DATAQ_5_3", Scal_Gaudi2::DupTrigger::dupTriggerRSRVD, 3},
    {"DATAQ_6_0", Scal_Gaudi2::DupTrigger::dupTriggerNicPri0_I, 0},
    {"DATAQ_7_0", Scal_Gaudi2::DupTrigger::dupTriggerNicPri1_I, 0},
    {"DATAQ_8_0", Scal_Gaudi2::DupTrigger::dupTriggerNicPri2_I, 0},
    {"DATAQ_9_0", Scal_Gaudi2::DupTrigger::dupTriggerNicPri3_I, 0},
    {"DATAQ_10_0", Scal_Gaudi2::DupTrigger::dupTriggerNicPri0_E, 0},
    {"DATAQ_11_0", Scal_Gaudi2::DupTrigger::dupTriggerNicPri1_E, 0},
    {"DATAQ_12_0", Scal_Gaudi2::DupTrigger::dupTriggerNicPri2_E, 0},
    {"DATAQ_13_0", Scal_Gaudi2::DupTrigger::dupTriggerNicPri3_E, 0}
};

static inline bool getDupTriggerByName(const std::string& dupTriggerName, Scal_Gaudi2::DupTrigger& dupTrigger)
{
    for (const auto& dup : c_dup_trigger_name_to_enum)
    {
        if (dupTriggerName == dup.name)
        {
            dupTrigger = dup.dupTrigger;
            return true;
        }
    }

    return false;
}

static inline bool getDupTriggerIndexByName(const std::string& dupTriggerName, unsigned& dupTransDataQIndex)
{
    for (const auto& dup : c_dup_trigger_name_to_enum)
    {
        if (dupTriggerName == dup.name)
        {
            dupTransDataQIndex = dup.dup_trans_data_q_index;
            return true;
        }
    }

    return false;
}

struct SchedulerDupEngineInfo
{
    const char* name;
    uint64_t    dupEngDevLocalAddress;
};

static const SchedulerDupEngineInfo c_schedulers_dup_eng_dev_local_address[] =
{
    {"ARCFARM_0",    mmARC_FARM_ARC0_DUP_ENG_DUP_TPC_ENG_ADDR_0 },
    {"ARCFARM_1",    mmARC_FARM_ARC1_DUP_ENG_DUP_TPC_ENG_ADDR_0 },
    {"ARCFARM_2",    mmARC_FARM_ARC2_DUP_ENG_DUP_TPC_ENG_ADDR_0 },
    {"ARCFARM_3",    mmARC_FARM_ARC3_DUP_ENG_DUP_TPC_ENG_ADDR_0 },
    {"DCORE1_MME_0", mmDCORE1_MME_QM_ARC_DUP_ENG_DUP_TPC_ENG_ADDR_0},
    {"DCORE3_MME_0", mmDCORE3_MME_QM_ARC_DUP_ENG_DUP_TPC_ENG_ADDR_0}
};

static inline bool schedulerName2DupEngLocalAddress(const std::string& schedulerName, uint64_t& dupEngDevLocalAddress)
{
    for (const SchedulerDupEngineInfo& info : c_schedulers_dup_eng_dev_local_address)
    {
        if (schedulerName == info.name)
        {
            dupEngDevLocalAddress = info.dupEngDevLocalAddress;
            return true;
        }
    }

    return false;
}

struct GroupName2GroupIndex
{
    const char* name;
    unsigned    index;
};

static const GroupName2GroupIndex c_group_name_2_group_index[] = {
    {"MME_COMPUTE_GROUP",  ScalComputeGroups::SCAL_MME_COMPUTE_GROUP},
    {"TPC_COMPUTE_GROUP",  ScalComputeGroups::SCAL_TPC_COMPUTE_GROUP},
    {"EDMA_COMPUTE_GROUP", ScalComputeGroups::SCAL_EDMA_COMPUTE_GROUP},
    {"RTR_COMPUTE_GROUP", ScalComputeGroups::SCAL_RTR_COMPUTE_GROUP},
    {"PDMA_TX_CMD_GROUP", ScalComputeGroups::SCAL_PDMA_TX_CMD_GROUP},
    {"PDMA_TX_DATA_GROUP", ScalComputeGroups::SCAL_PDMA_TX_DATA_GROUP},
    {"PDMA_RX_GROUP", ScalComputeGroups::SCAL_PDMA_RX_GROUP},
    {"NIC_RECEIVE_SCALE_UP_GROUP", ScalNetworkScaleUpReceiveGroups::SCAL_NIC_RECEIVE_SCALE_UP_GROUP},
    {"NIC_RECEIVE_SCALE_OUT_GROUP", ScalNetworkScaleOutReceiveGroups::SCAL_NIC_RECEIVE_SCALE_OUT_GROUP},
    {"NIC_SEND_SCALE_UP_GROUP", ScalNetworkScaleUpSendGroups::SCAL_NIC_SEND_SCALE_UP_GROUP},
    {"NIC_SEND_SCALE_OUT_GROUP", ScalNetworkScaleOutSendGroups::SCAL_NIC_SEND_SCALE_OUT_GROUP},
    {"EDMA_NETWORK_SCALE_UP_SEND_GROUP0", ScalNetworkScaleUpSendGroups::SCAL_EDMA_NETWORK_SCALE_UP_SEND_GROUP0},
    {"EDMA_NETWORK_GC_REDUCTION_GROUP0", ScalNetworkGarbageCollectorAndReductionGroups::SCAL_EDMA_NETWORK_GC_REDUCTION_GROUP0},
    {"EDMA_NETWORK_SCALE_OUT_SEND_GROUP0", ScalNetworkScaleOutSendGroups::SCAL_EDMA_NETWORK_SCALE_OUT_SEND_GROUP0},
    {"EDMA_NETWORK_SCALE_UP_RECV_GROUP0", ScalNetworkScaleUpReceiveGroups::SCAL_EDMA_NETWORK_SCALE_UP_RECV_GROUP0},
    {"EDMA_NETWORK_SCALE_UP_SEND_GROUP1", ScalNetworkScaleUpSendGroups::SCAL_EDMA_NETWORK_SCALE_UP_SEND_GROUP1},
    {"EDMA_NETWORK_GC_REDUCTION_GROUP1", ScalNetworkGarbageCollectorAndReductionGroups::SCAL_EDMA_NETWORK_GC_REDUCTION_GROUP1},
    {"EDMA_NETWORK_SCALE_OUT_SEND_GROUP1", ScalNetworkScaleOutSendGroups::SCAL_EDMA_NETWORK_SCALE_OUT_SEND_GROUP1},
    {"PDMA_NETWORK_SCALE_OUT_SEND_GROUP", ScalNetworkScaleOutSendGroups::SCAL_PDMA_NETWORK_SCALE_OUT_SEND_GROUP},
    {"EDMA_NETWORK_SCALE_UP_RECV_GROUP1", ScalNetworkScaleUpReceiveGroups::SCAL_EDMA_NETWORK_SCALE_UP_RECV_GROUP1},
    {"PDMA_NETWORK_SCALE_OUT_RECV_GROUP", ScalNetworkScaleOutReceiveGroups::SCAL_PDMA_NETWORK_SCALE_OUT_RECV_GROUP}};

static inline bool groupName2GroupIndex(const std::string groupName, unsigned& groupIndex)
{
    for (const auto& group : c_group_name_2_group_index)
    {
        if (groupName == group.name)
        {
            groupIndex = group.index;
            return true;
        }
    }

    return false;
};

struct priority_name_to_number
{
    std::string name;
    unsigned    priority;
};

static const priority_name_to_number c_priority_name_to_number[] =
{
    {"PRIORITY_HIGH", SCAL_HIGH_PRIORITY_STREAM},
    {"PRIORITY_LOW", SCAL_LOW_PRIORITY_STREAM}
};

static inline bool getPriorityByName(const std::string& priorityName, unsigned& priority)
{
    for (const auto& pri : c_priority_name_to_number)
    {
        if (priorityName == pri.name)
        {
            priority = pri.priority;
            return true;
        }
    }

    return false;
}

static inline uint64_t getSmBase(unsigned dcoreID)
{
    uint64_t smBase = 0;
    switch (dcoreID)
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
        assert(0);
    }
    return smBase;
}
