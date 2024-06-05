#include "gaudi3_mme_hal_reader.h"
#include "include/mme_common/mme_common_enum.h"
#include "drm/habanalabs_accel.h"
#include "gaudi3/asic_reg_structs/sob_objs_regs.h"
#include "gaudi3/asic_reg/gaudi3_blocks.h"
#include "gaudi3/mme.h"
#include "gaudi3/gaudi3.h"

namespace gaudi3
{
unsigned MmeHalReader::getDieNr() const
{
    return MAX_NUM_OF_DIES;
}
unsigned MmeHalReader::getDcoreNr() const
{
    return 4;
}
unsigned MmeHalReader::getMmeNr() const { return gaudi3::Mme::MME_CORE_MASTERS_NR; }
unsigned MmeHalReader::getEuNr() const { return gaudi3::Mme::MME_CORES_NR; }
unsigned MmeHalReader::getClSize() const { return gaudi3::Mme::c_cl_size; }
unsigned MmeHalReader::getMemoryClSize() const
{
    return 2 * getClSize();
}
unsigned MmeHalReader::getAccumsNr() const { return gaudi3::Mme::c_mme_accums_nr; }
unsigned MmeHalReader::getSBSize() const { return gaudi3::Mme::c_mme_sb_size; }
unsigned MmeHalReader::getMaxSBReuse() const { return gaudi3::Mme::c_mme_max_sb_reuse; }
unsigned MmeHalReader::getSinglePortBw() const
{
    return 200;
}
unsigned MmeHalReader::getLFSRSeedsNr() const { return gaudi3::Mme::c_mme_lfsr_seeds_nr; }
unsigned MmeHalReader::getSizeOfDescNumIterMinus1() const
{
    return sizeof(gaudi3::Mme::Desc::numIterationsMinus1);
}
unsigned MmeHalReader::getSizeOfOuterLoopSizeMinus1() const
{
    return sizeof(gaudi3::Mme::MmeOuterLoop::sizeMinus1);
}

unsigned MmeHalReader::getSizeOfKernelSizeDim0() const { return sizeof(gaudi3::Mme::MmeKernelSize::dim[0]); }

// The reduction tree required alignment of common dim.
// "common-dim % getBaseCommonDim(...)" must be 0.
unsigned MmeHalReader::getNumElementsForCommonDimAlignment(MmeCommon::EMmeDataType dataType,
                                                           MmeCommon::EMmeOpType opType) const
{
    //  dma operations
    if (opType == MmeCommon::e_mme_memcpy) return 1;
    if (opType == MmeCommon::e_mme_trans)
    {
        return isTypeFp8(dataType) ? 2 : 1;
    }

    //  regular mme operations
    switch (dataType)
    {
        case MmeCommon::e_type_fp8_143:
        case MmeCommon::e_type_fp8_152:
            return 8;
        case MmeCommon::e_type_bf16:
            return 8;
        case MmeCommon::e_type_tf32:
        case MmeCommon::e_type_fp16:
        case MmeCommon::e_type_ufp16:
            return 2;
        case MmeCommon::e_type_fp32:
            return 1;
        default:
            MME_ASSERT(0, "invalid data type");
    }
    return 0;
}

//  some of these use Gaudi2 fields since habanalabs repo doesnt support gaudi3 yet.
//  currently coral uses the same solution, update once specs is updated.

/*
 * 64 SOBs reserved for completion Q
 * 328 SOBs reserved for sync stream
 */
#define GAUDI3_DCORE0_FIRST_AVAILABLE_SYNC_OBJECT 392

/*
 * 64 monitors reserved for completion Q
 * 164 monitors reserved for sync stream
 */
#define GAUDI3_DCORE0_FIRST_AVAILABLE_MONITOR 228

// queue size from coral
#define GAUDI3_SIM_QUEUE_ID_SIZE 70

uint64_t MmeHalReader::getSramStart(unsigned dieNr) const
{
    dieNr = dieNr == 0 ? getDieNr() : dieNr;
    if (dieNr == 1) return SRAM_BASE_ADDR + SRAM_MODE_0_SINGLE_DIE_OFFSET;
    if (dieNr == 2) return SRAM_BASE_ADDR + SRAM_MODE_0_DOUBLE_DIE_OFFSET;

    MME_ASSERT(0, "invalid number of dies");
    return SRAM_BASE_ADDR;
}
uint64_t MmeHalReader::getSramSize(unsigned dieNr) const
{
    dieNr = dieNr == 0 ? getDieNr() : dieNr;
    if (dieNr == 1) return SRAM_SIZE - SRAM_MODE_0_SINGLE_DIE_OFFSET;
    if (dieNr == 2) return SRAM_SIZE - SRAM_MODE_0_DOUBLE_DIE_OFFSET;

    MME_ASSERT(0, "invalid number of dies");
    return SRAM_SIZE;
}
uint64_t MmeHalReader::getHBMStart() const { return DRAM_PHYS_BASE; }
uint64_t MmeHalReader::getHBMSize() const { return 0x1000000000000ull; } // should be taken from spec eventually
uint64_t MmeHalReader::getSMStart() const { return mmHD0_SYNC_MNGR_OBJS_BASE; }
unsigned MmeHalReader::getFirstMonitor() const
{
    return GAUDI3_DCORE0_FIRST_AVAILABLE_MONITOR + GAUDI3_SIM_QUEUE_ID_SIZE;
}
unsigned MmeHalReader::getMonitorNr() const
{
    return sizeof(gaudi3::block_sob_objs::mon_arm_0) / sizeof(gaudi3::block_sob_objs::mon_arm_0[0]);
}
unsigned MmeHalReader::getMinSoIdx() const { return GAUDI3_SIM_QUEUE_ID_SIZE + GAUDI3_DCORE0_FIRST_AVAILABLE_SYNC_OBJECT; }
unsigned MmeHalReader::getMaxSoIdx() const { return sizeof(gaudi3::block_sob_objs::sob_obj_0) / sizeof(uint32_t); }

}  // namespace gaudi3
