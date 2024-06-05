#include "gaudi2_mme_hal_reader.h"
#include "include/mme_common/mme_common_enum.h"
#include "drm/habanalabs_accel.h"
#include "gaudi2/asic_reg_structs/sob_objs_regs.h"
#include "gaudi2/asic_reg/gaudi2_blocks.h"
#include "gaudi2/mme.h"
#include "gaudi2/gaudi2.h"
#include "mme_assert.h"

// values copied from LKD.
#define GAUDI2_DCORE0_FIRST_AVAILABLE_SYNC_OBJECT 392
#define GAUDI2_DCORE0_FIRST_AVAILABLE_MONITOR     228

namespace Gaudi2
{
unsigned MmeHalReader::getDieNr() const
{
    return 1;
}
unsigned MmeHalReader::getDcoreNr() const
{
    return 1;
}
unsigned MmeHalReader::getMmeNr() const { return Gaudi2::Mme::MME_CORE_PAIR_SIZE; }
unsigned MmeHalReader::getEuNr() const { return Gaudi2::Mme::MME_CORES_NR; }
unsigned MmeHalReader::getClSize() const { return Gaudi2::Mme::c_cl_size; }
unsigned MmeHalReader::getMemoryClSize() const
{
    return getClSize();
}
unsigned MmeHalReader::getAccumsNr() const { return Gaudi2::Mme::c_mme_accums_nr; }
unsigned MmeHalReader::getSBSize() const { return Gaudi2::Mme::c_mme_sb_size; }
unsigned MmeHalReader::getMaxSBReuse() const { return Gaudi2::Mme::c_mme_max_sb_reuse; }
unsigned MmeHalReader::getSinglePortBw() const
{
    return 200;
}
unsigned MmeHalReader::getLFSRSeedsNr() const { return Gaudi2::Mme::c_mme_lfsr_seeds_nr; }
unsigned MmeHalReader::getSizeOfDescNumIterMinus1() const
{
    return sizeof(Gaudi2::Mme::Desc::numIterationsMinus1);
}
unsigned MmeHalReader::getSizeOfOuterLoopSizeMinus1() const
{
    return sizeof(Gaudi2::Mme::MmeOuterLoop::sizeMinus1);
}

unsigned MmeHalReader::getSizeOfKernelSizeDim0() const { return sizeof(Gaudi2::Mme::MmeKernelSize::dim[0]); }

// The reduction tree required alignment of common dim.
// "common-dim % getBaseCommonDim(...)" must be 0.
unsigned MmeHalReader::getNumElementsForCommonDimAlignment(MmeCommon::EMmeDataType dataType,
                                                           MmeCommon::EMmeOpType opType) const
{
    switch (dataType)
    {
        case MmeCommon::e_type_fp16:
        case MmeCommon::e_type_bf16:
            return 8;
        case MmeCommon::e_type_fp32:
            return 2;
        case MmeCommon::e_type_fp32_ieee:
            return 1;
        case MmeCommon::e_type_tf32:
            return 4;
        case MmeCommon::e_type_fp8_143:
        case MmeCommon::e_type_fp8_152:
            return 16;
        default:
            MME_ASSERT(0, "invalid data type");
    }
    return 0;
}

// memory space configurations
uint64_t MmeHalReader::getSramStart(unsigned dieNr) const
{
    return SRAM_BASE_ADDR;
}
uint64_t MmeHalReader::getSramSize(unsigned dieNr) const
{
    return SRAM_SIZE;
}
uint64_t MmeHalReader::getHBMStart() const { return DRAM_PHYS_BASE; }
uint64_t MmeHalReader::getHBMSize() const { return 0x1000000000000ull; } // should be taken from spec eventually
uint64_t MmeHalReader::getSMStart() const { return mmDCORE0_SYNC_MNGR_OBJS_BASE; }
unsigned MmeHalReader::getFirstMonitor() const { return GAUDI2_DCORE0_FIRST_AVAILABLE_MONITOR; }
unsigned MmeHalReader::getMonitorNr() const
{
    return sizeof(gaudi2::block_sob_objs::mon_arm) / sizeof(gaudi2::block_sob_objs::mon_arm[0]);
}
unsigned MmeHalReader::getMinSoIdx() const { return GAUDI2_QUEUE_ID_SIZE + GAUDI2_DCORE0_FIRST_AVAILABLE_SYNC_OBJECT; }
unsigned MmeHalReader::getMaxSoIdx() const { return sizeof(gaudi2::block_sob_objs::sob_obj) / sizeof(uint32_t); }

}  // namespace Gaudi2
