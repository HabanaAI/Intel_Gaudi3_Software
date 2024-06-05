#include "gaudi_mme_hal_reader.h"
#include "gaudi/mme.h"
#include "include/gaudi/new_descriptor_generator/mme_common.h"
#include "mme_assert.h"

namespace gaudi
{
unsigned MmeHalReader::getDieNr() const
{
    return 1;
}
unsigned MmeHalReader::getDcoreNr() const
{
    return 1;
}
    unsigned MmeHalReader::getMmeNr() const { return Mme::MME_MASTERS_NR; }
    unsigned MmeHalReader::getEuNr() const { return Mme::MME_CORES_NR; }
    unsigned MmeHalReader::getClSize() const
    {
        return GAUDI_MME_C_CL_SIZE;
    }
    unsigned MmeHalReader::getMemoryClSize() const
    {
        return getClSize();
    }
    unsigned MmeHalReader::getAccumsNr() const { return Mme::c_mme_accums_nr; }
    unsigned MmeHalReader::getSBSize() const { return Mme::c_mme_sb_size; }
    unsigned MmeHalReader::getMaxSBReuse() const { return Mme::c_mme_max_sb_reuse; }
    unsigned MmeHalReader::getSinglePortBw() const
    {
        return 200;
    }
    unsigned MmeHalReader::getLFSRSeedsNr() const { return Mme::c_mme_lfsr_seeds_nr; }
    unsigned MmeHalReader::getSizeOfDescNumIterMinus1() const { return sizeof(Mme::Desc::numIterationsMinus1); }
    unsigned MmeHalReader::getSizeOfOuterLoopSizeMinus1() const { return sizeof(Mme::MmeOuterLoop::sizeMinus1); }
    unsigned MmeHalReader::getSizeOfKernelSizeDim0() const { return sizeof(Mme::MmeKernelSize::dim[0]); }

    // memory space configurations
    uint64_t MmeHalReader::getSramStart(unsigned dieNr) const
    {
        MME_ASSERT(0, "not yet implemented");
        return 0;
    }
    uint64_t MmeHalReader::getSramSize(unsigned dieNr) const
    {
        MME_ASSERT(0, "not yet implemented");
        return 0;
    }
    uint64_t MmeHalReader::getHBMStart() const
    {
        MME_ASSERT(0, "not yet implemented");
        return 0;
    }
    uint64_t MmeHalReader::getHBMSize() const
    {
        MME_ASSERT(0, "not yet implemented");
        return 0;
    }
    uint64_t MmeHalReader::getSMStart() const
    {
        MME_ASSERT(0, "not yet implemented");
        return 0;
    }
    unsigned MmeHalReader::getMinSoIdx() const
    {
        MME_ASSERT(0, "not yet implemented");
        return 0;
    }
    unsigned MmeHalReader::getMaxSoIdx() const
    {
        MME_ASSERT(0, "not yet implemented");
        return 0;
    }
    unsigned MmeHalReader::getFirstMonitor() const
    {
        MME_ASSERT(0, "not yet implemented");
        return 0;
    }
    unsigned MmeHalReader::getMonitorNr() const
    {
        MME_ASSERT(0, "not yet implemented");
        return 0;
    }
}  // namespace gaudi
