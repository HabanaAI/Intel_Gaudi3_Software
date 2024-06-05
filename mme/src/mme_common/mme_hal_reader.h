#ifndef MME__HAL_READER_H
#define MME__HAL_READER_H

#include "include/mme_common/mme_common_enum.h"

namespace MmeCommon
{
class MmeHalReader
{
public:
    virtual unsigned getDcoreNr() const = 0;
    virtual unsigned getMmePerDie() const { return getMmeNr() / getDieNr(); }
    virtual unsigned getDieNr() const = 0;
    virtual unsigned getMmeNr() const = 0;
    virtual unsigned getEuNr() const = 0;
    virtual unsigned getClSize() const = 0;
    virtual unsigned getMemoryClSize() const = 0;
    virtual unsigned getAccumsNr() const = 0;
    virtual unsigned getSBSize() const = 0;
    virtual unsigned getMaxSBReuse() const = 0;
    virtual unsigned getSinglePortBw() const = 0;
    virtual unsigned getLFSRSeedsNr() const = 0;
    virtual unsigned getSizeOfDescNumIterMinus1() const = 0;
    virtual unsigned getSizeOfOuterLoopSizeMinus1() const = 0;
    virtual unsigned getSizeOfKernelSizeDim0() const = 0;
    virtual unsigned getNumElementsForCommonDimAlignment(MmeCommon::EMmeDataType dataType,
                                                         MmeCommon::EMmeOpType opType) const = 0;
    virtual MmeCommon::ChipType getChipType() const = 0;
    virtual unsigned getClkFreqMHz() const = 0;
    virtual bool isGemmMappedToBatchGemm() const = 0;

    // memory space configurations
    virtual uint64_t getSramStart(unsigned dieNr = 0) const = 0;
    virtual uint64_t getSramSize(unsigned dieNr = 0) const = 0;
    virtual uint64_t getHBMStart() const = 0;
    virtual uint64_t getHBMSize() const = 0;
    virtual uint64_t getSMStart() const = 0;
    virtual unsigned getFirstMonitor() const = 0;
    virtual unsigned getMonitorNr() const = 0;
    virtual unsigned getMinSoIdx() const = 0;
    virtual unsigned getMaxSoIdx() const = 0;
    virtual unsigned getSyncObjectMaxValue() const { return 32768; }

    virtual ~MmeHalReader() = default;
};

}  // namespace MmeCommon

#endif //MME__HAL_READER_H
