#ifndef MME__GAUDI3_MME_HAL_READER_H
#define MME__GAUDI3_MME_HAL_READER_H

#include "src/mme_common/mme_hal_reader.h"

namespace gaudi3
{
class MmeHalReader : private MmeCommon::MmeHalReader
{
public:
    static const MmeCommon::MmeHalReader& getInstance()
    {
        static const MmeHalReader mmeHalReader;
        return mmeHalReader;
    }

    virtual ~MmeHalReader() = default;
private:
    MmeHalReader() = default;

    virtual unsigned getDcoreNr() const override;
    virtual unsigned getDieNr() const override;
    virtual unsigned getMmeNr() const override;
    virtual unsigned getEuNr() const override;
    virtual unsigned getClSize() const override;
    virtual unsigned getMemoryClSize() const override;
    virtual unsigned getAccumsNr() const override;
    virtual unsigned getSBSize() const override;
    virtual unsigned getMaxSBReuse() const override;
    virtual unsigned getSinglePortBw() const override;
    virtual unsigned getLFSRSeedsNr() const override;
    virtual unsigned getSizeOfDescNumIterMinus1() const override;
    virtual unsigned getSizeOfOuterLoopSizeMinus1() const override;
    virtual unsigned getSizeOfKernelSizeDim0() const override;
    virtual MmeCommon::ChipType getChipType() const override { return MmeCommon::ChipType::e_mme_Gaudi3; }
    virtual unsigned getClkFreqMHz() const override { return 1600; };  // current PLDM clk update in the future
    // The reduction tree required alignment of common dim.
    // "common-dim % getBaseCommonDim(...)" must be 0.
    virtual unsigned getNumElementsForCommonDimAlignment(MmeCommon::EMmeDataType dataType,
                                                         MmeCommon::EMmeOpType opType) const override;

    // checks - to be deleted soon
    virtual bool isGemmMappedToBatchGemm() const override { return false; }

    // memory space configurations
    virtual uint64_t getSramStart(unsigned dieNr) const override;
    virtual uint64_t getSramSize(unsigned dieNr) const override;
    virtual uint64_t getHBMStart() const override;
    virtual uint64_t getHBMSize() const override;
    virtual uint64_t getSMStart() const override;
    virtual unsigned getFirstMonitor() const override;
    virtual unsigned getMonitorNr() const override;
    virtual unsigned getMinSoIdx() const override;
    virtual unsigned getMaxSoIdx() const override;
};

}  // namespace gaudi3

#endif //MME__GAUDI3_MME_HAL_READER_H
