#pragma once

// eager includes (relative to src/eager/lib/)
#include "eager_brain_base.h"

// synapse-internal includes (relative to src/)
#include "hal_reader/gaudi3/hal.h"

namespace eager_mode
{
namespace gaudi3_spec_info
{
class EagerMmeBrain final : public EagerMmeBrainBase
{
    static constexpr uint64_t CONV_PACKING_THRESHOLD = 50000;
    static constexpr uint64_t ROLLUP_TIME            = 512;

public:
    constexpr EagerMmeBrain() : EagerMmeBrainBase(CONV_PACKING_THRESHOLD, ROLLUP_TIME) {}

    virtual MmeCommon::ChipType getChipType() const override { return MmeCommon::e_mme_Gaudi3; }

    virtual const SupportedGeometries& getSupportedGeometries() const override
    {
        static const SupportedGeometries supportedMmeGeometries = {
            SupportedGeometryProperties {MmeCommon::e_mme_geometry_2xh, 512, 512},
            SupportedGeometryProperties {MmeCommon::e_mme_geometry_2xw, 1024, 256},
            SupportedGeometryProperties {MmeCommon::e_mme_geometry_4xh, 256, 1024},
            SupportedGeometryProperties {MmeCommon::e_mme_geometry_4xw, 2048, 128}};
        return supportedMmeGeometries;
    }
    virtual unsigned getTotalMmeSize() const override { return 512 * 512; }  // used 2xh dimensions

    virtual bool isConcurrencySupportedForOutputDataType(synDataType outputDataType) const override
    {
        // TODO SW-136691: The following condition is a temporary workaround for Gaudi3
        if ((outputDataType == syn_type_fp16 || outputDataType == syn_type_bf16))
        {
            // Check that the concurrency level does not exceed 16: TODO SW-137230
            // The concurrency level defines the number of partial outputs that are produced by the Gemms
            // that are concurrently used. When the data type is either fp16 or bf16, each partial result
            // is produced in lower accuracy (compared to fp32 if cd concurrency does not take place). It is
            // proved that the impact on accuracy up to 16 partial results is small enough.
            constexpr unsigned maxGemmsNr = gaudi3::halFullChipSpecificInfo.numMmeEngines *
                                            gaudi3::hal::numMmeCoresPerEngine * gaudi3::hal::maxNumMmeGemmsPerCore;
            constexpr bool isSupported = maxGemmsNr <= 16;
            return isSupported;
        }
        return true;
    }

protected:
    virtual unsigned getMaxMmeLength() const override { return 2048; }
};

}  // namespace gaudi3_spec_info

}  // namespace eager_mode
