#pragma once

// eager includes (relative to src/eager/lib/)
#include "eager_brain_base.h"

namespace eager_mode
{
namespace gaudi2_spec_info
{
class EagerMmeBrain final : public EagerMmeBrainBase
{
    static constexpr uint64_t CONV_PACKING_THRESHOLD = 50000;
    static constexpr uint64_t ROLLUP_TIME            = 256;

public:
    constexpr EagerMmeBrain() : EagerMmeBrainBase(CONV_PACKING_THRESHOLD, ROLLUP_TIME) {}

    virtual MmeCommon::ChipType getChipType() const override { return MmeCommon::e_mme_Gaudi2; }

    virtual const SupportedGeometries& getSupportedGeometries() const override
    {
        static const SupportedGeometries supportedMmeGeometries = {
            SupportedGeometryProperties {MmeCommon::e_mme_geometry_2xh, 256, 512},
            SupportedGeometryProperties {MmeCommon::e_mme_geometry_2xw, 512, 256},
            SupportedGeometryProperties {MmeCommon::e_mme_geometry_4xh, 128, 1024},
            SupportedGeometryProperties {MmeCommon::e_mme_geometry_4xw, 1024, 128}};
        return supportedMmeGeometries;
    }
    virtual unsigned getTotalMmeSize() const override { return 256 * 512; }  // used 2xh dimensions
    virtual bool isConcurrencySupportedForOutputDataType(synDataType outputDataType) const override { return true; }

protected:
    virtual unsigned getMaxMmeLength() const override { return 1024; }
};

}  // namespace gaudi2_spec_info

}  // namespace eager_mode
