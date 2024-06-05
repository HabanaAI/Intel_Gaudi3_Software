#include "mme_common/common_sbreuse.h"
#include "gaudi3/mme.h"
#include "gaudi3_mme_hal_reader.h"
#include "include/mme_common/mme_common_enum.h"

namespace gaudi3
{
class Gaudi3SBReuse : public MmeCommon::CommonSBReuse
{
public:
    Gaudi3SBReuse(const MmeCommon::MmeLayerParams& params,
                  const MmeCommon::CommonGeoAttr& geoAttr,
                  const MmeCommon::MmeRecipe& recipe)
    : CommonSBReuse(params, geoAttr, gaudi3::MmeHalReader::getInstance(), recipe) {};
    virtual ~Gaudi3SBReuse() override = default;

    virtual void setDescSbRepeatSteps(unsigned repeatDenseSteps, unsigned repeatSpatialSteps, void* descPtr) override;
    virtual void setDescSbRepeatMask(MmeCommon::EMmeLoopMask repeatDenseMask,
                                     MmeCommon::EMmeLoopMask repeatSpatialMask,
                                     void* descPtr) override;
    virtual void setDescBrainsAgu(uint8_t denseBitMaskLoops, uint8_t spatialBitMaskLoops, void* descPtr) override;
    virtual void
    setDescAccums(unsigned rollAccums, bool isAccumOutputEnable, bool isStoreOutputEnable, void* descPtr) override;
};
}  // namespace gaudi3