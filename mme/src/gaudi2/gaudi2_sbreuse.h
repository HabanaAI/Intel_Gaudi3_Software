#include "gaudi2_mme_hal_reader.h"
#include "mme_common/common_sbreuse.h"
#include "gaudi2/mme.h"
#include "include/mme_common/mme_common_enum.h"

namespace Gaudi2
{
class Gaudi2SBReuse : public MmeCommon::CommonSBReuse
{
public:
    Gaudi2SBReuse(const MmeCommon::MmeLayerParams& params,
                  const MmeCommon::CommonGeoAttr& geoAttr,
                  const MmeCommon::MmeRecipe& recipe)
    : CommonSBReuse(params, geoAttr, Gaudi2::MmeHalReader::getInstance(), recipe) {};
    virtual ~Gaudi2SBReuse() override = default;

    virtual void setDescSbRepeatSteps(unsigned repeatDenseSteps, unsigned repeatSpatialSteps, void* descPtr) override;
    virtual void setDescSbRepeatMask(MmeCommon::EMmeLoopMask repeatDenseMask,
                                     MmeCommon::EMmeLoopMask repeatSpatialMask,
                                     void* descPtr) override;
    virtual void setDescBrainsAgu(uint8_t denseBitMaskLoops, uint8_t spatialBitMaskLoops, void* descPtr) override;
    virtual void
    setDescAccums(unsigned rollAccums, bool isAccumOutputEnable, bool isStoreOutputEnable, void* descPtr) override;
};
}  // namespace Gaudi2