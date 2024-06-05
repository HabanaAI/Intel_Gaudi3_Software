#include "common_geo_attr.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/recipe.h"
#include "include/mme_common/recipe_generator.h"
namespace MmeCommon
{
struct DimensionSteps
{
    unsigned denseStepsNr = 0;
    unsigned spatialStepsNr = 0;
    unsigned filterStepsNr = 0;
};
struct LoopMasks
{
    EMmeLoopMask denseLoopMask = e_mme_outer_loop;
    EMmeLoopMask spatialLoopMask = e_mme_outer_loop;
    EMmeLoopMask accumDimLoopMask = e_mme_outer_loop;
};

// Class that handles SB repeats info
class CommonSBReuse
{
public:
    CommonSBReuse(const MmeLayerParams& params,
                  const CommonGeoAttr& geoAttr,
                  const MmeCommon::MmeHalReader& mmeHal,
                  const MmeCommon::MmeRecipe& recipe);
    virtual ~CommonSBReuse() = default;
    void configDescSBReuse(void* descPtr);

protected:
    virtual void setDescSbRepeatSteps(unsigned repeatDenseSteps, unsigned repeatSpatialSteps, void* descPtr) = 0;
    virtual void setDescSbRepeatMask(MmeCommon::EMmeLoopMask repeatDenseMask,
                                     MmeCommon::EMmeLoopMask repeatSpatialMask,
                                     void* descPtr) = 0;
    virtual void setDescBrainsAgu(uint8_t denseBitMaskLoops, uint8_t spatialBitMaskLoops, void* descPtr) = 0;
    virtual void
    setDescAccums(unsigned rollAccums, bool isAccumOutputEnable, bool isStoreOutputEnable, void* descPtr) = 0;

    const CommonGeoAttr& m_geoAttr;
    const MmeCommon::MmeRecipe& m_recipe;
    const MmeLayerParams& m_params;
    const MmeCommon::MmeHalReader& m_mmeHal;

private:
    uint8_t getFilterBitMaskLoops();
    void setSteps();
    void setLoopMasks();
    void setAguBitMaskLoopsAndRepeatSteps(void* descPtr);
    void setRepeatLoopMasks(void* descPtr);
    void setAccumsData(void* descPtr);
    void updateSpatialStepsNr(unsigned stepsNr);

    DimensionSteps m_steps;  // The number of steps in each Dimension
    LoopMasks m_masks;  // The mask of the loop that is associated with the dimension
};
}  // namespace MmeCommon