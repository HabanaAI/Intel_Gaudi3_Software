
#include "common_sbreuse.h"
#include <cstdint>
#include "mme_hal_reader.h"

namespace MmeCommon
{
CommonSBReuse::CommonSBReuse(const MmeLayerParams& params,
                             const CommonGeoAttr& geoAttr,
                             const MmeCommon::MmeHalReader& mmeHal,
                             const MmeCommon::MmeRecipe& recipe)
: m_params(params), m_recipe(recipe), m_geoAttr(geoAttr), m_mmeHal(mmeHal)
{
    if (m_params.isConvOperation() || m_params.isGemmOperation())
    {
        setSteps();
        setLoopMasks();
    }
}
//  calculate number of steps in each dimension
void CommonSBReuse::setSteps()
{
    const auto& fcdView = m_recipe.curFcd();
    const auto& spView = m_recipe.curSp();
    const auto& convView = m_recipe.curNonSpatial();
    bool isDedw = (m_params.isDedwOperation());

    m_steps.denseStepsNr = div_round_up(fcdView.viewSize, m_geoAttr.getGeometryWidth());
    unsigned spatialSize = isDedw ? convView.sizes[1] : spView.viewSize;
    m_steps.spatialStepsNr = div_round_up(spatialSize, m_geoAttr.getGeometryHeight());
    if (m_params.isDedwOrGemm())
    {
        unsigned batchLoops[c_batchDimNr] = {convView.sizes[2], convView.sizes[3], convView.sizes[4]};
        if (!isDedw)
        {
            // the recipe currently doesnt divide the concurrent dim for bgemms.
            unsigned concurrentDim = m_geoAttr.getConcurrentDim() - GEMM_DIM_B1;
            batchLoops[concurrentDim] = div_round_up(batchLoops[concurrentDim], m_geoAttr.getGeometryConcurrency());
        }
        m_steps.filterStepsNr = batchLoops[0] * batchLoops[1] * batchLoops[2];
    }
    else
    {
        m_steps.filterStepsNr = 1;
    }
}

// Setup loop mask
void CommonSBReuse::setLoopMasks()
{
    const MmeCommon::EMmePattern pattern = m_params.strategy.pattern;
    //  In fwd and dedx the filters are accumulated in the EU
    m_masks.accumDimLoopMask = m_params.isFwdOrDedx() ? e_mme_conv_loop_2 : e_mme_gemm_loop;
    const bool dontUseSpatialLoop = m_params.isDedwOrGemm();
    const EMmeLoopDim dim = dontUseSpatialLoop ? EMmeLoopDim::dim_c : EMmeLoopDim::dim_s;
    m_masks.spatialLoopMask = pattern2LoopMask(pattern, dim);
    m_masks.denseLoopMask = pattern2LoopMask(pattern, EMmeLoopDim::dim_k);
}

uint8_t CommonSBReuse::getFilterBitMaskLoops()
{
    uint8_t filterBitMaskLoops = 0;
    EMmeLoopMask filterLoopMask = e_mme_gemm_loop;
    if (m_params.isDedwOperation())  // can reuse on filter only in dedw
    {
        filterLoopMask = pattern2LoopMask(m_params.strategy.pattern, EMmeLoopDim::dim_f);
        EMmeLoopMask curFilterLoopMask = filterLoopMask;
        for (unsigned convDim = 0; convDim < MME_MAX_CONV_DIMS - 1; ++convDim)
        {
            MME_ASSERT(curFilterLoopMask <= e_mme_outer_loop, "loop mask cannot be above outer_loop");

            filterBitMaskLoops |= getLoopFromLoopMask(curFilterLoopMask);
            do
            {
                curFilterLoopMask = (EMmeLoopMask)((((unsigned) curFilterLoopMask) << 1) + 1);
            } while (curFilterLoopMask == e_mme_tetris_loop);
        }
    }
    return filterBitMaskLoops;
}
void CommonSBReuse::updateSpatialStepsNr(unsigned stepsNr)
{
    m_steps.spatialStepsNr = stepsNr;
}

// Setup bit mask loops
void CommonSBReuse::setAguBitMaskLoopsAndRepeatSteps(void* descPtr)
{
    uint8_t aguDenseBitMaskLoops = 0;
    unsigned sbRepeatDenseStepsNr = 0;
    if (m_recipe.reuseA())
    {
        aguDenseBitMaskLoops = getLoopFromLoopMask(m_masks.denseLoopMask);
        sbRepeatDenseStepsNr = m_steps.denseStepsNr;
    }
    if (m_params.isGemmDmaOperation())
    {
        MME_ASSERT(!m_recipe.reuseA(), "A operand cannot be reused in gemm transpose operation");
        // the first loop is used to advance B over the CD, A uses the gemm loop so it needs to mask the relevant loop.
        aguDenseBitMaskLoops = e_mme_conv_loop_0;
    }

    uint8_t aguSpatialBitMaskLoops = 0;
    unsigned sbRepeatSpatialStepsNr = 0;
    if (m_recipe.reuseB())
    {
        uint8_t filterBitMaskLoops = getFilterBitMaskLoops();
        uint8_t spatialBitMaskLoops = getLoopFromLoopMask(m_masks.spatialLoopMask);
        bool isDedw = (m_params.isDedwOperation());
        const MmeCommon::EMmePattern pattern = m_params.strategy.pattern;
        switch (pattern)
        {
            case e_mme_sp_reduction_ckf:
                if (m_geoAttr.isOperandFullyBroadcasted(e_mme_op_b) && (m_steps.denseStepsNr == 1))
                {
                    aguSpatialBitMaskLoops = spatialBitMaskLoops | filterBitMaskLoops;
                    sbRepeatSpatialStepsNr = m_steps.filterStepsNr * m_steps.spatialStepsNr;
                    updateSpatialStepsNr(sbRepeatSpatialStepsNr);
                }
                else
                {
                    aguSpatialBitMaskLoops = filterBitMaskLoops;
                    sbRepeatSpatialStepsNr = m_steps.filterStepsNr;
                    m_masks.spatialLoopMask = pattern2LoopMask(pattern, EMmeLoopDim::dim_f);
                    updateSpatialStepsNr(sbRepeatSpatialStepsNr);
                }
                break;
            case e_mme_sp_reduction_kcf:
            case e_mme_sp_reduction_cfk:
            case e_mme_sp_reduction_kfc:
            case e_mme_sp_reduction_fck:
                // C and F are adjacent, check if it is possible to reuse on both of them.
                // in cfk/fck spatialBitMaskLoops will be used only if k==1 (no movement in k)
                if (m_geoAttr.isOperandFullyBroadcasted(e_mme_op_b))
                {
                    //  fully broadcased reuse on both C and F
                    aguSpatialBitMaskLoops = spatialBitMaskLoops | filterBitMaskLoops;
                    sbRepeatSpatialStepsNr = m_steps.filterStepsNr * m_steps.spatialStepsNr;
                    updateSpatialStepsNr(sbRepeatSpatialStepsNr);
                }
                else
                {
                    //  not fully broadcasted reuse of first spatial movement
                    bool filterFirst = pattern == e_mme_sp_reduction_kcf || pattern == e_mme_sp_reduction_cfk;
                    aguSpatialBitMaskLoops = filterFirst ? filterBitMaskLoops : spatialBitMaskLoops;
                    sbRepeatSpatialStepsNr = filterFirst ? m_steps.filterStepsNr : m_steps.spatialStepsNr;
                    updateSpatialStepsNr(sbRepeatSpatialStepsNr);
                }
                break;
            case e_mme_sp_reduction_fkc:
                if (m_geoAttr.isOperandFullyBroadcasted(e_mme_op_b) && (m_steps.denseStepsNr == 1))
                {
                    aguSpatialBitMaskLoops = spatialBitMaskLoops | filterBitMaskLoops;
                    sbRepeatSpatialStepsNr = m_steps.filterStepsNr * m_steps.spatialStepsNr;
                    updateSpatialStepsNr(sbRepeatSpatialStepsNr);
                }
                else
                {
                    aguSpatialBitMaskLoops = spatialBitMaskLoops;
                    sbRepeatSpatialStepsNr = m_steps.spatialStepsNr;
                }
                break;
            case e_mme_z_reduction_skf:
            case e_mme_z_reduction_ksf:
                // no movement on filter dimensions, a reuse is simply the spatial size.
                aguSpatialBitMaskLoops = spatialBitMaskLoops;
                sbRepeatSpatialStepsNr = m_steps.spatialStepsNr;
                break;
            default:
                MME_ASSERT(0, "invalid pattern");
        }
    }

    setDescSbRepeatSteps(sbRepeatDenseStepsNr, sbRepeatSpatialStepsNr, descPtr);
    setDescBrainsAgu(aguDenseBitMaskLoops, aguSpatialBitMaskLoops, descPtr);
}

void CommonSBReuse::setRepeatLoopMasks(void* descPtr)
{
    const bool reuseA = m_recipe.reuseA();
    const bool reuseB = m_recipe.reuseB();

    MmeCommon::EMmeLoopMask sbRepeatDenseMask = e_mme_gemm_loop;
    MmeCommon::EMmeLoopMask sbRepeatSpatialMask = e_mme_gemm_loop;

    // Set repeat masks
    switch (m_recipe.reuseType())
    {
        case e_mme_no_reuse:  // No reuse
            MME_ASSERT(!reuseA && !reuseB, "reuse should be false in no-reuse mode");
            sbRepeatDenseMask = e_mme_gemm_loop;
            sbRepeatSpatialMask = e_mme_gemm_loop;
            break;
        case e_mme_1d_reuse_a:  // Reuse operand A only
            MME_ASSERT(reuseA && !reuseB, "expected reuse on operand A only");
            sbRepeatDenseMask = m_masks.accumDimLoopMask;
            sbRepeatSpatialMask = e_mme_gemm_loop;
            break;
        case e_mme_1d_reuse_b:  // Reuse operand B only
            MME_ASSERT(!reuseA && reuseB, "expected reuse on operand B only");
            sbRepeatDenseMask = e_mme_gemm_loop;
            sbRepeatSpatialMask = m_masks.accumDimLoopMask;
            break;
        case e_mme_2d_reuse_ab:  // A reused first then B
            MME_ASSERT(reuseA && reuseB, "expected reuse on both operand in 2d reuse mode");
            sbRepeatDenseMask = m_masks.accumDimLoopMask;
            sbRepeatSpatialMask = m_masks.spatialLoopMask;
            break;
        case e_mme_2d_reuse_ba:  // B reused first then A
            MME_ASSERT(reuseA && reuseB, "expected reuse on both operand in 2d reuse mode");
            sbRepeatDenseMask = m_masks.denseLoopMask;
            sbRepeatSpatialMask = m_masks.accumDimLoopMask;
            break;
        default:
            MME_ASSERT(0, "invalid reuse type");
    }
    setDescSbRepeatMask(sbRepeatDenseMask, sbRepeatSpatialMask, descPtr);
}
void CommonSBReuse::setAccumsData(void* descPtr)
{
    // The number of accumulator inc to do after the last rollup
    unsigned rollAccums = 0;
    if (m_recipe.isLastPartial())
    {
        if (m_geoAttr.getDoubleAccumsBit())
        {
            //  in case doubleAccums isnt used the Acc index must be an even value.
            //  an odd value an only be reach in doubleAccums mode since in regular mode accum index is always
            //  incremented by 2. so whenever a doubleAccum activation finishes on an odd Accum, roll ahead by 1.
            unsigned stepsNr = m_steps.denseStepsNr * m_steps.spatialStepsNr * m_steps.filterStepsNr;
            rollAccums = stepsNr % 2;
        }
    }
    else
    {
        const unsigned accumNr = (m_steps.spatialStepsNr != 1) ? m_steps.spatialStepsNr : m_steps.denseStepsNr;
        if (m_geoAttr.getDoubleAccumsBit())
        {
            rollAccums = (2 * m_mmeHal.getAccumsNr()) - accumNr;
        }
        else
        {
            // Multiplying by 2 as half ACC is the default unit. In this case we use 1 ACC ie two units.
            rollAccums = 2 * (m_mmeHal.getAccumsNr() - accumNr);
        }
    }
    setDescAccums(rollAccums, m_recipe.isAccumEn(), m_recipe.isStoreEn(), descPtr);
}

void CommonSBReuse::configDescSBReuse(void* descPtr)
{
    if (m_params.isConvOperation() || m_params.isGemmOperation())
    {
        setAguBitMaskLoopsAndRepeatSteps(descPtr);
        setRepeatLoopMasks(descPtr);
        setAccumsData(descPtr);
    }
}

}  // namespace MmeCommon
