#include "gaudi_linear_ranges.h"

using namespace MmeCommon;

namespace gaudi
{
/*
 * simulate the tensor walk using the chip specific agu implementation
 * this should be used only for complex walks.
 * due to extremely long run time this should be avoided at all cost
 */
void RoiCalculator::addSimulatedTensor(const EMmeOperand operand, MmeActivation& act, OverlapRoi& roi) const
{
    const SkipData& skipData = getSkipDataForOperand(operand, act, m_params->opType);
    if (skipData.skipActivation) return;

    std::vector<AguRanges> operandRanges;
    operandRanges.resize(std::max(act.numSignals, (unsigned) 1));
    EMmeGaudiInternalAgu internalAguType = mmeOperandToGaudiAgu(operand, m_params->opType, act.getDesc(0).header.transO);
    switch (internalAguType)
    {
        case e_mme_agu_shared:
            aguStatsGetRanges(true, true, &act.getDesc(1), 0, &operandRanges);
            if (!skipData.skipNorthActivation) aguStatsGetRanges(true, true, &act.getDesc(0), 0, &operandRanges);
            break;
        case e_mme_agu_local:
            aguStatsGetRanges(true, false, &act.getDesc(1), 0, &operandRanges);
            if (!skipData.skipNorthActivation) aguStatsGetRanges(true, false, &act.getDesc(0), 0, &operandRanges);
            break;
        case e_mme_agu_out:
            aguStatsGetRanges(false, false, &act.getDesc(1), 0, &operandRanges);
            if (!skipData.skipNorthActivation) aguStatsGetRanges(false, false, &act.getDesc(0), 0, &operandRanges);
            break;
        default:
            MME_ASSERT(0, "invalid operation");
    }
    unsigned numSkips = skipData.skipSignalsNr;
    aguRanges2Rois(numSkips, &operandRanges, roi);
}

bool RoiCalculator::isStoreEn(MmeActivation& act) const
{
    return act.getDesc(0).header.storeEn;
}

}  // namespace gaudi
