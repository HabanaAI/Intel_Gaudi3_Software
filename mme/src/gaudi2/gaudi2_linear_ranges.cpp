#include "gaudi2_linear_ranges.h"

using namespace MmeCommon;

namespace Gaudi2
{
inline void getOutputAguIdx(const Mme::Desc& desc, unsigned& aguCount, std::array<EMmeOperandIdx, 4>& aguIdx)
{
    if (desc.brains.aguOut0.masterEn)
    {
        MME_ASSERT(desc.brains.aguOut0.slaveEn, "slave aguOut0 should be enabled");
        aguIdx[aguCount++] = Gaudi2::e_mme_agu_cout0_idx;
    }
    if (desc.brains.aguOut1.masterEn)
    {
        MME_ASSERT(desc.brains.aguOut1.slaveEn, "slave aguOut1 should be enabled");
        aguIdx[aguCount++] = Gaudi2::e_mme_agu_cout1_idx;
    }
}

unsigned RoiCalculator::mmeOperand2aguIdx(const EMmeOpType opType,
                                          const EMmeOperand operand,
                                          const Mme::Desc& desc,
                                          std::array<EMmeOperandIdx, 4>& aguIdx) const
{
    const unsigned aguReadsAMask = desc.header.aguReadsA;
    const unsigned aguReadsBMask = desc.header.aguReadsB;

    unsigned aguCount = 0;
    switch (operand)
    {
        case e_mme_op_x:
            if (opType != e_mme_dedx && opType != e_mme_transposed_dedx)
            {
                for (unsigned i = 0; i < e_mme_agu_cout0_idx; i++)
                {
                    if (aguReadsAMask & (1 << i))
                    {
                        aguIdx[aguCount++] = (EMmeOperandIdx) i;
                    }
                }
            }
            else
            {
                getOutputAguIdx(desc, aguCount, aguIdx);
            }
            break;

        case e_mme_op_w:
            if (!isDedwOperation(opType))
            {
                for (unsigned i = 0; i < e_mme_agu_cout0_idx; i++)
                {
                    if (aguReadsBMask & (1 << i))
                    {
                        aguIdx[aguCount++] = (EMmeOperandIdx) i;
                    }
                }
            }
            else
            {
                getOutputAguIdx(desc, aguCount, aguIdx);
            }
            break;

        case e_mme_op_y:
            if (opType == e_mme_fwd || opType == e_mme_ab || opType == e_mme_atb || opType == e_mme_abt ||
                opType == e_mme_atbt || opType == e_mme_reductionAdd)
            {
                getOutputAguIdx(desc, aguCount, aguIdx);
            }
            else
            {
                unsigned aguReadsMask;
                if (opType == e_mme_dedx || opType == e_mme_transposed_dedx)
                {
                    aguReadsMask = aguReadsAMask;
                }
                else
                {
                    MME_ASSERT(isDedwOperation(opType), "invalid operation");
                    aguReadsMask = aguReadsBMask;
                }

                for (unsigned i = 0; i < e_mme_agu_cout0_idx; i++)
                {
                    if (aguReadsMask & (1 << i))
                    {
                        aguIdx[aguCount++] = (EMmeOperandIdx) i;
                    }
                }
            }
            break;
        case e_mme_op_o:
            getOutputAguIdx(desc, aguCount, aguIdx);
            break;
        default:
            MME_ASSERT(0, "invalid operand");
    }

    return aguCount;
}

void RoiCalculator::simulateAgu(const EMmeOperand operand,
                                const Mme::Desc& desc,
                                std::vector<Gaudi2::AguRanges>* ranges) const
{
    std::array<EMmeOperandIdx, 4> aguIdx;

    unsigned aguNr = mmeOperand2aguIdx(m_params->opType, operand, desc, aguIdx);

    for (unsigned i = 0; i < aguNr; ++i)
    {
        genAddresses(&desc, aguIdx[i], m_params->opType, /*master=*/true, ranges);
    }
    for (unsigned i = 0; i < aguNr; ++i)
    {
        genAddresses(&desc, aguIdx[i], m_params->opType, /*master=*/false, ranges);
    }
}

static void aguRanges2Rois(const unsigned signalBase, std::vector<AguRanges>* ranges, OverlapRoi& roi)
{
    auto& subRois = *roi.subRois;
    for (unsigned i = 0; i < ranges->size(); i++)
    {
        OverlapSubRoi* currSubRoi = nullptr;
        AguRanges::const_iterator cbegin;
        AguRanges::const_iterator cend;
        (*ranges)[i].getCoveredSegments(0, UINT64_MAX, cbegin, cend);
        uint64_t prevStart = 0;
        uint64_t prevEnd = 0;
        for (auto it = cbegin; it != cend; ++it)
        {
            if (it->second.valid)
            {
                if (!currSubRoi)
                {
                    subRois.push_back(OverlapSubRoi());
                    currSubRoi = &subRois.back();
                    currSubRoi->relSoIdx = signalBase + i;
                }

                if (prevEnd != it->first)
                {
                    if (prevEnd != prevStart)
                    {
                        currSubRoi->ranges.push_back(DataRange<uint64_t>(prevStart, prevEnd));
                    }

                    prevStart = it->first;
                    prevEnd = it->first + it->second.size;
                }
                else
                {
                    prevEnd += it->second.size;
                }
            }
        }

        if (prevEnd != prevStart)
        {
            currSubRoi->ranges.push_back(DataRange<uint64_t>(prevStart, prevEnd));
        }
    }
}

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
    simulateAgu(operand, act.getDesc(0), &operandRanges);
    if (skipData.skipNorthActivation == false) simulateAgu(operand, act.getDesc(1), &operandRanges);

    unsigned numSkips = skipData.skipSignalsNr;
    aguRanges2Rois(numSkips, &operandRanges, roi);
}

/*
 * Count the number of signals ahead of the current activation for which its
 * inputs will still be needed.
 *
 * It is needed because we will skip the calculation of inputs that we already
 * simulated before, using this we know when will they last be needed and the
 * overlap will be adjusted accordingly.
 */
void RoiCalculator::countSkipSignals(ActivationVec& activations)
{
    unsigned i = 0;
    for (auto it = activations.begin(); it != activations.end(); it++)
    {
        auto& act = *it;
        MME_ASSERT(i + act.skipDataA.skipDescsNr < activations.size(),
                   "number of skip descriptors for operand A is larger than total num of desc");
        MME_ASSERT(i + act.skipDataB.skipDescsNr < activations.size(),
                   "number of skip descriptors for operand B is larger than total num of desc");
        unsigned iterations = std::max(act.skipDataA.skipDescsNr, act.skipDataB.skipDescsNr);
        auto skipIter = it;
        for (unsigned j = 0; j < iterations; ++j)
        {
            act.skipDataA.skipSignalsNr += (j < act.skipDataA.skipDescsNr ? skipIter->numSignals : 0);
            act.skipDataB.skipSignalsNr += (j < act.skipDataB.skipDescsNr ? skipIter->numSignals : 0);
            skipIter++;
        }
        i++;
    }
}

bool RoiCalculator::isStoreEn(MmeActivation& act) const
{
    return act.getDesc(0).header.storeEn0;
}

/*
 * Sets whether for a certain operand we can skip the ROI calculation for the
 * activation. this happens for every activation that is not in the first column
 * or the first row of the output.
 *
 * this will happen when activations are split either on the dense or the
 * spatial dimension, this will allow us to skip calculating for A or B
 * respectively.
 *
 * for the activation that will not be skipped, calculate how many activation
 * will be skipped ahead of it. in other words, when is the last activation that
 * will need to read the same operand as the current activation.
 */
void ConvRoiCalculator::setSkipRoiCalc(const MmeCommon::CommonGeoAttr& geoAttr, MmeActivation& act) const
{
    unsigned colsNr = m_recipe->getFcdSubviews().size();
    unsigned colIdx = m_recipe->getIterator().fcdIdx();

    // in dedw the spatial progression in output is expressed by the convIdx
    // instead of spIdx
    unsigned rowsNr;
    unsigned partialsNr;
    unsigned cRowIdx;
    if (m_params->isDedwOperation())
    {
        rowsNr = m_recipe->getNonSpatialSubviews().size();
        partialsNr = m_recipe->getSpSubviews().size();
        cRowIdx = m_recipe->getIterator().nonSpatialIdx();
    }
    else
    {
        rowsNr = m_recipe->getSpSubviews().size();
        partialsNr = m_recipe->getNonSpatialSubviews().size();
        cRowIdx = m_recipe->getIterator().spIdx();
    }

    act.skipDataA.skipActivation = (0 != colIdx);
    act.skipDataB.skipActivation = (0 != cRowIdx);
    act.skipDataC.skipActivation = !act.getDesc(0).header.storeEn0 && !act.getDesc(0).header.storeEn1 &&
                                   !act.getDesc(1).header.storeEn0 && !act.getDesc(1).header.storeEn1;

    GeometryGrid mmeGrid = geoAttr.getMmeGrid();
    act.skipDataA.skipNorthActivation = mmeGrid.fcd > 1;  // skipA if both MMEs read the same A
    act.skipDataB.skipNorthActivation = mmeGrid.spatial > 1;  // skipB if both MMEs read the same B
    act.skipDataC.skipNorthActivation = !act.getDesc(1).header.storeEn0 && !act.getDesc(1).header.storeEn1;

    if (m_recipe->raster)
    {
        act.skipDataA.skipDescsNr = (0 == colIdx ? (colsNr - 1) * partialsNr : 0);
        act.skipDataB.skipDescsNr = (0 == cRowIdx ? colsNr * (rowsNr - 1) * partialsNr : 0);
    }
    else
    {
        act.skipDataA.skipDescsNr = (0 == colIdx ? (colsNr - 1) * rowsNr * partialsNr : 0);
        act.skipDataB.skipDescsNr = (0 == cRowIdx ? (rowsNr - 1) * partialsNr : 0);
    }
    act.skipDataC.skipDescsNr = 0;

    act.skipDataA.skipSignalsNr = 0;
    act.skipDataB.skipSignalsNr = 0;
    act.skipDataC.skipSignalsNr = 0;
}

void BGemmRoiCalculator::setSkipRoiCalc(const MmeCommon::CommonGeoAttr& geoAttr, MmeActivation& act) const
{
    for (SkipData* skipData : {&act.skipDataA, &act.skipDataB, &act.skipDataC})
    {
        skipData->skipActivation = false;
        skipData->skipNorthActivation = false;
        skipData->skipDescsNr = 0;
        skipData->skipSignalsNr = 0;
    }
}

}  // namespace Gaudi2
