#include "gaudi/headers/mme_descriptor_comparator.h"
#include "gaudi/mme_test/mme_test.h"

bool MmeDescriptorComparator::compare(const std::list<std::pair<gaudi::MmeActivation, gaudi::MmeActivation>>& newActivations,
                                      const std::list<std::pair<gaudi::MmeActivation, gaudi::MmeActivation>>& oldActivations)
{
    if (newActivations.size() == oldActivations.size())
    {
        atomicColoredPrint(COLOR_RED,
                           "[NEW_MME_STACK] Compare: activation list size doesnt match ! new : %lu | old : %lu",
                           newActivations.size(), oldActivations.size())
        return false;
    }
    unsigned long listLength = newActivations.size();
    bool status = true;
    auto newActListIt = newActivations.begin();
    auto oldActListIt = oldActivations.begin();
    for (unsigned actIdx = 0; actIdx < listLength; actIdx++)
    {
        // south activation
        status &= compareActivation(newActListIt->first, oldActListIt->first);
        if (!status)
        {
            m_where += " in activation " + std::to_string(actIdx) + " (south)";
            status = false;
            break;
        }
        // north activation
        status &= compareActivation(newActListIt->second, oldActListIt->second);
        if (!status)
        {
            m_where += " in activation " + std::to_string(actIdx) + " (north)";
            status = false;
            break;
        }
        newActListIt++;
        oldActListIt++;
    }
    if (!status)
    {
        printError();
    }
    return status;
}

bool MmeDescriptorComparator::compareActivation(const gaudi::MmeActivation& newAct,
                                                const gaudi::MmeActivation& oldAct)
{
    bool status = true;
    do
    {
        status &= compareDesc(newAct.getDesc(0), oldAct.getDesc(0));
        if (!status)
        {
            m_where += " in south desc";
            break;
        }
        status &= compareDesc(newAct.getDesc(1), oldAct.getDesc(1));
        if (!status)
        {
            m_where += " in north desc";
            break;
        }
        status &= compareOverLapRoi(newAct.roiX, oldAct.roiX);
        if (!status)
        {
            m_where += " in roiX";
            break;
        }
        status &= compareOverLapRoi(newAct.roiW, oldAct.roiW);
        if (!status)
        {
            m_where += " in roiW";
            break;
        }
        status &= compareOverLapRoi(newAct.roiY, oldAct.roiY);
        if (!status)
        {
            m_where += " in roiY";
            break;
        }
        status &= (newAct.numSignals == oldAct.numSignals);
        if (!status)
        {
            m_where += " in numSignals";
            break;
        }
    } while(false);
    return status;
}



bool MmeDescriptorComparator::compareDesc(const Mme::Desc& newDesc,
                                          const Mme::Desc& oldDesc)
{
    if (!compareAttribute(newDesc.header.ddw, oldDesc.header.ddw, "header"))
    {
        return false;
    }
    if (!compareAttribute(newDesc.conv.kernelSizeMinus1.dw, oldDesc.conv.kernelSizeMinus1.dw, "conv.kernelSizeMinus1"))
    {
        return false;
    }
    for (unsigned i = 0; i < Mme::c_mme_max_conv_dims; i++)
    {
        if (!compareAttribute(newDesc.conv.associatedDims[i].w,
                              oldDesc.conv.associatedDims[i].w,
                              "conv.associatedDims[" + std::to_string(i) + "]"))
        {
            return false;
        }
    }
    if (!compareAttribute(newDesc.numIterationsMinus1, oldDesc.numIterationsMinus1, "numIterationsMinus1"))
    {
        return false;
    }
    if (!compareAttribute(newDesc.outerLoop.dw, oldDesc.outerLoop.dw, "outerLoop"))
    {
        return false;
    }
    if (!compareTensor(newDesc.tensorS, oldDesc.tensorS, "tensorS"))
    {
        return false;
    }
    if (!compareAgu(newDesc.aguS, oldDesc.aguS, "aguS"))
    {
        return false;
    }
    if (!compareTensor(newDesc.tensorL, oldDesc.tensorL, "tensorL"))
    {
        return false;
    }
    if (!compareTensor(newDesc.tensorO, oldDesc.tensorO, "tensorO"))
    {
        return false;
    }
    for (unsigned i = 0 ; i< Mme::e_mme_local_and_remote; i++)
    {
        if (!compareAgu(newDesc.aguL[i], oldDesc.aguL[i], "aguL[" + std::to_string(i) + "]"))
        {
            return false;
        }
        if (!compareAgu(newDesc.aguO[i], oldDesc.aguO[i], "aguO[" + std::to_string(i) + "]"))
        {
            return false;
        }
    }
    if (!compareAttribute(newDesc.sbRepeat.dw, oldDesc.sbRepeat.dw, "sbRepeat"))
    {
        return false;
    }
    if (!compareAttribute(newDesc.rateLimiter.dw, oldDesc.rateLimiter.dw, "rateLimiter"))
    {
        return false;
    }
    if (!compareAttribute(newDesc.syncObject.ddw, oldDesc.syncObject.ddw, "syncObject"))
    {
        return false;
    }
    if (!compareAttribute(newDesc.axiUserData.dw, oldDesc.axiUserData.dw, "axiUserData"))
    {
        return false;
    }
    if (!compareAttribute(newDesc.perfEvtS.dw, oldDesc.perfEvtS.dw, "perEvtS"))
    {
        return false;
    }
    for (unsigned i = 0; i < Mme::e_mme_local_and_remote; i++)
    {
        if (!compareAttribute(newDesc.perfEvtL[i].dw,
                              oldDesc.perfEvtL[i].dw,
                              "perfEventL[" + std::to_string(i) + "]"))
        {
            return false;
        }
        if (!compareAttribute(newDesc.perfEvtO[i].dw,
                              oldDesc.perfEvtO[i].dw,
                              "perfEventO[" + std::to_string(i) + "]"))
        {
            return false;
        }
    }
    if (!compareAttribute(newDesc.paddingValueS,
                          oldDesc.paddingValueS,
                          "paddingValueS"))
    {
        return false;
    }
    if (!compareAttribute(newDesc.paddingValueL,
                          oldDesc.paddingValueL,
                          "paddingValueL"))
    {
        return false;
    }
    if (!compareAttribute(newDesc.metaData.aguS,
                          oldDesc.metaData.aguS,
                          "metaData.aguS"))
    {
        return false;
    }
    for (unsigned i = 0; i < Mme::e_mme_local_and_remote; i++)
    {
        if (!compareAttribute(newDesc.metaData.aguL[i],
                              oldDesc.metaData.aguL[i],
                              "metaData.aguL" + std::to_string(i) + "]"))
        {
            return false;
        }
        if (!compareAttribute(newDesc.metaData.aguO[i],
                              oldDesc.metaData.aguO[i],
                              "metaData.aguO" + std::to_string(i) + "]"))
        {
            return false;
        }
    }
    if (!compareAttribute(newDesc.pcu.dw,
                          oldDesc.pcu.dw,
                          "pcu"))
    {
        return false;
    }
    if (!compareAttribute(newDesc.sw.dw,
                          oldDesc.sw.dw,
                          "sw"))
    {
        return false;
    }
    return true;
}

bool MmeDescriptorComparator::compareOverLapRoi(const OverlapRoi& newRoi,
                                                const OverlapRoi& oldRoi)
{
    if (!compareAttribute(newRoi.subRois->size(), oldRoi.subRois->size(), "subRois vector size"))
    {
        return false;
    }
    auto newIt = newRoi.subRois->begin();
    auto oldIt = oldRoi.subRois->begin();
    for (unsigned i = 0; i < newRoi.subRois->size(); i++)
    {
        if (!compareAttribute(newIt->relSoIdx, oldIt->relSoIdx, "subRoi[" + std::to_string(i) + "].relSoIdx"))
        {
            return false;
        }
        if (!compareAttribute(newIt->ranges.size(),
                              oldIt->ranges.size(),
                              "subRoi[" + std::to_string(i) + "].ranges vector size"))
        {
            return false;
        }
        auto newRangesIt = newIt->ranges.begin();
        auto oldRangesIt = oldIt->ranges.begin();
        for (unsigned j = 0; j < newIt->ranges.size(); j++)
        {
            if (compareAttribute(newRangesIt->start(),
                                 oldRangesIt->start(),
                                 "subRoi[" + std::to_string(i) + "].ranges[" + std::to_string(j) + "].m_start"))
            {
                return false;
            }
            if (compareAttribute(newRangesIt->end(),
                                 oldRangesIt->end(),
                                 "subRoi[" + std::to_string(i) + "].ranges[" + std::to_string(j) + "].m_end"))
            {
                return false;
            }
            newRangesIt++;
            oldRangesIt++;
        }
        newIt++;
        oldIt++;
    }

    if (!compareAttribute(newRoi.isSram, oldRoi.isSram, "isSram"))
    {
        return false;
    }
    if (!compareAttribute(newRoi.isL0, oldRoi.isL0, "isL0"))
    {
        return false;
    }
    if (!compareAttribute(newRoi.isReduction, oldRoi.isReduction, "isReduction"))
    {
        return false;
    }
    if (!compareAttribute(newRoi.isLocalSignal, oldRoi.isLocalSignal, "isLocalSignal"))
    {
        return false;
    }
    if (!compareAttribute(newRoi.offset, oldRoi.offset, "offset"))
    {
        return false;
    }
    return true;
}

void MmeDescriptorComparator::printError()
{
    atomicColoredPrint(COLOR_RED,
                       "[NEW_MME_STACK] Compare: diff in %s \n", m_where.c_str())
}

template<typename T>
bool MmeDescriptorComparator::compareAttribute(const T& newAttrib, const T& oldAttrib, std::string description)
{
    if (newAttrib != oldAttrib)
    {
        m_where += description;
        return false;
    }
    return true;
}

bool MmeDescriptorComparator::compareTensor(const Mme::MmeTensorDesc& newTensor,
                                            const Mme::MmeTensorDesc& oldTensor,
                                            std::string tensorName)
{
    for (unsigned i = 0; i < Mme::c_mme_max_tensor_dims; i++)
    {
        if (!compareAttribute(newTensor.validElements[i],
                              oldTensor.validElements[i],
                              tensorName + ".validElements[" + std::to_string(i) + "]"))
        {
            return false;
        }
        if (!compareAttribute(newTensor.loopStride[i],
                              oldTensor.loopStride[i],
                              tensorName + ".loopStride[" + std::to_string(i) + "]"))
        {
            return false;
        }
        if (i < Mme::c_mme_max_tensor_dims-1)
        {
            if (!compareAttribute(newTensor.roiSize[i],
                                  oldTensor.roiSize[i],
                                  tensorName + ".roiSize[" + std::to_string(i) + "]"))
            {
                return false;
            }
            if (!compareAttribute(newTensor.spatialStrides[i],
                                  oldTensor.spatialStrides[i],
                                  tensorName + ".spatialStrides[" + std::to_string(i) + "]"))
            {
                return false;
            }
        }
    }
    if (!compareAttribute(newTensor.spatialSizeMinus1,
                          oldTensor.spatialSizeMinus1,
                          tensorName + ".spatialSizeMinus1"))
    {
        return false;
    }
    return true;
}

bool MmeDescriptorComparator::compareAgu(const Mme::MmeAguCoreDesc& newAgu,
                                         const Mme::MmeAguCoreDesc& oldAgu,
                                         std::string aguName)
{
    for (unsigned i = 0; i < Mme::c_mme_max_tensor_dims; i++)
    {
        if (!compareAttribute(newAgu.roiBaseOffset[i],
                              oldAgu.roiBaseOffset[i],
                              aguName + ".roiBaseOffset[" + std::to_string(i) + "]"))
        {
            return false;
        }
        if (i < Mme::c_mme_max_tensor_dims-1)
        {
            if (!compareAttribute(newAgu.startOffset[i],
                                  oldAgu.startOffset[i],
                                  aguName + ".startOffset[" + std::to_string(i) + "]"))
            {
                return false;
            }
        }
    }
    return true;
}


