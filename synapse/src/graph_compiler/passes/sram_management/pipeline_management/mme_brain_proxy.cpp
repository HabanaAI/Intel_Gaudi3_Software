#include "mme_brain_proxy.h"
#include "mme/mme_brain_ifc.h"
#include "compilation_hal_reader.h"

MmeCommon::MmeLayerParams MmeBrainProxy::getRecommendedMmeLayerParams(const NodePtr& node, bool isGeoPreferredShort)
{
    MMENodePtr mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
    return mmeNode->getMmeBrainIfc()->getRecommendedMmeLayerParams(isGeoPreferredShort);
}

MmeCommon::PerfAttr MmeBrainProxy::getRecommendedConfigMmePerf(const NodePtr& node)
{
    MMENodePtr mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
    return mmeNode->getMmeBrainIfc()->getRecommendedConfigMmePerf();
}

unsigned MmeBrainProxy::getRecommendedGeometryWidth(const NodePtr& node)
{
    const MmeCommon::MmeLayerParams mmeParams = getRecommendedMmeLayerParams(node, false);
    const MmeCommon::ChipType chipType = getMmeChipType(CompilationHalReader::getHalReader()->getDeviceType());
    return MmeCommon::MmeBrain::getGeometryWidth(chipType, mmeParams);
}

unsigned MmeBrainProxy::getRecommendedGeometryHeight(const NodePtr& node)
{
    const MmeCommon::MmeLayerParams mmeParams = getRecommendedMmeLayerParams(node, true);
    const MmeCommon::ChipType chipType = getMmeChipType(CompilationHalReader::getHalReader()->getDeviceType());
    return MmeCommon::MmeBrain::getGeometryHeight(chipType, mmeParams);
}

unsigned MmeBrainProxy::getRecommendedGeometryConcurrency(const NodePtr& node)
{
    MMENodePtr mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
    return mmeNode->getMmeBrainIfc()->getRecommendedGeometryConcurrency();
}

unsigned MmeBrainProxy::getRecommendedGeometryAxisElements(const NodePtr& node, unsigned inputIdx)
{
    return (inputIdx == 0) ? getRecommendedGeometryHeight(node) : getRecommendedGeometryWidth(node);
}

unsigned MmeBrainProxy::getRecommendedGeometryAxisElementsForNonMasterOperand(const NodePtr& node,
                                                                              unsigned       masterInputIdx)
{
    return (masterInputIdx == 0) ? getRecommendedGeometryWidth(node) : getRecommendedGeometryHeight(node);
}

MmeCommon::ChipType MmeBrainProxy::getMmeChipType(const synDeviceType deviceType)
{
    return MmeBrainIfc::getMmeChipType(deviceType);
}
