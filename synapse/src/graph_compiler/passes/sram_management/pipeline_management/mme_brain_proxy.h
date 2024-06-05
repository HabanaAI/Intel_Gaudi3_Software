#pragma once

#include "include/mme_common/mme_brain.h"
#include "types.h"

// TODO SW-64000 - turn the proxy API to be device agnostic, with gaudi2 impl, and consider caching it.

class MmeBrainProxy
{
public:
    static MmeCommon::MmeLayerParams getRecommendedMmeLayerParams(const NodePtr& node, bool isGeoPreferredShort);
    static MmeCommon::PerfAttr       getRecommendedConfigMmePerf(const NodePtr& node);
    static unsigned                  getRecommendedGeometryWidth(const NodePtr& node);
    static unsigned                  getRecommendedGeometryHeight(const NodePtr& node);
    static unsigned                  getRecommendedGeometryConcurrency(const NodePtr& node);
    static unsigned                  getRecommendedGeometryAxisElements(const NodePtr& node, unsigned inputIdx);
    static unsigned getRecommendedGeometryAxisElementsForNonMasterOperand(const NodePtr& node, unsigned masterInputIdx);
    static MmeCommon::ChipType       getMmeChipType(const synDeviceType deviceType);
};