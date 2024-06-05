#pragma once

#include "habana_graph.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/gaudi3/mme_descriptor_generator.h"
#include "platform/gaudi3/graph_compiler/gaudi3_types.h"

namespace gaudi3
{
// builder class for mme descriptors.
// this class will translate the mme node to mmeLayerParams, taking care of undecided attributes (geometry, walking
// pattern etc.) and call the mme-stack to generate descriptors. input: node output: mmeDescriptorGenerator object -
// which is the container for the descriptors throughout the graph
class MmeDescriptorBuilder
{
public:
    MmeDescriptorBuilder(const HabanaGraph& graph);
    MmeDescriptorGeneratorPtr createParamsAndActivations(const NodePtr& node, MmeLayerParams& params);
    void generateLayerParamsFromNode(const MMENodePtr& mmeNode, MmeLayerParams& params);
    void                      generateLayerParamsFromRoi(const MMENodePtr& mmeNode,
                                                         MmeLayerParams&   params,
                                                         DcoreROI&         dcoreRoi,
                                                         unsigned          dcoreIdx);

private:
    void generateLayerParamsCommon(const MMENodePtr& mmeNode, MmeLayerParams& params);
    void setTensors(const MMENodePtr& mmeNode, MmeCommon::MmeLayerParams& params);
    void setTensorsFromRoi(const MMENodePtr&          mmeNode,
                           const DcoreROI&            dcoreRoi,
                           MmeCommon::MmeLayerParams& params,
                           TensorPtr                  outTensor);
    void setSpSizeAndBase(MmeCommon::MmeLayerParams& params);
    void setCacheDirectives(const MMENodePtr& mmeNode, MmeLayerParams& params, bool useDefault = false);
    void setConvParams(const MMENodePtr& mmeNode, MmeCommon::MmeLayerParams& params);
    void setControls(const MMENodePtr& mmeNode, MmeCommon::MmeLayerParams& params);
    void setStrategy(const MMENodePtr& mmeNode, MmeCommon::MmeLayerParams& params);
    void chooseStrategy(const MMENodePtr& node, const TensorPtr& xTensor, const TensorPtr& wTensor, MmeLayerParams& layerParams);
    void chooseSignalingMode(MmeLayerParams& params);
    void chooseRoundingMode(const MMENodePtr& mmeNode, MmeLayerParams& params);
    void getDcoreOffsetBasedOnSmallTensor(DcoreRoisVec& dcoreROIs) const;

    void printDebugInfoAndSetAnnotations(MmeDescriptorGeneratorPtr& descGenerator,
                                         const MMENodePtr&          mmeNode,
                                         const MmeLayerParams&      params,
                                         std::optional<bool>        cdPerforation = std::nullopt);

    void chooseRecommendedParamsForEager(MmeLayerParams& layerParams, const MmeCommon::MmeBrainOperationModes& brainOperationModes);
    void chooseGeometryForEager(const MmeNode* mmeNode, MmeLayerParams& layerParams);
    void chooseStrategyForEager(const MmeNode* mmeNode, MmeLayerParams& layerParams);

    bool isCDPerforated(const MMENodePtr& mmeNode, const MmeLayerParams& params);
    void setTensorsForReduction(const MMENodePtr& mmeNode, MmeLayerParams& params);
    void generateLayerParamsForReduction(const MMENodePtr& mmeNode, MmeLayerParams& params, unsigned reductionLevel);

    const HabanaGraph& m_graph;
    bool                m_isEagerMode = false;
    MmeCommon::ChipType m_chipType    = MmeCommon::e_mme_Gaudi3;
};

}  // namespace gaudi3
