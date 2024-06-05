#pragma once

#include "habana_graph.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/gaudi2/mme_descriptor_generator.h"
#include "platform/gaudi2/graph_compiler/gaudi2_types.h"
#include <memory>
namespace gaudi2
{

class MMEBrain
{
public:
    static void generateLayerParamsFromNode(HabanaGraph& g, const NodePtr& node, MmeLayerParams& params);

    static std::shared_ptr<Gaudi2::MmeDescriptorGenerator>
    createDescriptorGenerator(HabanaGraph& g, const NodePtr& node, MmeLayerParams& params);

private:
    static void printDebugInfoAndSetAnnotations(MmeDescriptorGeneratorPtr& descGenerator,
                                                MmeNode*                   mmeNode,
                                                const MmeLayerParams&      params);
    static void chooseRecommendedParamsForEager(MmeLayerParams& layerParams, const MmeCommon::MmeBrainOperationModes& brainOperationModes);
    static void chooseStrategyForEager(HabanaGraph& g, const MmeNode* mmeNode, MmeLayerParams& layerParams);
    static void chooseStrategy(const HabanaGraph& g,
                               const NodePtr      node,
                               TensorPtr          xTensor,
                               TensorPtr          wTensor,
                               TensorPtr          yTensor,
                               bool               setPipeline,
                               MmeLayerParams&    layerParams);
    static void chooseSignalingMode(const MmeNode* mmeNode, MmeLayerParams& params);
    static void
    chooseRoundingMode(const MmeNode* mmeNode, MmeLayerParams& params, synDataType outputElementType);

    static std::string createTensorString(const MmeCommon::MmeTensorView& tensor);
    static void        printMmeParams(const MmeLayerParams& params);
    static void        printMmeRecipeInfo(const std::vector<std::string>& mmeRecipeInfo);
    static void        printMmePerf(const MmeCommon::PerfAttr& perfAttr);
    static void        patchActivations(const MmeNode& mmeNode,  MmeDescriptorGenerator& descGenerator);
};

}  // namespace gaudi2
