#include "node.h"

#include "passes.h"
#include "habana_graph.h"
#include "perf_lib_layer_params.h"
#include "node_factory.h"
#include "graph_editor.h"

constexpr std::string_view s_bf16            = "bf16";
constexpr std::string_view s_f32             = "f32";
constexpr std::string_view s_layerNormsufix1 = "_stage1";
constexpr std::string_view s_layerNormsufix2 = "_stage2";

const unsigned NUM_TPC_ENGINES = 4;

/** utility function to extract guid's name **/
static void getFullGuidName(const Tensor& mainTensor, int stageId, std::string& guid)
{
    guid = fmt::format("lpnorm_frobenius_stage{}_fwd_{}",
                       stageId,
                       mainTensor.getElementType() == syn_type_bf16 ? s_bf16 : s_f32);
}

static bool splitfrobeniusNormFwdNode(HabanaGraph& g, const NodePtr& node)
{
    const TensorVector& inputs          = node->getInputs();
    const TensorPtr&    inputFeatureMap = inputs[0];

    TSize inputSizes[Tensor::c_tensorMaxDim] = {};
    inputFeatureMap->getAllSizesInElements(inputSizes, Tensor::c_tensorMaxDim);

    // verify commputation of chunk size
    ns_LpNormFroStage1Kernel::Params stage1Params;
    stage1Params.chunkSize = std::max(inputSizes[1] / g.getNumTpcEng(), static_cast<TSize>(NUM_TPC_ENGINES));
    int numChunks          = div_round_up(inputSizes[1], stage1Params.chunkSize);

    TSize intermediateSizes[Tensor::c_tensorMaxDim] = {};
    inputFeatureMap->getAllSizesInElements(intermediateSizes, Tensor::c_tensorMaxDim);
    intermediateSizes[0] = 1;
    intermediateSizes[1] = numChunks;

    TensorVector stage1Output = {std::make_shared<Tensor>(2, intermediateSizes, syn_type_float)};
    std::string  nodeGuid;
    getFullGuidName(*inputFeatureMap, 1, nodeGuid);

    NodePtr frobeniusNormStage1Node =
        NodeFactory::createGenericTPCNode(inputs /*stage1Inputs*/,
                                          stage1Output,
                                          &stage1Params,
                                          nodeGuid,
                                          fmt::format("{}{}", node->getNodeName(), s_layerNormsufix1));

    const TensorVector& outputs = node->getOutputs();

    getFullGuidName(*inputFeatureMap, 2, nodeGuid);
    NodePtr frobeniusNormStage2Node =
        NodeFactory::createGenericTPCNode(stage1Output,
                                          outputs /*stage2Output*/,
                                          nullptr,
                                          nodeGuid,
                                          fmt::format("{}{}", node->getNodeName(), s_layerNormsufix2));

    return (GraphEditor::replaceNodes(g, {node}, {frobeniusNormStage1Node, frobeniusNormStage2Node}) ==
            REPLACE_NODE_SUCCESS);
}

bool splitFrobeniusLayerNorm(HabanaGraph& g)
{
    auto fn = [](const NodePtr& node){return (node->getNodeType() == Node::TYPE_FROBENIUS_NORM_NODE);};
    const NodeVector nodes = g.getTopoSortedNodesCond(fn);

    for (const pNode& node : nodes)
    {
        if (!splitfrobeniusNormFwdNode(g, node))
        {
            LOG_ERR(GC, "{}: failed splitting frobeniusNormFwd node {}", HLLOG_FUNC, node->getNodeName());
            return false;
        }
    }
    return true;
}