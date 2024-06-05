#include "passes.h"
#include "habana_graph.h"
#include "perf_lib_layer_params.h"
#include "node_factory.h"
#include "graph_editor.h"

typedef enum
{
    FEATURE_MAP_TENSOR_INDEX    = 0,
    GRAD_INPUT_TENSOR_INDEX     = 1,
    MEAN_TENSOR_INDEX           = 2,
    LSTD_TENSOR_INDEX           = 3,
    GAMMA_TENSOR_INDEX          = 4
} LayerNormInputsIndices;

typedef enum
{
    GRAD_OUTPUT_TENSOR_INDEX    = 0,
    GRAD_BETA_TENSOR_INDEX      = 1,
    GRAD_GAMMA_TENSOR_INDEX     = 2
} LayerNormOutputsIndices;

static std::string getFullGuidName(const Tensor& mainTensor, const std::string_view stageId)
{
    return fmt::format("layer_norm_stage{}_bwd_{}",
                       stageId,
                       mainTensor.getElementType() == syn_type_bf16 ? "bf16" : "f32");
}

static bool splitLayerNormBwdNode(HabanaGraph& g, TPCNodePtr& node)
{
    const TensorVector& inputs    = node->getInputs();
    const TensorVector& outputs   = node->getOutputs();
    const TensorPtr&    gradInput = inputs[GRAD_INPUT_TENSOR_INDEX];

    TSize width = gradInput->getSizeInElements(1);

    // Chunk size and number of chunks calculation are according to TPC specs
    ns_LayerNormStage1Kernel::Params params;
    if (GCFG_ENABLE_LAYER_NORM_BWD_EXPERIMENTAL_SPLIT.value())
    {
        // From TPC doc:
        // Since chunk Partitioning is done only over dim1, and unrolled by 4 along that axis, most probably that for
        // Width(dim1 size) < numTpcEngines*Unroll (8*4 on Gaudi, 24*4 on Gaudi2), it would be more efficient to use the
        // single Layer norm BWD (layer_norm_bwd), since 2-stage kernel cannot be used at that shape (partitioning is
        // not allowed along Dim2 and 3 as dBeta and dGamma require them, see formulas above).

        // From Aviv:
        // Even if width < numTpc*Unroll, it is not worth it to use the single node version since it has index space
        // size of 1,1,1,1, so it can't be divided to engines or pipelined. Ignoring this recommendation for now.

        constexpr TSize unroll = 4;

        // We want to allow up to 16 logical ROIs / slices with full TPC engines utilization
        const TSize desiredNumChunks = 16 * g.getNumTpcEng();
        double      chunkSize        = static_cast<double>(width) / static_cast<double>(desiredNumChunks);

        // Need to round to 4 for accuracy (and performance)
        TSize chunkSizeInElements = round_to_multiple(static_cast<TSize>(chunkSize), unroll);
        params.chunkSize =
            std::max(chunkSizeInElements, TSize(unroll));  // in case chunkSize < 1.0, chunkSizeInElements may be 0
    }
    else
    {
        // Legacy split method - saved until the experimental method is proved stable.
        params.chunkSize = std::max(floor(width / g.getNumTpcEng()), 4.0);
    }

    // From TPC doc:
    // Dim1 size of the dGamma and dBeta intermediate output tensors should be equal to the maximum number of chunks.
    // Where, maximum number of chunks = ceil(dim1*dim2*dim3/chunkSize)
    uint64_t pixels    = gradInput->getDenseSizeInElements() / gradInput->getSizeInElements(0);
    int      numChunks = ceil(static_cast<float>(pixels) / params.chunkSize);

    LOG_DEBUG(GC, "splitLayerNormBwd node {} is using chunkSize={}, numChunks={}", node->getNodeName(),
             params.chunkSize, numChunks);

    TSize gradIntermediateSizes[Tensor::c_tensorMaxDim];
    gradInput->getAllSizesInElements(gradIntermediateSizes, Tensor::c_tensorMaxDim);
    gradIntermediateSizes[1] = numChunks;
    gradIntermediateSizes[2] = 1;
    gradIntermediateSizes[3] = 1;

    TensorPtr gradGammaIntermediate = std::make_shared<Tensor>(2, gradIntermediateSizes, syn_type_single);
    gradGammaIntermediate->setName(node->getNodeName() + "_grad_gamma_intermedtiate");

    TensorPtr gradBetaIntermediate = std::make_shared<Tensor>(2, gradIntermediateSizes, syn_type_single);
    gradBetaIntermediate->setName(node->getNodeName() + "_grad_beta_intermedtiate");

    // layer_norm_bwd kernel expect gradInput as second input while layer_norm_stage1_bwd expect it as the first input
    TensorVector stage1Inputs = {gradInput,
                                 inputs[FEATURE_MAP_TENSOR_INDEX],
                                 inputs[MEAN_TENSOR_INDEX],
                                 inputs[LSTD_TENSOR_INDEX],
                                 inputs[GAMMA_TENSOR_INDEX]};

    TensorVector stage1Outputs = {outputs[GRAD_OUTPUT_TENSOR_INDEX],
                                  gradGammaIntermediate,
                                  gradBetaIntermediate};

    NodePtr layerNormStage1Node = NodeFactory::createGenericTPCNode(stage1Inputs,
                                                                    stage1Outputs,
                                                                    &params,
                                                                    getFullGuidName(*gradInput, "1"),
                                                                    fmt::format("{}_stage1", node->getNodeName()));

    TensorVector stage2Inputs = {gradGammaIntermediate, gradBetaIntermediate};
    TensorVector stage2Outputs       = {outputs[GRAD_GAMMA_TENSOR_INDEX], outputs[GRAD_BETA_TENSOR_INDEX]};
    NodePtr      layerNormStage2Node = NodeFactory::createGenericTPCNode(stage2Inputs,
                                                                    stage2Outputs,
                                                                    &params,
                                                                    getFullGuidName(*gradInput, "2"),
                                                                    fmt::format("{}_stage2", node->getNodeName()));

    if (GraphEditor::replaceNodes(g, {node}, {layerNormStage1Node, layerNormStage2Node}) != REPLACE_NODE_SUCCESS)
    {
        LOG_ERR(GC, "{}: failed splitting layer norm node {}", HLLOG_FUNC, node->getNodeName());
        return false;
    }

    LOG_DEBUG(GC, "Split LayerNorm node {} into {} and {} nodes", node->getNodeName(),
             layerNormStage1Node->getNodeName(),
             layerNormStage2Node->getNodeName());
    return true;
}

bool splitLayerNormBwd(HabanaGraph& g)
{
    if (GCFG_SKIP_LAYER_NORM_BWD_SPLIT.value())
    {
        LOG_TRACE(GC, "Skipping splitLayerNormBwd pass. (GCFG_SKIP_LAYER_NORM_BWD_SPLIT=true)");
        return true;
    }

    const NodeVector nodes = g.getExeSortedNodes();
    for (NodePtr node : nodes)
    {
        TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);

        if (tpcNode != nullptr && tpcNode->isGuidPrefix("layer_norm_bwd"))
        {
            if (!splitLayerNormBwdNode(g, tpcNode)) return false;
        }
    }

    return true;
}
