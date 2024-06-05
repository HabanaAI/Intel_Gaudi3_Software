#include "compilation_hal_reader.h"
#include "habana_graph.h"
#include "habana_pass.h"
#include "mme_node.h"
#include "dedx_node.h"
#include "tensor.h"
#include "node_factory.h"
#include "transpose_utils.h"
#include <memory>

using DeToDXNodePtr = std::shared_ptr<DeToDxNode>;

NodeVector replaceWithTransposedDedx(DeToDXNodePtr dedxNode)
{
    NodeVector ret;

    TensorPtr weight = dedxNode->getWOperand();
    TensorPtr dedy   = dedxNode->getYOperand();
    TensorPtr dedx   = dedxNode->getXOperand();
    TensorPtr shape  = dedxNode->getShapeOperand();

    TransposePermutationArray permutation = getIdentityPermutation(weight->getDim());
    permutation[0]                        = (TransposePermutationDim)1;
    permutation[1]                        = (TransposePermutationDim)0;

    synTransposeParamsNDims transposeParams = permutationToParams(permutation);
    TensorPtr               wTransposed     = getTensorAfterTranspose(*weight, permutation);

    // transpose weights
    ret.emplace_back(NodeFactory::createInternalNode({weight},
                                                     {wTransposed},
                                                     &transposeParams,
                                                     NodeFactory::transposeNodeTypeName,
                                                     fmt::format("{}_transpose_weight", dedxNode->getNodeName())));

    TensorPtr reversedW = wTransposed;
    if (GCFG_LOWER_DEDX_REVERSE_WEIGHTS.value())
    {
        // reverse weights
        reversedW = wTransposed->cloneGeometry();
        reversedW->setName(fmt::format("{}_reversed", wTransposed->getName()));

        int       axis       = WEIGHT_DIM_S;
        TSize     sizes      = {1};
        TensorPtr axisTensor = std::make_shared<Tensor>(1, &sizes, syn_type_int32);
        axisTensor->setTensorBuffer(&axis, sizeof(int), syn_type_int32, true);
        axisTensor->setMemorySectionID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);

        synAxisParams reverseParams = {.axis = axis};
        std::string   reverseGuid   = "reverse_";
        ret.emplace_back(
            NodeFactory::createInternalNode({wTransposed, axisTensor},
                                            {reversedW},
                                            &reverseParams,
                                            reverseGuid.append(getDtypeSuffixFromSynDataType(dedy->getElementType())),
                                            fmt::format("{}_reverse", dedxNode->getNodeName())));
    }

    // create transposedDedx node
    NodePtr transposedDedx = NodeFactory::createInternalNode(
        dedxNode->isDynamicShape() ? TensorVector {dedy, reversedW, shape} : TensorVector {dedy, reversedW},
        {dedx},
        &dedxNode->getConvolutionParams(),
        NodeFactory::transposedDeDx3DNodeTypeName,
        fmt::format("{}_transposed_dedx", dedxNode->getNodeName()));
    transposedDedx->getNodeAnnotation() = dedxNode->getNodeAnnotation();
    // copy packing meta data
    transposedDedx->getNodeAnnotation().mmeMetaData.packing[PACKING_X] =
        dedxNode->getNodeAnnotation().mmeMetaData.packing[PACKING_X];
    ret.emplace_back(transposedDedx);

    return ret;
}

bool shouldLowerDeDx(DeToDXNodePtr dedxNode)
{
    TensorPtr weight          = dedxNode->getWOperand();
    unsigned  kSizeInBytes    = weight->getSizeInBytes(WEIGHT_DIM_K);
    unsigned  sSizeInElements = weight->getSizeInElements(WEIGHT_DIM_S);
    auto      params          = dedxNode->getConvolutionParams();

    TensorPtr    dedy          = dedxNode->getYOperand();
    bool         strideCond    = dedy->getStrideInElements(1) == weight->getSizeInElements(0);
    bool         isNoDilation  = params.dilation[0] == 1;
    unsigned int packingFactor = dedxNode->getNodeAnnotation().mmeMetaData.packing[PACKING_X];
    HB_ASSERT(packingFactor != 0, "packing factor is 0");

    if (weight->getElementType() == syn_type_fp8_143 || weight->getElementType() == syn_type_fp8_152)
    {
        // f8 reverse kernel not implemented yet
        return false;
    }

    bool isSubProblemsCase = params.stride[0] / packingFactor != 1 || params.stride[1] != 1 || params.stride[2] != 1;

    if (sSizeInElements > 1 && kSizeInBytes < CompilationHalReader::getHalReader()->getCacheLineSizeInBytes() &&
        strideCond && isNoDilation && !isSubProblemsCase)
    {
        return true;
    }

    return false;
}

bool lowerDedx(HabanaGraph& g)
{
    if (!GCFG_ENABLE_LOWER_DEDX.value())
    {
        return true;
    }

    // find dedx mme nodes
    for (auto& node : g.getSortedMMENodes())
    {
        if (node->getNodeType() != Node::TYPE_DEDX) continue;
        DeToDXNodePtr dedxNode = std::dynamic_pointer_cast<DeToDxNode>(node);
        HB_ASSERT_PTR(dedxNode);

        if (!shouldLowerDeDx(dedxNode)) continue;

        NodeVector newNodes = replaceWithTransposedDedx(dedxNode);
        if (GraphEditor::replaceNodes(g, {node}, newNodes) != REPLACE_NODE_SUCCESS)
        {
            LOG_ERR(GC, "{}: Cannot replace nodes for node {}", HLLOG_FUNC, node->getNodeName());
            return false;
        }
    }

    return true;
}