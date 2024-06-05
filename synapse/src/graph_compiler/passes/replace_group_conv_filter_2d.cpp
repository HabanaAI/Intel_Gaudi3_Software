#include "habana_graph.h"
#include "graph_traits.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "graph_editor.h"
#include "convolution_to_filter_converter.h"

bool validateReplaceGroupConvConditions(const NodePtr& foundNode)
{
    std::shared_ptr<ConvolutionNode> convNode = std::dynamic_pointer_cast<ConvolutionNode>(foundNode);

    if (convNode == nullptr)
    {
        return false;
    }
    if (convNode->is3DConvolution())
    {   //TODO - Add support for 3D convolution [SW-23583]
        return false;
    }
    if (convNode->getInput(TENSOR_WEIGHT)->getDim() != MAX_DIMENSIONS_NUM - 1)
    {
        return false;
    }
    // Actual C dimension in weights tensor is divided by nGroups for group convolutions. (C/nGroups)
    // in the case of C == nGroups, C dimension will be equal to 1. otherwise, can't be converted to filter2d.
    if (convNode->inputDimNameToSize(TENSOR_WEIGHT, 'C') != 1)
    {
        return false;
    }
    //If we have a C_IN this optimization will not work. filter doesn't have a CIN.
    if (convNode->getInput(TENSOR_CIN) != nullptr)
    {
        return false;
    }
    if (convNode->getConvolutionParams().nGroups <= 1)
    {
        return false;
    }

    //The main condition for the feature- C = k = nGroups
    if (convNode->inputDimNameToSize(TENSOR_IFM, 'C') == convNode->inputDimNameToSize(TENSOR_WEIGHT, 'K') &&
        convNode->inputDimNameToSize(TENSOR_WEIGHT, 'K') == convNode->getConvolutionParams().nGroups)
    {
        return true;
    }
    else
    {
        // --> main condition was not met! (i.e. C,k,nGroups aren't equal)
        return false;
    }
}

bool replaceGroupConvFilter2d(HabanaGraph& g)
{
    // TODO: this pass is currently enabled only for inference.
    // once SW-23221 is fixed it should be enabled also for training
    if (!GCFG_GAUDI_ENABLE_GROUP_CONV_TO_FILTER_2D.value()
        && !g.getTraits().inferenceGraph())
    {
        return true;
    }
    NodeVector nodes = g.getExeSortedNodes();
    //Replace convolution when K=C=N with filter 2d
    for (const NodePtr& foundNode : nodes)
    {
        if (!validateReplaceGroupConvConditions(foundNode))
        {
            continue;
        }
        if (!ConvToFilterConverter::replaceConvNodeWithFilter2D(foundNode, g))
        {
            LOG_ERR(GC, "{}: failed to replace conv node {} with filter2d", HLLOG_FUNC, foundNode->getNodeName());
            return false;
        }
    }
    return true;
}
