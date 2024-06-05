#include "convolution_to_filter_converter.h"
#include "passes.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "graph_editor.h"
#include "graph_traits.h"
#include "data_type_utils.h"

bool ConvToFilterConverter::replaceConvNodeWithFilter2D(NodePtr foundNode, HabanaGraph& g)
{
    std::shared_ptr<ConvolutionNode> convNode = std::dynamic_pointer_cast<ConvolutionNode>(foundNode);
    HB_ASSERT(convNode != nullptr, "Unexpected nullptr Node in replaceConvNodeWithFilter2D.");
    std::string strFilterName;

    strFilterName = NodeFactory::filter2dNodeTypeName;
    //Create the name with the type suffix
    if (!g.getTraits().inferenceGraph())
    {
        strFilterName += "_fwd";
    }

    std::string reluGuid = NodeFactory::reluNodeTypeName;
    synDataType type     = convNode->getNodePrecision();
    if (!g.getTraits().inferenceGraph())
    {
        type = convNode->getInput(TENSOR_WEIGHT)->getElementType();
    }
    if (type != syn_type_na)
    {
        std::string_view dt = getDtypeSuffixFromSynDataType(type);
        strFilterName       = fmt::format("{}_{}", strFilterName, dt);
        reluGuid            = fmt::format("{}_{}", reluGuid, dt);
    }

    //create parameters to new filter using the convolution parameters
    synConvolution3DParamsV2    convolutionParams = convNode->getConvolutionParams();
    ns_SpatialReduction::Params filter2dParams    = getFilter2DParamsFromConvParams(convolutionParams);

    std::string weightLayoutAsStr {convNode->getInputLayouts()[TENSOR_WEIGHT].toString()};

    //Remove the C dimension from layout string. It should be 1 in size- C is checked before.
    weightLayoutAsStr.erase(std::remove(weightLayoutAsStr.begin(), weightLayoutAsStr.end(), 'C'), weightLayoutAsStr.end());
    unsigned weightSize = weightLayoutAsStr.size();
    TSize newSizes[weightSize];
    int newSizeIdx = 0;

    //Here we squeeze the C dimension from the sizes
    for (int convSizeIdx = 0; convSizeIdx < convNode->getInput(TENSOR_WEIGHT)->getDim(); convSizeIdx++)
    {
        HB_ASSERT(newSizeIdx < weightSize, "Error- incorrect weightSize in replaceConvNodeWithFilter2D");
        if (convNode->getInputLayouts()[TENSOR_WEIGHT].getIndexByName('C') != convSizeIdx)
        {
            newSizes[newSizeIdx] = convNode->getInput(TENSOR_WEIGHT)->getSizeInElements(convSizeIdx);
            newSizeIdx++;
        }
    }
    convNode->getInput(TENSOR_WEIGHT)->reshape(weightSize, newSizes, nullptr);

    //In case of relu fused to grouped conv, need to plant relu node as filter_2d does not support fused relu
    TensorPtr convOutput = convNode->getOutput(0);
    NodePtr reluNode;
    if (convolutionParams.activation.reluEnable)
    {
        TensorPtr reluInput = convOutput->clone();
        reluNode            = NodeFactory::createNode({reluInput},
                                           {convOutput},
                                           nullptr,
                                           reluGuid,
                                           fmt::format("{}_relu_node", convNode->getNodeName()));
        convOutput = reluInput;
    }

    // Important note:
    // all convolution options that are not supported in filter 2d should be attended.
    // (This is true mostly for goya2)

    //Create the filter node
    TensorVector filterNodeInputs = {convNode->getInput(TENSOR_IFM),
                                     convNode->getInput(TENSOR_WEIGHT),
                                     convNode->getInput(TENSOR_BIAS)};

    // Create node Properties
    Node::NodeProperties properties;
    gc::Layout weightLayout = gc::Layout(weightLayoutAsStr);
    gc::Layout biasLayout = convNode->getInput(TENSOR_BIAS) ? convNode->getInputLayouts()[TENSOR_BIAS] : gc::Layout();
    properties.inputLayouts = { gc::Layout(convNode->getInputLayouts()[TENSOR_IFM]), weightLayout, biasLayout };
    properties.outputLayouts = convNode->getOutputLayouts();

    NodePtr    filter2dNode = NodeFactory::createNode(filterNodeInputs,
                                                   {convOutput},
                                                   nullptr,
                                                   strFilterName,
                                                   fmt::format("{}_filter2d", convNode->getNodeName()),
                                                   properties);
    TPCNodePtr filterNode = std::dynamic_pointer_cast<TPCNode>(filter2dNode);
    HB_ASSERT(filterNode, "Error- Failed to create TPC Node");

    //Copy the parameters to the buffer inside the node.
    filterNode->storeParamsInBuffer(&filter2dParams, sizeof(filter2dParams));

    // set filter2d inputs as int16 limited same as MME nodes
    for (const TensorPtr& input : filter2dNode->getInputs())
    {
        if (input == nullptr) continue;
        input->setInt16Limited(false);
    }

    NodeList nodesToRemove = { foundNode };
    NodeList nodesToAdd    = { filter2dNode };

    if (convolutionParams.activation.reluEnable)
    {
        nodesToAdd.emplace_back(reluNode);
    }

    return (GraphEditor::replaceNodes(g, nodesToRemove, nodesToAdd) == REPLACE_NODE_SUCCESS);
}

ns_SpatialReduction::Params
ConvToFilterConverter::getFilter2DParamsFromConvParams(const synConvolution3DParamsV2& convolutionParams)
{
    ns_SpatialReduction::Params         filter2dParams;
    filter2dParams.dilation_h           = convolutionParams.dilation[CONV_DIL_HEIGHT];
    filter2dParams.dilation_w           = convolutionParams.dilation[CONV_DIL_WIDTH];
    filter2dParams.kernel_h             = convolutionParams.kernel[CONV_KERNEL_HEIGHT];
    filter2dParams.kernel_w             = convolutionParams.kernel[CONV_KERNEL_WIDTH];
    filter2dParams.pad_h_begin          = convolutionParams.padding[CONV_PAD_TOP];
    filter2dParams.pad_h_end            = convolutionParams.padding[CONV_PAD_BOTTOM];
    filter2dParams.pad_w_begin          = convolutionParams.padding[CONV_PAD_LEFT];
    filter2dParams.pad_w_end            = convolutionParams.padding[CONV_PAD_RIGHT];
    filter2dParams.stride_h             = convolutionParams.stride[CONV_STRIDE_HEIGHT];
    filter2dParams.stride_w             = convolutionParams.stride[CONV_STRIDE_WIDTH];
    filter2dParams.pooling_convention   = POOLING_CONVENTION_VALID;
    return filter2dParams;
}
