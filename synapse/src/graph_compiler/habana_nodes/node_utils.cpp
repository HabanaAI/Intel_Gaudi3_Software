#include "node_utils.h"
#include "convolution_node.h"
#include "dedw_node.h"
#include "dedx_node.h"
#include "habana_nodes.h"
#include "types.h"

synConvolution3DParamsV2 getConvolutionParams(const Node& node)
{
    const ConvBaseNode* convNode = dynamic_cast<const ConvBaseNode*>(&node);
    return convNode != nullptr ? convNode->getConvolutionParams() : synConvolution3DParamsV2{};
}


float getScaleFromTensor(TensorPtr tensor)
{
    void*       data = tensor->getAddress();
    synDataType type = tensor->getElementType();
    HB_ASSERT(type == syn_type_float || type == syn_type_bf16, "scale tensor should be float32 or bfloat16");
    HB_ASSERT(data != nullptr, "scale tensor is not initialized");
    float scale = type == syn_type_bf16 ? float(*reinterpret_cast<bfloat16*>(data)) :
                                          float(*reinterpret_cast<float*>(data));
    return scale;
}

float getConvertNodeScale(const NodePtr& n)
{
    TensorPtr scaleTensor = n->getInput(CONVERT_INV_SCALE_IDX);
    float     scale       = getScaleFromTensor(scaleTensor);
    return scale;
}

bool isReshapeNode(const NodePtr& node)
{
    Node::eNodeType nodeType = node->getNodeType();
    return (nodeType == Node::TYPE_INTERNAL_RESHAPE || nodeType == Node::TYPE_STATIC_RESHAPE ||
            nodeType == Node::TYPE_PHYSICAL_RESHAPE);
}

bool isLogicalReshape(const NodePtr& node)
{
    return node->isLogicalOperation() && isReshapeNode(node);
}

bool isFp8GemmGuid(NodePtr node)
{
    return node->getGUID().find("fp8_gemm") != std::string_view::npos;
}

bool isFp8ConvGuid(NodePtr node)
{
    return node->getGUID().find("conv2d_fp8") != std::string_view::npos;
}

bool isFp8MmeCguid(NodePtr node)
{
    return isFp8GemmGuid(node) || isFp8ConvGuid(node);
}

bool isConvertFp8Node(NodePtr node)
{
    return isConvertToFp8Node(node) || isConvertFromFp8Node(node);
}

bool isConvertToFp8Node(NodePtr node)
{
    std::string_view guid    = node->getGUID();
    std::size_t      convert = guid.find("convert_to_fp8");
    return (convert != std::string_view::npos);
}

bool isConvertFromFp8Node(NodePtr node)
{
    std::string_view guid    = node->getGUID();
    std::size_t      convert = guid.find("convert_from_fp8");
    return (convert != std::string_view::npos);
}

bool hasScalarInput(const NodePtr& node)
{
    for (const TensorPtr& inputTensor : node->getInputs())
    {
        if (inputTensor->isStaticParam() && inputTensor->getTotalElements() == 1)
        {
            return true;
        }
    }
    return false;
}