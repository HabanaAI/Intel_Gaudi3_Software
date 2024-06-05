#include "physical_reshape_node.h"
#include "utils.h"
#include "habana_graph.h"
#include "graph_traits.h"
#include "node_factory.h"
#include "data_type_utils.h"

PhysicalReshapeNode::PhysicalReshapeNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: MultiNode(inputs, outputs, name, TYPE_PHYSICAL_RESHAPE)
{
    m_shapeInferenceFunctionID = ShapeFuncID::SIF_RESHAPE;
}

NodePtr PhysicalReshapeNode::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    if (GCFG_USE_DMA_IN_PHYSICAL_RESHAPE.value())
    {
        return NodePtr(new PhysicalReshapeNode(inputs, outputs, name));
    }

    auto        inputType = inputs[0]->getElementType();
    std::string dtype     = fmt::format("reshape_{}", getDtypeSuffixFromSynDataType(inputType));
    return TPCNode::createNode(inputs, outputs, nullptr, 0, dtype, name);
}

bool PhysicalReshapeNode::requiresPhysicalReshapeToHandleDynamicity(const Tensor& in, const Tensor& out)
{
    // We only need physical reshape when we reshape accross the dynamic dim, and there's only one dim.
    // In some cases, even when a dynamic shape is introduced, the reshape preserves the
    // binary layout of the elements, and thus we can solely perform a logical reshape.
    // e.g. we can reshape those logically:
    //      (1-200, 1) to (1-200)
    //      (2, 4, 1-200, 1) to (8, 1-200)
    //      (3, 1-200, 2, 3) to (3, 1-20, 6)
    // On the contrary (and obviously) we can't reshape logically those:
    //    (1-200, 2) to (2, 1-200)
    //    (1-20, 2) to (2-40)
    //    (1-5, 2, 1-5) to (1-5, 2, 1-5)
    //    (1-30, 2, 5-6) to (5-6, 2, 1-30)
    // The minimal test to check whether we can or can't perform a logical reshape
    // is whether there's only 1 dynamic dim and the multiplication of
    // sizes of dimensions lower than the dynamic dim are equal

    if (!in.isDynamicShape() && !out.isDynamicShape())
    {
        return false;
    }
    // can return false in case of dynamic when the reshape practically ignores the dynamic shape.
    int dynamicInDim = -1;
    int dynamicOutDim = -1;
    for (size_t i = 0; i < in.getDim(); i++)
    {
        if (in.isDynamicDim(i))
        {
            if (dynamicInDim == -1)
            {
                dynamicInDim = i;
            }
            else
            {
                return true; // more than 1 dynamic dim
            }
        }
    }
    for (size_t i = 0; i < out.getDim(); i++)
    {
        if (out.isDynamicDim(i))
        {
            if (dynamicOutDim == -1)
            {
                dynamicOutDim = i;
            }
            else
            {
                return true; // more than 1 dynamic dim
            }
        }
    }

    if (in.isZeroSizedDataTensor())
    {
        return false;  // the input tensor is zero-sized, thus the output will be DATA_TENSOR and not dynamic
    }

    HB_ASSERT(dynamicInDim != -1 && dynamicOutDim != -1, "both input and output should be dynamic tensors");

    const auto& inSizes = in.getAllSizesInElements();
    const auto& outSizes = out.getAllSizesInElements();

    return !(multiplyElements(inSizes.begin() + dynamicInDim + 1, inSizes.begin() + in.getDim()) ==
                 multiplyElements(outSizes.begin() + dynamicOutDim + 1, outSizes.begin() + out.getDim()));
}

NodeList PhysicalReshapeNode::extract()
{
    NodeList result;

    TensorPtr serializedTensor = m_inputs[0]->clone(false, false);

    NodePtr serializeNode = NodeFactory::createNode({m_inputs[0]}, {serializedTensor}, nullptr,
                                                    NodeFactory::getSerializeNodeGUID(),
                                                    getNodeName() + "_serialize");

    TensorPtr reshapedTensor = m_outputs[0]->clone(false, false);

    bool enforceLogical = true;
    NodePtr reshapeNode = NodeFactory::createNode({serializedTensor, m_inputs[1]}, {reshapedTensor}, &enforceLogical,
                                                  NodeFactory::reshapeNodeTypeName,
                                                  getNodeName() + "_logical_reshape");

    NodePtr deserializeNode = NodeFactory::createNode({reshapedTensor}, {m_outputs[0]}, nullptr,
                                                  NodeFactory::getDeserializeNodeGUID(),
                                                  getNodeName() + "_deserialize");

    result.push_back(serializeNode);
    result.push_back(reshapeNode);
    result.push_back(deserializeNode);
    return result;
}

NodePtr PhysicalReshapeNode::clone() const
{
    return NodePtr(new PhysicalReshapeNode(*this));
}

bool PhysicalReshapeNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return (g.getTraits().trainingGraph());
}

bool PhysicalReshapeNode::validateNode() const
{
    if (m_inputs.size() != 2 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 2 inputs and 1 output)");
        return false;
    }
    if (!m_inputs.back()->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "Invalid inputs, expecting shape tensor at index 1");
        return false;
    }
    unsigned outTotalElem(getOutput(TENSOR_OFM)->getDenseSizeInElements());
    unsigned inTotalElem(getInput(TENSOR_IFM)->getDenseSizeInElements());

    if (outTotalElem != inTotalElem)
    {
        LOG_ERR(HABANA_NODE, "Output tensor and input tensor of {} doesn't match in elements' count ( {} , {} )",
                getNodeName(), outTotalElem , inTotalElem);
        return false;
    }

    return true;
}
