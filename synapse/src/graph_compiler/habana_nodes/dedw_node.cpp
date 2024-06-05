#include "dedw_node.h"

#include "graph_traits.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "synapse_types_operators.h"
#include "tensor_shape.h"
#include "utils.h"

DeToDwNode::DeToDwNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: BaseClass(inputs, outputs, name, Node::TYPE_DEDW, SIF_CONV_DEDW)
{
}

bool DeToDwNode::is3DConvolutionGuid() const
{
    return BaseClass::getGUID() == NodeFactory::deDw3DNodeTypeName;
}

NodePtr DeToDwNode::createNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          userParams,
                               unsigned            userParamsSize,
                               std::string_view    guid,
                               std::string_view    name)
{
    DeToDwNode* DeToDwNode_node = new DeToDwNode(inputs, outputs, name);
    DeToDwNode_node->BaseClass::setGUID(guid);
    DeToDwNode_node->setParams(userParams, userParamsSize);
    return NodePtr(DeToDwNode_node);
}

NodePtr DeToDwNode::clone() const
{
    return NodePtr(new DeToDwNode(*this));
}

TensorSemanticType DeToDwNode::getParamSemanticType(const TensorPtr& param) const
{
   return TYPE_ACTIVATION;
}

TensorShape DeToDwNode::getInputShape(const TensorShape& output, uint32_t outputIdx, uint32_t inputIdx) const
{
    HB_ASSERT(outputIdx == TENSOR_DEDW, "output index mismatch, real:{}, expected:{}", outputIdx, TENSOR_OFM);
    // TODO
    TensorShape inputShape = Node::getInputShape(output, outputIdx, inputIdx);
    return inputShape;
}

bool DeToDwNode::validateNode() const
{
    if (m_inputs.size() != 2 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 2 inputs and 1 output)");
        return false;
    }
    return BaseClass::validateNode();
}

bool DeToDwNode::RunOnCpu()
{
    HB_ASSERT(false, "currently not implemented");
    return false;
}

bool DeToDwNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return g.getTraits().trainingGraph() && BaseClass::validateNodeForGraph(g);
}

bool DeToDwNode::isOperandTransposed(const TensorPtr& t) const
{
    return t == getXOperand();
}

TensorPtr DeToDwNode::getXOperand() const
{
    return getInput(TENSOR_X_BWD);
}

TensorPtr DeToDwNode::getYOperand() const
{
    return getInput(TENSOR_DEDY);
}

TensorPtr DeToDwNode::getWOperand() const
{
    return getOutput(TENSOR_DEDW);
}

bool DeToDwNode::isSpatialSlicingSupported(unsigned dim) const
{
    bool dedwSupported = GCFG_SRAM_SLICER_4D_DEDW_SPATIAL_SLICE_ENABLED.value();

    return (ConvBaseNode::isSpatialSlicingSupported(dim) && dedwSupported);
}

TSize DeToDwNode::getMinSpatialDimOutputROI(unsigned dim) const
{
    HB_ASSERT(false, "DEDW has no spatial dim in output tensor");
    return 0;
}