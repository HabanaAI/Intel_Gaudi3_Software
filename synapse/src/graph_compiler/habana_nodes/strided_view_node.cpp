#include "strided_view_node.h"

#include "defs.h"
#include "h2d_tensors.h"
#include "node_factory.h"
#include "physical_memory_ops_nodes.h"
#include "smf/shape_func_registry.h"
#include "strided_op_node_utils.h"
#include "strided_view_logical_node.h"
#include "synapse_common_types.hpp"
#include "transpose_utils.h"
#include "types_exception.h"
#include "types.h"
#include "utils.h"

StridedViewNode::StridedViewNode(const TensorVector& in,
                                 const TensorVector& out,
                                 UserParams          params,
                                 std::string_view    name)
: MultiNode(in, out, name, TYPE_STRIDED_VIEW, SIF_STRIDED_VIEW)
{
    setParams(params, sizeof(synStridedOpParams));
}

NodePtr StridedViewNode::createNode(const TensorVector& inputs,
                                    const TensorVector& outputs,
                                    UserParams          userParams,
                                    std::string_view    guid,
                                    std::string_view    name)
{
    HB_ASSERT(!outputs.empty(), "no output for node {}", name);
    HB_ASSERT(!inputs.empty(), "no input for node {}", name);
    return NodePtr(new StridedViewNode(inputs, outputs, userParams, name));
}

void StridedViewNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    synStridedOpParams params;
    if (userParams == nullptr)
    {
        HB_ASSERT(m_inputs.size() == MAX_NUM_INPUTS || m_inputs.size() == MAX_NUM_INPUTS - 1,
                  "StridedView missing node params. expected either synStridedOpParams or strides and offset shape "
                  "tensors or h2d strides tensor. node {}",
                  m_name);
        params = StridedOpUtils::createParamsFromShapeTensors(m_inputs[STRIDES_TENSOR], m_inputs.size() == MAX_NUM_INPUTS ? m_inputs[OFFSET_TENSOR] : nullptr);
    }
    else
    {
        if (userParamsSize != sizeof(synStridedOpParams))
        {
            LOG_ERR(HABANA_NODE, "StridedViewNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synStridedOpParams));
        }
        params = *(synStridedOpParams*)userParams;
    }
    for (unsigned dim = m_outputs[0]->getDim(); dim < HABANA_DIM_MAX; dim++)
    {
        params.strides[dim] = 0;
    }
    LOG_TRACE(HABANA_NODE,
              "StridedView name - {}, params - {}",
              m_name,
              StridedOpUtils::stridedOpParamsString(params, m_outputs[0]->getDim()));
    m_params = params;
}

bool StridedViewNode::isDynamicShape() const
{
    const TensorPtr& strides = m_inputs.size() > STRIDES_TENSOR ? m_inputs[STRIDES_TENSOR] : nullptr;

    return Node::isDynamicShape() || (strides && strides->isHost2DeviceTensor() && isDynamicStridedDmaH2DTensorDynamic(strides));
}

bool StridedViewNode::validateViewNode(const Node* node, const synStridedOpParams& params, bool validateAccess)
{
    const TensorVector& inputs  = node->getInputs();
    const TensorVector& outputs = node->getOutputs();
    const std::string&  name    = node->getNodeName();

    if (outputs.size() != 1 || inputs.size() == 0 || inputs.size() > 4) // TODO: [SW-107441]
    {
        LOG_ERR(HABANA_NODE,
                "StridedView Node {}, expects 1 data output, 1 data input and 3 optional shape inputs",
                name);
        return false;
    }

    const TensorPtr& in      = inputs[TENSOR_IFM];
    const TensorPtr& shape   = inputs.size() > SHAPE_TENSOR ? inputs[SHAPE_TENSOR] : nullptr;
    const TensorPtr& strides = inputs.size() > STRIDES_TENSOR ? inputs[STRIDES_TENSOR] : nullptr;
    const TensorPtr& offset  = inputs.size() > OFFSET_TENSOR ? inputs[OFFSET_TENSOR] : nullptr;
    const TensorPtr& out     = outputs[TENSOR_OFM];

    if (in->isDynamicShape() || out->isDynamicShape())
    {
        if (!shape)
        {
            LOG_ERR(HABANA_NODE, "StridedView Node {} which is dynamic, must have a shape tensor", name);
            return false;
        }
        if (!out->compareGeometry(*shape))
        {
            LOG_ERR(HABANA_NODE, "StridedView Node {}, needs a shape tensor with the same shape as it's output", name);
        }
    }

    if (in->getElementType() != out->getElementType())
    {
        LOG_ERR(HABANA_NODE, "StridedView Node {}, expects same data type for input and output", name);
        return false;
    }

    // dynamic strides and offset flavor
    if (strides)
    {
        if (!strides->isHost2DeviceTensor() && !offset)
        {
            LOG_ERR(HABANA_NODE,
                    "StridedView Node {}, with dynamic strides tesnor, requires a offset shape tensor as well",
                    name);
            return false;
        }

        if (!strides->isHost2DeviceTensor() && (!strides->isShapeTensor() || !offset->isShapeTensor())) // TODO: [SW-107441]
        {
            LOG_ERR(HABANA_NODE, "StridedView Node {}, strides tensor and offset tensor must be shape tensors", name);
            return false;
        }
    }

    // verify that the last strided element does not exceed the original tensor size
    if (validateAccess && !node->isDynamicShape() && !StridedOpUtils::verifyStridedAccess(in, out, params))
    {
        LOG_ERR(HABANA_NODE,
                "StridedView Node {}: Invalid params operation {}",
                name,
                StridedOpUtils::stridedOpParamsString(params, out->getDim()));
        return false;
    }

    return true;
}

bool StridedViewNode::validateNode() const
{
    if (!MultiNode::validateNode())
    {
        return false;
    }

    bool valid = validateViewNode(this, m_params);
    if (!valid)
    {
        return false;
    }
    return true;
}

NodePtr StridedViewNode::clone() const
{
    return NodePtr(new StridedViewNode(*this));
}

void StridedViewNode::printParamsRawData() const
{
    Node::printParamsRawData((void*)&m_params, sizeof(m_params));
}

bool StridedViewNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return true;
}

/*
when having irregular stride on FCD we "wrap" the node with expand dims node and squeeze node.
    T -> [VIEW] -> T'       strides = [x,y,..]
    S -----^
turns into:
    T ------------> [VIEW] --> [squeeze] --> T'       strides = [1,x,y,..]
    S --[expandDim]---^
*/
void StridedViewNode::handleFcdStride(NodeList&           nodes,
                                      TensorVector&       inputs,
                                      TensorVector&       outputs,
                                      synStridedOpParams& params,
                                      const std::string&  name)
{
    const TensorPtr& out     = outputs[TENSOR_OFM];
    const TensorPtr& shape   = inputs.size() > SHAPE_TENSOR ? inputs[SHAPE_TENSOR] : nullptr;
    const TensorPtr& strides = inputs.size() > STRIDES_TENSOR ? inputs[STRIDES_TENSOR] : nullptr;

    // create new params strided view node
    params = StridedOpUtils::createExpandedParams(params, out->getDim());

    // create shrink node
    unsigned  dim          = 0;
    TensorPtr expandedView = createExpandedTensor(out, 0);
    NodePtr   shrinkNode =
        NodeFactory::createNode({expandedView}, {out}, &dim, NodeFactory::squeezeNodeTypeName, name + "_squeeze");
    nodes.push_back(shrinkNode);

    // handle shape
    if (shape != nullptr)
    {
        std::tuple<TensorPtr, NodePtr> expanded = expandTensor(shape, 0);
        nodes.push_back(std::get<1>(expanded));
        inputs[SHAPE_TENSOR] = std::get<0>(expanded);  // new shape tensor for strided view node
    }

    // handle dynamic strides tensor
    if (strides != nullptr)
    {
        std::tuple<TensorPtr, NodePtr> expanded = StridedOpUtils::expandH2DTensor(strides, 0);
        nodes.push_back(std::get<1>(expanded));
        inputs[STRIDES_TENSOR] = std::get<0>(expanded);  // new h2d tensor for dynamic strided dma node
    }

    outputs[TENSOR_OFM] = expandedView;  // new output for strided view node
}

/*
when having dynamic shape on T we cannot gurantee to perform strided view correctly (see flatten).
So we add a Serialize node before the view operation.
Deserialize is not needed since the view consumer will deserialize it (it has strides that match the serialized tensor)

    T -> [VIEW] -> T'
    S -----^
turns into:
    T -->[Serialize]--> [VIEW] -> T'
    S ---------------------^
*/
void StridedViewNode::handleDynamicInput(NodeList&           nodes,
                                         TensorVector&       inputs,
                                         TensorVector&       outputs,
                                         synStridedOpParams& params,
                                         const std::string&  name)
{
    TensorPtr serializedTensor = inputs[INPUT_TENSOR]->clone(false, false);
    serializedTensor->setConnectAtomicNodes();

    NodePtr   serializeNode    = NodeFactory::createNode({inputs[INPUT_TENSOR]},
                                                    {serializedTensor},
                                                    nullptr,
                                                    NodeFactory::getSerializeNodeGUID(),
                                                    name + "_serialize");

    inputs[INPUT_TENSOR] = serializedTensor;  // new input for strided view node
    nodes.push_back(serializeNode);
}

/*
if the strides and offset are dynamic, need to handle this case with a physical node

    T_real ---> [VIEW] --> T'

turns into:

    T_real ---> [VIEW] ---> [DMA] ---> T'

dynamic stride and offset tensors will be moved to the DMA node.

*/
void StridedViewNode::handleDynamicStrides(NodeList&           nodes,
                                           TensorVector&       inputs,
                                           TensorVector&       outputs,
                                           synStridedOpParams& params,
                                           const std::string&  name)
{
    TensorPtr newOutput     = outputs[0]->clone(false, false);
    newOutput->setConnectAtomicNodes();

    bool      isSrc         = true;
    TensorPtr stridesTensor = inputs[STRIDES_TENSOR];
    inputs.pop_back();  // remove strides tensor from view input
    stridesTensor->setHostOnly();

    synDynamicStridedDmaH2dTensor* h2dData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(stridesTensor->getHostMaxData());

    HB_ASSERT(h2dData->num_strides == outputs[0]->getDim(), "stridesTensor doesn't match output dim");

    if (!isDynamicStridedDmaH2DTensorDynamic(stridesTensor))
    {
        HB_ASSERT(params.baseOffset == h2dData->offset,
                    "stridedOpParams don't match offset tensor");
        for (unsigned i = 0; i < h2dData->num_strides; i++)
        {
            HB_ASSERT(params.strides[i] == h2dData->strides[i],
                        "stridedOpParams don't match strides tensor");
        }
        return;
    }

    // dynamic strides and offset handling
    NodePtr dmaNode = NodeFactory::createNode({newOutput, stridesTensor},
                                              {outputs[0]},
                                              &isSrc,
                                              NodeFactory::getDynamicStridedMemcpyNodeGUID(),
                                              name + "_dynamic_strided_memcopy");

    // save original size to dynamic dma node for later use (validation during patching)
    dynamic_cast<NodeWithParentInfo*>(dmaNode.get())->setParentInfo(inputs[INPUT_TENSOR]->getTotalSizeInBytes());

    outputs[0] = newOutput;
    nodes.push_back(dmaNode);
}

/*  sequence:  [X,Y,Z] -> strided_view -> [x,y,z]
    turns into:
    [X,Y,Z](64b) -> reinterpret-> [X,Y,Z*2](32b)-> reshape-> [X,Y,Z,2] -> strided_view -> [x,y,z,2]->
                    reshape-> [x,y,z*2](32b)-> reinterpret-> [x,y,z] (64b)
*/
void StridedViewNode::handle64BitInput(NodeList&           nodes,
                                       TensorVector&       inputs,
                                       TensorVector&       outputs,
                                       synStridedOpParams& params,
                                       const std::string&  name)
{
    NodeList retval;
    HB_ASSERT(inputs.size() <= 3, "{}: expected up to 3 inputs for strided view!", __func__);
    HB_ASSERT(outputs.size() == 1, "{}: expected single output for strided view!", __func__);
    // arbitrary choice of 32u since mem move ops are agnostic to signedness
    constexpr synDataType dtype       = syn_type_uint32;
    unsigned originalDim = outputs[0]->getDim();
    // cast to 32b + reshape data tensors
    auto [newInput, reinterpretIn, reshapeIn]    = reinterpret64BitTensor(inputs[0], true, dtype);
    auto [newOutput, reinterpretOut, reshapeOut] = reinterpret64BitTensor(outputs[0], false, dtype);
    nodes.emplace_back(std::move(reinterpretIn));
    nodes.emplace_back(std::move(reinterpretOut));
    nodes.emplace_back(std::move(reshapeIn));
    nodes.emplace_back(std::move(reshapeOut));
    inputs[0]  = newInput;
    outputs[0] = newOutput;

    // handle shape tensor
    if (inputs.size() > 1)
    {
        auto [expandedTensor, expandNode] = expandShapeTensorWithValue(inputs[1], 0, /* fillValue */ 2);
        inputs[1]                         = expandedTensor;
        nodes.emplace_back(std::move(expandNode));
    }

    // transform params
    params = StridedOpUtils::createReinterpretedParams(params, originalDim);
    if (inputs.size() > 2)
    {
        auto [newH2D, reinterpretH2D] = StridedOpUtils::reinterpretH2DTensor(inputs[2], 2 /* factor */);
        inputs[2]                     = newH2D;
        nodes.emplace_back(std::move(reinterpretH2D));
    }
}

NodeList StridedViewNode::extract()
{
    NodeList           ret;
    TensorVector       newInputs  = m_inputs;
    TensorVector       newOutputs = m_outputs;
    synStridedOpParams newParams  = m_params;

    // new api - convert shape tensors to h2d if required (remove when H2D becomes standard)
    if (m_inputs.size() > STRIDES_TENSOR && m_inputs[STRIDES_TENSOR]->isShapeTensor())
    {
        StridedOpUtils::convertShapeToH2D(ret, newInputs, newOutputs, newParams, m_name);
    }

    if (is64BitOperands())
    {
        handle64BitInput(ret, newInputs, newOutputs, newParams, m_name);
    }

    // strides on FCD - expand dims
    if (newParams.strides[0] != 1 ||
        (m_inputs.size() > STRIDES_TENSOR && isDynamicStridedDmaH2DTensorFcdStrided(m_inputs[STRIDES_TENSOR])))
    {
        LOG_WARN(GC, "FCD strides for StridedView node: {}. will result in non-utilized DMA node!", m_name);
        handleFcdStride(ret, newInputs, newOutputs, newParams, m_name);
    }

    if (m_inputs[INPUT_TENSOR]->isDynamicShape())  // input is dynamic - serialize it
    {
        handleDynamicInput(ret, newInputs, newOutputs, newParams, m_name);
    }

    if (m_inputs.size() > STRIDES_TENSOR)
    {
        handleDynamicStrides(ret, newInputs, newOutputs, newParams, m_name);
    }

    NodePtr viewNode =
        NodeFactory::createNode(newInputs, newOutputs, &newParams, NodeFactory::logicalStridedViewTypeName, m_name);
    ret.push_back(viewNode);

    return ret;
}
