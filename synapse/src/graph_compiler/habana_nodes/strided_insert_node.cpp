#include "strided_insert_node.h"

#include "h2d_tensors.h"
#include "node_factory.h"
#include "physical_memory_ops_nodes.h"
#include "smf/shape_func_registry.h"
#include "strided_insert_logical_node.h"
#include "strided_op_node_utils.h"
#include "types_exception.h"
#include "utils.h"

StridedInsertNode::StridedInsertNode(const TensorVector& in,
                                     const TensorVector& out,
                                     UserParams          params,
                                     std::string_view    name)
: MultiNode(in, out, name, TYPE_STRIDED_INSERT, SIF_STRIDED_INSERT)
{
    setParams(params, sizeof(synStridedOpParams));
}

NodePtr StridedInsertNode::createNode(const TensorVector& inputs,
                                      const TensorVector& outputs,
                                      UserParams          userParams,
                                      std::string_view    guid,
                                      std::string_view    name)
{
    HB_ASSERT(!outputs.empty(), "no output for node {}", name);
    HB_ASSERT(inputs.size() >= MIN_NUM_INPUTS, "no insert input for node {}", name);
    return NodePtr(new StridedInsertNode(inputs, outputs, userParams, name));
}

void StridedInsertNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    synStridedOpParams params;
    if (userParams == nullptr)
    {   // TODO: [SW-107441]
        HB_ASSERT(m_inputs.size() == MAX_NUM_INPUTS || m_inputs.size() == MAX_NUM_INPUTS - 1,
                  "StridedInsert missing node params. expected either synStridedOpParams or strides and offset shape "
                  "tensors or h2d strides tensor. node {}",
                  m_name);
        params = StridedOpUtils::createParamsFromShapeTensors(m_inputs[STRIDES_TENSOR], m_inputs.size() == MAX_NUM_INPUTS ? m_inputs[OFFSET_TENSOR] : nullptr);
    }
    else
    {
        if (userParamsSize != sizeof(synStridedOpParams))
        {
            LOG_ERR(HABANA_NODE, "StridedInsertNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synStridedOpParams));
        }
        params = *(synStridedOpParams*)userParams;
    }
    for (unsigned dim = m_inputs[INSERT_TENSOR]->getDim(); dim < HABANA_DIM_MAX; dim++)
    {
        params.strides[dim] = 0;
    }
    LOG_TRACE(HABANA_NODE,
              "StridedInsertNode name - {}, params - {}",
              m_name,
              StridedOpUtils::stridedOpParamsString(params, m_inputs[INSERT_TENSOR]->getDim()));
    m_params = params;
}

bool StridedInsertNode::isDynamicShape() const
{
    const TensorPtr& strides = m_inputs.size() > STRIDES_TENSOR ? m_inputs[STRIDES_TENSOR] : nullptr;

    return Node::isDynamicShape() || (strides && strides->isHost2DeviceTensor() && isDynamicStridedDmaH2DTensorDynamic(strides));
}

bool StridedInsertNode::validateInsertNode(const Node* node, const synStridedOpParams& params, bool validateAccess)
{
    const TensorVector& inputs  = node->getInputs();
    const TensorVector& outputs = node->getOutputs();
    const std::string&  name    = node->getNodeName();

    if (outputs.size() != 1 || inputs.size() < MIN_NUM_INPUTS || inputs.size() > MAX_NUM_INPUTS)
    {
        LOG_ERR(HABANA_NODE,
                "StridedInsert Node {}, expects 1 data output, 1 data input and 3 optional shape inputs",
                name);
        return false;
    }

    const TensorPtr& inOriginal = inputs[ORIGINAL_TENSOR];
    const TensorPtr& inInsert   = inputs[INSERT_TENSOR];
    const TensorPtr& strides    = inputs.size() > STRIDES_TENSOR ? inputs[STRIDES_TENSOR] : nullptr;
    const TensorPtr& offset     = inputs.size() > OFFSET_TENSOR ? inputs[OFFSET_TENSOR] : nullptr;
    const TensorPtr& out        = outputs[TENSOR_OFM];

    if (inInsert->getElementType() != out->getElementType() || inOriginal->getElementType() != out->getElementType())
    {
        LOG_ERR(HABANA_NODE, "StridedInsert Node {}, expects same data type for input and output", name);
        return false;
    }

    if (!inOriginal->compareGeometry(*out))
    {
        LOG_ERR(HABANA_NODE, "StridedInsert Node {}, expects same shape for input{} and output", name, ORIGINAL_TENSOR);
        return false;
    }

    // dynamic strides and offset flavor
    if (strides)
    {
        if (!strides->isHost2DeviceTensor() && !offset)
        {
            LOG_ERR(HABANA_NODE,
                    "StridedInsert Node {}, with dynamic strides tesnor, requires a offset shape tensor as well",
                    name);
            return false;
        }

        if (!strides->isHost2DeviceTensor() && (!strides->isShapeTensor() || !offset->isShapeTensor()))  // TODO: [SW-107441]
        {
            LOG_ERR(HABANA_NODE, "StridedInsert Node {}, strides tensor and offset tensor must be shape tensors", name);
            return false;
        }
    }

    // validate that the last strided element does not exceed the original tensor size
    if (validateAccess && !node->isDynamicShape() && !StridedOpUtils::verifyStridedAccess(out, inInsert, params))
    {
        LOG_ERR(HABANA_NODE,
                "StridedInsert Node {}: Invalid params operation {}",
                name,
                StridedOpUtils::stridedOpParamsString(params, inInsert->getDim()));
        return false;
    }

    return true;
}

bool StridedInsertNode::validateNode() const
{
    if (!MultiNode::validateNode())
    {
        return false;
    }

    if (!validateInsertNode(this, m_params))
    {
        return false;
    }

    return true;
}

NodePtr StridedInsertNode::clone() const
{
    return NodePtr(new StridedInsertNode(*this));
}

void StridedInsertNode::printParamsRawData() const
{
    Node::printParamsRawData((void*)&m_params, sizeof(m_params));
}

bool StridedInsertNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return true;
}

/*
when having irregular stride on FCD we "wrap" the node with expand dims node and squeeze node.
    T_real ---> [INSERT] --> T'       strides = [x,y,..]
    T_insert -----^
turns into:
    T_real ---------------> [INSERT] --> T'       strides = [1,x,y,..]
    T_insert ---[expandDim]-----^
*/
void StridedInsertNode::handleFcdStride(NodeList&           nodes,
                                        TensorVector&       inputs,
                                        TensorVector&       outputs,
                                        synStridedOpParams& params,
                                        const std::string&  name)
{
    const TensorPtr& in      = inputs[INSERT_TENSOR];
    const TensorPtr& strides = inputs.size() > STRIDES_TENSOR ? inputs[STRIDES_TENSOR] : nullptr;

    // create expanded view params
    params = StridedOpUtils::createExpandedParams(params, in->getDim());

    // create expand node
    std::tuple<TensorPtr, NodePtr> expanded = expandTensor(in, 0);
    nodes.push_back(std::get<1>(expanded));

    // handle dynamic strides tensor
    if (strides != nullptr)
    {
        std::tuple<TensorPtr, NodePtr> expanded = StridedOpUtils::expandH2DTensor(strides, 0);
        nodes.push_back(std::get<1>(expanded));
        inputs[STRIDES_TENSOR] = std::get<0>(expanded);  // new h2d tensor for dynamic strided dma node
    }

    inputs[INSERT_TENSOR] = std::get<0>(expanded);
}

/*
when having dynamic shape on T_real (and T') we cannot gurantee to perform strided insert correctly (see flatten).
So we "wrap" the insert operation with Serialize and Deserialize nodes:

    T_real ---> [INSERT] --> T'
    T_insert -----^
turns into:
    T_real --->[Serialize]--> [INSERT] -->[Deserialize]--> T'
    T_insert -------------------^
*/
void StridedInsertNode::handleDynamicInput(NodeList&           nodes,
                                           TensorVector&       inputs,
                                           TensorVector&       outputs,
                                           synStridedOpParams& params,
                                           const std::string&  name)
{
    HB_ASSERT(outputs[0]->isDynamicShape(), "strided insert node output is not dynamic but the input is");

    TensorPtr serializedTensor = inputs[ORIGINAL_TENSOR]->clone(false, false);
    serializedTensor->setConnectAtomicNodes();

    NodePtr serializeNode = NodeFactory::createNode({inputs[ORIGINAL_TENSOR]},
                                                    {serializedTensor},
                                                    nullptr,
                                                    NodeFactory::getSerializeNodeGUID(),
                                                    name + "_serialize");

    TensorPtr originalTensor = outputs[0]->clone(false, false);
    originalTensor->setConnectAtomicNodes();

    NodePtr deserializeNode = NodeFactory::createNode({originalTensor},
                                                      {outputs[0]},
                                                      nullptr,
                                                      NodeFactory::getDeserializeNodeGUID(),
                                                      name + "_deserialize");

    // create new serialized input for strided insert
    inputs[ORIGINAL_TENSOR] = serializedTensor;
    outputs[0]              = originalTensor;

    nodes.push_back(serializeNode);
    nodes.push_back(deserializeNode);
}

/*
if the strides and offset are dynamic, need to handle this case with a physical node
    T_real ---> [INSERT] --> T'
    T_insert -----^
turns into:
    T_real ----------------> [INSERT] ----> T'
    T_insert ----[DMA]----------^

dynamic stride and offset tensors will be moved to the DMA node.
*/
void StridedInsertNode::handleDynamicStrides(NodeList&           nodes,
                                             TensorVector&       inputs,
                                             TensorVector&       outputs,
                                             synStridedOpParams& params,
                                             const std::string&  name)
{
    TensorPtr stridesTensor = inputs[STRIDES_TENSOR];
    inputs.pop_back();  // remove strides tensor from view input
    stridesTensor->setHostOnly();

    synDynamicStridedDmaH2dTensor* dynStridesData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(stridesTensor->getHostMaxData());

    HB_ASSERT(dynStridesData->num_strides == inputs[INSERT_TENSOR]->getDim(), "stridesTensor doesn't match input dim");

    if (!isDynamicStridedDmaH2DTensorDynamic(stridesTensor))
    {
        HB_ASSERT(params.baseOffset == dynStridesData->offset,
                    "stridedOpParams don't match offset tensor");
        for (unsigned i = 0; i < dynStridesData->num_strides; i++)
        {
            HB_ASSERT(params.strides[i] == dynStridesData->strides[i],
                        "stridedOpParams don't match strides tensor");
        }
        return;
    }

    // dynamic strides and offset handling
    TensorPtr newInput = inputs[INSERT_TENSOR]->clone(false, false);
    newInput->setConnectAtomicNodes();

    bool      isSrc    = false;
    NodePtr   dmaNode  = NodeFactory::createNode({inputs[INSERT_TENSOR], stridesTensor},
                                                 {newInput},
                                                 &isSrc,
                                                 NodeFactory::getDynamicStridedMemcpyNodeGUID(),
                                                 name + "_dynamic_strided_memcopy");

    // save original size to dynamic dma node for later use (validation during patching)
     dynamic_cast<NodeWithParentInfo*>(dmaNode.get())->setParentInfo(inputs[ORIGINAL_TENSOR]->getTotalSizeInBytes());

    inputs[INSERT_TENSOR] = newInput;
    nodes.push_back(dmaNode);
}

/*  sequence:  [X,Y,Z] -> strided_insert -> [X,Y,Z]
               [x,y,z]-----^
    turns into:
    [X,Y,Z](64b) -> reinterpret-> [X,Y,Z*2](32b)-> reshape-> [X,Y,Z,2] -> strided_insert -> [X,Y,Z*2]-> ...
    [x,y,z](64b) -> reinterpret-> [x,y,z*2](32b)-> reshape-> [x,y,z,2] -----^

                    ... -> reshape-> [X,Y,Z*2](32b)-> reinterpret-> [X,Y,Z] (64b)
*/
void StridedInsertNode::handle64BitInput(NodeList&           nodes,
                                         TensorVector&       inputs,
                                         TensorVector&       outputs,
                                         synStridedOpParams& params,
                                         const std::string&  name)
{
    NodeList retval;
    // arbitrary choice of 32u since mem move ops are agnostic to signedness
    constexpr synDataType dtype = syn_type_uint32;
    HB_ASSERT(inputs.size() == 2 || inputs.size() == 3, "expected up to 2 inputs for strided insert!");
    HB_ASSERT(outputs.size() == 1, "expected single output for strided insert!");
    unsigned originalDim = outputs[0]->getDim();
    // cast to 32b + reshape data tensors
    for (unsigned i = 0; i <= StridedInsertNode::INSERT_TENSOR; ++i)
    {
        auto [newInput, reinterpretIn, reshapeIn] = reinterpret64BitTensor(inputs[i], true, dtype);
        nodes.emplace_back(std::move(reinterpretIn));
        nodes.emplace_back(std::move(reshapeIn));
        inputs[i] = newInput;
    }
    auto [newOutput, reinterpretOut, reshapeOut] = reinterpret64BitTensor(outputs[0], false, dtype);
    nodes.emplace_back(std::move(reinterpretOut));
    nodes.emplace_back(std::move(reshapeOut));
    outputs[0] = newOutput;

    // transform params
    params = StridedOpUtils::createReinterpretedParams(params, originalDim);
    if (inputs.size() == 3)
    {
        auto [newH2D, reinterpretH2D] = StridedOpUtils::reinterpretH2DTensor(inputs[2], 2 /* factor */);
        inputs[2]                     = newH2D;
        nodes.emplace_back(std::move(reinterpretH2D));
    }
}

NodeList StridedInsertNode::extract()
{
    NodeList           ret;
    TensorVector       newInputs  = m_inputs;
    TensorVector       newOutputs = m_outputs;
    synStridedOpParams newParams  = m_params;

    // new api - convert shape tensors to h2d if required (remove when H2D becomes standard) (TODO: [SW-107441])
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
        LOG_WARN(GC, "FCD strides for StridedInsert node: {}. will result in non-utilized DMA node!", m_name);
        handleFcdStride(ret, newInputs, newOutputs, newParams, m_name);
    }

    if (m_inputs[ORIGINAL_TENSOR]->isDynamicShape())  // input is dynamic - serialize it
    {
        handleDynamicInput(ret, newInputs, newOutputs, newParams, m_name);
    }

    if (m_inputs.size() > MIN_NUM_INPUTS)
    {
        handleDynamicStrides(ret, newInputs, newOutputs, newParams, m_name);
    }

    NodePtr viewNode =
        NodeFactory::createNode(newInputs, newOutputs, &newParams, NodeFactory::logicalStridedInsertTypeName, m_name);
    ret.push_back(viewNode);

    return ret;
}
