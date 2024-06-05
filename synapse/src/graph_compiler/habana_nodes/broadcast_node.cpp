#include "broadcast_node.h"

#include "broadcast_node_creator.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "hal_reader/hal_reader.h"
#include "huge_tensor_broadcast_slicer.h"
#include "node_factory.h"

namespace
{
struct BroadcastParams
{
    unsigned dim;
    unsigned size;
    BroadcastParams(const unsigned dim, const unsigned size) : dim(dim), size(size) {}
};
using BroadcastParamsVector = std::vector<BroadcastParams>;
}  // anonymous namespace

LogicalBroadcastNode::LogicalBroadcastNode(const TensorVector& inputs,
                                           const TensorVector& outputs,
                                           const std::string&  name)
: LogicalOpNode(inputs, outputs, name, OUTPUT_TO_INPUT, TYPE_INTERNAL_BROADCAST, SIF_BROADCAST)
{
}

bool LogicalBroadcastNode::validateNode() const
{
    if (!LogicalOpNode::validateNode())
    {
        return false;
    }

    if (!BroadcastNode::validateBroadcast(this))
    {
        return false;
    }

    if (getInput(TENSOR_IFM)->getSizeInElements(0) != getOutput(TENSOR_OFM)->getSizeInElements(0))
    {
        LOG_ERR(HABANA_NODE, "Broadcast logical node {} cannot broadcast FCD", m_name);
        return false;
    }

    return true;
}

NodePtr LogicalBroadcastNode::clone() const
{
    return NodePtr(new LogicalBroadcastNode(*this));
}

void LogicalBroadcastNode::runLogicalOperation() const
{
    // set output tensor as alias to the input tensor

    TensorPtr in  = getInput(TENSOR_IFM);
    TensorPtr out = getOutput(TENSOR_OFM);

    NSizeArray   outputSizes = out->getNSizesInElements();
    NStrideArray outputStrides = {1};

    HB_ASSERT(!in->isStridedOnFCD(), "input to broadcast {} is strided on fcd", in->getName());
    HB_ASSERT(!out->isStridedOnFCD(), "out to broadcast {} is strided on fcd", in->getName());

    for (unsigned i = 1; i < Tensor::c_tensorMaxNDim; ++i)
    {
        if (in->getSizeInElements(i) != out->getSizeInElements(i))
        {
            HB_ASSERT(in->getSizeInElements(i) == 1, "support broadcast only to dimension with size equal to 1");
            outputStrides[i] = 0;
        }
        else
        {
            outputStrides[i] = in->getStrideInBytes(i);
        }
    }
    outputStrides[0] = in->getElementSizeInBytes();
    out->reshape(out->getDim(), outputSizes.data(), outputStrides.data());

    out->setAsAliasSubTensor(in);
}

bool LogicalBroadcastNode::isRedundantNode() const
{
    return isBasicRedundant();
}

bool LogicalBroadcastNode::RunOnCpu()
{
    // todo: Implement LogicalBroadcastNode (SW-2371)
    return false;
}

FcdBroadcastNode::FcdBroadcastNode(const TensorVector& inputs, const TensorVector& outputs, const std::string& name)
: BaseClass(inputs, outputs, name, TYPE_FCD_BROADCAST, SIF_BROADCAST)
{
}

NodePtr FcdBroadcastNode::clone() const
{
    return NodePtr(new FcdBroadcastNode(*this));
}

bool FcdBroadcastNode::validateNode() const
{
    return BroadcastNode::validateBroadcast(this);
}

bool FcdBroadcastNode::RunOnCpu()
{
    NodeList nodes = extract();
    for (auto n : nodes)
    {
        bool ret = n->RunOnCpu();
        if (!ret) return false;
    }

    return true;
}

// create a tensor that will fit the output of Flatten op
TensorPtr FcdBroadcastNode::createFlattenedTensor(const TensorPtr& tensor, unsigned axis)
{
    HB_ASSERT(tensor->getDim() > axis, "unable to flatten {}D tensor by axis {}", tensor->getDim(), axis);
    TensorPtr flattenedTensor = tensor->clone(false, false);
    SizeArray flattenMaxSize  = {1, 1, 1, 1, 1};
    SizeArray flattenMinSize  = {1, 1, 1, 1, 1};
    unsigned  i               = 0;
    for (; i < axis + 1; ++i)
    {
        flattenMaxSize[0] *= tensor->getSizeInElements(i);
        flattenMinSize[0] *= tensor->getMinimalSizeInElements(i);
    }
    for (unsigned j = i; j < tensor->getDim(); ++j)
    {
        flattenMaxSize[1] *= tensor->getSizeInElements(j);
        flattenMinSize[1] *= tensor->getMinimalSizeInElements(j);
    }
    flattenedTensor->reshape(2, flattenMaxSize.data(), nullptr, flattenMinSize.data());
    return flattenedTensor;
}

// For static shapes the FCD broadcast implementation is using a broadcast node and a reshape node.
// The input for the broadcast is reshaped in a way that the FCD dim is set to 1.
// The reshape node is used to reshape the output back to the original FCD output.
// For dynamic shapes the broadcast node requires a shape tensor as a second input.
// The shape tensor is used both as a input for the broadcast node and for the injected reshpae.
// For the broadcast node it needs to be reshaped in the same way that the input is reshaped in the static case.
TensorPtr FcdBroadcastNode::handleShapeTensor(const TensorPtr& shapeTensor, NodeList& nodes) const
{
    TensorPtr expandDimsOut = createExpandedTensor(shapeTensor, 0);

    synExpandDimsParams expandDimsParams;
    expandDimsParams.axis  = 0;
    NodePtr expandDimsNode = NodeFactory::createNode({shapeTensor},
                                                     {expandDimsOut},
                                                     &expandDimsParams,
                                                     sizeof(expandDimsParams),
                                                     NodeFactory::expandDimsShapeNodeTypeName,
                                                     m_name + "_expand_dims");
    nodes.push_back(expandDimsNode);
    return expandDimsOut;
}

// create expanded dimensions for broadcast so we do the broadcast on W dimension (instead of FCD)
// works only if the tensors are up to 4D.
// [1,Y,Z] ->BroadcastFCD-> [X,Y,Z] turns into:
// [1,Y,Z] ->Reshape-> [1,1,Y,Z] ->BroadcastW->  [1,X,Y,Z] ->Reshape-> [X,Y,Z]
void FcdBroadcastNode::extractNodesWithReducedDims(const TensorPtr& input,
                                                   const TensorPtr& output,
                                                   const TensorPtr& shapeTensor,
                                                   NodeList&        nodes) const
{
    TensorPtr    broadcastInput  = createExpandedTensor(input, 0);
    TensorPtr    broadcastOutput = createExpandedTensor(output, 0);
    TensorVector broadcastInputs = {broadcastInput};
    TensorVector shrinkInputs    = {broadcastOutput};

    // handle DSD - create shape tensors for broadcast and reshape
    if (shapeTensor)
    {
        TensorPtr expandDimsOut = handleShapeTensor(shapeTensor, nodes);
        shrinkInputs.push_back(shapeTensor);
        broadcastInputs.push_back(expandDimsOut);
    }

    // create expand, broadcastW and shrink nodes
    synExpandDimsParams expandDimsParams;
    expandDimsParams.axis = 0;
    NodePtr expandNode    = NodeFactory::createNode({input},
                                                 {broadcastInput},
                                                 &expandDimsParams,
                                                 sizeof(expandDimsParams),
                                                 NodeFactory::expandDimsNodeTypeName,
                                                 m_name + "_expand");
    NodePtr broadcastNode = NodeFactory::createNode(broadcastInputs,
                                                    {broadcastOutput},
                                                    nullptr,
                                                    0,
                                                    NodeFactory::broadcastNodeTypeName,
                                                    m_name);
    NodePtr shrinkNode    = NodeFactory::createNode(shrinkInputs,
                                                 {output},
                                                 nullptr,
                                                 0,
                                                 NodeFactory::reshapeNodeTypeName,
                                                 m_name + "_reshape_final");
    // return nodes created
    nodes.push_back(expandNode);
    nodes.push_back(broadcastNode);
    nodes.push_back(shrinkNode);
}

// handles the case of 5D:
// flatten the input and output into 2D. Hence:
// [1, Y, Z, W, V] ->broadcast-> [X, Y, Z, W, V] Will become: [1, Y*Z*W*V] ->broadcast-> [X, Y*Z*W*V]
void FcdBroadcastNode::reduceDims(TensorPtr& input,
                                  TensorPtr& output,
                                  TensorPtr& shapeTensor,
                                  NodeList&  nodes,
                                  unsigned   maxTensorDim) const
{
    synFlattenParams flattenDimsParams;
    flattenDimsParams.axis = 0;
    TensorPtr flatOutput   = createFlattenedTensor(output, flattenDimsParams.axis);

    if (input->getDim() >= maxTensorDim)
    {
        TensorPtr flatInput   = createFlattenedTensor(input, flattenDimsParams.axis);
        NodePtr   flattenNode = NodeFactory::createNode({input},
                                                      {flatInput},
                                                      &flattenDimsParams,
                                                      sizeof(flattenDimsParams),
                                                      NodeFactory::flattenNodeTypeName,
                                                      m_name + "_flatten");
        nodes.push_back(flattenNode);
        input = flatInput;
    }
    else
    {
        // currently no support for: FCD broadcast + 5D + broadcast on other axes
        HB_ASSERT(input->getDenseSizeInElements() == 1, "{}, unspported broadcast", m_name);
    }

    TensorVector reshapeInputs = {flatOutput};
    if (shapeTensor)
    {
        TensorPtr flatShapeTensor = createFlattenedTensor(shapeTensor, flattenDimsParams.axis);
        NodePtr   flattenShape    = NodeFactory::createNode({shapeTensor},
                                                       {flatShapeTensor},
                                                       &flattenDimsParams,
                                                       sizeof(flattenDimsParams),
                                                       NodeFactory::flattenShapeNodeTypeName,
                                                       m_name + "_flatten_dims");
        nodes.push_back(flattenShape);
        reshapeInputs.push_back(shapeTensor);
        shapeTensor = flatShapeTensor;
    }

    NodePtr reshapeNode = NodeFactory::createNode(reshapeInputs,
                                                  {output},
                                                  nullptr,
                                                  0,
                                                  NodeFactory::reshapeNodeTypeName,
                                                  m_name + "_reshape");
    nodes.push_back(reshapeNode);
    output = flatOutput;
}

NodeList FcdBroadcastNode::extract(const HabanaGraph& g)
{
    NodeList  ret;
    TensorPtr input       = getInput(TENSOR_IFM);
    TensorPtr output      = getOutput(TENSOR_OFM);
    TensorPtr shapeTensor = getInput(TENSOR_SHAPE_BROADCAST);

    // handle too many dims case: covert to 2D
    unsigned maxTensorDim = SYN_GAUDI_MAX_TENSOR_DIM;
    if (g.isDynamicShape())
    {
        maxTensorDim = SYN_MAX_TENSOR_DIM;  // only supports up to 5 when dynamic shapes
    }
    if (output->getDim() >= maxTensorDim)
    {
        reduceDims(input, output, shapeTensor, ret, maxTensorDim);  // modifies all parameters
    }
    // use extraction that works for up to Tensor::c_tensorMaxDim - 1
    extractNodesWithReducedDims(input, output, shapeTensor, ret);
    return ret;
}

bool FcdBroadcastNode::validateNodeForGraph(const HabanaGraph&) const
{
    return true;
}

BroadcastNode::BroadcastNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: BaseClass(inputs, outputs, name, TYPE_BROADCAST, SIF_BROADCAST)
{
}

NodePtr BroadcastNode::clone() const
{
    return NodePtr(new BroadcastNode(*this));
}

bool BroadcastNode::validateNode() const
{
    return BroadcastNode::validateBroadcast(this);
}

bool BroadcastNode::RunOnCpu()
{
    NodeList nodes = extract();
    for (auto n : nodes)
    {
        bool ret = n->RunOnCpu();
        if (!ret) return false;
    }

    return true;
}
NodeList BroadcastNode::extract()
{
    if (GCFG_ENABLE_HUGE_TENSOR_SLICING.value() && HugeTensorBroadcastSlicer::doesRequireSlicing(this))
    {
        // [CID: 86456] False positive - Uninitialized scalar variable defects caused by usage of std::optional,
        // https://community.synopsys.com/s/article/FP-Uninitialized-scalar-variable-defects-caused-by-usage-of-std-optional
        auto ret = HugeTensorBroadcastSlicer(this, std::nullopt).slice();
        return {ret.begin(), ret.end()};
    }
    if (GCFG_MAKE_BROADCAST_PHYSICAL.value())
    {
        return BroadcastNodeCreator::createBroadcast(clone(), m_graphTraits->getHalReader()->getCacheLineSizeInBytes());
    }
    else if (m_inputs[0]->getSizeInElements(0) != m_outputs[0]->getSizeInElements(0) &&
             m_inputs[0]->getSizeInElements(0) == 1)
    {
        return {NodePtr(new FcdBroadcastNode(m_inputs, m_outputs, m_name))};
    }
    else
    {
        return {NodePtr(new LogicalBroadcastNode(m_inputs, m_outputs, m_name))};
    }
}

bool BroadcastNode::validateNodeForGraph(const HabanaGraph&) const
{
    return true;
}

bool BroadcastNode::validateBroadcast(const Node* node)
{
    const TensorVector& inputs  = node->getInputs();
    const TensorVector& outputs = node->getOutputs();
    if ((inputs.size() != 1 && inputs.size() != 2) || outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE,
                "{}, Invalid number of operands (expecting 1 or 2 inputs and 1 output)",
                node->getNodeName());
        return false;
    }

    if (inputs.size() == 2 && !inputs.back()->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "{}, Invalid inputs, expecting shape tensor at index 1", node->getNodeName());
        return false;
    }

    const Tensor* input0  = inputs[0].get();
    const Tensor* output0 = outputs[0].get();

    if (inputs.size() == 1)
    {
        for (unsigned i = 0; i < output0->getDim(); ++i)
        {
            if (output0->isDynamicDim(i) != input0->isDynamicDim(i))
            {
                LOG_ERR(HABANA_NODE, "{}, exists broadcasted dim, but shape tensor not provided", node->getNodeName());
                return false;
            }
        }
    }

    if (output0->getDim() < input0->getDim())
    {
        LOG_ERR(HABANA_NODE, "{}, output dim is smaller the input dim", node->getNodeName());
        return false;
    }

    if (output0->getElementType() != input0->getElementType())
    {
        LOG_ERR(HABANA_NODE, "{}, input and output have different data types", node->getNodeName());
        return false;
    }

    for (unsigned i = 0; i < output0->getDim(); ++i)
    {
        unsigned dim_size = input0->getSizeInElements(i);
        if (dim_size != output0->getSizeInElements(i) && dim_size != 1)
        {
            LOG_ERR(HABANA_NODE,
                    "{}, cannot broadcast from {} to {}",
                    node->getNodeName(),
                    input0->getDimSizesStr(),
                    output0->getDimSizesStr());
            return false;
        }
    }

    return true;
}

NodePtr BroadcastNode::createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    return NodePtr(new BroadcastNode(inputs, outputs, name));
}

DMABroadcastNode::DMABroadcastNode(const TensorVector& in, const TensorVector& out, const std::string& name)
: DMANode(in, out, name, DMA_TYPE_INTERNAL, SIF_BROADCAST)
{
    m_outputs[0]->getTensorAnnotation().dataInfo.mustBeDense = true;
    HB_ASSERT(out[0]->isDenseLayout(),
              "Tensor {} can't be sparse layout because it is DMABroadcastNode output",
              out[0]->getName());
}

void DMABroadcastNode::replaceOutput(unsigned index, const TensorPtr& newTensor)
{
    HB_ASSERT(index == 0, "Can't replace not existing output");
    m_outputs[0]->getTensorAnnotation().dataInfo.mustBeDense = false;
    newTensor->getTensorAnnotation().dataInfo.mustBeDense = true;
    Node::replaceOutput(index, newTensor);
}

bool DMABroadcastNode::validateNode() const
{
    if (m_inputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Broadcast operation: Invalid number of operands (expecting 1 input)");
        return false;
    }

    if (m_inputs[0]->getDim() > 2)
    {
        LOG_ERR(HABANA_NODE, "Broadcast operation: Invalid number of operands (input of dim 1 or 2)");
        return false;
    }

    if (isDynamicShape())
    {
        LOG_ERR(HABANA_NODE, "Dma broadcast not support dynamic shapes");
        return false;
    }

    if (m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Broadcast Invalid number of operands (expecting 1 output)");
        return false;
    }

    if (m_inputs[0]->getDim() == 2 && m_inputs[0]->getSizeInElements(0) != m_outputs[0]->getSizeInElements(0))
    {
        LOG_ERR(HABANA_NODE, "Dma Broadcast supported only on the SCD");
        return false;
    }

    return DMANode::validateNode() && BroadcastNode::validateBroadcast(this);
}

NodePtr DMABroadcastNode::clone() const
{
    return NodePtr(new DMABroadcastNode(*this));
}

bool DMABroadcastNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return DMANode::validateNodeForGraph(g);
}

DMA_OP_TYPE DMABroadcastNode::getOpType() const
{
    return DMA_OP_TYPE::DMA_OP_BROADCAST;
}

bool DMABroadcastNode::RunOnCpu()
{
    return true;
}
