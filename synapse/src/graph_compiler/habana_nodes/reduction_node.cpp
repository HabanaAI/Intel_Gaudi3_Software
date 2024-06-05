#include "reduction_node.h"

#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "node_io_manager.h"

#include "access_pattern_generator.h"
#include "tensor_shape.h"
#include "types_exception.h"
#include "utils.h"

ReductionNode::ReductionNode(const TensorVector& in, const TensorVector& out, UserParams params, std::string_view name)
: LogicalOpNode(in, out, name, INPUT_TO_OUTPUT, TYPE_INTERNAL_REDUCTION, SIF_REDUCTION)
{
    setParams(params, sizeof(unsigned));
}

NodePtr ReductionNode::createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    return NodePtr(new ReductionNode(inputs, outputs, userParams, name));
}

void ReductionNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    ReductionOperation reductionOperation;
    if (userParams != nullptr)
    {
        if (userParamsSize != sizeof(ReductionOperation))
        {
            LOG_ERR(HABANA_NODE, "ReductionNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name);
        }
        reductionOperation = *(ReductionOperation*)userParams;
    }
    else
    {
        reductionOperation = REDUCTION_ADD;
    }
    LOG_TRACE(HABANA_NODE, "ReductionNode name - {}, params - reduction Operation={}", m_name, reductionOperation);
    m_reductionOperation = reductionOperation;
}

bool ReductionNode::validateNode() const
{
    if (!LogicalOpNode::validateNode())
    {
        return false;
    }

    // validate reduction operation
    if (m_reductionOperation >= ENUM_MAX_REDUCTION_OPERATIONS)
    {
        LOG_ERR(HABANA_NODE, "Reduction Node {}: Invalid reduction operation", m_reductionOperation);
        return false;
    }

    // validate tensors dimension and sizes
    if (m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Reduction Node {}: Invalid number of operands (expecting 1 output)", m_name);
        return false;
    }
    if (m_inputs.size() == 0)
    {
        LOG_ERR(HABANA_NODE, "Reduction Node {}: Invalid number of operands (expecting at least 1 input)", m_name);
        return false;
    }

    TensorPtr pOutTensor = m_outputs.front();

    //Validate match in geometry in other dimensions
    for (TensorPtr t : m_inputs)
    {
        if (t->getDim() != pOutTensor->getDim())
        {
            LOG_ERR(HABANA_NODE, "Reduction Node {}: Output and Input tensors have a different number of dimensions", m_name);
            return false;
        }

        for (unsigned d = 1; d < t->getDim(); ++d)
        {
            if (t->getSizeInElements(d) != pOutTensor->getSizeInElements(d))
            {
                LOG_ERR(HABANA_NODE, "Reduction Node {}: Output tensor and input tensors have a different size along dimension {}", m_name, d);
                return false;
            }
        }
    }

    return true;
}

NodePtr ReductionNode::clone() const
{
    return NodePtr(new ReductionNode(*this));
}

NStrideArray ReductionNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& real = getAliasDirection() == OUTPUT_TO_INPUT ? m_inputs[0] : m_outputs[0];
    NStrideArray     ret  = {1};
    real->getNStridesInBytes(ret.data());
    return ret;
}

void ReductionNode::runLogicalOperation() const
{
    TensorPtr outTensor = m_outputs.front();
    HB_ASSERT(!outTensor->isStridedOnFCD(), "output to reduction {} is strided on fcd", outTensor->getName());

    bool outputDense = outTensor->isDenseLayout();

    for (TensorPtr t : m_inputs)
    {
        HB_ASSERT(!t->isStridedOnFCD(), "input to reduction {} is strided on fcd", t->getName());
        if (! outputDense)
        {
            StrideArray strides = outTensor->getAllStridesInBytes();
            t->reshape(outTensor->getDim(), outTensor->getAllSizesInElements().data(), strides.data());
        }
        t->setAsAliasSubTensor(outTensor);
    }
}


TensorShape ReductionNode::getInputShape(const TensorShape& outputShape, uint32_t outputIndex, uint32_t inputIdx) const
{
    const TensorPtr& tensor = getInput(inputIdx);
    if (tensor == nullptr)
    {
        LOG_ERR(HABANA_NODE, "Node has no input.");
        throw(NodeHasNoInput(getNodeName()));
    }
    SizeArray size;
    tensor->getAllSizesInElements(size);
    TensorShape inputShape(tensor->getDim(), size);

    return inputShape;
}


bool ReductionNode::validateNodeForGraph(const HabanaGraph& g) const
{
    // REDUCTION_SET does not require HW support
    return ReductionInfo::isReductionSet(m_reductionOperation) || g.getHALReader()->isSRAMReductionSupported();
}

ReductionOperation ReductionNode::getReductionOperation() const
{
    return m_reductionOperation;
}

void ReductionNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_reductionOperation, sizeof(m_reductionOperation));
}

// If reduction node is consuming a memset without shape tensor (internally created), this memset will not update the
// shape of its output. In order for the reduction shape inference to work, the output shape should be updated by
// one of the other producer to the reduction.
bool ReductionNode::linkConsumedMemsetShape(const HabanaGraph& graph) const
{
    TensorVector memsetOutputs;
    NodePtr nonMemsetNode;
    TensorPtr nonMemsetOutput;
    for (const TensorPtr& input : getInputs())
    {
        NodePtr producer = graph.getTensorProducer(input);
        if (producer && producer->isMemset() && producer->getNumInputs() == 0)
        {
            memsetOutputs.push_back(input);
        }
        else if (!nonMemsetNode)
        {
            nonMemsetNode = producer;
            nonMemsetOutput = input;
        }
    }
    CHECK_RET_FALSE(memsetOutputs.empty() || nonMemsetOutput != nullptr,
            "Reduction ({}) node is expected to have at least one input that is not produced by internal memset", getNodeName());

    for (const TensorPtr& memsetOutput : memsetOutputs)
    {
        if (nonMemsetNode)
        {
            nonMemsetNode->getShapeNode()->addPostSifUpdate(nonMemsetOutput, memsetOutput);
        }
        else
        {
            memsetOutput->setMinSize(nonMemsetOutput->getAllMinimalSizesInElements().data());
        }
    }

    return true;
}

gc::access_pattern::NodeAccessPatternPtr ReductionNode::generateNodeAccessPattern() const
{
    return gc::access_pattern::AccessPatternReductionGenerator::generate(this);
}