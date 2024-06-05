
#include "node_io_manager.h"
#include "logical_op_node.h"
#include "handle_memory_reuse.h"
#include "defs.h"
#include "types.h"
#include <iterator>

constexpr AliasDirection getOppositeDirection(AliasDirection direction)
{
    static_assert(DIRECTION_MAX == 2);
    return AliasDirection(1 - direction);
}

LogicalOpNode::LogicalOpNode(const TensorVector& inputs,
                             const TensorVector& outputs,
                             std::string_view    name,
                             AliasDirection      direction,
                             eNodeType           type,
                             ShapeFuncID         sifId)
: Node(inputs, outputs, name, type, sifId, false)
{
    m_io                      = std::make_unique<LogicalNodeIOManager>(this);
    m_aliasTensorDirection    = direction;
    m_runLogicalOperationDone = false;
    m_isPureLogical           = false;
}

LogicalOpNode::LogicalOpNode(const LogicalOpNode& other)
: Node(other),
  m_aliasTensorDirection(other.m_aliasTensorDirection),
  m_runLogicalOperationDone(other.m_runLogicalOperationDone),
  m_isPureLogical(other.m_isPureLogical)
{
    m_io = std::make_unique<LogicalNodeIOManager>(this);
}

LogicalOpNode& LogicalOpNode::operator=(const LogicalOpNode& other)
{
    if (this != &other)
    {
        Node::operator        =(other);
        m_aliasTensorDirection    = other.m_aliasTensorDirection;
        m_runLogicalOperationDone = other.m_runLogicalOperationDone;
        m_isPureLogical           = other.m_isPureLogical;
        m_io                      = std::make_unique<LogicalNodeIOManager>(this);
    }
    return *this;
}

bool LogicalOpNode::incompatibleWithNextNode(Node::eNodeType type) const
{
    //If no type is given (so the default TYPE_MAX is used) then the node is a graph output
    if (type == TYPE_INTERNAL_CONCAT) return m_type != TYPE_INTERNAL_EXPAND_DIMS;

    if (isAliasStrided())
    {
        //Node messes with strides, so check if the consumer (next node) is convenient.
        //We can improve this by considering the axis of the operation
        switch (type)
        {
        case Node::TYPE_INTERNAL_EXPAND_DIMS:
            return false;
        default:
            //Let's be pessimistic for now - some reshapes may also work if they copy the strides
            return true;
        }
    }

    //Else - the node is a flavor of reshaping, so the output tensor is dense
    return false;
}

bool LogicalOpNode::validateNode() const
{
    if (getInput(TENSOR_IFM) == nullptr || getOutput(TENSOR_OFM) == nullptr)
    {
        // Logical node should have at least one input and one output
        LOG_ERR(HABANA_NODE, "{} node got missing key tensors", getNodeName());
        return false;
    }

    return true;
}


synDataType LogicalOpNode::getRequiredInputType(uint32_t tensorIdx) const
{
    return getRealTensor()->getElementType();
}

synDataType LogicalOpNode::getRequiredOutputType(uint32_t tensorIdx) const
{
    return getRealTensor()->getElementType();
}

bool LogicalOpNode::validateAlias() const
{
    TensorVector allTensors = getOperands();
    for (TensorPtr tensor: allTensors)
    {
        if (tensor->isAliasedTensor())
        {
            if (std::find(allTensors.begin(), allTensors.end(), tensor->getAliasTensor()) != allTensors.end())
            {
                if(getRealTensor() != tensor->getAliasTensor())
                {
                    LOG_CRITICAL(HABANA_NODE, "Tensor has unmatched aliases");
                    return false;
                }
            }
        }
    }
    return true;
}

bool LogicalOpNode::validateNodeForGraph(const HabanaGraph& g) const
{
    // Logical nodes are valid in all graphs
    return true;
}

NStrideArray LogicalOpNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& t = (m_aliasTensorDirection == INPUT_TO_OUTPUT) ? getInput(idx) : getOutput(idx);
    NStrideArray     ret = {1};
    t->getNStridesInBytes(ret.data());
    return ret;
}

void LogicalOpNode::swapAliasDirection()
{
    if (!canSwapAliasDirection())
    {
        LOG_ERR(HABANA_NODE, "Alias direction can't be set for {}", getNodeName());
        HB_ASSERT(false, "Alias direction can't be set for {}", getNodeName());
    }
    m_aliasTensorDirection = getOppositeDirection(m_aliasTensorDirection);
}

TensorPtr LogicalOpNode::getRealTensor() const
{
    if (m_aliasTensorDirection == INPUT_TO_OUTPUT)
    {
        HB_ASSERT(m_outputs.size() - getNumOutputsShapeTensors() == 1, "size mismatch");
        return getOutput(TENSOR_OFM);
    }
    else
    {
        HB_ASSERT((m_inputs.size() - getNumInputsShapeTensors()) == 1, "size mismatch");
        return getInput(TENSOR_IFM);
    }
}

TensorVector LogicalOpNode::getAliasTensors() const
{
    TensorVector        ret;
    const TensorVector& aliases = (m_aliasTensorDirection == INPUT_TO_OUTPUT) ? m_inputs : m_outputs;
    for (const TensorPtr& t : aliases)
    {
        if (t->isShapeTensor()) continue;
        ret.push_back(t);
    }
    return ret;
}

LogicalOpNode::ResolveStatus LogicalOpNode::resolveAliasDirection(IndicesVec& requireInputMemcpy, IndicesVec& requireOutputMemcpy)
{
    bool origDirValid  = aliasDirectionValid(getAliasDirection(), requireInputMemcpy, requireOutputMemcpy);
    IndicesVec tmp;
    bool oppositeValid = canSwapAliasDirection() && aliasDirectionValid(getOppositeDirection(getAliasDirection()), tmp, tmp);

    if (oppositeValid)
    {
        if (origDirValid) return ResolveStatus::AliasDirectionVaried;
        swapAliasDirection();
    }
    else if (!origDirValid)
    {
        HB_ASSERT(!requireInputMemcpy.empty() || !requireOutputMemcpy.empty(), "Logical node {} require memcpy with no indices", getNodeName());
        return ResolveStatus::MemcpyNeeded;
    }

    return ResolveStatus::Success;
}

void LogicalOpNode::setRunLogicalOperationDone()
{
    HB_ASSERT(getRunLogicalOperationDone() == false, "Logical operation {} already done!", getNodeName());
    m_runLogicalOperationDone = true;
}

bool LogicalOpNode::isBasicRedundant() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        return false;
    }

    TensorPtr input  = getInput(0);
    TensorPtr output = getOutput(0);

    unsigned inputDim  = input->getDim();
    unsigned outputDim = output->getDim();

    /* In order for this node to do nothing, first thing the in/out dims should be equal */
    if (inputDim != outputDim)
    {
        return false;
    }

    if (input->getElementType() != output->getElementType())
    {
        return false;
    }

    return (input->compareGeometry(*output));
}

bool LogicalOpNode::aliasDirectionValid(AliasDirection direction, IndicesVec& requireInputMemcpy, IndicesVec& requireOutputMemcpy) const
{
    if (direction == INPUT_TO_OUTPUT)
    {
        return aliasDirectionValid(m_outputs[0], m_inputs, requireOutputMemcpy, requireInputMemcpy);
    }
    else
    {
        return aliasDirectionValid(m_inputs[0], m_outputs, requireInputMemcpy, requireOutputMemcpy);
    }
}

bool LogicalOpNode::aliasDirectionValid(const TensorPtr& realTensor, const TensorVector& aliases, IndicesVec& realMemcpy, IndicesVec& aliasesMemcpy) const
{
    bool canBeReal = realTensor->isTrivialStrided() || canHandleStridedRealTensor();
    if (!canBeReal)
    {
        realMemcpy.push_back(0);
    }
    bool canBeAlias = canAllTensorsBeAlias(realTensor, aliases, aliasesMemcpy);
    return canBeAlias && canBeReal;
}

bool LogicalOpNode::canAllTensorsBeAlias(const TensorPtr&    realTensor,
                                         const TensorVector& aliases,
                                         IndicesVec&         requireMemcpy) const
{
    requireMemcpy.clear();
    for (auto tensorIter = aliases.begin(); tensorIter < aliases.end(); ++tensorIter)
    {
        const auto& t = *tensorIter;
        if (unlikely(!t)) continue;

        size_t tensorIdx = std::distance(aliases.begin(), tensorIter);

        bool isUserManagedTensor = t->isUserManagedDram();
        if (isUserManagedTensor && t->getMemorySectionID() == realTensor->getMemorySectionID() &&
            !isAliasStrided(tensorIdx) && MemoryReuseHandler::isExactOverlap(t, Tensor::getRealTensor(realTensor)))
        {
            // special case where a persistent tensor is an alias to another persistent tensor,
            // and the both have the exact same address, size, and strides.
            // This is frequent use case for slice\strided insert in practice, but in theory can also apply to
            // reshape\identity and other logical nodes.
            isUserManagedTensor = false;
        }

        if (isUserManagedTensor || t->isAliasedTensor() || t->isStaticParam() || t->isRealInLogical() ||
            t->isRealInAliasing() || (std::find(aliases.begin(), tensorIter, t) != tensorIter) ||
            (t->getTensorAnnotation().dataInfo.mustBeDense && isAliasStrided(tensorIdx)))
        {
            requireMemcpy.emplace_back(tensorIdx);
        }
    }
    return requireMemcpy.empty();
}

bool LogicalOpNode::constFoldingForReshape()
{
    TensorPtr pInputTensor  = getInput(TENSOR_IFM);
    TensorPtr pOutputTensor = getOutput(TENSOR_OFM);

    if (pInputTensor == nullptr)
    {
        return false;
    }

    if (pInputTensor->isStaticParam())
    {
        bool copyData = pInputTensor->getShouldFreeBuffer();
        pOutputTensor->setTensorBuffer(pInputTensor->getData(),
                                       pInputTensor->getBufferSizeInBytes(),
                                       pInputTensor->getBufferDataType(),
                                       copyData);
        return true;
    }

    char* pInputMap  = nullptr;
    char* pOutputMap = nullptr;

    // Getting the input and output buffers
    pInputMap  = static_cast<char*>(pInputTensor->map());
    pOutputMap = static_cast<char*>(pOutputTensor->map());

    DataRange<uint64_t> inputRange((uint64_t)pInputMap, (uint64_t)pInputMap + pInputTensor->getTotalSizeInBytes());
    DataRange<uint64_t> outputRange((uint64_t)pOutputMap, (uint64_t)pOutputMap + pOutputTensor->getTotalSizeInBytes());
    if (!inputRange.isOverlap(outputRange))
    {
        memcpy(pOutputMap, pInputMap, pInputTensor->getTotalSizeInBytes());
    }
    else if (pOutputMap != pInputMap)
    {
        memmove(pOutputMap, pInputMap, pInputTensor->getTotalSizeInBytes());
    }
    return true;
}

void LogicalOpNode::runAndSetLogicalOp()
{
    runLogicalOperation();
    setRunLogicalOperationDone();
}

void LogicalOpNode::resetLogicalOp()
{
    if (m_runLogicalOperationDone)
    {
        const TensorVector& aliasTensors = m_aliasTensorDirection == INPUT_TO_OUTPUT? m_inputs : m_outputs;
        for (const TensorPtr& t : aliasTensors)
        {
            t->resetAliasing();
        }
        m_runLogicalOperationDone = false;
    }
}
