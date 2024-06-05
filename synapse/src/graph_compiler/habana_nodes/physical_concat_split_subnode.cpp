#include "physical_concat_split_subnode.h"

#include <cstddef>

#include "habana_graph.h"

#include "types_exception.h"

// This .cpp file has implementations of class template memner functions.
// This is OK because we instantiate the class template explicitly
// at the end of the file. We only need two instantiations.

template <class BASE>
PhysicalConcatSplitSubnode<BASE>::PhysicalConcatSplitSubnode(const TensorVector& in,
                                                       const TensorVector& out,
                                                       std::string_view    name,
                                                       UserParams          params)
: PhysicalMemoryOpNode<BASE>(in, out, name, SIF_DMA_PHYS_CONCAT_SPLIT)
{
    setParams(params, sizeof(synPhysicalConcatSplitSubnodeParams));
}

NodePtr PhysicalConcatSplitSubnodeDMA::createNode(const TensorVector& inputs,
                                               const TensorVector& outputs,
                                               UserParams          userParams,
                                               std::string_view    guid,
                                               std::string_view    name)

{
    return NodePtr(new PhysicalConcatSplitSubnodeDMA(inputs, outputs, name, userParams));
}

NodePtr PhysicalConcatSplitSubnodeDMA::clone() const
{
    return NodePtr(new PhysicalConcatSplitSubnodeDMA(*this));
}

NodePtr PhysicalConcatSplitSubnodeTPC::createNode(const TensorVector& inputs,
                                               const TensorVector& outputs,
                                               UserParams          userParams,
                                               std::string_view    guid,
                                               std::string_view    name)

{
    return NodePtr(new PhysicalConcatSplitSubnodeTPC(inputs, outputs, name, userParams));
}

NodePtr PhysicalConcatSplitSubnodeTPC::clone() const
{
    return NodePtr(new PhysicalConcatSplitSubnodeTPC(*this));
}

template <class BASE>
void PhysicalConcatSplitSubnode<BASE>::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParamsSize != sizeof(synPhysicalConcatSplitSubnodeParams))
    {
        LOG_ERR(HABANA_NODE, "PhysicalConcatSplitSubnode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(this->m_name, userParamsSize, sizeof(synPhysicalConcatSplitSubnodeParams));
    }
    synPhysicalConcatSplitSubnodeParams params = *reinterpret_cast<synPhysicalConcatSplitSubnodeParams*>(userParams);
    m_concatSplitDim                           = params.concatSplitDim;
    m_nodeNumberInConcatSplit                  = params.nodeNumberInConcatSplit;
    m_isSplit                                  = params.isSplit;
    LOG_TRACE(
        HABANA_NODE,
        "PhysicalConcatSplitSubnode name - {}, params - concatSplitDim={}, nodeNumberInConcatSplit={}, isSplit={}",
        this->getNodeName(),
        this->m_concatSplitDim,
        this->m_nodeNumberInConcatSplit,
        this->m_isSplit);
}

template <class BASE>
bool PhysicalConcatSplitSubnode<BASE>::validateNode() const
{
    if (this->m_inputs.empty())
    {
        LOG_ERR(HABANA_NODE, "PhysicalConcatSplit operation: Invalid number of operands (expecting >=1 inputs, got 0)");
        return false;
    }

    if (m_isSplit && this->m_inputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE,
                "PhysicalConcatSplit operation: Invalid number of operands (expecting 1, got {})",
                this->m_inputs.size());
        return false;
    }

    if (this->m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE,
                "PhysicalConcatSplit operation: Invalid number of operands (expecting 1 output, got {})",
                this->m_outputs.size());
        return false;
    }

    // Do not call validateNode from either MemcpyDMANode or  DMANode
    return Node::validateNode();
}

template <class BASE>
NodePtr PhysicalConcatSplitSubnode<BASE>::clone() const
{
    return NodePtr(new PhysicalConcatSplitSubnode(*this));
}

template <class BASE>
bool PhysicalConcatSplitSubnode<BASE>::validateNodeForGraph(const HabanaGraph& g) const
{
    return BASE::validateNodeForGraph(g);
}

template <class BASE>
bool PhysicalConcatSplitSubnode<BASE>::RunOnCpu()
{
    std::shared_ptr<Tensor> in  = this->getInput(TENSOR_IFM);
    std::shared_ptr<Tensor> out = this->getOutput(TENSOR_OFM);
    HB_ASSERT(in->getTotalSizeInBytes() == out->getTotalSizeInBytes(),
              "DMAMemcpyNode::RunOnCpu: mismatch between input and output size");
    HB_ASSERT(in->getDim() == 4, "DMAMemcpyNode::RunOnCpu: currenly only 4 dims tensors are supported");
    HB_ASSERT(in->getElementSizeInBytes() >= 1,
              "DMAMemcpyNode::RunOnCpu: tensors with less than one byte element size are currently not supported");

    char* pOut = static_cast<char*>(out->map());
    char* pIn  = static_cast<char*>(in->map());

    HB_ASSERT(pOut, "Output tensor for node {} is not mapped", this->getNodeName());
    HB_ASSERT(pIn, "Input tensor for node {} is not mapped", this->getNodeName());

    for (auto b = 0; b < in->getSizeInElements(3); b++)
    {
        for (auto h = 0; h < in->getSizeInElements(2); h++)
        {
            for (auto w = 0; w < in->getSizeInElements(1); w++)
            {
                for (auto c = 0; c < in->getSizeInElements(0); c++)
                {
                    auto inOffsetBytes = static_cast<unsigned long>(c * in->getElementSizeInBytes()) + w * in->getStrideInBytes(1) +
                                         h * in->getStrideInBytes(2) + b * in->getStrideInBytes(3);
                    auto outOffsetBytes = static_cast<unsigned long>(c * out->getElementSizeInBytes()) + w * out->getStrideInBytes(0) +
                                          h * out->getStrideInBytes(2) + b * out->getStrideInBytes(3);
                    std::memcpy(pOut + outOffsetBytes, pIn + inOffsetBytes, in->getElementSizeInBytes());
                }
            }
        }
    }

    return true;
}

template <typename BASE>
std::vector<Node::NodeDynamicShapeProjection>
PhysicalConcatSplitSubnode<BASE>::getDynamicShapeProjectionsTensors() const
{
    std::vector<Node::NodeDynamicShapeProjection> projections;
    // Only look at the first input tensor, the rest are information tensors
    TensorPtr inputTensor = this->getInput(0);
    for (unsigned dim = 0; dim < inputTensor->getDim(); dim++)
    {
        // Always consider m_concatDim even when the main input is a static tensor,
        // because we will need to patch its origin
        if (!inputTensor->isDynamicDim(dim) && dim != m_concatSplitDim) continue;

        Node::NodeDynamicShapeProjection projection;
        projection.isOutput  = false;
        projection.tensorDim = dim;
        projection.indexSpaceDim = dim;
        projection.tensorIdx = 0;
        projections.push_back(projection);
    }
    return projections;
}

template <typename BASE>
void PhysicalConcatSplitSubnode<BASE>::calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const
{
    PhysicalMemoryOpNode<BASE>::calculateLinearRanges(tRoi, n, isInput);
    if (isSrcDynamicStrided() == isInput)
    {
        this->fixLinearRangesToRealParentStart(tRoi);
    }
}


template class PhysicalConcatSplitSubnode<DMAMemcpyNode>;
template class PhysicalConcatSplitSubnode<TPCMemcpyNode>;
