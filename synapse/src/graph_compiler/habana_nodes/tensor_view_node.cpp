#include "utils.h"

#include "tensor_view_node.h"

#include "sif/shape_inference_metadata.h"

TensorViewNode::TensorViewNode(const TensorPtr& realTensor,
                               bool             accessInput,
                               std::string_view name,
                               Node::eNodeType  type)
: LogicalOpNode(accessInput ? TensorVector({realTensor}) : TensorVector(),
                accessInput ? TensorVector() : TensorVector({realTensor}),
                name,
                accessInput ? OUTPUT_TO_INPUT : INPUT_TO_OUTPUT,
                type,
                SIF_TENSOR_VIEW),
  m_accessInput(accessInput)
{
}

NodePtr TensorViewNode::createNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   UserParams          userParams,
                                   std::string_view    guid,
                                   std::string_view    name)
{
    HB_ASSERT_PTR(userParams);
    auto                params = reinterpret_cast<const TensorViewParams*>(userParams);
    const TensorVector& real   = params->accessInput ? inputs : outputs;
    const TensorVector& views  = params->accessInput ? outputs : inputs;

    HB_ASSERT(real.size() == 1, "TensorView node {}: expected a single real tensor", name);
    HB_ASSERT(views.size() == params->dimsOffsets.size(),
              "TensorView node {}: number of views {} different than params {}",
              name,
              views.size(),
              params->dimsOffsets.size());

    auto tensorViewNode = std::shared_ptr<TensorViewNode>(new TensorViewNode(real[0], params->accessInput, name));
    for (unsigned i = 0; i < views.size(); i++)
    {
        tensorViewNode->addView(views[i], params->dimsOffsets[i]);
    }

    return tensorViewNode;
}

NodePtr TensorViewNode::clone() const
{
    return NodePtr(new TensorViewNode(*this));
}

NStrideArray TensorViewNode::calculateAliasStrides(unsigned idx) const
{
    if (m_keepStrides)
    {
        return BaseClass::calculateAliasStrides(idx);
    }
    else
    {
        const TensorPtr& real = getAliasDirection() == OUTPUT_TO_INPUT ? m_inputs[0] : m_outputs[0];
        NStrideArray     ret  = {1};
        real->getNStridesInBytes(ret.data());
        return ret;
    }
}

void TensorViewNode::runLogicalOperation() const
{
    for (int accessIdx = 0; accessIdx < m_dimsOffsets.size(); ++accessIdx)
    {
        const TensorPtr& realTensor = m_accessInput ? m_inputs.front() : m_outputs.front();
        const TensorPtr& view       = m_accessInput ? getOutput(accessIdx) : getInput(accessIdx);

        uint64_t offset = getBaseByteOffsetOfView(accessIdx);
        if (!m_keepStrides)
        {
            uint64_t strides[Tensor::c_numOfNStrides] = {0};
            realTensor->getNStridesInBytes(strides);
            view->setAsSliceSubTensor(realTensor, offset, strides);
        }
        else  // WA for user-defined strides.
        {
            view->setAsAliasSubTensor(realTensor, offset);
        }
    }
}

void TensorViewNode::addView(const TensorPtr& viewTensor, const SizeVector& dimsOffsets)
{
    const TensorPtr& real  = m_accessInput ? m_inputs[0] : m_outputs[0];
    auto&            views = m_accessInput ? m_outputs : m_inputs;

    HB_ASSERT(viewTensor->getDim() == real->getDim(),
              "view has a different dim {} than real tensor! {}",
              viewTensor->getDim(),
              real->getDim());
    HB_ASSERT(dimsOffsets.size() == real->getDim(),
              "dimsOffsets size {} doesn't match viewed tensor! {}",
              dimsOffsets.size(),
              real->getDim());

    views.push_back(viewTensor);
    m_dimsOffsets.push_back(dimsOffsets);
}

bool TensorViewNode::RunOnCpu()
{
    int elemSizeInBytes = getOutput(TENSOR_OFM)->getElementSizeInBytes();
    for (int tIndex = 0; tIndex < m_dimsOffsets.size(); ++tIndex)
    {
        if (!m_accessInput)
        {
            elemSizeInBytes = getInput(tIndex)->getElementSizeInBytes();
        }
        uint64_t     byteBaseOffset = getBaseByteOffsetOfView(tIndex);
        TensorPtr    aliasedTensor = m_accessInput ? getOutput(tIndex) : getInput(tIndex);
        TensorPtr    realTensor     = m_accessInput ? getInput(0) : getOutput(0);

        TSize sizes[Tensor::c_tensorMaxDim];
        aliasedTensor->getAllSizesInElements(sizes, Tensor::c_tensorMaxDim);
        uint64_t totalElements = aliasedTensor->getDenseSizeInElements();

        char* inputData = (char*)getInput(m_accessInput ? 0 : tIndex)->map();
        char* outputData = (char*)getOutput(m_accessInput ? tIndex : 0)->map();

        for (uint64_t sizePos = 0; sizePos < totalElements; ++sizePos)
        {
            TOffset index[Tensor::c_tensorMaxDim];
            findIndex(sizes, aliasedTensor->getDim(), sizePos, index);
            uint64_t byteOffset = byteBaseOffset;
            for (uint32_t dim = 0; dim < aliasedTensor->getDim(); ++dim)
            {
                byteOffset += index[dim] * aliasedTensor->getStrideInBytes(dim);
            }
            if (m_accessInput)
            {
                memcpy(outputData + (sizePos * elemSizeInBytes), inputData + byteOffset, elemSizeInBytes);
            }
            else
            {
                memcpy(outputData + byteOffset, inputData + (sizePos * elemSizeInBytes), elemSizeInBytes);
            }
        }
    }
    return true;
}

bool TensorViewNode::isAliasStrided(unsigned idx) const
{
    const TensorPtr& realTensor = m_accessInput ? m_inputs.front() : m_outputs.front();
    if (!realTensor->isDenseLayout()) return true;
    // non-trivial strides for real tensor will be copied to alias
    const TensorPtr& aliasTensor = m_accessInput ? getOutput(idx) : getInput(idx);
    HB_ASSERT_PTR(aliasTensor);
    TStride expectedStride = aliasTensor->getElementSizeInBytes();
    for (unsigned dim = 0; dim < realTensor->getDim(); dim++)
    {
        if (realTensor->getStrideInBytes(dim) != expectedStride) return true;
        // stride that will be copied from real tensor to alias, doesn't match the expected trivial stride of alias
        expectedStride *= aliasTensor->getSizeInElements(dim);
    }
    return false;
}

bool TensorViewNode::isAliasStrided() const
{
    const auto& aliasTensors = m_accessInput ? m_outputs : m_inputs;
    for (unsigned i = 0; i < aliasTensors.size(); i++)
    {
        if (isAliasStrided(i)) return true;
    }
    return false;
}

uint64_t TensorViewNode::getBaseByteOffsetOfView(unsigned idx) const
{
    const TensorPtr& realTensor = m_accessInput ? m_inputs.front() : m_outputs.front();
    uint64_t         offset     = 0;
    for (unsigned dim = 0; dim < realTensor->getDim(); dim++)
    {
        offset += realTensor->getStrideInBytes(dim) * m_dimsOffsets[idx][dim];
    }
    return offset;
}

SifNodeParams TensorViewNode::getShapeInferenceFunctionUserParams()
{
    uint32_t bufferSize = getShapeInferenceFunctionUserParamsSize();
    m_sifMetadataBuffer.resize(bufferSize);

    auto                metadata = reinterpret_cast<SifTensorViewMetadata*>(m_sifMetadataBuffer.data());
    const TensorVector& views    = m_accessInput ? m_outputs : m_inputs;
    metadata->header = {m_accessInput, views.size()};

    for(size_t i = 0; i < views.size(); ++i)
    {
        metadata->data[i].dims = views[i]->getDim();
        std::copy(m_dimsOffsets[i].begin(),
                  m_dimsOffsets[i].begin() + ARRAY_SIZE(metadata->data[i].offsets),
                  metadata->data[i].offsets);
        views[i]->getAllSizesInElements(metadata->data[i].sizes, SYN_MAX_TENSOR_DIM);
    }

    return reinterpret_cast<SifNodeParams>(metadata);
}

size_t TensorViewNode::getShapeInferenceFunctionUserParamsSize() const
{
    uint32_t viewsNr = m_accessInput ? m_outputs.size() : m_inputs.size();
    uint32_t headerSize = sizeof(SifTensorViewHeader);
    uint32_t dataSize   = viewsNr * sizeof(SifTensorViewData);
    return headerSize + dataSize;
}

void TensorViewNode::setKeepStrides()
{
    // this is a hack for using user-defined strides on the tensor view tensors.
    // since shape inference doesn't know the user strides, it will not work in this case.
    m_keepStrides = true;
    HB_ASSERT(!isDynamicShape(), "cannot override logical operation strides in dynamic shape mode!");
}