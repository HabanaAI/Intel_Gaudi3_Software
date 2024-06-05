#include "dynamic_split_node.h"

#include "defs.h"
#include "sif/shape_inference_metadata.h"
#include "types_exception.h"

DynamicSplitNode::DynamicSplitNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   UserParams          userParams,
                                   std::string_view    name,
                                   eNodeType           type)
: BaseClass(inputs, outputs, name, OUTPUT_TO_INPUT, type, SIF_DYNAMIC_SPLIT, userParams)
{
    setParams(userParams, sizeof(synSplitParams));
}

NodePtr DynamicSplitNode::createNode(const TensorVector& inputs,
                                     const TensorVector& outputs,
                                     UserParams          userParams,
                                     std::string_view    guid,
                                     std::string_view    name)
{
    return NodePtr(new DynamicSplitNode(inputs, outputs, userParams, name));
}

void DynamicSplitNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "DynamicSplitNode userParams is null");
        throw InvalidNodeParamsException(m_name, "userParams");
    }
    if (userParamsSize != sizeof(synSplitParams) && userParamsSize != 0)
    {
        LOG_ERR(HABANA_NODE, "DynamicSplitNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synSplitParams));
    }
    m_aggDim = reinterpret_cast<synSplitParams*>(userParams)->axis;
    LOG_TRACE(HABANA_NODE, "DynamicSplitNode name - {}, params - dim={}", m_name, m_aggDim);
}

NodePtr DynamicSplitNode::clone() const
{
    return NodePtr(new DynamicSplitNode(*this));
}

bool DynamicSplitNode::RunOnCpu()
{
    // TODO
    return false;
}

SifNodeParams DynamicSplitNode::getShapeInferenceFunctionUserParams()
{
    if (m_sifMetadataBuffer.empty())
    {
        m_sifMetadataBuffer.resize(sizeof(SifDynamicSplitMetadata));

        auto* metadata = reinterpret_cast<SifDynamicSplitMetadata*>(m_sifMetadataBuffer.data());

        metadata->axis = getAggregationDim();
    }

    return reinterpret_cast<SifNodeParams>(m_sifMetadataBuffer.data());
}

size_t DynamicSplitNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifDynamicSplitMetadata);
}

bool DynamicSplitNode::validateNode() const
{
    const unsigned DATA_TENSOR_INDEX  = 0;
    const unsigned SHAPE_TENSOR_INDEX = 1;

    // check inputs
    CHECK_RET_FALSE(m_inputs.size() == 2, "Dynamic split node must have 2 inputs");
    CHECK_RET_FALSE(m_inputs[DATA_TENSOR_INDEX]->getTensorType() == DATA_TENSOR ||
                        m_inputs[0]->getTensorType() == DATA_TENSOR_DYNAMIC,
                    "The first input must be a data tensor");
    CHECK_RET_FALSE(m_inputs[SHAPE_TENSOR_INDEX]->isHostShapeTensor(), "The second tensor must be a host shape tensor");
    CHECK_RET_FALSE(m_inputs[SHAPE_TENSOR_INDEX]->getDim() == 3, "Shape tensor must be [5, N, 2]");
    CHECK_RET_FALSE(m_inputs[SHAPE_TENSOR_INDEX]->getSizeInElements(0) == SYN_MAX_TENSOR_DIM,  // XXX HABANA_DIM_MAX?
                    "Shape tesnor dim 0 must be of size {}",
                    SYN_MAX_TENSOR_DIM);
    CHECK_RET_FALSE(m_inputs[SHAPE_TENSOR_INDEX]->getSizeInElements(2) == 2, "Shape tensor dim 2 must be of size 2");

    // check outputs
    auto nDataOutputs = m_outputs.size();
    auto dims         = m_inputs[DATA_TENSOR_INDEX]->getDim();

    for (unsigned i = 0; i < nDataOutputs; ++i)
    {
        CHECK_RET_FALSE(m_outputs[i]->getTensorType() == DATA_TENSOR_DYNAMIC, "Outputs must be dynamic data tensors");
        CHECK_RET_FALSE(m_outputs[i]->getDim() == dims,
                        "Output tensor dimension is not the same as input tensor dimension");
    }

    return true;
}
