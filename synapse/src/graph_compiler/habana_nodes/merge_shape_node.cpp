#include "merge_shape_node.h"

#include "types_exception.h"

#include <string_view>

MergeShapesNode::MergeShapesNode(const TensorVector& inputs,
                                 const TensorVector& outputs,
                                 UserParams          userParams,
                                 std::string_view    name)
: BaseClass(inputs, outputs, name, AliasDirection::INPUT_TO_OUTPUT, TYPE_MERGE_SHAPES, SIF_MERGE_SHAPES)
{
    setParams(userParams, sizeof(SifMergeShapesMetadata));
}

void MergeShapesNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "MergeShapesNode userParams is null");
        throw InvalidNodeParamsException(m_name, "userParams");
    }
    if (userParamsSize != sizeof(SifMergeShapesMetadata))
    {
        LOG_ERR(HABANA_NODE, "MergeShapesNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(SifMergeShapesMetadata));
    }
    SifMergeShapesMetadata params = *(SifMergeShapesMetadata*)userParams;
    LOG_TRACE(HABANA_NODE, "MergeShapesNode name - {}, params: {}", m_name, paramsToString(params));
    m_params = params;
}

bool MergeShapesNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() >= 1, "MergeShapesNode Expects at least 1 input");
    CHECK_RET_FALSE(m_outputs.size() == 1 && m_outputs.front()->isShapeTensor(),
                    "MergeShapesNode Expects 1 output that is a shape tensor");

    for (const TensorPtr& in : m_inputs)
    {
        CHECK_RET_FALSE(in->isShapeTensor(), "MergeShapesNode expects all input to be shape tensors");
    }

    CHECK_RET_FALSE(getOutput(0)->getDim() == m_params.outputDim, "MergeShapesNode output dim doesn't match params");
    for (unsigned i = 0; i < m_params.outputDim; i++)
    {
        if (m_params.dimMap[i].inputIdx == -1) continue;
        CHECK_RET_FALSE(m_params.dimMap[i].inputIdx < m_inputs.size(),
                        "MergeShapesNode dim map points to out of range input");
        CHECK_RET_FALSE(m_params.dimMap[i].dimIdx < m_inputs[m_params.dimMap[i].inputIdx]->getDim(),
                        "MergeShapesNode dim map points to out of range input dim");
    }

    return BaseClass::validateNode();
}

std::string MergeShapesNode::paramsToString(const SifMergeShapesMetadata& params)
{
    std::stringstream ss;
    ss << "outputDim: " << params.outputDim << ", fillValue: " << params.fillValue << ". DimMap: (";
    for (unsigned i = 0; i < params.outputDim; i++)
    {
        ss << "[" << params.dimMap[i].inputIdx << "," << params.dimMap[i].dimIdx << "]";
    }
    ss << ")";
    return ss.str();
}

NodePtr MergeShapesNode::createNode(const TensorVector& inputs,
                                    const TensorVector& outputs,
                                    UserParams          userParams,
                                    std::string_view    guid,
                                    std::string_view    name)
{
    return NodePtr(new MergeShapesNode(inputs, outputs, userParams, name));
}

NodePtr MergeShapesNode::clone() const
{
    return NodePtr(new MergeShapesNode(*this));
}

SifNodeParams MergeShapesNode::getShapeInferenceFunctionUserParams()
{
    return reinterpret_cast<SifNodeParams>(&m_params);
}

size_t MergeShapesNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(m_params);
}
