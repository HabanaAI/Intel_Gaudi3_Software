#include "dynamic_reshape_shape_node.h"

#include "types_exception.h"

#include <string_view>

DynamicReshapeShapeNode::DynamicReshapeShapeNode(const TensorVector& inputs,
                                                 const TensorVector& outputs,
                                                 UserParams          userParams,
                                                 std::string_view    name)
: BaseClass(inputs, outputs, name, Node::TYPE_DYNAMIC_RESHAPE_SHAPE, SIF_DYNAMIC_RESHAPE)
{
    setParams(userParams, sizeof(dynamicReshapeParams));
    // validate equation and generate sif params
    m_parsing_status = parseEquation();
}

bool DynamicReshapeShapeNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 1, "DynamicReshapeShapeNode expects 1 input");

    CHECK_RET_FALSE(m_outputs.size() == 1 && m_outputs.front()->isShapeTensor(),
                    "DynamicReshapeShapeNode expects 1 output that is a shape tensor");

    CHECK_RET_FALSE(m_parsing_status, "Failed to parse the reshape equation");

    // Inheritance order is ShapeOperationNode : ReshapeNode : LogicalOpNode, use LogicalOpNode's validation
    return BaseClass::BaseClass::BaseClass::validateNode();
}

bool DynamicReshapeShapeNode::validateDynamicShapes() const
{
    // This node is supposed to be created for dynamic shapes only
    if (!isDynamicShape())
    {
        LOG_ERR(HABANA_NODE, "Static shape passed to node {}", getNodeName());
        return false;
    }
    return true;
}

NodePtr DynamicReshapeShapeNode::createNode(const TensorVector& inputs,
                                            const TensorVector& outputs,
                                            UserParams          userParams,
                                            std::string_view    guid,
                                            std::string_view    name)
{
    return NodePtr(new DynamicReshapeShapeNode(inputs, outputs, userParams, name));
}

void DynamicReshapeShapeNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParamsSize != sizeof(dynamicReshapeParams))
    {
        LOG_ERR(HABANA_NODE, "DynamicReshapeShapeNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(dynamicReshapeParams));
    }
    dynamicReshapeParams* params = reinterpret_cast<dynamicReshapeParams*>(userParams);
    m_equation                   = params->equation;
    LOG_TRACE(HABANA_NODE, "DynamicReshapeShapeNode name - {}, params - equation={}", getNodeName(), m_equation);
}

NodePtr DynamicReshapeShapeNode::clone() const
{
    return NodePtr(new DynamicReshapeShapeNode(*this));
}

static std::vector<std::string> tokenize(std::string_view str)
{
    std::vector<std::string> tokens;
    std::string              t;
    for (char c : str)
    {
        if (c == ',')
        {
            tokens.push_back(std::exchange(t, ""));
            continue;
        }
        t.push_back(c);
    }
    tokens.push_back(std::move(t));

    return tokens;
}

bool DynamicReshapeShapeNode::parseEquation()
{
    // to be performed on input shape to get the output shape.
    // The format is "{input dim list}->{output dim list}", where each one is comma-separated
    // list of single letter labels and arithmetic operators and/or numeric constants.
    // e.g. given input shape [5,6,7] and equation "a,b,c->a*b,c" yields [30,7]
    //                        [5,6,7]              "a,b,c->a*b*c,1,1"    [210,1,1]

    size_t split = m_equation.find("->");

    if (split == std::string::npos || m_equation.find("->", split + 1) != std::string::npos)
    {
        LOG_ERR(HABANA_NODE, "Expecting exactly one '->' in reshape equation: \"{}\"", m_equation);
        return false;
    }

    m_input_labels = tokenize(m_equation.substr(0, split));  // maps input dimension to label
    m_output_eq    = m_equation.substr(split + 2);           // output tensor equation string

    // Verify all labels contain a single character and are unique
    std::array<bool, 26> label_array = {false};
#define LBL2IDX(c) ((c) - 'a')
    for (auto const& s : m_input_labels)
    {
        if (s.length() != 1)
        {
            LOG_ERR(HABANA_NODE, "Only single character input labels are supported (found {})", s);
            return false;
        }
        if (label_array[LBL2IDX(s[0])] == true)
        {
            LOG_ERR(HABANA_NODE, "All input labels should be unique (found repeating label {})", s);
            return false;
        }
        label_array[LBL2IDX(s[0])] = true;
    }
    // Verify output equations do not contain unknown dimension labels
    for (char const c : m_output_eq)
    {
        if (std::isalpha(c) && label_array[LBL2IDX(c)] == false)
        {
            LOG_ERR(HABANA_NODE, "Found label {} in the output tensor that does not exist in the input tensor", c);
            return false;
        }
    }
#undef LBL2IDX
    return true;
}

SifNodeParams DynamicReshapeShapeNode::getShapeInferenceFunctionUserParams()
{
    if (!m_parsing_status) return nullptr;

    if (m_sifMetadataBuffer.empty())
    {
        m_sifMetadataBuffer.resize(getShapeInferenceFunctionUserParamsSize());

        SifDynamicReshapeMetadata* metadata = reinterpret_cast<SifDynamicReshapeMetadata*>(m_sifMetadataBuffer.data());

        metadata->input_dims = m_input_labels.size();
        for (size_t i = 0; i < metadata->input_dims; i++)
        {
            metadata->input_dims_to_labels[i] = m_input_labels[i][0];
        }
        strncpy(metadata->output_eq, m_output_eq.c_str(), MAX_USER_PARAMS_SIZE-1);
        metadata->output_eq[MAX_USER_PARAMS_SIZE-1] = 0;
    }

    return reinterpret_cast<SifNodeParams>(m_sifMetadataBuffer.data());
}

size_t DynamicReshapeShapeNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifDynamicReshapeMetadata);
}

char DynamicReshapeShapeNode::getLabelForDim(unsigned dim)
{
    return 'a' + dim;
}

ComponentVector DynamicReshapeShapeNode::initComponentVector(unsigned rank)
{
    ComponentVector res;
    for (unsigned i = 0; i < rank; i++)
    {
        res.push_back(std::string {getLabelForDim(i)});
    }
    return res;
}

std::string DynamicReshapeShapeNode::makeReshapeEquation(const ComponentVector& c, unsigned input_rank)
{
    std::string eq;
    for (unsigned i = 0; i < input_rank; i++)
    {
        eq.push_back(getLabelForDim(i));
        if (i < input_rank - 1) eq.push_back(',');
    }
    eq += "->";
    for (unsigned i = 0; i < c.size(); i++)
    {
        if (c[i] == "")  // skip empty dims
        {
            // if empty dim is the last, remove extra comma
            if (i == c.size() - 1) eq.pop_back();
            continue;
        }

        eq += c[i];
        if (i < c.size() - 1) eq.push_back(',');
    }
    return eq;
}

EinsumExpandShapeNode::EinsumExpandShapeNode(const TensorVector& inputs,
                                             const TensorVector& outputs,
                                             UserParams          userParams,
                                             std::string_view    name)
: BaseClass(inputs, outputs, name, Node::TYPE_EINSUM_EXPAND_SHAPE, SIF_EINSUM_EXPAND)
{
    setParams(userParams, sizeof(einsumExtractParams));
}

bool EinsumExpandShapeNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 2 || m_inputs.size() == 3,
                    "EinsumExpandShapeNode expects 2 or 3 inputs (Batch tensor and 1 or 2 Einsum input tensors)");

    CHECK_RET_FALSE(m_outputs.size() == 1 && m_outputs.front()->isShapeTensor(),
                    "EinsumExpandShapeNode expects 1 output that is a shape tensor");

    // Inheritance order is ShapeOperationNode : ReshapeNode : LogicalOpNode, use LogicalOpNode's validation
    return BaseClass::BaseClass::BaseClass::validateNode();
}

bool EinsumExpandShapeNode::validateDynamicShapes() const
{
    // This node is supposed to be created for dynamic shapes only
    if (!isDynamicShape())
    {
        LOG_ERR(HABANA_NODE, "Static shape passed to node {}", getNodeName());
        return false;
    }
    return true;
}

NodePtr EinsumExpandShapeNode::createNode(const TensorVector& inputs,
                                          const TensorVector& outputs,
                                          UserParams          userParams,
                                          std::string_view    guid,
                                          std::string_view    name)
{
    return NodePtr(new EinsumExpandShapeNode(inputs, outputs, userParams, name));
}

void EinsumExpandShapeNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParamsSize != sizeof(einsumExtractParams))
    {
        LOG_ERR(HABANA_NODE, "EinsumExpandShapeNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(einsumExtractParams));
    }
    einsumExtractParams* params = reinterpret_cast<einsumExtractParams*>(userParams);
    m_freeDims1                 = params->freeDims[0];
    m_freeDims2                 = params->freeDims[1];
    LOG_TRACE(HABANA_NODE,
              "EinsumExpandShapeNode name - {}, params - freeDims1={}, freeDims2={}",
              getNodeName(),
              toString(m_freeDims1, ','),
              toString(m_freeDims2, ','));
}

NodePtr EinsumExpandShapeNode::clone() const
{
    return NodePtr(new EinsumExpandShapeNode(*this));
}

template<typename T>
static size_t getContainerSize(T& c)
{
    return c.size() * sizeof(*c.data());
}

SifNodeParams EinsumExpandShapeNode::getShapeInferenceFunctionUserParams()
{
    if (m_sifMetadataBuffer.empty())
    {
        m_sifMetadataBuffer.resize(getShapeInferenceFunctionUserParamsSize());

        SifEinsumExpandShapeMetadata* metadata =
            reinterpret_cast<SifEinsumExpandShapeMetadata*>(m_sifMetadataBuffer.data());

        metadata->numOfFreeDimsInFirstInput = m_freeDims1.size();
        if (m_freeDims1.data())
        {
            memcpy(metadata->freeDimsInFirstInput, m_freeDims1.data(), getContainerSize(m_freeDims1));
        }

        if (m_inputs.size() > 2)
        {
            metadata->numOfFreeDimsInSecondInput = m_freeDims2.size();
            if (m_freeDims2.data())
            {
                memcpy(metadata->freeDimsInSecondInput, m_freeDims2.data(), getContainerSize(m_freeDims2));
            }
        }
    }

    return reinterpret_cast<SifNodeParams>(m_sifMetadataBuffer.data());
}

size_t EinsumExpandShapeNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifEinsumExpandShapeMetadata);
}
