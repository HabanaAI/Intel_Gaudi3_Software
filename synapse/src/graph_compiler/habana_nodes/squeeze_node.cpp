#include "squeeze_node.h"

#include "access_pattern_generator.h"
#include "defs.h"

#include "synapse_common_types.h"
#include "types_exception.h"
#include "utils.h"

#include <algorithm>

SqueezeNode::SqueezeNode(const pTensor&   input,
                         const pTensor&   output,
                         UserParams       userParams,
                         std::string_view name,
                         eNodeType        nodeType)
: BaseClass({input}, {output}, name, nodeType, SIF_SQUEEZE)
{
    setParams(userParams, sizeof(synSqueezeParams));
}

NodePtr SqueezeNode::createNode(const TensorVector& inputs,
                                const TensorVector& outputs,
                                UserParams          userParams,
                                std::string_view    guid,
                                std::string_view    name)
{
    HB_ASSERT(inputs.size() == 1 && outputs.size() == 1, "Squeeze expects a single input and output");
    return NodePtr(new SqueezeNode(inputs[TENSOR_IFM], outputs[TENSOR_OFM], userParams, name));
}

gc::access_pattern::NodeAccessPatternPtr SqueezeNode::generateNodeAccessPattern() const
{
    return gc::access_pattern::AccessPatternSqueezeGenerator::generate(this, getSqueezedDims());
}

void SqueezeNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    m_sifParams = {0};
    if (userParams != nullptr)
    {
        if (userParamsSize != sizeof(synSqueezeParams))
        {
            LOG_ERR(HABANA_NODE, "SqueezeNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synSqueezeParams));
        }
        m_axis = ((synSqueezeParams*)userParams)->axis;
    }

    if (m_axis.is_set())
    {
        m_sifParams.squeezeDim[m_axis.value()] = 1;
    }
    else
    {
        for (unsigned dim = 0; dim < getInput(0)->getDim() && dim < ARRAY_SIZE(m_sifParams.squeezeDim); dim++)
        {
            if (getInput(0)->getSizeInElements(dim) == 1)
            {
                m_sifParams.squeezeDim[dim] = 1;
            }
        }
    }
    LOG_TRACE(HABANA_NODE,
              "SqueezeNode name - {}, params - squeezeDim={}",
              getNodeName(),
              arrayToString(m_sifParams.squeezeDim, ','));
}

NodePtr SqueezeNode::clone() const
{
    return NodePtr(new SqueezeNode(*this));
}

bool SqueezeNode::isAliasStrided() const
{
    const TensorPtr& real = getRealTensor();
    return !real->isDenseLayout();
}

DimVector SqueezeNode::getSqueezedDims() const
{
    DimVector ret;
    for (unsigned dim = 0; dim < m_inputs.front()->getDim(); dim++)
    {
        if (isDimSqueezed(dim))
        {
            ret.push_back(dim);
        }
    }
    return ret;
}

bool SqueezeNode::isDimSqueezed(unsigned dim) const
{
    const TensorPtr& in = m_inputs.front();
    if (dim >= in->getDim()) return false;
    return (m_axis == dim) || (!m_axis.is_set() && in->getSizeInElements(dim) == 1);
}

bool SqueezeNode::canHandleStridedRealTensor() const
{
    if (getAliasDirection() == OUTPUT_TO_INPUT)  // output is the alias
    {
        // we need to make sure that the squeezed tensor isn't strided on the FCD.
        // for example, if the input is strided on dimension 1, the resulting output will be strided on
        // dimension 0.
        const TensorPtr& in       = m_inputs.front();
        unsigned         firstDim = 0;
        while (isDimSqueezed(firstDim))
        {
            firstDim++;
        }
        return firstDim == 0 || (in->getStrideInBytes(firstDim) ==
                                 (in->getElementSizeInBytes() * in->getDenseStrideInElements(firstDim)));
    }
    return true;
}

void SqueezeNode::runLogicalOperation() const
{
    const TensorPtr& in                               = m_inputs.front();
    const TensorPtr& out                              = m_outputs.front();
    uint64_t         strides[Tensor::c_numOfNStrides] = {0};

    unsigned dim = 0;
    if (this->getAliasDirection() == INPUT_TO_OUTPUT)  // output is real
    {
        unsigned squeezedDims = 0;  // count number of squeezed dims so far
        for (; dim < in->getDim() + 1; dim++)
        {
            if (isDimSqueezed(dim))
            {
                // assign trivial stride on squeezed dimensions
                strides[dim] = in->getElementSizeInBytes() * in->getDenseStrideInElements(dim);
                squeezedDims++;
            }
            else
            {
                // on dims that are not squeezed, assign the outputs stride
                strides[dim] = out->getStrideInBytes(dim - squeezedDims);
            }
        }
    }
    else  // input is real
    {
        unsigned squeezedDims = 0;
        for (; dim + squeezedDims < in->getDim() + 1; dim++)
        {
            while (isDimSqueezed(dim + squeezedDims))
            {
                squeezedDims++;
            }
            strides[dim] = in->getStrideInBytes(dim + squeezedDims);
        }
        for (; dim < out->getDim() + 1; dim++)
        {
            // in case number of output dims is larger that squeezed dims (happens in some cases)
            HB_ASSERT(out->getSizeInElements(dim) == 1, "invalid dims for {}", getNodeName());
            strides[dim] = in->getStrideInBytes(in->getDim());
        }
    }

    const TensorPtr& real     = (getAliasDirection() == OUTPUT_TO_INPUT) ? in : out;
    const TensorPtr& alias    = (getAliasDirection() == OUTPUT_TO_INPUT) ? out : in;
    const auto&      sizes    = alias->getAllNSizesInElements();
    const auto&      minSizes = alias->getNMinimalSizesInElements();
    alias->reshape(alias->getDim(), sizes.data(), strides, minSizes.data());
    alias->setAsAliasSubTensor(real);
}

bool SqueezeNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Squeeze Node invalid number of operands");
        return false;
    }

    const auto& in  = getInput(TENSOR_IFM);
    const auto& out = getOutput(TENSOR_OFM);

    if (out->getDim() > in->getDim())
    {
        LOG_ERR(HABANA_NODE, "Squeeze Node output dimensionality should be smaller or equal to input");
        return false;
    }
    const NSizeArray inputSizes = in->getNSizesInElements();

    if (m_axis.is_set())
    {
        if (m_axis.value() >= in->getDim())
        {
            LOG_ERR(HABANA_NODE, "Squeeze Node dimension in axis out of bounds");
            return false;
        }
        if (in->getDim() != out->getDim() + 1)
        {
            LOG_ERR(HABANA_NODE, "Squeeze Node dimensions mismatch");
            return false;
        }
        if (inputSizes[m_axis.value()] != 1)
        {
            LOG_ERR(HABANA_NODE, "Squeeze Node dimension in axis must be equal to 1");
            return false;
        }
    }
    NSizeArray       expectedOutput = calculateExpectedOutput(inputSizes);
    const NSizeArray output         = out->getNSizesInElements();
    for (unsigned i = 0; i < HABANA_DIM_MAX; i++)
    {
        if (expectedOutput[i] != output[i])
        {
            LOG_ERR(HABANA_NODE,
                    "Squeeze Node invalid size for dimension {} expected {} got {}",
                    i,
                    expectedOutput[i],
                    output[i]);
            return false;
        }
    }
    return Node::validateNode();
}

NSizeArray SqueezeNode::calculateExpectedOutput(const NSizeArray& input) const
{
    NSizeArray ret;
    ret.fill(1);
    unsigned count = 0;
    if (m_axis.is_set())
    {
        for (unsigned dim = 0; dim < HABANA_DIM_MAX; dim++)
        {
            if (dim == m_axis.value())
            {
                continue;
            }
            else
            {
                ret[count] = input[dim];
                count++;
            }
        }
    }
    else
    {
        for (unsigned dim = 0; dim < HABANA_DIM_MAX; dim++)
        {
            if (input[dim] == 1)
            {
                continue;
            }
            else
            {
                ret[count] = input[dim];
                count++;
            }
        }
    }
    return ret;
}

SifNodeParams SqueezeNode::getShapeInferenceFunctionUserParams()
{
    return reinterpret_cast<SifNodeParams>(&m_sifParams);
}

size_t SqueezeNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(m_sifParams);
}

bool SqueezeNode::isNode64BitCompatible() const
{
    return true;
}

void SqueezeNode::permuteParams(const PermutationVector& inputPermutations)
{
    HB_ASSERT(inputPermutations.size() == 1,
              "Expect single input permutation when permuting params in squeeze node {}",
              getNodeName());
    if (m_axis.is_set())
    {
        m_axis.set(inputPermutations[0].permuteDim(m_axis.value()));
    }
    inputPermutations[0].permuteShape(m_sifParams.squeezeDim, inputPermutations[0].size());
}

SqueezeShapeNode::SqueezeShapeNode(const TensorPtr& inputs,
                                   const TensorPtr& outputs,
                                   UserParams       userParams,
                                   std::string_view name)
: BaseClass(inputs, outputs, userParams, name, TYPE_SQUEEZE_SHAPE)
{
}

bool SqueezeShapeNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() == 1, "SqueezeShapeNode Expects 1 input");

    CHECK_RET_FALSE(m_outputs.size() == 1 && m_outputs.front()->isShapeTensor(),
                    "SqueezeShapeNode Expects 1 output that is a shape tensor");

    return BaseClass::validateNode();
}

NodePtr SqueezeShapeNode::createNode(const TensorVector& inputs,
                                     const TensorVector& outputs,
                                     UserParams          userParams,
                                     std::string_view    guid,
                                     std::string_view    name)
{
    HB_ASSERT(inputs.size() == 1 && outputs.size() == 1, "Squeeze expects a single input and output");
    LOG_TRACE(HABANA_NODE, "SqueezeShapeNode name - {}", name);
    return NodePtr(new SqueezeShapeNode(inputs[TENSOR_IFM], outputs[TENSOR_OFM], userParams, name));
}

NodePtr SqueezeShapeNode::clone() const
{
    return NodePtr(new SqueezeShapeNode(*this));
}