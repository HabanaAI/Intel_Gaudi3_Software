#include "concatenate_node.h"

#include "aggregate_fcd_node.h"
#include "node_factory.h"
#include "node_io_manager.h"
#include "tensor.h"
#include "tensor_shape.h"
#include "utils.h"

ConcatenateNode::ConcatenateNode(const TensorVector& in,
                                 const TensorVector& out,
                                 UserParams          userParams,
                                 std::string_view    name)
: BaseClass(in, out, name, INPUT_TO_OUTPUT, TYPE_INTERNAL_CONCAT, SIF_CONCATENATE, userParams)
{
    //Concat is a pragma, nothing actually executes
}

NodePtr ConcatenateNode::createNode(const TensorVector& inputs,
                                    const TensorVector& outputs,
                                    UserParams          userParams,
                                    std::string_view    guid,
                                    std::string_view    name)
{
    return createConcatNode(inputs, outputs, userParams, guid, name, false);
}

NodePtr ConcatenateNode::createNodeInternal(const TensorVector& inputs,
                                            const TensorVector& outputs,
                                            UserParams          userParams,
                                            std::string_view    guid,
                                            std::string_view    name)
{
    return createConcatNode(inputs, outputs, userParams, guid, name, true);
}

NodePtr ConcatenateNode::createConcatNode(const TensorVector& inputs,
                                          const TensorVector& outputs,
                                          UserParams          userParams,
                                          std::string_view    guid,
                                          std::string_view    name,
                                          bool                isInternalNode)
{
    unsigned dim = 0;

    if (userParams != nullptr)
    {
        dim = reinterpret_cast<synConcatenateParams*>(userParams)->axis;
        LOG_TRACE(HABANA_NODE, "ConcatenateNode name - {}, params - dim={}", name, dim);
    }
    else
    {
        userParams = (void*)&dim;
    }

    if (checkIfPhysicalConcat(inputs, outputs, dim))
    {
        return NodeFactory::createNode(inputs, outputs, &dim, sizeof(dim), NodeFactory::physicalConcatNodeTypeName, name);
    }

    std::set<TensorPtr> concatInputsSet {std::begin(inputs), std::end(inputs)};
    TensorVector        operands = inputs;
    operands.insert(operands.end(), outputs.begin(), outputs.end());
    bool isPartOfRmwSection =
        std::any_of(operands.begin(), operands.end(), [](const TensorPtr& t) { return t->isPartOfRMWSection(); });

    bool enableFcdOptimization =
        (dim == 0) && !isInternalNode && !isPartOfRmwSection &&
        !is64BitOperands(inputs,
                         outputs) &&  // Don't optimize when operation is 64bit, not supported on physical operations
        (concatInputsSet.size() ==
         inputs.size());  // Don't optimize in case we have duplicated inputs to enable replacing concat with
                          // broadcast (optimizeConcatOp works on logical nodes)

    if (enableFcdOptimization)
    {
        // Optimized version for concat on FCD.
        return NodePtr(new ConcatFcdNode(inputs, outputs, name));
    }

    return NodePtr(new ConcatenateNode(inputs, outputs, userParams, name));
}

NodePtr ConcatenateNode::createNodeLogicalInternal(const TensorVector& inputs,
                                                   const TensorVector& outputs,
                                                   UserParams          userParams,
                                                   std::string_view    guid,
                                                   std::string_view    name)
{
    unsigned dim = 0;

    if (userParams != nullptr)
    {
        dim = reinterpret_cast<synConcatenateParams*>(userParams)->axis;
        LOG_TRACE(HABANA_NODE, "ConcatenateNode name - {}, params - dim={}", name, dim);
    }
    else
    {
        userParams = (void*)&dim;
    }

    return NodePtr(new ConcatenateNode(inputs, outputs, userParams, name));
}

bool ConcatenateNode::checkIfPhysicalConcat(const TensorVector& inputs, const TensorVector& outputs, unsigned dim)
{
    if (inputs.size() <= 1)
    {
        return false;
    }

    for (const pTensor& tensor : inputs)
    {
        if (tensor->isDynamicDim(dim))
        {
            return true;
        }
    }

    for (const pTensor& tensor : outputs)
    {
        if (tensor->isDynamicDim(dim))
        {
            return true;
        }
    }

    return false;
}

NodePtr ConcatenateNode::clone() const
{
    return NodePtr(new ConcatenateNode(*this));
}

bool ConcatenateNode::RunOnCpu()
{
    unsigned                                elemsUntilConcat = 0;
    unsigned                                elemsFromConcat  = 0;
    TensorPtr                               output           = getOutput(TENSOR_OFM);
    std::vector<std::pair<char*, unsigned>> pInputs;
    for (auto t : m_inputs)
    {
        elemsUntilConcat += t->getStrideInElements(m_aggDim + 1);
        pInputs.push_back(std::pair<char*, unsigned>(static_cast<char*>(t->map()), t->getStrideInBytes(m_aggDim + 1)));

        unsigned SP = t->getTotalElements() / t->getStrideInElements(m_aggDim + 1);
        if (elemsFromConcat == 0)
        {
            elemsFromConcat = SP;
        }
        else
        {
            HB_ASSERT(elemsFromConcat == SP, "size inconsistency between inputs");
        }
    }
    HB_ASSERT(elemsUntilConcat == output->getStrideInElements(m_aggDim + 1),
              "channel inconsistency between inputs and output");
    unsigned outputSP = output->getTotalElements() / output->getStrideInElements(m_aggDim + 1);
    UNUSED(outputSP);
    HB_ASSERT(elemsFromConcat == outputSP, "spatial size inconsistency between inputs and output");
    char* pOutput = static_cast<char*>(output->map());

    for (unsigned spIndex = 0; spIndex < elemsFromConcat; ++spIndex)
    {
        for (auto& it : pInputs)
        {
            memcpy(pOutput, it.first, it.second);
            it.first += it.second;
            pOutput  += it.second;
        }
    }
    return true;
}

TensorShape ConcatenateNode::getInputShape(const TensorShape& outputShape, uint32_t outputIndex, uint32_t inputIdx) const
{
    HB_ASSERT(outputIndex == TENSOR_OFM, "output index mismatch, real:{}, expected:{}", outputIndex, TENSOR_OFM);
    const TensorPtr& input = getInput(inputIdx);
    HB_ASSERT_PTR(input);

    NCoordArray base(outputShape.getNBases());
    NSizeArray  size(outputShape.getNSizes());
    if (base[m_aggDim] !=0 ||
        size[m_aggDim] != getOutput(0)->getSizeInElements(m_aggDim))
    {
        return Node::getInputShape(outputShape, outputIndex, inputIdx);
    }
    else
    {
        uint32_t inputShapeDim = input->getDim();
        size[m_aggDim] = m_aggDim < inputShapeDim ? input->getSizeInElements(m_aggDim) : 1;

        return TensorShape(inputShapeDim,
                           size,
                           base);
    }
}

SifNodeParams ConcatenateNode::getShapeInferenceFunctionUserParams()
{
    return getShapeInferenceFunctionUserParams(m_metadata, m_aggDim, m_inputs);
}

SifNodeParams ConcatenateNode::getShapeInferenceFunctionUserParams(Settable<SifConcatenateMetadata>& metadata,
                                                                   const unsigned                    aggregationDim,
                                                                   const TensorVector&               inputs)
{
    metadata.set({aggregationDim, inputs.back()->isShapeTensor()});
    return (SifNodeParams)&metadata.value();
}

size_t ConcatenateNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifConcatenateMetadata);
}

void ConcatenateNode::setInputLayouts(const LayoutVector& layouts)
{
    if (layouts.size() == m_inputs.size() + 1)
    {
        // we just removed an input, fix the layouts too
        LayoutVector newLayouts(layouts.begin(), layouts.begin() + m_inputs.size());
        Node::setInputLayouts(newLayouts);
    }
    else
    {
        Node::setInputLayouts(layouts);
    }
}
