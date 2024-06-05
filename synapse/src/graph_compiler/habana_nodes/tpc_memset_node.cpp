
#include "tpc_memset_node.h"
#include "habana_graph.h"
#include "data_type_utils.h"

TPCMemsetNode::TPCMemsetNode(const TensorVector&        in,
                             const TensorVector&        out,
                             std::string_view           name,
                             ns_ConstantKernel::Params& params)
: TPCNode(in, out, name, &params, sizeof(ns_ConstantKernel::Params))
{
}

pNode TPCMemsetNode::createNode(const TensorVector& in,
                                const TensorVector& out,
                                UserParams          userParams,
                                std::string_view    guid,
                                std::string_view    name)
{
    HB_ASSERT(userParams == nullptr, "TPCMemsetNode: currently only memset to zero is supported");
    ns_ConstantKernel::Params params = {};
    params.constant.f = 0;
    return pNode(new TPCMemsetNode(in, out, name, params));
}

pNode TPCMemsetNode::clone() const
{
    return pNode(new TPCMemsetNode(*this));
}

bool TPCMemsetNode::validateNode() const
{
    if ((m_inputs.size() != 0 && m_inputs.size() != 1) || (m_outputs.size() != 1))
    {
        LOG_ERR(HABANA_NODE, "TPCMemsetNode: Invalid number of operands (expecting 0 or 1 input and 1 output)");
        return false;
    }
    if (m_inputs.size() == 1 && !m_inputs[0]->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "TPCMemsetNode: input tensor must be a shape tensor");
        return false;
    }

    return TPCNode::validateNode();
}

bool TPCMemsetNode::isMemset() const
{
    return true;
}

void TPCMemsetNode::setGUID(const StringViewWithHash& guidAndHash)
{
    Node::setGUID(fmt::format("constant_{}", getDtypeSuffixFromSynDataType(getOutput(0)->getElementType())));
}