#include "cache_warmup_node.h"
#include "data_type_utils.h"

CacheWarmupNode::CacheWarmupNode(const TensorVector&        in,
                                 const TensorVector&        out,
                                 std::string_view           name,
                                 ns_ConstantKernel::Params& params)
: TPCNode(in, out, name, &params, sizeof(ns_ConstantKernel::Params))
{
}

pNode CacheWarmupNode::createNode(const TensorVector& in,
                                  const TensorVector& out,
                                  UserParams          userParams,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    HB_ASSERT(userParams == nullptr, "CacheWarmupNode (guid - {}): currently only memset to zero is supported", guid);
    ns_ConstantKernel::Params params = {};
    return pNode(new CacheWarmupNode(in, out, name, params));
}

pNode CacheWarmupNode::clone() const
{
    return pNode(new CacheWarmupNode(*this));
}

bool CacheWarmupNode::validateNode() const
{
    if ((m_inputs.size() != 0) || (m_outputs.size() != 1))
    {
        LOG_ERR(HABANA_NODE, "CacheWarmupNode: Invalid number of operands (expecting 0 input and 1 output)");
        return false;
    }
    // The strides of the tensor should be in increasing order
    if (!areStridesAscending(getOutput(0)->getNStridesInElements(), getOutput(0)->getDim())) return false;
    return TPCNode::validateNode();
}

// Behave as memset to allow the scheduler to optimize the node location to be close to the producer
bool CacheWarmupNode::isMemset() const
{
    return true;
}

void CacheWarmupNode::setGUID(const StringViewWithHash& guidAndHash)
{
    Node::setGUID(
        fmt::format("{}_{}", guidAndHash.getKey(), getDtypeSuffixFromSynDataType(getOutput(0)->getElementType())));
}