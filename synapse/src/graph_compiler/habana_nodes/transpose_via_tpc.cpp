#include "node_factory.h"
#include "kernel_db.h"
#include "data_type_utils.h"
#include "graph_traits.h"

#include "synapse_common_types.h"
#include "transpose_via_tpc.h"

bool TransposeViaTPC::canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    const Tensor* input = transposeNodeParams.input.get();
    if (LogicalTransposeNode::isSupportedPermutation(*input,
                                                     *transposeNodeParams.output,
                                                     transposeNodeParams.permutation))
    {
        return false;
    }
    synDataType in_type  = input->getElementType();
    auto        deviceId = deviceTypeToDeviceID(hal->getDeviceType());
    return !getTransposeGuid(in_type, deviceId).empty();
}

NodeVector TransposeViaTPC::extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    ns_TransposeKernel::Params params        = getParams(transposeNodeParams.permutation);
    const TensorPtr&           input         = transposeNodeParams.input;
    synDataType                in_type       = input->getElementType();
    auto                       deviceId      = deviceTypeToDeviceID(hal->getDeviceType());
    std::string                transposeGuid = getTransposeGuid(in_type, deviceId);
    HB_ASSERT(!transposeGuid.empty(), "Unexpected empty transpose guid");
    NodePtr transposeNode = NodeFactory::createGenericTPCNode({input},
                                                              {transposeNodeParams.output},
                                                              nullptr,
                                                              transposeGuid,
                                                              transposeNodeParams.nodeName.value_or("noName"));
    static_cast<TPCNode*>(transposeNode.get())->storeParamsInBuffer(&params, sizeof(ns_TransposeKernel::Params));
    return {transposeNode};
}

uint64_t TransposeViaTPC::calculateCost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    return m_costModel.getCost(extract(transposeNodeParams, hal));
}

ns_TransposeKernel::Params TransposeViaTPC::getParams(const TransposePermutationArray& permutation) const
{
    ns_TransposeKernel::Params params;
    for (unsigned int dim = 0; dim < SYN_MAX_TENSOR_DIM - 1; ++dim)
    {
        if (dim < permutation.size())
        {
            params.axes[permutation[dim]] = dim;
        }
        else
        {
            params.axes[dim] = dim;
        }
    }
    return params;
}

std::string TransposeViaTPC::getTransposeGuid(synDataType type, tpc_lib_api::DeviceId deviceId) const
{
    static const StringWithHash transposeGuidInference("transpose_f32");
    if (KernelDB::instance().isKernelExist(transposeGuidInference, deviceId))
    {
        if (type == syn_type_single)
        {
            return transposeGuidInference.getKey();
        }
        else
        {
            return "";
        }
    }
    return fmt::format("transpose_fwd_{}", getDtypeSuffixFromSynDataType(type));
}