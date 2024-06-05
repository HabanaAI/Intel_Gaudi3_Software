#include <synapse_types.h>
#include <vector>
#include "tpc_kernel_names.h"
#include "infra/defs.h"

std::vector<std::string> getBN1Guids(Direction direction)
{
    std::vector<std::string> guidList;
    guidList.reserve(3);
    for (const auto datatype : {syn_type_bf16, syn_type_float, syn_type_fp8_152})
    {
        guidList.emplace_back(getBN1Guid(direction, datatype));
    }
    return guidList;
}

std::vector<std::string> getBN2Guids(Direction direction)
{
    std::vector<std::string> guidList;
    guidList.reserve(9);
    for (const auto op : {BN_OPS_BN, BN_OPS_BN_ACTIVATION, BN_OPS_BN_ADD_ACTIVATION})
    {
        for (const auto datatype : {syn_type_bf16, syn_type_float, syn_type_fp8_152})
        {
            guidList.emplace_back(getBN2Guid(op, direction, datatype));
        }
    }
    return guidList;
}
