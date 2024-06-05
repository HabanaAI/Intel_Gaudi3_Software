#pragma once

// synapse-internal includes (relative to src/)
#include "graph_compiler/types.h"
#include "graph_compiler/habana_nodes/tpc_node.h"

// synapse-internal includes (relative to specs_external/)
#include "perf_lib_layer_params.h"

// eager includes (relative to src/eager/lib/)
#include "node_info/eager_node.h"
#include "recipe_gen/recipe_hal_base.h"

namespace eager_mode
{
// Drop constant\cast operation acting on a const input in favor of an output const tensor.
// this reduces compile time as we do not need to process the extra physical node (tpc nodes are expensive).
// this reduces device time as we have less nodes to execute on the device,
// saving DRAM read\write latency and sync scheme latency, translating into
// a few microseconds savings per dropped node.
// and we also usually reduce recipe size by a few to several KBs most of the times as the kernels
// are a few to several KBs and most constants we deal with are 1 dimensional scalars,
// where the kernel does not appear more than a few times.
// For cases where the kernel is still needed due to non dropped nodes the overhead
// to recipe size is still not large as we have a bound on the total size of added const
// tensors.
// Additionaly each tpc node also contributes to recipe size due to execution and patching blobs
// which are in the scope of hundreds of bytes, so const tensors smaller than that would also always
// reduce the recipe size.
class ConstantTensorOptimizer
{
public:
    ConstantTensorOptimizer(const RecipeHalBase& recipeHal) : m_recipeHal(recipeHal) {}
    bool tryReplaceNodeByConstTensor(const TensorVector& inputs,
                                     const TensorVector& outputs,
                                     UserParams          userParams,
                                     unsigned            userParamsSize,
                                     std::string_view    guid) const;

    bool tryReplaceNodeByConstTensor(const EagerNode& node) const
    {
        if (node.getEngineType() != EngineType::TPC) return false;
        // memsets are not relevant to the optimization and are frequent
        // so we need to make the check early for minimal overhead.
        if (node->isMemset()) return false;
        const auto& tpcNode = *node.get<TPCNode>();
        return tryReplaceNodeByConstTensor(node->getInputs(),
                                           node->getOutputs(),
                                           tpcNode.getParams(),
                                           tpcNode.getParamsSize(),
                                           node->getGUID());
    }

private:
    bool canReplaceConstantByConstTensor(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         std::string_view    guid) const;
    bool canReplaceCastByConstTensor(const TensorVector& inputs,
                                     const TensorVector& outputs,
                                     std::string_view    guid,
                                     CastF32RoundMode_t  roundingMode) const;

    bool tryReplaceConstantByConstTensor(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         UserParams          userParams,
                                         unsigned            userParamsSize,
                                         std::string_view    guid) const;
    bool tryReplaceCastByConstTensor(const TensorVector& inputs,
                                     const TensorVector& outputs,
                                     UserParams          userParams,
                                     unsigned            userParamsSize,
                                     std::string_view    guid) const;

    static bool
    tryReplaceCastFromFloatToIntByConstTensor(const TensorPtr& output, float val, CastF32RoundMode_t roundingMode);
    static bool
    tryReplaceCastFromFloatToFloatByConstTensor(const TensorPtr& output, float val, CastF32RoundMode_t roundingMode);
    template<typename T>
    static bool tryReplaceCastFromIntByConstTensor(const TensorPtr& output, T val);

    static bool tryReplaceConstantByConstTensor(const TensorPtr& output, ns_ConstantKernel::Params val);
    static bool
    tryReplaceCastByConstTensor(const TensorPtr& input, const TensorPtr& output, CastF32RoundMode_t roundingMode);

    const RecipeHalBase& m_recipeHal;
};

}  // namespace eager_mode
