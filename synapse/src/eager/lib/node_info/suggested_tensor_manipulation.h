#pragma once

// synapse-internal includes (relative to src/)
#include "graph_compiler/passes/optimize_tpc_kernels.h"

namespace eager_mode
{
class EagerModeSuggestedManipulationHandler final : public SuggestedManipulationHandlerBase
{
public:
    EagerModeSuggestedManipulationHandler(HabanaGraph& graph, TPCNode& node)
    : SuggestedManipulationHandlerBase(graph, node)
    {
        m_skipDynamicNodeHandling = true;
    }
    bool        applySuggestedTensorManipulation() override;
    static bool shouldSkipSuggestedTensorManipulation(TPCNode& node, const HabanaGraph& graph);

private:
    bool        isTensorSparse(const TensorPtr& tensor) const override;
    std::string getSuggestedTensorManipulationNodeName(tpc_lib_api::TensorOperationType opType,
                                                       unsigned                         tensorIdx,
                                                       bool                             isInput) override;
    void        setManipulatedTensorNameName(Tensor& t, std::string_view nameSuffix) override;
};

}  // namespace eager_mode