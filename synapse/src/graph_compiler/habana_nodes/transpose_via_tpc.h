#pragma once

#include <perf_lib_layer_params.h>

#include "transpose_nodes_creator.h"
#include "transpose_strategies.h"
#include "tpc_transpose_cost_model.h"

class TransposeViaTPC : public TransposeViaPhysical
{
public:
    bool             canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    NodeVector       extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    std::string_view strategyName() const override { return "Transpose by TPC"; }
    uint64_t calculateCost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;

private:
    TpcTransposeCostModel      m_costModel;
    ns_TransposeKernel::Params getParams(const TransposePermutationArray& permutation) const;
    std::string                getTransposeGuid(synDataType type, tpc_lib_api::DeviceId deviceId) const;
};