#pragma once

#include "transpose_nodes_creator.h"
#include "graph_compiler/mme/mme_brain_ifc.h"

class MmeTransposeCostModel : public TransposeCostModel
{
public:
    uint64_t getCost(const TensorPtr& input, const TransposePermutationArray& permutation) const override
    {
        HB_ASSERT(false,
                  "MmeTransposeCostModel::getCost(const TensorPtr& input, const TransposePermutationArray& "
                  "permutation) - Not implemented");
        return 0;
    }

    uint64_t getCost(const NodeVector& extractedNodes) const override
    {
        LOG_ERR(GC, "Use of cost model that is not implemented yet");  // TODO: remove error [SW-113510] [SW-116267]
        uint64_t cost = 0;

        for (auto node : extractedNodes)
        {
            if (!node->isTranspose()) continue;
            if (node->isLogicalOperation())
            {
                // TODO - uncomment once [SW-116267] is resolved
                // cost += getLogicalTransposeCost(node);
            }
            else
            {
                std::shared_ptr<MmeNode> mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
                HB_ASSERT_PTR(mmeNode);
                // TODO - uncomment once [SW-113510] is resolved
                // cost += MmeBrainIfc::getRecommendedConfigMmePerf(node).expectedRuntimeCycles;
            }
        }

        return cost;
    }

private:
    uint64_t getLogicalTransposeCost(const NodePtr& node) const
    {
        const auto& logicalTranspose = std::dynamic_pointer_cast<LogicalTransposeNode>(node);
        HB_ASSERT_PTR(logicalTranspose);
        const auto& input  = logicalTranspose->getInput(0);
        const auto& output = logicalTranspose->getOutput(0);

        MmeCommon::MmeLayerParams denseMemcpyParams =
            MmeBrainIfc::getMmeMemcpyLayerParams(input->getAllSizesInElements(), input->getAllStridesInBytes());
        MmeCommon::PerfAttr denseMemcpyPerf = MmeBrainIfc::getMmePerfFromParams(denseMemcpyParams);

        MmeCommon::MmeLayerParams stridedMemcpyParams =
            MmeBrainIfc::getMmeMemcpyLayerParams(output->getAllSizesInElements(), output->getAllStridesInBytes());
        MmeCommon::PerfAttr stridedMemcpyPerf = MmeBrainIfc::getMmePerfFromParams(stridedMemcpyParams);

        return stridedMemcpyPerf.expectedRuntimeCycles - denseMemcpyPerf.expectedRuntimeCycles;
    }
};