#include "gaudi3_graph.h"
#include "types.h"

namespace gaudi3
{

bool alignTransposeViaGemmOutput(Gaudi3Graph& g)
{
    for (const NodePtr& node : g.getExeSortedNodes())
    {
        if (g.runsOnMME(node))
        {
            MMENodePtr mmeNode = std::static_pointer_cast<MmeNode>(node);
            HB_ASSERT(mmeNode != nullptr, "node runs on mme but can't be cast to mme node");
            if (mmeNode->isTransposeViaGemm())
            {
                const TensorPtr& transposeOutput = mmeNode->getOutput(TENSOR_OFM);

                NodeList consumers = g.getTensorConsumers(transposeOutput);

                bool noLogicalConsumers = std::none_of(consumers.begin(),
                                                       consumers.end(),
                                                       [](NodePtr n) { return n->isLogicalOperation(); });

                // Align the strides of the transpose output to cache line
                if (!transposeOutput->isPersistent() && noLogicalConsumers)
                {
                    transposeOutput->alignStridesToCacheLine(); // will do nothing if FCD < cacheline
                }
            }
        }
    }
    return true;
}

}  // namespace gaudi3