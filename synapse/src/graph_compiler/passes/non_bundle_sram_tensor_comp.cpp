#include "habana_graph.h"
#include "liveness_analysis.h"
#include "node.h"
#include "tensor.h"
#include "non_bundle_sram_tensor_comp.h"

NonBundleSramTensorComp::NonBundleSramTensorComp(const HabanaGraph* graph)
: m_graph(graph)
{
}

bool NonBundleSramTensorComp::operator ()(const pTensor &tensor)
{
    if (tensor->inSram())
    {
        pNode producer = m_graph->getTensorProducer(tensor);
        if (producer && !producer->getNodeAnnotation().bundleInfo.is_set() && !tensor->isPartOfRMWSection())
        {
            return true;
        }
    }
    return false;
}

uint64_t NonBundleSramTensorComp::getGraphNonBundleTensorSramSize(const HabanaGraph* graph)
{
    uint64_t sramSizeForNonBundleTensors = 0;
    if (GCFG_NON_BUNDLE_SRAM_ALLOCATION_FACTOR.value() >= 1.0)
    {
        const uint64_t maxLiveSramCapacity = NonBundleSramTensorComp::getMaxLiveSramCapacityForNonBundleTensors(graph);
        sramSizeForNonBundleTensors = (float)maxLiveSramCapacity * GCFG_NON_BUNDLE_SRAM_ALLOCATION_FACTOR.value();
        if (sramSizeForNonBundleTensors > 0)
        {
            sramSizeForNonBundleTensors =
                std::max(sramSizeForNonBundleTensors, GCFG_MIN_SRAM_SIZE_FOR_NON_BUNDLE_TENSORS.value());
        }
        LOG_DEBUG(GC,
                  "Total SRAM size for non-bunble tensors: {} (max live SRAM capcity={}, allocation factor={}, min "
                  "SRAM size={})",
                  sramSizeForNonBundleTensors,
                  maxLiveSramCapacity,
                  GCFG_NON_BUNDLE_SRAM_ALLOCATION_FACTOR.value(),
                  GCFG_MIN_SRAM_SIZE_FOR_NON_BUNDLE_TENSORS.value());
    }
    return sramSizeForNonBundleTensors;
}

uint64_t NonBundleSramTensorComp::getMaxLiveSramCapacityForNonBundleTensors(const HabanaGraph* graph)
{
    LivenessAnalysis analyzer(graph, NonBundleSramTensorComp(graph));

    return analyzer.getGraphMaxCapacity();
}
