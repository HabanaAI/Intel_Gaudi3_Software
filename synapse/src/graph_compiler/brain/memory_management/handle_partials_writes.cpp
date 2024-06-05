#include "hal_reader/hal_reader.h"
#include "partial_writes_detection.h"
#include "habana_graph.h"
#include "brain_conf.h"
#include "add_cache_warmup.h"

using namespace gc::layered_brain;

bool handlePartialsWrites(HabanaGraph& g)
{
    // Partials write is supported only for layered brain
    if (!GCFG_ENABLE_LAYERED_PIPELINE_BRAIN.value()) return true;
    // Check that partials handling is enabled for any engine type
    if (!GCFG_ENABLE_LB_PARTIALS_WRITE_TPC_HANDLING.value() && !GCFG_ENABLE_LB_PARTIALS_WRITE_MME_HANDLING.value())
    {
        return true;
    }
    HB_ASSERT(g.getHALReader()->isCacheSupported(), "partials handling is relevant only for devices with cache");

    auto graphNodes = g.getNodes();
    for (const NodePtr& node : graphNodes)
    {
        if (!node) continue;
        // AddCacheWarmup may replace graph nodes. Check that the node is still valid
        if (!g.containsNode(node)) continue;
        // AddCacheWarmup may call GraphEditor::replaceTensor. output is copied to protect its validity after
        // replacement
        for (TensorPtr output : node->getOutputs())
        {
            if (!output || output->isShapeTensor()) continue;

            PartialWritesDetector partialsDetect(output, g);
            const auto            decision = partialsDetect.checkTensor();
            if (!decision.has_value())
            {
                LOG_DEBUG(LB_PARTIALS, "{}: Partials detection rejects solution", HLLOG_FUNC);
                return false;
            }
            // Apply memset/memread decisions
            if (decision.value().warmupCache)
            {
                // TODO SW-155619 handle single core memset
                LOG_DEBUG(LB_PARTIALS,
                          "{}: Warmup cache required{}",
                          HLLOG_FUNC,
                          decision.value().allocInSingleDCore ? " on single DCore" : "");
                AddCacheWarmup(g, output, decision.value().allocInSingleDCore).run();
            }
        }
    }
    return true;
}