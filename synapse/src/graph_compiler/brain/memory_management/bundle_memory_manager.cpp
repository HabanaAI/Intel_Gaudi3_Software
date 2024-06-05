#include "bundle_memory_manager.h"

#include "bundle_memory_preprocessor.h"
#include "bundle_memory_insulator.h"
#include "bundle_sram_allocator.h"
#include "bundle_memory_placer.h"
#include "bundle_memory_directive_executor.h"

using namespace gc::layered_brain;

BundleMemoryManager::BundleMemoryManager(HabanaGraph& graph, const BundleNodes& bundleNodes)
: m_graph(graph), m_nodes(bundleNodes)
{
}

bool BundleMemoryManager::placeTiles()
{
    TempLogContextSetter logCtx {
        fmt::format("BundleMemoryManager:bundle_{}",
                    std::to_string(m_nodes.front()->getNodeAnnotation().bundleInfo->bundleIndex))};
    preProcess();
    setPlacementDirectives();
    return executeDirectives();
}

void BundleMemoryManager::preProcess()
{
    m_db = getPreprocessor(m_nodes)->buildMemUsageDB();
}

void BundleMemoryManager::setPlacementDirectives()
{
    auto placer    = getPlacer();
    auto allocator = getAllocator();
    for (auto& stepEntry : m_db.steps)
    {
        placer->placeStepSlices(stepEntry, allocator.get());
    }
}

bool BundleMemoryManager::executeDirectives()
{
    auto executor = getExecutor();
    for (const auto& sliceAndEntry : m_db.slices)
    {
        bool res = executor->executeDirectivesFor(sliceAndEntry.first);
        if (!res) return false;
    }
    return true;
}

std::unique_ptr<BundleMemoryPreProcessor> BundleMemoryManager::getPreprocessor(const BundleNodes& bundleNodes)
{
    // Can be replaced by factory/abstract-factory call
    return std::unique_ptr<BundleMemoryPreProcessor>(new POCBundleMemoryPreProcessor(m_graph, bundleNodes));
}

std::unique_ptr<BundleMemoryInsulator> BundleMemoryManager::getInsulator(const BundleNodes& bundleNodes)
{
    // Can be replaced by factory/abstract-factory call
    return std::unique_ptr<BundleMemoryInsulator>(new POCBundleMemoryInsulator {m_graph, bundleNodes});
}

std::unique_ptr<BundleSRAMAllocator> BundleMemoryManager::getAllocator()
{
    // Can be replaced by factory/abstract-factory call
    uint64_t inputMemoryBudget = SlicingBrain::knobs.maxSRAMCapInBytes;
    return std::unique_ptr<BundleSRAMAllocator>(new POCBundleSRAMAllocator {inputMemoryBudget});
}

std::unique_ptr<BundleMemoryPlacer> BundleMemoryManager::getPlacer()
{
    // Can be replaced by factory/abstract-factory call
    return std::unique_ptr<BundleMemoryPlacer>(new POCBundleMemoryPlacer {m_db});
}

std::unique_ptr<BundleMemoryDirectiveExecutor> BundleMemoryManager::getExecutor()
{
    // Can be replaced by factory/abstract-factory call
    return std::unique_ptr<BundleMemoryDirectiveExecutor>(new POCBundleMemoryDirectiveExecutor {m_graph, m_db});
}
