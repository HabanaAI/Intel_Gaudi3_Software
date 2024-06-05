#include "node_cache_setter.h"

#include "habana_graph.h"
#include "node_access_cache_setter.h"
#include "brain_conf.h"

using namespace gc::layered_brain;

NodeCacheSetter::NodeCacheSetter(HabanaGraph&         graph,
                                 const BundleNodes&   nodes,
                                 const MemoryUsageDB& db,
                                 BundleCacheState&    cacheStateTracker,
                                 unsigned             pipelineDepth)
: m_graph(graph),
  m_nodes(nodes),
  m_db(db),
  m_cacheState(cacheStateTracker),
  m_allocator(m_cacheState),
  m_pipelineDepth(pipelineDepth)
{
    LOG_DEBUG(LB_CACHE_MNGR,
              "Node cache setter initialized with effective cache budget of {}B ({:.2}MB)",
              m_cacheState.totalFree(),
              bToMb(m_cacheState.totalFree()));
}

bool NodeCacheSetter::setDirectives(size_t nodeIdx, CacheRequirementsAnalyzerIfc* requirementAnalyzer)
{
    m_currentNodeIdx = nodeIdx;
    m_currentNode    = m_nodes.at(nodeIdx);
    if (shouldSkipNode(m_currentNode)) return true;

    LOG_DEBUG(LB_CACHE_MNGR, "Setting cache directives for node: {}", m_currentNode->getNodeName());

    initAnnotations(m_currentNode);

    // The order of the following calls matters. The outputs allocation must happen before the inputs allocation, so
    // that the inputs will not be freed to make room for the outputs. It is also important that both calls happen, so
    // the return statement should not be shortened to:
    // return setOutputDirectives(...) && setInputDirectives(...)
    // or a failure in caching the outputs would result in omitting caching the inputs.
    CachingResult result = setOutputDirectives(requirementAnalyzer);
    result += setInputDirectives(requirementAnalyzer);

    releaseAll(result.release);
    ensureDependencies(result.dependencies);

    logNodeSummary(result);

    return result.successful;
}

bool NodeCacheSetter::shouldSkipNode(const NodePtr& node)
{
    return node == nullptr || node->isLogicalOperation() || node->isShapeOperation();
}

bool NodeCacheSetter::shouldSkipTensor(const TensorPtr& t)
{
    return t == nullptr || t->isShapeTensor();
}

NodeCacheSetter::CachingResult NodeCacheSetter::setInputDirectives(CacheRequirementsAnalyzerIfc* requirementAnalyzer)
{
    CachingResult allInputsResult {};
    for (size_t inputIdx = 0; inputIdx < m_currentNode->getNumInputs(); inputIdx++)
    {
        const TensorPtr& input = m_currentNode->getInput(inputIdx);
        if (shouldSkipTensor(input)) continue;

        NodeAccessCacheSetter accessSetter(m_currentNode);
        accessSetter.input(inputIdx);
        auto req = requirementAnalyzer->inputRequirement(m_currentNodeIdx, inputIdx);

        CachingResult inputResult = cacheAccess(input, req, accessSetter);
        if (inputResult.successful)
        {
            registerYieldingOption(inputIdx, req);
        }
        else
        {
            LOG_DEBUG(LB_CACHE_MNGR, "Failed to cache input[{}] of node {}", inputIdx, m_currentNode->getNodeName());
        }
        allInputsResult += inputResult;
    }
    return allInputsResult;
}

NodeCacheSetter::CachingResult NodeCacheSetter::setOutputDirectives(CacheRequirementsAnalyzerIfc* requirementAnalyzer)
{
    CachingResult allOutputsResult {};
    for (size_t outputIdx = 0; outputIdx < m_currentNode->getNumOutputs(); outputIdx++)
    {
        const TensorPtr& output = m_currentNode->getOutput(outputIdx);
        if (shouldSkipTensor(output)) continue;

        NodeAccessCacheSetter accessSetter(m_currentNode);
        accessSetter.output(outputIdx);
        auto req = requirementAnalyzer->outputRequirement(m_currentNodeIdx, outputIdx);

        CachingResult outputResult = cacheAccess(output, req, accessSetter);
        if (!outputResult.successful)
        {
            LOG_DEBUG(LB_CACHE_MNGR, "Failed to cache output[{}] of node {}", outputIdx, m_currentNode->getNodeName());
        }
        allOutputsResult += outputResult;
    }
    return allOutputsResult;
}

NodeCacheSetter::CachingResult NodeCacheSetter::cacheAccess(const TensorPtr&       tensor,
                                                            const Requirement&     requirements,
                                                            NodeAccessCacheSetter& accessSetter)
{
    CachingResult cachingResult {};
    if (requirements.cachingRequired())
    {
        LOG_DEBUG(LB_CACHE_MNGR,
                  "Trying to cache tensor: {} - Cap: {}B ({:.2}MB), will {}be released.",
                  tensor->getName(),
                  requirements.capacity,
                  bToMb(requirements.capacity),
                  requirements.releaseRequired() ? "" : "not ");

        AllocResult allocRes     = allocateCache(tensor, requirements);
        cachingResult.successful = allocRes.successful;
        if (allocRes.successful)
        {
            LOG_DEBUG(LB_CACHE_MNGR,
                      "    Allocation successful. Dependencies: [{}]. Free cache: {}B",
                      toString(allocRes.dependencies, ','),
                      m_cacheState.totalFree());

            // Fill all the details of the access only if the allocation is successful
            accessSetter.directive(requirements.directive);
            accessSetter.cacheClass(requirements.cacheClass);
            cachingResult.dependencies.insert(allocRes.dependencies.begin(), allocRes.dependencies.end());
        }
        else
        {
            LOG_DEBUG(LB_CACHE_MNGR,
                      "    Allocation failed. Missing {}B ({:.2}MB)",
                      allocRes.missingCapacity,
                      bToMb(allocRes.missingCapacity));
        }
    }
    // Release may be required even when caching is not required. Release is based on whether the access to the data is
    // the last one. Caching requirement is based on whether the data is accessed more than once. If the cdata is
    // already cached, it may be accessed only once and still require releasing.
    if (requirements.releaseRequired())
    {
        // setting release access may apply cache directive even if the allocation of the cache failed,
        // so only set release if the allocation was successful or the slice is already cached.
        if ((cachingResult.successful && requirements.cachingRequired()) || m_cacheState.isCached(cacheKey(tensor)))
        {
            setCacheAccessRelease(requirements, accessSetter);
            cachingResult.release.insert(tensor);
        }
    }

    accessSetter.set();

    return cachingResult;
}

void NodeCacheSetter::setCacheAccessRelease(const Requirement& requirements, NodeAccessCacheSetter& accessSetter)
{
    accessSetter
        .directive(requirements.releaseCacheDirective())  // For class based degrade (no CME)
        .cacheClass(requirements.releaseCacheClass())     // For class based degrade (no CME)
        .cmAction(requirements.cmAction())
        .mcid(allocateMCID(requirements));
}

NodeCacheSetter::AllocResult NodeCacheSetter::allocateCache(const TensorPtr& tensor, const Requirement& requirements)
{
    auto key = cacheKey(tensor);
    auto res = m_allocator.allocate(key, requirements.capacity);

    if (!res.successful && GCFG_ENABLE_LB_CACHE_YIELDING.value())
    {
        LOG_DEBUG(LB_CACHE_MNGR,
                  "    Allocation failed. Missing {}B ({:.2}MB). Trying to yield some living buffers.",
                  res.missingCapacity,
                  bToMb(res.missingCapacity));
        if (tryYielding(res.missingCapacity))
        {
            res = m_allocator.allocate(key, requirements.capacity);
        }
    }

    if (res.successful)
    {
        m_cacheState.addAccess(key, m_currentNodeIdx);
    }
    return res;
}

TensorPtr NodeCacheSetter::cacheKey(const TensorPtr& tensor) const
{
    const auto& properties = m_db.slices.at(tensor).properties;
    if (properties.realSlice)
    {
        return properties.realSlice;
    }
    else
    {
        return tensor;
    }
}

LogicalMcid NodeCacheSetter::allocateMCID(const Requirement& requirements)
{
    switch (requirements.cmAction())
    {
        case CacheMaintenanceAction::DEGRADE:
            return m_graph.getCodeGenerator()->getNextMCID(MCIDGenerator::DEGRADE);
        case CacheMaintenanceAction::DISCARD:
            return m_graph.getCodeGenerator()->getNextMCID(MCIDGenerator::DISCARD);
        default:
            return 0;
    }
}

void NodeCacheSetter::releaseAll(const TensorSet& tensors)
{
    std::unordered_set<TensorPtr> uniqueKeys;

    // In case a node accesses the same entry several times (e.g. inputs that are aliases of one another), need to free
    // just once.
    for (const TensorPtr& t : tensors)
    {
        uniqueKeys.insert(cacheKey(t));
    }

    for (const TensorPtr& key : uniqueKeys)
    {
        releaseByCacheKey(key);
    }
}

void NodeCacheSetter::releaseByCacheKey(const TensorPtr& key)
{
    bool res = m_allocator.free(key);
    HB_ASSERT(res, "Unexpected free of uncached tensor: {}", key->getName());
}

void NodeCacheSetter::ensureDependencies(const std::set<size_t>& depIndices)
{
    LOG_DEBUG(LB_CACHE_MNGR,
              "Ensuring dependencies of {} with nodes in indices: [{}]",
              m_currentNode->getNodeName(),
              toString(depIndices, ','));

    // Traversing bwd to increase the chance that adding ctrl-edge with later nodes will result in preventing the need
    // to add ctrl-edges to earlier nodes
    for (auto it = depIndices.rbegin(); it != depIndices.rend(); it++)
    {
        const NodePtr& blocking = m_nodes.at(*it);
        ensureDependency(blocking);
    }
}
void NodeCacheSetter::ensureDependency(const NodePtr& blocker)
{
    LOG_DEBUG(LB_CACHE_MNGR, "Ensuring order {}->{}", blocker->getNodeName(), m_currentNode->getNodeName());
    if (!skipAddingSync(blocker))
    {
        LOG_DEBUG(LB_CACHE_MNGR, "  Non deterministic order - adding ctrl edge");
        m_graph.addControlDependency(blocker, m_currentNode, Tensor::ControlEdgeType::SYNC);
    }
}

bool NodeCacheSetter::skipAddingSync(const NodePtr& blocker) const
{
    enum SyncSkippingMode
    {
        SYNC_ALL = 0,
        SKIP_SAME_ENGINE,
        RESERVED,
        SKIP_ALL,
    };

    // No need to add sync for data dependency blocker
    if (m_graph.isAncestor(blocker, m_currentNode)) return true;

    switch (GCFG_LAYERED_BRAIN_CACHE_THRESHING_PREVENTION_MODE.value())
    {
        case SYNC_ALL:
            return false;  // Sync all => skip none
        case SKIP_SAME_ENGINE:
            return sameEngine(blocker, m_currentNode);
        case SKIP_ALL:
            return true;
        default:
            HB_ASSERT(false,
                      "Unsupported cache threshing prevention mode: {}",
                      GCFG_LAYERED_BRAIN_CACHE_THRESHING_PREVENTION_MODE.value());
    }
}

bool NodeCacheSetter::sameEngine(const NodePtr& a, const NodePtr& b) const
{
    if (m_graph.runsOnMME(a) && m_graph.runsOnMME(b)) return true;
    if (m_graph.runsOnTPC(a) && m_graph.runsOnTPC(b)) return true;
    // If more logical engines are used (DMA, XPS, ...) they can be added here. For now assuming everything runs on MME
    // or TPC
    HB_ASSERT(m_graph.runsOnMME(a) || m_graph.runsOnTPC(a), "Unexpected engine for node {}", a->getNodeName());
    HB_ASSERT(m_graph.runsOnMME(b) || m_graph.runsOnTPC(b), "Unexpected engine for node {}", b->getNodeName());
    return false;
}

void NodeCacheSetter::initAnnotations(NodePtr& node)
{
    initCMDContainer(node->getNodeAnnotation().inputsCacheMetaData, node->getNumInputs());
    initCMDContainer(node->getNodeAnnotation().outputsCacheMetaData, node->getNumOutputs());
}

void NodeCacheSetter::initCMDContainer(std::vector<CacheMetaData>& operandsCMDVec, size_t numOperands)
{
    operandsCMDVec.resize(numOperands, defaultCMD());
}

CacheMetaData NodeCacheSetter::defaultCMD()
{
    CacheMetaData ret {};
    ret.cacheDirective = CacheDirective::NoAllocate;
    return ret;
}

void NodeCacheSetter::registerYieldingOption(size_t& inputIdx, Requirement& req)
{
    if (!GCFG_ENABLE_LB_CACHE_YIELDING.value()) return;

    const TensorPtr& key = cacheKey(m_currentNode->getInput(inputIdx));
    if (req.yieldAllowed())
    {
        m_yieldQueue.addCandidate(m_currentNode, key, inputIdx, req);
    }
    else
    {
        m_yieldQueue.erase(key);
    }
}

bool NodeCacheSetter::tryYielding(uint64_t requiredCapacity)
{
    const auto& currentThread = m_currentNode->getNodeAnnotation().bundleInfo->threadIndex;
    if (!currentThread) return false;  // Can't yield without knowing the current thread

    auto options = yieldingOptions(*currentThread);

    const auto& potential = options.availableCapacity;
    LOG_DEBUG(LB_CACHE_MNGR,
              "    > Yielding potential in thread-{}: {} ({:.2}MB), required: {} ({:.2}MB), so {}yielding.",
              *currentThread,
              potential,
              bToMb(potential),
              requiredCapacity,
              bToMb(requiredCapacity),
              potential < requiredCapacity ? "NOT " : "");
    if (potential < requiredCapacity)
    {
        return false;  // Not enough capacity to yield
    }

    yield(requiredCapacity, options);
    return true;
}

CacheYieldQueue::YieldingOptions NodeCacheSetter::yieldingOptions(size_t currentThread) const
{
    if (currentThread < m_pipelineDepth) return {};  // Can't yield so soon.

    const auto maxYieldingThread = currentThread - m_pipelineDepth;
    auto       yieldingOptions =
        m_yieldQueue.yieldingOptions([&](const auto& candidate) { return candidate.threadIdx <= maxYieldingThread; });
    return yieldingOptions;
}

void NodeCacheSetter::yield(uint64_t requiredCapacity, const CacheYieldQueue::YieldingOptions& options)
{
    uint64_t yieldedCapacity = 0;
    for (const auto& key : options.sortedYieldingKeys)
    {
        if (yieldedCapacity >= requiredCapacity) break;

        LOG_TRACE(LB_CACHE_MNGR, "    >> Yielding tensor: {}", key->getName());

        const auto& candidate = m_yieldQueue.get(key);

        yieldedCapacity += m_cacheState.capacity(key);
        executeYieldRelease(candidate);

        LOG_DEBUG(LB_CACHE_MNGR,
                  "    >> Yielded tensor: {}, tensor capacity: {}, total so far: {}.",
                  candidate.cacheKey->getName(),
                  m_cacheState.capacity(key),
                  yieldedCapacity);

        m_yieldQueue.erase(key);
    }
    HB_ASSERT(yieldedCapacity >= requiredCapacity,
              "Unexpected failure to yield enough cache. Yielded: {}, required: {}",
              yieldedCapacity,
              requiredCapacity);
}

void NodeCacheSetter::executeYieldRelease(const CacheYieldQueue::Candidate& candidate)
{
    const auto& recipe = candidate.releaseRecipe;

    NodeAccessCacheSetter accessSetter(recipe.node);
    accessSetter.input(recipe.inputIdx);
    setCacheAccessRelease(recipe.requirement, accessSetter);
    accessSetter.set();

    releaseByCacheKey(candidate.cacheKey);
}

void NodeCacheSetter::logNodeSummary(const CachingResult& result) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(LB_CACHE_MNGR)) return;

    LOG_DEBUG(LB_CACHE_MNGR, "Node {} cache accesses:", m_currentNode->getNodeName());
    size_t idx = 0;
    for (const auto& cmd : m_currentNode->getNodeAnnotation().inputsCacheMetaData)
    {
        LOG_DEBUG(LB_CACHE_MNGR,
                  "  inputCMD[{}]: directive: {}, class: {}, action: {}, MCID: {}",
                  idx++,
                  cmd.cacheDirective,
                  cmd.cacheClass,
                  cmd.cmAction,
                  cmd.mcid);
    }
    idx = 0;
    for (const auto& cmd : m_currentNode->getNodeAnnotation().outputsCacheMetaData)
    {
        LOG_DEBUG(LB_CACHE_MNGR,
                  "  outputCMD[{}]: directive: {}, class: {}, action: {}, MCID: {}",
                  idx++,
                  cmd.cacheDirective,
                  cmd.cacheClass,
                  cmd.cmAction,
                  cmd.mcid);
    }
    LOG_DEBUG(LB_CACHE_MNGR, "  ====");
    if (result.dependencies.empty())
    {
        LOG_DEBUG(LB_CACHE_MNGR, "  No dependencies");
    }
    else
    {
        for (const auto& depIdx : result.dependencies)
        {
            LOG_DEBUG(LB_CACHE_MNGR, "  Dependency: idx: {}, name: {}", depIdx, m_nodes[depIdx]->getNodeName());
        }
    }
}