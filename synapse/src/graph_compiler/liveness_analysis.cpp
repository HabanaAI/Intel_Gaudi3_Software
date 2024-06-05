#include "liveness_analysis.h"

#include "allocators_utils.h"
#include "habana_graph.h"

#include "tensor.h"

#include <tuple>

static LivenessAnalysis::TensorLifetimeMap GetTensorLifetimes(const HabanaGraph&          graph,
                                                              const TensorsCompatibility& tensorsCompatibility)
{
    LOG_TRACE(LIVA, "{}", HLLOG_FUNC);

    LivenessAnalysis::TensorLifetimeMap tensorLifetimes;
    uint32_t                            maxIdx = graph.getMaxExecutionOrderedIndex();
    uint32_t                            minIdx = graph.getMinExecutionOrderedIndex();
    for (const pNode& node : graph.getExeSortedNodes())
    {
        for (const auto* tensorVec : {&node->getInputs(), &node->getOutputs()})
        {
            for (auto t : *tensorVec)
            {
                GET_REAL_TENSOR_IF_NULL_CONTINUE(t);
                if (!tensorsCompatibility(t)) continue;

                const auto idx = node->getExecutionOrderedIndex();

                LivenessAnalysis::TensorLifetimeMap::iterator it;
                bool                                          inserted;
                std::tie(it, inserted) = tensorLifetimes.emplace(t, Lifetime {idx, idx});
                if (!inserted)
                {
                    it->second.m_end = idx;
                }
                if (t->isPersistent())
                {
                    // output tensor lifetime [producer - maxIdx]
                    // input  tensor lifetime [0 - maxIdx]
                    it->second.m_end = maxIdx;
                    if (graph.getNumberOfTensorProducers(t) == 0)
                    {
                        it->second.m_start = minIdx;
                    }
                }
            }
        }
    }
    return tensorLifetimes;
}

std::pair<uint64_t, uint64_t>
LivenessAnalysis::GetMaxCapacity(size_t nodeCount, const LivenessAnalysis::TensorLifetimeMap& tensorLifetimes)
{
    LOG_TRACE(LIVA, "{}", HLLOG_FUNC);

    if (!nodeCount) return {};

    m_usedMem.resize(nodeCount);
    for (const auto& v : tensorLifetimes)
    {
        const uint64_t space = getWriteSpaceForTensor(v.first);
        for (uint32_t i = v.second.m_start; i <= v.second.m_end; ++i)
        {
            m_usedMem[i] += space;
        }
    }
    auto     max_elem       = std::max_element(begin(m_usedMem), end(m_usedMem));
    uint64_t maxCapacityIdx = std::distance(begin(m_usedMem), max_elem);

    return std::pair(*max_elem, maxCapacityIdx);
}

static bool operator<(const Lifetime& a, const Lifetime& b)
{
    using T = std::pair<uint64_t, uint64_t>;
    return T(a.m_start, a.m_end) < T(b.m_start, b.m_end);
}

bool lifetimeIntersectsWith(const Lifetime& lifetime, const Lifetime& otherLifetime)
{
    uint32_t maxStart = std::max(lifetime.m_start, otherLifetime.m_start);
    uint32_t minEnd   = std::min(lifetime.m_end, otherLifetime.m_end);

    if (maxStart > minEnd) return false;  // no intersection
    return true;
}

// print maxCapacity and tensors lifetimes during maxCapacity sorted by size
static void debugPrint(const LivenessAnalysis::TensorLifetimeMap& lifetimes,
                       unsigned                                   nodeCount,
                       uint64_t                                   maxCapacity,
                       uint64_t                                   maxCapacityIdx)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(LIVA)) return;

    using T = std::pair<TensorPtr, Lifetime>;

    if (maxCapacity==0) return;

    std::vector<T> sorted;
    std::copy_if(lifetimes.begin(), lifetimes.end(), std::back_inserter(sorted), [maxCapacityIdx](const T& v) {
        return (maxCapacityIdx >= v.second.m_start  && v.second.m_end >= maxCapacityIdx);
    });
    std::sort(begin(sorted), end(sorted), [](const T& a, const T& b) {
        using A = std::pair<uint64_t, Lifetime>;

        return A(getWriteSpaceForTensor(a.first), a.second) > A(getWriteSpaceForTensor(b.first), b.second);
    });

    std::vector<std::string> lines;
    lines.reserve(sorted.size());
    std::transform(begin(sorted), end(sorted), std::back_inserter(lines), [](const T& v) {
        return fmt::format("\t\"{}\": [{},{}] aligned size Bytes {}", v.first->getName(), v.second.m_start, v.second.m_end, getWriteSpaceForTensor(v.first));
    });

    LOG_TRACE(LIVA,
              "LivenessAnalysis set up with nodeCount {}, maxCapacity {} maxCapacityIdx {} and lifetimes:\n{}",
              nodeCount,
              maxCapacity,
              maxCapacityIdx,
              fmt::join(begin(lines), end(lines), "\n"));
}

LivenessAnalysis::LivenessAnalysis(const HabanaGraph* graph, const TensorsCompatibility& tensorsCompatibility)
: m_nodeCount(graph->getNumNodes()), m_tensorLifetimeMap(GetTensorLifetimes(*graph, tensorsCompatibility))
{
    std::tie(m_maxCapacity, m_maxCapacityIdx) = GetMaxCapacity(m_nodeCount, m_tensorLifetimeMap);
    debugPrint(m_tensorLifetimeMap, m_nodeCount, m_maxCapacity, m_maxCapacityIdx);
}

TensorSetVector LivenessAnalysis::getLiveTensorsEntries() const
{
    LOG_TRACE(LIVA, "{}", HLLOG_FUNC);

    TensorSetVector res(m_nodeCount);

    for (const auto& v : m_tensorLifetimeMap)
    {
        pTensor t = v.first;
        GET_REAL_TENSOR_IF_NULL_CONTINUE(t);

        for (uint32_t i = v.second.m_start; i <= v.second.m_end; ++i)
        {
            res[i].insert(t);
        }
    }

    return res;
}

bool LivenessAnalysis::wasRealTensorEncounteredBeforeNode(const pNode& node, pTensor tensor) const
{
    const auto it = m_tensorLifetimeMap.find(tensor);
    HB_ASSERT(it != m_tensorLifetimeMap.end(), "Unknown tensor (\"{}\")", tensor->getName());

    return it->second.m_start < node->getExecutionOrderedIndex();
}

bool LivenessAnalysis::isRealTensorAliveAfterNode(const pNode& node, pTensor tensor) const
{
    GET_REAL_TENSOR_IF_NULL_RETURN_VAL(tensor, false);

    const auto it = m_tensorLifetimeMap.find(tensor);
    HB_ASSERT(it != m_tensorLifetimeMap.end(), "Unknown tensor (\"{}\")", tensor->getName());

    const auto  idx      = node->getExecutionOrderedIndex();
    const auto& lifetime = it->second;
    return lifetime.m_start <= idx && idx < lifetime.m_end;
}

Lifetime LivenessAnalysis::getTensorLifeTime(TensorPtr tensor) const
{
    GET_REAL_TENSOR_IF_NULL_RETURN_VAL(tensor, Lifetime({-1, -1}));

    const auto it = m_tensorLifetimeMap.find(tensor);
    HB_ASSERT(it != m_tensorLifetimeMap.end(), "Unknown tensor (\"{}\")", tensor->getName());
    return it->second;
}

const LivenessAnalysis::LiveAndDieTensors& LivenessAnalysis::liveAndDieTensors() const
{
    if (m_liveAndDieTensors.empty())
    {
        m_liveAndDieTensors.resize(m_nodeCount + 1);
        for (const auto& tLife : m_tensorLifetimeMap)
        {
            m_liveAndDieTensors[tLife.second.m_start].m_live.push_back(tLife.first);
            // m_end pointing to the last node in use, the next node is where the tensor is destroyed
            HB_ASSERT(m_liveAndDieTensors.size() > tLife.second.m_end + 1, "index out of bound");
            m_liveAndDieTensors[tLife.second.m_end + 1].m_die.push_back(tLife.first);
        }
        for (auto& liveAndDie : m_liveAndDieTensors)
        {
            std::sort(liveAndDie.m_live.begin(), liveAndDie.m_live.end(), TensorComparator());
            std::sort(liveAndDie.m_die.begin(), liveAndDie.m_die.end(), TensorComparator());
        }
    }

    return m_liveAndDieTensors;
}
