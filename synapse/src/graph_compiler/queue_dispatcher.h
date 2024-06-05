#pragma once

#include <vector>
#include "types.h"
#include "queue_command.h"
#include "command_queue.h"
#include "llvm/small_vector.h"

class CommandQueue;
class HabanaGraph;

using llvm_vecsmall::SmallVectorImpl;

class QueueDispatcher
{
public:
    // In case of enableRoundRobin = true, make sure that setDescriptorSignaling doesn't happen during descriptor
    // generation, but during dispatch
    QueueDispatcher(const std::string& dispatcherName, bool enableRoundRobin = false);

    virtual ~QueueDispatcher();

    virtual void init(uint32_t numEngines, uint32_t activeEnginesMask, HabanaGraph* g);

    unsigned                            getNumEngines() const { return m_queues.size(); }
    const CommandQueuePtr&              getQueue(unsigned engineIndex);
    const std::vector<CommandQueuePtr>& getQueues() { return m_queues; };

    /* CommandQueue API */
    virtual unsigned GetBinarySize(bool isSetup = true);
    virtual unsigned GetBinarySize(unsigned engineIndex, bool isSetup = true);
    virtual unsigned GetAllCpDmaCmdSize();
    virtual unsigned GetCpDmaCmdSize(unsigned engineIndex);
    virtual void     Print() const;
    virtual void     dispatchNode(const pNode& n, HabanaGraph* g, bool isSetup);

protected:
// Shorthand for Pipeline Descriptors Map: rows are pipeline level, columns are physical engine
#define PipelineDescMap std::vector<std::vector<Settable<DescriptorWrapper<DescType>>>>

    virtual CommandQueue* createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g) = 0;

    virtual void updateEmptyJobDescWrapper(void* wrapper) {}

    template<class DescType>
    void fillPipelineMapping(unsigned                                            numDescs,
                             const SmallVectorImpl<DescriptorWrapper<DescType>>& descriptorsWrappers,
                             PipelineDescMap&                                    pipelineDescriptors,
                             bool                                                gotExecutionInfo              = true,
                             bool                                                allowManyDescForEngInPipeline = false,
                             DescType*                                           pEmptyJobDesc = nullptr);

    template<class DescType>
    DescriptorWrapper<DescType> createWrapperForEmptyJobDescriptor(DescType* pEmptyJobDesc = nullptr);

    template<class DescType>
    void setInactiveQueueState(PipelineDescMap& pipelineDescriptors);

    void addPreSyncScheme(const pNode& n, bool isSetup);
    void addPostSyncScheme(const pNode& n, bool isSetup);

    template<class DescType>
    void addDescsToNode(const pNode& n, PipelineDescMap& pipelineDescriptors, bool isSetup);

    virtual void addEmptyJob(const NodePtr& n, uint32_t pipeLevel, CommandQueuePtr queue, bool isLastPipelineLevel);

    template<class DescType>
    void assignPredicateToDescriptors(PipelineDescMap& pipelineDescriptors);

    std::vector<CommandQueuePtr>  m_queues;
    uint32_t                      m_activeQueues;
    uint32_t                      m_workEngIdx;
    bool                          m_roundRobinMode;
    std::vector<bool>             m_emptyJobHardStopAdded;
    const std::string             m_dispatcherName;
    uint16_t                      m_sendSyncEventsMask;
    std::vector<uint64_t>         m_baseRegsCache; // list of cache resident section IDs
};


template <typename T>
T* downcaster(QueueDispatcher* qd)
{
    T* ptr = dynamic_cast<T*>(qd);
    HB_ASSERT_PTR(ptr);
    return ptr;
}

template <typename T>
const T* downcaster(const QueueDispatcher* qd)
{
    const T* ptr = dynamic_cast<const T*>(qd);
    HB_ASSERT_PTR(ptr);
    return ptr;
}

template<class DescType>
DescriptorWrapper<DescType> QueueDispatcher::createWrapperForEmptyJobDescriptor(DescType* pEmptyJobDesc)
{
    DescriptorWrapper<DescType> emptyJobDescWrap(*pEmptyJobDesc);

    updateEmptyJobDescWrapper(&emptyJobDescWrap);

    return emptyJobDescWrap;
}

template<class DescType>
void QueueDispatcher::fillPipelineMapping(unsigned                                            numDescs,
                                          const SmallVectorImpl<DescriptorWrapper<DescType>>& descriptorsWrappers,
                                          PipelineDescMap&                                    pipelineDescriptors,
                                          bool                                                gotExecutionInfo,
                                          bool      allowManyDescForEngInPipeline,
                                          DescType* pEmptyJobDesc)
{
    if (allowManyDescForEngInPipeline)
    {
        HB_ASSERT(gotExecutionInfo, "execution info must be provided when multiple desc per engine is allowed");
    }

    for (unsigned descIndex = 0 ; descIndex < numDescs ; ++descIndex)
    {
        const DescriptorWrapper<DescType>& descWrapper = descriptorsWrappers[descIndex];

        unsigned pipelineLevel;
        if (gotExecutionInfo)
        {
            const ExecutionInfo& execInfo = descWrapper.getExecutionInfo();
            pipelineLevel = execInfo.pipelineLevel;
        }
        else
        {
            pipelineLevel = descIndex / getNumEngines();
        }

        if ((pipelineLevel + 1) > pipelineDescriptors.size())
        {
            pipelineDescriptors.resize(pipelineLevel + 1);
        }
        pipelineDescriptors[pipelineLevel].emplace_back(descWrapper);
    }

    for (std::vector<Settable<DescriptorWrapper<DescType>>>& pipelineDesc : pipelineDescriptors)
    {
        HB_ASSERT(pipelineDesc.size() <= getNumEngines() || allowManyDescForEngInPipeline,
                  "Engine is expected to work only once in pipeline level");

        unsigned expectedSize = round_to_multiple(pipelineDesc.size(), getNumEngines());

        // If we were given descriptor for empty job, fill the missing entries up to the next multiple of num
        // engines with that descriptor; otherwise, just fill the missing entries with empty placeholders.
        if (pEmptyJobDesc)
        {
            while (pipelineDesc.size() < expectedSize)
            {
                pipelineDesc.emplace_back(createWrapperForEmptyJobDescriptor(pEmptyJobDesc));
            }
        }
        else
        {
            pipelineDesc.resize(expectedSize);  // fill with placeholders only
        }
    }
}

template<class DescType>
void QueueDispatcher::setInactiveQueueState(PipelineDescMap& pipelineDescriptors)
{
    if (m_activeQueues == m_queues.size()) return; // Nothing to do

    uint32_t descriptorSet = m_workEngIdx;
    for (const auto& pipeDescs : pipelineDescriptors)
    {
        for (uint32_t descIdx = 0; descIdx < pipeDescs.size(); ++descIdx)
        {
            if (pipeDescs[descIdx].is_set())
            {
                uint32_t queueIdx = (m_roundRobinMode? descriptorSet++ : descIdx) % m_queues.size();
                std::shared_ptr<CommandQueue>& queue = m_queues[queueIdx];
                if (!queue->isQueueActive())
                {
                    // Just set the queue state: active/inactive. Init commands were already handled
                    queue->setQueueAsActive();

                    if (++m_activeQueues == m_queues.size()) return;
                }
            }
        }
    }
}

template<class DescType>
void QueueDispatcher::addDescsToNode(const pNode& n, PipelineDescMap& pipelineDescriptors, bool isSetup)
{
    unsigned numEngines = getNumEngines();

    for (unsigned pipeLevel = 0 ; pipeLevel < pipelineDescriptors.size() ; ++pipeLevel)
    {
        std::vector<Settable<DescriptorWrapper<DescType>>>& pipeDescs    = pipelineDescriptors[pipeLevel];
        unsigned engineRounds = div_round_up(pipeDescs.size(), numEngines);
        unsigned setElements = std::count_if(pipeDescs.begin(),
                                             pipeDescs.end(),
                                             [](Settable<DescriptorWrapper<DescType>> k){return k.is_set();});
        unsigned descIdx = 0;

        // check if it's the last descriptor in node
        bool isLastPipelineLevel = (pipeLevel == pipelineDescriptors.size() - 1);

        // In some scenarios (e.g rotator) there might be more descriptors than engines for a single ROI.
        // Therefore we split the descriptors to engines in rounds.
        // Only descriptors in first round should monitor for previous completions.
        // Only last set of #engines descriptors should signal completion.
        for (unsigned engineRoundIdx = 0 ; engineRoundIdx < engineRounds; ++engineRoundIdx)
        {
            if (!m_roundRobinMode)
            {
                // If not working in round robin mode, reset the start working index to 0
                m_workEngIdx = 0;
            }
            // first round of descriptors should have monitors
            bool isFirstInEnginePerLevel = (engineRoundIdx == 0);

            // Num of descriptors might be larger then num of engines
            unsigned numEnginesToCover = std::min<unsigned>(pipeDescs.size(), numEngines);
            uint32_t runningEngIdx = m_workEngIdx;
            for (unsigned engineIdx = 0 ; engineIdx < numEnginesToCover ; ++engineIdx, ++descIdx)
            {
                // last set of #engines descriptors should signal
                bool isLastInEnginePerLevel = (numEngines > setElements) || (descIdx >= (setElements - numEngines));
                uint32_t currentQueue = pipeDescs[descIdx].is_set() ? m_workEngIdx : runningEngIdx;
                runningEngIdx = (runningEngIdx + 1) % numEngines;
                DescCommandQueue<DescType>* queue = downcaster<DescCommandQueue<DescType>>(m_queues[currentQueue]);
                assert(queue->GetEngineIndex() == currentQueue);

                if (pipeDescs[descIdx].is_set())
                {
                    LOG_TRACE(QMAN, "{}: AddPartialNode desc Idx {} to engine index {} of Node {},"
                                    "first engine per level {}, last engine per level {}",
                                    m_dispatcherName, descIdx, currentQueue, n->getNodeName(),
                                    isFirstInEnginePerLevel, isLastInEnginePerLevel);

                    DescriptorWrapper<DescType>& descWrapper = pipeDescs[descIdx].value();

                    queue->AddPartialNode(n,
                                          descWrapper,
                                          &pipeDescs,
                                          pipeLevel,
                                          isSetup,
                                          m_baseRegsCache,
                                          isLastPipelineLevel,
                                          nullptr,  // preSyncCmds
                                          nullptr,  // postSyncCmds
                                          isFirstInEnginePerLevel,
                                          isLastInEnginePerLevel);

                    m_emptyJobHardStopAdded[currentQueue] = false;
                    m_workEngIdx = runningEngIdx;
                }
                else
                {
                    if (engineRounds > 1) continue;  // engine got a desc in previous round, no need to add an empty job

                    // For legacy sync scheme, make sure sync was prepared by the sync scheme pass; otherwise, continue
                    std::vector<SyncInteraction>& syncScheme = n->getNodeAnnotation().syncScheme;
                    if (!GCFG_ARC_ARCHITECTURE.value() && (queue->GetEngineIndex() < syncScheme.size()))
                    {
                        const SyncInteraction& engineSyncScheme = syncScheme[queue->GetEngineIndex()];
                        if (pipeLevel < engineSyncScheme.pipelineSyncs.size())
                        {
                            const auto&                 pipeSyncs = engineSyncScheme.pipelineSyncs[pipeLevel];
                            std::shared_ptr<SyncObject> sync      = pipeSyncs.sync;

                            if (!sync.get() && pipeSyncs.monitors.empty() && pipeSyncs.cpSyncs.empty()) continue;
                        }
                    }
                    addEmptyJob(n, pipeLevel, m_queues[currentQueue], isLastPipelineLevel);
                }
            }
        }
    }
}

template<class DescType>
void QueueDispatcher::assignPredicateToDescriptors(PipelineDescMap& pipelineDescriptors)
{
    uint32_t descriptorSet = m_workEngIdx;
    for (unsigned pipeLevel = 0; pipeLevel < pipelineDescriptors.size(); ++pipeLevel)
    {
        std::vector<Settable<DescriptorWrapper<DescType>>>& pipeDescs = pipelineDescriptors[pipeLevel];
        for (unsigned descIdx = 0; descIdx < pipeDescs.size(); ++descIdx)
        {
            if (!pipeDescs[descIdx].is_set()) continue;
            DescriptorWrapper<DescType> descWrapper = pipeDescs[descIdx].value();
            uint32_t                    queueIdx    = (m_roundRobinMode ? descriptorSet++ : descIdx) % m_queues.size();
            DescCommandQueue<DescType>* queue       = downcaster<DescCommandQueue<DescType>>(m_queues[queueIdx]);
            // Predicate is assigned according to the engine ID + 1 (predicate #0 is reserved).
            // For example TPC0 should have predicate #1, TPC1 should have predicate #2 and so on.
            // If for instance TPC3 is not working in the chip, it's predicate should not be used either; thus,
            // TPC4 should still have predicate #5.
            descWrapper.getExecutionInfo().predicate = queue->GetEngineID() + 1;
            pipeDescs[descIdx].set(descWrapper);
        }
    }
}

struct QueueDispatcherParams
{
    unsigned parallelLevel;
    unsigned index;
    QueueDispatcherParams(unsigned p, unsigned i) : parallelLevel(p), index(i) {}
    QueueDispatcherParams() : parallelLevel(0), index(0) {}
    bool operator<(const QueueDispatcherParams& b) const
    {
        return (parallelLevel == b.parallelLevel) ? index < b.index : parallelLevel < b.parallelLevel;
    }
};
using QueueDispatcherPtr = std::shared_ptr<QueueDispatcher>;
using QueueDispatcherMap = std::map<QueueDispatcherParams, QueueDispatcherPtr>; // used for multi DMA dispatchers where key is parallel level
