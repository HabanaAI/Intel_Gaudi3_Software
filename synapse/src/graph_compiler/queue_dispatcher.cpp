#include "queue_dispatcher.h"

#include "habana_graph.h"
#include "habana_nodes.h"
#include "hal_reader/hal_reader.h"

QueueDispatcher::QueueDispatcher(const std::string& dispatcherName, bool enableRoundRobin)
: m_activeQueues(0), m_workEngIdx(0), m_roundRobinMode(enableRoundRobin), m_dispatcherName(dispatcherName), m_sendSyncEventsMask(0)
{
}

QueueDispatcher::~QueueDispatcher()
{
}

void QueueDispatcher::addEmptyJob(const NodePtr& n, uint32_t pipeLevel, CommandQueuePtr queue, bool isLastPipelineLevel)
{
    // In ARC mode 3 we cannot signal from the QMAN and thus must handle empty job from the engine itself
    HB_ASSERT(!GCFG_ARC_ARCHITECTURE.value(), "default implementation of addEmptyJob is prohibited in ARC architecture");

    std::vector<QueueCommandPtr> cmds;
    uint32_t currentQueue = queue->GetEngineIndex();
    const auto& pipelineSyncs = n->getNodeAnnotation().syncScheme[currentQueue].pipelineSyncs[pipeLevel];
    std::shared_ptr<SyncObject> sync = pipelineSyncs.sync;
    if (n->getNodeAnnotation().prevSyncId[pipeLevel][currentQueue].is_set() && !m_emptyJobHardStopAdded[currentQueue])
    {
        cmds.push_back(queue->getCommandFactory().getWaitForSemaphore(
            n->getNodeAnnotation().prevSyncId[pipeLevel][currentQueue].value(),
            queue->getFirstFenceMonitorId(),
            MONITOR_SO_OP_EQ,
            n->getNodeAnnotation().prevSyncVal[pipeLevel][currentQueue],
            Settable<uint8_t>(),
            ID_0));
    }
    m_emptyJobHardStopAdded[currentQueue] = true;

    for (const auto& monitor : pipelineSyncs.monitors)
    {
        queue->getMonitorCmds(monitor, cmds);
    }

    for (const auto& cpSync : pipelineSyncs.cpSyncs)
    {
        queue->getSignalCmds(cpSync, cmds);
    }

    if (sync.get())
    {
        cmds.push_back(queue->getCommandFactory().getSignalSemaphore(sync->id, sync->value, sync->operation, sync->barrier));
    }
    for (auto& cmd : cmds)
    {
        queue->PushBack(std::move(cmd), false);
    }
}

void QueueDispatcher::init(uint32_t numEngines, uint32_t activeEnginesMask, HabanaGraph* g)
{
    unsigned index = 0;
    for (unsigned id = 0 ; id < numEngines ; ++id)
    {
        if ((activeEnginesMask & (1 << id)))
        {
            CommandQueuePtr q(createCommandQueue(id, index, g));
            HB_ASSERT(q != nullptr, "{}: failed to allocate queue for {}, engine {} index {}",
                      __func__, m_dispatcherName, id, index);
            m_queues.push_back(q);
            index++;
        }
    }
    m_emptyJobHardStopAdded.assign(m_queues.size(), false);
    m_baseRegsCache.resize(g->getHALReader()->getBaseRegistersCacheSize(), std::numeric_limits<uint64_t>::max());
}

const CommandQueuePtr& QueueDispatcher::getQueue(unsigned engineIndex)
{
    HB_ASSERT(engineIndex < getNumEngines(), "engine id out of range");
    return m_queues[engineIndex];
}

unsigned QueueDispatcher::GetBinarySize(bool isSetup)
{
    unsigned totalBinarySize = 0;

    for (auto q : m_queues)
    {
        totalBinarySize += q->GetBinarySize(isSetup);
    }

    return totalBinarySize;
}

unsigned QueueDispatcher::GetBinarySize(unsigned engineIndex, bool isSetup)
{
    HB_ASSERT(engineIndex < getNumEngines(), "engine id out of range");
    return m_queues[engineIndex]->GetBinarySize(isSetup);
}

unsigned QueueDispatcher::GetAllCpDmaCmdSize()
{
    unsigned totalSize = 0;

    for (auto q : m_queues)
    {
        totalSize += q->GetCpDmaCmdSize();
    }

    return totalSize;
}

unsigned QueueDispatcher::GetCpDmaCmdSize(unsigned engineIndex)
{
    HB_ASSERT(engineIndex < getNumEngines(), "engine id out of range");
    return m_queues[engineIndex]->GetCpDmaCmdSize();
}

void QueueDispatcher::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GC)) return;

    unsigned i = 0;
    for (const auto& q : m_queues)
    {
        LOG_DEBUG(GC, "   Engine{}:", i++);
        q->Print();
    }
}

void QueueDispatcher::dispatchNode(const pNode& n, HabanaGraph* g, bool isSetup)
{
    // Update base registers cache if this node has something to update
    if (n->getNodeAnnotation().baseRegsCacheUpdate.size())
    {
        // Update local cache snapshot to be used later by the descriptor loading function (addLoadDesc)
        for (const auto& cacheEntry : n->getNodeAnnotation().baseRegsCacheUpdate)
        {
            m_baseRegsCache.at(cacheEntry.indexInCache) = cacheEntry.sectionID;
        }

        // Push to all physical queues commands (with their patch-points) that load the new entries to the hardware
        for (auto q : m_queues)
        {
            q->loadBaseRegsCacheUpdate(n);
        }

        LOG_DEBUG(BASE_REGS_CACHE, "Node {} is Cache Updater (dispatcher: {})", n->getNodeName(), m_dispatcherName);
        if (log_level_at_least(synapse::LogManager::LogType::BASE_REGS_CACHE, 0))
        {
            LOG_TRACE(BASE_REGS_CACHE, "  Cache snapshot at this node (entry #: sectionID):");
            for (auto i = 0; i < m_baseRegsCache.size(); i++)
            {
                LOG_TRACE(BASE_REGS_CACHE, "    entry {}: 0x{:x}", i, m_baseRegsCache[i]);
            }
        }
    }

    // on top of per-engine dispatching any engine can have a delay added
    unsigned waitCycles = n->getNodeAnnotation().waitCycles;
    if (waitCycles)
    {
        for (auto q : m_queues)
        {
            q->AddSuspend(waitCycles);
        }
    }
}

void QueueDispatcher::addPreSyncScheme(const pNode& n, bool isSetup)
{
    for (unsigned i = 0 ; i < m_queues.size() ; ++i)
    {
        if (m_queues[i]->isQueueActive())
        {
            m_queues[i]->addPreSyncScheme(n, isSetup);
        }
    }
}

void QueueDispatcher::addPostSyncScheme(const pNode& n, bool isSetup)
{
    for (unsigned i = 0 ; i < m_queues.size() ; ++i)
    {
        if (m_queues[i]->isQueueActive())
        {
            m_queues[i]->addPostExeSyncs(n, isSetup);
        }
    }
}
