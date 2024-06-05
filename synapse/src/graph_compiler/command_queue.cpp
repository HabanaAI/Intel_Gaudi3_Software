#include "command_queue.h"

#include "compilation_hal_reader.h"
#include "graph_compiler/queue_command.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "infra/defs.h"
#include "infra/settable.h"
#include "monitor_setup_manager.h"
#include "node.h"
#include "node_roi.h"
#include "queue_command_factory.h"
#include "sync_types.h"
#include "tensor.h"
#include "utils.h"

#include <iterator>
#include <list>
#include <memory>

class ParamsManager;
class SyncConventions;

unsigned getMaxTpcEnginesCount()
{
    return CompilationHalReader::getHalReader()->getNumTpcEngines();
}

unsigned getMaxDmaEnginesCount()
{
    return CompilationHalReader::getHalReader()->getNumDmaEngines();
}

unsigned getMaxMmeEnginesCount()
{
    return CompilationHalReader::getHalReader()->getNumMmeEngines();
}

unsigned getMaxRotatorEnginesCount()
{
    return CompilationHalReader::getHalReader()->getNumRotatorEngines();
}

unsigned getMcidBaseRegsFirstIndex()
{
    return CompilationHalReader::getHalReader()->getMcidBaseRegsFirstIndex();
}

//---------------------------------------------------------
//--------------------- CommandQueue ----------------------
//------------------ Base for all queues ------------------
//---------------------------------------------------------

CommandQueue::CommandQueue(const QueueCommandFactory& cmdFactory,
                           unsigned                   queId,
                           HabanaDeviceType           devType)
  : m_commandFactory(cmdFactory),
    m_queueId(queId),
    m_deviceType(devType),
    m_engineId(0),
    m_engineIndex(0),
    m_stream(0),
    m_maxStreams(1),
    m_packetSizeActivate(0),
    m_packetSizeLoad(0),
    m_packetSizeExe(0),
    m_params(nullptr),
    m_activeQueue(false),
    m_sendSyncEvents(false),
    m_armMonBeforeDesc(GCFG_CODE_GEN_ARM_MON_BEFORE_DESC.value()),
    m_qmanMutexRequired(false)
{}

void CommandQueue::Clear()
{
    m_queueLoad.clear();
    m_queueActivate.clear();
    m_queueExe.clear();
    m_initialQueueCommands.clear();
    m_packetSizeActivate = 0;
    m_packetSizeLoad = 0;
    m_packetSizeExe = 0;
}

unsigned CommandQueue::GetBinarySize(bool isSetup, bool isLoad /*= false*/) const
{
    if (isSetup)
    {
        if (isLoad)
        {
            return m_packetSizeLoad;
        }
        else
        {
            return m_packetSizeActivate;
        }
    }
    else
    {
        return m_packetSizeExe;
    }
}

unsigned CommandQueue::Size(bool isSetup, bool isLoad) const
{
    if (isSetup)
    {
        if (isLoad)
        {
            return m_queueLoad.size();
        }
        else
        {
            return m_queueActivate.size();
        }
    }
    else
    {
        return m_queueExe.size();
    }
}

unsigned CommandQueue::GetQueueID() const
{
    return m_queueId;
}

unsigned CommandQueue::GetEngineID() const
{
    return m_engineId;
}

unsigned CommandQueue::GetEngineIndex() const
{
    return m_engineIndex;
}

HabanaDeviceType CommandQueue::GetDeviceType() const
{
    return m_deviceType;
}

unsigned CommandQueue::GetLogicalQueue() const
{
    return -1;
}

unsigned CommandQueue::GetEngineStream() const
{
    return m_stream;
}

bool CommandQueue::Empty() const
{
    return m_queueLoad.size() == 0 && m_queueActivate.size() == 0 && m_queueExe.size() == 0;
}

void CommandQueue::PushBack(std::unique_ptr<QueueCommand> cmd, bool isSetup /*= true*/, bool isLoad /*= false */)
{
    HB_ASSERT_PTR(cmd);

    if (isSetup)
    {
        if (isLoad)
        {
            m_packetSizeLoad += cmd->GetBinarySize();
            m_queueLoad.emplace_back(std::move(cmd));
        }
        else
        {
            m_packetSizeActivate += cmd->GetBinarySize();
            m_queueActivate.emplace_back(std::move(cmd));
        }
    }
    else
    {
        m_packetSizeExe += cmd->GetBinarySize();
        m_queueExe.emplace_back(std::move(cmd));
    }
}


void CommandQueue::SerializePB(gc_recipe::generic_packets_container* pktCon, bool isSetup, bool isLoad /*= false*/) const
{
    auto& queueToUpdate = (isSetup ? (isLoad ? m_queueLoad : m_queueActivate) : m_queueExe);

    if (m_params != nullptr)
    {
        for (auto& p : queueToUpdate)
        {
            p->WritePB(pktCon, m_params);
        }
    }
    else
    {
        for (auto& p : queueToUpdate)
        {
            p->WritePB(pktCon);
        }
    }
}

void CommandQueue::SetParams(ParamsManager* params)
{
    m_params = params;
}

const std::string CommandQueue::getName() const
{
    return fmt::format("{}(stream={})(engineId={})(queueId={})",
                       getDeviceName(m_deviceType),
                       m_stream,
                       m_engineId,
                       m_queueId);
}

void CommandQueue::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;
    LOG_DEBUG(QMAN, "{}:", getName());
    LOG_DEBUG(QMAN, "  Load queue:");
    for (const auto& p : m_queueLoad)
    {
        p->Print();
    }

    LOG_DEBUG(QMAN, "  Activate queue:");
    for (const auto& p : m_queueActivate)
    {
        p->Print();
    }

    LOG_DEBUG(QMAN, "  Execute queue:");
    for (const auto& p : m_queueExe)
    {
        p->Print();
    }
}

QueueCommand& CommandQueue::operator[] (int index)
{
    return *m_queueExe[index];
}

unsigned CommandQueue::GetCpDmaCmdSize() const
{
    return  m_commandFactory.getCpDmaCmdSize();
}

void CommandQueue::setMonIdToSetupMonitors(std::map<unsigned, MonObject>& setupMonitors)
{
    m_monIdToSetupMonitors = setupMonitors;
}

void CommandQueue::setMonIdAsSetup(unsigned monitorId)
{
    HB_ASSERT(m_usedSetupMonitors.find(monitorId) == m_usedSetupMonitors.end(), "Monitor id was already setup");
    m_usedSetupMonitors.insert(monitorId);
}

void CommandQueue::addPreSyncScheme(std::shared_ptr<Node> node, bool isSetup)
{
    const auto& syncScheme = node->getNodeAnnotation().syncScheme;
    if (GetEngineIndex() < syncScheme.size())
    {
        const std::list<SyncOrMonitor>& syncAndMon = syncScheme[GetEngineIndex()].preSyncsAndMon;
        for (const auto& syncOrMonitor : syncAndMon)
        {
            if (syncOrMonitor.type == SyncOrMonitor::SYNC_OBJ)
            {
                addSignal(syncOrMonitor.sync, isSetup);
            }
            else if (syncOrMonitor.type == SyncOrMonitor::MONITOR_OBJ)
            {
                addMonitor(syncOrMonitor.monitor, isSetup);
            }
        }
        auto patchableMonitors = syncScheme[GetEngineIndex()].patchableMonitors;
        for (const PatchableMonitor& patchableMonitor : patchableMonitors)
        {
            BasicFieldsContainerInfo bfci;

            auto syncObject = std::make_shared<SyncObjectAddressFieldInfo>(patchableMonitor.tensorId, node);
            bfci.add(syncObject);

            auto setup = getCommandFactory().getMonitorSetup(patchableMonitor.monObject.id,
                                                             patchableMonitor.monObject.signalSyncId,
                                                             patchableMonitor.monObject.setupValue,
                                                             0,
                                                             patchableMonitor.monObject.shouldInc);
            setup->SetContainerInfo(bfci);

            auto arm = getCommandFactory().getMonitorArm(patchableMonitor.monObject.syncId,
                                                         patchableMonitor.monObject.id,
                                                         patchableMonitor.monObject.operation,
                                                         patchableMonitor.monObject.armValue,
                                                         patchableMonitor.monObject.mask);

            // coupling the setup and arm command to guarantee their order of execution
            std::vector<QueueCommandPtr> cmdVector;
            cmdVector.push_back(std::move(setup));
            cmdVector.push_back(std::move(arm));

            auto setupAndArm = getCommandFactory().getSetupAndArmCommand(cmdVector);

            PushBack(std::move(setupAndArm), isSetup);
        }
    }
}

void CommandQueue::addPostExeSyncs(std::shared_ptr<Node> node, bool isSetup)
{
    const auto& syncScheme = node->getNodeAnnotation().syncScheme;
    if (GetEngineIndex() < syncScheme.size())
    {
        const std::list<SyncOrMonitor>& postSyncsOrMons = syncScheme[GetEngineIndex()].postSyncsAndMons;
        for (const auto& syncOrMon : postSyncsOrMons)
        {
            if (syncOrMon.type == SyncOrMonitor::SYNC_OBJ)
            {
                addSignal(syncOrMon.sync, isSetup);
            }
            else
            {
                addMonitor(syncOrMon.monitor, isSetup);
            }
        }
    }
}

const QueueCommandFactory& CommandQueue::getCommandFactory() const
{
    return m_commandFactory;
}

std::vector<std::pair<std::unique_ptr<QueueCommand>, bool>>& CommandQueue::getInitialQueueCommands()
{
    return m_initialQueueCommands;
}

bool CommandQueue::isQueueActive()
{
    return m_activeQueue;
}

void CommandQueue::setQueueAsActive()
{
    m_activeQueue = true;
}

void CommandQueue::getMonitorCmds(const MonObject&              monitor,
                                  std::vector<QueueCommandPtr>& cmds,
                                  MonitorCommandParts           parts,
                                  Settable<unsigned>            engID)
{
    // 1. Handle setup
    if (parts != MonitorCommandParts::Fence &&  // don't handle setup if the caller requested only the fence command
        m_usedSetupMonitors.find(monitor.id) == m_usedSetupMonitors.end())
    {
        if (m_monIdToSetupMonitors.count(monitor.id) == 0)
        {
            cmds.emplace_back(getCommandFactory().getMonitorSetup(monitor.id,
                                                                  monitor.signalSyncId,
                                                                  monitor.setupValue,
                                                                  0,
                                                                  monitor.shouldInc));
        }
        else
        {
            MonObject setupMon = m_monIdToSetupMonitors[monitor.id];

            HB_ASSERT(setupMon.id == monitor.id, "Monitor id does not match");
            // fence type
            LOG_DEBUG(QMAN,
                      "Setup monitor id {}, using setup monitor id {}, value {} for queue {}",
                      monitor.id,
                      setupMon.id,
                      setupMon.setupValue,
                      getName());
            if (setupMon.signalSyncId == FENCE_MONITOR_ID)
            {
                unsigned streamStop = monitor.predicateTheSetup ? m_maxStreams : 1;
                unsigned predicate  = monitor.predicateTheSetup ? 1 : 0;

                for (unsigned stream = 0; stream < streamStop; stream++)
                {
                    cmds.emplace_back(
                        getCommandFactory().getMonitorSetup(setupMon.id,
                                                            setupMon.fenceId,
                                                            GetDeviceType(),
                                                            engID.is_set() ? engID.value() : GetEngineID(),
                                                            setupMon.setupValue,
                                                            stream,
                                                            predicate + stream,
                                                            setupMon.shouldInc));
                }
            }
            else
            {
                cmds.emplace_back(getCommandFactory().getMonitorSetup(setupMon.id,
                                                                      setupMon.signalSyncId,
                                                                      setupMon.setupValue,
                                                                      0,
                                                                      setupMon.shouldInc));
            }
            setMonIdAsSetup(monitor.id);
        }
    }

    // 2. Handle arm and fence

    // Either they specifically requested only arm, or they wanted arm and fence but there is no fence in this monitor
    if (parts == MonitorCommandParts::SetupArm ||
        (parts == MonitorCommandParts::SetupArmFence && !monitor.fenceTargetVal.is_set()))
    {
        cmds.emplace_back(m_commandFactory.getMonitorArm(monitor.syncId,
                                                         monitor.id,
                                                         monitor.operation,
                                                         monitor.armValue,
                                                         monitor.mask));
    }
    // They wanted arm and fence and there is fence in this monitor
    else if (parts == MonitorCommandParts::SetupArmFence)
    {
        cmds.emplace_back(m_commandFactory.getWaitForSemaphore(monitor.syncId,
                                                               monitor.id,
                                                               monitor.operation,
                                                               monitor.armValue,
                                                               monitor.mask,
                                                               monitor.fenceId,
                                                               monitor.fenceTargetVal.value()));
    }
    // They specifically requested only fence, but we must make sure fence exist
    else if (monitor.fenceTargetVal.is_set())
    {
        HB_ASSERT(parts == MonitorCommandParts::Fence, "bug in logic around MonitorCommandParts");
        cmds.emplace_back(m_commandFactory.getFence(monitor.fenceId, monitor.fenceTargetVal.value()));
    }
}

void CommandQueue::getSignalCmds(const SyncObject& sync, std::vector<QueueCommandPtr>& cmds)
{
    cmds.emplace_back(m_commandFactory.getSignalSemaphore(sync.id,
                                                          sync.value,
                                                          sync.operation,
                                                          sync.barrier));
}

void CommandQueue::setArmMonBeforeDesc(bool value)
{
    m_armMonBeforeDesc = value;
}

unsigned CommandQueue::getFirstFenceMonitorId()
{
    for (auto& mon : m_monIdToSetupMonitors)
    {
        if (mon.second.signalSyncId == FENCE_MONITOR_ID)
        {
            return mon.first;
        }
    }
    HB_ASSERT(false, "At least 1 monitor should be setup for each queue");
    return 0;
}

void CommandQueue::addMonitor(const MonObject&    monitor,
                              bool                isSetup,
                              MonitorCommandParts parts,
                              Settable<unsigned>  engID)
{
    std::vector<QueueCommandPtr> cmds;
    getMonitorCmds(monitor, cmds, parts, engID);

    for (auto& cmd : cmds)
    {
        PushBack(std::move(cmd), isSetup);
    }
}

void CommandQueue::addSignal(const SyncObject& sync, bool isSetup)
{
    std::vector<QueueCommandPtr> cmds;
    getSignalCmds(sync, cmds);

    for (auto& cmd : cmds)
    {
        PushBack(std::move(cmd), isSetup);
    }
}

void CommandQueue::AddNode(const pNode& n, HabanaGraph* g, bool isSetup)
{}

void CommandQueue::AddSuspend(unsigned cyclesToWait)
{
    PushBack(QueueCommandPtr{m_commandFactory.getSuspend(cyclesToWait)}, false);
}

void CommandQueue::finalizeSetupPhase()
{}

void CommandQueue::finalizeInitQueue(bool isSetup)
{
    auto& initialQueueCommands = getInitialQueueCommands();
    for (auto iter = initialQueueCommands.begin(); iter != initialQueueCommands.end();)
    {
        auto& cmd = *iter;
        bool isActiveRequired = cmd.second;
        if (!isActiveRequired)
        {
            PushBack(std::move(cmd.first), isSetup);
            iter = initialQueueCommands.erase(iter);
        }
        else
        {
            // push to Activate queue
            PushBack(std::move(cmd.first), true);
            iter = initialQueueCommands.erase(iter);
        }
    }
}

void CommandQueue::finalizeQueue(bool isSetup)
{
    if (!isQueueActive())
    {
        auto& initialQueueCommands = getInitialQueueCommands();
        for (auto iter = initialQueueCommands.begin(); iter != initialQueueCommands.end();)
        {
            auto& cmd = *iter;
            bool isActiveRequired = cmd.second;
            if (!isActiveRequired)
            {
                PushBack(std::move(cmd.first), isSetup);
                iter = initialQueueCommands.erase(iter);
            }
            else
            {
                ++iter;
            }
        }
    }
}

void CommandQueue::loadBaseRegsCacheUpdate(const pNode& node)
{
    const std::vector<BaseRegsCacheEntry>& cacheEntries = node->getNodeAnnotation().baseRegsCacheUpdate;

    if (cacheEntries.empty()) return;

    std::vector<uint64_t>  sectionIDs;
    unsigned               lastCacheIndex;
    unsigned               firstCacheIndex;

    sectionIDs.push_back(cacheEntries[0].sectionID);
    firstCacheIndex = cacheEntries[0].indexInCache;
    lastCacheIndex  = cacheEntries[0].indexInCache;

    // Accumulate consecutive indices into single write-bulk, indices are in acsending order
    // (and if they aren't, it would be less efficient but still function properly)
    for (unsigned i = 1; i < cacheEntries.size(); i++)
    {
        if (cacheEntries[i].indexInCache == lastCacheIndex + 1) // consecutive index?
        {
            sectionIDs.push_back(cacheEntries[i].sectionID);
            lastCacheIndex = cacheEntries[i].indexInCache;
        }
        else
        {
            // we found a gap, time to create a write-bulk command
            QueueCommandPtr cmd = makeWriteBulkCacheEntries(firstCacheIndex, sectionIDs, node);
            PushBack(std::move(cmd), false); // false means execute part
            sectionIDs.clear();
            sectionIDs.push_back(cacheEntries[i].sectionID);
            firstCacheIndex = cacheEntries[i].indexInCache;
            lastCacheIndex  = cacheEntries[i].indexInCache;
        }
    }
    QueueCommandPtr cmd = makeWriteBulkCacheEntries(firstCacheIndex, sectionIDs, node);
    PushBack(std::move(cmd), false);
    // Add QmanDelay command as a WA for a HW issue - H6-3262 (https://jira.habana-labs.com/browse/H6-3262)
    // To avoid race between updating regs in cache to read them using wreg64
    QueueCommandPtr qmanDelayCmd = m_commandFactory.getQmanDelay();
    qmanDelayCmd->setAsBlobCommitter(node->getExecutionOrderedIndex());
    PushBack(std::move(qmanDelayCmd), false);
}

QueueCommandPtr CommandQueue::makeWriteBulkCacheEntries(unsigned                     cacheIndex,
                                                        const std::vector<uint64_t>& sectionIDs,
                                                        const pNode&                 node) const
{
    // WriteManyRegisters is expecting to get register values as 32bit elements. We provide values that
    // include the section IDs in the high bits to ensure blob compression will not drop the commands.
    // The patch points also contain section IDs and will be used in run-time to set the physical address.
    std::vector<uint32_t> values(sectionIDs.size() * 2, 0);
    uint64_t* p64 = (uint64_t*)values.data();
    for (uint64_t secID : sectionIDs)
    {
        *p64++ = getVirtualAddressForMemoryID(secID, 0);
    }

    QueueCommandPtr cmd = m_commandFactory.getWriteManyRegisters(
        m_commandFactory.getRegForBaseAddress(cacheIndex),
        values.size(),
        values.data());

    // Create patch-points for the addresses. We create LOW and HIGH and not FULL so we won't assume alignment,
    // they will be unified to FULL by the recipe generator if they will seat consecutively in the blob.
    BasicFieldsContainerInfo addrContainer;
    for (unsigned i = 0; i < sectionIDs.size(); ++i)
    {
        addrContainer.addAddressEngineFieldInfo(node,
                                                getMemorySectionNameForMemoryID(sectionIDs[i]),
                                                sectionIDs[i],
                                                0,         // low target address
                                                0,         // high target address
                                                i * 2,     // low field offset
                                                i * 2 + 1, // high field offset
                                                FIELD_MEMORY_TYPE_DRAM);
    }
    cmd->SetContainerInfo(addrContainer);
    return cmd;
}

void CommandQueue::pushAdditionalDynamicCmds(const NodePtr& node, unsigned pipeLevel, bool isLastPipelineLevel)
{
    pushAdditionalDynamicCmds4sfg(node, pipeLevel, isLastPipelineLevel);
    pushAdditionalDynamicCmds4mcidRollover(node, pipeLevel);
    pushAdditionalDynamicCmds4sobReset(node, pipeLevel); // must be last
}

void CommandQueue::pushAdditionalDynamicCmds4sfg(const NodePtr& node, unsigned pipeLevel, bool isLastPipelineLevel)
{
    // check if last descriptor of node needs to increment SFG sync object
    if (isLastPipelineLevel && node->getNodeAnnotation().sfgSyncObjValue.is_set())
    {
        // create sfg command and push to queue
        unsigned sfgSobInc = node->getNodeAnnotation().sfgSyncObjValue.value();
        PushBack(std::move(m_commandFactory.getSfgInc(sfgSobInc)), false);
        LOG_DEBUG(SFG, "Adding SFG cmd (sfgSobInc value={}) to {}", sfgSobInc, node->getNodeName());
    }
}

void CommandQueue::pushAdditionalDynamicCmds4mcidRollover(const NodePtr& node, unsigned pipeLevel)
{
    const std::list<NodeROI>& logicalRois = *(node->getLogicalRois());
    HB_ASSERT(pipeLevel < logicalRois.size(), "pipeLevel OOB");
    std::list<NodeROI>::const_iterator itrRoi = std::next(logicalRois.begin(), pipeLevel);
    for (unsigned rolloverId : itrRoi->rolloverIds)
    {
        unsigned sobTargetVal = node->getNodeAnnotation().arcSyncScheme[pipeLevel].emittedSigVal.value();
        PushBack(std::move(m_commandFactory.getMcidRollover(sobTargetVal)), false);
        LOG_DEBUG(CACHE_MAINT, "Adding mcid rollover cmd (rolloverId={}) to {}", rolloverId, getName());
    }
}

void CommandQueue::pushAdditionalDynamicCmds4sobReset(const pNode& node, unsigned pipeLevel)
{
    // check if current pipeline level requires sync object reset after it
    if (pipeLevel < node->getNodeAnnotation().arcSyncScheme.size() &&
        node->getNodeAnnotation().arcSyncScheme[pipeLevel].sobResetTotalNumEngs > 0)
    {
        // create ResetSobs command and push to queue
        unsigned sobTargetVal = node->getNodeAnnotation().arcSyncScheme[pipeLevel].emittedSigVal.value();
        unsigned totalNumEngs = node->getNodeAnnotation().arcSyncScheme[pipeLevel].sobResetTotalNumEngs;
        PushBack(std::move(m_commandFactory.getResetSobs(sobTargetVal, totalNumEngs)), false);
    }
}

const std::vector<std::unique_ptr<QueueCommand>>& CommandQueue::getCommands(bool isSetup, bool isLoad) const
{
    return isSetup ? (isLoad ? m_queueLoad : m_queueActivate) : m_queueExe;
}

//---------------------------------------------------------
//--------------------- DmaCommandQueue -------------------
//----------------- Base for all DMA queues ---------------
//---------------------------------------------------------

DmaCommandQueue::DmaCommandQueue(const QueueCommandFactory&  cmdFactory,
                                 unsigned                    queId,
                                 HabanaDeviceType            devType)
: CommandQueue(cmdFactory, queId, devType)
{
}

DmaCommandQueue::~DmaCommandQueue() = default;

void DmaCommandQueue::AddNode(const pNode& n, HabanaGraph* g, bool isSetup)
{
    HB_ASSERT_PTR(n);
    HB_ASSERT_PTR(g);

    addPreSyncScheme(n, isSetup);

    const std::list<NodeROI> &roiList = *(g->GetNodeROIs(n));

    if (isSetup)
    {
        for (const NodeROI &roi : roiList)
        {
            pushQueueCommand(n, roi, isSetup, false);
        }
    }
    else
    {
        const auto& syncScheme = n->getNodeAnnotation().syncScheme[GetEngineIndex()];
        // Signal complete if sync scheme intended to
        bool wrComplete    = ! syncScheme.pipelineSyncs.empty();

        uint32_t roiIdx = 0;
        for (const NodeROI &roi : roiList)
        {
            for (const auto &monitor : syncScheme.pipelineSyncs[roiIdx].monitors)
            {
                addMonitor(monitor, isSetup);
            }
            // add cpSyncs before the descriptor is executed
            for (const auto &sync : syncScheme.pipelineSyncs[roiIdx].cpSyncs)
            {
                addSignal(sync, isSetup);
            }

            pushQueueCommand(n, roi, isSetup, wrComplete);
            ++roiIdx;
        }
    }

    addPostExeSyncs(n, isSetup);
}

//---------------------------------------------------------
//------------------- DmaHostToDevQueue -------------------
//---------------------------------------------------------

DmaHostToDevQueue::DmaHostToDevQueue(const QueueCommandFactory&  cmdFactory,
                                     unsigned                    queId,
                                     const SyncConventions&      syncConventions)
: DmaCommandQueue(cmdFactory, queId, DEVICE_DMA_HOST_DEVICE),
  m_syncConventions(syncConventions)
{
}

DmaHostToDevQueue::~DmaHostToDevQueue() = default;

void DmaHostToDevQueue::pushQueueCommand(const pNode& n, const NodeROI& roi, bool isSetup, bool wrComplete)
{
    std::shared_ptr<Tensor> t = n->getOutput(0);
    unsigned elemSizeInBits = t->getElementSizeInBits();
    static const uint32_t SRC_DWORD_OFFSET_FROM_HEADER = 0;

    QueueCommandPtr cmd = nullptr;
    if (! isSetup && t->tensorAllocatedInSram())
    {
        cmd =  m_commandFactory.getDmaHostToSram((char *) t->getAddress() + safeBitsToByte(roi.baseOffset[0] * elemSizeInBits),
                                                 t->getSramOffset() + safeBitsToByte(roi.baseOffset[0] * elemSizeInBits),
                                                 safeBitsToByte(roi.size[0] * elemSizeInBits),
                                                 wrComplete,
                                                 n->getContextId());
        PushBack(std::move(cmd), isSetup, false);
    }
    else if (isSetup && (std::dynamic_pointer_cast<DMANode>(n))->getDmaType() == DMA_TYPE_INTERNAL)
    {
        // TODO: remove entire else case when supporting only internal queue in activate

        const pTensor& input  = (n->getNumInputs() == 1? n->getInput(0) : t);

        uint64_t src = input->getDramOffset() + safeBitsToByte(roi.baseOffset[0] * elemSizeInBits);
        uint64_t dst = t->getSramOffset() + safeBitsToByte(roi.baseOffset[0] * elemSizeInBits);

        cmd =  m_commandFactory.getDmaDramToSram(src,
                                                 dst,
                                                 safeBitsToByte(roi.size[0] * elemSizeInBits),
                                                 wrComplete,
                                                 n->getContextId());

        BasicFieldsContainerInfo cmdAddressFieldsInfo;
        uint64_t                 memID = getMemoryIDFromVirtualAddress(src);

        cmdAddressFieldsInfo.addAddressEngineFieldInfo(n,
                                                       getMemorySectionNameForMemoryID(memID),
                                                       memID,
                                                       src,
                                                       SRC_DWORD_OFFSET_FROM_HEADER,
                                                       FIELD_MEMORY_TYPE_DRAM);

        cmd->SetContainerInfo(cmdAddressFieldsInfo);

        PushBack(std::move(cmd), isSetup, false);
    }
    else //to Dram
    {
        cmd =  m_commandFactory.getDmaHostToDram((char *) t->getAddress() + safeBitsToByte(roi.baseOffset[0] * elemSizeInBits),
                                                 t->getDramOffset() + safeBitsToByte(roi.baseOffset[0] * elemSizeInBits),
                                                 safeBitsToByte(uint64_t(roi.size[0]) * uint64_t(elemSizeInBits)),
                                                 wrComplete,
                                                 n->getContextId());
        PushBack(std::move(cmd), isSetup, true);
    }
}

//---------------------------------------------------------
//------------------- DmaDevToHostQueue -------------------
//---------------------------------------------------------

DmaDevToHostQueue::DmaDevToHostQueue(const QueueCommandFactory&  cmdFactory,
                                     unsigned                    queId)
: DmaCommandQueue(cmdFactory, queId, DEVICE_DMA_DEVICE_HOST)
{
}

DmaDevToHostQueue::~DmaDevToHostQueue() = default;

void DmaDevToHostQueue::pushQueueCommand(const pNode& n, const NodeROI& roi, bool isSetup, bool wrComplete)
{
    std::shared_ptr<Tensor> t = n->getInput(0);
    unsigned elemSizeBits = t->getElementSizeInBits();

    QueueCommandPtr cmd = nullptr;
    if (t->tensorAllocatedInSram())
    {
        cmd =  m_commandFactory.getDmaSramToHost((char *) t->getAddress() + safeBitsToByte(roi.baseOffset[0] * elemSizeBits),
                                                 t->getSramOffset() + safeBitsToByte(roi.baseOffset[0] * elemSizeBits),
                                                 safeBitsToByte(roi.size[0] * elemSizeBits),
                                                 wrComplete,
                                                 n->getContextId());
    }
    else //from Dram
    {
        cmd =  m_commandFactory.getDmaDramToHost((char *) t->getAddress() + safeBitsToByte(roi.baseOffset[0] * elemSizeBits),
                                                 t->getDramOffset() + safeBitsToByte(roi.baseOffset[0] * elemSizeBits),
                                                 safeBitsToByte(roi.size[0] * elemSizeBits),
                                                 wrComplete,
                                                 n->getContextId());
    }
    PushBack(std::move(cmd), isSetup);
}

//---------------------------------------------------------
//------------------ DmaDramToSramQueue -------------------
//---------------------------------------------------------

DmaDramToSramQueue::DmaDramToSramQueue(const QueueCommandFactory&  cmdFactory,
                                       unsigned                    queId,
                                       HabanaDeviceType            type,
                                       SyncConventions&            syncConventions)
: DmaCommandQueue(cmdFactory, queId, type),
  m_syncConventions(syncConventions)
{
}

DmaDramToSramQueue::~DmaDramToSramQueue() = default;

void DmaDramToSramQueue::pushQueueCommand(const pNode& n, const NodeROI& roi, bool isSetup, bool wrComplete)
{
    const pTensor& output = n->getOutput(0);
    const pTensor& input  = (n->getNumInputs() == 1? n->getInput(0) : output);
    unsigned elemSize     = output->getElementSizeInBytes();

    QueueCommandPtr cmd =  m_commandFactory.getDmaDramToSram(input->getDramOffset() + roi.baseOffset[0] * elemSize,
                                                           output->getSramOffset() + roi.baseOffset[0] * elemSize,
                                                           roi.size[0] * elemSize,
                                                           wrComplete,
                                                           n->getContextId());
    PushBack(std::move(cmd), isSetup);
}

//---------------------------------------------------------
//------------------ DmaSramToDramQueue -------------------
//---------------------------------------------------------

DmaSramToDramQueue::DmaSramToDramQueue(const QueueCommandFactory&  cmdFactory,
                                       unsigned                    queId)
: DmaCommandQueue(cmdFactory, queId, DEVICE_DMA_SRAM_DRAM)
{
}

DmaSramToDramQueue::~DmaSramToDramQueue() = default;

void DmaSramToDramQueue::pushQueueCommand(const pNode& n, const NodeROI& roi, bool isSetup, bool wrComplete)
{
    const pTensor& input  = n->getInput(0);
    const pTensor& output = (n->getNumOutputs() == 1 ? n->getOutput(0) : input);
    unsigned elemSize     = input->getElementSizeInBytes();

    QueueCommandPtr cmd =  m_commandFactory.getDmaSramToDram(output->getDramOffset() + roi.baseOffset[0] * elemSize,
                                                           input->getSramOffset() + roi.baseOffset[0] * elemSize,
                                                           roi.size[0] * elemSize,
                                                           wrComplete,
                                                           n->getContextId());
    PushBack(std::move(cmd), isSetup);
}

//-------------------------------------------------------------------------------------------
// Templeate implementation is inlined in desc_command_queue.inl and included from the H file
//-------------------------------------------------------------------------------------------
