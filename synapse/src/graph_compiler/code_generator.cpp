#include "code_generator.h"

#include "code_generation/tensor_size_validator.h"
#include "defs.h"
#include "dma_cost_model.h"
#include "habana_graph.h"
#include "tensor.h"
#include "tpc_node.h"
#include "types_exception.h"

#include <sys/types.h>

CodeGenerator::CodeGenerator(const CodeGenerator& other, HabanaGraph* graph)
: m_kernelsAddrMap(other.m_kernelsAddrMap),
  m_kernelsPrintf(other.m_kernelsPrintf),
  m_usingPrintf(other.m_usingPrintf),
  m_NOPKernel(other.m_NOPKernel),
  m_graph(graph),
  m_dramBaseAddr(other.m_dramBaseAddr),
  m_dramSize(other.m_dramSize),
  m_sramSize(other.m_sramSize),
  m_synapseSramBaseAddr(other.m_synapseSramBaseAddr),
  m_programDataBlobs(other.m_programDataBlobs),
  m_mcidConverter(other.m_mcidConverter),
  m_cmeCommands(other.m_cmeCommands)
{
}

CodeGenerator& CodeGenerator::operator=(const CodeGenerator& other)
{
    if (this != &other)
    {
        clear();
        m_kernelsAddrMap = other.m_kernelsAddrMap;
        m_kernelsPrintf  = other.m_kernelsPrintf;
        m_usingPrintf    = other.m_usingPrintf;
        m_NOPKernel      = other.m_NOPKernel;
        m_synapseSramBaseAddr = other.m_synapseSramBaseAddr;
        m_dramBaseAddr        = other.m_dramBaseAddr;
        m_dramSize            = other.m_dramSize;
        m_sramSize            = other.m_sramSize;
        m_programDataBlobs    = other.m_programDataBlobs;
        m_mcidConverter       = other.m_mcidConverter;
        m_cmeCommands         = other.m_cmeCommands;
    }
    return *this;
}

void CodeGenerator::clear()
{
    m_kernelsAddrMap.clear();
    m_kernelsPrintf.clear();
    m_usingPrintf = false;
    m_NOPKernel.nopKernelOffset.unset();
    m_NOPKernel.nopKernelSection = 0;
    m_NOPKernel.nopKernelSize    = 0;
    m_synapseSramBaseAddr        = 0;
    m_dramBaseAddr               = 0;
    m_dramSize                   = 0;
    m_programDataBlobs.clear();
    m_physicalRois.clear();
    m_mmeDispatcher.reset();
    m_tpcDispatcher.reset();
    m_rotatorDispatcher.reset();
    m_cmeCommands.clear();
}

bool CodeGenerator::init()
{
    initAllocators();
    try
    {
        addAllPasses();
    }
    catch (PassFailedException& ex)
    {
        LOG_ERR(GC, "Registering all code generation passes failed");
        return false;
    }
    return true;
}

void CodeGenerator::generate(const HabanaGraph* graph)
{
    if (!TensorSizeValidator(*graph).validateTensors(*graph)) throw InvalidTensorSizeException();
    runPassManager();
    addAllDescriptors();
    // initQueues(); temporary until addAllDescriptos will move to codeGen
    generateQueues();
    // generateRecipes(*graph); temporary until addAllDescriptos will move to codeGen
}

std::list<NodeROI>* CodeGenerator::getNodeROIs(const NodePtr n) const
{
    return m_graph->GetNodeROIs(n);
}

const std::shared_ptr<HalReader>& CodeGenerator::getHALReader() const
{
    return m_graph->getHALReader();
}

void CodeGenerator::initQueues()
{
    addInitialSyncs();
    initDMAQueues();
    fillSetupNodes();
}

// Same implementation for gaudi2/gaudi3/eager CodeGen. gaudi/greco reimplement
void CodeGenerator::initTPCQueues()
{
    std::vector<std::vector<QueueCommandPtr>> cmdsToQueues = getInitTPCQueuesCmds();

    for (unsigned tpcEng = 0; tpcEng < m_tpcDispatcher->getNumEngines(); ++tpcEng)
    {
        const CommandQueuePtr&        queue       = m_tpcDispatcher->getQueue(tpcEng);
        std::vector<QueueCommandPtr>& cmdsToQueue = cmdsToQueues[tpcEng];

        // Put all TPC init commands in the queue's init commands vector
        for (auto& cmd : cmdsToQueue)
        {
            queue->getInitialQueueCommands().emplace_back(std::move(cmd), false);
        }
    }
}

std::vector<std::vector<QueueCommandPtr>> CodeGenerator::getInitTPCQueuesCmds()
{
    std::vector<std::vector<QueueCommandPtr>> cmdsToQueues;
    bool                                      shouldPrefetch = GCFG_ENABLE_TPC_ICACHE_PREFETCH.value();
    cmdsToQueues.resize(m_tpcDispatcher->getNumEngines());

    for (unsigned tpcEng = 0; tpcEng < m_tpcDispatcher->getNumEngines(); ++tpcEng)
    {
        std::vector<QueueCommandPtr>& cmdsToQueue = cmdsToQueues[tpcEng];
        const QueueCommandFactory&    cmdFactory  = getCommandFactory();

        // Prefetch the instruction cache
        if (shouldPrefetch)
        {
            // Commit the blob so far to ensure the invalidate command will preced the prefetch.
            // If we won't do that, we will end up with reverse order due to patching blob order (prefetch is patched).
            prefetchTPCKernels(cmdsToQueue);
        }
        else  // in gaudi we can't prtefetch due to HW bug we invalidate
        {
            // Invalidate instruction cache
            cmdsToQueue.push_back(cmdFactory.getInvalidateTPCCaches());
        }
    }

    return cmdsToQueues;
}

void CodeGenerator::prefetchTPCKernels(std::vector<QueueCommandPtr>& cmdsToQueue)
{
    // Add prefetch TPC kernel i-cache command
    const QueueCommandFactory&                          cmdFactory       = getCommandFactory();
    const std::pair<deviceAddrOffset, deviceAddrOffset> lowerUpperBounds = getKernelsLowerAndUpperBounds();
    deviceAddrOffset                                    kernelBase       = lowerUpperBounds.first;
    deviceAddrOffset                                    kernelTop        = lowerUpperBounds.second;
    uint64_t                                            kernelsTotalSize = kernelTop - kernelBase;

    if (kernelsTotalSize > m_graph->getHALReader()->getTPCICacheSize())
    {
        LOG_WARN(GC, "Total kernel size {} exceeds TPC i-cache size", kernelsTotalSize);
    }
    if (kernelsTotalSize > 0)
    {
        LOG_DEBUG(GC, "Pre-fetching kernels of size {} from {} to {}", kernelsTotalSize, kernelBase, kernelTop);
        ptrToInt kernelBaseAddr;

        kernelBaseAddr.u64  = kernelBase;
        auto kernelsAddrCmd = cmdFactory.getUploadKernelsAddr(kernelBaseAddr.u32[0], kernelBaseAddr.u32[1]);
        cmdsToQueue.push_back(std::move(kernelsAddrCmd));
    }
}

NodePtr CodeGenerator::getNodeSharedPtr(const Node& node)
{
    return const_cast<Node&>(node).shared_from_this();
}

void CodeGenerator::setupQueuesMonitors()
{
    for (const std::pair<const uint32_t, CommandQueuePtr>& queue : m_commandQueueById)
    {
        m_monitorSetupManager->initSetupMonitorForQueue(queue.second);
    }
}

void CodeGenerator::downstreamSetupNode(NodePtr n)
{
    m_downstreamQueue->AddNode(n, m_graph, true);
}

CommandQueuePtr CodeGenerator::getActivateDramSramQueue()
{
    HB_ASSERT(0, "Activate dram sram queue unsupported");
    return nullptr;
}

void CodeGenerator::addExecuteDMANode(NodePtr n, uint32_t* inputDmaInd, uint32_t* outputDmaInd)
{
    DMANode* dmaNode = dynamic_cast<DMANode*>(n.get());

    switch (dmaNode->getDmaType())
    {
        case DMA_TYPE_UPSTREAM:
            m_upstreamQueue->AddNode(n, m_graph, false);
            m_outputDmaIndices.push_back(std::make_pair(n->getInput(0), *outputDmaInd));
            *outputDmaInd += m_graph->GetNodeROIs(n)->size();
            break;
        case DMA_TYPE_DOWNSTREAM:
            // Host to SRAM or DRAM depending on tensor location
            m_downstreamQueue->AddNode(n, m_graph, false);
            m_inputDmaIndices.push_back(std::make_pair(n->getOutput(0), *inputDmaInd));
            *inputDmaInd += m_graph->GetNodeROIs(n)->size();
            break;
        default:
            fillQueuesWithDmaNode(n);
            break;
    }
}

void CodeGenerator::fillSetupNodes()
{
    for (auto n : m_graph->getSetupNodes())
    {
        downstreamSetupNode(n);

        // There should be only one tensor in the outputs list of a DMA node.
        const TensorPtr& tensor = n->getOutput(0);

        // if setup node should not be prefetched, and it should eventually be in SRAM
        // downstream to SRAM also during setup phase (dram-sram copy)
        if (!tensor->getTensorAnnotation().memorySpaceInfo.prefetchInfo.prefetch)
        {
            if (tensor->tensorAllocatedInSram())
            {
                std::shared_ptr<DMANode> dmaNode = std::dynamic_pointer_cast<DMANode>(n);
                HB_ASSERT_PTR(dmaNode);

                DMA_TYPE originalDmaType = dmaNode->getDmaType();
                dmaNode->setDmaType(DMA_TYPE_INTERNAL);

                // use external queue if sram-dram queue is not supported for setup nodes (activate stage)
                CommandQueuePtr cmdQueuePtr = getActivateDramSramQueue();
                cmdQueuePtr->AddNode(dmaNode, m_graph, true);

                dmaNode->setDmaType(originalDmaType);
            }
        }
    }
}

void CodeGenerator::downloadProgramDataBlobs()
{
    const QueueCommandFactory& cmdFactory  = getCommandFactory();
    int                        blobCounter = 0;
    for (auto blob : m_programDataBlobs)
    {
        char* hostAddr = nullptr;

        if (blob->hostAddrPtr != nullptr)
        {
            HB_ASSERT(blob->hostAddrSharedPtr == nullptr,
                      "hostAddrPtr and hostAddrSharedPtr cannot be set simultanusly");
            hostAddr = blob->hostAddrPtr;
        }
        else if (blob->hostAddrSharedPtr != nullptr)
        {
            HB_ASSERT(blob->hostAddrPtr == nullptr, "hostAddrPtr and hostAddrSharedPtr cannot be set simultanusly");
            hostAddr = blob->hostAddrSharedPtr.get();
        }
        else
        {
            HB_ASSERT(0, "host pointer was not set, this is impossible");
        }

        QueueCommandPtr dma{cmdFactory.getDmaHostToDram(hostAddr,
                                                        blob->deviceAddr,
                                                        blob->binSize,
                                                        false)};
        m_downstreamQueue->PushBack(std::move(dma), true, true);

        LOG_DEBUG(GC, "DMA program-data blob number {} from 0x{:x} to Dram (virtual) 0x{:x}, size {}",
                  blobCounter,
                  (uint64_t)hostAddr,
                  maskOutMemoryID(blob->deviceAddr),
                  blob->binSize);

        blobCounter++;
    }
}

std::map<uint32_t, std::list<SyncOrMonitor>>& CodeGenerator::getFinalSyncInstructions(bool bIsActivate /*false*/)
{
    return m_finalSyncInstructionsByQueueId;
}

void CodeGenerator::addFinalSyncs(bool bIsActivate /*false*/)
{
    // Add final sync instructions
    for (const auto& syncsAndQueueId : getFinalSyncInstructions(bIsActivate))
    {
        auto commandQueueIter = m_commandQueueById.find(syncsAndQueueId.first);
        HB_ASSERT(commandQueueIter != m_commandQueueById.end(),
                  "Got final syncs to unrecognized queue id {}",
                  syncsAndQueueId.first);
        const CommandQueuePtr& queue = commandQueueIter->second;
        for (const SyncOrMonitor& finalSync : syncsAndQueueId.second)
        {
            if (finalSync.type == SyncOrMonitor::SYNC_OBJ)
            {
                queue->addSignal(finalSync.sync, bIsActivate);
            }
            else
            {
                queue->addMonitor(finalSync.monitor, bIsActivate);
            }
        }
    }
}

void CodeGenerator::markLastCommandInQueuesForCommit(bool bLastCommand /* false*/)
{
    // We want to assure that the last command in all queues up until this point will commit the blob.
    // Currently we use it in 2 places:
    // 1. At the end of init phase
    // 2. To mark the very last command in each queue

    for (const std::pair<const uint32_t, CommandQueuePtr>& queue : m_commandQueueById)
    {
        if (queue.second->Size(true))
        {
            auto& lastCommand = queue.second->getCommands(true).back();
            // use the previously stored node exe index if any; otherwise, we will get the default
            lastCommand->setAsBlobCommitter(lastCommand->getNodeExecutionIndex());
            LOG_DEBUG(QMAN,
                      "Marking last command in activate queue (for queue id: {}) as committer",
                      queue.second->GetQueueID());
        }

        if (queue.second->Size(false))
        {
            auto& lastCommand = queue.second->getCommands(false).back();
            // use the previously stored node exe index if any; otherwise, we will get the default
            lastCommand->setAsBlobCommitter(lastCommand->getNodeExecutionIndex());
            LOG_DEBUG(QMAN, "Marking last command in execute queue (for queue id: {}) as committer",
                      queue.second->GetQueueID());
        }
    }
}

void CodeGenerator::finalizeQueues(bool isSetup)
{
    for (const std::pair<const uint32_t, CommandQueuePtr>& queue : m_commandQueueById)
    {
        queue.second->finalizeQueue(isSetup);
    }

    addFinalSyncs();

    markLastCommandInQueuesForCommit(true);
}

void CodeGenerator::finalizeInitQueues(bool isSetup)
{
    for (const std::pair<const uint32_t, CommandQueuePtr>& queue : m_commandQueueById)
    {
        queue.second->finalizeInitQueue(isSetup);
    }

    markLastCommandInQueuesForCommit();
}

void CodeGenerator::finalizeFillQueues()
{
    addFinalSyncs();

    markLastCommandInQueuesForCommit(true);
}

void CodeGenerator::fillQueues()
{
    uint32_t inputDmaInd  = 0;
    uint32_t outputDmaInd = 0;

    for (auto n : m_graph->getExeSortedNodes())
    {
        if (m_graph->runsOnMME(n))
        {
            m_mmeDispatcher->dispatchNode(n, m_graph, false);
        }
        else if (m_graph->runsOnTPC(n))
        {
            m_tpcDispatcher->dispatchNode(n, m_graph, false);
        }
        else if (m_graph->runsOnRotator(n))
        {
            m_rotatorDispatcher->dispatchNode(n, m_graph, false);
        }
        else if (n->isDma())
        {
            // All DMA's that are not in the m_setupNodes should be in the execute queue...
            // except the intermediates
            addExecuteDMANode(n, &inputDmaInd, &outputDmaInd);
        }
        else
        {
            HB_ASSERT(n->isLogicalOperation(), "Unknown node type");
        }
        generateCmeCommands(n);
    }
}

void CodeGenerator::printQueues() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;
    LOG_DEBUG(QMAN, "Queue Dump:");
    LOG_DEBUG(QMAN,
              "    The Queue Dump is obsolete and disabled, look down the log file for \"Program Dump\" or \"Start of "
              "Program\" to see the queue commands.");

    // We no longer dump the queues as they provide misleading information. The final command order is altered during
    // recipe generation when we group the commands to blobs. Therefore, it is better to print the commands from the
    // actual blobs. Open the comment below only if you want to see the queues content before recipe generation but
    // beware that the order of commands is not final at this stage.

    // for (const std::pair<const uint32_t, CommandQueuePtr>& queue : m_commandQueueById)
    // {
    //     queue.second->Print();
    // }
}

unsigned int CodeGenerator::getKernelsBinarySize() const
{
    unsigned int ret = 0;
    for (const auto& kernel : m_kernelsAddrMap)
    {
        ret += kernel.second->binSize;
    }
    return ret;
}

deviceAddrOffset CodeGenerator::getKernelAddress(kernelID kid) const
{
    auto it = m_kernelsAddrMap.find(kid);
    HB_ASSERT(it != m_kernelsAddrMap.end(), "Request for address for kernel with ID {} failed", kid);
    return it->second->deviceAddr;
}

deviceAddrOffset CodeGenerator::getKernelAddress(kernelID kid, bool& wasFound) const
{
    auto it = m_kernelsAddrMap.find(kid);
    if (it == m_kernelsAddrMap.end())
    {
        wasFound = false;
        return 0;
    }
    wasFound = true;
    return it->second->deviceAddr;
}

void CodeGenerator::configNOPKernel(deviceAddrOffset addrOffset, uint64_t section, unsigned int kernelSize)
{
    m_NOPKernel.nopKernelOffset.set(addrOffset);
    m_NOPKernel.nopKernelSection = section;
    m_NOPKernel.nopKernelSize    = kernelSize;
}

const DeviceAddrOffsetPair CodeGenerator::getKernelsLowerAndUpperBounds() const
{
    if (m_kernelsAddrMap.empty())
    {
        return {0, 0};
    }

    // sort tpc kernel addresses by execution order
    const NodeSet&   nodes             = m_graph->getNodes();
    deviceAddrOffset lowerBoundAddress = std::numeric_limits<deviceAddrOffset>::max();
    deviceAddrOffset upperBoundAddress = 0;
    for (const NodePtr& n : nodes)
    {
        if (!n || !HabanaGraph::runsOnTPC(n)) continue;
        const auto&            tpcNode    = static_cast<TPCNode&>(*n);
        kernelID               kid        = tpcNode.getUniqueID();
        const ProgramDataBlob& kernelAddr = *m_kernelsAddrMap.at(kid);
        lowerBoundAddress                 = std::min(lowerBoundAddress, kernelAddr.deviceAddr);
        upperBoundAddress                 = std::max(upperBoundAddress, kernelAddr.deviceAddr + kernelAddr.binSize);
        LOG_DEBUG(GC, "Kernel {} fits in size", tpcNode.getGUID());
    }

    // whenever there is nop kernel, we need to take its offset/size into lowerBound/upperBound
    // calculation. Currently, it's relevant only for gaudi2
    if (m_NOPKernel.nopKernelOffset.is_set())
    {
        if (m_NOPKernel.nopKernelOffset.value() < lowerBoundAddress)
        {
            lowerBoundAddress = m_NOPKernel.nopKernelOffset.value();
        }

        if (m_NOPKernel.nopKernelOffset.value() + m_NOPKernel.nopKernelSize > upperBoundAddress)
        {
            upperBoundAddress = m_NOPKernel.nopKernelOffset.value() + m_NOPKernel.nopKernelSize;
        }
    }

    return {lowerBoundAddress, upperBoundAddress};
}

uint64_t CodeGenerator::getNextMemorySectionID(SectionIDGenerator::AllocationManagementType allocType)
{
    return m_MemSectionIDGenerator.nextSectionId(allocType);
}

uint64_t CodeGenerator::getNumberOfMemorySections(SectionIDGenerator::AllocationManagementType allocType) const
{
    uint64_t numberOfSections = m_MemSectionIDGenerator.getNumberOfMemorySections(allocType);
    if (GCFG_INTERNAL_TEST.value())
    {
        std::unordered_set<uint64_t> uniqueIds;
        // it may be that a test didn't go through the API and therefor, the MemSectionIDGenerator wasn't used
        for (const TensorPtr& t : m_graph->getTensors())
        {
            uint64_t sectionId = t->getMemorySectionID();
            if (allocType == SectionIDGenerator::GC_ALLOCATED_SECTIONS)
            {
                auto userSectionId = t->getTensorAnnotation().nonPersistentSectionInfo.sectionId;
                sectionId          = userSectionId.is_set() ? userSectionId.value() : 0;
            }
            uniqueIds.insert(sectionId);
        }
        numberOfSections = uniqueIds.size();
    }
    return numberOfSections;
}

LogicalMcid CodeGenerator::getNextMCID(MCIDGenerator::MCIDType mcidType)
{
    return m_mcidGenerator.nextMCID(mcidType);
}

void CodeGenerator::initSram(uint64_t sramSize, uint64_t sramBaseAddr)
{
    setSramSize(sramSize);
    setSramBaseAddr(sramBaseAddr);
}

void CodeGenerator::initDram(uint64_t dramSize, uint64_t dramBaseAddr)
{
    setDramSize(dramSize);
    setDramBaseAddr(dramBaseAddr);
}

void CodeGenerator::addCommandQueue(const CommandQueuePtr& queue)
{
    HB_ASSERT_PTR(queue);
    HB_ASSERT(m_commandQueueById.find(queue->GetQueueID()) == m_commandQueueById.end(), "Queue was already created");

    m_commandQueueById[queue->GetQueueID()] = queue;
}

std::map<HabanaDeviceType, std::vector<std::list<SyncOrMonitor>>>& CodeGenerator::getInitialSyncInstructionsByQueueId()
{
    return m_initialSyncInstructionsByQueue;
}

void CodeGenerator::registerDispatcher(QueueDispatcher& dispatcher)
{
    for (const auto& queuePtr : dispatcher.getQueues())
    {
        addCommandQueue(queuePtr);
    }
}

const QueueCommandFactory& CodeGenerator::getCommandFactory() const
{
    // since cmd factories are identical in all queues we return mme factory which always exist
    // But... if it does not (Media), we will use the TPC and otherwise ASSERT
    if ((m_mmeDispatcher != nullptr) && (m_mmeDispatcher->getNumEngines() != 0))
    {
        return m_mmeDispatcher->getQueue(0)->getCommandFactory();
    }
    else if ((m_tpcDispatcher != nullptr) && (m_tpcDispatcher->getNumEngines() != 0))
    {
        return m_tpcDispatcher->getQueue(0)->getCommandFactory();
    }
    else
    {
        HB_ASSERT(false, "MME and TPC dispatchers does not exists");
        return m_tpcDispatcher->getQueue(0)->getCommandFactory();  // For compilation only
    }
}

// Used by gaudi and greco
// For greco: bIsActivate determines if we push it to execute or activate queues
void CodeGenerator::addInitialSyncs(bool bIsActivate /*false*/)
{
    for (const auto& deviceTypeToQueue : m_initialSyncInstructionsByQueue)
    {
        const HabanaDeviceType                       deviceType      = deviceTypeToQueue.first;
        const std::vector<std::list<SyncOrMonitor>>& queueSyncOrMons = deviceTypeToQueue.second;
        for (unsigned engineId = 0; engineId < queueSyncOrMons.size(); ++engineId)
        {
            const std::list<SyncOrMonitor>& syncOrMons = queueSyncOrMons[engineId];
            auto commandQueueIter                      = m_commandQueueById.find(getQueueID(deviceType, engineId));

            if (commandQueueIter == m_commandQueueById.end()) continue;
            const CommandQueuePtr& queue = commandQueueIter->second;
            for (SyncOrMonitor finalSync : syncOrMons)
            {
                std::vector<QueueCommandPtr> cmds;

                if (finalSync.type == SyncOrMonitor::SYNC_OBJ)
                {
                    if (bIsActivate)
                    {
                        // Add engine barrier to sync activate commands completion (greco only)
                        finalSync.sync.barrier |= ENGINE_BARRIER;
                    }
                    queue->getSignalCmds(finalSync.sync, cmds);
                }
                else
                {
                    queue->getMonitorCmds(finalSync.monitor, cmds);
                }

                for (auto& cmd : cmds)
                {
                    queue->getInitialQueueCommands().emplace_back(std::move(cmd), bIsActivate);
                }
            }
        }
    }
}
