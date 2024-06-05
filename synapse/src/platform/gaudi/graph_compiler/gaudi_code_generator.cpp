
#include "gaudi_code_generator.h"
#include "code_generation/code_generator_factory.h"
#include "code_generator.h"
#include "gaudi_graph.h"
#include "mme_dispatcher.h"
#include "queue_command.h"
#include "tpc_dispatcher.h"
#include "sync/sync_conventions.h"
#include "platform/gaudi/graph_compiler/sync/monitor_setup_manager.h"
#include "patch_point_generator.h"
#include "descriptor_generator.h"

using namespace gaudi;
CodeGeneratorPtr CodeGeneratorFactory::instantiateGaudiCodeGenerator(HabanaGraph* graph)
{
    return std::make_unique<GaudiCodeGenerator>(graph);
}

GaudiCodeGenerator::GaudiCodeGenerator(HabanaGraph* graph) : CodeGenerator(graph)
{
    m_workspaceAllocator = createAllocator(MEMORY_HEAP_NON_CYCLIC_ALLOCATOR, "HBM_WORKSPACE");
    m_programAllocator   = createAllocator(MEMORY_SLAB_ALLOCATOR, "HBM_PROGRAM");
    m_sramAllocator      = createAllocator(MEMORY_HEAP_ALLOCATOR, "SRAM");
    m_syncObjectManager   = std::make_shared<SyncObjectManager>(graph->getHALReader()->getNumSyncObjects(),
                                                                graph->getHALReader()->getNumMonitors(),
                                                                gaudi::SyncConventions::instance());
    m_monitorSetupManager = std::make_shared<MonitorSetupManagerGaudi>(m_syncObjectManager);
}

GaudiCodeGenerator::GaudiCodeGenerator(const GaudiCodeGenerator& other,
                                       HabanaGraph*              graph,
                                       bool                      cloneAllocators /*false*/)
: CodeGenerator(other, graph)
{
    m_syncObjectManager = std::make_shared<SyncObjectManager>(*other.m_syncObjectManager);
    m_monitorSetupManager = std::make_shared<MonitorSetupManagerGaudi>(*other.m_monitorSetupManager, m_syncObjectManager);
    if (cloneAllocators)
    {
        m_workspaceAllocator = other.m_workspaceAllocator->Clone();
        m_programAllocator   = other.m_programAllocator->Clone();
        m_sramAllocator      = other.m_sramAllocator->Clone();
    }
    else
    {
        m_workspaceAllocator = createAllocator(MEMORY_HEAP_NON_CYCLIC_ALLOCATOR, "HBM_WORKSPACE");
        m_programAllocator   = createAllocator(MEMORY_SLAB_ALLOCATOR, "HBM_PROGRAM");
        m_sramAllocator      = createAllocator(MEMORY_HEAP_ALLOCATOR, "SRAM");
        if (other.m_allocatorsWereInit) initAllocators();
    }
}

CodeGeneratorPtr GaudiCodeGenerator::clone(HabanaGraph* graph, bool cloneAllocators /*false*/) const
{
    return CodeGeneratorPtr(new GaudiCodeGenerator(*this, graph, cloneAllocators));
}

GaudiCodeGenerator& GaudiCodeGenerator::operator=(const GaudiCodeGenerator& other)
{
    if (this != &other)
    {
        CodeGenerator::operator=(other);
        GaudiCodeGenerator tmp(other, other.m_graph);
        m_commandQueueById.swap(tmp.m_commandQueueById);
        std::swap(m_completionQueue, tmp.m_completionQueue);
        std::swap(m_mmeDispatcher, tmp.m_mmeDispatcher);
        std::swap(m_tpcDispatcher, tmp.m_tpcDispatcher);
        std::swap(m_dmaDispatchers, tmp.m_dmaDispatchers);
        std::swap(m_workspaceAllocator, tmp.m_workspaceAllocator);
        std::swap(m_programAllocator, tmp.m_programAllocator);
        std::swap(m_sramAllocator, tmp.m_sramAllocator);
        std::swap(m_recipeGenerator, tmp.m_recipeGenerator);
        std::swap(m_syncObjectManager, tmp.m_syncObjectManager);
        std::swap(m_monitorSetupManager, tmp.m_monitorSetupManager);
    }
    return *this;
}

void GaudiCodeGenerator::initTPCQueues()
{
    std::vector<std::vector<QueueCommandPtr>> cmdsToQueues = getInitTPCQueuesCmds();
    std::vector<QueueCommandPtr> loadPredicates = getLoadPredicateCmds(m_graph->getHALReader()->getNumTpcEngines());

    for (unsigned tpcEng = 0; tpcEng < m_tpcDispatcher->getNumEngines(); ++tpcEng)
    {
        const CommandQueuePtr&        queue       = m_tpcDispatcher->getQueue(tpcEng);
        std::vector<QueueCommandPtr>& cmdsToQueue = cmdsToQueues[tpcEng];

        // Load predicate according to engine ID
        cmdsToQueue.push_back(std::move(loadPredicates[queue->GetEngineID()]));

        // Put all TPC init commands in the queue's init commands vector
        for (auto& cmd : cmdsToQueue)
        {
            queue->getInitialQueueCommands().emplace_back(std::move(cmd), false);
        }
    }
}

std::vector<QueueCommandPtr> GaudiCodeGenerator::getLoadPredicateCmds(unsigned numPreds, unsigned firstPredVal /*=1*/)
{
    static const uint32_t ADDR_DWORD_OFFSET_FROM_HEADER = 0;

    deviceAddrOffset predTableDevAddr = getPredTableDeviceAddr(numPreds, firstPredVal);

    std::vector<QueueCommandPtr> ret;

    for (uint64_t lineIdx = 0; lineIdx < numPreds; ++lineIdx)
    {
        deviceAddrOffset predLineAddr =
            predTableDevAddr + (lineIdx * m_graph->getHALReader()->getCacheLineSizeInBytes());
        QueueCommandPtr          loadPred = std::make_unique<gaudi::LoadPredicates>(predLineAddr);
        BasicFieldsContainerInfo afci;
        uint64_t                 memID = getMemoryIDFromVirtualAddress(predLineAddr);
        afci.addAddressEngineFieldInfo(nullptr,
                                       getMemorySectionNameForMemoryID(memID),
                                       memID,
                                       predLineAddr,
                                       ADDR_DWORD_OFFSET_FROM_HEADER,
                                       FIELD_MEMORY_TYPE_DRAM);
        loadPred->SetContainerInfo(afci);
        ret.push_back(std::move(loadPred));
    }

    return ret;
}

deviceAddrOffset GaudiCodeGenerator::getPredTableDeviceAddr(unsigned numPreds, unsigned firstPredVal)
{
    unsigned i = 0;
    for (; i < m_predicateTables.size(); ++i)
    {
        const PredicateTable& existTable = m_predicateTables[i];

        // Check if an existing table can cover the requested predicates
        if (firstPredVal >= existTable.firstPredVal &&                                 // lower bound check
            firstPredVal + numPreds <= existTable.firstPredVal + existTable.numPreds)  // upper bound check
        {
            break;  // yes, we found an existing table that can serve us
        }
    }

    if (i < m_predicateTables.size())
    {
        return m_predicateTables[i].deviceAddr;
    }
    else
    {
        return createPredicateTable(numPreds, firstPredVal);
    }
}

// Create predicate table and return its device address
deviceAddrOffset GaudiCodeGenerator::createPredicateTable(unsigned numPreds, unsigned firstPredVal)
{
    HB_ASSERT(firstPredVal > 0, "predicate 0 is reserved and cannot be used by SW");
    HB_ASSERT(numPreds < m_graph->getHALReader()->getNumPredicateBits(),
              "max num of predicates is {}",
              m_graph->getHALReader()->getNumPredicateBits());

    // Allocate host memory
    uint64_t predTableSizeInBytes = m_graph->getHALReader()->getCacheLineSizeInBytes() * (uint64_t)numPreds;
    std::shared_ptr<char> hostTable(new char[predTableSizeInBytes], [](char* p) { delete[] p; });

    // Fill the predicate table in host memory
    // Example:
    //   The following is a table of 8 predicates. Note that predicate 0 is reserved and shouldn't be used.
    //   Each digit represents a 32bit value, total of 128 bytes per line.
    //   Currently we only support single predicate per line.
    //
    //   31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
    //   -----------------------------------------------------------------------------------------------
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
    //
    memset(hostTable.get(), 0, predTableSizeInBytes);
    uint32_t* p = (uint32_t*)hostTable.get();
    for (unsigned i = 0, j = firstPredVal; i < numPreds; ++i, ++j)
    {
        p[j] = 1;
        p += (m_graph->getHALReader()->getCacheLineSizeInBytes() / sizeof(uint32_t));  // move to next line
    }

    // Allocate HBM memory
    Settable<deviceAddrOffset> deviceAddr = m_programAllocator->Allocate(predTableSizeInBytes,
                                                               m_graph->getHALReader()->getCacheLineSizeInBytes());
    HB_ASSERT(deviceAddr.is_set(), "Failed to allocate predicate table on HBM");

    // Register the table as data-blob so it will get downloaded to the HBM
    registerProgramDataBlobForDownload(hostTable, deviceAddr.value(), predTableSizeInBytes);

    // Save the table in class member
    m_predicateTables.emplace_back(numPreds, firstPredVal, deviceAddr.value(), hostTable);

    return deviceAddr.value();
}

void GaudiCodeGenerator::addAllDescriptors()
{
    DescriptorGenerator generator(this);
    for (auto n : m_graph->getNodes())
    {
        n->accept(&generator);
    }
}

void GaudiCodeGenerator::updateMmeNodeDescriptorWrapper(const MmeNode&        node,
                                                        const gaudi::MmeDesc& mmeDesciptor,
                                                        NodeROI&              roi)
{
    NodePtr nodeShared = getNodeSharedPtr(node);
    HB_ASSERT_PTR(nodeShared);

    MmeDescWrapper descWrapper(mmeDesciptor);
    descWrapper.getBasicFieldsContainerInfo().setRoi(&roi);

    gaudi::GaudiMMEPatchPointGenerator ppGenerator;
    ppGenerator.generateMmePatchPoints(node, descWrapper);

    m_mmeNodesDescriptorsWrappers[nodeShared].push_back(descWrapper);
}

void GaudiCodeGenerator::updateTPCDescriptorWrapper(const TPCNode&                      node,
                                                    const gaudi::TpcDesc&               tpcDescriptor,
                                                    const ValidityMask<gaudi::TpcDesc>& tpcMask,
                                                    NodeROI&                            roi)
{
    NodePtr nodeShared = getNodeSharedPtr(node);
    HB_ASSERT_PTR(nodeShared);

    TPCDescWrapper descWrapper(tpcDescriptor, tpcMask);
    descWrapper.getExecutionInfo().pipelineLevel = roi.pipelineLevel;
    descWrapper.getBasicFieldsContainerInfo().setRoi(&roi);

    gaudi::GaudiTPCPatchPointGenerator ppGenerator;
    ppGenerator.generateTpcPatchPoints(node, descWrapper);
    getTPCNodeDescriptorsWrappers(nodeShared).push_back(descWrapper);
}

void GaudiCodeGenerator::updateDMADescriptorWrapper(const DMANode&                      node,
                                                    const gaudi::DmaDesc&               dmaDescriptor,
                                                    const ValidityMask<gaudi::DmaDesc>& dmaMask,
                                                    NodeROI&                            roi)
{
    NodePtr nodeShared = getNodeSharedPtr(node);
    HB_ASSERT_PTR(nodeShared);

    DMADescWrapper descWrapper(dmaDescriptor, dmaMask);
    descWrapper.getExecutionInfo().pipelineLevel = roi.pipelineLevel;
    descWrapper.getBasicFieldsContainerInfo().setRoi(&roi);

    GaudiDMAPatchPointGenerator ppGenerator;
    ppGenerator.generateDmaPatchPoints(node, descWrapper);
    getDMANodeDescriptorsWrappers(nodeShared).push_back(descWrapper);
}

TPCDescriptorsWrappers& GaudiCodeGenerator::getTPCNodeDescriptorsWrappers(const NodePtr& n)
{
    return m_tpcNodesDescriptorsWrappers[n];
}

DMADescriptorsWrappers& GaudiCodeGenerator::getDMANodeDescriptorsWrappers(const NodePtr& n)
{
    return m_dmaNodesDescriptorsWrappers[n];
}

MmeDescriptorsWrappers& GaudiCodeGenerator::getMmeNodeDescriptorsWrappers(const NodePtr& n)
{
    return m_mmeNodesDescriptorsWrappers[n];
}

void GaudiCodeGenerator::initAllocators()
{
    // At this point we give the allocators the full HBM size, later we will check that we didn't exceed the max size
    m_workspaceAllocator->Init(m_dramSize, getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE));
    m_programAllocator->Init(m_dramSize, getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA));
    m_sramAllocator->Init(m_sramSize, m_synapseSramBaseAddr);
    m_allocatorsWereInit = true;
}

void GaudiCodeGenerator::generateRecipes(const HabanaGraph& graph)
{
    // Generate the recipe in internal representation, the serialize() will put it into a recipe_t structure
    m_recipeGenerator.reset(new gaudi::GaudiRecipeGenerator(&graph));

    m_recipeGenerator->generateRecipes(graph.isDynamicShape());

    m_recipeGenerator->print();
}

recipe_t* GaudiCodeGenerator::serializeDataPlane(RecipeAllocator* recipeAlloc) const
{
    CHECK_RET_NULL(m_recipeGenerator, "Invoke compile before serialize");
    return m_recipeGenerator->serializeDataPlaneGraph(recipeAlloc);
}

shape_plane_graph_t* GaudiCodeGenerator::serializeShapePlane(RecipeAllocator* recipeAlloc) const
{
    CHECK_RET_NULL(m_recipeGenerator, "Invoke compile before serialize");
    return m_recipeGenerator->serializeShapePlane(recipeAlloc);
}

void GaudiCodeGenerator::initQueues()
{
    m_completionQueue.reset(new gaudi::CompletionQueue());
    m_mmeDispatcher = std::make_shared<gaudi::MmeDispatcher>(GCFG_MME_SYNC_TRACE_EN_MASK.value(), m_graph); //temporary using graph
    m_tpcDispatcher = std::make_shared<gaudi::TPCDispatcher>(GCFG_TPC_ENGINES_ENABLED_MASK.value(),
                                                             GCFG_TPC_SYNC_TRACE_EN_MASK.value(),
                                                             m_graph);

    HB_ASSERT_PTR(m_mmeDispatcher);
    HB_ASSERT_PTR(m_completionQueue);
    HB_ASSERT_PTR(m_tpcDispatcher);

    addCommandQueue(m_completionQueue);
    registerDispatcher(*m_mmeDispatcher);
    registerDispatcher(*m_tpcDispatcher);
    for (const std::pair<const QueueDispatcherParams, QueueDispatcherPtr>& dispatcher : m_dmaDispatchers)
    {
        HB_ASSERT_PTR(dispatcher.second);
        registerDispatcher(*(dispatcher.second));
    }
    setupQueuesMonitors();

    CodeGenerator::initQueues();
    initTPCQueues();
    finalizeInitQueues(false);
}

void GaudiCodeGenerator::fillQueuesWithDmaNode(NodePtr node)
{
    // Dispatch internal DMA
    std::shared_ptr<DMANode> dmaNode = std::dynamic_pointer_cast<DMANode>(node);
    if (dmaNode->getDmaType() == DMA_TYPE_INTERNAL)
    {
        QueueDispatcherParams params = QueueDispatcherParams(dmaNode->parallelLevel(), dmaNode->dispatcherIndex());
        HB_ASSERT(m_dmaDispatchers.count(params) != 0,
                  "node name: {}, have level: {}, and index: {}, but not exists dispatcher with those params",
                  node->getNodeName(),
                  dmaNode->parallelLevel(),
                  dmaNode->dispatcherIndex());
        m_dmaDispatchers[params]->dispatchNode(node, m_graph, false);
    }
}

void GaudiCodeGenerator::addExecuteDMANode(NodePtr n, uint32_t* inputDmaInd, uint32_t* outputDmaInd)
{
    fillQueuesWithDmaNode(n);
}

void GaudiCodeGenerator::fillQueues()
{
    // Base class handles common queues
    CodeGenerator::fillQueues();

    finalizeFillQueues();
}

unsigned GaudiCodeGenerator::getQueueID(HabanaDeviceType type, unsigned id)
{
    return gaudi::getQueueID(type, id);
}