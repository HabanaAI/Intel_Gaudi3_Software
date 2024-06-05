#include "gaudi2_code_generator.h"

#include "code_generation/code_generator_factory.h"
#include "code_generator.h"
#include "patch_point_generator.h"
#include "descriptor_generator.h"
#include "gaudi2_graph.h"
#include "hal_conventions.h"
#include "mme_dispatcher.h"
#include "recipe_allocator.h"
#include "rotator_dispatcher.h"
#include "tpc_dispatcher.h"
#include "types_exception.h"

CodeGeneratorPtr CodeGeneratorFactory::instantiateGaudi2CodeGenerator(HabanaGraph* graph)
{
    return std::make_unique<Gaudi2CodeGenerator>(graph);
}

Gaudi2CodeGenerator::Gaudi2CodeGenerator(HabanaGraph* graph) : CodeGenerator(graph)
{
    m_workspaceAllocator = createAllocator(MEMORY_HEAP_NON_CYCLIC_ALLOCATOR, "HBM_WORKSPACE");
    m_programAllocator   = createAllocator(MEMORY_SLAB_ALLOCATOR, "HBM_PROGRAM");
    m_sramAllocator      = createAllocator(MEMORY_HEAP_ALLOCATOR, "SRAM");
}

Gaudi2CodeGenerator::Gaudi2CodeGenerator(const Gaudi2CodeGenerator& other,
                                         HabanaGraph*               graph,
                                         bool                       cloneAllocators /*false*/)
: CodeGenerator(other, graph)
{
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

Gaudi2CodeGenerator& Gaudi2CodeGenerator::operator=(const Gaudi2CodeGenerator& other)
{
    if (this != &other)
    {
        CodeGenerator::operator=(other);
        Gaudi2CodeGenerator tmp(other, other.m_graph);
        std::swap(m_tpcDispatcher, tmp.m_tpcDispatcher);
        std::swap(m_mmeDispatcher, tmp.m_mmeDispatcher);
        std::swap(m_dmaDispatchers, tmp.m_dmaDispatchers);
        std::swap(m_rotatorDispatcher, tmp.m_rotatorDispatcher);
        std::swap(m_workspaceAllocator, tmp.m_workspaceAllocator);
        std::swap(m_programAllocator, tmp.m_programAllocator);
        std::swap(m_sramAllocator, tmp.m_sramAllocator);
        std::swap(m_recipeGenerator, tmp.m_recipeGenerator);
    }
    return *this;
}

CodeGeneratorPtr Gaudi2CodeGenerator::clone(HabanaGraph* graph, bool cloneAllocators /*false*/) const
{
    return CodeGeneratorPtr(new Gaudi2CodeGenerator(*this, graph, cloneAllocators));
}

void Gaudi2CodeGenerator::initAllocators()
{
    m_workspaceAllocator->Init(m_dramSize, getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE));
    m_programAllocator->Init(m_dramSize, getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA));
    m_sramAllocator->Init(m_sramSize, m_synapseSramBaseAddr);
    m_allocatorsWereInit = true;
}

void Gaudi2CodeGenerator::generateRecipes(const HabanaGraph& graph)
{
    // Generate the recipe in internal representation, the serialize() will put it into a recipe_t structure
    m_recipeGenerator.reset(new gaudi2::Gaudi2RecipeGenerator(&graph));

    m_recipeGenerator->generateRecipes(graph.isDynamicShape());

    m_recipeGenerator->print();
}

recipe_t* Gaudi2CodeGenerator::serializeDataPlane(RecipeAllocator* recipeAlloc) const
{
    CHECK_RET_NULL(m_recipeGenerator, "Invoke compile before serialize");
    try
    {
        return m_recipeGenerator->serializeDataPlaneGraph(recipeAlloc);
    }
    catch (const SynapseException& exc)
    {
        LOG_ERR(SYN_API, "Failed to serializeDataPlaneGraph due to: {}", exc.what());
        recipeAlloc->freeAll();
    }

    return nullptr;
}

shape_plane_graph_t* Gaudi2CodeGenerator::serializeShapePlane(RecipeAllocator* recipeAlloc) const
{
    CHECK_RET_NULL(m_recipeGenerator, "Invoke compile before serialize");
    return m_recipeGenerator->serializeShapePlane(recipeAlloc);
}
void Gaudi2CodeGenerator::setupQueuesMonitors()
{
}

void Gaudi2CodeGenerator::initQueues()
{
    m_mmeDispatcher = std::make_shared<gaudi2::MmeDispatcher>(GCFG_MME_SYNC_TRACE_EN_MASK.value(), m_graph); //temporary using graph
    m_tpcDispatcher = std::make_shared<gaudi2::TpcDispatcher>(GCFG_TPC_ENGINES_ENABLED_MASK.value() &
                                                                  m_graph->getHALReader()->getTpcEnginesMask(),
                                                              GCFG_TPC_SYNC_TRACE_EN_MASK.value(),
                                                              m_graph);
    m_rotatorDispatcher = std::make_shared<gaudi2::RotatorDispatcher>(GCFG_ROTATOR_SYNC_TRACE_EN_MASK.value(), m_graph);

    HB_ASSERT_PTR(m_mmeDispatcher);
    HB_ASSERT_PTR(m_tpcDispatcher);
    HB_ASSERT_PTR(m_rotatorDispatcher);
    registerDispatcher(*m_mmeDispatcher);
    registerDispatcher(*m_tpcDispatcher);

    for (const std::pair<const QueueDispatcherParams, QueueDispatcherPtr>& dispatcher : m_dmaDispatchers)
    {
        HB_ASSERT_PTR(dispatcher.second);
        registerDispatcher(*(dispatcher.second));
    }

    // PLDM image does not include rotator
    if (!GCFG_RUNNING_ON_PLDM.value())
    {
        registerDispatcher(*m_rotatorDispatcher);
    }
    setupQueuesMonitors();
    initTPCQueues();

    CodeGenerator::initQueues();

    finalizeInitQueues(false);

    // Add the additional SFG init command as the very last init command
    addSFGInitCmd();
}

void Gaudi2CodeGenerator::addSFGInitCmd()
{
    for (const std::pair<const uint32_t, CommandQueuePtr>& queue : getCommandQueueById())
    {
        HabanaDeviceType devType = queue.second->GetDeviceType();

        if (m_deviceSfgInitValue.find(devType) != m_deviceSfgInitValue.end())
        {
            unsigned        initVal = m_deviceSfgInitValue[devType];
            QueueCommandPtr sfgCmd  = getCommandFactory().getSfgInit(initVal);
            // Do not set switch-bit as this command will be pushed to dynamic blob just before nop-switch_cq command
            queue.second->PushBack(std::move(sfgCmd), false);
        }
    }
}

void Gaudi2CodeGenerator::addAllDescriptors()
{
    gaudi2::DescriptorGenerator generator(this);
    for (auto n : m_graph->getExeSortedNodes())  // must be in execution order
    {
        n->accept(&generator);
    }
    LOG_DEBUG(GC, "total DMA descriptors in graph  {}", getNumDmaNodeDescriptorsWrappers());
}

gaudi2::TpcDescriptorsWrappers& Gaudi2CodeGenerator::getTpcNodeDescriptorsWrappers(const NodePtr& n)
{
    return m_tpcNodesDescriptorsWrappers[n];
}
gaudi2::DmaDescriptorsWrappers& Gaudi2CodeGenerator::getDmaNodeDescriptorsWrappers(const NodePtr& n)
{
    return m_dmaNodesDescriptorsWrappers[n];
}
gaudi2::MmeDescriptorsWrappers& Gaudi2CodeGenerator::getMmeNodeDescriptorsWrappers(const NodePtr& n)
{
    return m_mmeNodesDescriptorsWrappers[n];
}
gaudi2::RotatorDescriptorsWrappers& Gaudi2CodeGenerator::getRotateNodeDescriptorsWrappers(const NodePtr& n)
{
    return m_rotNodesDescriptorsWrappers[n];
}

unsigned Gaudi2CodeGenerator::getNumDmaNodeDescriptorsWrappers()
{
    return m_dmaNodesDescriptorsWrappers.size();
}

void Gaudi2CodeGenerator::updateTPCDescriptorWrapper(const TPCNode&                       node,
                                                     const gaudi2::TpcDesc&               tpcDescriptor,
                                                     const ValidityMask<gaudi2::TpcDesc>& tpcMask,
                                                     const tpc_wd_ctxt_t&                 tpcFwCtx,
                                                     NodeROI&                             roi)
{
    NodePtr nodeShared = getNodeSharedPtr(node);
    HB_ASSERT_PTR(nodeShared);
    gaudi2::TpcDescWrapper descWrapper(tpcDescriptor, tpcMask);
    descWrapper.getExecutionInfo().pipelineLevel = roi.pipelineLevel;
    descWrapper.getBasicFieldsContainerInfo().setRoi(&roi);
    descWrapper.getBasicFieldsContainerInfoForCtx().setRoi(&roi);
    descWrapper.setFwCtx(tpcFwCtx);

    gaudi2::Gaudi2TPCPatchPointGenerator ppGenerator;
    ppGenerator.generateTpcPatchPoints(node, descWrapper);
    getTpcNodeDescriptorsWrappers(nodeShared).push_back(descWrapper);
}

void Gaudi2CodeGenerator::updateDMADescriptorWrapper(const DMANode&                       node,
                                                     const gaudi2::DmaDesc&               dmaDescriptor,
                                                     const ValidityMask<gaudi2::DmaDesc>& dmaMask,
                                                     const edma_wd_ctxt_t&                dmaFwCtx,
                                                     NodeROI&                             roi)
{
    NodePtr nodeShared = getNodeSharedPtr(node);
    HB_ASSERT_PTR(nodeShared);

    gaudi2::DmaDescWrapper descWrapper(dmaDescriptor, dmaMask);
    descWrapper.getExecutionInfo().pipelineLevel = roi.pipelineLevel;
    descWrapper.getBasicFieldsContainerInfo().setRoi(&roi);
    descWrapper.getBasicFieldsContainerInfoForCtx().setRoi(&roi);
    descWrapper.setFwCtx(dmaFwCtx);

    gaudi2::Gaudi2DMAPatchPointGenerator ppGenerator;
    ppGenerator.generateDmaPatchPoints(node, descWrapper);
    getDmaNodeDescriptorsWrappers(nodeShared).push_back(descWrapper);
}

void Gaudi2CodeGenerator::updateMmeNodeDescriptorWrapper(const MmeNode& node, const gaudi2::MmeDesc& desc, NodeROI& roi)
{
    NodePtr nodeShared = getNodeSharedPtr(node);
    HB_ASSERT(nodeShared != nullptr, "Invalid node object");

    gaudi2::MmeDescWrapper descWrapper(desc);
    descWrapper.getBasicFieldsContainerInfo().setRoi(&roi);
    descWrapper.getBasicFieldsContainerInfoForCtx().setRoi(&roi);

    gaudi2::Gaudi2MMEPatchPointGenerator ppGenerator;
    ppGenerator.generateMmePatchPoints(node, getMmeNodeDescriptorGenerator(nodeShared), descWrapper);

    getMmeNodeDescriptorsWrappers(nodeShared).push_back(descWrapper);
}

void Gaudi2CodeGenerator::updateRotatorDescriptorWrapper(const RotateNode&                        node,
                                                         const gaudi2::RotatorDesc&               rotatorDescriptor,
                                                         const ValidityMask<gaudi2::RotatorDesc>& rotateMask,
                                                         const rot_wd_ctxt_t&                     rotFwCtx,
                                                         NodeROI&                                 roi)
{
    NodePtr nodeShared = getNodeSharedPtr(node);
    HB_ASSERT_PTR(nodeShared);

    gaudi2::RotatorDescWrapper descWrapper(rotatorDescriptor, rotateMask);
    descWrapper.getBasicFieldsContainerInfo().setRoi(&roi);
    descWrapper.setFwCtx(rotFwCtx);
    descWrapper.getExecutionInfo().pipelineLevel = roi.pipelineLevel;

    gaudi2::Gaudi2RotatorPatchPointGenerator ppGenerator;
    ppGenerator.generateRotatorPatchPoints(node, descWrapper);
    // [CID: 42187] False positive - coverity ignores std::map and std::set default c'tor
    getRotateNodeDescriptorsWrappers(nodeShared).push_back(descWrapper);
}

void Gaudi2CodeGenerator::setMmeNodeDescriptorGenerator(const NodePtr& n, gaudi2::MmeDescriptorGeneratorPtr& descGenerator)
{
    HB_ASSERT(m_mmeNodesDescriptorGenerator.find(n) == m_mmeNodesDescriptorGenerator.end(), "object already in map");
    m_mmeNodesDescriptorGenerator.emplace(n, descGenerator);
}

gaudi2::MmeDescriptorGenerator& Gaudi2CodeGenerator::getMmeNodeDescriptorGenerator(const NodePtr& n)
{
    auto descGenerator = m_mmeNodesDescriptorGenerator.find(n);
    HB_ASSERT(descGenerator != m_mmeNodesDescriptorGenerator.end(), "object not in map");
    return *descGenerator->second;
}

unsigned Gaudi2CodeGenerator::getQueueID(HabanaDeviceType type, unsigned id)
{
    return gaudi2::getQueueID(type, id);
}

void Gaudi2CodeGenerator::addInitialSyncs(bool bIsActivate /*false*/)
{
    CodeGenerator::addInitialSyncs(bIsActivate);
}

void Gaudi2CodeGenerator::fillQueuesWithDmaNode(NodePtr node)
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

void Gaudi2CodeGenerator::addFinalSyncs(bool bIsActivate /*false*/)
{
    if (getFinalSyncInstructions().empty()) return;

    HB_ASSERT(getFinalSyncInstructions().size() == 1, "only 1 queue with final syncs is expected (1 completion flow)");

    // Push the completion flow to the queue
    CodeGenerator::addFinalSyncs(bIsActivate);
}

void Gaudi2CodeGenerator::fillQueues()
{
    // Base class handles common queues
    CodeGenerator::fillQueues();
    finalizeFillQueues();
}