#include "gaudi3_code_generator.h"

#include "code_generation/code_generator_factory.h"
#include "gaudi3_eng_arc_command.h"
#include "gaudi3_graph.h"
#include "hal_conventions.h"
#include "mme_dispatcher.h"
#include "recipe_allocator.h"
#include "rotator_dispatcher.h"
#include "tpc_dispatcher.h"
#include "types_exception.h"

CodeGeneratorPtr CodeGeneratorFactory::instantiateGaudi3CodeGenerator(HabanaGraph* graph)
{
    return std::make_unique<Gaudi3CodeGenerator>(graph);
}

Gaudi3CodeGenerator::Gaudi3CodeGenerator(HabanaGraph* graph) : CodeGenerator(graph)
{
    m_workspaceAllocator = createAllocator(MEMORY_HEAP_NON_CYCLIC_ALLOCATOR, "HBM_WORKSPACE");
    m_programAllocator   = createAllocator(MEMORY_SLAB_ALLOCATOR, "HBM_PROGRAM");
    m_sramAllocator      = createAllocator(MEMORY_HEAP_ALLOCATOR, "SRAM");
}

Gaudi3CodeGenerator::Gaudi3CodeGenerator(const Gaudi3CodeGenerator& other,
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

Gaudi3CodeGenerator& Gaudi3CodeGenerator::operator=(const Gaudi3CodeGenerator& other)
{
    if (this != &other)
    {
        CodeGenerator::operator=(other);
        Gaudi3CodeGenerator tmp(other, other.m_graph);
        std::swap(m_mmeDispatcher, tmp.m_mmeDispatcher);
        std::swap(m_tpcDispatcher, tmp.m_tpcDispatcher);
        std::swap(m_rotatorDispatcher, tmp.m_rotatorDispatcher);
        std::swap(m_workspaceAllocator, tmp.m_workspaceAllocator);
        std::swap(m_programAllocator, tmp.m_programAllocator);
        std::swap(m_sramAllocator, tmp.m_sramAllocator);
        std::swap(m_recipeGenerator, tmp.m_recipeGenerator);
    }
    return *this;
}

CodeGeneratorPtr Gaudi3CodeGenerator::clone(HabanaGraph* graph, bool cloneAllocators /*false*/) const
{
    return CodeGeneratorPtr(new Gaudi3CodeGenerator(*this, graph, cloneAllocators));
}

void Gaudi3CodeGenerator::initAllocators()
{
    m_workspaceAllocator->Init(m_dramSize, getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE));
    m_programAllocator->Init(m_dramSize, getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA));
    m_sramAllocator->Init(m_sramSize, m_synapseSramBaseAddr);
    m_allocatorsWereInit = true;
}

void Gaudi3CodeGenerator::generateRecipes(const HabanaGraph& graph)
{
    // Generate the recipe in internal representation, the serialize() will put it into a recipe_t structure
    m_recipeGenerator.reset(new gaudi3::Gaudi3RecipeGenerator(&graph));

    m_recipeGenerator->generateRecipes(graph.isDynamicShape());

    m_recipeGenerator->print();
}

void Gaudi3CodeGenerator::generateCmeCommands(const NodePtr& n)
{
    PhysicalMcid physicalMcid;
    bool         changeToDegrade;

    if (n->isLogicalOperation()) return;
    auto rois = n->getLogicalRois();
    HB_ASSERT_PTR(rois);

    for (const NodeROI& roi : *rois)
    {
        // Process each ROI cme tasks
        // Cache maintenance commands handling is first
        for (const CmCmd& cmCmd : roi.cmeTasks.cmCmds)
        {
            if (cmCmd.op == DEGRADE)
            {
                m_mcidConverter.convertDegrade(cmCmd.mcid, physicalMcid);
                m_cmeCommands.push_back(std::make_shared<Gaudi3CmeDegradeArcCommand>(cmCmd.deps, physicalMcid));
            }
            else if (cmCmd.op == DISCARD)
            {
                m_mcidConverter.convertReleaseDiscard(cmCmd.mcid, physicalMcid, changeToDegrade);

                if (changeToDegrade)
                {
                    m_cmeCommands.push_back(std::make_shared<Gaudi3CmeDegradeArcCommand>(cmCmd.deps, physicalMcid, true));
                }
                else
                {
                    m_cmeCommands.push_back(std::make_shared<Gaudi3CmeDiscardArcCommand>(cmCmd.deps, physicalMcid));
                }
            }
        }
        // Rollover handling is second
        if (roi.cmeTasks.rollover.doRollover)
        {
            bool incMme = roi.cmeTasks.rollover.rolloverEngineBitmap & 1; // bit 0 for mme
            bool incRot = roi.cmeTasks.rollover.rolloverEngineBitmap & 2; // bit 1 for rot
            m_cmeCommands.push_back(std::make_shared<Gaudi3CmeMcidRolloverArcCommand>(incMme, incRot));
            m_mcidConverter.slideRolloverWindow();
            LOG_DEBUG(CACHE_MAINT, "Adding mcid rollover cmd (rolloverId={}) to CME", roi.cmeTasks.rollover.rolloverId);
        }
        // SOB reset handling must be last
        if (roi.cmeTasks.sobReset.sobResetTotalNumEngs > 0)
        {
            m_cmeCommands.push_back(std::make_shared<Gaudi3CmeResetSobsArcCommand>(roi.cmeTasks.sobReset.sobResetTotalNumEngs));
            LOG_DEBUG(CACHE_MAINT, "Adding sob reset cmd (resetId={}) to CME", roi.cmeTasks.sobReset.sobResetId);
        }
    }
}

recipe_t* Gaudi3CodeGenerator::serializeDataPlane(RecipeAllocator* recipeAlloc) const
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

void Gaudi3CodeGenerator::initQueues()
{
    m_mmeDispatcher = std::make_shared<gaudi3::MmeDispatcher>(GCFG_MME_SYNC_TRACE_EN_MASK.value(), m_graph); //temporary using graph
    m_tpcDispatcher = std::make_shared<gaudi3::TpcDispatcher>(GCFG_TPC_ENGINES_ENABLED_MASK.value() &
                                                                  m_graph->getHALReader()->getTpcEnginesMask(),
                                                              GCFG_TPC_SYNC_TRACE_EN_MASK.value(),
                                                              m_graph);
    m_rotatorDispatcher = std::make_shared<gaudi3::RotatorDispatcher>(GCFG_ROTATOR_SYNC_TRACE_EN_MASK.value(), m_graph);
    HB_ASSERT_PTR(m_mmeDispatcher);
    HB_ASSERT_PTR(m_tpcDispatcher);
    HB_ASSERT_PTR(m_rotatorDispatcher);
    registerDispatcher(*m_mmeDispatcher);
    registerDispatcher(*m_tpcDispatcher);

    // PLDM image does not include rotator
    if (!GCFG_RUNNING_ON_PLDM.value())
    {
        registerDispatcher(*m_rotatorDispatcher);
    }

    initTPCQueues();
    finalizeInitQueues(false);
}

void Gaudi3CodeGenerator::addExecuteDMANode(NodePtr n, uint32_t* inputDmaInd, uint32_t* outputDmaInd)
{
    HB_ASSERT(0, "Gaudi3 graph should not contain DMA nodes");
}

void Gaudi3CodeGenerator::fillQueues()
{
    // Base class handles common queues
    CodeGenerator::fillQueues();
    finalizeFillQueues();
}

unsigned Gaudi3CodeGenerator::getQueueID(HabanaDeviceType type, unsigned id)
{
    return gaudi3::getQueueID(type, id);
}

shape_plane_graph_t* Gaudi3CodeGenerator::serializeShapePlane(RecipeAllocator* recipeAlloc) const
{
    CHECK_RET_NULL(m_recipeGenerator, "Invoke compile before serialize");
    return m_recipeGenerator->serializeShapePlane(recipeAlloc);
}

// Remove redundant dependencies from the input map using archived overlap data-base identified by overlapId.
// A redundant dependency is a dependency that is already satisfied by another dependency in the map.
void Gaudi3CodeGenerator::removeRedundantDependencies(DependencyMap& depMap, unsigned overlapId) const
{
    // sanity check for input
    std::for_each(depMap.begin(), depMap.end(), [](auto& d) { HB_ASSERT(d.first < gaudi3::LOGICAL_QUEUE_MAX_ID, "OOB"); });
    const gaudi3::Overlap& overlap = *m_overlapArchive.at(overlapId);
    bool toBeRemoved[gaudi3::LOGICAL_QUEUE_MAX_ID] = {0};

    for (auto pivot : depMap)
    {
        unsigned pivotEngineId = pivot.first;
        if (toBeRemoved[pivotEngineId]) continue;
        unsigned pivotSignalIdx = pivot.second - 1; // convert from "1-based signal realm" to "0-based index realm"
        const gaudi3::Overlap::DependencyCtx& depCtx = overlap.getSignalCtx(pivotEngineId, pivotSignalIdx);

        // compare pivot dependency to all candidate dependencies and collect the redundancies
        for (auto candidate : depMap)
        {
            unsigned engineId = candidate.first;
            if (toBeRemoved[engineId] || engineId == pivotEngineId) continue;
            unsigned signalIdx = candidate.second - 1; // convert from "1-based signal realm" to "0-based index realm"
            if (depCtx.valid[engineId] && signalIdx <= depCtx.signalIdx[engineId]) toBeRemoved[engineId] = true;
        }
    }

    // erase redundancies in-place
    for (unsigned engId = 0; engId < gaudi3::LOGICAL_QUEUE_MAX_ID; engId++)
    {
        if (toBeRemoved[engId]) depMap.erase(engId);
    }
}
