#include "eager_graph.h"

// eager includes (relative to src/eager/lib/)
#include "chip_info.h"
#include "debug_tools/eager_graph_visualization.h"
#include "desc_gen/node2desc.h"
#include "node_info/eager_node.h"
#include "node_info/node_displacement.h"
#include "recipe_gen/recipe_templates.h"
#include "utils/general_defs.h"
#include "utils/numeric_utils.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/compilation_hal_reader.h"
#include "graph_compiler/graph_traits.h"
#include "graph_compiler/graph.h"
#include "graph_compiler/habana_global_conf.h"
#include "graph_compiler/habana_graph.h"
#include "graph_compiler/habana_nodes/transpose_node.h"
#include "graph_compiler/passes/alloc_utils.h"
#include "graph_compiler/passes/generate_profiler_debug_info.h"
#include "graph_compiler/types.h"

// synapse api (relative to include/)
#include "synapse_common_types.h"

#include <hl_gcfg/hlgcfg.hpp>

// std includes
#include <bitset>
#include <memory>
#include <optional>

namespace eager_mode
{
EagerGraph::EagerGraph(synDeviceType deviceType)
: HabanaGraph(false),
  m_chipType(synDeviceType2ChipType(deviceType)),
  m_tensorNameBuilder("t_"),
  m_nodeNameBuilder("n_"),
  m_nodesContainer(*this),
  m_node2Desc(*this),
  m_programAllocator("HBM_PROGRAM"),
  m_workspaceAllocator("WORKSPACE", 0, false, false),
  m_isDebugInfoEnabled(GCFG_ENABLE_PROFILER.value())
{
    m_graphTraits = std::make_shared<GraphTraits>(deviceType, CompilationMode::Eager);
    GlobalConfManager::instance().setDeviceType(deviceType);
    getGraphAnnotation().memoryStrategyParams.dramInfo.enableDramAlloc = true;
}

EagerGraph::EagerGraph(const EagerGraph& other)
: HabanaGraph(other),
  m_chipType(other.m_chipType),
  m_graphState(other.m_graphState),
  m_tensorNameBuilder(other.m_tensorNameBuilder),
  m_nodeNameBuilder(other.m_nodeNameBuilder),
  m_nodesContainer(other.m_nodesContainer, *this),
  m_node2Desc(*this),
  m_programAllocator("HBM_PROGRAM"),
  m_workspaceAllocator("WORKSPACE", 0, false, false),
  m_isDebugInfoEnabled(other.m_isDebugInfoEnabled)
{
}

std::optional<uint32_t> EagerGraph::getNextTPCKernelUniqueId()
{
    return m_nextTPCKernelUniqueId++;
}

bool EagerGraph::isValidForEager(synDeviceType deviceType)
{
    return ChipInfo::isValidForEager(synDeviceType2ChipType(deviceType));
}

HabanaGraphPtr EagerGraph::clone(bool cloneAllocators, bool keepMappings) const
{
    // For Eager we do not need to clone the allocators since the initialization only takes places
    // at compilation phase.
    return HabanaGraphPtr(new EagerGraph(*this));
}

bool EagerGraph::moveNodesToGraph(HabanaGraph& outputGraph)
{
    if (unlikely(!transitionGraphState(GraphState::FALLBACK_STARTED))) return false;
    return m_nodesContainer.downloadOriginalNodesToHabanaGraph(outputGraph);
}

QueueDispatcherParams EagerGraph::getEagerDMADispatcherParams() const
{
    uint32_t numEngines = std::bitset<32>(getHALReader()->getInternalDmaEnginesMask()).count();
    EAGER_ASSERT(numEngines >= 1, "Not enough engines to execute dma operations");
    return QueueDispatcherParams(numEngines, 0);
}

bool EagerGraph::performMaxShapeInference()
{
    if (unlikely(!transitionGraphState(GraphState::SHAPE_INFERENCE_STARTED))) return false;
    return m_nodesContainer.performMaxShapeInference();
}

bool EagerGraph::compileGraph()
{
    if (unlikely(!transitionGraphState(GraphState::COMPILATION_STARTED))) return false;
    if (calcTypeForCompilation() != getCompilationMode())
    {
        EAGER_REPORT_ERROR("{}: Eager doesn't support compilation", HLLOG_FUNC);
        return false;
    }

    // Must come before nodes addition as it handles aux tensor allocation
    const auto& hal = *m_graphTraits->getHalReader();
    uint64_t    dramSizeInBytes = hal.getDRAMSizeInBytes();
    m_workspaceAllocator.Init(dramSizeInBytes, getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_WORKSPACE));
    m_programAllocator.Init(dramSizeInBytes, getVirtualAddressForMemoryID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA));

    // Node addition handles
    // - Node Displacement (Extraction/ Transformation/ Replacement)
    // - TPC Kernel aliased tensors update
    // - TPC Kernel Loading (In order to add memset and reduce nodes if required for the kernel outputs)
    // - TPC Aux tensor allocation (Since aux tensor list is given at loading time)
    if (!m_nodesContainer.downloadOriginalNodesToEagerGraph())
    {
        EAGER_REPORT_ERROR("{}: Failed to add the nodes to the graph", HLLOG_FUNC);
        return false;
    }
    if (m_nodesContainer.getNodes().getPhysicalNodesNr() == 0)
    {
        // Note: An empty graph is possible even when `m_nodesContainer.getNodes().empty() == false` in case of ZST
        return true;
    }

    // Done before desc generation because patchTensors will use the contextId we set here.
    if (unlikely(m_isDebugInfoEnabled))
    {
        generateProfilerDebugInfo(*this);
    }

    if (m_node2Desc.init(m_nodesContainer.getNodes(),
                         m_nodesContainer.getGlobalDependencies().getLatestPhysicalProducers()) == false)
    {
        return false;
    }

    if (!allocateTensors())
    {
        EAGER_REPORT_ERROR("{}: Failed to allocate tensors", HLLOG_FUNC);
        return false;
    }

#ifndef NDEBUG
    LOG_DEBUG(GRAPH_DATA, "Final graph data");
    for (const NodePtr& n : getExeSortedNodes())
    {
        if (n == nullptr) continue;
        n->print();
    }
#endif

    if (!m_node2Desc.generateDescriptors())
    {
        EAGER_REPORT_ERROR("{}: Failed to generate descriptors", HLLOG_FUNC);
        return false;
    }

    const SyncSchemeManagerBase& syncSchemeManager = ChipInfo::getSyncSchemeManager(getChipType());
    syncSchemeManager.generateNodesArcSyncScheme(m_node2Desc);
    syncSchemeManager.generateWorkDistributionContexts(m_node2Desc);

    return true;
}

bool EagerGraph::compile()
{
    // Must come before nodes addition
    CompilationHalReaderSetter compHalReaderSetter(this);

    // SW-171881 will get rid of this temporary configuration change
    class TmpGcfgChange
    {
    public:
        TmpGcfgChange()
        {
            GCFG_ENABLE_TRANSPOSE_VIA_GEMM.setValue(false);
        }
        ~TmpGcfgChange()
        {
            GCFG_ENABLE_TRANSPOSE_VIA_GEMM.setValue(m_origGcfg);
        }
        bool m_origGcfg = GCFG_ENABLE_TRANSPOSE_VIA_GEMM.value();
    } tmpGcfgChange;

    if (compileGraph() == false) return false;

    // Note that we pass in the original tensors since persistent tensors have to be preserved in the recipe for query
    if (m_nodesContainer.getNodes().getPhysicalNodesNr() > 0)
    {
        if (unlikely(GCFG_GRAPH_VISUALIZATION.value()))
        {
            visualizeGraph(*this, "eager_final_graph");
        }

        bool addNOPKernel = false;
        if (GCFG_ENABLE_EAGER_NOP_IN_RECIPE.value() && GCFG_ENABLE_EAGER_ARCH_OPTIMIZATIONS.value())
        {
            // for a single program data blob we can avoid copy altogether and just point
            // to the retrieved tpc kernel.
            addNOPKernel = (m_programDataBlobManager.isProgramDataBlobCopyRequired() ||
                            m_programDataBlobManager.getProgramDataBlobs().size() > 1);

            // In case we copy TPC kernels binaries anyway we would like to plant a NOP kernel
            if (addNOPKernel)
            {
                const KernelInfo& NOPKernel = eager_mode::RecipeTemplates::getInstance().getNOPKernelInfo(m_chipType);
                // if NOPKernel is not valid or failed to load into programDataBlobManager it won't be added to recipe
                addNOPKernel =
                    NOPKernel.kernelBinary != nullptr ? loadNOPKernelToProgramDataBlobManager(NOPKernel) : false;
            }
        }

        const unsigned           cacheLineSizeInBytes = getHALReader()->getCacheLineSizeInBytes();
        const WorkspaceSizesType workspaceSize =
            alignUpTo(m_workspaceAllocator.GetCurrentlyUsed(), cacheLineSizeInBytes);
        const WorkspaceSizesType programDataSize =
            alignUpTo(m_programAllocator.GetCurrentlyUsed(), cacheLineSizeInBytes);
        EAGER_ASSERT(m_node2Desc.isInitialized(), "");
        return m_recipeGenerator.generate(
            m_recipeName,
            m_node2Desc,
            m_programDataBlobManager,
            workspaceSize,
            programDataSize,
            m_nodesContainer.getOriginalTensors(),
            unlikely(m_isDebugInfoEnabled) ? std::make_optional(RecipeIdType {getRecipeDebugId()}) : std::nullopt,
            addNOPKernel);
    }
    else
    {
        // TODO: Should we still create an empty graph if GCFG_GRAPH_VISUALIZATION is set?

        return m_recipeGenerator.generateEmptyRecipe(m_recipeName,
                                                     getDeviceType(),
                                                     m_nodesContainer.getOriginalTensors());
    }
}

bool EagerGraph::loadNOPKernelToProgramDataBlobManager(const KernelInfo& NOPKernelInfo)
{
    Settable<deviceAddrOffset> newAddress =
        m_programAllocator.Allocate(NOPKernelInfo.kernelSize, getHALReader()->getCacheLineSizeInBytes());
    if (!newAddress.is_set())
    {
        getGraphAnnotation().errors.memoryAllocationError = true;
        EAGER_REPORT_ERROR("Failed to allocate NOP kernel for Eager");
        return false;
    }
    m_programDataBlobManager.registerNewTPCProgramDataBlobForDownload(NOPKernelInfo.kernelBinary,
                                                                      *newAddress,
                                                                      NOPKernelInfo.kernelSize,
                                                                      NOPKernelInfo.kernelId);
    return true;
}

bool EagerGraph::addNode(pNode node)
{
    if (!m_nodesContainer.addNewNode(node))
    {
        EAGER_REPORT_ERROR("{}: Failed to add node {} to eager graph", HLLOG_FUNC, node->getGUID());
        return false;
    }
    return true;
}

void EagerGraph::removeNode(pNode node, pNode newProducer)
{
    HabanaGraph::removeNode(node, newProducer);
}

CompilationMode EagerGraph::calcTypeForCompilation() const
{
    return m_nodesContainer.isEagerCompilationSupported() ? CompilationMode::Eager : CompilationMode::Graph;
}

recipe_t* EagerGraph::serializeDataPlane(RecipeAllocator* recipeAlloc) const
{
    return const_cast<recipe_t*>(m_recipeGenerator.getRecipe());
}

// Normally done as part of allocateTensors
bool EagerGraph::allocateTensors()
{
    unsigned cacheLineSize    = m_graphTraits->getHalReader()->getCacheLineSizeInBytes();

    for (const TensorPtr& tensor : m_nodesContainer.getNodes().getTensors().getTensors())
    {
        if (tensor->isAliasedTensor()) continue;

        if (tensor->isPersistent())
        {
            if (!assignVirtualAddressToUserPersistentTensor(tensor))
            {
                return false;
            }
        }
        else if (tensor->getMemorySectionID() == MEMORY_ID_RESERVED_FOR_WORKSPACE)
        {
            tensor->setTensorAlignment(cacheLineSize);
            if (!allocateTensor<false>(tensor, m_workspaceAllocator, false /*allowFailure*/))
            {
                return false;
            }
        }
        else if (tensor->getMemorySectionID() == MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
        {
            tensor->setTensorAlignment(cacheLineSize);
            if (!allocateTensor<false>(tensor, m_programAllocator, false /*allowFailure*/))
            {
                return false;
            }
            m_programDataBlobManager.registerProgramDataBlobForDownload(tensor->getData(),
                                                                        tensor->getDramOffset(),
                                                                        tensor->getTotalSizeInBytes());
        }
    }
    return true;
}

bool EagerGraph::generateExecutionSchedule() const
{
    if (m_cacheExeSortedNodes.empty())
    {
        const auto& nodes = m_nodesContainer.getNodes();
        m_cacheExeSortedNodes.append(nodes.begin(), nodes.end());
    }
    return true;
}

RecipeAllocator* EagerGraph::consumeEagerCompositeTemplateRecipeAllocator()
{
    return m_recipeGenerator.consumeRecipeAllocator();
}

HabanaGraphPtr EagerGraph::duplicate(TensorPtrMappingVec& tensorsMap, NodeIdMappingVec& nodesMap)
{
    EAGER_ASSERT(m_graphState == GraphState::NEW_GRAPH || m_graphState == GraphState::DUPLICATED,
                 "API is not supported mid or post compilation");
    // sort the origin graph nodes once so that all the duplicated targets get it for free.
    // if we fail the sort, we also fail the duplication.
    if (unlikely(m_graphState == GraphState::NEW_GRAPH &&
                 (!transitionGraphState(GraphState::DUPLICATED) || !m_nodesContainer.prepareForGraphDuplication())))
    {
        return nullptr;
    }
    HabanaGraphPtr duplicateGraph      = clone();
    auto           duplicateEagerGraph = static_cast<EagerGraph*>(duplicateGraph.get());
    duplicateEagerGraph->m_nodesContainer.getDuplicateMappings(m_nodesContainer, tensorsMap, nodesMap);
    duplicateEagerGraph->m_duplicatedTarget = true;
    return duplicateGraph;
}

unsigned EagerGraph::getNumNodesPreCompilation()
{
    EAGER_ASSERT(m_graphState == GraphState::NEW_GRAPH || m_graphState == GraphState::DUPLICATED,
                 "API is not supported mid or post compilation");
    return m_nodesContainer.getOriginalNodesNr();
}

const TensorSet& EagerGraph::getTensors() const
{
    if (m_cacheAllTensors.empty())
    {
        for (const TensorPtr& tensor : m_nodesContainer.getNodes().getTensors().getTensors())
        {
            if (tensor == nullptr) continue;
            m_cacheAllTensors.insert(tensor);
        }
    }
    return m_cacheAllTensors;
}

const NodeSet& EagerGraph::getNodes() const
{
    if (m_cacheAllNodes.empty())
    {
        const auto& nodes = m_nodesContainer.getNodes();
        m_cacheAllNodes.insert(nodes.begin(), nodes.end());
    }
    EAGER_ASSERT(m_nodesContainer.getNodes().size() == m_cacheAllNodes.size(), "");
    return m_cacheAllNodes;
}

NodePtr EagerGraph::getNodeByID(synNodeId nodeID) const
{
    const EagerNode* node = m_nodesContainer.findNodeByID(nodeID);
    if (node != nullptr) return *node;
    return nullptr;
}

const EagerMmeBrainBase& EagerGraph::getEagerMmeBrain() const
{
    return ChipInfo::getEagerMmeBrain(m_chipType);
}

void EagerGraph::addControlDependency(const NodeSet& blockingSet, const NodeSet& blockedSet, Tensor::ControlEdgeType controlType)
{
    EAGER_REPORT_ERROR("Unsupported API: 'addControlDependency'");
}

void EagerGraph::addControlDependency(const NodePtr& blockingNode,const NodePtr& blockedNode, Tensor::ControlEdgeType controlType)
{
    EAGER_REPORT_ERROR("Unsupported API: 'addControlDependency'");
}

void EagerGraph::removeNodeControlDependencies(const NodePtr& node, Tensor::ControlEdgeType controlType)
{
    EAGER_REPORT_ERROR("Unsupported API: 'removeNodeControlDependencies'");
}

void EagerGraph::removeNodeControlDependency(const NodePtr& Node, const TensorPtr& ctrlEdge, Node::eParamUsage usage)
{
    EAGER_REPORT_ERROR("Unsupported API: 'removeNodeControlDependency'");
}

bool EagerGraph::transitionGraphState(GraphState newState)
{
    bool transitionAllowed = true;
    switch (newState)
    {
        case GraphState::NEW_GRAPH:
            transitionAllowed = false;
            EAGER_ASSERT(transitionAllowed, "transition to NEW_GRAPH isn't allowed.");
            break;
        case GraphState::DUPLICATED:
            transitionAllowed = m_graphState == GraphState::NEW_GRAPH;
            EAGER_ASSERT(transitionAllowed, "DUPLICATED can only be set from NEW_GRAPH");
            break;
        case GraphState::SHAPE_INFERENCE_STARTED:
            transitionAllowed = m_graphState == GraphState::NEW_GRAPH || m_graphState == GraphState::DUPLICATED;
            EAGER_ASSERT(transitionAllowed, "SHAPE_INFERENCE_STARTED can only be set from NEW_GRAPH or DUPLICATED");
            break;
        case GraphState::COMPILATION_STARTED:
            transitionAllowed = m_graphState == GraphState::NEW_GRAPH || m_graphState == GraphState::DUPLICATED ||
                                m_graphState == GraphState::SHAPE_INFERENCE_STARTED;
            EAGER_ASSERT(transitionAllowed,
                         "COMPILATION_STARTED can only be set from either NEW_GRAPH or DUPLICATED or "
                         "SHAPE_INFERENCE_STARTED");
            break;
        case GraphState::FALLBACK_STARTED:
            return true;
        default:
            transitionAllowed = false;
            EAGER_ASSERT_0;
    }

    if (!transitionAllowed) return false;
    m_graphState = newState;
    return true;
}

std::pair<unsigned, unsigned> EagerGraph::getBreakpointsAndNodeROINr(const NodePtr& n) const
{
    // TODO : add DISABLE_NON_SIGNALING_ROI_BREAKPOINT support if needed.
    // Given we currently have a single logical roi for tpc\mme, this might be applicable
    // to dma engine only for the moment.
    if (n->isLogicalOperation()) return {};
    const auto& execSeq = m_node2Desc.getExecSequence();
    auto        iter    = std::find_if(execSeq.begin(), execSeq.end(), [&](const SingleNode2Desc& nodeDesc) {
        return n.get() == &nodeDesc.getDescGen().getNode();
    });
    if (iter == execSeq.end())
    {
        EAGER_ASSERT(false, "physical node must have a non empty logical ROI");
        return {};
    }
    unsigned roiNr = iter->getDescGen().getLogicalRoiNr();
    return {roiNr, roiNr};
}

}  // namespace eager_mode
