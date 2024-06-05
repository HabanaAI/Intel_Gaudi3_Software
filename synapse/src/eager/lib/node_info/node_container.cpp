#include "node_container.h"

// eager includes (relative to src/eager/lib/)
#include "chip_info.h"
#include "debug_tools/eager_graph_visualization.h"
#include "eager_graph.h"
#include "node_info/eager_node.h"
#include "node_info/node_displacement.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/types.h"
#include "infra/log_manager.h"

// std includes
#include <algorithm>

namespace eager_mode
{
NodesContainer::NodesContainer(EagerGraph& eagerGraph)
: m_eagerGraph(eagerGraph),
  m_tensorSizeThresholdForParallelExec(ChipInfo::getTensorSizeThresholdForParallelExec(eagerGraph.getChipType())),
  m_orgNodes(/*isOriginalGraph*/ true, m_tensorSizeThresholdForParallelExec),
  m_nodes(/*isOriginalGraph*/ false, m_tensorSizeThresholdForParallelExec),
  m_nodeDisplacement(eagerGraph, m_nodes),
  m_execSequencer(m_orgNodes.getTensors().allowParallelExecHandling())
{
}

NodesContainer::NodesContainer(const NodesContainer& other, EagerGraph& eagerGraph)
: m_eagerGraph(eagerGraph),
  m_tensorSizeThresholdForParallelExec(other.m_tensorSizeThresholdForParallelExec),
  m_orgNodes(other.m_orgNodes),
  m_nodes(/*isOriginalGraph*/ false, m_tensorSizeThresholdForParallelExec),
  m_nodeDisplacement(eagerGraph, m_nodes),
  m_areOrgNodesSupported(other.m_areOrgNodesSupported),
  m_execSequencer(m_orgNodes.getTensors().allowParallelExecHandling())
{
    m_duplicationMap.constructTensorMapping(m_orgNodes.getTensors(), other.m_orgNodes.getTensors());
}

bool NodesContainer::addNewNode(const EagerNode& node)
{
    // Accumulate original nodes and update Eager-support-check flag
    EAGER_ASSERT(m_orgNodes.isAddingNewNodesEnabled(), "Adding new nodes is prevented at this stage");
    if (m_areOrgNodesSupported)
    {
        if (!m_nodeDisplacement.isUserNodeSupported(node))
        {
            EAGER_LOG_WARN("{}: Node {} is not supported in eager mode, will fall back to graph mode",
                           HLLOG_FUNC,
                           node->getGUID());
            m_areOrgNodesSupported = false;
        }
        else if (m_orgNodes.size() >= GCFG_MAX_NODES_IN_EAGER_GRAPH.value())
        {
            EAGER_LOG_WARN("{}: Nodes number exceeded Eager's max limit of {} nodes, will fallback to graph mode",
                           HLLOG_FUNC,
                           GCFG_MAX_NODES_IN_EAGER_GRAPH.value());
            m_areOrgNodesSupported = false;
        }
    }
    addNewOriginalNode(node);
    return true;  // This flow returns true all time
}

void NodesContainer::addNewOriginalNode(const EagerNode& node)
{
    EAGER_ASSERT(m_orgNodes.isAddingNewNodesEnabled(), "Wrong flow");
    m_orgNodes.push_back(node);
}

// Use when fall back to Graph mode
bool NodesContainer::downloadOriginalNodesToHabanaGraph(HabanaGraph& graph)
{
    EAGER_ASSERT(!isEagerCompilationSupported() && !m_orgNodes.empty(), "Wrong flow");
    m_orgNodes.disableAddingNewNodes();
#ifndef NDEBUG
    printNodes(m_orgNodes, "Original nodes");
#endif  // NDEBUG

    for (const auto& node : m_orgNodes)
    {
        if (!GraphEditor::addNode(graph, node))
        {
            EAGER_REPORT_ERROR("{}: Failed to download node {} to Habana graph", HLLOG_FUNC, node->getGUID());
            return false;
        }
    }

    return true;
}

// Use when all original nodes are supported by Eager
bool NodesContainer::downloadOriginalNodesToEagerGraph()
{
    EAGER_ASSERT(m_areOrgNodesSupported, "Wrong flow");
    if (unlikely(m_orgNodes.isAddingNewNodesEnabled() && !lockAndSortUserNodes()))
    {
        return false;
    }
#ifndef NDEBUG
    printNodes(m_orgNodes, "Original nodes");
#endif  // NDEBUG

    const bool visualizeGraphs = GCFG_GRAPH_VISUALIZATION.value();
    // first we just collect the extracted nodes but do not yet add them to the node container
    for (EagerNode& node : m_orgNodes)
    {
        if (!m_nodeDisplacement.processUserNode(node))
        {
            LOG_ERR(EAGER, "{}: Failed to extract node {} for Eager graph", HLLOG_FUNC, node->getGUID());
            return false;
        }
    }
    m_nodeDisplacement.markUserNodeExtractionCompletion();
    m_nodeDisplacement.fuseTransposes();
    // add the extracted nodes to the node container
    for (int userNodeIndex = 0; userNodeIndex < m_orgNodes.size(); ++userNodeIndex)
    {
        const EagerNode& node = m_orgNodes[userNodeIndex];
        m_nodeDisplacement.downloadExtractedNodes(userNodeIndex);
        // Reorder new nodes that have been added
        if (unlikely(!m_execSequencer.reorderLast(m_nodes))) return false;

        if (unlikely(visualizeGraphs && !m_nodes.empty()))
        {
            visualizeGraph(m_nodeDisplacement.m_eagerGraph, fmt::format("{}_extracted", node->getNodeName()));
        }
    }

    // m_nodes contains all nodes including logical ones but not resulting memcpy from the logical pass, yet
    if (m_nodeDisplacement.hasLogicalNodes())
    {
        if (unlikely(!m_nodeDisplacement.processLogicalNodes(m_execSequencer)))
        {
            LOG_ERR(EAGER, "processLogicalNodes failed!");
            return false;
        }
        m_execSequencer.redoSerialDependencies(m_nodes.size());
    }

    // After the following locking there will be no way to process new node,
    // Doing so brings undefined behavior and wrong functionality.
    m_nodes.disableAddingNewNodes();
    m_execSequencer.disableProcessingNewNodes(m_nodes);

#ifndef NDEBUG
    printNodesWithTensorDetails(m_nodes, "all nodes");
#endif  // NDEBUG

    return true;
}

const EagerNode* NodesContainer::findNodeByID(synNodeId nodeID) const
{
    if (!m_nodes.empty()) return m_nodes.findNodeByID(nodeID);
    return m_orgNodes.findNodeByID(nodeID);
}

namespace
{
class ShapeInferenceDataLayoutAdjuster
{
public:
    ShapeInferenceDataLayoutAdjuster(const EagerGraph& g, const NodePtr& node);
    ~ShapeInferenceDataLayoutAdjuster();
    bool layoutsValidationPassed() const { return m_validLayouts; }

private:
    using TensorBoolVec = SmallVector<bool, MAX_TENSOR_NR>;
    void applyPermutations(const TensorVector&      tensors,
                           const PermutationVector& permutationsToApply,
                           const TensorBoolVec&     permutationsToRestore,
                           bool                     applyInverse,
                           bool                     restoreFlow);
    bool anyPermutedTensorsExist() const;
    void updateNodePermutationsWithTensorPermutations();

    const NodePtr&    m_node;
    bool              m_layoutAdjustmentRequired = false;
    bool              m_validLayouts             = true;
    PermutationVector m_inputPermutations;
    TensorBoolVec     m_inputPermutationsToRestore;
    PermutationVector m_outputPermutations;
    TensorBoolVec     m_outputPermutationsToRestore;
    using TensorPtrSet = chromium_small_set::small_set<std::set<Tensor*>, MAX_TENSOR_NR>;
    TensorPtrSet m_inferedTensors;
};

ShapeInferenceDataLayoutAdjuster::ShapeInferenceDataLayoutAdjuster(const EagerGraph& g, const NodePtr& node)
: m_node(node),
  m_inputPermutationsToRestore(m_node->getNumInputs()),
  m_outputPermutationsToRestore(m_node->getNumOutputs())
{
    auto& ioManager            = m_node->getNodeIOManager();
    bool  permutedTensorsExist = anyPermutedTensorsExist();
    bool  nodeIsAllDontCare    = ioManager.nodeisAllDontCare();

    if (!permutedTensorsExist && nodeIsAllDontCare) return;

    if (nodeIsAllDontCare)
    {
        m_inputPermutations.resize(m_node->getNumInputs());
        m_outputPermutations.resize(m_node->getNumOutputs());
    }
    else
    {
        ioManager.setSupportedIOLayouts(g.getDeviceType());
        if (!ioManager.permutationsRequired())
        {
            ioManager.setDefaultIOLayouts();
            ioManager.permute(m_inputPermutations, m_outputPermutations);
            ioManager.setSupportedLayouts(LayoutVector(m_node->getNumInputs()), LayoutVector(m_node->getNumOutputs()));
        }
        else
        {
            ioManager.permute(m_inputPermutations, m_outputPermutations);
        }
    }
    if (permutedTensorsExist)
    {
        updateNodePermutationsWithTensorPermutations();
    }
    applyPermutations(m_node->getInputs(), m_inputPermutations, m_inputPermutationsToRestore, false, false);
    applyPermutations(m_node->getOutputs(), m_outputPermutations, m_outputPermutationsToRestore, true, false);
    m_layoutAdjustmentRequired = true;
}

ShapeInferenceDataLayoutAdjuster::~ShapeInferenceDataLayoutAdjuster()
{
    if (m_layoutAdjustmentRequired)
    {
        applyPermutations(m_node->getInputs(), m_inputPermutations, m_inputPermutationsToRestore, true, true);
        applyPermutations(m_node->getOutputs(), m_outputPermutations, m_outputPermutationsToRestore, false, true);
    }
}

bool ShapeInferenceDataLayoutAdjuster::anyPermutedTensorsExist() const
{
    auto anyPermutedTensorsFound = [](const TensorVector& tensors) {
        return std::any_of(tensors.begin(), tensors.end(), [](const TensorPtr& t) {
            return t && t->getPermutation().has_value() && !t->getPermutation()->isIdentity();
        });
    };
    return anyPermutedTensorsFound(m_node->getInputs()) || anyPermutedTensorsFound(m_node->getOutputs());
}

void ShapeInferenceDataLayoutAdjuster::updateNodePermutationsWithTensorPermutations()
{
    auto updatePermutations = [this](PermutationVector&  nodePermutations,
                                     const TensorVector& tensors,
                                     TensorBoolVec&      permutationsToRestore,
                                     bool                applyInverse) {
        if (unlikely(!m_validLayouts)) return;
        for (int i = 0; i < nodePermutations.size(); i++)
        {
            if (!tensors[i]) continue;
            bool noTensorPermutation =
                !tensors[i]->getPermutation().has_value() || tensors[i]->getPermutation()->isIdentity();
            if (!noTensorPermutation)
            {
                permutationsToRestore[i] = true;
            }
            if (nodePermutations[i].isEmpty())
            {
                if (tensors[i]->getPermutation().has_value())
                {
                    nodePermutations[i] = applyInverse ? tensors[i]->getPermutation()->getInversePermutation()
                                                       : *tensors[i]->getPermutation();
                }
                else
                {
                    nodePermutations[i].setIdentityPermutation(tensors[i]->getDim());
                }
                continue;
            }
            if (noTensorPermutation) continue;
            gc::Permutation newNodePermutation =
                applyInverse ? nodePermutations[i].getInversePermutation() : nodePermutations[i];
            if (newNodePermutation != *tensors[i]->getPermutation())
            {
                EAGER_REPORT_ERROR("node has both supplied layout and a different supplied permutation");
                m_validLayouts = false;
                return;
            }
        }
    };
    updatePermutations(m_inputPermutations, m_node->getInputs(), m_inputPermutationsToRestore, false);
    updatePermutations(m_outputPermutations, m_node->getOutputs(), m_outputPermutationsToRestore, true);
}

void ShapeInferenceDataLayoutAdjuster::applyPermutations(const TensorVector&      tensors,
                                                         const PermutationVector& permutationsToApply,
                                                         const TensorBoolVec&     permutationsToRestore,
                                                         bool                     applyInverse,
                                                         bool                     restoreFlow)
{
    for (int i = 0; i < permutationsToApply.size(); i++)
    {
        Tensor* t = tensors[i].get();
        if (!t) continue;
        if (t->isPropSet(synTensorPropGeometryMax))
        {
            gc::Permutation actualPermutationToApply =
                applyInverse ? permutationsToApply[i].getInversePermutation() : permutationsToApply[i];

            TSize sizes[Tensor::c_tensorMaxNDim];
            t->getAllNSizesInElements(sizes);
            actualPermutationToApply.permuteShape(sizes, t->getDim());

            if (restoreFlow && m_inferedTensors.count(t) > 0)
            {
                t->reshape(t->getDim(), sizes);
                if (permutationsToRestore[i])
                {
                    t->setPermutation(actualPermutationToApply);
                }
            }
            else
            {
                TStride strides[Tensor::c_numOfNStrides];
                t->getNStridesInBytes(strides);
                actualPermutationToApply.permuteShape(strides, t->getDim());
                t->reshape(t->getDim(), sizes, strides);
                if (restoreFlow && permutationsToRestore[i])
                {
                    // we always take actualPermutationToApply.getInversePermutation() but in case applyInverse
                    // was set then it is already available through permutationsToApply[i].
                    t->setPermutation(applyInverse ? permutationsToApply[i]
                                                   : actualPermutationToApply.getInversePermutation());
                }
                else
                {
                    t->unsetPermutation();
                }
            }
        }
        else
        {
            m_inferedTensors.insert(t);
            t->unsetPermutation();
        }
    }
}

}  // namespace

bool NodesContainer::performMaxShapeInference()
{
    // no point in shape inference if we know we'll fallback to graph mode
    if (unlikely(!isEagerCompilationSupported()))
    {
        return false;
    }

    if (unlikely(m_orgNodes.isAddingNewNodesEnabled() && !lockAndSortUserNodes()))
    {
        return false;
    }

    // we only set synTensorPropGeometryMax for infered shapes
    // but setting synTensorPropGeometryMax by the user sets all 3 in Tensor::setGeometry
    auto areAllOperandsShapesSetByUser = [](const TensorVector& tensors) {
        return std::all_of(tensors.begin(), tensors.end(), [](const TensorPtr& t) {
            return t == nullptr || (t->isPropSet(synTensorPropGeometryDim) && t->isPropSet(synTensorPropGeometryMin) &&
                                    t->isPropSet(synTensorPropGeometryMax));
        });
    };

    for (auto& node : m_orgNodes)
    {
        if (areAllOperandsShapesSetByUser(node->getInputs()) && areAllOperandsShapesSetByUser(node->getOutputs()))
        {
            continue;
        }

        // need to adjust data layout prior to max shape inference
        ShapeInferenceDataLayoutAdjuster dataLayoutAdjuster(m_eagerGraph, node);
        if (unlikely(!dataLayoutAdjuster.layoutsValidationPassed())) return false;
        if (unlikely(!node->inferOutputsSizes(m_eagerGraph.getDeviceType(),
                                              /*inferMax*/ true,
                                              /*forbidInvalid*/ true,
                                              /*skipStatic=*/false)))
        {
            LOG_ERR(EAGER, "Failed to update output shape for node: \"{}\"", node->getNodeName());
            return false;
        }
    }
    return true;
}

void NodesContainer::getDuplicateMappings(const NodesContainer&             orig,
                                          HabanaGraph::TensorPtrMappingVec& tensorsMap,
                                          HabanaGraph::NodeIdMappingVec&    nodesMap)
{
    EAGER_ASSERT(orig.m_orgNodes.size() == m_orgNodes.size(), "Duplicate graph has an unexpected number of nodes");
    auto origGraphNodesIter      = orig.m_orgNodes.begin();
    auto origGraphNodesIterEnd   = orig.m_orgNodes.end();
    auto duplicateGraphNodesIter = m_orgNodes.begin();
    for (; origGraphNodesIter != origGraphNodesIterEnd; ++origGraphNodesIter, ++duplicateGraphNodesIter)
    {
        nodesMap.emplace_back((*origGraphNodesIter)->getId(), (*duplicateGraphNodesIter)->getId());
    }
    // also handles tensors not attached to nodes
    for (auto& [origTensor, newTensor] : tensorsMap)
    {
        newTensor = m_duplicationMap.getNewTensor(origTensor);
        if (newTensor == nullptr)
        {
            newTensor = origTensor->clone(false, true, true, TensorNameClonePolicy::COPY_NAME);
        }
    }
    m_duplicationMap.clear();
}

bool NodesContainer::lockAndSortUserNodes()
{
    EAGER_ASSERT(m_orgNodes.isAddingNewNodesEnabled(), "lockAndSortUserNodes should only be called once");
    m_orgNodes.disableAddingNewNodes();
    // sort the original nodes once pre node extraction.
    // cases are:
    // 1. we sorted once for the source graph for the duplicate API.
    // 2. we sort it pre graph compilation (shape inference failed so graph was created from scratch).
    if (!m_execSequencer.reorderAll(m_orgNodes))
    {
        LOG_ERR(EAGER, "failed to sort user nodes for Eager graph");
        return false;
    }
    return true;
}

}  // namespace eager_mode
