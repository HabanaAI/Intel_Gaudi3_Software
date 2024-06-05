#include "bundle_slicer.h"

#include "graph_editor.h"
#include "graph_visualization.h"
#include "habana_graph.h"
#include "logical_op_node.h"
#include "node.h"
#include "operation_slice.h"
#include "post_slicing_op_handler.h"
#include "slicing_utils.h"
#include "sram_management/bundle.h"
#include "types.h"

#include <algorithm>

BundleSlicer::BundleSlicer(const HabanaGraph& graph, uint32_t bundleIdx, BundleType bundleType)
: m_graph(graph), m_bundleIdx(bundleIdx), m_opIdx(0), m_bundleType(bundleType)
{
}

void BundleSlicer::sliceBundle(const Bundle& bundle, HabanaGraph& graph)
{
    SET_TEMP_LOG_CONTEXT(fmt::format("Bundle#{}", bundle.index()));

    BundleSlicer slicer(graph, bundle.index(), bundle.type());

    bool runsOnMme = false;
    bool runsOnTpc = false;
    bool runsOnDma = false;
    for (const auto& node : bundle.getNodes())
    {
        if (HabanaGraph::runsOnTPC(node))
        {
            runsOnTpc = true;
        }
        else if (HabanaGraph::runsOnMME(node))
        {
            runsOnMme = true;
        }
        else if (node->isDma())
        {
            runsOnDma = true;
        }
        slicer.m_bundleNodes.insert(node);
    }
    BundleEngine bundleEngine = runsOnMme && runsOnTpc ? BundleEngine::ENGINE_MME_TPC
                                : runsOnMme            ? BundleEngine::ENGINE_MME
                                : runsOnTpc            ? BundleEngine::ENGINE_TPC
                                                       : BundleEngine::ENGINE_UNDEFINED;
    if (runsOnDma && !runsOnTpc && !runsOnMme)
    {
        bundleEngine = BundleEngine::ENGINE_DMA;
    }
    HB_ASSERT(bundleEngine != BundleEngine::ENGINE_UNDEFINED, "Bundle {} has no MME nor TPC nodes", bundle.index());

    // Analyze the operations
    for (const auto& operation : bundle.getSolution().operations)
    {
        slicer.addOperation(operation);
    }

    try
    {
        auto tmpGraph = graph.createEmptyGraph();
        // copy memory coherence data - so that control edges will be copied along with the nodes
        tmpGraph->getGraphAnnotation().memoryCoherence = graph.getGraphAnnotation().memoryCoherence;
        // copy flash attention data
        tmpGraph->getGraphAnnotation().flashAttentionDb = graph.getGraphAnnotation().flashAttentionDb;
        // Add the new nodes to temporary graph
        // If it fails, an exception is thrown
        slicer.addGraphNodes(*tmpGraph);

        // Remove bundle nodes from graph
        if (GraphEditor::isInGroupDependencies(graph, bundle.getNodes()))
        {
            LOG_WARN(BE_SLICER, "Fusion of bundle {} nodes is not allowed", bundle.getName());
            return;
        }

        // Merge temporary graph
        NodeVector tmpGraphNodes = tmpGraph->getExeSortedNodes();

        for (const pNode& node : tmpGraphNodes)
        {
            HB_ASSERT(node->getNodeAnnotation().bundleInfo.is_set(), "Adding bundle node without bundle index set");
            node->getNodeAnnotation().bundleInfo->bundleEngine = bundleEngine;
        }
        GraphEditor::removeNodes(*tmpGraph, tmpGraphNodes);  // Prevent node from being in two graphs
        GraphEditor::replaceNodes(graph, bundle.getNodes(), tmpGraphNodes, true);
    }
    catch (std::exception& e)
    {
        // Original graph is not changed
        LOG_ERR(BE_SLICER, "Failed to slice bundle {}, error {}", bundle.getName(), e.what());
        if (!GCFG_ENABLE_SLICER_SILENT_FAILURE.value())
        {
            throw e;
        }
    };
}

void BundleSlicer::addOperation(const Operation& op)
{
    logOperation(op);
    // Clone the original operation node for this sliced operation
    pNode sliceNode = getSliceNode(op.originalNode);

    // Creat a slice of each tensor, and attach it to the tensor section and the slice node
    replaceOperandsWithSliceTensors(sliceNode, op.inputs, true);
    replaceOperandsWithSliceTensors(sliceNode, op.outputs, false);

    m_sliceNodes.push_back(sliceNode);

    ++m_opIdx;
}

template<typename Container>
void BundleSlicer::replaceOperandsWithSliceTensors(NodePtr&         sliceNode,
                                                   const Container& sliceReferences,
                                                   bool             isInputsContainer)
{
    // The sliceNode may not inherit/implement OperationSlice interface, so use dynamic cast.
    auto operationSlicePtr = std::dynamic_pointer_cast<OperationSlice>(sliceNode);

    unsigned index = 0;
    for (const pSliceReference& sliceRef : sliceReferences)
    {
        auto sliceTensor = getSliceTensor(sliceRef, isInputsContainer, index);
        sliceNode->replaceFirstTensor(sliceRef->operand->originalTensor, sliceTensor);

        if (sliceRef->operand->postSlicingHandler != nullptr)
        {
            sliceRef->operand->postSlicingHandler->updateSlicedNode(sliceNode, sliceRef);
        }

        if (operationSlicePtr)
        {
            operationSlicePtr->addTensorSliceOffset(sliceTensor,
                                                    sliceRef->operand->originalTensor,
                                                    SlicedOperandUtils::getSliceOffsets(sliceRef));
        }
        index++;
    }
}

pTensor BundleSlicer::getSliceTensor(const pSliceReference& sliceRef, bool inputSlice, uint32_t tensorIdx)
{
    // First check if the sliced operand has already been inserted to the sections map.
    auto sectionIter = m_sections.find(sliceRef->operand);

    if(sectionIter == m_sections.end())
    {
        // Insert a section for the sliced operand, get multi-buffer ID if needed.
        Settable<uint64_t> multiBufferId;
        // Use multi-buffer in the following cases:
        // 1) Operands with double-buffer
        // 2) Operands with single-buffer where first slice is smaller - to avoid OOM in epoch-allocator due
        //    to inner bundle fragmentation.
        // 3) Node Chain sharing buffers across the chain
        if ((sliceRef->operand->numOfBuffers > 1 || sliceRef->operand->isFirstSliceSmaller() ||
             sliceRef->operand->sharedChainMultiBuf) &&
            sliceRef->operand->resideInSRAM)
        {
            if (sliceRef->operand->sharedChainMultiBuf && m_sharedChainMultiBufId.has_value())
            {
                // Other tensors of this chain already allocated a multi buf ID
                multiBufferId.set(*m_sharedChainMultiBufId);
            }
            else
            {
                multiBufferId.set(
                    m_graph.getCodeGenerator()->getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS));
                if (sliceRef->operand->sharedChainMultiBuf)
                {
                    // The first tensor in this chain - allocate a new multi buf ID and save the value for the rest of
                    // the chain tensors
                    m_sharedChainMultiBufId = multiBufferId.value();
                    LOG_DEBUG(BE_SLICER,
                              "MultiBuf level {} set for sectionId {}",
                              c_sharedMultiBufLevel,
                              multiBufferId.value());
                }
            }
        }

        sectionIter = m_sections
                          .insert(std::make_pair(sliceRef->operand,
                                                 TensorSection(*m_graph.getHALReader(),
                                                               sliceRef->operand,
                                                               m_bundleIdx,
                                                               m_bundleType,
                                                               multiBufferId,
                                                               getNextTensorSectionIdx())))
                          .first;
    }

    // Create new slice in case its an output - producing new slice
    pTensor ret;
    if (inputSlice)
    {
        // Input to the slice operation is consumed from the section
        ret = sectionIter->second.addConsumeSlice(sliceRef->coordinates, m_opIdx);
    }
    else
    {
        // Output of the slice operation is producer for the section
        ret = sectionIter->second.addProduceSlice(sliceRef->coordinates, m_opIdx);
    }

    m_memcpyScheduler.addOperationBuffer(sliceRef, m_opIdx, inputSlice, sectionIter->second.getSectionIdx(), tensorIdx);
    ret->getTensorAnnotation().origBigTensor = sliceRef->operand->originalTensor;

    return ret;
}

static void collectTensorNodes(const HabanaGraph& g, const TensorPtr& t, /*IN,OUT*/ NodeVector& plannedNodes)
{
    const auto& prod = g.getTensorProducer(t);
    const auto& cons = g.getTensorConsumers(t);
    if (prod) plannedNodes.push_back(prod);
    plannedNodes.insert(plannedNodes.end(), cons.begin(), cons.end());
}

// Get all tensors connected to rootTensor via logical ops recursively
static std::set<TensorPtr> getLogicalTensorSet(const HabanaGraph& g, const TensorPtr& rootTensor)
{
    std::set<TensorPtr> res {rootTensor};

    NodeVector plannedNodes;
    collectTensorNodes(g, rootTensor, plannedNodes);

    std::set<NodePtr> handledNodes;
    while (!plannedNodes.empty())
    {
        const auto n = plannedNodes.back();
        plannedNodes.pop_back();

        if (!handledNodes.insert(n).second) continue;
        if (!n->isLogicalOperation()) continue;

        for (const auto& t : n->getOperands())
        {
            if (!res.insert(t).second) continue;
            collectTensorNodes(g, t, plannedNodes);
        }
    }

    return res;
}

std::unordered_set<TensorPtr> BundleSlicer::getEvictions() const
{
    std::unordered_set<TensorPtr> res;

    // Keep track of uninteresting tensors for the second pass to avoid looking
    // up the whole logical set for each of the views (avoiding O(n^2))
    std::unordered_set<TensorPtr> skipTensors;

    // First Pass: Forced (non-internal to bundle) evictions
    for (const auto& section : m_sections)
    {
        const auto& tensor = section.first->originalTensor;
        if (shouldTensorBeEvicted(tensor, m_graph, m_bundleNodes))
        {
            res.insert(tensor);
            skipTensors.insert(tensor);
        }
    }

    // Second Pass: Optional optimization evictions (DRAM->DRAM) forcing slices to be views into a single tensor
    // Needs to be done after the first pass is completed to avoid repeated evictions of different views
    if (GCFG_ENABLE_HBM_SLICES_ALLOCATION_OPTIMIZATION.value())
    {
        for (const auto& section : m_sections)
        {
            if (section.first->resideInSRAM) continue;

            const auto& tensor = section.first->originalTensor;
            if (skipTensors.count(tensor) == 0)
            {
                const std::set<TensorPtr> lts = getLogicalTensorSet(m_graph, tensor);
                skipTensors.insert(lts.begin(), lts.end());
                if (std::none_of(begin(lts), end(lts), [&](const TensorPtr& t) { return res.count(t) != 0; }))
                {
                    LOG_DEBUG(BE_SLICER, "HBM eviction enabled for tensor {} to force concat", tensor->getName());
                    res.insert(tensor);
                }
            }
            if (res.count(tensor) == 0)
            {
                LOG_DEBUG(BE_SLICER, "HBM eviction of tensor {} already done through another view", tensor->getName());
            }
        }
    }

    return res;
}

void BundleSlicer::addGraphNodes(HabanaGraph& slicedBundleGraph)
{
    HB_ASSERT(slicedBundleGraph.getExeSortedNodes().empty(), "sliced bundle graph is empty");

    for (const auto& sliceNode : m_sliceNodes)
    {
        GraphEditor::addNode(slicedBundleGraph, sliceNode, false);
    }
    createTempGraphVisualization(slicedBundleGraph, "operations");

    const std::unordered_set<TensorPtr> evictedTensors = getEvictions();
    for (auto& section : m_sections)
    {
        const auto& tensor = section.first->originalTensor;
        section.second.generateGraphSection(slicedBundleGraph, evictedTensors.count(tensor) != 0);
    }

    m_memcpyScheduler.scheduleBundleGraph(slicedBundleGraph);
    createTempGraphVisualization(slicedBundleGraph, "operations_and_data_moves");
}

pNode BundleSlicer::getSliceNode(const pNode& origNode) const
{
    pNode ret;
    ret = origNode->getSlice();
    ret->setName(fmt::format("{}_bundle_{}/op_{}", origNode->getNodeName(), m_bundleIdx, m_opIdx));
    ret->getNodeAnnotation().bundleInfo.set(BundleInfo(m_bundleIdx, m_bundleType, m_opIdx));
    ret->getNodeAnnotation().origBigNode = origNode;
    if (!origNode->getNodeAnnotation().fusedNodes.empty())
    {
        ret->getNodeAnnotation().fusedNodes = origNode->getNodeAnnotation().fusedNodes;
    }

    return ret;
}

bool BundleSlicer::shouldTensorBeEvicted(const TensorPtr&   tensor,
                                         const HabanaGraph& fullGraph,
                                         const NodeSet&     bundleNodes)
{
    NodePtr producer = fullGraph.getTensorProducer(tensor);
    if (producer == nullptr)
    {
        LOG_TRACE(BE_SLICER, "Tensor {} is a graph input and should not be evicted", tensor->getName());
        return false;
    }
    if ((producer != nullptr) && (bundleNodes.count(producer) == 0))
    {
        LOG_TRACE(BE_SLICER,
                  "Tensor {} is generated by {} which is not a part of the bundle and should not be evicted",
                  tensor->getName(),
                  producer->getNodeName());
        return false;
    }

    if (tensor->isUserManagedDram())
    {
        LOG_TRACE(BE_SLICER, "Tensor {} is user managed and should be evicted", tensor->getName());
        return true;
    }

    NodeList consumersInOriginalGraph = fullGraph.getTensorConsumers(tensor);
    for (const auto& node : consumersInOriginalGraph)
    {
        if (bundleNodes.count(node) == 0)
        {
            LOG_TRACE(BE_SLICER,
                      "Tensor {} is needed in a different bundle and should be evicted. Needed by node: {}",
                      tensor->getName(),
                      node->getNodeName());
            return true;
        }
    }

    LOG_TRACE(BE_SLICER, "Tensor {} is fully consumed in bundle", tensor->getName());
    return false;
}

void BundleSlicer::createTempGraphVisualization(HabanaGraph& tempGraph, const std::string& suffix)
{
    if (GCFG_SRAM_SLICER_GRAPH_VISUALIZATION.value())
    {
        GraphVisualization::graphVisualizationOnDemand(tempGraph, fmt::format("bundle_{:03}_{}", m_bundleIdx, suffix));
    }
}

void BundleSlicer::logOperation(const Operation& op) const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(BE_SLICER)) return;

    LOG_TRACE(BE_SLICER, "Add operation {}: {}", m_opIdx, op.originalNode->getNodeName());
    LOG_TRACE(BE_SLICER, "    Inputs: ");
    logOperands(op.inputs);
    LOG_TRACE(BE_SLICER, "    Outputs: ");
    logOperands(op.outputs);
}

void BundleSlicer::logOperands(const std::vector<pSliceReference>& operands) const
{
    for (uint32_t idx = 0; idx < operands.size(); ++idx)
    {
        bool inSram = operands[idx]->operand->resideInSRAM || operands[idx]->operand->originalTensor->inSram();
        LOG_TRACE(BE_SLICER, "        {}. {}({}) coord [{}], size [{}] from [{}]",
                idx,
                operands[idx]->operand->originalTensor->getName(),
                inSram ? "SRAM" : "DRAM",
                toString(operands[idx]->coordinates, ','),
                toString(operands[idx]->operand->chunkDimensions, ','),
                toString(operands[idx]->operand->originalTensor->getAllSizesInElements(), ','));
    }
}
