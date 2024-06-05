#include "add_cache_warmup.h"
#include "log_manager.h"
#include "tpc_kernel_loader.h"
#include "handle_memory_reuse.h"
#include "graph_editor.h"
#include "node_factory.h"
#include "types.h"
#include "reduction_node.h"
#include "slicer/node_dcore_rois_setter.h"
#include "compilation_hal_reader.h"

using namespace gc::layered_brain;
using namespace gc::access_pattern;

// m_origTensor is copied to protect its validity after GraphEditor::replaceTensor
AddCacheWarmup::AddCacheWarmup(HabanaGraph& graph, const TensorPtr& tensor, bool allocInSingleDCore)
: m_graph(graph),
  m_origTensor(tensor),
  m_producer(m_graph.getTensorProducer(m_origTensor)),
  m_singleDCore(allocInSingleDCore)
{
    HB_ASSERT_PTR(m_producer);
    m_producerOutputIdx = m_producer->getOutputIndexOfTensor(m_origTensor);
}

bool AddCacheWarmup::shouldHandle() const
{
    if (!GCFG_ENABLE_ADD_CACHE_WARMUP.value())
    {
        LOG_TRACE(LB_PARTIALS, "Cache warmup is disabled");
        return false;
    }
    if (!GCFG_ENABLE_CACHE_WARMUP_ON_SINGLE_DCORE.value() && shouldAllocTensorInSingleDcore())
    {
        LOG_TRACE(LB_PARTIALS, "Cache warmup is disabled when required for single DCore");
        return false;
    }

    HB_ASSERT(!m_producer->isLogicalOperation(), "Cache warmup isn't required for logical node");
    // The strides of the tensor should be in increasing order for cache warmup node requirements
    if (!areStridesAscending(m_origTensor->getNStridesInElements(), m_origTensor->getDim()))
    {
        LOG_WARN(LB_PARTIALS, "Cache warmup kernels doesn't allow transposed output, strides must be ascending");
        return false;
    }
    // check if output is already reduced
    if (m_origTensor->getRealReductionInfo().isReductionEnabled)
    {
        LOG_TRACE(LB_PARTIALS, "Cache warmup node isn't added for reduced output");
        return false;
    }
    // Cache warmup kernels does not supported 64 bit data types
    if (m_origTensor->is64BitElementSize())
    {
        LOG_WARN(LB_PARTIALS, "Cache warmup kernels are not supported for 64 bit DT");
        return false;
    }

    // TODO [SW-168305] - unblock when the cache WU kernels support dynamicity
    if (m_origTensor->isDynamicShape())
    {
        LOG_WARN(LB_PARTIALS, "Cache warmup kernels are not supported for dynamic shapes");
        return false;
    }

    if (is8BitFloat(m_origTensor->getElementType()))
    {
        LOG_WARN(LB_PARTIALS, "Cache warmup GUIDs are not supported for fp8");
        return false;
    }

    if (isPerforatedOnFCD(m_producer, m_origTensor))
    {
        LOG_WARN(LB_PARTIALS, "Cache warmup is not allowed if producer is perforated on dim 0");
        return false;
    }
    if (m_producer->getNodeAnnotation().perforationDim.has_value() &&
        tensorOverlapsOnPerforationDim(m_producer,
                                       m_producer->getNodeAnnotation().perforationDim.value(),
                                       m_origTensor) &&
        !shouldAllocTensorInSingleDcore())
    {
        LOG_WARN(LB_PARTIALS, "Cache warmup is blocked for Dcore ROIs overlap on the output.");
        // TODO SW-160325 - turn to assert after fixing failures. If the condition is valid, move it to
        // decision tree
        return false;
    }
    return true;
}

bool AddCacheWarmup::outputOverlapsWithInput() const
{
    const TensorPtr& output = m_producer->getOutput(m_producerOutputIdx);
    for (const TensorPtr& input : m_producer->getInputs())
    {
        if (!input || input->isShapeTensor()) continue;
        // In case the output is overlaped with the input, memsetting might override valid data
        if (MemoryReuseHandler::isStridedOverlap(input, output))
        {
            return true;
        }
    }
    return false;
}

bool AddCacheWarmup::shouldReplaceExistingMemset() const
{
    // SW-169160 - Disabled by default as currently, kernel implementation for CL aware memset
    // is not efficient enough. It's more efficient to use regular memset + memget.
    // Note - The scenario which memset is required + FCD > CL - was not seen in models within compilation job
    if (!GCFG_ENABLE_REPLACE_MEMSET_BY_CACHE_WARMUP.value()) return false;
    // For large FCD - use CL aware memset, instead of regular memset, as the function is the same.
    // For FCD smaller than CL - CL aware memset has bad performance, so use reular memset to zero mem, and use CL aware
    // memget to warmup cache.
    return (isMemsetBeforeExec() && !isFCDSmallerThanCL());
}

bool AddCacheWarmup::isFullyWritten() const
{
    // Only TPC nodes indicate they might not write the full output
    if (!HabanaGraph::runsOnTPC(m_producer)) return true;

    return checkedCast<TPCNode>(m_producer)->isOutputTensorFullyWritten(m_producerOutputIdx);
}

bool AddCacheWarmup::isMemsetBeforeExec() const
{
    // Only TPC nodes indicate they require memset Befor execution
    if (!HabanaGraph::runsOnTPC(m_producer)) return false;

    return checkedCast<TPCNode>(m_producer)
        ->isOutputTensorMemset(m_producerOutputIdx, deviceTypeToDeviceID(m_graph.getDeviceType()));
}

bool AddCacheWarmup::isFCDSmallerThanCL() const
{
    return m_producer->getOutput(m_producerOutputIdx)->getSizeInBytes(0) <
           m_graph.getHALReader()->getCacheLineSizeInBytes();
}

// Returns original <memset, reduction> nodes, which reset output before exec for producer
std::pair<NodePtr, NodePtr> AddCacheWarmup::getExistingMemsetReductionNodes() const
{
    if (!isMemsetBeforeExec())
    {
        return std::make_pair(nullptr, nullptr);
    }
    const auto& outputConsumers = m_graph.getTensorConsumers(m_producer->getOutput(m_producerOutputIdx));
    HB_ASSERT(outputConsumers.size() == 1,
              "expecting a single reduction consumer for memset before exec tensor, got {}",
              outputConsumers.size());
    const auto& reduction = *outputConsumers.begin();
    HB_ASSERT(reduction->getNodeType() == Node::TYPE_INTERNAL_REDUCTION,
              "expecting consumer to be reduction. Actual type {}",
              reduction->getNodeType());
    const auto& producers = m_graph.getNodeRealProducers(reduction, Node::TENSOR_TYPE_DATA);
    const auto& memsetIt =
        std::find_if(producers.begin(), producers.end(), [](const NodePtr& n) { return n->isMemset(); });
    HB_ASSERT(memsetIt != producers.end(), "expecting memset producer for memset before exec tensor");
    return std::make_pair(*memsetIt, reduction);
}

void AddCacheWarmup::run()
{
    if (!shouldHandle())
    {
        LOG_DEBUG(LB_PARTIALS,
                  "producer {} requires cache warmup for output {} but can't be handled",
                  m_producer->getNodeName(),
                  m_origTensor->getName());
        return;
    }
    LOG_DEBUG(LB_PARTIALS, "Partials writes cache warmup handling for {}", m_origTensor->getName());
    const auto& baseName = m_origTensor->getName();

    TensorPtr cacheWarmupOutput;
    TensorPtr reductionOutput;
    TensorPtr producerOutput;
    if (shouldReplaceExistingMemset())
    {
        auto [origMemset, origReduction] = getExistingMemsetReductionNodes();
        HB_ASSERT_PTR(origMemset);
        HB_ASSERT_PTR(origReduction);
        // reuse memset and reduction outputs
        cacheWarmupOutput = origMemset->getOutput(0);
        reductionOutput   = origReduction->getOutput(0);
        producerOutput    = m_origTensor;
    }
    else
    {
        // clone origin output
        cacheWarmupOutput = m_origTensor->clone();
        cacheWarmupOutput->setName(fmt::format("{}_cl_aware_zeros", baseName));
        reductionOutput = m_origTensor;
        producerOutput  = m_origTensor->clone();
        producerOutput->setName(fmt::format("{}_cl_aware_reduction_in", baseName));
    }
    NodePtr cacheWarmup = createCacheWarmupNode(cacheWarmupOutput);
    NodePtr reduction   = createReductionNode({cacheWarmupOutput, producerOutput}, reductionOutput);
    modifyGraph(cacheWarmup, reduction);
    setPerforation(cacheWarmup);
}

void AddCacheWarmup::setPerforation(NodePtr& cacheWarmup)
{
    // handle producer perforation before cache warmup perforation because they are dependent.
    setProducerPerforation();
    setCacheWarmupPerforation(cacheWarmup);
}

NodePtr AddCacheWarmup::createReductionNode(const TensorVector& inputs, const TensorPtr& output)
{
    // we assume that the cache manager handles this type and sets allocD for it
    ReductionOperation reductionOp = REDUCTION_UNORDERED_SET;
    HB_ASSERT(inputs.size() == 2, "reduction consumer of cache warmup should get 2 inputs");
    NodePtr reductionNode =
        NodeFactory::createNode(inputs,
                                {output},
                                &reductionOp,
                                NodeFactory::reductionNodeTypeName,
                                fmt::format("cache_warmup_reduction_for_tensor_{}", output->getName()));
    // logical operation pass already ran - handle locally to avoid retriggering it
    checkedCast<LogicalOpNode>(reductionNode)->runAndSetLogicalOp();
    return reductionNode;
}

// Return a cache warmup node, which provides the best performance, under the constraints of the output producer.
// The constraints and GUID selection is described in SW-162075.
NodePtr AddCacheWarmup::createCacheWarmupNode(const TensorPtr& warmupOutput)
{
    NodePtr cacheWarmupNode = createCacheWarmupNodeByGCFG(warmupOutput);
    if (cacheWarmupNode)
    {
        LOG_TRACE(LB_PARTIALS, "{}: Cache warmup selected by GCFG {}", HLLOG_FUNC, cacheWarmupNode->getNodeTypeStr());
        return cacheWarmupNode;
    }
    // Select node type by constraints on the output tensor
    if (isMemsetBeforeExec())
    {
        // Regular memset was already added as requested by the tpc kernel
        if (shouldReplaceExistingMemset())
        {
            // The caller should replace the existing memset and reduction nodes
            cacheWarmupNode = createClAwareNode(NodeFactory::clAwareMemsetNodeTypeName, warmupOutput);
        }
        else
        {
            // cl_aware_memset performance is bad on low FCD - keep the regular memset and add cl_aware_memget for
            // warmup.
            // TODO SW-167864 - can be replaced by perforating memset like the producer if possible
            cacheWarmupNode = createClAwareNode(NodeFactory::clAwareMemgetNodeTypeName, warmupOutput);
        }
    }
    else
    {
        if (!isFullyWritten() || outputOverlapsWithInput() || isFCDSmallerThanCL())
        {
            // Use cl_aware_memget when:
            // 1. not fully written or output overlap with input - to not override
            //    the existing tensor data
            // 2. low FCD (less than CL size) - hybrid kernel is doing only memgets anyway,
            //    using cl_aware_memget directly saves time
            cacheWarmupNode = createClAwareNode(NodeFactory::clAwareMemgetNodeTypeName, warmupOutput);
        }
        else
        {
            // Use the hybrid memset/memget kernel for best performance if FCD > CL
            cacheWarmupNode = createClAwareNode(NodeFactory::clAwareHybridNodeTypeName, warmupOutput);
        }
    }
    return cacheWarmupNode;
}

NodePtr AddCacheWarmup::createCacheWarmupNodeByGCFG(const TensorPtr& warmupOutput)
{
    enum class CacheWarmupKernel
    {
        AUTO = 0,
        CL_AWARE_MEMSET,
        CL_AWARE_MEMGET,
        CL_AWARE_HYBRID_MEMSET_MEMGET
    };
    const auto cacheWarmupKernel = static_cast<CacheWarmupKernel>(GCFG_PARTIALS_CACHE_WARMUP_KERNEL.value());

    switch (cacheWarmupKernel)
    {
        case CacheWarmupKernel::AUTO:
            return nullptr;  // let the caller create the node by its logic
        case CacheWarmupKernel::CL_AWARE_MEMSET:
            return createClAwareNode(NodeFactory::clAwareMemsetNodeTypeName, warmupOutput);
        case CacheWarmupKernel::CL_AWARE_MEMGET:
            return createClAwareNode(NodeFactory::clAwareMemgetNodeTypeName, warmupOutput);
        case CacheWarmupKernel::CL_AWARE_HYBRID_MEMSET_MEMGET:
            return createClAwareNode(NodeFactory::clAwareHybridNodeTypeName, warmupOutput);
        default:
            HB_ASSERT(false, "invalid PARTIALS_CACHE_WARMUP_KERNEL {}", cacheWarmupKernel);
    }
    return nullptr;
}

NodePtr AddCacheWarmup::createClAwareNode(std::string_view guid, const TensorPtr& output)
{
    const auto& baseName = m_origTensor->getName();
    NodePtr     cacheWarmupNode =
        NodeFactory::createNode({}, {output}, nullptr, guid, fmt::format("cache_warmup_for_{}", baseName));
    cacheWarmupNode->getNodeAnnotation().updateSplitToLogicalROIs(false);
    return cacheWarmupNode;
}

void AddCacheWarmup::modifyGraph(const NodePtr& cacheWarmup, const NodePtr& reduction)
{
    auto [origMemset, origReduction] = getExistingMemsetReductionNodes();
    if (shouldReplaceExistingMemset())
    {
        HB_ASSERT_PTR(origMemset);
        HB_ASSERT_PTR(origReduction);
        LOG_INFO(LB_PARTIALS, "Replace {} with {}", origMemset->getNodeName(), cacheWarmup->getNodeName());
        ReductionOperation reductionOp = checkedCast<ReductionNode>(reduction)->getReductionOperation();
        checkedCast<ReductionNode>(origReduction)->setParams(&reductionOp, sizeof(reductionOp));
        auto rc = GraphEditor::replaceNodes(m_graph, {origMemset}, {cacheWarmup});
        HB_ASSERT(rc == REPLACE_NODE_SUCCESS, "failed to replace cache warmup and reduction nodes with existing");
    }
    else
    {
        LOG_INFO(LB_PARTIALS, "Add {} ({})", cacheWarmup->getNodeName(), cacheWarmup->getNodeTypeStr());
        // The new producer output is input 1 of the reduction node
        auto newTensor = reduction->getInput(1);
        GraphEditor::replaceTensor(m_graph, m_producer, m_origTensor, newTensor);
        auto rc = GraphEditor::addNodes(m_graph, {cacheWarmup, reduction});
        HB_ASSERT(rc == true, "failed to add cache warmup and reduction nodes to the graph");
        if (origMemset)
        {
            // make sure the cache warmup runs after the memset, in case the memset didn't happen in the correct DCore
            m_graph.addControlDependency(origMemset, cacheWarmup, Tensor::ControlEdgeType::SCHEDULE);
        }
    }
    // Schedule the chache warmup node to be executed before the producer
    m_graph.addControlDependency(cacheWarmup, m_producer, Tensor::ControlEdgeType::SCHEDULE);
    // Copy producer bundle info to the new nodes
    if (m_producer->getNodeAnnotation().bundleInfo.is_set())
    {
        const BundleInfo bundleInfo = BundleInfo(m_producer->getNodeAnnotation().bundleInfo.value());
        reduction->getNodeAnnotation().bundleInfo.set(bundleInfo);
        cacheWarmup->getNodeAnnotation().bundleInfo.set(bundleInfo);
    }
}

void AddCacheWarmup::setProducerPerforation()
{
    // If the producer slice ROI isn't set, set the slice ROI for the producer as well.
    // It may be required if the producer is set in single DCore
    if (!m_producer->getNodeAnnotation().sliceROI.has_value())
    {
        m_producer->getNodeAnnotation().sliceROI = NodeTile(m_producer->getNodeAccessPattern()->getNodeResolution());
    }

    if (!m_producer->getNodeAnnotation().perforationDim.has_value() && canPerforateProducer())
    {
        // It is better to perforate the producer, because otherwise we will have to
        // execute it on a single dcore, for the cache warmup to be effective.

        // Select a perforation dim.
        // We want to select a dim that leads to the lowest amount of partials writes between engines from
        // different dcores, because this way the cache warmup is the most effective.
        // (Usually it means that we need to perforate on node dim that is mapped to
        // m_producer->getOutput(m_producerOutputIdx)'s most external dim).
        const NodeTile::Geometry& fullSliceShape = m_producer->getNodeAnnotation().sliceROI->geometry;
        auto perforationDim = findOptimalPerforationDim([this, &fullSliceShape](const Dim nodeDim) {
            return fullSliceShape[nodeDim] >= m_graph.getHALReader()->getNumDcores();
        });
        if (!perforationDim.has_value())
        {
            // Try to split the work to less dcores.
            perforationDim =
                findOptimalPerforationDim([&fullSliceShape](const Dim nodeDim) { return fullSliceShape[nodeDim] > 1; });
        }
        if (perforationDim.has_value())
        {
            LOG_DEBUG(LB_PARTIALS,
                      "Perforate node: {} on nodeDim: {}",
                      m_producer->getNodeName(),
                      perforationDim.value());
            // Call splitToDcoreROIs with granularity 1 because this node is perforated as standalone
            NodeDcoreROIsSetter(m_producer, CompilationHalReader::getHalReader()->getNumDcores())
                .splitToDcoreROIs(perforationDim.value(), 1, std::nullopt);
        }
    }

    if (shouldAllocTensorInSingleDcore())
    {
        // If the tensor requires access on a single DCore, and the output is not RMW, it means not all engines write to
        // it. The output might be written by representative engines or overriding each other. It's unknown which
        // engines on which DCore are going to write it. Thus, can't set a specific DCore to warmup the tensor, as other
        // DCore might do the actual write. This access pattern is currently not expected for any TPC kernel.
        // If the producer is not perforated we don't care about its access pattern and we set it as single dcore.
        HB_ASSERT(isRmwOutput() || !m_producer->getNodeAnnotation().perforationDim.has_value(),
                  "unexpected single DCore without RMW or unperforated producer {}",
                  m_producer->getNodeName());

        // Set the full node ROI in a single DCore if wasn't set by a previous output which required warmup
        if (!isPerforatedOnSingleDCore())
        {
            resetProducerPerforation();
            setNodeRoiInSingleDcore(m_producer, m_singleDCoreIndex);
        }
    }
}

std::optional<Dim>
AddCacheWarmup::findOptimalPerforationDim(std::function<bool(const Dim nodeDim)> pred) const
{
    const auto& output = m_producer->getOutput(m_producerOutputIdx);
    for (int tensorDim = output->getDim() - 1; tensorDim > 0; --tensorDim)
    {
        const auto mappedIndexSpaceDim = m_producer->getNodeAccessPattern()->getIndexSpaceDim(output, tensorDim);
        // We don't want to perforate on a dim that has intersections on any of the tensor's relevant dims.
        if (tensorOverlapsOnPerforationDim(m_producer, mappedIndexSpaceDim, output)) continue;

        if (pred(mappedIndexSpaceDim))
        {
            LOG_DEBUG(LB_PARTIALS,
                      "Node: {}, output {} tensorDim: {} is mapped to nodeDim: {}",
                      m_producer->getNodeName(),
                      output->getName(),
                      tensorDim,
                      mappedIndexSpaceDim);
            return mappedIndexSpaceDim;
        }
    }
    return std::nullopt;
}

bool AddCacheWarmup::canPerforateProducer() const
{
    return (!m_singleDCore && GCFG_ENABLE_LAYERED_BRAIN_PERFORATION.value() && !isOutputAllRequired());
}

bool AddCacheWarmup::isRmwOutput() const
{
    // Only TPC nodes indicate they RMW the output
    if (!HabanaGraph::runsOnTPC(m_producer)) return false;
    return checkedCast<TPCNode>(m_producer)
        ->isOutputTensorRmw(m_producerOutputIdx,
                            deviceTypeToDeviceID(CompilationHalReader::getHalReader()->getDeviceType()));
}

bool AddCacheWarmup::isOutputAllRequired() const
{
    if (!HabanaGraph::runsOnTPC(m_producer)) return false;
    const auto& tpcNode = checkedCast<TPCNode>(m_producer);
    return tpcNode->isOutputTensorAllRequired(m_producerOutputIdx);
}

// This function must be called after the cache warmup bundle info is set, as it overrides its fields
void AddCacheWarmup::setCacheWarmupPerforation(NodePtr& cacheWarmup)
{
    TpcKernelLoader kernelLoader(&m_graph);
    auto            res = kernelLoader.load(cacheWarmup);
    HB_ASSERT(res, "Could not instantiate node {} <GUID: {}>", cacheWarmup->getNodeName(), cacheWarmup->getGUID());

    // Init cache warmup slice ROI, required to set or validate its perforation
    cacheWarmup->getNodeAnnotation().sliceROI = NodeTile(cacheWarmup->getNodeAccessPattern()->getNodeResolution());

    if (shouldAllocTensorInSingleDcore())
    {
        setNodeRoiInSingleDcore(cacheWarmup, m_singleDCoreIndex);
    }
    else
    {
        // Project perforation from producer to the cache warmup node, through the common output
        projectPerforationByCommonTensor(m_producer,
                                         cacheWarmup,
                                         m_producer->getOutput(m_producerOutputIdx),
                                         cacheWarmup->getOutput(0));
    }
}

bool AddCacheWarmup::shouldAllocTensorInSingleDcore() const
{
    LOG_DEBUG(LB_PARTIALS,
              "{}: allocInSingleDCore {}, producer {} perforation dim, with {} dcore ROIs",
              HLLOG_FUNC,
              m_singleDCore,
              !m_producer->getNodeAnnotation().perforationDim.has_value() ? "doens't have" : "has",
              m_producer->getNodeAnnotation().m_dcoreROIs.size());
    // Allocate in single DCore if partials detection decided, or if the producer has no perforation
    return m_singleDCore || !m_producer->getNodeAnnotation().perforationDim.has_value();
}

bool AddCacheWarmup::isPerforatedOnSingleDCore() const
{
    // check that the producer full size is set to 1 dcore, and the rest are 0's
    auto  numDcores = CompilationHalReader::getHalReader()->getNumDcores();
    auto& dcoreROIs = m_producer->getNodeAnnotation().m_dcoreROIs;
    if (dcoreROIs.size() != numDcores) return false;

    const NodeTile::Geometry& fullSliceShape = m_producer->getNodeAnnotation().sliceROI->geometry;
    const NodeTile::Geometry  zeroSliceShape(fullSliceShape.size(), 0);
    for (unsigned dcoreIdx = 0; dcoreIdx < numDcores; dcoreIdx++)
    {
        const auto& dcoreShape = dcoreROIs.at(dcoreIdx).size;
        if (dcoreIdx == m_singleDCoreIndex)
        {
            if (!std::equal(fullSliceShape.begin(), fullSliceShape.end(), dcoreShape)) return false;
        }
        else
        {
            if (!std::equal(zeroSliceShape.begin(), zeroSliceShape.end(), dcoreShape)) return false;
        }
    }
    LOG_DEBUG(LB_PARTIALS, "{}: node already perforated on single DCore: {}", HLLOG_FUNC, m_producer->getNodeName());
    return true;
}

void AddCacheWarmup::resetProducerPerforation()
{
    auto& dcoreROIs = m_producer->getNodeAnnotation().m_dcoreROIs;
    if (!dcoreROIs.empty())
    {
        LOG_WARN(LB_PARTIALS,
                 "{}: node is perforated, but need to be executed in single dcore {}",
                 HLLOG_FUNC,
                 m_producer->getNodeName());
        dcoreROIs.clear();
        m_producer->getNodeAnnotation().perforationDim.reset();
        if (m_producer->getNodeAnnotation().bundleInfo.is_set())
        {
            m_producer->getNodeAnnotation().bundleInfo->perforationGroup.reset();
        }
    }
}

// TODO - SW-162552 - move to a dedicated component
void AddCacheWarmup::projectPerforationByCommonTensor(const NodePtr&   source,
                                                      NodePtr&         dest,
                                                      const TensorPtr& sourceTensor,
                                                      const TensorPtr& destTensor)
{
    std::vector<NodeTile> dcoreNodeTiles;
    std::optional<TensorTile> baseTensorTile;

    // map each source dcore roi to dest dcore roi through the common tensor
    auto numDcores = source->getNodeAnnotation().m_dcoreROIs.size();
    for (unsigned dcoreIdx = 0; dcoreIdx < numDcores; dcoreIdx++)
    {
        const auto& sourceDcoreRoi = source->getNodeAnnotation().m_dcoreROIs[dcoreIdx];
        NodeTile    destNodeTile(dest->getNodeAccessPattern()->getNodeResolution().size(), 0, 0);
        if (!doesArrayContainZeros(sourceDcoreRoi.size))
        {
            TensorTile tensorTile = getDcoreRoiTensorTile(sourceDcoreRoi, source, sourceTensor);
            if (!baseTensorTile.has_value())
            {
                baseTensorTile = tensorTile;
            }
            destNodeTile = getProjectedNodeTile(dest, destTensor, *baseTensorTile, tensorTile);
        }
        dcoreNodeTiles.push_back(destNodeTile);
    }

    std::optional<unsigned> prforationDim = getProjectedPerforationDim(source, dest, sourceTensor, destTensor);
    std::optional<unsigned> perforationGroup =
        prforationDim.has_value() && source->getNodeAnnotation().bundleInfo.is_set()
            ? source->getNodeAnnotation().bundleInfo->perforationGroup
            : std::nullopt;
    NodeDcoreROIsSetter(dest, numDcores).setDcoreROIs(dcoreNodeTiles, prforationDim, perforationGroup);
}

std::optional<unsigned> AddCacheWarmup::getProjectedPerforationDim(const NodePtr&   source,
                                                                   const NodePtr&   dest,
                                                                   const TensorPtr& sourceTensor,
                                                                   const TensorPtr& destTensor)
{
    std::optional<unsigned> destPerforationDim;
    auto                    tensorPerfDims = getTensorPerforationDims(source, sourceTensor);
    std::set<unsigned>      nodePerfDims;
    for (auto tensorDim : tensorPerfDims)
    {
        auto nodeDim = dest->getNodeAccessPattern()->getIndexSpaceDim(destTensor, tensorDim);
        nodePerfDims.insert(nodeDim);
    }
    HB_ASSERT(nodePerfDims.size() <= 1, "expecting a single perforation dim for {}", dest->getNodeName());
    if (nodePerfDims.size() == 1)
    {
        destPerforationDim = *nodePerfDims.begin();
    }
    return destPerforationDim;
}

void AddCacheWarmup::setNodeRoiInSingleDcore(const NodePtr& node, uint64_t singleDCoreIndex)
{
    HB_ASSERT(node->getNodeAnnotation().sliceROI.has_value(),
              "Missing sliced node ROI for node {}",
              node->getNodeName());
    const NodeTile&       fullSliceROI = node->getNodeAnnotation().sliceROI.value();
    auto                  numDcores    = CompilationHalReader::getHalReader()->getNumDcores();
    std::vector<NodeTile> dcoreNodeTiles {numDcores, NodeTile(fullSliceROI.geometry.size(), 0, 0)};
    dcoreNodeTiles.at(singleDCoreIndex) = fullSliceROI;

    NodeDcoreROIsSetter(node, numDcores).setDcoreROIs(dcoreNodeTiles, std::nullopt, std::nullopt);
}

// Returns the tensor dims which match the node annotation perforation dim.
// Returns empty vector when no tensor dim matches the perforation dim, or no value for node perforation dim.
std::vector<unsigned> AddCacheWarmup::getTensorPerforationDims(const NodePtr& node, const TensorPtr& tensor)
{
    std::vector<unsigned> perfDims;
    for (auto dim = 0; dim < tensor->getDim(); ++dim)
    {
        auto nodeDim = node->getNodeAccessPattern()->getIndexSpaceDim(tensor, dim);
        if (node->getNodeAnnotation().perforationDim == nodeDim)
        {
            perfDims.push_back(dim);
        }
    }
    return perfDims;
}

TensorTile AddCacheWarmup::getDcoreRoiTensorTile(const DcoreROI& dcoreROI, const NodePtr& node, const TensorPtr& tensor)
{
    NSizeArray dcoreRoiSize;
    std::copy(std::begin(dcoreROI.size), std::end(dcoreROI.size), dcoreRoiSize.begin());
    NStrideArray dcoreRoiOffset;
    std::copy(std::begin(dcoreROI.baseOffset), std::end(dcoreROI.baseOffset), dcoreRoiOffset.begin());
    NodeTile roiNodeTile =
        NodeTile(node->getNodeAccessPattern()->getNodeResolution().size(), dcoreRoiSize, dcoreRoiOffset);
    return node->getNodeAccessPattern()->getTensorTile(tensor, roiNodeTile);
}

NodeTile AddCacheWarmup::getProjectedNodeTile(const NodePtr&    node,
                                              const TensorPtr&  tensor,
                                              const TensorTile& baseTensorTile,
                                              const TensorTile& tensorTileQueriedDcore)
{
    TensorTile clippedDcoreTensorTile(tensorTileQueriedDcore);
    for (auto idx = 0; idx < tensor->getDim(); idx++)
    {
        // clip the size of the last slice according to tensor bounds
        auto dimensionOffset = tensorTileQueriedDcore.offset.at(idx) - baseTensorTile.offset.at(idx);
        clippedDcoreTensorTile.geometry.at(idx) =
            std::min(tensor->getSizeInElements(idx) - dimensionOffset, tensorTileQueriedDcore.geometry.at(idx));
        auto tensorGranularity = node->getNodeAccessPattern()->getTensorGranularity(tensor).geometry.at(idx);

        HB_ASSERT(!(clippedDcoreTensorTile.geometry.at(idx) > tensorGranularity &&
                    clippedDcoreTensorTile.geometry.at(idx) % tensorGranularity != 0),
                  "cache warmup node granulariy in dim {} does not allow slicing to element size {}",
                  idx,
                  clippedDcoreTensorTile.geometry.at(idx));
    }
    // TODO - SW-162552 - replace with projection through BVD container
    return node->getNodeAccessPattern()->getNodeTile(tensor, clippedDcoreTensorTile);
}

bool AddCacheWarmup::isPerforatedOnFCD(const NodePtr& node, const TensorPtr& tensor)
{
    auto fcdNodeDim = node->getNodeAccessPattern()->getIndexSpaceDim(tensor, 0);
    return (node->getNodeAnnotation().perforationDim == fcdNodeDim);
}

bool AddCacheWarmup::tensorOverlapsOnPerforationDim(const NodePtr& node, Dim nodeDim, const TensorPtr& tensor)
{
    const auto& tensorDims = node->getNodeAccessPattern()->getTensorDims(nodeDim, tensor);
    if (tensorDims.size() > 0)
    {
        const auto granuleIntersection = node->getNodeAccessPattern()->getTensorOverlap(tensor);
        return std::any_of(tensorDims.cbegin(), tensorDims.cend(), [&granuleIntersection](Dim tensorDim) {
            return granuleIntersection.geometry.at(tensorDim) > 0;
        });
    }
    // perforation dim wasn't found for this tensor - there's no perforation for it
    return true;
}
