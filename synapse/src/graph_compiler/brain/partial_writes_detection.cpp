#include "partial_writes_detection.h"
#include "brain_data.h"
#include "bundle_view.h"
#include "cache_types.h"
#include "defs.h"
#include "dma_cost_model.h"
#include "compilation_hal_reader.h"
#include "log_manager.h"
#include "synapse_common_types.h"
#include "reduced_dims_detector.h"

using namespace gc::layered_brain;

static TStride getRealTensorOffset(const TensorPtr& t)
{
    TStride   offset     = 0;
    TensorPtr realTensor = t;
    while (realTensor->isAliasedTensor())
    {
        offset += realTensor->getAliasedByteOffset();
        realTensor = realTensor->getAliasTensor();
    }
    return offset;
}

static bool isNodeOutsideTheBundle(const NodePtr& n, unsigned bundleIndex)
{
    return !n->getNodeAnnotation().bundleInfo.is_set() || n->getNodeAnnotation().bundleInfo->bundleIndex != bundleIndex;
}

class AnalysisLog
{
public:
    AnalysisLog(const TensorPtr& output, const NodePtr& producer, bool isBpt)
    : m_output(output),
      m_producer(producer),
      m_outputIndex(m_producer->getOutputIndexOfTensor(m_output)),
      m_isBpt(isBpt)
    {
    }

    void print(const std::string& header) const { LOG_DEBUG(LB_PARTIALS, "{}: {}", header, getCommonLog()); }

    void printTensorData() const
    {
        LOG_DEBUG(LB_PARTIALS, "Node {} ({})", m_producer->getNodeName(), m_producer->getNodeTypeStr());
        LOG_DEBUG(LB_PARTIALS, "Output index {} - Tile:", m_outputIndex);
        printOperand(m_output);
        const TensorPtr& bigTensor = m_output->getTensorAnnotation().origBigTensor;
        if (bigTensor)
        {
            LOG_DEBUG(LB_PARTIALS, "Big Tensor:");
            printOperand(bigTensor);
        }
        const TensorPtr& real = Tensor::getRealTensor(m_output);
        if (real != m_output)
        {
            LOG_DEBUG(LB_PARTIALS, "Real Tensor:");
            printOperand(real);
            printAliasChain();
        }
    }

private:
    std::string getCommonLog() const
    {
        return fmt::format("{} out index {} {} sizes={}",
                           m_producer->getNodeTypeStr(),
                           m_outputIndex,
                           m_producer->getNodeAnnotation().bundleInfo.is_set() ? (m_isBpt ? "BPT" : "intermediate")
                                                                               : "unbundled",
                           m_output->getDimSizesStr());
    }

    void printOperand(const TensorPtr& t) const
    {
        LOG_DEBUG(LB_PARTIALS, "    {}", t->getName());
        LOG_TRACE(LB_PARTIALS,
                  "    offset in real (0x{:x}) ({}) sizes={} minSizes={} strides={} isDense={} isPersistent={}",
                  getRealTensorOffset(t),
                  getStringFromSynDataType(t->getElementType()),
                  t->getDimSizesStr(),
                  t->getDimSizesStr(false, true),
                  t->getStridesStr(),
                  t->isDenseLayout(),
                  t->isPersistent() ? "1" : "0");
        if (t->isAliasedTensor())
        {
            LOG_TRACE(LB_PARTIALS, "    is an alias of {}", Tensor::getRealTensor(t)->getName());
        }
    }

    void printAliasChain() const
    {
        LOG_DEBUG(LB_PARTIALS, "Alias chain:");
        TensorPtr nextAlias = m_output;
        while (nextAlias->isAliasedTensor())
        {
            LOG_DEBUG(LB_PARTIALS,
                      "{} is aliased to {} with offset {}",
                      nextAlias->getName(),
                      nextAlias->getAliasTensor()->getName(),
                      nextAlias->getAliasedByteOffset());
            nextAlias = nextAlias->getAliasTensor();
        }
    }

    TensorPtr m_output;
    NodePtr   m_producer;
    unsigned  m_outputIndex;
    bool      m_isBpt;
};

PartialWritesDetector::PartialWritesDetector(const TensorPtr& output, const HabanaGraph& g)
: m_output(output), m_graph(g), m_producer(g.getTensorProducer(output))
{
    HB_ASSERT_PTR(m_output);
    HB_ASSERT_PTR(m_producer);
    m_outputIndex = m_producer->getOutputIndexOfTensor(m_output);
}

std::optional<PartialWritesDetector::PartialsDecision> PartialWritesDetector::checkTensor() const
{
    PartialsDecision decision;
    if (!shouldHandleNode(m_producer))
    {
        return decision;
    }
    if (m_output->getTotalSizeInBytes() == 0)
    {
        LOG_TRACE(LB_PARTIALS, "{}: zero sized output is discarded", HLLOG_FUNC);
        return decision;
    }
    LOG_TRACE(LB_PARTIALS,
              "{}: Check {} output: tensor {} - node {}",
              HLLOG_FUNC,
              m_producer->getNodeTypeStr(),
              m_output->getName(),
              m_producer->getNodeName());

    AnalysisLog log(m_output, m_producer, isSliceOfBPT());

    if (isAllRequired())
    {
        LOG_DEBUG(LB_PARTIALS, "{}: All required", HLLOG_FUNC);
        return checkUnsliceableTensor();
    }
    else
    {
        if (m_output->isDynamicDim(0))
        {
            LOG_DEBUG(LB_PARTIALS, "{}: Dynamic dim 0", HLLOG_FUNC);
            log.print("=> setWarmupCacheRequired: dynamic FCD");
            setWarmupCacheRequired(decision);
            return decision;
        }
        else
        {
            if (isIsmeAligned() && !isPartialWrite())
            {
                if (isTpcSparseAccess())
                {
                    LOG_DEBUG(LB_PARTIALS, "{}: Sparse access", HLLOG_FUNC);
                    return checkUnsliceableTensor();
                }
                else
                {
                    if (isReducedTensor())
                    {
                        LOG_DEBUG(LB_PARTIALS, "{}: Reduced tensor", HLLOG_FUNC);
                        if (perforatedOnNonReducedDim())
                        {
                            LOG_DEBUG(LB_PARTIALS, "{}: Can perforate non reduced dim", HLLOG_FUNC);
                            return checkTensorSizeAndStrides();
                        }
                        else
                        {
                            LOG_DEBUG(LB_PARTIALS, "{}: Can't perforate non reduced dim", HLLOG_FUNC);
                            log.print(
                                fmt::format("=> setAllocFullTensorInSingleDcore: IS reduction, no perforation, {}",
                                            isRmw() ? "RMW" : "no RMW"));
                            setAllocFullTensorInSingleDcore(decision);
                            if (isTensorLargerThanCache())
                            {
                                LOG_DEBUG(LB_PARTIALS, "{}: Tensor is too large to fit single DCore cache", HLLOG_FUNC);
                                return std::nullopt;
                            }
                            return decision;
                        }
                    }
                    else
                    {
                        return checkTensorSizeAndStrides();
                    }
                }
            }
            else
            {
                LOG_DEBUG(LB_PARTIALS, "{}: Partial write or unaligned ISME", HLLOG_FUNC);
                log.print(fmt::format("=> setWarmupCacheRequired: {}",
                                      !isIsmeAligned() ? fmt::format("ISME unaligned: ISME {}", getFcdIsmeInBytes())
                                                       : "partial write"));
                setWarmupCacheRequired(decision);
                return decision;
            }
        }
    }
}

PartialWritesDetector::PartialsDecision PartialWritesDetector::checkUnsliceableTensor() const
{
    PartialsDecision decision;

    AnalysisLog log(m_output, m_producer, isSliceOfBPT());

    if (isTensorLargerThanCache())
    {
        LOG_DEBUG(LB_PARTIALS, "{}: tensor larger than cache", HLLOG_FUNC);
        log.print("=> setWriteInNoAlloc: tensor too large");
        setWriteInNoAlloc(decision);
    }
    else
    {
        LOG_DEBUG(LB_PARTIALS, "{}: tensor not larger than cache", HLLOG_FUNC);
        log.print(fmt::format("=> setAllocFullTensorInSingleDcore: {} unsliceable tensor {}",
                              isAllRequired() ? "all required" : "sparse",
                              isRmw() ? "RMW" : "no RMW"));
        setAllocFullTensorInSingleDcore(decision);
    }
    return decision;
}

PartialWritesDetector::PartialsDecision PartialWritesDetector::checkTensorSizeAndStrides() const
{
    PartialsDecision decision;

    AnalysisLog log(m_output, m_producer, isSliceOfBPT());

    TStride fcdStride               = m_output->getAllStridesInBytes().at(1);
    bool    sliceStridesAligned     = isSizeCLAligned(fcdStride, true /* allowExactHalfCL */);
    TStride offsetInBig             = getRealTensorOffset(m_output);
    bool    sliceOffsetAlignedInBPT = isSliceOfBPT() ? isSizeCLAligned(offsetInBig) : true;
    if (sliceStridesAligned && sliceOffsetAlignedInBPT)
    {
        LOG_TRACE(LB_PARTIALS, "{}: No partials writes, tensor is good", HLLOG_FUNC);
    }
    else
    {
        LOG_TRACE(LB_PARTIALS, "{}: fcdStride {}, offsetInBig {}", HLLOG_FUNC, fcdStride, offsetInBig);
        log.print(fmt::format("=> setWarmupCacheRequired: unaligned: tile FCD stride {} {}, offset {}",
                              fcdStride,
                              sliceStridesAligned ? "(aligned)" : "(unaligned)",
                              isSizeCLAligned(offsetInBig) ? "(aligned)" : "(unaligned)"));
        setWarmupCacheRequired(decision);
    }
    return decision;
}

bool PartialWritesDetector::shouldHandleNode(const NodePtr& node) const
{
    bool isSupportedBundlingState = node->getNodeAnnotation().bundleInfo.is_set() ||
                                    GCFG_ENABLE_LB_PARTIALS_WRITE_UNBUNDELED_NODES_HANDLING.value();
    bool isSupportedMME = HabanaGraph::runsOnMME(node) && GCFG_ENABLE_LB_PARTIALS_WRITE_MME_HANDLING.value();
    bool isSupportedTpc = HabanaGraph::runsOnTPC(node) && GCFG_ENABLE_LB_PARTIALS_WRITE_TPC_HANDLING.value();
    return isSupportedBundlingState && (isSupportedMME || isSupportedTpc);
}

bool PartialWritesDetector::isAllRequired() const
{
    if (!m_producer || !HabanaGraph::runsOnTPC(m_producer)) return false;
    const auto& tpcNode = static_cast<TPCNode&>(*m_producer);
    return tpcNode.isOutputTensorAllRequired(m_outputIndex);
}

bool PartialWritesDetector::isRmw() const
{
    // Relevant only to TPC nodes
    auto tpcNode = std::dynamic_pointer_cast<TPCNode>(m_producer);
    if (!tpcNode) return false;
    return tpcNode->isOutputTensorRmw(m_outputIndex,
                                      deviceTypeToDeviceID(CompilationHalReader::getHalReader()->getDeviceType()));
}

bool PartialWritesDetector::isPartialWrite() const
{
    if (HabanaGraph::runsOnMME(m_producer))
    {
        const auto& mmeNode = static_cast<MmeNode&>(*m_producer);
        return mmeNode.isOutputTensorPartialWrites(m_outputIndex);
    }
    else if (HabanaGraph::runsOnTPC(m_producer))
    {
        const auto& tpcNode = static_cast<TPCNode&>(*m_producer);
        return tpcNode.isOutputTensorPartialWrites(m_outputIndex);
    }
    return false;
}

bool PartialWritesDetector::isIsmeAligned() const
{
    // Need to check for TPC node only
    if (!HabanaGraph::runsOnTPC(m_producer)) return true;
    bool ismeAligned = isSizeCLAligned(getFcdIsmeInBytes());
    // If the node dim corresponding to the output FCD has small resolution - it won't create a bad race on the cache -
    // avoid handling this case
    const auto&                          accessPattern = m_producer->getNodeAccessPattern();
    const NodeAccessPattern::Resolution& resolution    = accessPattern->getNodeResolution();
    auto                                 fcdNodeDim    = accessPattern->getIndexSpaceDim(m_output, 0);
    auto                                 fcdResolution = resolution.at(fcdNodeDim);
    return ismeAligned || fcdResolution == 1;
}

bool PartialWritesDetector::isSizeFullCLMult(TSize bytes, bool allowExactHalfCL) const
{
    const TSize sizeToAlignTo = CompilationHalReader::getHalReader()->getCacheLineSizeInBytes();
    bool        aligned       = (bytes % sizeToAlignTo == 0);
    // In some cases the perf degradation is avoided if the tensor is exactly half CL size
    bool sizeIsHalfCL = allowExactHalfCL ? (bytes == sizeToAlignTo / 2) : false;
    return aligned || sizeIsHalfCL;
}

bool PartialWritesDetector::isSizeCLAligned(TSize bytes, bool allowExactHalfCL) const
{
    if (!GCFG_ENABLE_LB_PARTIALS_DETECTION_HALF_CL_REFINEMENT.value())
    {
        return isSizeFullCLMult(bytes, allowExactHalfCL);
    }
    const TSize sizeToAlignTo = CompilationHalReader::getHalReader()->getCacheLineSizeInBytes() / 2;
    // In some cases perf degradation is avoided if tensor is aligned to half CL. It happens because of these two ECOs:
    // Cache ECO: Coalesce 2 transactions of 128B to 1 transaction of 256B.
    // TPC ECO: Coalesce 2 consecutive partial transactions as long the coalescing point is the cache-line middle(128B).
    return (bytes % sizeToAlignTo == 0);
}

bool PartialWritesDetector::isTpcSparseAccess() const
{
    return (HabanaGraph::runsOnTPC(m_producer) && m_output->getTensorAnnotation().sparseAccess);
}

bool PartialWritesDetector::isTensorLargerThanCache() const
{
    auto cacheSize = CompilationHalReader::getHalReader()->getSRAMSizeInBytes();
    cacheSize /= CompilationHalReader::getHalReader()->getNumDcores();
    cacheSize *= GCFG_FRAGMENTATION_COMPENSATION_FACTOR.value();
    return (m_output->getTotalSizeInBytes() > cacheSize);
}

bool PartialWritesDetector::isReducedTensor() const
{
    // Check reduction only for TPC nodes
    if (!HabanaGraph::runsOnTPC(m_producer)) return false;
    ReducedDimsDetector detector(m_producer);
    auto                reducedDims = detector.getReducedNodeDimsForOutput(m_output);
    return !reducedDims.empty();
}

TSize PartialWritesDetector::getFcdIsmeInBytes() const
{
    auto granularity = m_producer->getNodeAccessPattern()->getTensorGranularity(m_output);
    return granularity.geometry.at(0) * m_output->getElementSizeInBytes();
}

bool PartialWritesDetector::perforatedOnNonReducedDim() const
{
    // If the node is perforated - the slicer found a non reduced BVD to slice.
    // If dcore rois is empty - node isn't perforated
    return !m_producer->getNodeAnnotation().m_dcoreROIs.empty();
}

bool PartialWritesDetector::isSliceOfBPT() const
{
    // This function handles a bundled sliced tensor - block non bundled producers
    if (!m_producer->getNodeAnnotation().bundleInfo.is_set()) return false;

    TensorPtr real             = Tensor::getRealTensor(m_output);
    auto      realLastProducer = m_graph.getTensorProducer(real);

    // The slice is written to the real tensor -
    // Check if the real tensor is outside the bundle - either its producer is the join node on the bundle boundary, or
    // it's another node after the join. If it's after the join it must be outside the bundle (different or none bundle
    // index).
    bool isBpt = Node::isJoinNode(realLastProducer) ||
                 isNodeOutsideTheBundle(realLastProducer, m_producer->getNodeAnnotation().bundleInfo->bundleIndex);
    return isBpt;
}

void PartialWritesDetector::setWarmupCacheRequired(PartialsDecision& decision) const
{
    LOG_TRACE(LB_PARTIALS, "{}", HLLOG_FUNC);
    decision.warmupCache = true;
    setWriteInAllocD(decision);
}

void PartialWritesDetector::setWriteInNoAlloc(PartialsDecision& decision) const
{
    LOG_TRACE(LB_PARTIALS, "{}", HLLOG_FUNC);
    decision.cacheDirective = CacheDirective::NoAllocate;
    AnalysisLog log(m_output, m_producer, isSliceOfBPT());
    log.printTensorData();
}

void PartialWritesDetector::setAllocFullTensorInSingleDcore(PartialsDecision& decision) const
{
    LOG_TRACE(LB_PARTIALS, "{}", HLLOG_FUNC);
    decision.allocInSingleDCore = true;
    if (!isRmw())
    {
        // Handling of non RMW is defined for representative IS elements writing this tensor,
        // but it's unexpected besides for BN, which is not expected to get here for !RMW outputs
        LOG_WARN(LB_PARTIALS, "{} unexpected single core without RMW", HLLOG_FUNC);
    }
    AnalysisLog log(m_output, m_producer, isSliceOfBPT());
    log.print("=> setWarmupCacheRequired: tensor in single DCore");
    setWarmupCacheRequired(decision);
}

void PartialWritesDetector::setWriteInAllocD(PartialsDecision& decision) const
{
    LOG_TRACE(LB_PARTIALS, "{}", HLLOG_FUNC);
    decision.cacheDirective = CacheDirective::DcoreAllocate;
    AnalysisLog log(m_output, m_producer, isSliceOfBPT());
    log.printTensorData();
}