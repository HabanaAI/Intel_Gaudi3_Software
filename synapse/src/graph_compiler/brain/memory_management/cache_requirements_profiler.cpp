#include "cache_requirements_profiler.h"

#include "reduction_node.h"
#include "types.h"

using namespace gc::layered_brain;

InputCacheUsageProfile CacheRequirementProfiler::inputProfile(size_t opIdx, size_t inputIdx)
{
    InputCacheUsageProfile profile {};

    initInputAnalysis(opIdx, inputIdx);
    setProducerProperties(profile);
    setConsumersProfile(profile);
    setBPTAccess(profile);
    setAllRequired(profile);
    setSize(profile);
    setNumReads(profile);

    return profile;
}

OutputCacheUsageProfile CacheRequirementProfiler::outputProfile(size_t opIdx, size_t outputIdx)
{
    OutputCacheUsageProfile profile {};

    initOutputAnalysis(opIdx, outputIdx);
    setRealSliceConsumersProperties(profile);
    setPerforation(profile);
    setSize(profile);

    return profile;
}

void CacheRequirementProfiler::initInputAnalysis(size_t nodeIdx, size_t inputIdx)
{
    initAnalysis(nodeIdx, inputIdx);
    initSlices(m_node->getInputs());
}

void CacheRequirementProfiler::initOutputAnalysis(size_t nodeIdx, size_t outputIdx)
{
    initAnalysis(nodeIdx, outputIdx);
    initSlices(m_node->getOutputs());
}

void CacheRequirementProfiler::initAnalysis(size_t nodeIdx, size_t operandIdx)
{
    m_nodeIdx    = nodeIdx;
    m_operandIdx = operandIdx;
    m_node       = sliceNode(nodeIdx);

    HB_ASSERT(!m_node->isLogicalOperation(), "Cache profile for logical operations is not supported (nor make sense)");
}

void CacheRequirementProfiler::initSlices(const TensorVector& operands)
{
    m_slice     = operands.at(m_operandIdx);
    m_realSlice = properties(m_slice).realSlice ? properties(m_slice).realSlice : m_slice;
}

void CacheRequirementProfiler::setProducerProperties(InputCacheUsageProfile& profile) const
{
    if (properties(m_slice).producingStep)
    {
        profile.produced = true;

        const NodePtr& producer = sliceNode(*properties(m_slice).producingStep);
        profile.localized       = isTensorPerforated() && samePerforation(producer, m_node);
    }
}

bool CacheRequirementProfiler::samePerforation(const NodePtr& producer, const NodePtr& consumer) const
{
    // The currently profiled operation can't be logical. This means that producer and consumer can't both be logical
    // operations. When the profiled operation is the consumer, it's possible that the producer would be logical. When
    // the profiled operation is the producer, we search for the first physical consumer, so that can't be logical.
    HB_ASSERT(!consumer->isLogicalOperation(),
              "Unexpected perforation match with logical consumer ({} [{}])",
              consumer->getNodeName(),
              consumer->getNodeTypeStr());

    if (producer->isLogicalOperation())
    {
        return samePerforationLogicalProducer(producer, consumer);
    }

    return samePerforationPhysicalNodes(producer, consumer);
}

bool CacheRequirementProfiler::samePerforationLogicalProducer(const NodePtr& producer, const NodePtr& consumer) const
{
    bool same = true;
    for (const TensorPtr& in : producer->getInputs())
    {
        if (!in || in->isShapeTensor()) continue;
        auto producerStep = properties(in).producingStep;
        if (!producerStep)
        {
            LOG_DEBUG(LB_CACHE_MNGR,
                      "Matching perforation with a logical producer ({}), which has an input without a producer in "
                      "the bundle: {}. Locality is impossible.",
                      producer->getNodeName(),
                      in->getName());
            return false;
        }
        same &= samePerforation(sliceNode(*producerStep), consumer);  // recursive call
    }
    // Log(s) will be printed in the recursive calls.
    return same;
}

bool CacheRequirementProfiler::samePerforationPhysicalNodes(const NodePtr& producer, const NodePtr& consumer) const
{
    if (!producer->getNodeAnnotation().perforationDim.has_value() ||
        !consumer->getNodeAnnotation().perforationDim.has_value())
    {
        LOG_DEBUG(LB_CACHE_MNGR,
                  "One or more of the operations isn't actually perforated: [{}, {}]",
                  producer->getNodeName(),
                  consumer->getNodeName());
        return false;
    }

    HB_ASSERT(producer->getNodeAnnotation().bundleInfo.is_set(),
              "producer missing bundle info {}",
              producer->getNodeName());
    const auto& producerPerforationGroup = producer->getNodeAnnotation().bundleInfo->perforationGroup;
    HB_ASSERT(consumer->getNodeAnnotation().bundleInfo.is_set(),
              "consumer missing bundle info {}",
              consumer->getNodeName());
    const auto& consumerPerforationGroup = consumer->getNodeAnnotation().bundleInfo->perforationGroup;
    if (!producerPerforationGroup.has_value() || !consumerPerforationGroup.has_value())
    {
        LOG_DEBUG(LB_CACHE_MNGR,
                  "One or more of the operations doesn't have perforation group: [{}, {}]",
                  producer->getNodeName(),
                  consumer->getNodeName());
        return false;
    }

    // Perfortion group represents nodes which were perforated on the same BVD
    bool samePerforationBvd = (producerPerforationGroup == consumerPerforationGroup);

    LOG_TRACE(LB_CACHE_MNGR,
              "{} and {} are {}perforated the same.",
              producer->getNodeName(),
              consumer->getNodeName(),
              samePerforationBvd ? "" : "not ");
    return samePerforationBvd;
}

void CacheRequirementProfiler::setConsumersProfile(InputCacheUsageProfile& profile) const
{
    const auto physicalConsumers = physicalConsumerSteps();

    profile.lastConsumer =
        physicalConsumers.empty() || *std::max_element(physicalConsumers.begin(), physicalConsumers.end()) == m_nodeIdx;

    profile.nofConsumers = physicalConsumers.size();
}

std::vector<size_t> CacheRequirementProfiler::physicalConsumerSteps() const
{
    std::vector<size_t> steps;
    for (const auto& step : properties(m_realSlice).consumingSteps)
    {
        if (!sliceNode(step)->isLogicalOperation()) steps.push_back(step);
    }
    return steps;
}

void CacheRequirementProfiler::setBPTAccess(InputCacheUsageProfile& profile) const
{
    profile.bpt = properties(m_realSlice).joinedBy != nullptr || !properties(m_realSlice).producingStep;
}

void CacheRequirementProfiler::setAllRequired(InputCacheUsageProfile& profile) const
{
    // TODO [SW-147166] - improve this condition. Needs more sophisticated testing (for example, compare the
    // granularity with the shape of the big tensor - requires to always have a big tensor to any slice in the
    // test).

    // If the tensor is allRequired, its overlap will have zero offset. This is not the only condition, but it's
    // simple and should be enough for most cases.
    auto ap = m_node->getNodeAccessPattern();
    HB_ASSERT(ap,
              "No access pattern for node {} (guid: {}, type: {})",
              m_node->getNodeName(),
              m_node->getGUID(),
              m_node->getNodeTypeStr());

    auto ol             = ap->getTensorOverlap(m_slice);
    auto isZero         = [](auto val) { return val == 0; };
    profile.allRequired = std::all_of(ol.offset.begin(), ol.offset.end(), isZero);
}

void CacheRequirementProfiler::setNumReads(InputCacheUsageProfile& profile) const
{
    if (HabanaGraph::runsOnMME(m_node))
    {
        HB_ASSERT_PTR(m_strategy);
        const auto& qor = m_strategy->getNodeQORs(bigNode(m_node));

        bool        isDedw      = m_node->getNodeType() == Node::TYPE_DEDW;
        const auto& memAttrIdx0 = isDedw ? qor->perfAttr.memoryAttrB : qor->perfAttr.memoryAttrA;
        const auto& memAttrIdx1 = isDedw ? qor->perfAttr.memoryAttrA : qor->perfAttr.memoryAttrB;

        switch (m_operandIdx)
        {
            case 0:
                profile.totalReads = memAttrIdx0.accessesPerChip;
                profile.dcoreReads = memAttrIdx0.accessesPerDcore;
                break;
            case 1:
                profile.totalReads = memAttrIdx1.accessesPerChip;
                profile.dcoreReads = memAttrIdx1.accessesPerDcore;
                break;
            case TENSOR_AUX_CD_SCRATCHPAD:
            case TENSOR_AUX_CD_REDUCTION:  // TODO [SW-175795] : separate memoryAttrAux for the different aux tensors
                profile.totalReads = qor->perfAttr.memoryAttrAux.accessesPerChip;
                profile.dcoreReads = qor->perfAttr.memoryAttrAux.accessesPerDcore;
                break;
            default:
                HB_ASSERT(0, "Unsupported input index: {} for MME node {}", m_operandIdx, m_node->getNodeName());
        }
    }
    else if (HabanaGraph::runsOnTPC(m_node))
    {
        // TODO [SW-147168] - totalReads and dcoreReads
        if (profile.allRequired)
        {
            const auto numTPCs   = m_graph.getHALReader()->getNumTpcEngines();
            const auto numDcores = m_graph.getHALReader()->getNumDcores();
            HB_ASSERT(numDcores > 0, "Invalid num dcores {}", numDcores);
            profile.totalReads   = numTPCs;
            profile.dcoreReads   = numTPCs / numDcores;
        }
    }
}

void CacheRequirementProfiler::setRealSliceConsumersProperties(OutputCacheUsageProfile& profile) const
{
    profile.hasConsumers = bool(findFirstPhysicalConsumerIdx());
    setRMWAccess(profile);
}

void CacheRequirementProfiler::setRMWAccess(OutputCacheUsageProfile& profile) const
{
    auto reductionIdx = findReductionConsumerIdx();
    if (reductionIdx)
    {
        const NodePtr& reduction = sliceNode(*reductionIdx);
        if (isRMWReduction(dynamic_cast<ReductionNode*>(reduction.get())))
        {
            profile.rmw           = true;
            profile.lastRmwWriter = (findLastReductionProducerIdx(reduction) == m_nodeIdx);
        }
    }
}

std::optional<size_t> CacheRequirementProfiler::findReductionConsumerIdx() const
{
    for (int step : properties(m_realSlice).consumingSteps)
    {
        if (sliceNode(step)->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            return step;
        }
    };
    return std::nullopt;
}

bool CacheRequirementProfiler::isRMWReduction(const ReductionNode* reductionNode) const
{
    HB_ASSERT_PTR(reductionNode);
    // REDUCTION_UNORDERED_SET may be used for partials writes, so is considered as single DCore reduction
    return (reductionNode->getReductionOperation() != REDUCTION_SET);
}

size_t CacheRequirementProfiler::findLastReductionProducerIdx(const NodePtr& reductionNode) const
{
    size_t maxReductionProducerIdx = m_nodeIdx;
    for (const TensorPtr& input : reductionNode->getInputs())
    {
        HB_ASSERT(properties(input).producingStep,
                  "Can't have reduction node without a producer to one of its inputs (Reduction: {}, input: {}).",
                  reductionNode->getNodeName(),
                  input->getName());
        maxReductionProducerIdx = std::max(maxReductionProducerIdx, size_t(*properties(input).producingStep));
    }
    return maxReductionProducerIdx;
}

void CacheRequirementProfiler::setPerforation(OutputCacheUsageProfile& profile) const
{
    auto firstConsumerIdx = findFirstPhysicalConsumerIdx();
    if (firstConsumerIdx)
    {
        const NodePtr& consumer                = sliceNode(*firstConsumerIdx);
        profile.localized                      = isTensorPerforated() && samePerforation(m_node, consumer);
    }
}

std::optional<size_t> CacheRequirementProfiler::findFirstPhysicalConsumerIdx() const
{
    const auto& consumerIndices = properties(m_realSlice).consumingSteps;
    if (consumerIndices.empty()) return std::nullopt;

    auto compareConsumers = [&](int lhs, int rhs) {
        if (sliceNode(lhs)->isLogicalOperation() ^ sliceNode(rhs)->isLogicalOperation())
        {
            // one of them is not logical - it should be considered the minimal
            // rhs is logical => lhs is not => "lhs < rhs" so should return true.
            // rhs is not logical => lhs is => "lhs > rhs" so should return false.
            return sliceNode(rhs)->isLogicalOperation();
        }
        // either both logical or both physical => compare schedule
        return lhs < rhs;
    };
    auto idx = *std::min_element(consumerIndices.begin(), consumerIndices.end(), compareConsumers);

    if (sliceNode(idx)->isLogicalOperation()) return std::nullopt;  // no physical consumers

    return idx;
}

bool CacheRequirementProfiler::isTensorPerforated() const
{
    const auto& perforationDim = m_node->getNodeAnnotation().perforationDim;
    if (!perforationDim.has_value())
    {
        return false;
    }

    const auto& nodeAP = m_node->getNodeAccessPattern();
    HB_ASSERT_PTR(nodeAP);
    for (auto tensorDim = 0; tensorDim < m_slice->getDim(); tensorDim++)
    {
        auto nodeDim = nodeAP->getIndexSpaceDim(m_slice, tensorDim);
        if (nodeDim == perforationDim.value())
        {
            return true;
        }
    }
    return false;
}

void InputCacheUsageProfile::log() const
{
    LOG_TRACE(LB_CACHE_MNGR,
              "InputCacheUsageProfile"
              ": totalReads = {}"
              ", dcoreReads = {}"
              ", size = {}B"
              ", nofConsumers = {}"
              ", produced = {}"
              ", localized = {}"
              ", lastConsumer = {}"
              ", allRequired = {}"
              ", bpt = {}",
              totalReads,
              dcoreReads,
              size,
              nofConsumers,
              produced,
              localized,
              lastConsumer,
              allRequired,
              bpt);
}

void OutputCacheUsageProfile::log() const
{
    LOG_TRACE(LB_CACHE_MNGR,
              "OutputCacheUsageProfile: "
              ", size = {}B"
              ", rmw = {}"
              ", lastRmwWriter = {}"
              ", hasConsumers = {}"
              ", localized = {}",
              size,
              rmw,
              lastRmwWriter,
              hasConsumers,
              localized);
}
