#include "fusion_candidates_collector.h"

void TpcRecompileDb::registerTpcFusion(const RecompileCacheKey& key, const RecompileCacheValue& value)
{
    auto guidInMap = m_recompileCache.find(key);
    if (guidInMap == m_recompileCache.end())
    {
        m_recompileCache.insert({key, value});
    }
}

std::optional<TpcRecompileDb::RecompileCacheValue> TpcRecompileDb::getFusionFromDb(const RecompileCacheKey& key)
{
    auto guidInMap = m_recompileCache.find(key);
    if (guidInMap != m_recompileCache.end())
    {
        return guidInMap->second;
    }
    return {};
}

unsigned getLlvmTensorIdx(const NodePtr& node, const TensorPtr& tensor)
{
    unsigned ret = 0;
    for (const auto& op : node->getOperands())
    {
        if (op == nullptr || op->isTensorAuxOrShapeOutput()) continue;
        if (op == tensor) return ret;
        ret++;
    }
    return ret;
}

CandidateInfo::CandidateInfo(const NodePtr& node, const TensorPtr& connectingTensor, bool isInput)
: m_candidate(node), m_connectingTensor(connectingTensor), m_isInput(isInput)
{
    m_gcTensorIdx =
        isInput ? node->getInputIndexOfTensor(connectingTensor) : node->getOutputIndexOfTensor(connectingTensor);
    m_llvmTensorIdx = getLlvmTensorIdx(node, connectingTensor);
}

std::optional<CandidateInfo> FusionCandidatesCollector::getProducerCandidate(const HabanaGraph& g,
                                                                             const TensorPtr&   tensor)
{
    if (auto producer = g.getTensorProducer(tensor))
    {
        return CandidateInfo(producer, tensor, false);
    }
    return {};
}

CandidatesInfoVector FusionCandidatesCollector::getConsumerCandidates(const HabanaGraph& g, const TensorPtr& tensor, const NodePtr& spill)
{
    CandidatesInfoVector consumerCandidates;
    const auto&          consumers = g.getTensorConsumers(tensor);
    for (const auto& consumer : consumers)
    {
        if (consumer == spill) continue;
        consumerCandidates.push_back(CandidateInfo(consumer, tensor, true));
    }
    return consumerCandidates;
}

CandidatesInfoVector FusionCandidatesCollector::getSpillFusionCandidates(const HabanaGraph& g, const NodePtr& spill)
{
    const TensorPtr& connectingTensor = spill->getInput(0);
    auto producerCandidate  = getProducerCandidate(g, connectingTensor);
    auto consumerCandidates = getConsumerCandidates(g, connectingTensor, spill);
    if (producerCandidate.has_value()) consumerCandidates.push_back(producerCandidate.value());
    return consumerCandidates;
}
