
#include "reduction_handler.h"
#include "habana_graph.h"
#include "types.h"
#include "node_factory.h"
#include "cast_nodes_handler.h"

using namespace gc::layered_brain;

void ReductionHandler::addProducerForTensorSlice(const TensorPtr& slice,
                                                 const NodePtr&   sliceProducer,
                                                 const NodePtr&   origProducer,
                                                 unsigned         producerOutputIdx)
{
    SlicedTensorProducers& producers = m_slicedTensorProducers[slice];
    if (producers.sliceProducers.empty())
    {
        // First producer for this slice
        producers.origProducer = origProducer;
        producers.outputIdx    = producerOutputIdx;
    }
    else
    {
        HB_ASSERT(producers.origProducer == origProducer, "Expected the same original producer for all slices");
        HB_ASSERT(producers.outputIdx == producerOutputIdx, "Expected the same reduction output idx for all slices");
    }
    producers.sliceProducers.push_back(sliceProducer);
}

bool ReductionHandler::requiresReduction(const SlicedTensorProducers& producers) const
{
    return (producers.sliceProducers.size() > 1) || requiresMemset(producers.origProducer);
}

// For TPC nodes - reduction operation is UNORDERED_SET.
// For all other cases - reduction operation is ADD.
unsigned ReductionHandler::getReductionOp(const NodePtr& origProducer) const
{
    if (origProducer && HabanaGraph::runsOnTPC(origProducer))
    {
        return ReductionOperation::REDUCTION_UNORDERED_SET;
    }
    return ReductionOperation::REDUCTION_ADD;
}

NodePtr ReductionHandler::handleReductionDataType(const NodePtr& reductionNode) const
{
    TensorPtr origReductionOutput = reductionNode->getOutput(0);
    for (const auto& tensor : reductionNode->getInputs())
    {
        HB_ASSERT_PTR(tensor);
        tensor->setElementType(HIGH_PRECISION_DATA_TYPE_FOR_REDUCTION);
    }
    TensorPtr reductionOutputBeforeCast = origReductionOutput->clone(false, false, false);
    reductionOutputBeforeCast->setName(fmt::format("reduction_for_{}_before_cast", origReductionOutput->getName()),
                                       true);
    reductionOutputBeforeCast->setElementType(HIGH_PRECISION_DATA_TYPE_FOR_REDUCTION);
    reductionNode->replaceOutput(0, reductionOutputBeforeCast);
    NodePtr castNode = CastNodeHandler::createCastNode(
        reductionOutputBeforeCast,
        origReductionOutput,
        fmt::format("{}cast_for_{}", getNodeNamePrefix(), reductionNode->getNodeName()));
    HB_ASSERT_PTR(castNode);
    if (m_bundleIdx.has_value())
    {
        castNode->getNodeAnnotation().bundleInfo.set(BundleInfo(*m_bundleIdx, BundleType::UNDEFINED, 0));
    }
    LOG_DEBUG(LB_SLICER,
              "Create a cast node {} ({} -> {}) after reduction node {}",
              castNode->getNodeName(),
              getStringFromSynDataType(castNode->getInput(0)->getElementType()),
              getStringFromSynDataType(castNode->getOutput(0)->getElementType()),
              reductionNode->getNodeName());

    return castNode;
}

bool ReductionHandler::requiresCastForReduction(const TensorPtr& tensor, const NodePtr& origProducer) const
{
    return (m_requireCast.find(origProducer) != m_requireCast.end()) &&
           (tensor->getElementType() != HIGH_PRECISION_DATA_TYPE_FOR_REDUCTION);
}

bool ReductionHandler::requiresMemset(const NodePtr& origProducer) const
{
    return (m_requireMemset.find(origProducer) != m_requireMemset.end());
}

NodePtr ReductionHandler::addMemsetForReduction(const NodePtr& reduction) const
{
    auto memsetTensor = reduction->getOutput(0)->clone(false, false, false);
    memsetTensor->setName(reduction->getOutput(0)->getName() + "_zeros");
    auto memsetNode = NodeFactory::createNode(
        {},
        {memsetTensor},
        nullptr,
        NodeFactory::memsetNodeTypeName,
        fmt::format("{}memset_for_{}", getNodeNamePrefix(), reduction->getOutput(0)->getName()));
    HB_ASSERT_PTR(memsetNode);
    if (m_bundleIdx.has_value())
    {
        memsetNode->getNodeAnnotation().bundleInfo.set(BundleInfo(*m_bundleIdx, BundleType::UNDEFINED, 0));
    }
    reduction->addInput(reduction->getNumInputs(), memsetNode->getOutput(0));
    LOG_DEBUG(LB_SLICER,
              "Create a memset node {} for reduction {}",
              memsetNode->getNodeName(),
              reduction->getNodeName());
    return memsetNode;
}

std::string ReductionHandler::getNodeNamePrefix() const
{
    if (m_bundleIdx.has_value())
    {
        return fmt::format("lb_bundle_{}/", m_bundleIdx.value());
    }
    return "";
}

NodeVector ReductionHandler::createReductionNodes() const
{
    NodeVector newNodes;
    for (const auto& [slicedTensor, producers] : m_slicedTensorProducers)
    {
        if (requiresReduction(producers))
        {
            TensorVector reductionInputs;
            for (const auto& producer : producers.sliceProducers)
            {
                TensorPtr reductionInput = slicedTensor->clone(false, false, false);
                reductionInput->setName(
                    fmt::format("reduction_{}_for_{}", producers.outputIdx, producer->getNodeName()),
                    true);
                reductionInputs.emplace_back(reductionInput);
                producer->replaceOutput(producers.outputIdx, reductionInput);
            }
            unsigned reductionOp = getReductionOp(producers.origProducer);
            NodePtr  reductionNode = NodeFactory::createNode(
                reductionInputs,
                {slicedTensor},
                &reductionOp,
                NodeFactory::reductionNodeTypeName,
                fmt::format("{}reduction_for_tensor_{}", getNodeNamePrefix(), slicedTensor->getName()));
            if (m_bundleIdx.has_value())
            {
                const auto bundleInfo = BundleInfo(*m_bundleIdx, BundleType::UNDEFINED, 0);
                reductionNode->getNodeAnnotation().bundleInfo.set(bundleInfo);
            }
            LOG_DEBUG(LB_SLICER,
                      "Create a reduction node {} for tensor {}, number of producers {}, reduction operation {}",
                      reductionNode->getNodeName(),
                      slicedTensor->getName(),
                      producers.sliceProducers.size(),
                      reductionOp);
            newNodes.emplace_back(reductionNode);
            if (requiresCastForReduction(slicedTensor, producers.origProducer))
            {
                NodePtr castNode = handleReductionDataType(reductionNode);
                newNodes.emplace_back(castNode);
            }
            if (requiresMemset(producers.origProducer))
            {
                NodePtr memsetNode = addMemsetForReduction(reductionNode);
                newNodes.emplace_back(memsetNode);
            }
        }
    }
    return newNodes;
}