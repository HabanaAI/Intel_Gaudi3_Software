#include "batch_norm_eviction_fuser.h"

#include "graph_editor.h"
#include "bundle.h"
#include "bundle_slicer.h"
#include "slicing_strategy.h"
#include "sram_management/slice_mapping.h"
#include "tpc_kernel_loader.h"
#include "tpc_kernel_names.h"
#include "types.h"
#include <algorithm>
#include <iterator>

BatchNormStagesEvictionFuser::BatchNormStagesEvictionFuser(HabanaGraph&              graph,
                                                           const pBundle&            bundle,
                                                           const SlicingStrategyPtr& strategy)
: m_graph {graph}, m_bundleNodes {bundle->getNodes().begin(), bundle->getNodes().end()}, m_strategy {strategy}
{
}

template<typename CONTAINER>
static inline bool inGuidList(const std::string& guid, const CONTAINER& guidList)
{
    return std::find(guidList.begin(), guidList.end(), guid) != guidList.end();
}

void BatchNormStagesEvictionFuser::fuseEvictions(bool stage1Fwd, bool stage2Fwd, bool stage1bwd)
{
    static const auto fwdStage1Guids = getBN1Guids(Direction::FWD);
    static const auto fwdStage2Guids = getBN2Guids(Direction::FWD);
    static const auto bwdStage1Guids = getBN1Guids(Direction::BWD);

    for (auto& node : m_bundleNodes)
    {
        if (inGuidList(node->getGUID(), fwdStage1Guids) && stage1Fwd)
        {
            checkAndFuseBN1FwdEviction(node);
        }
        else if (inGuidList(node->getGUID(), fwdStage2Guids) && stage2Fwd)
        {
            checkAndFuseBN2FwdEviction(node);
        }
    }
}

// Does tensor require eviction from the bundle
bool BatchNormStagesEvictionFuser::requiresEviction(const TensorPtr& tensor) const
{
    if (findBundledTensorConsumers(tensor).empty())
    {
        return false;
    }
    return BundleSlicer::shouldTensorBeEvicted(tensor, m_graph, m_bundleNodes);
}

NodeVector BatchNormStagesEvictionFuser::findBundledTensorConsumers(const TensorPtr& tensor) const
{
    NodeVector bundledConsumers;
    for (const auto& consumer : m_graph.getTensorConsumers(tensor))
    {
        if (m_bundleNodes.find(consumer) != m_bundleNodes.end())
        {
            bundledConsumers.push_back(consumer);
        }
    }
    return bundledConsumers;
}

void BatchNormStagesEvictionFuser::reInstantiateTpcNode(NodePtr& node) const
{
    TpcKernelLoader kernelLoader(&m_graph);
    auto            res = kernelLoader.load(node);
    HB_ASSERT(res,
              "Could not re-instantiate node {} <GUID: {}> with additional output",
              node->getNodeName(),
              node->getGUID());
}

// Replace the evicted and the new tensor in the strategy sliced operands.
// Returns the new sliced operand of the evicted tensor that needs to be remapped in the strategy.
pSlicedOperand BatchNormStagesEvictionFuser::generateEvictedSlicedOperand(pSlicedOperand   orig,
                                                                          const TensorPtr& newIntermediateTensor)
{
    pSlicedOperand slicedEvictedTensor = std::make_shared<SlicedOperand>(*orig);
    slicedEvictedTensor->resideInSRAM  = false;
    // Avoid aligning the evicted tensor in HBM
    slicedEvictedTensor->alignWithCacheLine = false;
    m_strategy->getSlicingData().bundleTensors.push_back(slicedEvictedTensor);
    orig->originalTensor = newIntermediateTensor;
    return slicedEvictedTensor;
}

void BatchNormStagesEvictionFuser::checkAndFuseBN1FwdEviction(NodePtr bn1Fwd)
{
    auto bnIfm = bn1Fwd->getInput(BN1_FWD_IFM_INPUT_IDX);
    if (requiresEviction(bnIfm))
    {
        if (bn1Fwd->getNumOutputs() < 2)
        {
            // Add eviction to the graph
            auto bnIfmCopy = fuseBN1FwdEviction(bn1Fwd, bnIfm);
            // bnIfm was an intermediate tensor in the bundle. bnIfmCopy replaced it as an intermediate. bnIfm is now an
            // output from the batch_norm node and is no longer an intermediate tensor in the bundle.
            // => need to update the strategy.
            replaceBN1IfmInStrategy(bn1Fwd, bnIfm, bnIfmCopy);
        }
        else
        {
            SLC_WARN("{} ({}) input needs eviction, but optional output is already used.",
                     bn1Fwd->getNodeName(),
                     bn1Fwd->getGUID());
        }
    }
}

// Performs the fusion in the graph and returns the new intermediate tensor that is used in the bundle.
TensorPtr BatchNormStagesEvictionFuser::fuseBN1FwdEviction(NodePtr& bn1Fwd, const TensorPtr& bnIfm)
{
    // Transform this subgraph: (bnIfmProducer) -> [BN_IFM] -> (BN1_fwd) -> [sigmas]
    // To this graph: (bnIfmProducer) -> [BN_IFM WS copy] -> (BN1_fwd) -> [Sigmas]
    //                                                           |
    //                                                           +------> [BN_IFM]
    LOG_DEBUG(GC, "Fusing bn1fwd eviction for {}", bn1Fwd->getNodeName());
    auto bnIfmProducer  = m_graph.getTensorProducer(bnIfm);
    auto bnIfmOutputIdx = bnIfmProducer->getOutputIndexOfTensor(bnIfm);
    auto bnIfmCopy      = bnIfm->clone(false, false, false);

    // In case the original BN was instantiated already, need to reset it before adding additional operands
    // as we do not wish to update the access pattern cache due to node re-addition.
    TPCNode* bn1FwdTPCNode   = static_cast<TPCNode*>(bn1Fwd.get());
    bool     wasInstantiated = bn1FwdTPCNode->isInstantiated();
    bn1FwdTPCNode->resetInstantiated();

    bnIfmCopy->setName(bnIfm->getName() + "_bundled_intermediate_copy");
    GraphEditor::replaceOutput(m_graph, bnIfmProducer, bnIfmOutputIdx, bnIfmCopy);
    GraphEditor::replaceInput(m_graph, bn1Fwd, BN1_FWD_IFM_INPUT_IDX, bnIfmCopy);
    GraphEditor::editNode(m_graph, bn1Fwd, [&]() { bn1Fwd->emplaceOutput(BN1_FWD_IFM_COPY_OUTPUT_IDX, bnIfm); });

    // After adding an additional output to the BN node, it needs to be re-instantiated.
    if (wasInstantiated)
    {
        reInstantiateTpcNode(bn1Fwd);
    }

    return bnIfmCopy;
}

// Create a new sliced operand for the original BN IFM and swap the original sliced operand's tensor to the copy.
// Update the mapping to point to the new output as well.
void BatchNormStagesEvictionFuser::replaceBN1IfmInStrategy(const NodePtr&   bn1Fwd,
                                                           const TensorPtr& bnIfm,
                                                           const TensorPtr& bnIfmCopy)
{
    pSlicedOperand origSlicedIfm = m_strategy->getSlicingData().getSlicedOperand(bnIfm);
    HB_ASSERT(origSlicedIfm != nullptr, "{} was not found in strategy's sliced operands.", bnIfm->getName());

    pSlicedOperand newSlicedIfm = generateEvictedSlicedOperand(origSlicedIfm, bnIfmCopy);

    SlicedOperandList bnInputs;
    SlicedOperandList bnOutputs;
    std::tie(bnInputs, bnOutputs) = getBN1SlicedOperands(origSlicedIfm);
    bnOutputs.push_back(newSlicedIfm);

    if (GCFG_ENABLE_PIPELINE_MANAGEMENT.value() || !GCFG_IGNORE_INDEX_SPACE_FOR_SLICING.value())
    {
        m_strategy->getSlicingData().setOperandSliceForwardMapping(
            origSlicedIfm,
            AccessPatternSliceMapper::createFwdMapping(bn1Fwd, bnInputs, bnOutputs));
    }
    else
    {
        m_strategy->getSlicingData().setOperandSliceForwardMapping(
            origSlicedIfm,
            TrivialSliceMapper::mapSlicedOperandForward(bnInputs, bnOutputs));
    }
}

// Find the sliced inputs and outputs of the BN node from the strategy sliced operands for mapping.
BatchNormStagesEvictionFuser::InOutSlicedOpLists
BatchNormStagesEvictionFuser::getBN1SlicedOperands(const pSlicedOperand& bn1SlicedIfm) const
{
    // Use the existing mapping to find the inupt and output slice references, to extract the slicedOperands
    auto pair = m_strategy->getSlicingData().getFwdMappedSlicedOperands(bn1SlicedIfm);

    HB_ASSERT(!pair.first.empty() && !pair.second.empty(),
              "Expected a pair of non-empty list of operand in fwd mapping of {}",
              bn1SlicedIfm->originalTensor->getName());

    return pair;
}
BatchNormStagesEvictionFuser::InOutSlicedOpLists
BatchNormStagesEvictionFuser::getBN2SlicedOperands(const pSlicedOperand& bn2SlicedOfm) const
{
    // Use the existing mapping to find the inupt and output slice references, to extract the slicedOperands
    auto pair = m_strategy->getSlicingData().getBwdMappedSlicedOperands(bn2SlicedOfm);

    HB_ASSERT(!pair.first.empty() && !pair.second.empty(),
              "Expected a pair of non-empty list of operand in fwd mapping of {}",
              bn2SlicedOfm->originalTensor->getName());

    return pair;
}

void BatchNormStagesEvictionFuser::checkAndFuseBN2FwdEviction(NodePtr bn2Fwd)
{
    const auto bnOfm = bn2Fwd->getOutput(BN2_FWD_OFM_IDX);
    if (requiresEviction(bnOfm))
    {
        if (bn2Fwd->getNumOutputs() >= 4)
        {
            SLC_WARN("{} ({}) output needs eviction, but optional output is already used.",
                     bn2Fwd->getNodeName(),
                     bn2Fwd->getGUID());
            return;
        }

        auto bnOfmCopy = fuseBN2FwdEviction(bn2Fwd, bnOfm);  // Graph modifications
        // bnOfm was an intermediate tensor in the bundle. bnOfmCopy replaced it as an intermediate. bnOfm is now the
        // 2nd output from the batch_norm node and is no longer an intermediate tensor in the bundle.
        // => need to update the strategy.
        replaceBN2OfmInStrategy(bn2Fwd, bnOfm, bnOfmCopy);
    }
}

// Performs the fusion in the graph and returns the new intermediate tensor that is used in the bundle.
TensorPtr BatchNormStagesEvictionFuser::fuseBN2FwdEviction(NodePtr& bn2Fwd, const TensorPtr& bnOfm)
{
    // Transform this subgraph: (BN2_Fwd) -> [bnOfm] -> (bundledOfmConsumer)
    // To this subgraph: (BN2_Fwd)-> [bnOfm WS Copy] -> (bundledOfmConsumer)
    //                       |
    //                       +-----> [bnOfm]
    //
    // In case there are multiple bundled consumers to the tensor, they would all read the new copy (not drawn for
    // simplicity).

    // 0. Create BnOfm copy and rename appropriately.
    LOG_DEBUG(GC, "Fusing bn2fwd eviction for {}", bn2Fwd->getNodeName());
    auto bnOfmCopy = bnOfm->clone(false, false, false);
    bnOfmCopy->setName(bnOfm->getName() + "_bundled_intermediate_copy");

    // 1. Connect bnOfmCopy to consumers of bnOfm and disconnect bnOfm from them.
    // The tensor may have multiple bundled consumers. Need to replace the the input in all of them.
    const auto& bundledConsumers = findBundledTensorConsumers(bnOfm);
    for (const auto& bundledOfmConsumer : bundledConsumers)
    {
        auto bnOfmInputIndex = bundledOfmConsumer->getInputIndexOfTensor(bnOfm);
        GraphEditor::replaceInput(m_graph, bundledOfmConsumer, bnOfmInputIndex, bnOfmCopy);
    }

    // In case the original BN was instantiated already, need to reset it before adding additional operands
    // as we do not wish to update the access pattern cache due to node re-addition.
    TPCNode* bn2FwdTPCNode   = static_cast<TPCNode*>(bn2Fwd.get());
    bool     wasInstantiated = bn2FwdTPCNode->isInstantiated();
    bn2FwdTPCNode->resetInstantiated();

    // 2.
    GraphEditor::editNode(m_graph, bn2Fwd, [&]() { bn2Fwd->emplaceOutput(0, bnOfmCopy); });
    pSlicedOperand origSlicedOfm = m_strategy->getSlicingData().getSlicedOperand(bnOfm);
    // Adding an optional output to the BN node means it needs re-instantiation
    if (wasInstantiated)
    {
        reInstantiateTpcNode(bn2Fwd);
    }
    return bnOfmCopy;
}

// Create a new sliced operand for the original BN OFM and swap the original sliced operand's tensor to the copy.
// No need to fix mappings, since the non-primary TPC outputs are not mapped to when using bwd mapping.
void BatchNormStagesEvictionFuser::replaceBN2OfmInStrategy(const NodePtr&   bn2Fwd,
                                                           const TensorPtr& bnOfm,
                                                           const TensorPtr& bnOfmCopy)
{
    pSlicedOperand origSlicedOfm = m_strategy->getSlicingData().getSlicedOperand(bnOfm);
    HB_ASSERT(origSlicedOfm != nullptr, "{} was not found in strategy's sliced operands.", bnOfm->getName());

    pSlicedOperand    newSlicedOfm = generateEvictedSlicedOperand(origSlicedOfm, bnOfmCopy);
    SlicedOperandList bnInputs;
    SlicedOperandList bnOutputs;
    std::tie(bnInputs, bnOutputs) = getBN2SlicedOperands(origSlicedOfm);
    std::vector<pSlicedOperand> bnInputsVec(bnInputs.begin(), bnInputs.end());
    bnOutputs.push_back(newSlicedOfm);
    std::vector<pSlicedOperand> bnOutputsVec(bnOutputs.begin(), bnOutputs.end());

    if (GCFG_ENABLE_PIPELINE_MANAGEMENT.value() || !GCFG_IGNORE_INDEX_SPACE_FOR_SLICING.value())
    {
        m_strategy->getSlicingData().setOperandSliceBackwardMapping(
            origSlicedOfm,
            AccessPatternSliceMapper::createBwdMapping(bn2Fwd, bnInputsVec, bnOutputsVec));
    }
    else
    {
        m_strategy->getSlicingData().setOperandSliceBackwardMapping(
            origSlicedOfm,
            TrivialSliceMapper::mapOutputToInputs(bn2Fwd, bnInputs, newSlicedOfm));
    }
}
