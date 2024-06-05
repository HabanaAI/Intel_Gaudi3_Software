#include "fusion_handlers.h"
#include "operation_slice.h"
#include <compilation_hal_reader.h>
#include "handle_memory_reuse.h"
#include "handle_logical_operations.h"

static std::string_view getRecompileDeviceName(synDeviceType deviceType)
{
    switch (deviceType)
    {
        case synDeviceGaudi:
            return "gaudi";
        case synDeviceGaudi2:
            return "gaudi2";
            break;
        default:
            LOG_ERR(SPILL_FILL, "{}: device type :{} is not supported for spill fill fusion", HLLOG_FUNC, deviceType);
            HB_ASSERT(0, "{}: device type :{} is not supported for spill fill fusion", __FUNCTION__, deviceType);
            return "";
    }
}

bool TpcFusionHandler::recompileKernel(const TPCNodePtr& node,
                                       unsigned          tensorIdxToDuplicate,
                                       void*&            resultElf,
                                       unsigned&         resultElfSize)
{
    auto instance = node->getInstance();
    // TODO: remove -tpc-enable-duplicate-loads once llvm enables it by default
    std::string options = fmt::format("-mllvm -tpc-enable-duplicate-loads -mllvm -duplicate-output={} -mllvm -dontAnalysis={} -march={}",
                                      tensorIdxToDuplicate,
                                      !GCFG_DUMP_TPC_COST_MODEL_DATA.value(),
                                      getRecompileDeviceName(CompilationHalReader::getHalReader()->getDeviceType()));

    TpcElfTools::TpcElfStatus result = TpcElfTools::RecompileKernel(instance.kernel.kernelElf,
                                                                    instance.kernel.elfSize,
                                                                    options.c_str(),
                                                                    resultElf,
                                                                    resultElfSize);
    if (result != TpcElfTools::TpcElfStatus::TPC_ELF_SUCCESS)
    {
        LOG_ERR(GC,
                "Failed to recompile kernel {}, attempted to perform: {}, error: {}",
                node->getNodeName(),
                options,
                result);
        return false;
    }
    LOG_INFO(GC,
             "recompileKernel finished successfully for kernel: {} , new size {}",
             node->getNodeName(),
             resultElfSize);
    return true;
}

bool TpcFusionHandler::isValidForFusion(HabanaGraph& g, const NodePtr& directive, const CandidateInfo& candidate)
{
    auto tpcNode = std::dynamic_pointer_cast<TPCNode>(candidate.getNode());
    if (!tpcNode) return false;

    if (candidate.getConnectingTensor()->isPartOfRMWSection())
    {
        LOG_DEBUG(SPILL_FILL,
                  "Candidate {} performs rmw on the connecting tensor with the directive {}, fusion is not allowed.",
                  tpcNode->getNodeName(),
                  directive->getNodeName());
        return false;
    }

    if (tpcNode->hasHighRankOperand())
    {
        LOG_DEBUG(SPILL_FILL,
                  "Candidate {} has a high rank operand, fusion is not allowed.",
                  tpcNode->getNodeName());
        return false;
    }

    // TODO [SW-109323] - Support fusion for dynamic shapes
    if (tpcNode->isDynamicShape())
    {
        LOG_DEBUG(SPILL_FILL,
                  "Candidate {} has an operand with dynamic shape, fusion is not allowed.",
                  tpcNode->getNodeName());
        return false;
    }

    if (tpcNode->getTotalNumDescriptors() >= MAX_TENSOR_NR)
    {
        LOG_DEBUG(SPILL_FILL,
                      "Candidate {} does not have enough tensor descriptors available, fusion is not allowed.",
                      tpcNode->getNodeName());
        return false;
    }

    auto accessPattern = candidate.isInput()
                             ? tpcNode->getInstance().inputTensorAccessPattern[candidate.getOrigTensorIdx()]
                             : tpcNode->getInstance().outputTensorAccessPattern[candidate.getOrigTensorIdx()];
    if (!accessPattern.fullyAccessedOnce)
    {
        LOG_DEBUG(
            SPILL_FILL,
            "Candidate {} only partially accesses the connecting tensor with the directive {}, fusion is not allowed.",
            tpcNode->getNodeName(),
            directive->getNodeName());
        return false;
    }

    // Validate that fusion doesn't create a cycle in the graph:
    // Make sure none of the spill’s consumers is an ancestor of the candidate
    auto directiveConsumers   = g.getNodeConsumers(directive);
    bool foundCycle = std::any_of(directiveConsumers.begin(), directiveConsumers.end(), [&](const NodePtr& consumer) {
        return g.getNumberOfPaths(consumer, tpcNode) != 0;
    });
    if (foundCycle)
    {
        LOG_DEBUG(SPILL_FILL,
                  "Fusion of candidate {} and directive {} will cause a cycle in the graph, fusion is not allowed.",
                  tpcNode->getNodeName(),
                  directive->getNodeName());
        return false;
    }

    // Block fusion in cases the directive's output's memory overlaps with any of the tpc's outputs. The method is to
    // find the potential real tensors of each of the output tensors (directive's and tpc's) and check their overlap.
    const auto& potentialRealDirectiveOutputs = LogicalOpsHandler::getPotentialRealTensors(g, directive->getOutput(0), INPUT_TO_OUTPUT);
    for (const auto& tpcOutput : tpcNode->getOutputs())
    {
        if (!tpcOutput) continue;
        const auto& potentialRealCandidateOutputs = LogicalOpsHandler::getPotentialRealTensors(g, tpcOutput, INPUT_TO_OUTPUT);
        for (const auto& potentialRealCandidateOutput : potentialRealCandidateOutputs)
        {
            bool isDirectiveOverlap =
                std::any_of(potentialRealDirectiveOutputs.begin(),
                            potentialRealDirectiveOutputs.end(),
                            [&](const TensorPtr& potentialRealDirectiveOutput) {
                                return MemoryReuseHandler::isDenseOverlap(potentialRealCandidateOutput,
                                                                          potentialRealDirectiveOutput);
                            });
            if (isDirectiveOverlap)
            {
                LOG_DEBUG(SPILL_FILL,
                          "Candidate {} has an output tensor that has memory overlap with the output of directive {}, "
                          "fusion is not allowed.",
                          tpcNode->getNodeName(),
                          directive->getNodeName());
                return false;
            }
        }
    }

    return true;
}

bool TpcFusionHandler::isValidLlvmTensorIdPostRecompile(const TPCNodePtr& tpcCandidate, unsigned tensorIdToDuplicate, void*& elf, unsigned elfSize) const
{
    auto     dupTensors          = TPCNode::getDuplicateTensorsFromElf(elf, elfSize);
    auto     origAndDupIds       = dupTensors.find(tensorIdToDuplicate);
    if (origAndDupIds == dupTensors.end())
    {
        HB_ASSERT(0, "Duplicated id {} is not found in duplicateTensors mapping from llvm", tensorIdToDuplicate);
    }
    unsigned expectedLlvmId = tpcCandidate->getNumInputs() + tpcCandidate->getNumOutputs();
    unsigned givenLlvmId    = origAndDupIds->second;
    if (givenLlvmId != expectedLlvmId)
    {
        LOG_WARN(SPILL_FILL,
                  "The expected id for duplication {} is different than the given id by llvm {} for node {}",
                  expectedLlvmId,
                  givenLlvmId,
                  tpcCandidate->getNodeName());
        return false;
    }
    return true;
}

bool TpcFusionHandler::handleTpcDoubleStore(const NodePtr& directive, const CandidateInfo& candidate)
{
    auto tpcCandidate = std::dynamic_pointer_cast<TPCNode>(candidate.getNode());

    HB_ASSERT_PTR(tpcCandidate);
    auto     cachedFusionOpt = m_tpcFusionDb.getFusionFromDb({candidate.getNode()->getGUID(), candidate.getTensorIdToDuplicate()});
    void*    resultElf       = nullptr;
    unsigned resultElfSize   = 0;
    if (cachedFusionOpt.has_value())
    {
        const auto& [fusionSucceeded, elf, elfSize] = cachedFusionOpt.value();
        if (!fusionSucceeded)
        {
            LOG_DEBUG(SPILL_FILL, "Spill fusion found in cache as failed fusion. Not performing spill fusion");
            return false;
        }
        resultElf     = elf;
        resultElfSize = elfSize;
        LOG_DEBUG(SPILL_FILL, "Spill fusion found in cache - performing spill fusion");
    }
    else
    {
        unsigned tensorIdToDuplicate = candidate.getTensorIdToDuplicate();
        if (!recompileKernel(tpcCandidate, tensorIdToDuplicate, resultElf, resultElfSize) ||
            !isValidLlvmTensorIdPostRecompile(tpcCandidate, tensorIdToDuplicate, resultElf, resultElfSize))
        {
            LOG_DEBUG(SPILL_FILL, "Double store fusion failed");
            m_tpcFusionDb.registerTpcFusion({m_candidate.getNode()->getGUID(), m_candidate.getTensorIdToDuplicate()},
                                            {false, resultElf, resultElfSize});
            return false;
        }
    }
    m_tpcFusionDb.registerTpcFusion({m_candidate.getNode()->getGUID(), m_candidate.getTensorIdToDuplicate()},
                                    {true, resultElf, resultElfSize});
    tpcCandidate->setDoubleStore(resultElf, resultElfSize, candidate.getOrigTensorIdx(), candidate.isInput());
    return true;
}

bool TpcFusionHandler::fuse(const NodePtr& directive)
{
    LOG_INFO(SPILL_FILL,
             "Attempting to fuse directive {} to node {}",
             directive->getNodeName(),
             m_candidate.getNode()->getNodeName());
    if (!handleTpcDoubleStore(directive, m_candidate))
    {
        return false;
    }
    // Remove sfd node from graph
    GraphEditor::removeNode(m_graph, directive);

    // Mark the tensor as double store output
    directive->getOutput(0)->getTensorAnnotation().isDoubleStoreTensor = true;

    // Add new output to the desired candidate
    GraphEditor::editNode(m_graph, m_candidate.getNode(), [&]() { m_candidate.getNode()->addOutput(directive->getOutput(0)); });

    // Add tensor offset (which will be used later for tpc slice roi generation)
    if (auto operationSlicePtr = std::dynamic_pointer_cast<OperationSlice>(m_candidate.getNode()))
    {
        TensorPtr doubleStore =
            m_candidate.getNode()->getOutputs().back();  // TODO SW-106237: when we double-store multiple tensors at once, this
                                             // should be done differently and according to the tensor ids from the elf
        TensorPtr origTensor = operationSlicePtr->getOriginalTensor(m_candidate.getConnectingTensor());
        operationSlicePtr->addTensorSliceOffset(
            doubleStore,
            origTensor,
            operationSlicePtr->getTensorSliceOffset(m_candidate.getConnectingTensor()));
        LOG_INFO(SPILL_FILL, "Fusion succeeded");
        return true;
    }
    return true;
}
