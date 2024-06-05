#include "mme_shared_input.h"
#include "slicing_utils.h"
#include "slicing_brain.h"
#include "flatten_mme.h"
#include "tensor.h"

std::list<pBundleExpansion> SharedMMEInputCandidateHandler::findSharedMMEInputCandidate(const pMmeSlicingStrategy& strategy,
                                                                                        const HabanaGraph& graph) const
{
    std::list<pBundleExpansion> expansions;
    const MmeSlicingStrategy::MmeSlicingData& slicingData = strategy->getMmeSlicingData();
    pNode masterNode = graph.getTensorProducer(slicingData.masterOperand->originalTensor);

    for (auto& it : findSharedMMEInputConsumers(masterNode, graph))
    {
        const pNode& slaveNode     = it.first;
        const pTensor& sharedInput = it.second;
        const pSlicedOperand& sharedOperand = (slicingData.getWide()->originalTensor == sharedInput) ?
                                              slicingData.getWide() : slicingData.getNarrow();

        // shared operand can only be sliced on a single dimension
        // in case of batch gemm can be sliced on all batch dimensions
        unsigned actualSlicedDims = 0;
        for (unsigned i = 0; i < sharedOperand->originalTensor->getDim(); ++i)
        {
            if (SlicedOperandUtils::isSlicedOnDimension(sharedOperand, i))
            {
                actualSlicedDims++;
                if (i >= DIM_GEMM_BATCH && masterNode->isBatchGemm() && slaveNode->isBatchGemm()) break;
            }
        }
        if (actualSlicedDims > 1) continue;

        pBundleExpansion candidate = std::make_shared<BundleExpansion>();
        candidate->role = BundleExpansion::SharedInputConsumer;
        candidate->bundleNode      = masterNode;
        candidate->nodeToStitch    = slaveNode;
        candidate->stitchedOperand = sharedOperand;

        auto nonSharedInputOperand = SlicedOperandUtils::createNonSharedOperandOfNode(sharedOperand, slaveNode);
        auto slaveOutputOperand = std::make_shared<SlicedOperand>(slaveNode->getOutput(0));
        candidate->slaveOperands.setInput(nonSharedInputOperand);
        candidate->slaveOperands.setOutput(slaveOutputOperand);

        const auto& slaveNodeTensors = slaveNode->getInputs();
        auto shapeTensorIt = std::find_if(slaveNodeTensors.begin(), slaveNodeTensors.end(),
                           [] (const pTensor& t) { return t && t->isShapeTensor(); });

        if(shapeTensorIt != slaveNodeTensors.end())
        {
            candidate->slaveOperands.setShapeOperand(std::make_shared<SlicedOperand>(*shapeTensorIt));
        }

        expansions.push_back(candidate);
    }

    return expansions;
}

// find all MME nodes with shared input
SlaveNodeToSharedTensor SharedMMEInputCandidateHandler::findSharedMMEInputConsumers(const pNode& masterNode, const HabanaGraph& graph) const
{
    SlaveNodeToSharedTensor sharedInputConsumersMap;
    for (auto& input : masterNode->getInputs())
    {
        if (!input || input->isShapeTensor()) continue;
        const NodeList& consumers = graph.getTensorConsumers(input);
        for (auto& c : consumers)
        {
            if (c == masterNode || !HabanaGraph::runsOnMME(c) || (c->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM))
                continue;

            //TODO SW-18666 Need to allow 4D solver for flattenable nodes, and decide based on master strategy
            if (MMENodeFlattener::canFlattenMMENode(masterNode) && !MMENodeFlattener::canFlattenMMENode(c))
            {
                continue;
            }

            sharedInputConsumersMap[c] = input;
        }
    }
    return sharedInputConsumersMap;
}

bool SharedMMEInputCandidateHandler::isCandidateValidForStrategy(const pBundleExpansion& candidate,
                                                                 const pMmeSlicingStrategy& strategy)
{
    // check that candidate input resides in sram
    uint64_t sramCap = strategy->calculateMetrics().SRAMCapacity;
    sramCap += candidate->additionalSRAMCapacity;
    // make sure shared operand is sliced on only one dimension
    unsigned slicedDimCount = 0;
    bool slicedOnBatch = false;
    for (unsigned dim = 0; dim < candidate->stitchedOperand->originalTensor->getDim(); dim++)
    {
        if (SlicedOperandUtils::isSlicedOnDimension(candidate->stitchedOperand, dim))
        {
            slicedDimCount++;
            if (dim >= DIM_GEMM_BATCH && candidate->nodeToStitch->isBatchGemm() && candidate->bundleNode->isBatchGemm())
            {
                slicedOnBatch = true;
                break;
            }
        }
    }
    bool isValidCandidate = true;
    if (slicedDimCount > 1)
    {
        isValidCandidate = false;
        SLC_TRACE("Cannot stitch shared input consumer {} to strategy since candidate is sliced on more then one dim",
                   candidate->nodeToStitch->getNodeName());
    }
    else if (candidate->additionalSRAMCapacity == 0)
    {
        isValidCandidate = false;
        SLC_TRACE("Cannot stitch shared input consumer to strategy since no valid slicing was found");
    }
    else if (sramCap > SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        isValidCandidate = false;
        SLC_TRACE("Cannot stitch shared input consumer {} to strategy since candidate additional SRAM capacity exceeds max capacity",
                  candidate->nodeToStitch->getNodeName());
    }
    else if (candidate->nodeToStitch->isBatchGemm() && candidate->bundleNode->isBatchGemm())
    {
        if (!slicedOnBatch)  // ignore batch gemms that not sliced on the batch
        {
            isValidCandidate = false;
            SLC_TRACE("Cannot stitch shared input consumer {} to strategy since candidate is Batch Gemm and not sliced "
                      "on batch",
                      candidate->nodeToStitch->getNodeName());
        }
        else
        {
            const auto* slaveBgemm = dynamic_cast<BatchGemmNode*>(candidate->nodeToStitch.get());
            HB_ASSERT_PTR(slaveBgemm);
            for (unsigned dim = DIM_GEMM_BATCH; dim < candidate->stitchedOperand->originalTensor->getDim(); ++dim)
            {
                if ((candidate->stitchedOperand->chunkDimensions[dim] > 1) && !slaveBgemm->isSymmetricLayout())
                {
                    // TODO - SW-41350 : Support broadcast Batch-Gemm in MME
                    isValidCandidate = false;
                    SLC_TRACE(
                        "Cannot stitch shared input consumer {} to strategy - broadcast batch-gemm is not supported",
                        candidate->nodeToStitch->getNodeName());
                }
            }
        }
    }

    return isValidCandidate;
}

uint64_t SharedMMEInputCandidateHandler::getCandidateAdditionalCapacity(const pBundleExpansion& candidate)
{
    SLC_TRACE("Calculating additional SRAM capacity for candidate - {}", candidate->nodeToStitch->getNodeName());
    uint64_t additionalSramCap = 0;
    pSlicedOperand nonSharedInput = candidate->slaveOperands.getInput();
    if (nonSharedInput->resideInSRAM)
    {
        additionalSramCap = static_cast<uint64_t>(SlicedOperandUtils::getSliceSizeInBytes(nonSharedInput)) *
                            nonSharedInput->numOfBuffers;
        // if shared operand is sliced on the common dim regarding slave node -
        // we also need to add the slave node output to the SRAM capacity as partials are created with datatype of float
        if (SlicedOperandUtils::isSlicedOnCommonDim(candidate->stitchedOperand, candidate->nodeToStitch))
        {
            additionalSramCap += candidate->nodeToStitch->getOutput(0)->getDenseSizeInElements() * sizeof(syn_type_float);
        }
    }

    return additionalSramCap;
}

pBundleExpansion SharedMMEInputCandidateHandler::getCandidateFromStrategy(const MmeSlicingStrategy* strategy,
                                                                          bool& isValidForRole)
{
    pBundleExpansion candidate = strategy->getMmeSlicingData().getRoleCandidates()[BundleExpansion::SharedInputConsumer];
    isValidForRole = true;
    if (!candidate || !candidate->nodeToStitch)
    {
        for (const pBundleExpansion& invalidCandidate : strategy->getMmeSlicingData().getInvalidCandidates())
        {
            if (invalidCandidate->role == BundleExpansion::SharedInputConsumer)
            {
                candidate      = invalidCandidate;
                isValidForRole = false;
            }
        }

    }
    return candidate;
}
