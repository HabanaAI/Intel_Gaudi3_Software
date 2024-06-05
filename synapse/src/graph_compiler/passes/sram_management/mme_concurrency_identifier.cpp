#include "defs.h"
#include "habana_global_conf.h"
#include "include/mme_common/mme_common_enum.h"
#include "reductions.h"
#include "settable.h"
#include "synapse_common_types.h"
#include "graph_editor.h"
#include "node_factory.h"
#include "mme_concurrency_identifier.h"
#include "types.h"
#include "cast_nodes_handler.h"
#include "mme/mme_brain_ifc.h"
#include "brain_conf.h"

bool MmeConcurrencyIdentifier::allProducerNodesEligibleForCdConcurrency(const HabanaGraph& g, const NodePtr node)
{
    const auto& inputTensors = node->getInputs();
    if (inputTensors.size() == 0)
    {
        return false;  // no producers
    }
    for (auto pTensor : inputTensors)
    {
        NodePtr producerNode = g.getTensorProducer(pTensor);
        if (!producerNode ||
            producerNode->getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn != MmeCommon::TurnedOn)
        {
            return false;
        }
    }
    return true;
}

void MmeConcurrencyIdentifier::resetCdConcurrencyOfAllProducers(const HabanaGraph& g, const NodePtr node)
{
    const auto& inputTensors = node->getInputs();
    for (auto pTensor : inputTensors)
    {
        NodePtr producerNode = g.getTensorProducer(pTensor);
        if (producerNode)
        {
            producerNode->getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn = MmeCommon::TurnedOff;
        }
    }
}

//========= Methods that check if the node is eligible for cd concurrency =============
bool MmeConcurrencyIdentifier::gaudiMmenodeIsCandidateForCdConcurrency(const NodePtr& node) const
{
    // 1. Op is dedw
    if (node->getNodeType() != Node::TYPE_DEDW)
    {
        return false;
    }
    // 2. Output type is float32
    if (node->getOutput(0)->getElementType() != syn_type_single)
    {
        return false;
    }
    // 3. Output size is up to a single MME size
    auto      inputDT    = node->getInput(0)->getElementType();
    SizeArray outputDims = node->getOutput(0)->getAllSizesInElements();
    unsigned  outW       = outputDims[0];
    unsigned  outH       = outputDims[1];
    if (outW > m_graph.getHALReader()->getMmeSymmetricWidthInElems(inputDT) ||
        outH > m_graph.getHALReader()->getMmeSymmetricHeightInElems(inputDT))
    {
        return false;
    }

    // 4. Conv params are gemm convertible (kernel: 1x1x1, stride: 1x1x1, padding: 0x0x0)
    const std::shared_ptr<MmeNode> mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
    if (!mmeNode->canBeConvertedToGEMM())
    {
        return false;
    }

    return true;
}

// Check that the concurrency level does not exceed 16: TODO SW-137230
// The concurrency level defines the number of partial outputs that are produced by the Gemms
// that are concurrently used. When the data type is either fp16 or bf16, each partial result
// is produced in lower accuracy (compared to fp32 if cd concurrency does not take place). It is
// proved that the impact on accuracy up to 16 partial results is small enough.
bool MmeConcurrencyIdentifier::isMaxGemmNrSupportedForFp16() const
{
    unsigned maxGemmsNr = m_graph.getHALReader()->getNumMmeEngines() *
                          m_graph.getHALReader()->getNumMmeCoresPerEngine() *
                          m_graph.getHALReader()->getNumMmeMaxGemmsPerCore();
    return maxGemmsNr <= 16;
}

bool MmeConcurrencyIdentifier::outputInReducibleMemory(const NodePtr& node)
{
    // TODO SW-136691: The following condition is a temporary workaround for Gaudi3
    const synDataType outputDataType = node->getOutput(0)->getElementType();
    if ((outputDataType == syn_type_fp16 || outputDataType == syn_type_bf16))
    {
        if (isMaxGemmNrSupportedForFp16() == false) return false;
    }

    // Check output tensor 0
    bool outputInSram = node->getOutput(0)->inSram();
    if (!m_graph.getHALReader()->isReducibleMemory(outputInSram ? MemoryType::MEMORY_TYPE_SRAM
                                                                : MemoryType::MEMORY_TYPE_DRAM))
    {
        return false;
    }
    // Check output tensor 1, if exists
    if (node->getOutput(1) != nullptr)
    {
        bool outputInSram = node->getOutput(1)->inSram();
        if (!m_graph.getHALReader()->isReducibleMemory(outputInSram ? MemoryType::MEMORY_TYPE_SRAM
                                                                    : MemoryType::MEMORY_TYPE_DRAM))
        {
            return false;
        }
    }
    return true;
}

bool MmeConcurrencyIdentifier::dataTypeSupportsCdConcurrency(synDataType outputDataType)
{
    // TODO 106087: Apply to nodes of output of fp8 as well. Currently applying to such nodes fail because
    // no cast from fp32 to fp8 is available
    if (outputDataType == syn_type_tf32)
    {
        return false;  // TF32 not supported yet
    }
    if (m_graph.getDeviceType() == synDeviceGaudi3 && outputDataType == syn_type_fp8_143)
    {
        return false; // cast kernel to fp8_143 is not yet implemented.
    }

    return true;
}

bool MmeConcurrencyIdentifier::nodeIsCandidateForCdConcurrency(const NodePtr& node)
{
    auto mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
    if (mmeNode == nullptr)
    {
        return false;
    }

    // The current code produces non-deterministic results because the order of writes in the MME is
    // non deterministic. Therefore the optimization should be enabled only if non-deterministic
    // results are allowed (or we force the non-deterministic optimization)
    if (node->getDeterministic() && !GCFG_FORCE_MME_CD_CONCURRENCY_NON_DETERMINISTIC.value())
    {
        return false;
    }
    // Verify that output tensor is located in reducible memory
    if (!outputInReducibleMemory(node))
    {
        return false;
    }
    synDataType outputDataType = node->getOutput(0)->getElementType();
    if (!dataTypeSupportsCdConcurrency(outputDataType))
    {
        return false;
    }

    // Gaudi uses dedwAsBgemm, for which we check the optimization conditions locally.
    // For all other devices we use CD Concurrency for which we use the mme brain
    if (m_graph.getDeviceType() == synDeviceGaudi)
    {
        if (gaudiMmenodeIsCandidateForCdConcurrency(node))
        {
            return true;
        }
        return false;
    }
    else
    {
        if (!mmeNode->getMmeBrainIfc()->opSupportsChoosingCdConcurrency())
        {
            return false;
        }
    }

    return true;
}

// In some data types we prefer to perform the accumulation of the cd concurrency partial results in fp32.
// There is no need to add if the output is already in fp32.
bool MmeConcurrencyIdentifier::dataTypeRequiresAccumulationFp32Reduction(synDataType outputDataType)
{
    return !gc::reduction::datatypeValidForAccumulation(outputDataType);
}


void MmeConcurrencyIdentifier::addFloat32OutputNodeAsNeeded(const NodePtr& mmeNode, unsigned outputIdx)
{
    // The function inserts float32 tensor and cast node
    // Schematically, graph before
    //     mmeNode -> mmeOutput
    // Graph after
    //     mmeNode -> mmeOutputFloat32 -> castNode -> mmeOutput
    TensorPtr mmeOutput = mmeNode->getOutput(outputIdx);
    if (dataTypeRequiresAccumulationFp32Reduction(mmeOutput->getElementType()))
    {
        // Create the float32 output tensor
        TensorPtr mmeOutputFp32 = mmeOutput->clone(false, false, false);
        mmeOutputFp32->setElementType(syn_type_float);
        mmeOutputFp32->setName(fmt::format("{}_Fp32", mmeOutput->getName()));
        GraphEditor::replaceOutput(m_graph, mmeNode, outputIdx, mmeOutputFp32);
        /*
         * Originally added to remove unnecessary cast, that wasn't detected in
         * remove contiguous casts since the "opposite" cast was already removed by
         * eliminate redundant nodes pass.
         */
        m_graph.turnOnPredicate(PREDICATE_ID_ELIMINATE_REDUNDANT_NODES);

        // Create the cast node
        NodePtr castNode = CastNodeHandler::createCastNode(mmeOutputFp32,
                                                           mmeOutput,
                                                           fmt::format("{}_cast", mmeOutput->getName()),
                                                           m_graph.getDeviceId());

        // Add the cast node to the graph. Maintain tracking of origin nodes for debug purposes
        castNode->setOriginNodes(mmeNode->getOriginNodes());
        GraphEditor::addNode(m_graph, castNode);


        // Associate the cast node with the current bundle, if exists
        const auto& mmeBundleInfo = mmeNode->getNodeAnnotation().bundleInfo;
        if (mmeBundleInfo.is_set())
        {
            castNode->getNodeAnnotation().bundleInfo = mmeBundleInfo;
        }
    }
}

bool MmeConcurrencyIdentifier::nodeHasCdConcurrencyAndNotFeedReduction(const HabanaGraph& g, NodePtr node)
{
    if (node->getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn != MmeCommon::TurnedOn)
    {
        return false;
    }
    // Check if the consumer is Reduction node
    NodeList consumerNodes = g.getTensorConsumers(node->getOutput(0));
    if (!consumerNodes.empty() && consumerNodes.front()->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
    {
        // We have a hidden assumption that if the node has been already sliced than the output tensor
        // is float32 or otherwise the slicing may impact accuracy of results.
        synDataType outputDataType = node->getOutput(0)->getElementType();
        HB_ASSERT((outputDataType == syn_type_float) || (outputDataType == syn_type_hb_float),
                  "Identified node that feeds reduction with non-fp32 output");
        return false;
    }
    return true;
}

bool MmeConcurrencyIdentifier::isCdConcurrencyEnabled(const HabanaGraph& g)
{
    bool cdConcurrencyEnabled = GCFG_ENABLE_MME_CD_CONCURRENCY.value();
    if (cdConcurrencyEnabled)
    {
        // in case LB is enabled - it controls the memset\reduction additions.
        if (GCFG_ENABLE_LAYERED_PIPELINE_BRAIN.value())
        {
            cdConcurrencyEnabled = false;
        }
    }
    return cdConcurrencyEnabled;
}

bool MmeConcurrencyIdentifier::scanGraph()
{
    if (!isCdConcurrencyEnabled(m_graph))
    {
        return true;
    }

    // 2 loops:
    // 1. Go over all mme nodes and identify if they are candidates to cd and batch concurrencies by setting
    //    the appropriate annotation to Undefined
    // 2. For every mme node call mme brain to finalize the concurrency and set it to either TurnedOn or TurnedOff
    //    Insert float32 output if needed

    // Pass 1: Identify candidates
    for (const NodePtr& node : m_graph.getExeSortedNodes())
    {
        auto mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
        if (mmeNode != nullptr)
        {
            // check if the node is candidate for cd concurrency
            if (nodeIsCandidateForCdConcurrency(node))
            {
                if (m_graph.getDeviceType() == synDeviceGaudi)
                {
                    // In Gaudi we do not use mme brain to choose between concurrencies, it is set here
                    mmeNode->getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn =
                        MmeCommon::TurnedOn;
                }
                else
                {
                    // Set to undefined so that it will be chosen by the mme brain
                    mmeNode->getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn =
                        MmeCommon::Undefined;
                }
            }
            else
            {
                mmeNode->getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn = MmeCommon::TurnedOff;
            }
        }
        else
        {
            node->getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn = MmeCommon::TurnedOff;
        }
    }

    // Pass 2: Finalize the concurrencies, and create a vector of all nodes to which cd concurrency is applied
    for (const NodePtr& node : m_graph.getExeSortedNodes())
    {
        auto mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
        if (mmeNode != nullptr)
        {
            // Choose the concurrency
            if (m_graph.getDeviceType() != synDeviceGaudi)
            {
                mmeNode->getMmeBrainIfc()->setRecommendedConcurrency();
            }
        }
        // If the node if Reduction, check if all its producers are eligible for cd concurrency. If not, reset
        // their cd concurrency annotation. If yes, add it to the cd list
        if (node->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            if (!allProducerNodesEligibleForCdConcurrency(m_graph, node))
            {
                resetCdConcurrencyOfAllProducers(m_graph, node);
            }
        }
        if (nodeHasCdConcurrencyAndNotFeedReduction(m_graph, node))
        {
            // If the output data type is not float, add float32 tensor as an output of the node + cast to the
            // original data type. This maintains the accuracy of calculations.
            addFloat32OutputNodeAsNeeded(node, 0);
            if (node->getOutput(1) != nullptr)  // in case of secondary output
            {
                addFloat32OutputNodeAsNeeded(node, 1);
            }
        }
    }

    return true;
}
