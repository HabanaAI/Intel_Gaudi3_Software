#include "defs.h"
#include "habana_global_conf.h"
#include "include/mme_common/mme_common_enum.h"
#include "settable.h"
#include "synapse_common_types.h"
#include "graph_editor.h"
#include "node_factory.h"
#include "mme_concurrency_identifier.h"
#include "mme_concurrency_memset.h"
#include "types.h"
#include "cast_nodes_handler.h"
#include "mme/mme_brain_ifc.h"


void MmeConcurrencyMemset::createMemsetNodeAndTensor(const TensorPtr& refTensor,
                                                         NodePtr&         memsetNode,
                                                         TensorPtr&       memsetTensor)
{
    memsetTensor = refTensor->clone(false, false, false);
    // Set the memset tensor in the same memory type as the ref tensor
    if (refTensor->inSram())
    {
        memsetTensor->setTensorInSram();
    }
    else
    {
        memsetTensor->setTensorInWorkspace();
    }

    memsetTensor->setName(refTensor->getName() + "_zeros");
    // Create the memset node with memsetTensor as output
    memsetNode = NodeFactory::createNode({},
                                         {memsetTensor},
                                         nullptr,
                                         0,
                                         NodeFactory::memsetNodeTypeName,
                                         refTensor->getName() + "_memset");
}

void MmeConcurrencyMemset::addMemsetNodeToReduction(const NodePtr& reductionNode)
{
    auto inputTensors = reductionNode->getInputs();

    // Step 1: Create the memset tensor and node and add to the graph
    TensorPtr memsetTensor;
    NodePtr   memsetNode;
    createMemsetNodeAndTensor(inputTensors[0], memsetNode, memsetTensor);

    // Maintain tracking of origin nodes for debug purposes
    memsetNode->setOriginNodes(reductionNode->getOriginNodes());

    GraphEditor::addNode(m_graph, memsetNode);
    // Step 2: Add the memset tensor as input to the Reduction node
    GraphEditor::editNode(m_graph, reductionNode, [&]() {
        reductionNode->addInput(reductionNode->getNumInputs(), memsetTensor);
    });

    // Step 3: Run over all input nodes: update their bundle info and mask CD Concurrency optimization
    const auto&        reductionBundleInfo = reductionNode->getNodeAnnotation().bundleInfo;
    Settable<unsigned> minOperationIdx;  // Minimal operation index of the reduction producers
    for (const TensorPtr& pTensor : inputTensors)
    {
        NodePtr producerNode = m_graph.getTensorProducer(pTensor);
        // Verify that CD Concurrency optimization is already enabled
        HB_ASSERT(producerNode->getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn ==
                      MmeCommon::TurnedOn,
                  "Expected cd concurrency to be turned on");
        // Update the bundle info
        if (reductionBundleInfo.is_set())
        {
            const auto& producerBundleInfo = producerNode->getNodeAnnotation().bundleInfo;
            HB_ASSERT(producerBundleInfo.is_set() &&
                          (producerBundleInfo->bundleIndex == reductionBundleInfo->bundleIndex),
                      "All the MME producers must have the same bundle idx as the reduction node");
            if (!minOperationIdx.is_set() || (producerBundleInfo->operationIndex < minOperationIdx.value()))
            {
                minOperationIdx = producerBundleInfo->operationIndex;
            }
        }
    }

    if (reductionBundleInfo.is_set())
    {
        // Add the memset node to the bundle
        memsetNode->getNodeAnnotation().bundleInfo = reductionBundleInfo;
        // Update the memsest operation index so it will be scheduled before its first dependent node.
        memsetNode->getNodeAnnotation().bundleInfo->operationIndex = minOperationIdx.value();
    }
}

void MmeConcurrencyMemset::addMemsetAndReductionToOutput(const NodePtr mmeNode, unsigned outputIdx)
{
    // The mme node writes to an output tensor called here mmeOutput.
    // This function adds memset and reduction node in order to preset the mmeOutput to zero.
    // Schematically:
    //   MmeNode ->   beforeReductionTensor ---->
    //                                       ReductionNode -> mmeOutput
    //   MemsetNode -> memsetTensor        ---->
    //
    TensorPtr mmeOutput = mmeNode->getOutput(outputIdx);

    // Step 1: Create the memset node & tensor
    TensorPtr memsetTensor;
    NodePtr   memsetNode;
    createMemsetNodeAndTensor(mmeOutput, memsetNode, memsetTensor);

    // Step 2: Create the Reduction node & tensor
    TensorPtr beforeReductionTensor = mmeOutput->clone(false, false, false);
    beforeReductionTensor->setTensorInWorkspace();  // todo AlonG: Required due to bug SW-95251
    beforeReductionTensor->setName(mmeNode->getOutput(outputIdx)->getName() + "_before_reduction");
    NodePtr reductionNode = NodeFactory::createNode({memsetTensor, beforeReductionTensor},
                                                    {mmeOutput},
                                                    nullptr,
                                                    0,
                                                    NodeFactory::reductionNodeTypeName,
                                                    mmeOutput->getName() + "_reduction");

    // Step 3: Add all nodes to the graph
    GraphEditor::replaceOutput(m_graph, mmeNode, outputIdx, beforeReductionTensor);
    NodeList newNodes = {reductionNode, memsetNode};
    for (auto node : newNodes)
    {
        // Maintain tracking of origin nodes for debug purposes
        node->setOriginNodes(mmeNode->getOriginNodes());
        GraphEditor::addNode(m_graph, node);
    };

    // Step 4: Associate all new nodes with the current bundle
    const auto& mmeBundleInfo = mmeNode->getNodeAnnotation().bundleInfo;
    if (mmeBundleInfo.is_set())
    {
        for (auto node : newNodes)
        {
            node->getNodeAnnotation().bundleInfo = mmeBundleInfo;
        }
    }
}

void MmeConcurrencyMemset::addMemsetAndReductionNodes(const NodePtr mmeNode)
{
    // Verify that cd concurrency is already enabled
    HB_ASSERT(mmeNode->getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn == MmeCommon::TurnedOn,
              "Expected cd concurrency to be turned on");

    // The mme output is replaced by a sub-graph that includes memset and reduction nodes
    // In case of secondary output, this is applied to the second output as well
    addMemsetAndReductionToOutput(mmeNode, 0);

    if (mmeNode->getOutput(1) != nullptr)  // in case of secondary output
    {
        addMemsetAndReductionToOutput(mmeNode, 1);
    }
}

bool MmeConcurrencyMemset::addMemsetNodes()
{
    if (!MmeConcurrencyIdentifier::isCdConcurrencyEnabled(m_graph))
    {
        return true;
    }

    // The pass adds memset nodes where needed:
    // 1. If a node has cd concurrency indication and its consumer is not reduction, add both reduction and memset
    // 2. If the node is reduction, and all its producers have cd concurrency indication, add memset only

    for (const NodePtr& node : m_graph.getExeSortedNodes())
    {
        if (MmeConcurrencyIdentifier::nodeHasCdConcurrencyAndNotFeedReduction(m_graph, node))
        {
            addMemsetAndReductionNodes(node);
        }
        if (node->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            if (MmeConcurrencyIdentifier::allProducerNodesEligibleForCdConcurrency(m_graph, node))
            {
                addMemsetNodeToReduction(node);
            }
            else
            {
                MmeConcurrencyIdentifier::resetCdConcurrencyOfAllProducers(m_graph, node);
            }
        }
    }
    
    return true;
}
