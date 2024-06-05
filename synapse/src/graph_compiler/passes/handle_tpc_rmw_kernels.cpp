#include <bitset>

#include "graph_editor.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "tpc_node.h"
#include "memset_node_output.h"

static void setRMWOutput(const NodePtr& node, const TensorPtr& tensor, HabanaGraph& graph, bool addMemset)
{
    HB_ASSERT(tensor->getTotalSizeInBytes() <= GCFG_MAX_RMW_TENSOR_BYTES.value(),
              "Node {} , GUID {} has RMW indication to a large tensor, size: {}, max allowed: {}",
              node->getNodeName(),
              node->getGUID(),
              tensor->getTotalSizeInBytes(),
              GCFG_MAX_RMW_TENSOR_BYTES.value());

    LOG_DEBUG(GC, "Set RMW output {} for node {}, GUID: {}", tensor->getName(), node->getNodeName(), node->getGUID());

    GraphEditor::removeNode(graph, node);

    bool addMemcpyNode = false;
    TensorPtr sramTensor = tensor;
    if (! tensor->inSram() || tensor->getElementType() != syn_type_float)
    {
        addMemcpyNode = true;
        sramTensor = tensor->clone();
        // Reduction should happen in float
        // (if the original tensor is bfloat the memcpy will be replaced with cast in selectMemcpyNode pass)
        sramTensor->setElementType(syn_type_float);
        sramTensor->setTensorInSram();
        sramTensor->setMemorySectionID(getSramMemoryID());
        sramTensor->setName(tensor->getName() + "_sram");
    }
    TensorPtr sramTensorReducted = sramTensor->clone(false, false);
    sramTensorReducted->setTensorInSram();
    sramTensorReducted->setName(sramTensor->getName() + "_reducted");

    node->replaceFirstTensor(tensor, sramTensor);

    // In order to have reduction on a given tensor, we need the following steps
    // 1. Add a reduction node for the tensor to be later marked as RMW.
    // 2. memcpy the reduction output to the original tensor (memcpy node)
    // 3. If memset is required by kernel, add a memset node for reduction.
    NodeList     newNodes;
    TensorVector reductionInputs;
    newNodes.push_back(node);
    if (addMemset)
    {
        TensorPtr memsetTensor = sramTensor->clone();
        memsetTensor->setTensorInSram();
        memsetTensor->setName(sramTensor->getName() + "_zeros");
        MemsetNodeOutputManager memsetNodeOutputManager(graph, memsetTensor);
        NodePtr                 memsetNode =
            memsetNodeOutputManager.createMemsetForReduction(memsetTensor, sramTensorReducted->getName());
        newNodes.push_back(memsetNode);
        reductionInputs.push_back(memsetTensor);
    }
    reductionInputs.push_back(sramTensor);
    NodePtr reductionNode = NodeFactory::createNode(reductionInputs,
                                                    {sramTensorReducted},
                                                    nullptr,
                                                    0,
                                                    NodeFactory::reductionNodeTypeName,
                                                    sramTensor->getName() + "_reduction");
    newNodes.push_back(reductionNode);
    if (addMemcpyNode)
    {
        NodePtr memcpyNode = NodeFactory::createNode({sramTensorReducted}, {tensor}, nullptr, 0, NodeFactory::memcpyNodeTypeName, sramTensorReducted->getName() + "_memcpy");
        newNodes.push_back(memcpyNode);
    }

    // make sure fused nodes delay is propagated
    GraphEditor::updateWaitCycles({node}, newNodes);

    // Add the nodes
    std::for_each(newNodes.begin(), newNodes.end(), [&graph, &node](const NodePtr& newNode) {
        // Maintain tracking of origin nodes for debug purposes
        newNode->setOriginNodes(node->getOriginNodes());
        GraphEditor::addNode(graph, newNode);
    });
}

static bool reductionEnabledForTensor(const TensorPtr& tensor, HabanaGraph& graph)
{
    if (! tensor->inSram()) return false;

    const auto& consumers = graph.getTensorConsumers(tensor);
    return (consumers.size() == 1 &&
            consumers.front()->getNodeType() == Node::TYPE_INTERNAL_REDUCTION);
}

static bool canTensorBeReducted(const TensorPtr& tensor)
{
    // Reduction happens in float
    // Currently bf16 kernels dosn't support different data type
    return tensor->getElementType() == syn_type_float;
}

// TPC kernels that wish to use HW feature - read modify write AKA reduction
// is handled at this pass
bool handleTpcRmwKernels(HabanaGraph& graph)
{
    // Don't pass on graph if feature disabled
    if (GCFG_MAX_RMW_TENSOR_BYTES.value() == 0) return true;

    const NodeSet& allNodes = graph.getNodes();
    for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end();)
    {
        TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(*nodeIter);
        ++nodeIter;
        if (tpcNode)
        {
            std::bitset<BITS_PER_BYTE * sizeof(uint16_t)> rmwMask =
                tpcNode->getRmwOutputMask(deviceTypeToDeviceID(graph.getDeviceType()));

            // Don't handle if all bits are marked
            // LLVM compiler can't tell statically which tensors need RMW so it marks them all
            // For now, don't handle such cases
            // TODO LLVM-527: once the ticket is resolved, remove the all check
            if (rmwMask.none() || rmwMask.all()) continue;

            //Sum rmw tensor size, call setRMWOutput only if total size < max allowed rmw bytes
            size_t requiredRmwSramBytes = 0;
            TensorVector rmwOutputs;
            for (uint32_t index = 0; index < tpcNode->getNumOutputs(); ++index)
            {
                // rmwMask has a bit for both inputs and outputs ordered from the first input to the last output
                // disregard aux and output describing shape tensors - they are not passed to kernel
                size_t outputBitPos = tpcNode->getNumInputsToKernel() + index;
                if (rmwMask.test(outputBitPos))
                {
                    const TensorPtr& output = tpcNode->getOutput(index);
                    if (!reductionEnabledForTensor(output, graph) && canTensorBeReducted(output) &&
                        !output->isPartOfRMWSection())  // tensors already in RMW section should be ignored
                    {
                        requiredRmwSramBytes += output->getTotalSizeInBytes();
                        rmwOutputs.push_back(output);
                    }
                }
            }
            if (requiredRmwSramBytes <= GCFG_MAX_RMW_TENSOR_BYTES.value())
            {
                for (const auto& output : rmwOutputs)
                {
                    bool isMemsetBeforeExecution =
                        tpcNode->isOutputTensorMemset(tpcNode->getOutputIndexOfTensor(output),
                                                      deviceTypeToDeviceID(graph.getDeviceType()));
                    setRMWOutput(tpcNode, output, graph, isMemsetBeforeExecution);
                }
            }
            else
            {
                LOG_WARN(GC, "Cannot set RMW output for node {} outputs as outputs are too large, requested: {}, allowed: {}",
                        tpcNode->getNodeName(),
                        requiredRmwSramBytes,
                        GCFG_MAX_RMW_TENSOR_BYTES.value());
            }
        }
    }

    return true;
}