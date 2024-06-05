#include "memset_node_output.h"

#include "compiler_types.h"
#include "graph_editor.h"
#include "passes.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "reduction_node.h"
#include "memset_node.h"
#include "tensor.h"
#include "tpc_memset_node.h"
#include "tpc_node.h"
#include "types.h"
#include <memory>

MemsetNodeOutputManager::MemsetNodeOutputManager(HabanaGraph& g, const TensorPtr& tensor) : m_graph(g), m_tensor(tensor)
{
    if (m_graph.getCompilationMode() == CompilationMode::Graph)
    {
        // Todo [SW-101068] support memset for reduction on TPC with dynamic shape
        bool isDynamicShape = m_tensor->isDynamicShape();
        m_preferTpc = m_tensor->getTotalSizeInBytes() >= GCFG_RUN_MEMSET_ON_DMA_THRESHOLD.value() && !isDynamicShape;
    }
}

std::string MemsetNodeOutputManager::getNodeName(std::string_view prefix, std::string_view suffix)
{
    return fmt::format("{}_{}", prefix, suffix);
}

void MemsetNodeOutputManager::setTensorName(Tensor& tensor, std::string_view prefix, std::string_view suffix)
{
    tensor.setName(fmt::format("{}_{}", prefix, suffix));
}

NodePtr MemsetNodeOutputManager::createMemsetForReduction(const TensorPtr& zerosOutput, const std::string& baseName)
{
    // Todo [SW-100556] no const kernel for u32 dtype
    const HalReader& halReader         = *m_graph.getTraits().getHalReader();
    bool             supportedTpcDtype = halReader.isTPCMemsetSupportedDataType(zerosOutput->getElementType());
    const char* guid;
    if (m_preferTpc && supportedTpcDtype)
    {
        guid = NodeFactory::tpcMemsetNodeTypeName;
    }
    else
    {
        guid = NodeFactory::memsetNodeTypeName;
    }
    NodePtr memsetNode =
        NodeFactory::createInternalNode({}, {zerosOutput}, nullptr, guid, getNodeName(baseName, "memset_zero"));
    return memsetNode;
}

std::tuple<TensorPtr, NodePtr, NodePtr> MemsetNodeOutputManager::extract(const Node& node)
{
    const auto& baseName = m_tensor->getName();

    TensorPtr inTensor = m_tensor->clone(false, true, false, TensorNameClonePolicy::EMPTY_NAME);
    setTensorName(*inTensor, baseName, "reduction_in");

    TensorPtr zerosTensor = inTensor->clone(false, true, false, TensorNameClonePolicy::EMPTY_NAME);
    setTensorName(*zerosTensor, baseName, "zeros");
    // Set isReductionEnabled and reductionOperation since this pass runs after markReductionInputs
    // Passes order:
    // 1) markReductionInputs - marks tensors as reducible
    // 2) loadTpcKernels - loads tpc kernels params and sets Reducible flag for kernels
    // 3) memsetNodeOutput - adding reduction node and sets its input tensors reduction params
    ReductionOperation reductionOp               = REDUCTION_SET;
    TensorAnnotation& inAnn = inTensor->getTensorAnnotation();
    inAnn.tensorReductionInfo.isReductionEnabled = false;
    inAnn.tensorReductionInfo.reductionOperation = reductionOp;

    TensorAnnotation& zeroAnn = zerosTensor->getTensorAnnotation();
    zeroAnn.tensorReductionInfo.isReductionEnabled = false;
    zeroAnn.tensorReductionInfo.reductionOperation = reductionOp;

    NodePtr memsetNode = createMemsetForReduction(zerosTensor, baseName);

    NodePtr reduceNode = NodeFactory::createInternalNode({zerosTensor, inTensor},
                                                         {m_tensor},
                                                         &reductionOp,
                                                         NodeFactory::reductionNodeTypeName,
                                                         getNodeName(baseName, "reduction"));
    if (node.getNodeAnnotation().bundleInfo.is_set())
    {
        // Add the reduction and memset nodes to the bundle, otherwise it messes up the execution scheduler.
        // Set the reduction and memset nodes the same op index as TPC node, as it doesn't have to be unique.
        reduceNode->getNodeAnnotation().bundleInfo.set(node.getNodeAnnotation().bundleInfo.value());
        memsetNode->getNodeAnnotation().bundleInfo.set(node.getNodeAnnotation().bundleInfo.value());
    }

    // Input sub-graph: {input} -> [tpc node - memsetBeforeExecution=1] -> {output}
    // Output sub-graph: {input} -> [tpc node] -> [reduction node - reductionOp=REDUCTION_SET] -> {output}
    //                           [memset node] /

    // Maintain tracking of origin nodes for debug purposes
    const auto& originNodes = node.getOriginNodes();
    memsetNode->setOriginNodes(originNodes);
    reduceNode->setOriginNodes(originNodes);
    return {inTensor, memsetNode, reduceNode};
}

static void apply(HabanaGraph& g, TPCNodePtr& tpcNode)
{
    TensorVector memsetTensors = tpcNode->getMemsetBeforeExecTensors();
    for (const TensorPtr& tensor : memsetTensors)
    {
        if (!tensor->isReductionEnabled(true /*check set op*/) && !tensor->isPartOfRMWSection())
        {
            LOG_TRACE(GC, "Apply tensor memset for tensor {}, node {}", tensor->getName(), tpcNode->getNodeName());
            MemsetNodeOutputManager memsetNodeOutputManager(g, tensor);
            auto [inTensor, memsetNode, reduceNode] = memsetNodeOutputManager.extract(*tpcNode);
            GraphEditor::replaceTensor(g, tpcNode, tensor, inTensor);
            GraphEditor::addNodes(g, {memsetNode, reduceNode});
        }
    }
}

bool memsetNodeOutput(HabanaGraph& g)
{
    const NodeVector nodes = g.getExeSortedNodes();
    for (NodePtr node : nodes)
    {
        if (HabanaGraph::runsOnTPC(node))
        {
            auto tpcNode = std::static_pointer_cast<TPCNode>(node);
            apply(g, tpcNode);
        }
    }
    return true;
}
