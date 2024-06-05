#include "compilation_hal_reader.h"
#include "data_type_utils.h"
#include "memcpy_engine_manager.h"
#include "habana_graph.h"
#include "passes.h"
#include "node_factory.h"
#include "cast_nodes_handler.h"
#include "graph_editor.h"
#include "memset_node.h"
#include "dma_memcopy_node.h"
#include "synapse_common_types.h"
#include "utils.h"

NodeList MemcpyEngineManager::createConcreteNode(const HabanaGraph& graph, const NodePtr& node) const
{
    NodeList    nodes;
    switch (node->getNodeType())
    {
        case Node::TYPE_MEMCOPY:
            // Debug assert on mem copy assumptions
            HB_DEBUG_VALIDATE(validateMemCopy(node));
            nodes = createConcreteCopyNode(graph, node);
            break;
        case Node::TYPE_MEMSET:
        {
            const bool memsetAsMemcpy   = GCFG_GAUDI_MEMSET_HW_WA.value();
            const bool isGaudi1Platform = graph.getDeviceType() == synDeviceGaudi;
            HB_ASSERT(memsetAsMemcpy == isGaudi1Platform,
                      "Memset WA is only needed to Gaudi1 platform, lhs={}, rhs={}",
                      memsetAsMemcpy,
                      isGaudi1Platform);
            const MemsetNode* const memsetSemanticNode = dynamic_cast<MemsetNode*>(node.get());
            HB_ASSERT(memsetSemanticNode, "Memset semantics node is NULL");
            // For Dynshapes with WA we will create a copy node even if output is dynamic, since we will do a linear
            // memset on all the tensor anyway.
            if (memsetAsMemcpy && memsetSemanticNode->getOutput(0)->isDenseLayout())
            {
                nodes = createConcreteCopyNode(graph, node);
            }
            else
            {
                const auto memsetNode = createConcreteSetNode(graph, node);
                HB_ASSERT_PTR(memsetNode);
                nodes.emplace_back(std::move(memsetNode));
            }
            break;
        }
        break;
        default:
            return {};
    }
    if (nodes.empty())
    {
        LOG_DEBUG(GC,
                  "Could not create concrete node for semantic node {}[{}]",
                  node->getNodeName(),
                  node->getNodeTypeStr());
    }
    return nodes;
}

bool MemcpyEngineManager::selectEngine(HabanaGraph& graph)
{
    NodeVector nodes = graph.getExeSortedNodes();
    for (const NodePtr& node : nodes)
    {
        NodeList newNodes = createConcreteNode(graph, node);
        if (!newNodes.empty())
        {
            updateNodeAnnotation(node, newNodes);
            const auto replaceRet = GraphEditor::replaceNodes(graph, {node}, newNodes);
            HB_ASSERT(replaceRet == REPLACE_NODE_SUCCESS,
                      "Failed replacing semantic node {}[{}], replaceNodes retval: {}",
                      node->getNodeName(),
                      node->getNodeTypeStr(),
                      replaceRet);
        }
    }

    return true;
};

static bool isNDimCopy(const NodePtr& semanticNode)
{
    return (semanticNode->getNumInputsDataTensors() > 0 &&
            semanticNode->getInput(0)->getDim() > DMAMemcpyNode::MAX_SUPPORTED_DIM &&
            semanticNode->getOutput(0)->getDim() > DMAMemcpyNode::MAX_SUPPORTED_DIM);
}

NodeList MemcpyEngineManager::lower64bMemcpyTo32b(const NodePtr& semanticNode) const
{
    // Lower high rank (dim>5) 64b memcpy to utilize 32bit memcpy_nd kernel:
    // (FCD left)
    // Initial memcpy_nd:
    // [X,Y,Z]64b -> memcpy_nd_64b -> [X,Y,Z]64b
    //
    // Resulting sequence:
    // [X,Y,Z]64b -> (reinterpret cast) -> [2*X,Y,Z]32b -> memcpy_nd_32b -> [2*X,Y,Z]32b ->
    // -> (reinterpret cast) -> [X,Y,Z]64b
    NodeList nodes;
    HB_ASSERT_PTR(semanticNode);
    HB_ASSERT(isMemcpy(*semanticNode),
              "Expecting {}[{}] to be a memcpy",
              semanticNode->getNodeName(),
              semanticNode->getNodeTypeStr());
    const auto inputDtype  = semanticNode->getInput(0)->getElementType();
    const auto outputDtype = semanticNode->getOutput(0)->getElementType();
    HB_ASSERT(isSameBitRepresentation(inputDtype, outputDtype) && semanticNode->is64BitOperands(),
              "Expecting memcpy operands to have same bit representation and 64b operands");

    // Create node sequence: reinterpret_to_32b -> memcpy -> reinterpret_to_64b
    // cast to/from 32b + reshape data tensors
    const synDataType concreteType = (inputDtype == syn_type_int64) ? syn_type_int32 : syn_type_uint32;
    const auto [reinterpretIn, newInput] = reinterpretTensor(semanticNode->getInput(0), true /*isInput*/, concreteType);
    const auto [reinterpretOut, newOutput] =
        reinterpretTensor(semanticNode->getOutput(0), false /*isInput*/, concreteType);
    HB_ASSERT_PTR(newInput);
    HB_ASSERT_PTR(reinterpretIn);
    HB_ASSERT_PTR(newOutput);
    HB_ASSERT_PTR(reinterpretOut);

    // Create the memcpy_nd_32 node
    const auto memcpyNd = getDefaultCopyNdNode({newInput}, {newOutput}, semanticNode->getNodeName(), concreteType);
    HB_ASSERT_PTR(memcpyNd);

    for (const auto& n : {reinterpretIn, memcpyNd, reinterpretOut})
    {
        nodes.emplace_back(n);
    }
    return nodes;
}

NodePtr MemcpyEngineManager::createTpcMemset(const TensorVector& inputs,
                                             const TensorVector& outputs,
                                             const std::string&  name) const
{
    const auto tpcMemset = NodeFactory::createNode(inputs, outputs, nullptr, NodeFactory::tpcMemsetNodeTypeName, name);
    return tpcMemset;
}

NodePtr MemcpyEngineManager::getDefaultCopyNode(const TensorVector& inputs,
                                                const TensorVector& outputs,
                                                const std::string&  name) const
{
    // Default copy node for gaudi1/2 is dmaMemcpy
    // TODO: add assert based on CompilationHalReader that validates GC can use DMA engine
    HB_ASSERT(CompilationHalReader::getHalReader()->isGcDmaSupported(), "Expecting GC to support DMA engine");
    const auto defaultCopy =
        NodeFactory::createNode(inputs, outputs, nullptr, NodeFactory::dmaMemcpyNodeTypeName, name);

    DMAMemcpyNode* const dmaMemcpy = dynamic_cast<DMAMemcpyNode*>(defaultCopy.get());
    HB_ASSERT_PTR(dmaMemcpy);
    dmaMemcpy->setIsCreatedFromSemanticMemcpy();
    return defaultCopy;
}

NodePtr MemcpyEngineManager::getDefaultCopyNdNode(const TensorVector& inputs,
                                                  const TensorVector& outputs,
                                                  const std::string&  name,
                                                  synDataType         dtype) const
{
    HB_ASSERT(dtype != syn_type_na, "Expecting a valid input datatype");
    const auto memcpyNd = NodeFactory::createNode(
        inputs,
        outputs,
        nullptr,
        fmt::format("{}_{}", NodeFactory::memcpyNdNodeTypeName, getDtypeSuffixFromSynDataType(dtype)),
        name);
    return memcpyNd;
}

void MemcpyEngineManager::updateNodeAnnotation(const NodePtr& semanticNode, const NodeList& copySequence) const
{
    HB_ASSERT_PTR(semanticNode);
    // Propagate node annotation from semantic node to concrete copy/set node
    // and bundle info from semantic node to the remaining nodes in the concrete copy/set sequence
    for (const auto& n : copySequence)
    {
        if (isMemcpy(*n) || n->isMemset())
        {
            n->getNodeAnnotation() = semanticNode->getNodeAnnotation();
        }
        else
        {
            n->getNodeAnnotation().bundleInfo = semanticNode->getNodeAnnotation().bundleInfo;
        }
    }
}

// In case of memcpy_nd_64 function should return list of 3 nodes for memcpy_nd_64 operation converted to
// memcpy_nd_32 operation. In this case, we will add 2 nodes before/after the copy to handle the 32/64 cast
NodeList MemcpyEngineManager::createConcreteCopyNode(const HabanaGraph& graph, const NodePtr& semanticNode) const
{
    NodeList          ret;
    const synDataType outputDtype = semanticNode->getOutput(0)->getElementType();
    const synDataType inputDtype =
        (semanticNode->getNumInputsDataTensors() > 0) ? semanticNode->getInput(0)->getElementType() : outputDtype;

    if (!isSameBitRepresentation(inputDtype, outputDtype))
    {
        // different data type for input/output - use tpc cast node if possible
        const auto cast = CastNodeHandler::createCastNode(semanticNode->getInput(0),
                                                          semanticNode->getOutput(0),
                                                          fmt::format("{}_cast", semanticNode->getNodeName()),
                                                          graph.getDeviceId());
        HB_ASSERT_PTR(cast);
        ret.emplace_back(std::move(cast));
    }
    else
    {
        if (!isNDimCopy(semanticNode))
        {
            // use the default copy guid for low rank memcpy
            const auto memcpy =
                getDefaultCopyNode(semanticNode->getInputs(), semanticNode->getOutputs(), semanticNode->getNodeName());
            HB_ASSERT_PTR(memcpy);
            ret.emplace_back(std::move(memcpy));
        }
        else
        {
            // high rank (dim>5) copy
            if (semanticNode->is64BitOperands())
            {
                // lower high rank 64b memcpy to utilize high rank 32b memcpy kernel
                const auto nodes = lower64bMemcpyTo32b(semanticNode);
                HB_ASSERT(!nodes.empty(), "Expecting successful lowering of 64b memcpy to 32b");
                ret.insert(ret.end(), nodes.begin(), nodes.end());
            }
            else
            {
                const auto memcpyNd = getDefaultCopyNdNode(semanticNode->getInputs(),
                                                           semanticNode->getOutputs(),
                                                           semanticNode->getNodeName(),
                                                           inputDtype);
                HB_ASSERT_PTR(memcpyNd);
                ret.emplace_back(std::move(memcpyNd));
            }
        }
    }
    return ret;
}

NodePtr MemcpyEngineManager::createConcreteSetNode(const HabanaGraph& graph, const NodePtr& semanticNode) const
{
    // There is HW bug to support strided memset.
    // Current DMA HW W/A for strided memset has significant performance impact, so it is better to
    // perform strided memset using TPC.
    // Todo [SW-75906] Currently we don't support dynamic shapes in TPC
    NodePtr setNode;
    if (GCFG_GAUDI_MEMSET_HW_WA.value() && !semanticNode->getOutput(0)->isDenseLayout() &&
        !semanticNode->isDynamicShape())
    {
        const auto elementType = semanticNode->getOutput(0)->getElementType();
        HB_ASSERT(graph.getHALReader()->isTPCMemsetSupportedDataType(elementType),
                  "TPC memset not implemented for {}",
                  getStringFromSynDataType(elementType));
        setNode = createTpcMemset(semanticNode->getInputs(), semanticNode->getOutputs(), semanticNode->getNodeName());
    }
    else
    {
        setNode = NodeFactory::createNode(semanticNode->getInputs(),
                                          semanticNode->getOutputs(),
                                          nullptr,
                                          NodeFactory::dmaMemsetNodeTypeName,
                                          semanticNode->getNodeName());
    }
    return setNode;
}

// This validation should be executed after mem copy node was created, and before the mem copy engine is selected,
// and the node type is changed. This allows easily catching all types of mem copy (also casting), and exclude
// other DMA operations such as memset.
bool MemcpyEngineManager::validateMemCopy(const NodePtr& node) const
{
    // Validate mem copy node doesn't copy from SRAM to SRAM as it violates SRAM management decisions (note that
    // goya2 nullifies this function). Asserts in debug, but doesn't fail compilation, as this is recoverable with
    // performance penalty.
    if (node->getNodeType() == Node::TYPE_MEMCOPY)
    {
        // validtae the node has a single input and a single output
        if (node->getNumInputs() != 1)
        {
            LOG_ERR(GC, "Mem copy node should have a single input");
            return false;
        }
        if (node->getNumOutputs() != 1)
        {
            LOG_ERR(GC, "Mem copy node should have a single output");
            return false;
        }

        // validate no SRAM -> SRAM copy excluding copies that "fix" strides
        const auto& input  = node->getInput(0);
        const auto& output = node->getOutput(0);
        if (input->inSram() && output->inSram() && input->isTrivialStrided() && output->isTrivialStrided())
        {
            LOG_ERR(GC, "Mem copy node {} shouldn't copy memory from SRAM to SRAM", node->getNodeName());
            return false;
        }
    }

    return true;
}
