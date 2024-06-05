#include "graph_comparator.hpp"
#include "habana_graph.h"
#include "log_manager.h"
#include "node_utils.h"
#include "synapse_types_operators.h"

// Auxiliary strings for compared value names
const std::string ID             = "Id";
const std::string Name           = "Name";
const std::string DataType       = "Data type";
const std::string Dim            = "Dim";
const std::string BufferDataType = "Buffer data type";
const std::string NumInputs      = "Num inputs";
const std::string NumOutputs     = "Num outputs";
const std::string ParamsSize     = "Params size";
const std::string Params         = "Params";
const std::string Guid           = "Guid";
const std::string MaxSizes       = "Max sizes";
const std::string MinSizes       = "Min sizes";
const std::string Strides        = "Strides";
const std::string TensorType     = "Tensor type";
const std::string TensorPointers = "Tensor pointers";
const std::string NodePointers   = "Node pointers";
const std::string NodeType       = "Node type";
const std::string SectionId      = "Section ID";
const std::string SectionOffset  = "Section Offset";
const std::string SectionType    = "Section Type";
const std::string Scale          = "Scale";
const std::string ZeroPoint      = "Zero Point";
const std::string IsStaticTensor = "Is Static Tensor";
const std::string Permutations   = "Permutations";


void GraphComparator::compareTensors(const TensorPtr origTensor, const TensorPtr newTensor)
{
    if (!checkPointers(origTensor, newTensor, TensorPointers))
    {
        LOG_WARN(GC_TRANSLATION,
                 "Comparison of tensors failed due to a null tensor. Error - {}", m_errString.str());
        postCompareFailure();
        return;
    }
    if (origTensor == nullptr && newTensor == nullptr) return;

    checkNonEqual(origTensor->getId(), newTensor->getId(), ID);
    checkNonEqual(origTensor->getName(), newTensor->getName(), Name);

    checkEqual(origTensor->getElementType(), newTensor->getElementType(), DataType);
    checkEqual(origTensor->getTensorType(), newTensor->getTensorType(), TensorType);
    checkEqual(origTensor->getDim(), newTensor->getDim(), Dim);
    checkEqual(origTensor->getBufferDataType(), newTensor->getBufferDataType(), BufferDataType);
    checkEqual(origTensor->getMemorySectionID(), newTensor->getMemorySectionID(), SectionId);
    checkEqual(origTensor->getMemorySectionOffset(), newTensor->getMemorySectionOffset(), SectionOffset);
    checkEqual(origTensor->isPersistent(), newTensor->isPersistent(), SectionType);
    checkEqual(origTensor->isPartOfRMWSection(), newTensor->isPartOfRMWSection(), SectionType);
    checkEqual(origTensor->getScale(), newTensor->getScale(), Scale);
    checkEqual(origTensor->getZeroPoint(), newTensor->getZeroPoint(), ZeroPoint);
    checkEqual(origTensor->isStaticParam(), newTensor->isStaticParam(), IsStaticTensor);

    checkEqualBuffer(origTensor->getAllNSizesInElements().data(),
                     newTensor->getAllNSizesInElements().data(),
                     MaxSizes,
                     origTensor->getDim());
    checkEqualBuffer(origTensor->getNMinimalSizesInElements().data(),
                     newTensor->getNMinimalSizesInElements().data(),
                     MinSizes,
                     origTensor->getDim());
    checkEqualBuffer(origTensor->getNStridesInElements().data(),
                     newTensor->getNStridesInElements().data(),
                     Strides,
                     origTensor->getDim());

    // check that new tensor name starts with old tensor name
    if (newTensor->getName().rfind(origTensor->getName()) != 0)
    {
        m_errString << fmt::format("Name mismatch - original - {}, not a prefix of new - {}\n",
                                   origTensor->getName(),
                                   newTensor->getName());
        m_localCompareResult &= false;
    }
    if (!m_localCompareResult)
    {
        LOG_WARN(GC_TRANSLATION,"Compare of original tensor(id:{},name:{}) with new tensor(id:{},name:{}) failed. Errors - {}",
                 origTensor->getId(),
                 origTensor->getName(),
                 newTensor->getId(),
                 newTensor->getName(),
                 m_errString.str());
        postCompareFailure();
    }
}

// Specific compare of Conv node params,
// Needed since conv node rawParams vector contains uninitialized random values in some indexes (42,43,68,69)
void GraphComparator::compareConvNodesParams(const NodePtr origNode, const NodePtr newNode)
{
    const ConvBaseNode* origConvNode = dynamic_cast<const ConvBaseNode*>(origNode.get());
    const ConvBaseNode* newConvNode  = dynamic_cast<const ConvBaseNode*>(newNode.get());
    if (origConvNode == nullptr || newConvNode == nullptr)
    {
        m_errString << fmt::format("Can't compare conv nodes params, at least one the nodes is not a Conv Node");
        m_localCompareResult &= false;
    }
    else
    {
        if (origConvNode->getConvolutionParams() == newConvNode->getConvolutionParams()) return;
        m_errString << fmt::format("SynConvParams mismatch -\n original - {}\n new - {}\n",
                                   MmeNode::synConvolution3DParamsToString(origConvNode->getConvolutionParams()),
                                   MmeNode::synConvolution3DParamsToString(newConvNode->getConvolutionParams()));
        m_localCompareResult &= false;
    }
}

void GraphComparator::compareNodes(const NodePtr origNode, const NodePtr newNode)
{
    if (!checkPointers(origNode, newNode, NodePointers))
    {
        LOG_WARN(GC_TRANSLATION,
                 "Compare of nodes failed due to a null node. Error - {}", m_errString.str());
        postCompareFailure();
        return;
    }
    if (origNode == nullptr && newNode == nullptr) return;

    checkNonEqual(origNode->getId(), newNode->getId(), ID);

    checkEqual(origNode->getGUID(), newNode->getGUID(), Guid);
    checkEqual(origNode->getNumInputs(), newNode->getNumInputs(), NumInputs);
    checkEqual(origNode->getNumOutputs(), newNode->getNumOutputs(), NumOutputs);
    checkEqual(origNode->getParamsRawData().size(), newNode->getParamsRawData().size(), ParamsSize);
    checkEqual(origNode->getNodeType(), newNode->getNodeType(), NodeType);
    PermutationVector& origPermutations = origNode->getNodeAnnotation().inputPermutations;
    PermutationVector& newPermutations  = newNode->getNodeAnnotation().inputPermutations;

    if (origPermutations.size() != newPermutations.size())
    {
        m_errString << fmt::format(
            "Input permutations mismatch - original permutations size = {}, new permutations size = {}\n",
            origPermutations.size(),
            newPermutations.size());
        m_localCompareResult &= false;
    }
    else
    {
        for (unsigned i = 0; i < origPermutations.size(); i++)
        {
            if (origPermutations[i] != newPermutations[i])
            {
                m_errString << fmt::format("Input permutations mismatch for tensor index={} - "
                                           "{} != {}\n",
                                           i,
                                           origPermutations[i].toString(),
                                           newPermutations[i].toString());
                m_localCompareResult &= false;
            }
        }
    }

    if (origNode->getNodeType() == Node::TYPE_CONVOLUTION)
    {
        // conv node params need special compare
        compareConvNodesParams(origNode, newNode);
    }
    else
    {
        checkEqualBuffer(origNode->getParamsRawData().data(),
                         newNode->getParamsRawData().data(),
                         Params,
                         origNode->getParamsRawData().size());
    }

    if (!m_localCompareResult)
    {
        LOG_WARN(GC_TRANSLATION,
                 "Compare of original node(id:{},name:{}) with new node(id:{},name:{}) failed. Errors - {}",
                 origNode->getId(),
                 origNode->getNodeName(),
                 newNode->getId(),
                 newNode->getNodeName(),
                 m_errString.str());
        postCompareFailure();
    }
    //  lambda for iterating input/output tensors and run tensor compare
    auto iterateTensors = [&](auto tensorIterator, bool isInput) {
        auto originalTensors = isInput ? origNode->getInputs() : origNode->getOutputs();
        for (auto origTensor : originalTensors)
        {
            auto newTensor = *tensorIterator++;
            compareTensors(origTensor, newTensor);
        }
    };

    auto newInputsIterator = newNode->getInputs().begin();
    iterateTensors(newInputsIterator, true);

    auto newOutputsIterator = newNode->getOutputs().begin();
    iterateTensors(newOutputsIterator, false);
}

void GraphComparator::postCompareFailure()
{
    m_globalCompareResult &= false;
    // reset before starting next compare
    m_localCompareResult = true;
    m_errString.str("");
    m_errString.clear(); // clear internal flags
}

bool GraphComparator::compareGraphs(const HabanaGraph& originalGraph, HabanaGraph& newGraph)
{
    LOG_TRACE(GC_TRANSLATION, "Comparing original graph to new graph");

    unsigned origGraphNodeNum = originalGraph.getNumNodes();
    unsigned newGraphNodeNum  = newGraph.getNumNodes();
    if (origGraphNodeNum != newGraphNodeNum)
    {
        LOG_WARN(GC_TRANSLATION,"Nodes number in original graph {} is different from nodes number in new graph - {}",
                 origGraphNodeNum,
                 newGraphNodeNum);
        m_globalCompareResult &= false;
    }
    unsigned origGraphTensorNum = originalGraph.getTensors().size();
    unsigned newGraphTensorNum  = newGraph.getTensors().size();
    if (origGraphTensorNum != newGraphTensorNum)
    {
        LOG_WARN(GC_TRANSLATION,
                 "Tensors number in original graph {} is different from tensors number in new graph - {}",
                 origGraphTensorNum,
                 newGraphTensorNum);
        m_globalCompareResult &= false;
    }
    // compare new nodes to original nodes - should be the same except IDs
    auto oldGraphNodes = originalGraph.getExeSortedNodes();
    auto newGraphNodes = newGraph.getExeSortedNodes();
    auto newNodesIterator = newGraphNodes.begin();
    for (auto origNode : oldGraphNodes)
    {
        auto newNode = *newNodesIterator++;
        compareNodes(origNode, newNode);
    }
    if (newGraphNodeNum <= 100 && !originalGraph.isomorphicTo(newGraph))
    {
        // The isomorphic function explodes for large graphs, check why
        LOG_WARN(GC_TRANSLATION, "Original graph is not isomorphic to new graph");
        m_globalCompareResult &= false;
    }
    LOG_DEBUG(GC_TRANSLATION,"Finished comparing original graph to new graph, result - {}", m_globalCompareResult);
    return m_globalCompareResult;
}

template<typename T>
void GraphComparator::checkEqual(T origValue, T newValue, const std::string& valueName)
{
    if (origValue == newValue) return;
    m_errString << fmt::format("{} mismatch - original - {}, new - {}\n", valueName, origValue, newValue);
    m_localCompareResult &= false;
}

template<typename T>
void GraphComparator::checkNonEqual(T origValue, T newValue, const std::string& valueName)
{
    if (origValue != newValue) return;
    m_errString << fmt::format("{} match - original - {}, new - {}\n", valueName, origValue, newValue);
    m_localCompareResult &= false;
}

template<typename T>
void GraphComparator::checkEqualBuffer(T* origBuffer, T* newBuffer, const std::string& bufferName, unsigned bufferSize)
{
    if (origBuffer != nullptr && newBuffer == nullptr)
    {
        m_errString << fmt::format("{} mismatch - new buffer is null\n", bufferName);
        m_localCompareResult &= false;
        return;
    }
    if (origBuffer == nullptr && newBuffer != nullptr)
    {
        m_errString << fmt::format("can't compare {} - original buffer is null\n", bufferName);
        m_localCompareResult &= false;
        return;
    }
    if (origBuffer == nullptr && newBuffer == nullptr) return; // valid case

    for (unsigned i = 0; i < bufferSize; i++)
    {
        if (origBuffer[i] != newBuffer[i])
        {
            m_errString << fmt::format("{} mismatch - origBuffer[{}]={}, newBuffer[{}]={}\n",
                                       bufferName,
                                       i,
                                       origBuffer[i],
                                       i,
                                       newBuffer[i]);
            m_localCompareResult &= false;
        }
    }
}

template<typename T>
bool GraphComparator::checkPointers(std::shared_ptr<T> origPointer,
                                    std::shared_ptr<T> newPointer,
                                    const std::string& pointerName)
{
    if (origPointer.get() != nullptr && newPointer.get() == nullptr)
    {
        m_errString << fmt::format("{} mismatch - new pointer is null\n", pointerName);
        m_localCompareResult &= false;
        return false;
    }
    if (origPointer.get() == nullptr && newPointer.get() != nullptr)
    {
        m_errString << fmt::format("can't compare {} - original pointer is null\n", pointerName);
        m_localCompareResult &= false;
        return false;
    }
    return true;
}