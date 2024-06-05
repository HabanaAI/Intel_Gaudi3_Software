#pragma once

#include "tpc_fuser.h"

class HabanaGraph;
typedef PermutationVector PermutationsVector;

void createFuserTensorAttributes(pFuserTensorAttributes attributes, TensorPtr gcTensor);

void createFuserSection(pFuserSection section, TensorPtr gcTensor);

/* definitions of internal template methods implemented in corresponding cpp file.
 * This is done to avoid implementation in hpp file.
 * Internal template methods can't be called directly from other cpp files (won't compile).
 * They should be called by using the wrapping methods defined after them */
template<typename NodeVersion>
void internalCreateFuserNode(const HabanaGraph& g, std::shared_ptr<NodeVersion> fuserNode, const NodePtr node);

template<typename TensorVersion>
void internalCreateGCTensor(TensorPtr gcTensor, std::shared_ptr<TensorVersion> fuserTensor);

template<typename TensorVersion>
void internalCreateFuserTensor(std::shared_ptr<TensorVersion> fuserTensor, TensorPtr gcTensor);

template<typename NodeVersion, typename TensorVersion, typename EdgeVersion>
void internalCreateFuserNodeEdgesAndTensors(std::shared_ptr<NodeVersion> fuserNode,
                                            TensorVector&                tensors,
                                            bool                         isInputEdge,
                                            PermutationsVector&          permutations);

template<typename TensorVersion>
uint64_t internalGetCommonTensorSizeInBytes(std::shared_ptr<TensorVersion> fuserTensor);

template<typename TensorVersion>
void internalSetCommonTensorPermutation(std::shared_ptr<TensorVersion> fuserTensor, gc::Permutation& permutation);

/* Wrapper template methods calling the internal methods */
template<typename NodeVersion>
void createFuserNode(const HabanaGraph& g, std::shared_ptr<NodeVersion> fuserNode, const NodePtr node)
{
    internalCreateFuserNode(g, fuserNode, node);
}

template<typename TensorVersion>
void createGCTensor(TensorPtr gcTensor, std::shared_ptr<TensorVersion> fuserTensor)
{
    internalCreateGCTensor(gcTensor, fuserTensor);
}

template<typename TensorVersion>
void createFuserTensor(std::shared_ptr<TensorVersion> fuserTensor, TensorPtr gcTensor)
{
    internalCreateFuserTensor(fuserTensor, gcTensor);
}

template<typename NodeVersion, typename TensorVersion, typename EdgeVersion>
void createFuserNodeEdgesAndTensors(std::shared_ptr<NodeVersion> fuserNode,
                                    TensorVector&                tensors,
                                    bool                         isInputEdge,
                                    PermutationsVector&          permutations)
{
    internalCreateFuserNodeEdgesAndTensors<NodeVersion, TensorVersion, EdgeVersion>(fuserNode,
                                                                                    tensors,
                                                                                    isInputEdge,
                                                                                    permutations);
}

template<typename TensorVersion>
uint64_t getCommonTensorSizeInBytes(std::shared_ptr<TensorVersion> fuserTensor)
{
    return internalGetCommonTensorSizeInBytes(fuserTensor);
}

template<typename TensorVersion>
void setCommonTensorPermutations(std::shared_ptr<TensorVersion> fuserTensor, gc::Permutation& permutation)
{
    internalSetCommonTensorPermutation(fuserTensor, permutation);
}

/*
 * Helper class for validating Common IR graph extracted by complexGuid lib
 */
template<typename GraphVersion, typename TensorVersion, typename EdgeVersion>
class CommonIRGraphValidator
{
public:
    CommonIRGraphValidator(const GraphVersion*                            pCommonGraph,
                           const std::unordered_map<unsigned, TensorPtr>& externalGCTensors);
    void prepareForValidation();  // store tensors and nodes in data members for later validation
    bool validateNodes();
    bool validateRMWTensorsForNode(FuserNodeId commonNodeId);  // validate RMW tensors of node are in same section
    bool validateGraphInputsAndOutputs();                      // run several validations on input & output tensors
    // validate new RMW output is alias to other external
    bool validateNewRMWOutputTensor(std::shared_ptr<TensorVersion> commonTensor);
    // validate new persistent output is alias to other external output
    bool validateNewPersistentOutputTensor(std::shared_ptr<TensorVersion> newPersistentOutput);

private:
    // add tensors to member data structs used later for validation
    void storeTensor(EdgeVersion& commonEdge, FuserNodeId commonNodeId, bool isInput);

    const GraphVersion* m_pCommonGraph;  // the Common graph extracted by external lib
    /* each of the below members used for a different validation */
    const std::unordered_map<unsigned, TensorPtr>& m_externalGCTensors;  // original inputs & outputs of GC node
    std::set<FuserTensorId> m_externalGCPersistentOutputTensorsIds;      // original persistent outputs ids of GC node
    std::unordered_set<FuserTensorId> m_extractedTensorIDs;  // IDs of all Common tensors extracted by complexGuid
    // Common node to its RMW tensors
    std::unordered_map<FuserNodeId, std::unordered_set<std::shared_ptr<TensorVersion>>> m_nodesRMWTensors;
    // new input Common tensors of Common graph
    std::unordered_map<FuserTensorId, std::shared_ptr<TensorVersion>> m_newInputTensors;
    // new output Common tensors of Common graph
    std::unordered_map<FuserTensorId, std::shared_ptr<TensorVersion>> m_newOutputTensors;
};

/*
 * CommonIRGraphValidator
 */
template<typename GraphVersion, typename TensorVersion, typename EdgeVersion>
CommonIRGraphValidator<GraphVersion, TensorVersion, EdgeVersion>::CommonIRGraphValidator(
    const GraphVersion*                            pCommonGraph,
    const std::unordered_map<unsigned, TensorPtr>& externalGCTensors)
: m_pCommonGraph(pCommonGraph), m_externalGCTensors(externalGCTensors)
{
}
template<typename GraphVersion, typename TensorVersion, typename EdgeVersion>
void CommonIRGraphValidator<GraphVersion, TensorVersion, EdgeVersion>::prepareForValidation()
{
    // store tensors in internal data members for later validation
    for (auto& node : m_pCommonGraph->nodes)
    {
        for (auto& edge : node->inputEdges)
        {
            storeTensor(edge, node->uniqueIdentifier, true);
        }
        for (auto& edge : node->outputEdges)
        {
            storeTensor(edge, node->uniqueIdentifier, false);
        }
    }
    LOG_TRACE(GC_TPC_FUSER, "Number of new Common graph inputs/outputs tensors - {}", m_newInputTensors.size());
}

template<typename GraphVersion, typename TensorVersion, typename EdgeVersion>
void CommonIRGraphValidator<GraphVersion, TensorVersion, EdgeVersion>::storeTensor(EdgeVersion& commonEdge,
                                                                                   FuserNodeId  commonNodeId,
                                                                                   bool         isInput)
{
    if (commonEdge.tensor == nullptr) return;
    m_extractedTensorIDs.insert(commonEdge.tensor->uniqueIdentifier);
    if (commonEdge.tensor->section.type == gcapi::SECTION_RMW)
    {
        // store for RMW validation
        m_nodesRMWTensors[commonNodeId].insert(commonEdge.tensor);
    }

    if (commonEdge.targetNode.expired())  // empty pointer means graph input or output
    {
        if (m_externalGCTensors.find(commonEdge.tensor->uniqueIdentifier) == m_externalGCTensors.end())  // new tensor
        {  // store for new inputs & outputs validation
            if (isInput)
            {
                m_newInputTensors.insert({commonEdge.tensor->uniqueIdentifier, commonEdge.tensor});
            }
            else
            {
                m_newOutputTensors.insert({commonEdge.tensor->uniqueIdentifier, commonEdge.tensor});
            }
        }
        else if (!isInput)  // store external GC outputs ids for later validation of new persistent tensors
        {
            m_externalGCPersistentOutputTensorsIds.insert(commonEdge.tensor->uniqueIdentifier);
        }
    }
}

template<typename GraphVersion, typename TensorVersion, typename EdgeVersion>
bool CommonIRGraphValidator<GraphVersion, TensorVersion, EdgeVersion>::validateNodes()
{
    for (const auto& node : m_pCommonGraph->nodes)
    {
        if (!validateRMWTensorsForNode(node->uniqueIdentifier)) return false;
    }
    return true;
}

template<typename GraphVersion, typename TensorVersion, typename EdgeVersion>
bool CommonIRGraphValidator<GraphVersion, TensorVersion, EdgeVersion>::validateRMWTensorsForNode(
    FuserNodeId commonNodeId)
{
    // validate all RMW tensors of same node are in same section
    FuserSectionId testRMWSectionID    = 0;  // can't be a real RMW section id, it's reserved for workspace section
    FuserSectionId currentRMWSectionID = testRMWSectionID;
    for (const auto& tensor : m_nodesRMWTensors[commonNodeId])
    {
        if (currentRMWSectionID == testRMWSectionID)  // first RMW section encountered
        {
            currentRMWSectionID = tensor->section.id;
        }
        else  // we already encountered a RMW section
        {
            if (tensor->section.id != currentRMWSectionID)
            {
                LOG_ERR(GC_TPC_FUSER, "Extracted node has tensors in different RMW sections");
                return false;
            }
        }
    }
    return true;
}

template<typename GraphVersion, typename TensorVersion, typename EdgeVersion>
bool CommonIRGraphValidator<GraphVersion, TensorVersion, EdgeVersion>::validateGraphInputsAndOutputs()
{
    // validate that all original tensors were received, tensors that aren't needed may be omitted
    for (auto& pair : m_externalGCTensors)
    {
        bool tensorIsNeeded = !pair.second->isNotNeeded();
        if (tensorIsNeeded)
        {
            if (m_extractedTensorIDs.find(pair.first) == m_extractedTensorIDs.end())
            {
                LOG_ERR(GC_TPC_FUSER,
                        "Original tensor with id {} is needed but not found in extracted graph",
                        pair.first);
                return false;
            }
        }
    }
    /*
     * validate that all new Common tensors that are graph input or output fill one of below conditions :
     * 1. New output RMW tensor that is alias to other external RMW tensor
     * 2. New input tensors with initialized static data
     * 3. New output persistent tensor that is alias to other external persistent output tensor
     */
    for (const auto& tensor : m_newOutputTensors)
    {
        if (tensor.second->section.type == gcapi::SECTION_RMW && !validateNewRMWOutputTensor(tensor.second))
        {
            LOG_ERR(GC_TPC_FUSER,
                    "New output tensor with id {} is not a valid RMW tensor",
                    tensor.second->uniqueIdentifier);
            return false;
        }
        if (tensor.second->section.type == gcapi::SECTION_PERSISTENT &&
            !validateNewPersistentOutputTensor(tensor.second))
        {
            LOG_ERR(GC_TPC_FUSER,
                    "New output tensor with id {} is not a valid persistent tensor",
                    tensor.second->uniqueIdentifier);
            return false;
        }
    }
    for (const auto& tensor : m_newInputTensors)
    {
        if (!tensor.second->attributes.isInitialized || tensor.second->pData == nullptr)
        {
            LOG_ERR(GC_TPC_FUSER,
                    "New input tensor with id {} doesn't have initialized static data",
                    tensor.second->uniqueIdentifier);
            return false;
        }
    }
    return true;
}

template<typename GraphVersion, typename TensorVersion, typename EdgeVersion>
bool CommonIRGraphValidator<GraphVersion, TensorVersion, EdgeVersion>::validateNewRMWOutputTensor(
    std::shared_ptr<TensorVersion> newOutputRMWTensor)
{
    // validate that tensor is alias of an original external tensor
    for (const auto& externalTensor : m_externalGCTensors)
    {
        // t1 alias of t2 means - t1 , t2 in same section and offset, and t1 fits in t2
        if (externalTensor.second->isPartOfRMWSection() &&
            newOutputRMWTensor->section.id ==
                externalTensor.second->getTensorAnnotation().nonPersistentSectionInfo.sectionId.value() &&
            newOutputRMWTensor->section.offset ==
                externalTensor.second->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.value() &&
            getCommonTensorSizeInBytes(newOutputRMWTensor) <= externalTensor.second->getTotalSizeInBytes())
        {
            return true;
        }
    }
    return false;
}

template<typename GraphVersion, typename TensorVersion, typename EdgeVersion>
bool CommonIRGraphValidator<GraphVersion, TensorVersion, EdgeVersion>::validateNewPersistentOutputTensor(
    std::shared_ptr<TensorVersion> newPersistentOutput)
{
    // validate that tensor is alias of an original external output tensor
    for (const auto& tensorId : m_externalGCPersistentOutputTensorsIds)
    {
        auto externalPersistentOutputTensor = m_externalGCTensors.at(tensorId);
        // t1 alias of t2 means - t1 , t2 in same section and offset, and t1 fits in t2
        if (newPersistentOutput->section.id == externalPersistentOutputTensor->getMemorySectionID() &&
            newPersistentOutput->section.offset == externalPersistentOutputTensor->getMemorySectionOffset() &&
            getCommonTensorSizeInBytes(newPersistentOutput) <= externalPersistentOutputTensor->getTotalSizeInBytes())
        {
            return true;
        }
    }
    return false;
}
