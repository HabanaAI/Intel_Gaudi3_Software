#include "gc_protocol_utils.hpp"

/*
 * ProtocolIRGraphValidator
 */
ProtocolIRGraphValidator::ProtocolIRGraphValidator(const ProtocolGraph*                     pProtocolGraph,
                                                   const ir_translation_defs::NewTensorMap& externalGCTensors)
: m_pProtocolGraph(pProtocolGraph), m_externalGCTensors(externalGCTensors)
{
    prepareForValidation();
}

void ProtocolIRGraphValidator::prepareForValidation()
{
    // store tensors in internal data members for later validation
    m_pProtocolGraph->foreachNode([&](const ProtocolNode& node) {
        m_pProtocolGraph->foreachInputTensor(node.id, [&](const ProtocolTensor& tensor) {
            storeTensor(node.id, tensor, true);
            return true;
        });
        m_pProtocolGraph->foreachOutputTensor(node.id, [&](const ProtocolTensor& tensor) {
            storeTensor(node.id, tensor, false);
            return true;
        });
        return true;
    });
    LOG_TRACE(GC_TPC_FUSER, "Number of new Protocol graph inputs/outputs tensors - {}", m_newInputTensors.size());
}

void ProtocolIRGraphValidator::storeTensor(synNodeId protocolNodeId, const ProtocolTensor& tensor, bool isInput)
{
    // If already handled, skip.
    if (!m_extractedTensorIDs.insert(tensor.id).second) return;

    if (tensor.tensorSection->type == gc_protocol::SECTION_RMW)
    {
        // store for RMW validation
        m_nodesRMWTensors[protocolNodeId].insert(&tensor);
    }

    if (m_externalGCTensors.find(tensor.id) == m_externalGCTensors.end())  // new tensor
    {                                                                      // store for new inputs & outputs validation
        if (isInput)
        {
            m_newInputTensors.insert({tensor.id, &tensor});
        }
        else
        {
            m_newOutputTensors.insert({tensor.id, &tensor});
        }
    }
    else if (!isInput)  // store external GC outputs ids for later validation of new persistent tensors
    {
        m_externalGCPersistentOutputTensorsIds.insert(tensor.id);
    }
}

bool ProtocolIRGraphValidator::validateNodes()
{
    return m_pProtocolGraph->foreachNode([&](const ProtocolNode& node) {
        return validateRMWTensorsForNode(node.id);
    });
}

bool ProtocolIRGraphValidator::validateRMWTensorsForNode(synNodeId protocolNodeId)
{
    // validate all RMW tensors of same node are in same section
    synSectionId testRMWSectionID    = 0;  // can't be a real RMW section id, it's reserved for workspace section
    synSectionId currentRMWSectionID = testRMWSectionID;
    for (const auto& tensor : m_nodesRMWTensors[protocolNodeId])
    {
        if (currentRMWSectionID == testRMWSectionID)  // first RMW section encountered
        {
            currentRMWSectionID = tensor->tensorSection->id;
        }
        else  // we already encountered a RMW section
        {
            if (tensor->tensorSection->id != currentRMWSectionID)
            {
                LOG_ERR(GC_TPC_FUSER, "Extracted node has tensors in different RMW sections");
                return false;
            }
        }
    }
    return true;
}

bool ProtocolIRGraphValidator::validateGraphInputsAndOutputs()
{
    // validate that all original tensors were received, tensors that aren't needed may be omitted
    for (const auto& pair : m_externalGCTensors)
    {
        bool tensorIsNeeded = !pair.second->isNotNeeded();
        if (tensorIsNeeded)
        {
            if (m_extractedTensorIDs.count(pair.first))
            {
                LOG_ERR(GC_TPC_FUSER,
                        "Original tensor with id {} is needed but not found in extracted graph",
                        pair.first);
                return false;
            }
        }
    }
    /*
     * validate that all new Protocol tensors that are graph input or output fill one of below conditions :
     * 1. New output RMW tensor that is alias to other external RMW tensor
     * 2. New input tensors with initialized static data
     * 3. New output persistent tensor that is alias to other external persistent output tensor
     */
    for (const auto& tensor : m_newOutputTensors)
    {
        if (tensor.second->tensorSection->type == gc_protocol::SECTION_RMW &&
            !validateNewRMWOutputTensor(tensor.second))
        {
            LOG_ERR(GC_TPC_FUSER, "New output tensor with id {} is not a valid RMW tensor", tensor.second->id);
            return false;
        }
        if (tensor.second->tensorSection->type == gc_protocol::SECTION_PERSISTENT &&
            !validateNewPersistentOutputTensor(tensor.second))
        {
            LOG_ERR(GC_TPC_FUSER, "New output tensor with id {} is not a valid persistent tensor", tensor.second->id);
            return false;
        }
    }
    for (const auto& tensor : m_newInputTensors)
    {
        if (!tensor.second->attributes->isInitialized || tensor.second->pData == nullptr)
        {
            LOG_ERR(GC_TPC_FUSER,
                    "New input tensor with id {} doesn't have initialized static data",
                    tensor.second->id);
            return false;
        }
    }
    return true;
}

bool ProtocolIRGraphValidator::validateNewRMWOutputTensor(const ProtocolTensor* newOutputRMWTensor)
{
    // validate that tensor is alias of an original external tensor
    for (const auto& externalTensor : m_externalGCTensors)
    {
        // t1 alias of t2 means - t1 , t2 in same section and offset, and t1 fits in t2
        if (externalTensor.second->isPartOfRMWSection() &&
            newOutputRMWTensor->tensorSection->id ==
                externalTensor.second->getTensorAnnotation().nonPersistentSectionInfo.sectionId.value() &&
            newOutputRMWTensor->tensorSection->offset ==
                externalTensor.second->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.value())
        {
            return true;
        }
    }
    return false;
}

bool ProtocolIRGraphValidator::validateNewPersistentOutputTensor(const ProtocolTensor* newPersistentOutput)
{
    // validate that tensor is alias of an original external output tensor
    for (const auto& tensorId : m_externalGCPersistentOutputTensorsIds)
    {
        auto externalPersistentOutputTensor = m_externalGCTensors.at(tensorId);
        // t1 alias of t2 means - t1 , t2 in same section and offset, and t1 fits in t2
        if (newPersistentOutput->tensorSection->id == externalPersistentOutputTensor->getMemorySectionID() &&
            newPersistentOutput->tensorSection->offset == externalPersistentOutputTensor->getMemorySectionOffset())
        {
            return true;
        }
    }
    return false;
}
