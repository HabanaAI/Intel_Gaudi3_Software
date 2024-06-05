#pragma once

#include "gc_protocol.hpp"
#include "tpc_node.h"
#include "define_synapse_common.hpp"
#include "ir_to_synapse_translator_defs.hpp"

using namespace gc_protocol;

inline void createProtocolSection(gc_protocol::ProtocolTensorSection_t* section, const TensorPtr& gcTensor)
{
    HB_ASSERT(gcTensor != nullptr, "Tensor is null");
    if (gcTensor->isPersistent())
    {
        section->type   = gc_protocol::SECTION_PERSISTENT;
        section->offset = gcTensor->getMemorySectionOffset();
        section->id     = gcTensor->getMemorySectionID();
    }
    else if (gcTensor->isPartOfRMWSection())
    {
        section->type          = gc_protocol::SECTION_RMW;
        const auto& annotation = gcTensor->getTensorAnnotation();
        section->offset        = annotation.nonPersistentSectionInfo.offsetFromBase.value();
        section->id            = annotation.nonPersistentSectionInfo.sectionId.value();
    }
    else
    {
        section->type   = gc_protocol::SECTION_WORKSPACE;
        section->offset = 0;  // for workspace sections offset is 0;
        section->id     = MEMORY_ID_RESERVED_FOR_WORKSPACE;
    }
    LOG_DEBUG(GC_TRANSLATION,
              "Protocol IR section was created with id: {}, type: {} with offset: {} ",
              section->id,
              section->type,
              section->offset);
}

inline void
createProtocolAttributes(ProtocolTensorAttributes* attributes, const TensorPtr& gcTensor, bool isGraphOutput)
{
    HB_ASSERT(gcTensor != nullptr, "attributes and/or tensor are null");
    attributes->isNotNeeded    = gcTensor->isNotNeeded();
    attributes->isInitialized  = gcTensor->isStaticParam() && gcTensor->isBound();
    synTensorType gcTensorType = gcTensor->getTensorType();
    // synTensorType enum values are equal to ProtocolTensorType_t values
    attributes->tensorType = (ProtocolTensorType_t)gcTensorType;
    if (gcTensor->isStaticParam())
    {
        attributes->tensorDataType = toTpcLibDataType(gcTensor->getBufferDataType());
    }
    else
    {
        // Irrelevant in this case, filling with default value (has to be filled).
        attributes->tensorDataType = tpc_lib_api::NUM_DATATYPES;
    }
    attributes->isGraphOutput = isGraphOutput;
    LOG_DEBUG(GC_TRANSLATION,
              "Protocol IR attribute was created with tensorType: {}, isNotNeeded: {} isInitialized: {}"
              "tensorDataType: {} isGraphOutput: {}",
              attributes->tensorType,
              attributes->isNotNeeded,
              attributes->isInitialized,
              attributes->tensorDataType,
              attributes->isGraphOutput);
}

inline void createProtocolQuantizationParams(gc_protocol::ProtocolTensorQuantizationParams* protocolQuantParams,
                                             const TensorPtr&                               gcTensor)
{
    auto       isFP8               = is8BitFloat(gcTensor->getElementType());
    const auto gcQuantParams       = gcTensor->getQuantizationParams();
    protocolQuantParams->zeroPoint = gcQuantParams.zp();
    if (isFP8)
    {
        protocolQuantParams->fp8bias = gcQuantParams.expBias();
    }
    protocolQuantParams->scale = gcQuantParams.scale();

    if (isFP8)
    {
        LOG_DEBUG(GC_TRANSLATION,
                  "Protocol IR QuantizationParam was created with fp8ias: {} scale: {}",
                  protocolQuantParams->fp8bias,
                  protocolQuantParams->scale);
    }
    else
    {
        LOG_DEBUG(GC_TRANSLATION,
                  "Protocol IR QuantizationParam was created with zeroPoint: {} scale: {}",
                  protocolQuantParams->zeroPoint,
                  protocolQuantParams->scale);
    }
}

/*
 * Helper class for validating Protocol IR graph extracted by CGUID lib
 */
class ProtocolIRGraphValidator
{
public:
    ProtocolIRGraphValidator(const ProtocolGraph*                     pProtocolGraph,
                             const ir_translation_defs::NewTensorMap& externalGCTensors);
    bool validateNodes();
    bool validateRMWTensorsForNode(synNodeId protocolNodeId);  // validate RMW tensors of node are in same section
    bool validateGraphInputsAndOutputs();  // run several validations on input & output tensors
    // validate new RMW output is alias to other external
    bool validateNewRMWOutputTensor(const ProtocolTensor* protocolTensor);
    // validate new persistent output is alias to other external output
    bool validateNewPersistentOutputTensor(const ProtocolTensor* newPersistentOutput);

private:
    // add tensors to member data structs used later for validation
    void storeTensor(synNodeId protocolNodeId, const ProtocolTensor& tensor, bool isInput);
    // store tensors and nodes in data members for later validation
    void prepareForValidation();
    /* each of the below members used for a different validation */
    const ProtocolGraph* m_pProtocolGraph;  // the Protocol graph extracted by external lib
    // original inputs & outputs of GC node
    const ir_translation_defs::NewTensorMap& m_externalGCTensors;
    std::set<synNodeId>          m_externalGCPersistentOutputTensorsIds;  // original persistent outputs ids of GC node
    std::unordered_set<uint64_t> m_extractedTensorIDs;  // IDs of all Protocol tensors extracted by complexGuid
    // Protocol node to its RMW tensors
    std::unordered_map<synNodeId, std::unordered_set<const ProtocolTensor*>> m_nodesRMWTensors;
    // new input Protocol tensors of Protocol graph
    std::unordered_map<uint64_t, const ProtocolTensor*> m_newInputTensors;
    // new output Protocol tensors of Protocol graph
    std::unordered_map<uint64_t, const ProtocolTensor*> m_newOutputTensors;
};
