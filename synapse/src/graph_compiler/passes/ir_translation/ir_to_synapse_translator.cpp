#include "data_type_utils.h"
#include "habana_graph.h"
#include "ir_to_synapse_translator.hpp"
#include "node_factory.h"
#include "synapse_node_replacement.hpp"
#include "infer_shape_node.h"
#include "tpc_fuser.h"

using namespace gc_protocol;

bool AbstractIRToSynapseTranslator::createGCNode(const gc_protocol::ProtocolNode& node, NodePtr& createdNode)
{
    std::string_view nodeName(node.name.data, node.name.size);
    std::string_view nodeGuid(node.guid.data, node.guid.size);

    if (node.isShapeManipulationOp)
    {
        // Shape manipulation is a logical node that's used to store info about SIF.
        InferShapeParams params;
        // copy the fuserNode's guid
        memset(params.sifGUID, 0, tpc_lib_api::MAX_NODE_NAME);
        memcpy(params.sifGUID, nodeGuid.data(), tpc_lib_api::MAX_NODE_NAME - 1);
        params.isTpc           = true;
        // From CGUID side it's always a single sif
        params.multiSifInfo    = nullptr;
        // copy sif data that is received from CGUID.
        params.sifMetadata     = node.userParams.nodeParams;
        params.sifMetadataSize = node.userParams.nodeParamsSize;
        createdNode            = NodeFactory::createNode(m_inputs,
                                                         m_outputs,
                                                         &params,
                                                         sizeof(params),
                                                         NodeFactory::inferShapeNodeTypeName,
                                                         nodeName.data());
    }
    else
    {
        createdNode = NodeFactory::createNode(m_inputs,
                                              m_outputs,
                                              node.userParams.nodeParams,
                                              node.userParams.nodeParamsSize,
                                              nodeGuid.data(),
                                              nodeName.data());
    }
    if (!createdNode)
    {
        LOG_WARN(GC_TRANSLATION, "Failed to create node {}, with id {}", nodeName.data(), node.id);
        return false;
    }
    if (m_originalGraph->getInferenceMode() && HabanaGraph::runsOnMME(createdNode))
    {
        updateExpBiasForMmeNode(node, createdNode);
    }
    return true;
}

// Called by handleNode methods.
// First, invoke protocolGraph tensor iteration methods.
// Then create synapse node from the protocol IR node and store it in given pointer.
bool AbstractIRToSynapseTranslator::createGCNodeAndTensors(const ProtocolNode& node, NodePtr& createdNode)
{
    m_inputs.clear();
    m_outputs.clear();
    m_inputPermutations.clear();

    // Create input & output tensors
    if (!m_protocolGraph.foreachInputTensor(node.id, *this) || !m_protocolGraph.foreachOutputTensor(node.id, *this))
    {
        LOG_ERR(GC_TRANSLATION, "Error while iterating node tensors");
        return false;
    }

    LOG_TRACE(GC_TRANSLATION,
              "Synapse Translator - Handling IR node {}, id {}, guid - {}",
              node.name.data,
              node.id,
              node.guid.data);

    if (!createGCNode(node, createdNode)) return false;
    // We can skip actual node creation for Eager
    if (!createdNode) return true;

    // Validate created node
    for (const TensorVector* tensors : {&createdNode->getInputs(), &createdNode->getOutputs()})
    {
        for (const auto& tensor : *tensors)
        {
            if (tensor == nullptr || tensor->getDim() <= SYN_MAX_TENSOR_DIM) continue;

            if (!isTensorDimsValidForNode(tensor, createdNode, true))
            {
                LOG_ERR(GC_COMPLEX_GUID,
                        "Tensor {} of node {} dimensions validation failed when translating from protocolIR to GC node",
                        tensor->getName(),
                        createdNode->getNodeName());
                return false;
            }
        }
    }

    // If m_inputPermutations has at least one non-empty permutation,
    // then save the vector in the node input permutations.
    if (std::any_of(m_inputPermutations.begin(), m_inputPermutations.end(), [](const gc::Permutation& elem) {
            return !elem.isEmpty();
        }))
    {
        LOG_TRACE(GC_TRANSLATION, "Fill node input permutations for {}", createdNode->getNodeName());
        createdNode->getNodeAnnotation().inputPermutations = m_inputPermutations;
    }

    createdNode->setDeterministic(node.useDeterministic);

    LOG_TRACE(GC_TRANSLATION,
              "Finished Creating IR node with name {}, id {} to GC node",
              createdNode->getNodeName(),
              node.id);
    return true;
}

void AbstractIRToSynapseTranslator::setGCTensorName(const ProtocolTensor& irTensor, Tensor& gcTensor)
{
    gcTensor.setName(irTensor.name.data);
}

bool AbstractIRToSynapseTranslator::createGCTensor(const ProtocolTensor& irTensor, TensorVector& tensors)
{
    // below logic for creating GC tensor from IR tensor logic is taken from gc_interface_utils.cpp

    if (auto tensorIter = m_createdTensors.find(irTensor.id); tensorIter != m_createdTensors.end())
    {
        LOG_TRACE(GC_TRANSLATION, "Synapse Translator - gc tensor with id {} already created, using it", irTensor.id);
        tensors.push_back(tensorIter->second);
        return true;
    }

    // Case of new tensor
    LOG_TRACE(GC_TRANSLATION, "Converting IR tensor with name {}, id {} to gc tensor", irTensor.name.data, irTensor.id);

    synDataType tensorDataType = translateTensorDataType(irTensor.elementDataType);
    TensorPtr   gcTensor       = std::make_shared<Tensor>(tensorDataType, "");
    setGCTensorName(irTensor, *gcTensor);
    gcTensor->setProp(synTensorPropName);
    const TStride* strides = nullptr;

    // strides are valid if they are non-zero
    if (irTensor.strides != nullptr &&
        std::all_of(irTensor.strides, irTensor.strides + irTensor.rank + 1, [](unsigned stride) { return stride != 0; }))
    {
        // set strides if they are valid
        LOG_TRACE(GC_TRANSLATION, "Setting strides from IR tensor");
        strides = irTensor.strides;
    }

    gcTensor->reshape(irTensor.rank, irTensor.maxSizes, strides, irTensor.minSizes);
    gcTensor->setProp(synTensorPropGeometryMin);
    gcTensor->setProp(synTensorPropGeometryMax);
    gcTensor->setProp(synTensorPropGeometryDim);
    gcTensor->setProp(synTensorPropDeviceLayout);

    // translate attributes
    if (irTensor.attributes != nullptr)
    {
        if (irTensor.attributes->isInitialized)
        {
            // get tensor data
            HB_ASSERT(irTensor.pData != nullptr, "tensor data is null");
            synDataType bufferDataType = translateTensorDataType(irTensor.attributes->tensorDataType);
            // TODO SW-102046 - copy tensor data in case of new tensor created by MLIR
            // and in case of allocating new data buffer by protocolGraph
            gcTensor->setTensorBuffer(const_cast<void*>(irTensor.pData),
                                      gcTensor->getTotalSizeInBytes(),
                                      bufferDataType,
                                      false);
            gcTensor->setAsStaticParam();
            gcTensor->setProp(synTensorPropHostPtr);
        }
        else if (irTensor.attributes->tensorType == gc_protocol::HOST_TO_DEVICE_TENSOR)
        {
            // reshape above sets device size according to # of elements,
            // but we need to allocate buffer for two sets of data (max/min)
            gcTensor->setDeviceSizeInBytes(gcTensor->getTotalSizeInBytes() * 2);
            gcTensor->bind(new char[gcTensor->getTotalSizeInBytes()], true);
            gcTensor->setAsDataTypeMatchData();
            gcTensor->setProp(synTensorPropHostPtr);
            // ComplexGuid may create tensors without setting their max-dims explicitly relying on max-dims infer
            if (m_originalGraph)
            {
                m_originalGraph->turnOnPredicate(PREDICATE_ID_NODE_CREATED_WITHOUT_OUTPUT_SHAPE);
            }
            else
            {
                LOG_TRACE(GC_TRANSLATION,
                          "can't turn on PREDICATE_ID_NODE_CREATED_WITHOUT_OUTPUT_SHAPE since graph doesn't exist");
            }
        }
        gcTensor->setTensorType(synTensorType(irTensor.attributes->tensorType));
        // update exp bias only for quantization mode and training mode.
        if ((m_originalGraph->getQuantizationEnabled() || !m_originalGraph->getInferenceMode()) &&
            irTensor.attributes->quantizationParams != nullptr &&
            (is8BitFloat(gcTensor->getElementType()) || !gcTensor->isTypeFloat()))
        {
            QuantizationData quantizationParams(tensorDataType);
            quantizationParams.setScale(irTensor.attributes->quantizationParams->scale);
            quantizationParams.setExpBias(irTensor.attributes->quantizationParams->fp8bias);
            gcTensor->setQuantizationParams(quantizationParams);
            gcTensor->setProp(synTensorPropFpQuantMetadata);
        }
    }
    else
    {
        gcTensor->setTensorType(synTensorType::DATA_TENSOR);
    }
    // Translate section info
    if (irTensor.tensorSection != nullptr)
    {
        unsigned gcSectionID = getGCSectionId(*irTensor.tensorSection);
        auto     sectionType = m_sectionTypeMapping[irTensor.tensorSection->id];
        if (sectionType == gc_protocol::SECTION_PERSISTENT)
        {
            gcTensor->setMemoryDescriptor(synMemoryDescriptor(true));
            gcTensor->setMemorySectionID(gcSectionID);
            gcTensor->setMemorySectionOffset(irTensor.tensorSection->offset);
            gcTensor->setProp(synTensorPropSection);
        }
        else if (sectionType == gc_protocol::SECTION_RMW)
        {
            gcTensor->setTensorInSram(); // TODO In Gaudi2+ RMW is possible is HBM, understand if  to change this line
            auto& nonPersistentSectionInfo = gcTensor->getTensorAnnotation().nonPersistentSectionInfo;
            nonPersistentSectionInfo.sectionId.set(gcSectionID);
            nonPersistentSectionInfo.offsetFromBase.set(irTensor.tensorSection->offset);
            gcTensor->setProp(synTensorPropSection);
        }
    }

    if (!gcTensor->isPropsValid())  //  validate tensors properties
    {
        tensors.clear();
        LOG_ERR(GC_TRANSLATION, "Tensor {} has non valid props", gcTensor->getName());
        return false;
    }

    // Add the new tensor to mapping with the ProtocolIR id for future use
    m_createdTensors.emplace(irTensor.id, gcTensor);
    tensors.push_back(std::move(gcTensor));

    LOG_TRACE(GC_TRANSLATION, "Finished creating gc tensor from protocol");

    return true;
}

void AbstractIRToSynapseTranslator::updateExpBiasForMmeNode(const gc_protocol::ProtocolNode& node, NodePtr& createdNode)
{
    MmeExpBias mmeExpBias;
    int        inputIdx   = 0;
    bool       fp8biasSet = false;
    m_protocolGraph.foreachInputTensor(node.id, [&](const ProtocolTensor& irTensor) {
        if (irTensor.attributes->quantizationParams != nullptr)
        {
            LOG_TRACE(GC_TRANSLATION,
                      "MME node {} updating exp bias for input, idx {}, bias {}",
                      createdNode->getNodeName(),
                      inputIdx,
                      irTensor.attributes->quantizationParams->fp8bias);
            mmeExpBias.fp8BiasIn.push_back(irTensor.attributes->quantizationParams->fp8bias);
            fp8biasSet = true;
        }
        inputIdx++;
        return true;
    });
    m_protocolGraph.foreachOutputTensor(node.id, [&](const ProtocolTensor& irTensor) {
        if (irTensor.attributes->quantizationParams != nullptr)
        {
            LOG_TRACE(GC_TRANSLATION,
                      "MME node {} updating exp bias for output, bias {}",
                      createdNode->getNodeName(),
                      irTensor.attributes->quantizationParams->fp8bias);
            mmeExpBias.fp8BiasOut = irTensor.attributes->quantizationParams->fp8bias;
            fp8biasSet            = true;
        }
        return true;
    });
    if (fp8biasSet)
    {
        MMENodePtr mmeNode = std::dynamic_pointer_cast<MmeNode>(createdNode);
        HB_ASSERT(mmeNode, "could not downcast Node to MME Node");
        mmeNode->setMmeExpBias(mmeExpBias);
    }
}

/*
 * IRToSynapseTranslatorBase
 */

IRToSynapseTranslatorBase::IRToSynapseTranslatorBase(const gc_protocol::ProtocolGraph& graphProvider)
: AbstractIRToSynapseTranslator(graphProvider)
{
    m_originalSections.insert(MEMORY_ID_RESERVED_FOR_WORKSPACE);
}

bool IRToSynapseTranslatorBase::handleTensor(const gc_protocol::ProtocolTensor& tensor, TensorVector& tensors)
{
    // Check if tensor with specific id was already in graph, and use it if so
    if (auto tensorIter = m_originalTensors.find(tensor.id); tensorIter != m_originalTensors.end())
    {
        LOG_TRACE(GC_TRANSLATION, "Synapse Translator - gc tensor with id {} already exist in graph, using it", tensor.id);
        tensors.push_back(tensorIter->second);
        return true;
    }
    if (tensor.tensorSection != nullptr && !createProtocolSection(*tensor.tensorSection))
    {
        LOG_ERR(GC_TRANSLATION,
                "Failed to process section's information for tensor {} which is connected to node {}",
                tensor.name.data,
                m_originalNode->getNodeName());
        // TODO: should return false when CGUID resolves issues (https://jira.habana-labs.com/browse/SW-118354)
        //        return false;
    }
    return createGCTensor(tensor, tensors);
}

// Invoked from ProtocolGraph::foreachOutputTensor.
// First search if the tensor exist in original graph and store it if so.
bool IRToSynapseTranslatorBase::handleOutputTensor(const gc_protocol::ProtocolTensor& tensor)
{
    return handleTensor(tensor, m_outputs);
}

// Invoked from ProtocolGraph::foreachInputTensor.
// First search if the tensor exist in original graph and store it if so.
bool IRToSynapseTranslatorBase::handleInputTensor(const gc_protocol::ProtocolTensor& tensor)
{
    // Handle permutations for input tensors only.
    if (tensor.perm != nullptr)
    {
        m_inputPermutations.emplace_back(DimVector(tensor.perm, tensor.perm + tensor.rank));
    }
    else
    {
        m_inputPermutations.emplace_back();
    }
    return handleTensor(tensor, m_inputs);
}

/*
 * Get appropriate GC section id from section mappings that were updated during section preprocessing
 */
unsigned IRToSynapseTranslatorBase::getGCSectionId(const gc_protocol::ProtocolTensorSection_t& irSection)
{
    if (auto sectionIdIter = m_originalSections.find(irSection.id); sectionIdIter != m_originalSections.end())
    {
        LOG_TRACE(GC_TRANSLATION, "Section with id {} is an original section", irSection.id);
        return irSection.id;
    }
    auto sectionIdMapIter = m_sectionIdMapping.find(irSection.id);
    HB_ASSERT(sectionIdMapIter != m_sectionIdMapping.end(), "Unexpected section id {}", irSection.id);
    LOG_TRACE(GC_TRANSLATION, "Section with id {} is mapped to new GC section {}", irSection.id, sectionIdMapIter->second);
    return sectionIdMapIter->second;
}

void IRToSynapseTranslatorBase::storeTensorData(const TensorPtr& tensor)
{
    if (tensor == nullptr)
    {
        return;
    }
    m_originalTensors.emplace(tensor->getId(), tensor);
    // Store original workspace, persistent and RMW section ids, so we can know if new sections were added
    if (auto sectionId = tensor->getMemorySectionID(); sectionId >= MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR)
    {
        m_originalSections.insert(sectionId);
    }
}

bool IRToSynapseTranslatorBase::createProtocolSection(const gc_protocol::ProtocolTensorSection_t& section)
{
    // If a new section id save its type, else make sure the new type matches the old mapped type
    if (auto typeIter = m_sectionTypeMapping.find(section.id); typeIter == m_sectionTypeMapping.end())
    {
        m_sectionTypeMapping.emplace(section.id, section.type);
        LOG_DEBUG(GC_TRANSLATION, "New protocol section with id={} and type={}, saved in map", section.id, section.type);
    }
    else if (typeIter->second != section.type)
    {
        LOG_ERR(GC_TRANSLATION,
                "Section with id={} was mapped with type={}, but it's received now with type={}",
                section.id,
                typeIter->second,
                section.type);
        return false;
    }
    if (section.type == gc_protocol::SECTION_WORKSPACE)
    {
        // Workspace section , do nothing
        return true;
    }
    if (m_originalSections.find(section.id) == m_originalSections.end() &&
        m_sectionIdMapping.find(section.id) == m_sectionIdMapping.end())
    {
        // New section, validate and map to new GC section
        LOG_TRACE(GC_TRANSLATION, "A new section was created by protocolGraph with id {}", section.id);
        if (section.type == gc_protocol::SECTION_PERSISTENT)
        {
            LOG_ERR(GC_TRANSLATION, "Creating new persistent sections is not allowed");
            return false;
        }
        if (section.type == gc_protocol::SECTION_RMW)
        {
            HB_ASSERT(m_originalGraph, "Graph doesn't exist, can't create new RMW section");
            unsigned newGCSectionID =
                m_originalGraph->getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS);
            LOG_TRACE(GC_TRANSLATION,
                      "new section RMW with id {}, offset {} mapped to new GC section with id",
                      section.id,
                      section.offset,
                      newGCSectionID);
            m_sectionIdMapping.emplace(section.id, newGCSectionID);
        }
        else
        {
            LOG_ERR(GC_TRANSLATION, "Unknown section type {}, section id {}", section.type, section.id);
            return false;
        }
    }
    else
    {
        LOG_TRACE(GC_TRANSLATION, "Section with id {} already exist in graph", section.id);
    }
    return true;
}

/*
 * IRToSynapseTranslator
 */

bool IRToSynapseTranslator::startNodeTranslationToSynapse(HabanaGraph* graph, const NodePtr& origNode)
{
    LOG_DEBUG(GC_TRANSLATION, "Starting translation from protocolGraph to Synapse");
    HB_ASSERT(graph != nullptr, "Can't replace nodes while graph is null in synapse on demand translator");

    m_originalGraph       = graph;
    auto replacer = SynapseNodeReplacer(*graph);
    m_synapseNodeReplacer = &replacer;
    // Iterate over node's input and output tensors to store data.
    for (const auto& tensor : origNode->getInputs())
    {
        storeTensorData(tensor);
    }
    for (const auto& tensor : origNode->getOutputs())
    {
        storeTensorData(tensor);
    }
    m_originalNode = origNode;
    // foreachNode will invoke handleNode method implemented below
    if (!m_protocolGraph.foreachNode(*this))
    {
        return false;
    }

    return m_synapseNodeReplacer->replaceNodes(origNode->getId());
}

bool IRToSynapseTranslator::startTranslationToSynapse(HabanaGraph* graph)
{
    LOG_DEBUG(GC_TRANSLATION, "Starting translation from protocolGraph to Synapse");
    HB_ASSERT(graph != nullptr, "Can't replace nodes while graph is null in synapse on demand translator");

    m_originalGraph       = graph;
    auto replacer = SynapseNodeReplacer(*graph);
    m_synapseNodeReplacer = &replacer;
    for (const auto& tensor : graph->getTensors())
    {
        storeTensorData(tensor);
    }
    // foreachNode will invoke handleNode method implemented below
    bool res = m_protocolGraph.foreachNode(*this);
    return res;
}

// Invoked from (protocolGraph::foreachNode).
// After GC node is created, use NodeReplacer class to store replacement info and replace nodes if possible.
bool IRToSynapseTranslator::handleNode(const ProtocolNode& irNode)
{
    NodePtr newNode = nullptr;
    if (!createGCNodeAndTensors(irNode, newNode))
    {
        return false;
    }
    LOG_TRACE(GC_TRANSLATION, "Updating node replacement info");
    // TODO: use irNode data only, when fuser exporter is refactored without translator (meanwhile this data isn't sent)
    // the "Leader" id is the first id in replacedNodeIds field,
    // which is always the same for all nodes in a cluster since the order is kept.
    // it provides quick access to an instance of NodeClusterReplacementInfo.
    unsigned replacedNodeLeaderId = -1;
    if (m_originalNode != nullptr)
    {
        replacedNodeLeaderId = m_originalNode->getId();
        if (ClusteringUtils::canBeClusteredBasic(*m_originalGraph, newNode))
        {
            // Save original complex guid node details to recognize nodes extracted from specific complex guids.
            // It will be used in the tpc fuser for optimizing clustering creation.
            newNode->getNodeAnnotation().originalComplexGuidId = replacedNodeLeaderId;
            const char* originalGuid = m_originalNode->getGUID().c_str();
            strncpy(newNode->getNodeAnnotation().originalComplexGuid, originalGuid, tpc_lib_api::MAX_NODE_NAME - 1);
            LOG_DEBUG(GC_COMPLEX_GUID,
                      "Node {}, id {}, was extracted from complex guid node {}, with id {} and guid {}",
                      newNode->getNodeName(),
                      newNode->getId(),
                      m_originalNode->getNodeName(),
                      replacedNodeLeaderId,
                      originalGuid);
        }
        newNode->getNodeAnnotation().originatedFromCguid = true;
    }
    else if (irNode.replacedNodeIds.data != nullptr)
    {
        replacedNodeLeaderId = irNode.replacedNodeIds.data[0];
    }
    else
    {
        HB_ASSERT(false, "replacedNodeIds is null and leaderID isn't valid");
    }
    if (!m_synapseNodeReplacer->isNodeClusterReplacementInfoExist(replacedNodeLeaderId))
    {
        // TODO: restore when fuser exporter is refactored and translator is disabled (meanwhile this data isn't sent).
        // if (irNode.replacedNodeIds.data != nullptr)
        // {
        //      m_synapseNodeReplacer->createNodeClusterReplacementInfo(irNode);
        // }
        // else
        // {
            m_synapseNodeReplacer->createNodeClusterReplacementInfo(replacedNodeLeaderId);
        // }
    }
    m_synapseNodeReplacer->addSynapseNodeToNodeClusterReplacementInfo(newNode, replacedNodeLeaderId);
    // TODO: restore when fuser exporter is refactored and translator is disabled (meanwhile this data isn't sent).
    // replace nodes if possible
    //        if (m_synapseNodeReplacer->canReplaceNodes(replacedNodeLeaderId) &&
    //            !m_synapseNodeReplacer->replaceNodes(replacedNodeLeaderId))
    //        {
    //            return false;
    //        }

    if (irNode.blockingNodeIds.begin() != irNode.blockingNodeIds.end())
    {
        m_irIdToGCNodeBlockingNodes.emplace(
            irNode.id,
            ir_translation_defs::IdsVector(irNode.blockingNodeIds.begin(), irNode.blockingNodeIds.end()));
    }
    m_createdNodes.push_back(std::move(newNode));
    m_irIdToGCNodeIdx.emplace(irNode.id, m_createdNodes.size() - 1);
    return true;
}

/*
 * IRToSynapseDummyGraphTranslator
 */

bool IRToSynapseDummyGraphTranslator::startTranslationToSynapse(HabanaGraph* graph)
{
    LOG_DEBUG(GC_TRANSLATION, "Starting translation from protocolGraph to Synapse");
    // foreachNode will invoke handleNode method implemented below
    bool res = m_protocolGraph.foreachNode(*this);
    return res;
}

bool IRToSynapseDummyGraphTranslator::handleNode(const ProtocolNode& irNode)
{
    NodePtr newNode = nullptr;
    if (!createGCNodeAndTensors(irNode, newNode))
    {
        return false;
    }
    else if (newNode != nullptr)
    {
        m_createdNodes.push_back(newNode);
        return true;
    }

    return true;
}

bool IRToSynapseDummyGraphTranslator::handleTensor(const ProtocolTensor& tensor, TensorVector& tensors)
{
    return createGCTensor(tensor, tensors);
}
