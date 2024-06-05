#include "habana_graph.h"

#include "node_factory.h"
#include "tpc_node.h"
#include "utils.h"
#include "cast_nodes_handler.h"

#include "node_annotation.h"
#include "graph_editor.h"
#include "data_type_utils.h"
#include <memory>

pTensor createCastTensor(pTensor castFrom, synDataType toType, const std::string& name)
{
    pTensor castTensor = castFrom->clone(false, false);

    if (isQuantDtype(toType) && !castTensor->isQuantizationParamsExist(toType))
    {
        LOG_WARN(GC,
                 "Tensor {} requires cast to type {} but has no quantization params for that type",
                 castFrom->getName(),
                 getStringFromSynDataType(toType));
    }

    castTensor->resetAliasing();
    castTensor->maskOutput();

    castTensor->setElementType(toType);

    castTensor->setName(name);
    return castTensor;
}

static std::pair<std::shared_ptr<ns_CastKernel::ParamsV3>, unsigned>
getCastParams(const TensorPtr& inputTensor, const TensorPtr& outputTensor, tpc_lib_api::DeviceId deviceId)
{
    std::shared_ptr<ns_CastKernel::ParamsV3> paramsPtr(nullptr);
    unsigned                                 paramsSize = 0;
    if ((deviceId == tpc_lib_api::DEVICE_ID_GAUDI2 || deviceId == tpc_lib_api::DEVICE_ID_GAUDI3) &&
        (is8BitFloat(inputTensor->getElementType()) || is8BitFloat(outputTensor->getElementType())))
    {
        LOG_TRACE(GC, "Setting clip mode for fp8 cast");
        paramsPtr         = std::make_shared<ns_CastKernel::ParamsV3>();
        (*paramsPtr).mode = CastSatMode_t::CAST_CLIP;
        paramsSize        = sizeof(ns_CastKernel::ParamsV3);
    }
    return std::make_pair(paramsPtr, paramsSize);
}

NodePtr CastNodeHandler::createCastNode(const TensorPtr&      inputTensor,
                                        const TensorPtr&      outputTensor,
                                        std::string_view      nodeName,
                                        tpc_lib_api::DeviceId deviceId)
{
    std::string castKernelName = getCastGUID(inputTensor->getElementType(), outputTensor->getElementType());

    // If device id is passed, validate the kernel exist
    if (deviceId != tpc_lib_api::DEVICE_ID_MAX && !KernelDB::instance().isKernelExist(castKernelName, deviceId))
        return nullptr;

    LOG_DEBUG(GC, "Creating cast node with TPC kernel: {}", castKernelName);

    auto castParamsPair = getCastParams(inputTensor, outputTensor, deviceId);

    NodePtr node = NodeFactory::createNode({inputTensor},
                                           {outputTensor},
                                           castParamsPair.first.get(),
                                           castParamsPair.second,
                                           castKernelName,
                                           nodeName);
    NodeAnnotation& nodeAnnotation = node->getNodeAnnotation();
    nodeAnnotation.insertedNode = true;
    return node;
}

bool CastNodeHandler::createCastNodes(const pNode& node, tpc_lib_api::DeviceId deviceId)
{
    m_totalCreatedCasts   = 0;
    unsigned createdCasts = 0;

    bool ret = _createCastNodes(node, true, createdCasts, deviceId);
    m_totalCreatedCasts += createdCasts;

    if (ret)
    {
        ret = _createCastNodes(node, false, createdCasts, deviceId);
        m_totalCreatedCasts += createdCasts;
    }
    LOG_DEBUG(GC, "{}: total cast nodes created {}", __FUNCTION__, m_totalCreatedCasts);
    return ret;
}

bool CastNodeHandler::plantCastNodes(HabanaGraph& g)
{
    if (m_castInfoMap.empty())
    {
        return true;
    }
    LOG_DEBUG(GC, "Plant {} cast nodes",  m_castInfoMap.size());
    for (const auto& castInfoPair : m_castInfoMap)
    {
        pNode castNode = castInfoPair.second.castNode;
        pTensor castTensor = castInfoPair.first.tensor;
        SortableNodeMap<pTensor> nodeToReplacedTensor;

        if (castTensor == castNode->getInput(0))
        {
            pTensor replacementTensor = castNode->getOutput(0);
            const auto& consumers = castInfoPair.second.castNodeConsumers;
            for (const auto& consumer : consumers)
            {
                nodeToReplacedTensor.insert(std::make_pair(consumer, replacementTensor));
            }
        }
        else
        {
            pNode producer = castInfoPair.second.castNodeProducer;
            nodeToReplacedTensor.insert(std::make_pair(producer, castNode->getInput(0)));
        }

        for (const auto& changedNode : nodeToReplacedTensor)
        {
            LOG_DEBUG(GC,
                      "{}: Replacing {} tensor with {} tensor",
                      __FUNCTION__,
                      castTensor->getName(),
                      changedNode.second);
            pNode node = changedNode.first;
            GraphEditor::replaceTensor(g, node, castTensor, changedNode.second);
        }
        if (!GraphEditor::addNode(g, castNode))
        {
            LOG_ERR(GC, "{}: Failed to add {} node ", __FUNCTION__, castNode->getNodeName());
            return false;
        }
    }
    return true;
}

void CastNodeHandler::clear()
{
    m_totalCreatedCasts = 0;
    m_castInfoMap.clear();
}

bool CastNodeHandler::_createCastNodes(const pNode& node,
                                       bool castInput,
                                       unsigned& createdCasts,
                                       tpc_lib_api::DeviceId deviceId)
{
    const auto& tensors = castInput ? node->getInputs() : node->getOutputs();
    const std::string nodeName = node->getNodeName() + (castInput ? "_cast_input" : "_cast_output");
    createdCasts = 0;

    for (unsigned int index = 0; index < tensors.size(); ++index)
    {
        if (tensors[index] == nullptr || !tensors[index]->isDataTensor())
        {
            continue; // could happen in case of MME node with bias/cin tensors set to null
        }
        pTensor tensor = tensors[index];
        synDataType requiredType = castInput? node->getRequiredInputType(index) : node->getRequiredOutputType(index);

        if (requiredType != tensor->getElementType())
        {
            CastInfoKey castInfoKey;
            castInfoKey.tensor = tensor;
            castInfoKey.requiredType = requiredType;
            castInfoKey.castInput = castInput;
            auto castInfoIter = m_castInfoMap.find(castInfoKey);
            if (castInfoIter != m_castInfoMap.end())
            {
                HB_ASSERT(castInput, "castInput is false");
                castInfoIter->second.castNodeConsumers.push_back(node);
                createdCasts++;
                continue;
            }

            std::string curNodeName = nodeName + std::to_string(index);
            LOG_DEBUG(GC,"Trying to insert cast node ({})", curNodeName);

            CastInfo castInfo;
            pTensor castTensor = createCastTensor(tensor, requiredType, curNodeName);
            pTensor inputTensor = castInput ? tensor : castTensor;
            pTensor outputTensor = castInput ? castTensor : tensor;
            castInfo.castNode = createCastNode(inputTensor, outputTensor, curNodeName, deviceId);
            if (castInfo.castNode == nullptr)
            {
                return false;
            }
            if (castInput)
            {
                castInfo.castNodeConsumers.push_back(node);
            }
            else
            {
                castInfo.castNodeProducer = node;
            }
            m_castInfoMap.insert(std::make_pair(castInfoKey, castInfo));
            createdCasts++;
        }
    }
    return true;
}

bool CastNodeHandler::CastInfoKey::operator<(const CastInfoKey& rhs) const
{
    return (tensor->getId() != rhs.tensor->getId()? tensor->getId() < rhs.tensor->getId() : (requiredType != rhs.requiredType? requiredType < rhs.requiredType : castInput < rhs.castInput));
}
