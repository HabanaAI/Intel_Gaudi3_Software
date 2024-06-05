#include "infer_shape_node.h"

#include "types_exception.h"

#include "kernel_db.h"

InferShapeNode::InferShapeNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          userParams,
                               std::string_view    name)
: BaseClass(inputs, outputs, name, INPUT_TO_OUTPUT, Node::TYPE_INFER_SHAPE), MultiSifNodeInfoHelper()
{
    setParams(userParams, sizeof(InferShapeParams));
}

void InferShapeNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParamsSize != sizeof(InferShapeParams))
    {
        LOG_ERR(HABANA_NODE, "InferShapeNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(InferShapeParams));
    }
    InferShapeParams params    = *reinterpret_cast<InferShapeParams*>(userParams);
    m_shapeInferenceFunctionID = (ShapeFuncID)params.sifId;
    m_isTpc                    = params.isTpc;
    m_sifMetadataSize          = params.sifMetadataSize;
    if (m_sifMetadataSize > 0)
    {
        m_sifMetadataBuffer.resize(m_sifMetadataSize);
        memcpy(m_sifMetadataBuffer.data(), params.sifMetadata, m_sifMetadataSize);
    }

    if (m_isTpc)
    {
        m_sifGUID.assign(params.sifGUID);
        setMultiSifInfo(params.multiSifInfo);
    }
    LOG_TRACE(HABANA_NODE,
              "InferShapeNode name - {}, params - shapeInferenceFunctionID={}, isTpc={}, sifMetadataSize={}",
              getNodeName(),
              m_shapeInferenceFunctionID,
              m_isTpc,
              m_sifMetadataSize);
}

bool InferShapeNode::validateNode() const
{
    CHECK_RET_FALSE(m_inputs.size() > 0, "InferShapeNode expects at least 1 input");

    CHECK_RET_FALSE(std::find_if_not(m_outputs.begin(),
                                     m_outputs.end(),
                                     [](TensorPtr t) { return t->isShapeTensor() || t->isHost2DeviceTensor(); }) == m_outputs.end(),
                    "InferShapeNode expects all outputs to be shape and/or H2D tensors");

    return LogicalOpNode::validateNode();
}

bool InferShapeNode::validateDynamicShapes() const
{
    return true;
}

bool InferShapeNode::runShapeInferenceFunction(synDeviceType deviceType,
                                               SifParams*    params,
                                               SifOutputs*   outputs,
                                               bool          inferMax,
                                               bool          skipStatic)
{
    if (!m_isTpc)
    {
        return BaseClass::runShapeInferenceFunction(deviceType, params, outputs, inferMax, skipStatic);
    }
    else
    {
        auto deviceIdGlueFormat = deviceTypeToDeviceID(deviceType);
        tpc_lib_api::GlueCodeReturn   ret;
        tpc_lib_api::UniqueShapeInferenceHash sifID;

        // Get SIF id for the runtime
        if (!KernelDB::instance().GetKernelShapeInferenceFunctionID(deviceIdGlueFormat, m_sifGUID, &sifID))
        {
            LOG_ERR(GC, "Can't get shape inference function id for dynamic shape mock-tpc node {}", m_name);
            return false;
        }

        m_shapeInferenceFunctionID = sifID.Value;

        ret = KernelDB::instance().RunShapeInferenceFunction(deviceIdGlueFormat, m_sifGUID, params, outputs);

        if (ret != tpc_lib_api::GLUE_SUCCESS)
        {
            LOG_ERR(GC,
                    "Running shape inference for mock-tpc node {} guid {} error {}",
                    m_name,
                    getGUID(),
                    enumToString(ret));
            return false;
        }
    }

    return true;
}

uint64_t InferShapeNode::getShapeInferenceFunctionVersion() const
{
    if (!m_isTpc)
    {
        return Node::getShapeInferenceFunctionVersion();
    }

    // look it up in the database
    // but we need to find the SIF ID first
    // but we cannot find it without the device type which we do not have
    // but SIF should not depend on the device type
    // so just look in all of them

    synDeviceType devices[] { synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3 };

    ShapeFuncRegistry& sfr = ShapeFuncRegistry::instance();

    for (auto deviceType: devices)
    {

        auto deviceIdGlueFormat = deviceTypeToDeviceID(deviceType);
        tpc_lib_api::UniqueShapeInferenceHash globalSifID;

        // not found for this device... look for the next one
        if (!KernelDB::instance().GetKernelShapeInferenceFunctionID(deviceIdGlueFormat, m_sifGUID, &globalSifID))
            continue;

        sm_function_id_t sifId;
        sifId.sm_func_index = globalSifID.Value;
        uint64_t version = sfr.getSifVersion(sifId);

        if (version != (uint64_t)(-1))
            return version;
    }

    LOG_ERR(GC, "Can't get shape inference function version for node {}", m_name);
    return (uint64_t)(-1);
}

NodePtr InferShapeNode::createNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   UserParams          userParams,
                                   std::string_view    guid,
                                   std::string_view    name)
{
    return NodePtr(new InferShapeNode(inputs, outputs, userParams, name));
}

NodePtr InferShapeNode::clone() const
{
    return NodePtr(new InferShapeNode(*this));
}
