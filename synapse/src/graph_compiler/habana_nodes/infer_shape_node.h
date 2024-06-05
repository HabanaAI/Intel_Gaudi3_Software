#pragma once

#include "shape_op_node.h"
#include "node_visitor.h"
#include "habana_nodes.h"

struct InferShapeParams
{
    bool                              isTpc;
    u_int32_t                         sifId;
    char                              sifGUID[tpc_lib_api::MAX_NODE_NAME];
    SifNodeParams                     sifMetadata;
    size_t                            sifMetadataSize;
    std::shared_ptr<MultiSifNodeInfo> multiSifInfo;
};

class InferShapeNode : public ShapeOperationNode<LogicalOpNode>,
                       public MultiSifNodeInfoHelper
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = ShapeOperationNode<LogicalOpNode>;

    ~InferShapeNode() override = default;

    bool    validateNode() const override;
    bool    validateDynamicShapes() const override;
    NodePtr clone() const override;

    SifNodeParams getShapeInferenceFunctionUserParams() override
    {
        return reinterpret_cast<SifNodeParams>(m_sifMetadataBuffer.data());
    }

    size_t getShapeInferenceFunctionUserParamsSize() const override { return m_sifMetadataSize; }
    uint64_t getShapeInferenceFunctionVersion() const override;

    bool runShapeInferenceFunction(synDeviceType deviceType,
                                   SifParams*    params,
                                   SifOutputs*   outputs,
                                   bool          inferMax,
                                   bool          skipStatic) override;
    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

private:
    InferShapeNode(const TensorVector& inputs,
                   const TensorVector& outputs,
                   UserParams          userParams,
                   std::string_view    name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    bool                              m_isTpc;
    std::string                       m_sifGUID;
    std::vector<uint8_t>              m_sifMetadataBuffer;
    size_t                            m_sifMetadataSize;
};
