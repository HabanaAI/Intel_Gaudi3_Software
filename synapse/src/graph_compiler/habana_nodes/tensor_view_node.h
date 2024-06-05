#pragma once

#include "logical_op_node.h"
#include "node_visitor.h"

/**
 * This node represents a view into a tensor
 */
class TensorViewNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef LogicalOpNode BaseClass;

    // params to be passed to createNode method
    struct TensorViewParams
    {
        std::vector<SizeVector> dimsOffsets;
        bool                    accessInput;
    };

    /**
     * @param realTensor   IN  The tensor we wish to access
     * @param accessInput  IN  Whether the real tensor is input or output,
     *                         true indicates the real tensor is an input
     */
    TensorViewNode(const TensorPtr& realTensor  = std::make_shared<Tensor>(),
                   bool             accessInput = true,
                   std::string_view name        = "",
                   Node::eNodeType  type        = TYPE_TENSOR_VIEW);

    virtual NodePtr clone() const override;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    void addView(const TensorPtr& viewTensor, const SizeVector& dimsOffsets);

    bool realTensorIsInput() { return m_accessInput; }

    virtual bool RunOnCpu() override;

    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;

    virtual void runLogicalOperation() const override;

    virtual bool canHandleStridedRealTensor() const override { return !m_keepStrides; }

    void setKeepStrides();  // use tensor view with given strides, do not change them later.

    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override;

protected:
    virtual bool isAliasStrided(unsigned idx) const override;
    virtual bool isAliasStrided() const override;

private:
    std::vector<SizeVector> m_dimsOffsets;
    const bool              m_accessInput;
    std::vector<uint8_t>    m_sifMetadataBuffer;
    bool                    m_keepStrides = false;

    uint64_t getBaseByteOffsetOfView(unsigned idx) const;
};
