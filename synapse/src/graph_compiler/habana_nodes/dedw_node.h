#pragma once

#include "conv_base_node.h"
#include "node_visitor.h"


class DeToDwNode : public ConvBaseNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;
public:
    typedef ConvBaseNode BaseClass;

    virtual NodePtr clone() const override;

    virtual TensorSemanticType getParamSemanticType(const TensorPtr& param) const override;
    bool RunOnCpu() override;

    virtual bool validateNode() const override;

    virtual TensorShape getInputShape(const TensorShape& output, uint32_t outputIndex, uint32_t inputIdx) const override;
    virtual bool        validateNodeForGraph(const HabanaGraph& g) const override;

    virtual bool isOperandTransposed(const TensorPtr& t) const override;

    virtual bool is3DConvolutionGuid() const override;

    virtual TensorPtr getXOperand() const override;
    virtual TensorPtr getYOperand() const override;
    virtual TensorPtr getWOperand() const override;

    bool         isSpatialSlicingSupported(unsigned dim) const override;
    TSize        getMinSpatialDimOutputROI(unsigned dim) const override;

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              unsigned            userParamsSize,
                              std::string_view    guid,
                              std::string_view    name);

    DeToDwNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);
};
