#pragma once

#include "habana_nodes.h"
#include "logical_op_node.h"
#include "node_visitor.h"
#include "shape_op_node.h"
#include "synapse_common_types.h"
#include "sif/shape_inference_metadata.h"
#include "access_pattern.h"

/**
 * If no parameter is given- squeeze node will remove all dimensions which are equal to 1
 * Example- Axis=nullptr (1,2,1,5) -> (2,5)
 * If a parameter is given, only if the axis is equal to 1, it will be squeezed.
 * Example- Axis=2 (1,2,1,5)->(1,2,5)
 */
class SqueezeNode : public ReshapeNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef ReshapeNode BaseClass;

    virtual NodePtr clone() const override;

    virtual bool validateNode() const override;

    virtual bool validateDynamicShapes() const override { return true; }

    virtual void runLogicalOperation() const override;

    virtual bool isAliasStrided() const override;

    virtual bool canSwapAliasDirection() const override { return true; };

    virtual bool canHandleStridedRealTensor() const override;

    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;

    virtual size_t getShapeInferenceFunctionUserParamsSize() const override;

    virtual bool isNode64BitCompatible() const override;

    virtual void permuteParams(const PermutationVector& inputPermutations) override;

    inline bool isAxisSet() const { return m_axis.is_set(); }

    inline Settable<unsigned> axisToSqueeze() const { return m_axis; }

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    SqueezeNode(const pTensor&   input,
                const pTensor&   output,
                UserParams       userParams = nullptr,
                std::string_view name       = "",
                eNodeType        nodeType   = TYPE_SQUEEZE_NODE);

    virtual gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override;

private:
    static NodePtr     createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  std::string_view    guid,
                                  std::string_view    name);
    NSizeArray         calculateExpectedOutput(const NSizeArray& input) const;
    bool               isDimSqueezed(unsigned dim) const;
    DimVector          getSqueezedDims() const;

    Settable<unsigned> m_axis;
    SifSqueezeMetadata m_sifParams;
};

class SqueezeShapeNode : public ShapeOperationNode<SqueezeNode>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = ShapeOperationNode<SqueezeNode>;

    ~SqueezeShapeNode() override = default;

    bool    validateNode() const override;
    NodePtr clone() const override;

private:
    SqueezeShapeNode(const TensorPtr& inputs, const TensorPtr& outputs, UserParams userParams, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
