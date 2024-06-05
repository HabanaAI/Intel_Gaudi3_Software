#pragma once

#include "aggregation_node.h"
#include "node_visitor.h"
#include "sif/shape_inference_metadata.h"

class ConcatenateNode : public AggregationNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef AggregationNode BaseClass;

    virtual NodePtr clone() const override;
    bool RunOnCpu() override;

    virtual TensorShape getInputShape(const TensorShape& output, uint32_t outputIndex, uint32_t inputIdx) const override;

    static SifNodeParams getShapeInferenceFunctionUserParams(Settable<SifConcatenateMetadata>& m_metadata,
                                                             const unsigned                    aggregationDim,
                                                             const TensorVector&               inputs);

protected:
    SifNodeParams getShapeInferenceFunctionUserParams() override;
    size_t getShapeInferenceFunctionUserParamsSize() const override;
    Settable<SifConcatenateMetadata> m_metadata;

private:
    ConcatenateNode(const TensorVector& in, const TensorVector& out, UserParams userParams, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    static NodePtr createNodeInternal(const TensorVector& inputs,
                                      const TensorVector& outputs,
                                      UserParams          userParams,
                                      std::string_view    guid,
                                      std::string_view    name);

    static NodePtr createNodeLogicalInternal(const TensorVector& inputs,
                                             const TensorVector& outputs,
                                             UserParams          userParams,
                                             std::string_view    guid,
                                             std::string_view    name);

    static NodePtr createConcatNode(const TensorVector& inputs,
                                    const TensorVector& outputs,
                                    UserParams          userParams,
                                    std::string_view    guid,
                                    std::string_view    name,
                                    bool                isInternalNode);

    static bool checkIfPhysicalConcat(const TensorVector& inputs, const TensorVector& outputs, unsigned dim);

    virtual void setInputLayouts(const LayoutVector& layouts) override;
};
