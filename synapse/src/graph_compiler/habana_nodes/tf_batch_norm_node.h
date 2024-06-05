#pragma once

#include "node.h"
#include "node_visitor.h"

class TfBatchNormNode : public Node
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    virtual NodePtr                      clone()        const override;
    virtual bool                         validateNode() const override;
    virtual bool                         validateNodeForGraph(const HabanaGraph& g) const override;
    const synTfBatchNormalizationParams& getParams();

    void printParamsRawData() const override;
    virtual void setParams(UserParams userParams, unsigned int userParamsSize) override;

private:
    synTfBatchNormalizationParams m_params;

    TfBatchNormNode(const TensorVector& in, const TensorVector& out, std::string_view name, UserParams params);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class TfFusedBatchNormGradNode : public Node
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    virtual NodePtr                      clone()        const override;
    virtual bool                         validateNode() const override;
    virtual bool                         validateNodeForGraph(const HabanaGraph& g) const override;
    const synTfBatchNormalizationParams& getParams();

    void printParamsRawData() const override;
    virtual void setParams(UserParams userParams, unsigned int userParamsSize) override;

private:
    synTfBatchNormalizationParams m_params;

    TfFusedBatchNormGradNode(const TensorVector& in,
                             const TensorVector& out,
                             std::string_view    name,
                             UserParams          userParams);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
