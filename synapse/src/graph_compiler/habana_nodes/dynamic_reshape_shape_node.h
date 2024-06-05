#pragma once

#include "shape_op_node.h"
#include "node_visitor.h"
#include "habana_nodes.h"
#include <string>

using ComponentVector = std::vector<std::string>;

struct dynamicReshapeParams
{
    std::string equation;
};

class DynamicReshapeShapeNode : public ShapeOperationNode<ReshapeNode>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = ShapeOperationNode<ReshapeNode>;

    ~DynamicReshapeShapeNode() override = default;

    bool    validateNode() const override;
    bool    validateDynamicShapes() const override;
    NodePtr clone() const override;

    SifNodeParams getShapeInferenceFunctionUserParams();

    size_t getShapeInferenceFunctionUserParamsSize() const;

    static char                           getLabelForDim(unsigned dim);
    static ComponentVector                initComponentVector(unsigned rank);
    static std::string                    makeReshapeEquation(const ComponentVector& c, unsigned rank);
    static std::pair<SizeArray, unsigned> calculateShape(const SizeArray& sizes, const std::string& eq);
    virtual void                          setParams(UserParams userParams, unsigned userParamsSize) override;

private:
    DynamicReshapeShapeNode(const TensorVector& inputs,
                            const TensorVector& outputs,
                            UserParams          userParams,
                            std::string_view    name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    bool parseEquation();

    bool                     m_parsing_status;
    std::string              m_equation;
    std::vector<uint8_t>     m_sifMetadataBuffer;
    std::vector<std::string> m_input_labels;
    std::string              m_output_eq;
};

struct einsumExtractParams
{
    std::vector<unsigned> freeDims[2];
};

class EinsumExpandShapeNode : public ShapeOperationNode<ReshapeNode>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = ShapeOperationNode<ReshapeNode>;

    ~EinsumExpandShapeNode() override = default;

    bool    validateNode() const override;
    bool    validateDynamicShapes() const override;
    NodePtr clone() const override;
    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

    SifNodeParams getShapeInferenceFunctionUserParams();

    size_t getShapeInferenceFunctionUserParamsSize() const;

private:
    EinsumExpandShapeNode(const TensorVector& inputs,
                          const TensorVector& outputs,
                          UserParams          userParams,
                          std::string_view    name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    std::vector<unsigned> m_freeDims1;
    std::vector<unsigned> m_freeDims2;
    std::vector<uint8_t>  m_sifMetadataBuffer;
};
