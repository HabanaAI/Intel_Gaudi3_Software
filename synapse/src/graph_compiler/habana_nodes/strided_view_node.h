#pragma once

#include "logical_op_node.h"
#include "multi_node.h"
#include "node_visitor.h"
#include "synapse_common_types.hpp"
#include "types.h"

class StridedViewNode : public MultiNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;
    virtual bool    isDynamicShape() const override;

    static bool validateViewNode(const Node* node, const synStridedOpParams& params, bool validateAccess = true);

    virtual NodeList extract() override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

    void printParamsRawData() const override;

    bool isNode64BitCompatible() const override { return true; }

    bool isDataMovementMultiNode() const override { return true; };

    const synStridedOpParams& getParams() const { return m_params; }

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

    StridedViewNode(const TensorVector& in, const TensorVector& out, UserParams params, std::string_view name);

    enum StridedViewInputs
    {
        INPUT_TENSOR   = 0,
        SHAPE_TENSOR   = 1,
        STRIDES_TENSOR = 2,
        OFFSET_TENSOR  = 3,
        MAX_NUM_INPUTS
    };

private:
    synStridedOpParams m_params;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    static void handleFcdStride(NodeList&           nodes,
                                TensorVector&       inputs,
                                TensorVector&       outputs,
                                synStridedOpParams& params,
                                const std::string&  name);

    static void handleDynamicInput(NodeList&           nodes,
                                   TensorVector&       inputs,
                                   TensorVector&       outputs,
                                   synStridedOpParams& params,
                                   const std::string&  name);

    static void handle64BitInput(NodeList&           nodes,
                                 TensorVector&       inputs,
                                 TensorVector&       outputs,
                                 synStridedOpParams& params,
                                 const std::string&  name);

    static void handleDynamicStrides(NodeList&           nodes,
                                     TensorVector&       inputs,
                                     TensorVector&       outputs,
                                     synStridedOpParams& params,
                                     const std::string&  name);

    static void convertShapeToH2D(NodeList&           nodes,
                                  TensorVector&       inputs,
                                  TensorVector&       outputs,
                                  synStridedOpParams& params,
                                  const std::string&  name);
};
