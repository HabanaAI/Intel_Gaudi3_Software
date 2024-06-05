#pragma once

#include "node_visitor.h"
#include "multi_node.h"
#include "synapse_common_types.hpp"

class StridedInsertNode : public MultiNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;
    virtual bool    isDynamicShape() const override;

    static bool validateInsertNode(const Node* node, const synStridedOpParams& params, bool validateAccess = true);

    void printParamsRawData() const override;

    bool isNode64BitCompatible() const override { return true; }

    const synStridedOpParams& getParams() const { return m_params; }

    virtual NodeList extract() override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

    bool isDataMovementMultiNode() const override { return true; };

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

    StridedInsertNode(const TensorVector& in, const TensorVector& out, UserParams params, std::string_view name);

    enum StridedInsertInputs
    {
        ORIGINAL_TENSOR = 0,
        INSERT_TENSOR   = 1,
        STRIDES_TENSOR  = 2,
        OFFSET_TENSOR   = 3,
        MAX_NUM_INPUTS  = 4,
        MIN_NUM_INPUTS  = STRIDES_TENSOR
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

    static void handleDynamicStrides(NodeList&           nodes,
                                     TensorVector&       inputs,
                                     TensorVector&       outputs,
                                     synStridedOpParams& params,
                                     const std::string&  name);

    static void handle64BitInput(NodeList&           nodes,
                                 TensorVector&       inputs,
                                 TensorVector&       outputs,
                                 synStridedOpParams& params,
                                 const std::string&  name);
};
