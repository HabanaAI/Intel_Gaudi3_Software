#pragma once

#include "slice_node.h"

class SliceInsertNode : public SliceNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef SliceNode BaseClass;

    SliceInsertNode(const TensorVector& inputs,
                    const TensorVector& outputs,
                    UserParams          params,
                    unsigned            paramsSize,
                    std::string_view    name);

    NodePtr  clone() const override;
    NodeList extract() override;
    bool     validateNode() const override;

    TensorVector getDataInputs() const override { return {m_inputs[ORIGINAL_TENSOR], m_inputs[INSERT_TENSOR]}; }
    bool         hasShapeTensor() const override { return false; }
    uint32_t     getFirstShapeIndex() const override { return STEPS_TENSOR; }

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              unsigned            userParamsSize,
                              std::string_view    guid,
                              std::string_view    name);

    enum SliceInsertInputs
    {
        ORIGINAL_TENSOR = 0,
        INSERT_TENSOR   = 1,
        STEPS_TENSOR    = 2,
        H2D_TENSOR      = STEPS_TENSOR,
        STARTS_TENSOR   = 3,
        MAX_NUM_INPUTS,
    };

protected:
    NodePtr
    getSliceNode(const TensorVector& inputs, const TensorPtr& output, const SliceNodeStaticParams& params) override;

    NodePtr getLogicalNode(const TensorPtr&             unsliced,
                           const TensorPtr&             sliced,
                           const SliceNodeStaticParams& params) const override;

    TensorPtr getUnslicedTensor() const override;
    TensorPtr getSlicedTensor() const override;
    bool      isFwd() const override { return false; };
    bool      canTranspose() const override;
};
