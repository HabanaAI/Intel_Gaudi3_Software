#pragma once

#include "slice_node.h"

class SliceGradNode : public SliceNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef SliceNode BaseClass;

    SliceGradNode(const TensorVector& inputs,
                  const TensorVector& outputs,
                  UserParams          params,
                  unsigned            paramsSize,
                  std::string_view    name);

    NodePtr  clone() const override;
    NodeList extract() override;
    bool     validateNode() const override;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              unsigned            userParamsSize,
                              std::string_view    guid,
                              std::string_view    name);

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
    bool      shouldExtractToLogicalSlice() const override { return false; };

private:
    NodeList extractGradSlice() const;
};
