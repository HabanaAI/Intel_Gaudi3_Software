#include "static_reshape_shape_node.h"

StaticReshapeShapeNode::StaticReshapeShapeNode(const TensorVector&       inputs,
                                               const TensorVector&       outputs,
                                               synStaticReshapeSifParams params,
                                               std::string_view          name)
: BaseClass(inputs, outputs, params, name)
{
}

NodePtr StaticReshapeShapeNode::createNode(const TensorVector& inputs,
                                           const TensorVector& outputs,
                                           UserParams          userParams,
                                           std::string_view    guid,
                                           std::string_view    name)
{
    HB_ASSERT(!inputs.empty(), "missing input for reshape {}", name);
    HB_ASSERT(!outputs.empty() > 0, "missing input for reshape {}", name);
    return std::make_shared<StaticReshapeShapeNode>(inputs,
                                                    outputs,
                                                    createParamsFromTensors(inputs[0], outputs[0]),
                                                    name);
}
