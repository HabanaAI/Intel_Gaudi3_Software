/**
 * @file node_tensor_accessor.h
 * @brief
 * Access node tensor inlined
 * Templated on inpt / output and tensor type control / data
 */
#pragma once

#include "node.h"
#include "types.h"

template<Node::eParamUsage TUsage, Node::eTensorType TYPE = Node::TENSOR_TYPE_DATA>
class NodeTensorsAccessor
{
public:
    static const TensorVector& getTensors(const NodePtr& node);
    static const void          replace(const NodePtr& node, unsigned index, const TensorPtr& newTensor);
};

template<Node::eTensorType TYPE>
class NodeTensorsTypeAccessor
{
public:
    static const TensorVector& getInputs(const NodePtr& node);
    static const TensorVector& getOutputs(const NodePtr& node);

    static const void replaceInput(const NodePtr& node, unsigned index, const TensorPtr& newTensor);
    static const void replaceOutput(const NodePtr& node, unsigned index, const TensorPtr& newTensor);
};

template<>
class NodeTensorsTypeAccessor<Node::TENSOR_TYPE_DATA>
{
public:
    static const TensorVector& getInputs(const NodePtr& node) { return node->getInputs(); }
    static const TensorVector& getOutputs(const NodePtr& node) { return node->getOutputs(); };

    static const void replaceInput(const NodePtr& node, unsigned index, const TensorPtr& newTensor)
    {
        node->replaceInput(index, newTensor, Node::TENSOR_TYPE_DATA);
    }

    static const void replaceOutput(const NodePtr& node, unsigned index, const TensorPtr& newTensor)
    {
        node->replaceOutput(index, newTensor);
    }
};

template<>
class NodeTensorsTypeAccessor<Node::TENSOR_TYPE_CONTROL>
{
public:
    static const TensorVector& getInputs(const NodePtr& node) { return node->getControlInputs(); }
    static const TensorVector& getOutputs(const NodePtr& node) { return node->getControlOutputs(); };

    static const void replaceInput(const NodePtr& node, unsigned index, const TensorPtr& newTensor)
    {
        node->replaceInput(index, newTensor, Node::TENSOR_TYPE_CONTROL);
    }

    static const void replaceOutput(const NodePtr& node, unsigned index, const TensorPtr& newTensor)
    {
        node->replaceOutput(index, newTensor);
    }
};

template<Node::eTensorType TYPE>
class NodeTensorsAccessor<Node::USAGE_INPUT, TYPE>
{
public:
    static const TensorVector& getTensors(const NodePtr& node)
    {
        return NodeTensorsTypeAccessor<TYPE>::getInputs(node);
    }

    static const void replace(const NodePtr& node, unsigned index, const TensorPtr& newTensor)
    {
        NodeTensorsTypeAccessor<TYPE>::replaceInput(node, index, newTensor);
    }
};

template<Node::eTensorType TYPE>
class NodeTensorsAccessor<Node::USAGE_OUTPUT, TYPE>
{
public:
    static const TensorVector& getTensors(const NodePtr& node)
    {
        return NodeTensorsTypeAccessor<TYPE>::getOutputs(node);
    }

    static const void replace(const NodePtr& node, unsigned index, const TensorPtr& newTensor)
    {
        NodeTensorsTypeAccessor<TYPE>::replaceOutput(node, index, newTensor);
    }
};

template<Node::eParamUsage TUsage, Node::eTensorType TYPE = Node::TENSOR_TYPE_DATA>
static void runOnTensors(const NodePtr& node, std::function<void(const TensorPtr&)> func)
{
    const TensorVector& tensors = NodeTensorsAccessor<TUsage, TYPE>::getTensors(node);
    for (const auto& t : tensors)
    {
        func(t);
    }
}

template<Node::eParamUsage TUsage>
static void runOnTensorsForType(const NodePtr& node, Node::eTensorType type, std::function<void(const TensorPtr&)> func)
{
    if (type == Node::TENSOR_TYPE_DATA || type == Node::TENSOR_TYPE_ALL)
    {
        runOnTensors<TUsage, Node::TENSOR_TYPE_DATA>(node, func);
    }
    if (type == Node::TENSOR_TYPE_CONTROL || type == Node::TENSOR_TYPE_ALL)
    {
        runOnTensors<TUsage, Node::TENSOR_TYPE_CONTROL>(node, func);
    }
}