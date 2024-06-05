#pragma once
#include <string>
#include <optional>
#include <memory>
#include "types.h"
#include "tensor_shape.h"

/* The strategy classes must have 3 static function:
 * isFitForStrategy - returns true if the optimization can be used and needed
 * applyStrategy - create the new nodes
 * getStrategyName  - return the name of the optimization
 */
class MemcpyOptimizationStrategy
{
public:
    virtual bool             isFitForStrategy(const NodePtr& memcpy) const = 0;
    virtual NodeList         applyStrategy(const NodePtr& memcpy) const    = 0;
    virtual std::string_view getStrategyName() const                       = 0;
};

// this optimization is the end point optimization and it just create the new Dma node
class StridedMemcpyViaTransposeEngineStrategy : public MemcpyOptimizationStrategy
{
public:
    virtual bool             isFitForStrategy(const NodePtr& memcpy) const override;
    virtual NodeList         applyStrategy(const NodePtr& memcpy) const override;
    virtual std::string_view getStrategyName() const override { return "StridedMemcpyViaTransposeEngine"; }

private:
    NodePtr createReshape(const TensorPtr& input, const TensorPtr& output, const std::string& name) const;
    NodePtr createReinterpretCast(const TensorPtr& input, const TensorPtr& output, const std::string& name) const;
};

class AggregateFcdWithStaticReshapeBase : public MemcpyOptimizationStrategy
{
    virtual std::string_view getStrategyName() const override                                             = 0;
    virtual bool shouldBreakOnDimWhenCalculateNewFcdSize(const NodePtr& memcpy, const unsigned dim) const = 0;

protected:
    virtual bool               isFitForStrategy(const NodePtr& memcpy) const override;
    std::optional<TensorShape> getNewShape(const NodePtr& memcpy) const;

    NodePtr createStaticReshape(const TensorPtr& input, const TensorPtr& output, const std::string& name) const;
    // Returns list of the new nodes (include the memcpy), and the memcpy.
    std::pair<NodeList, NodePtr> createNewMemcpyWrappedByReshapes(const NodePtr&     memcpy,
                                                                  const TensorShape& newShape) const;
};

/* This optimization is to increase the fcd size to include all dims that are contiguous in memory.
This is done by wrapping the dma memcpy node in 2 static reshape nodes, for example:
original:  memcpy(sizes[2, 2, 256-512, 512, 1024])
optimized:  staticReshape(to optimized shape) -> memcpy(sizes[1024-2048, 512, 1024]) -> staticReshape(to original shape)
*/
class AggregateFcdWithStaticReshape : public AggregateFcdWithStaticReshapeBase
{
public:
    virtual bool             isFitForStrategy(const NodePtr& memcpy) const override;
    virtual NodeList         applyStrategy(const NodePtr& memcpy) const override;
    virtual std::string_view getStrategyName() const override { return "AggregateFcdWithStaticReshape"; }

private:
    virtual bool shouldBreakOnDimWhenCalculateNewFcdSize(const NodePtr& memcpy, const unsigned dim) const override
    {
        return false;
    }
};

class AggregateDynamicSliceFcdWithStaticReshape : public AggregateFcdWithStaticReshapeBase
{
public:
    virtual bool             isFitForStrategy(const NodePtr& memcpy) const override;
    virtual NodeList         applyStrategy(const NodePtr& memcpy) const override;
    virtual std::string_view getStrategyName() const override { return "AggregateDynamicSliceFcdWithStaticReshape"; }

private:
    NodeList     createDynamicSliceShapeOperations(const NodePtr& originalNode, const NodePtr& newNode) const;
    NodePtr      createMergeShapeNode(const TensorPtr& input, const unsigned numOfSqueezedDims) const;
    NodePtr      createTileShapeNode(const TensorPtr& input, const unsigned tile) const;
    virtual bool shouldBreakOnDimWhenCalculateNewFcdSize(const NodePtr& memcpy, const unsigned dim) const override;
};
