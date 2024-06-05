#pragma once

#include "defs.h"
#include "graph_traits.h"
#include "habana_nodes.h"
#include "habana_graph.h"
#include "logical_op_node.h"
#include "multi_node.h"
#include "node.h"
#include "broadcast_node.h"
#include "synapse_common_types.h"
#include "types.h"

namespace BroadcastNodeStrategy
{
class BaseStrategy
{
public:
    virtual void             extractNodes()       = 0;
    virtual std::string_view strategyName() const = 0;

    NodeVector getExtractedBroadcastNodes() const { return m_broadcastNodes; }
    // return the extracted broadcast nodes, but not the DMA broadcast
    NodeList getExtractedNodesExceptBroadcasts() const { return m_nodesExceptBroadcasts; }

protected:
    BaseStrategy(const TensorPtr& input, const TensorPtr& output, const TensorPtr shape, std::string_view name)
    : m_input(input), m_output(output), m_shapeTensor(shape), m_name(name)
    {
    }

    // common function for more than 1 strategies
    NodePtr createTwoDimTranspose(const TensorPtr& input, std::string_view suffix) const;

    void createNewBroadcast(const TensorVector& inputs, const TensorVector& outputs, std::string_view suffix = "");

    TensorPtr   m_input;
    TensorPtr   m_output;
    TensorPtr   m_shapeTensor;
    std::string m_name;
    // the extracted nodes:
    NodeVector m_broadcastNodes;
    NodeList   m_nodesExceptBroadcasts;
};

// in this strategy we create transpose before and after the broadcast,
// so we should perform the broadcast with higher utilization
// example: [1,4,32,256,2] ->(broadcast)-> [16,4,32,256,2] will replace with:
// [1,4,32,256,2] ->(transpose)-> [256,2,1,4,32] ->(broadcast)-> [256,2,16,4,32] ->(transpose)-> [16,4,32,256,2]
class TransposeStrategy : public BaseStrategy
{
public:
    TransposeStrategy(const TensorPtr& input,
                      const TensorPtr& output,
                      const TensorPtr  shape,
                      std::string_view name,
                      unsigned         dim)
    : BaseStrategy(input, output, shape, name), m_dim(dim)
    {
    }
    virtual void             extractNodes() override;
    virtual std::string_view strategyName() const override { return "TransposeStrategy"; }

private:
    TransposePermutationArray createShiftPermutation(const unsigned tensorDim, const unsigned axis) const;
    NodeList                  createShiftTransposeSequence(const TensorPtr& tensor,
                                                           const unsigned   axis,
                                                           bool             fromInput,
                                                           std::string_view suffix) const;
    unsigned                  m_dim;
};

// in this strategy we reshape the input and output to make them fit to the transpose strategy
// (it's supported only in static shape)
// we keep all dimensions up to the last broadcasted dim unchanged, and split the remaining sizes into 2 dims
// example in float (element size is 4 bytes): [2, 1, 2, 4, 8192] ->(broadcast)-> [2, 16, 2, 4, 8192] will replace with:
// [2, 1, 2, 4, 8192] ->(reshape)-> [2, 1, 16, 4096] ->(broadcast)-> [2, 16, 16, 4096] ->(reshape)-> [2, 16, 2, 4, 8192]
class PreTransposeStrategy : public BaseStrategy
{
public:
    PreTransposeStrategy(const TensorPtr& input,
                         const TensorPtr& output,
                         const TensorPtr  shape,
                         std::string_view name,
                         unsigned         splitBy,
                         unsigned         lastBroadcastedDim)
    : BaseStrategy(input, output, shape, name), m_splitBy(splitBy), m_lastBroadcastedDim(lastBroadcastedDim)
    {
    }
    virtual void             extractNodes() override;
    virtual std::string_view strategyName() const override { return "PreTransposeStrategy"; }

    // check if using this strategy will return the same broadcast
    static bool
    isRedundant(const TensorPtr& input, const TensorPtr& output, unsigned splitBy, unsigned lastBroadcastedDim);

private:
    static std::pair<TensorPtr, TensorPtr>
    creteNewTensors(const TensorPtr& input, const TensorPtr& output, unsigned splitBy, unsigned lastBroadcastedDim);

    unsigned m_splitBy;
    unsigned m_lastBroadcastedDim;
};

// in this strategy we reshape the input into 2-dim tensor ([tensor size, 1])
// then do broadcast on the SCD ([tensor size, Total broadcast size])
// and then return the tensor to expected shape with reshape and transpose (if needed)
// example: [X,1,Y,1,Z] ->(broadcast)-> [X,B1,Y,B2,Z] will replace with:
// [X,1,Y,1,Z]->(reshape)->[X*Y*Z,1]->(broadcast)->[X*Y*Z,B1*B2]->(reshape)->[X,Y,Z,B1,B2]->(transpose)->[X,B1,Y,B2,Z]
class FlattenStrategy : public BaseStrategy
{
public:
    FlattenStrategy(const TensorPtr&          input,
                    const TensorPtr&          output,
                    const TensorPtr           shape,
                    std::string_view          name,
                    const std::set<unsigned>& broadcastedDims)
    : BaseStrategy(input, output, shape, name), m_broadcastedDims(broadcastedDims)
    {
    }
    virtual void             extractNodes() override;
    virtual std::string_view strategyName() const override { return "FlattenStrategy"; }

private:
    TensorPtr createBroadcastOutputAfterFlatten() const;
    TensorPtr createReshapeShapeTensor(const TransposePermutationArray& permutation);
    TensorPtr createBroadcastShapeTensor(const TensorPtr& outputShape, const unsigned numDimsThatNotBroadcasted);

    std::set<unsigned> m_broadcastedDims;
};

// in this strategy we squeezed trivial dims (dims of size 1 in the output)
// example: [X, 1, 1, 1, 1] ->(broadcast)-> [X, B1, 1, B2, 1] will replace with:
// [X, 1, 1, 1, 1] ->(squeeze)-> [X, 1, 1] ->(broadcast)-> [X, B1, B2] ->(expand dim)-> [X, B1, 1, B2, 1]
class SqueezeStrategy : public BaseStrategy
{
public:
    SqueezeStrategy(const TensorPtr& input, const TensorPtr& output, const TensorPtr shape, std::string_view name)
    : BaseStrategy(input, output, shape, name)
    {
    }
    virtual void             extractNodes() override;
    virtual std::string_view strategyName() const override { return "SqueezeStrategy"; }
};

// in this strategy we expand input dim to output dim
// example: [X, Y, Z] ->(broadcast)-> [X, Y, Z, B1, B2] will replace with:
// [X, Y, Z] ->(expand dim)-> [X, Y, Z, 1, 1] ->(broadcast)-> [X, Y, Z, B1, B2]
class ExpandDimStrategy : public BaseStrategy
{
public:
    ExpandDimStrategy(const TensorPtr& input, const TensorPtr& output, const TensorPtr shape, std::string_view name)
    : BaseStrategy(input, output, shape, name)
    {
    }
    virtual void             extractNodes() override;
    virtual std::string_view strategyName() const override { return "ExpandDimStrategy"; }
};

// This strategy is used in case of dynamic broadcast dim. we do the full broadcast, and add a slice node after it.
// example: [X_min-X_max, 1] ->(broadcast)-> [X_min-X_max, B_min-B_max] will be replaced with:
// [X_min-X_max, 1] ->(broadcast)-> [X_min-X_max, B_max] ->(slice)-> [X_min-X_max, B_min-B_max]
class SliceStrategy : public BaseStrategy
{
public:
    SliceStrategy(const TensorPtr& input, const TensorPtr& output, const TensorPtr shape, std::string_view name)
    : BaseStrategy(input, output, shape, name)
    {
    }
    virtual void             extractNodes() override;
    virtual std::string_view strategyName() const override { return "SliceStrategy"; }
};

// in this strategy we split a broadcast with low utilization on the transpose after,
// to a small broadcast with low utilization, and a second broadcast with higher utilization
// (it's supported only in static shape)
// example in float (element size is 4 bytes): [4,1,16] ->(broadcast)-> [4,8192,16] will be replaced with:
// [4,1,16] ->(broadcast)-> [4,8,16] ->(reshape)-> [32,1,16] ->(broadcast)-> [32,1024,16] ->(reshape)->[4,8192,16]
class SplitStrategy : public BaseStrategy
{
public:
    SplitStrategy(const TensorPtr& input,
                  const TensorPtr& output,
                  const TensorPtr  shape,
                  std::string_view name,
                  unsigned         splitBy,
                  unsigned         broadcastDimSequenceStart,
                  unsigned         broadcastDimSequenceEnd,
                  unsigned         broadcastDimSequenceTotalSize)
    : BaseStrategy(input, output, shape, name),
      m_splitBy(splitBy),
      m_broadcastDimSequenceStart(broadcastDimSequenceStart),
      m_broadcastDimSequenceEnd(broadcastDimSequenceEnd),
      m_broadcastDimSequenceTotalSize(broadcastDimSequenceTotalSize)
    {
    }
    virtual void             extractNodes() override;
    virtual std::string_view strategyName() const override { return "SplitStrategy"; }

private:
    unsigned m_splitBy;  // the size of the first broadcast, it's divide the the full broadcast size

    // in case that there is a sequence of broadcasted dims, we can handle them like 1 broadcasted dim that his size
    // is the multiplication of all broadcast sizes in the sequence,
    // we also need to know where the sequence start and end
    unsigned m_broadcastDimSequenceStart;
    unsigned m_broadcastDimSequenceEnd;
    unsigned m_broadcastDimSequenceTotalSize;
};

// this is the final strategy, the input must be with 1 static broadcasted dim on the SCD
// and at most 1 additional dim, if the additional dim isn't dynamic it will be replaced by only DMA broadcast
// if not it will replace with log2(input size) concats, and slice node if the broadcast size is not power of 2
// example 1: [4, 1] ->(broadcast)-> [4, 205] will be replaced by [4, 1] ->(DMA broadcast)-> [4, 205]
// example 2: [4-8, 1] ->(broadcast)-> [4-8, 205] will be replaced by:
// [4-8, 1]  X2          ->(concat)-> [4-8, 2] (concat with same input)
// [4-8, 2]  X2          ->(concat)-> [4-8, 4]
// [4-8, 4]  X2          ->(concat)-> [4-8, 8]
// [4-8, 8]  X2          ->(concat)-> [4-8, 16]
// [4-8, 16] X2          ->(concat)-> [4-8, 32]
// [4-8, 32] X2          ->(concat)-> [4-8, 64]
// [4-8, 64] X2          ->(concat)-> [4-8, 128]
// [4-8, 128]            ->(slice) -> [4-8, 77]
// [4-8, 128], [4-8, 77] ->(concat)-> [4-8, 205]
class PhysicalBroadcastStrategy : public BaseStrategy
{
public:
    PhysicalBroadcastStrategy(const TensorPtr&       input,
                              const TensorPtr&       output,
                              const TensorPtr        shape,
                              std::string_view       name,
                              const HabanaDeviceType physicalEngineType)
    : BaseStrategy(input, output, shape, name), m_physicalEngineType(physicalEngineType)
    {
    }
    virtual void             extractNodes() override;
    virtual std::string_view strategyName() const override { return "PhysicalBroadcastStrategy"; }

private:
    NodePtr        duplicateTensor(const TensorPtr& tensor, const unsigned dim) const;
    void           performWithConcatsAndSlice();
    SliceNode::SliceNodeStaticParams createSliceParams(const unsigned remainder);
    std::string                      getTpcBroadcastGuid();

    const HabanaDeviceType m_physicalEngineType;
};

class TpcConstantKernelStrategy : public BaseStrategy
{
public:
    TpcConstantKernelStrategy(const TensorPtr& input,
                              const TensorPtr& output,
                              const TensorPtr  shape,
                              std::string_view name)
    : BaseStrategy(input, output, shape, name)
    {
    }
    virtual void             extractNodes() override;
    virtual std::string_view strategyName() const override { return "TpcConstantKernelStrategy"; }

private:
    std::string getGuid();
};

// in this strategy we replace the broadcast with identity Node
class IdentityStrategy : public BaseStrategy
{
public:
    IdentityStrategy(const TensorPtr& input, const TensorPtr& output, const TensorPtr shape, std::string_view name)
    : BaseStrategy(input, output, shape, name)
    {
    }
    virtual void             extractNodes() override;
    virtual std::string_view strategyName() const override { return "IdentityStrategy"; }
};

}  // namespace BroadcastNodeStrategy

class BroadcastNodeCreator
{
    using StrategyPtr = std::shared_ptr<BroadcastNodeStrategy::BaseStrategy>;

public:
    static NodeList createBroadcast(const NodePtr& node, const unsigned cacheLineSizeInBytes);
    static bool     isTrivialDim(const TensorPtr& t, const unsigned dim);  // trivial dim is dim of size 1 (min and max)

private:
    static constexpr float goodUtilization = 0.9;

    // since we support more than 1 broadcast node, we aggregate sequences of broadcasted dims to improve utilization
    struct BroadcastParams
    {
        unsigned dimStart;
        unsigned dimEnd;
        unsigned totalSize;
    };

    BroadcastNodeCreator(BroadcastNodeCreator& creator) = default;

    explicit BroadcastNodeCreator(const NodePtr& node, const unsigned cacheLineSizeInBytes);

    // the dense size in the memory before the first broadcasted dim
    unsigned getSizeBeforeFirstBroadcastedDim() const;
    // utilization formula: fcd size / ((num of DMA activations) X cache line size in bytes)
    // num of DMA activations = ceil(fcd size / cache line size in bytes)
    bool isGoodUtilization(unsigned fcdSizeInBytes) const;
    bool existsTrivialDim() const;
    bool isFcdBroadcast() const;
    // linear broadcast is broadcast that all the broadcasted dims are the outer dims,
    // in other words linear broadcast just duplicate the dense tensor as is.
    // positive example [[1.5, -3]](2,1) -> [[1.5, -3], [1.5, -3]](2,2), memory view: [1.5, -3, 1.5, -3]
    // negative example [[1.5], [-3]](1,2) -> [[1.5, -1.5], [-3, -3]](2,2), memory view: [1.5, 1.5, -3, -3]
    bool isLinearBroadcast() const;

    std::optional<unsigned>
    findBestSplit(const unsigned effectiveFcd, unsigned totalSize, std::string_view strategy) const;

    StrategyPtr tryToUseSplitStrategy(const unsigned effectiveFcd, const unsigned totalSize) const;

    std::shared_ptr<BroadcastNodeStrategy::BaseStrategy> findWinningStrategy();

    bool isRunOnTpc() const { return m_physicalBroadcastEngineType == HabanaDeviceType::DEVICE_TPC; }
    bool isRunOnDma() const { return m_physicalBroadcastEngineType == HabanaDeviceType::DEVICE_EDMA; }

    const TensorPtr              m_input;
    const TensorPtr              m_shapeTensor;
    const TensorPtr              m_output;
    const std::string            m_name;
    const unsigned               m_cacheLineSizeInBytes;
    const HabanaDeviceType       m_physicalBroadcastEngineType;
    std::set<unsigned>           m_broadcastedDims;
    std::vector<BroadcastParams> m_params;
};