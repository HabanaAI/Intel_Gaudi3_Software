#pragma once

#include "transpose_utils.h"
#include "types.h"
#include <optional>

class HabanaGraph;
struct StridedOpNodeInfo;

// in case the compilation mode is not LAZY or we did not go through
// Pytorch bridge, we might receive strided view\insert encoding a sequence
// of logical operations that took place on it.
// We wish to detect those and handle them separately for better device performance.
// In LAZY mode we have the bridge tracking the various logical operations that were
// applied to the view, such that the graph would be created with the corresponding
// nodes instead of a single strided view\insert encoding the final result.
// For Pytorch 2.X with torch.compile there is a different mechanism to avoid the
// need for the decoding bellow relying on "AOT autograd" feature and explained in
// https://confluence.habana-labs.com/display/Frameworks/Graph+mode#Graphmode-AOTAutogradforHPUcompiler
// We only handle the static use case.
// TODO: add support for ndim (SW-161081)

class StridedNodeSubOpDecoder
{
public:
    StridedNodeSubOpDecoder(const NodePtr& node, StridedOpNodeInfo& nodeInfo) : m_node(node), m_nodeInfo(nodeInfo) {}
    virtual bool                     canExtract() const = 0;
    virtual std::pair<NodePtr, bool> extract()          = 0;

protected:
    void                    fixupStridedViewZeroStrides();
    bool                    canDropStridedView() const;
    inline const TensorPtr& getViewTensor() const;

    const NodePtr&     m_node;
    StridedOpNodeInfo& m_nodeInfo;
};

// strided view can encode a broadcast (Pytorch expand method).
// This is represented through 0 strides with corresponding non trivial dimensions (> 1) on the output tensor.
// https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
// This is only applicable to strided view. The reasoning behind it can be described as follow:
// strided insert parameters describe how to write into the output
// while strided view parameters describe how to read the input into the output.
// For strided view we can read the same element many times and write it to new locations in the output.
// But for strided insert this operation is not applicable since it would translate into revisit of the same locations
// for the output tensor, leading to undefined behavior.
class StridedOpBroadcastDecoder final : public StridedNodeSubOpDecoder
{
public:
    StridedOpBroadcastDecoder(const NodePtr& node, StridedOpNodeInfo& nodeInfo)
    : StridedNodeSubOpDecoder(node, nodeInfo)
    {
    }
    bool                     canExtract() const override;
    std::pair<NodePtr, bool> extract() override;

private:
    void updateStridedOpNodeInfoParams(const TensorPtr& broadcastInput);
};

// strided view can encode a transpose (Pytorch permute method).
// This is represented through the node params strides which would not be monotinically non decreasing.
// https://pytorch.org/docs/stable/generated/torch.permute.html#torch.permute
// This is applicable to both strided view and strided insert where we might read the source tensor with a permutation
// as well as for strided insert where we might write to the output tensor the insert tensor with a
// permutation.
class StridedOpTransposeDecoder final : public StridedNodeSubOpDecoder
{
public:
    StridedOpTransposeDecoder(const NodePtr& node, StridedOpNodeInfo& nodeInfo)
    : StridedNodeSubOpDecoder(node, nodeInfo)
    {
    }
    bool                     canExtract() const override;
    std::pair<NodePtr, bool> extract() override;

private:
    using StrideAndDimVector = llvm_vecsmall::SmallVector<std::pair<TStride, unsigned>, tpc_lib_api::MAX_TENSOR_DIM>;
    NodePtr createTransposeNode(const TensorPtr&          origTensor,
                                const TensorPtr&          newTensor,
                                const StrideAndDimVector& strideAndDimVec,
                                bool                      isStridedView) const;
    void    updateStridedOpNodeInfoParams(const TensorPtr&                 newTensor,
                                          const TransposePermutationArray& inversePermutationArray,
                                          bool                             isStridedView);
};

// strided view\insert can encode a reshape (Pytorch reshape method).
// https://pytorch.org/docs/stable/generated/torch.reshape.html#torch-reshape
// For strided view it means the input and output have the same number of elements
// and default strides, while for strided insert it means the same for the insert tensor
// and output tensor. For strided insert we achieve a performance optimization on the device
// as we do not need to copy the view tensor, while for strided insert it would behave the
// same as the later replacement by a logical strided view in multi node extraction pass in case we would have not
// applied this decoding.

class StridedOpReshapeDecoder final : public StridedNodeSubOpDecoder
{
public:
    StridedOpReshapeDecoder(const NodePtr& node, StridedOpNodeInfo& nodeInfo) : StridedNodeSubOpDecoder(node, nodeInfo)
    {
    }
    bool                     canExtract() const override;
    std::pair<NodePtr, bool> extract() override;

private:
    inline const TensorPtr& getReshapeInputTensor() const;
};

class StridedOpDecoderStrategies
{
public:
    StridedOpDecoderStrategies(const NodePtr& node, StridedOpNodeInfo& nodeInfo);
    static constexpr size_t                                                    DECODER_STRATEGY_COUNT = 3;
    inline const std::array<StridedNodeSubOpDecoder*, DECODER_STRATEGY_COUNT>& getDecoderStrategies() const;

private:
    StridedOpBroadcastDecoder                                    m_broadcastDecoder;
    StridedOpTransposeDecoder                                    m_transposeDecoder;
    StridedOpReshapeDecoder                                      m_reshapeDecoder;
    std::array<StridedNodeSubOpDecoder*, DECODER_STRATEGY_COUNT> m_subOpDecoders;
};

class StridedOpDecoder
{
public:
    static bool       canExtract(const NodePtr& node);
    static NodeVector extract(const NodePtr& node, bool changeInPlace);

private:
    static const synStridedOpParams* getStridedOpParams(const NodePtr& node);
};