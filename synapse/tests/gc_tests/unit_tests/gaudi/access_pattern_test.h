#pragma once

#include "access_pattern.h"
#include "graph_optimizer_test.h"
#include <optional>

#include "gaudi2_graph.h"
#include "compilation_hal_reader.h"

using namespace gc::access_pattern;

struct ReshapeAccessPatternTestParams
{
    // Describe the geometry of a tensor in the pattern of 1x1x1...x<outerDimSize>x1x1x1...x<extra non-1 sized dims>
    struct DimParams
    {
        unsigned           m_outerOnes;
        unsigned           m_innerOnes;
        std::vector<TSize> m_reshapedDims;

        DimParams(unsigned outerOnes, unsigned innerOnes, std::initializer_list<TSize> reshapedDims)
        : m_outerOnes(outerOnes), m_innerOnes(innerOnes), m_reshapedDims(reshapedDims)
        {
        }
    };
    DimParams inputParams;
    DimParams outputParams;
};

class AccessPatternTest : public GraphOptimizerTest
{
protected:
    virtual void createNode() = 0;

    virtual bool isAllRequiredDim(unsigned dim, const TensorPtr& t) const { return false; };
    virtual bool isElementwiseDim(unsigned dim, const TensorPtr& t) const { return false; };

    void validateGranularityAndOverlap(const TensorPtr& tensor) const;
    void validateNodeDimsMapping(const TensorPtr& tensor) const;
    // validate mapping between any pair of tensors, for which the slicing dims are mapped to each other
    void validateTensorMapping(const TensorPtr& inTensor,
                               const TensorPtr& outTensor,
                               Dim              inputSlicingDim,
                               Dim              outputSlicingDim) const;

    TensorVector m_inputs;
    TensorPtr    m_output;
    NodePtr      m_node;

    Gaudi2Graph                m_graph;
    CompilationHalReaderSetter m_chrSetter {&m_graph};
};

// This test checks access pattern for reshapes that either adds/remove degenerated dimensions (with size 1) around the
// outer dimension, or flattens a few outer dimensions together ([d1, d2, d3, ...] -> [d1*d2*d3, ....] or vice versa).
// In the later case the not flattened side is referred to as the 'element-wise' tensor and the flattened is referred to
// as the 'reshaped' tensor.
class ReshapeAccessPatternTest
: public AccessPatternTest
, public testing::WithParamInterface<ReshapeAccessPatternTestParams>
{
protected:
    using DimParams = ReshapeAccessPatternTestParams::DimParams;

    std::vector<TSize> getOperandShape(const DimParams& dimParam);

    void createNode() override;

    TensorPtr createOperand(std::vector<TSize> shape);

    bool isAllRequiredDim(unsigned dim, const TensorPtr& t) const override;

    Dim getOuterDim(const TensorPtr& tensor, const DimParams& dimParams) const;

    const DimParams& dimParams(const TensorPtr& tensor) const;

    std::tuple<TensorPtr, TensorPtr> getElementWiseAndReshapedTensors() const;

    TensorTile::Size getExpectedFlattenedGranularity() const;

    void validateGranularityAndOverlap(const TensorPtr& tensor, const DimParams& dimParam) const;

    TensorTile::Size getExpectedOuterDimGranularity(const TensorPtr& tensor) const;
};

struct ExpandDimsAccessPatternParams
{
    unsigned           expandDim;
    std::vector<TSize> inputShape;
};

class ExpandDimsAccessPatternTest
: public AccessPatternTest
, public testing::WithParamInterface<ExpandDimsAccessPatternParams>
{
protected:
    void createNode() override;
    bool isAllRequiredDim(unsigned dim, const TensorPtr& t) const override;
    bool isElementwiseDim(unsigned dim, const TensorPtr& t) const override;
};

struct SqueezeAccessPatternParams
{
    std::vector<TSize> inputShape;
    unsigned           numOutputPaddingDims;
    DimVector          squeezeDims;
};

class SqueezeAccessPatternTest
: public AccessPatternTest
, public testing::WithParamInterface<SqueezeAccessPatternParams>
{
protected:
    void createNode() override;
    bool isAllRequiredDim(unsigned dim, const TensorPtr& t) const override;
    bool isElementwiseDim(unsigned dim, const TensorPtr& t) const override;
    bool isSqueezedDim(unsigned dim) const;
    Dim getMatchingOutDim(unsigned inDim) const;
};

struct GemmAccessPatternParams
{
    unsigned                heightA;
    unsigned                commonDim;
    unsigned                widthB;
    std::vector<unsigned>   batchDims;
    bool                    transposeA;
    bool                    transposeB;
    std::optional<unsigned> masksCommonDim;  // set to modify the op to masked bgemm
};

class GemmAccessPatternTest
: public AccessPatternTest
, public testing::WithParamInterface<GemmAccessPatternParams>
{
protected:
    TensorPtr createOperand(std::vector<TSize> shape, bool transposed, bool isMask);

    void createNode() override;

    bool isElementwiseDim(unsigned dim, const TensorPtr& t) const override;

    Dim getOutDimForInput(unsigned inputIdx, Dim inputDim) const;
    Dim getSpatialNonCommonDim(unsigned inputIdx) const;
    Dim getSpatialCommonDim(unsigned inputIdx) const;
};

class BiasedGemmAccessPatternTest : public GemmAccessPatternTest
{
};

struct ConvAccessPatternParams
{
    unsigned          batchDim;
    unsigned          ifmC;
    unsigned          ofmK;
    const std::string guid;
};

class ConvAccessPatternTest
: public AccessPatternTest
, public testing::WithParamInterface<ConvAccessPatternParams>
{
protected:
    void createNode() override;

    bool isAllRequiredDim(unsigned dim, const TensorPtr& t) const override;
    bool isElementwiseDim(unsigned dim, const TensorPtr& t) const override;

    Dim getNonCommonSlicableDim(unsigned inputIdx) const;

    Dim getOutDimForInput(unsigned inputIdx, Dim inputDim) const;

    Dim batchIdxSpcDim() const;
};