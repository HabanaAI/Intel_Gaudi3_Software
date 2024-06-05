#pragma once

#include "access_pattern.h"
#include "habana_nodes.h"
#include "tpc_slicing_blacklist.h"

namespace gc::access_pattern
{
using GlueCodeAP    = tpc_lib_api::TensorAccessPattern;
using GlueCodeDimAP = tpc_lib_api::DimIndexSpaceMapping;
using DimsMapping =
    std::vector<Dim>;  // Maps the vector's idx which represents the tensorDim, to a value that represents nodeDim

class AccessPatternFromGlueCodeGenerator
{
public:
    static NodeAccessPatternPtr generate(const TPCNode* tpcNode);

private:
    template<typename Container>
    static void addAllTensorsToAccessPattern(NodeAccessPatternPtr& ap,
                                      const Container&       tensors,
                                      const GlueCodeAP*      glueCodeAccessPatterns,
                                      DimSet                 blacklistedIndexSpaceDims);

    static void addAllShapeTensorsToAccessPattern(NodeAccessPatternPtr& ap, const TPCNode* tpcNode);

    static TensorAccessPatternPtr createTensorAccessPattern(const TensorPtr&  tensor,
                                                     const GlueCodeAP* glueCodeAccessPatterns,
                                                     Dim               indexSpaceDimForAllRequired,
                                                     DimSet            blacklistedIndexSpaceDims);

    static DimSet findBlacklistedIndexSpaceDims (const TPCNode* tpcNode);
};

class GlueCodeTensorAccessPattern : public TensorAccessPattern
{
public:
    struct DimMapping
    {
        virtual ~DimMapping() = default;
        // Single dimension "1D tile"
        struct DimTile : public TensorTile
        {
            DimTile(TensorTile::Size dimSize, TensorTile::Coord dimOffset) : TensorTile(1, dimSize, dimOffset) {}
        };

        // Get the tensor minimal tile size and offset in the dimension
        virtual DimTile getGranularity() const = 0;
        // Maps the node tile to an offset and size in the tensor dimension
        virtual DimTile mapNodeRange(const NodeTile& nodeTile) const = 0;
        // Since mapping tensor tile to node tile is not always possible, this method signature lets dim mappings that
        // can deduce the node tile from the tensor tile do so, and other dim mapping to simply not have any effect.
        virtual void updateNodeTile(NodeTile& nodeTile, DimTile tensorDimTile) const
        {
            // do nothing be default
        }
        // Get the index-space dim that this tensor dim is mapped to.
        virtual Dim getIndexSpaceDim() const = 0;
    };

    // For tensors that are marked as "all-required" - all the dims will be mapped to indexSpaceDimForAllRequired.
    GlueCodeTensorAccessPattern(const TensorPtr&  tensor,
                                const GlueCodeAP* glueCodeAccessPatterns,
                                Dim               indexSpaceDimForAllRequired,
                                DimSet            blacklistedDimensions = DimSet());

    // Used for tensors without glue code access pattern entry (such as aux tensors)
    // All the tensor dims will be mapped to indexSpaceDimForAllRequired.
    explicit GlueCodeTensorAccessPattern(const TensorPtr& tensor, Dim indexSpaceDimForAllRequired);

    // Interface implementation
    TensorTile getGranularity() const override;
    TensorTile getTensorTile(const NodeTile& nodeTile) const override;
    NodeTile   getNodeTile(const TensorTile&                    tensorTile,
                           const NodeAccessPattern::Resolution& nodeResolution) const override;
    Dim        getIndexSpaceDim(Dim tensorDim) const override;

private:
    std::unordered_map<Dim, std::unique_ptr<DimMapping>> m_dimMappings;

    // Add mapping for a specific dimension
    void mapTensorDimension(const TensorPtr& tensor,
                            Dim dimBase,  // Dimension is split to base and offset for high rank implementation.
                            Dim dimOffset,
                            const GlueCodeAP& glueCodeAccessPattern,
                            Dim               indexSpaceDimForAllRequired,
                            DimSet            blacklistedDims);

    // test if the entire dimension bundle is all required (indicates the the singular dimension glue code access
    // pattern is invalid)
    static bool areAllDimsAllRequired(const GlueCodeAP& glueCodeAP);
    // test if a dimension's access pattern is such that every node tile should map to the full tensor size in
    // the dimension.
    static bool isDimAllRequired(const GlueCodeDimAP& glueCodeDimAP, TensorTile::Size dimSize);
};

class AccessPatternReshapeGenerator
{
public:
    static NodeAccessPatternPtr generate(const ReshapeNode* reshapeNode);

private:
    static bool isValidOuterDimSize(Dim               outerDim,
                                    const NSizeArray& sizes,
                                    TensorTile::Size  minOuterDimSize,
                                    TensorTile::Size  maxOuterDimSize);
};

class ReshapeTensorAccessPattern : public TensorAccessPattern
{
public:
    ReshapeTensorAccessPattern(const TensorPtr& tensor, Dim mappedDim, NodeTile::Size numOfGranules);

    TensorTile getGranularity() const override;
    TensorTile getTensorTile(const NodeTile& nodeTile) const override;
    NodeTile   getNodeTile(const TensorTile& tensorTile, const NodeTile::Geometry& nodeResolution) const override;
    Dim        getIndexSpaceDim(Dim tensorDim) const override;

private:
    const Dim            m_tensorRank;
    const Dim            m_mappedDim;
    TensorTile::Geometry m_tensorGeometry;
    TensorTile::Size     m_granularity;
};

class AccessPatternTransposeGenerator
{
public:
    static NodeAccessPatternPtr generate(const Node* transposeNode, const TransposePermutationArray& permutation);
};

// Can be used when the tensor and the node resolution have the same shape and the tensor elements are mapped 1:1 to the
// not resolution.
class IdentityTensorAccessPattern : public TensorAccessPattern
{
public:
    explicit IdentityTensorAccessPattern(const TensorPtr& tensor) : m_rank(tensor->getDim()) {}

    TensorTile getGranularity() const override;
    TensorTile getTensorTile(const NodeTile& nodeTile) const override;
    NodeTile   getNodeTile(const TensorTile& tensorTile, const NodeTile::Geometry& nodeResolution) const override;
    Dim        getIndexSpaceDim(Dim tensorDim) const override;

private:
    Dim m_rank;
};

class AccessPatternReductionGenerator
{
public:
    static NodeAccessPatternPtr generate(const Node* node);
};

class TransposedTensorAccessPattern : public TensorAccessPattern
{
public:
    explicit TransposedTensorAccessPattern(const TensorPtr&                 tensor,
                                           const TransposePermutationArray& permutation,
                                           DimsMapping                      nodeDimPerTensorDim)
    : m_rank(tensor->getDim()),
      m_permutation(permutation.begin(), std::next(permutation.begin(), m_rank)),
      m_nodeDimPerTensorDim(nodeDimPerTensorDim)
    {
    }

    TensorTile getGranularity() const override;
    TensorTile getTensorTile(const NodeTile& nodeTile) const override;
    NodeTile   getNodeTile(const TensorTile& tensorTile, const NodeTile::Geometry& nodeResolution) const override;
    Dim        getIndexSpaceDim(Dim tensorDim) const override;

private:
    const Dim              m_rank;
    const std::vector<Dim> m_permutation;
    DimsMapping            m_nodeDimPerTensorDim;
};

class AccessPatternExpandDimsGenerator
{
public:
    static NodeAccessPatternPtr generate(const Node* expandDimsNode, Dim expandDim);
};

class ExpandDimsTensorAccessPattern : public TensorAccessPattern
{
public:
    explicit ExpandDimsTensorAccessPattern(const TensorPtr& tensor, Dim expandDim)
    : m_rank(tensor->getDim()), m_expandDim(expandDim)
    {
    }

    TensorTile getGranularity() const override;
    TensorTile getTensorTile(const NodeTile& nodeTile) const override;
    NodeTile   getNodeTile(const TensorTile& tensorTile, const NodeTile::Geometry& nodeResolution) const override;
    Dim        getIndexSpaceDim(Dim tensorDim) const override;

private:
    Dim m_rank;
    Dim m_expandDim;
};

class AccessPatternSqueezeGenerator
{
public:
    static NodeAccessPatternPtr generate(const Node* node, DimVector squeezeDims);
private:
    static bool isSqueezedDim(DimVector squeezeDims, Dim dim);
};

class SqueezeTensorAccessPattern : public TensorAccessPattern
{
public:
    SqueezeTensorAccessPattern(const TensorPtr& tensor, DimVector squeezeDims, DimsMapping nodeDimPerTensorDim)
    : m_rank(tensor->getDim()), m_squeezeDims(squeezeDims), m_nodeDimPerTensorDim(nodeDimPerTensorDim)
    {
    }

    TensorTile getGranularity() const override;
    TensorTile getTensorTile(const NodeTile& nodeTile) const override;
    NodeTile   getNodeTile(const TensorTile& tensorTile, const NodeTile::Geometry& nodeResolution) const override;
    Dim        getIndexSpaceDim(Dim tensorDim) const override;

private:
    bool isSqueezedDim(Dim dim) const;

    Dim         m_rank;
    DimVector   m_squeezeDims;
    DimsMapping m_nodeDimPerTensorDim;
};

class AccessPatternGemmGenerator
{
public:
    static NodeAccessPatternPtr generate(const GEMMNode* mmeNode);
};

// Access pattern for GEMM nodes hierarchy - GEMM/BGEMM FWD/BWD
class GemmTensorAccessPattern : public TensorAccessPattern
{
public:
    GemmTensorAccessPattern(const TensorPtr& tensor,
                            DimsMapping      nodeDimPerTensorDim,
                            bool             transposed,
                            Dim              indexSpaceDimForBroadcastedBatchDims);

    TensorTile getGranularity() const override;
    TensorTile getTensorTile(const NodeTile& nodeTile) const override;
    NodeTile   getNodeTile(const TensorTile& tensorTile, const NodeTile::Geometry& nodeResolution) const override;
    Dim        getIndexSpaceDim(Dim tensorDim) const override;

private:
    bool isBroadcastedBatchDim(Dim tensorDim) const;

    const Dim            m_tensorRank;
    TensorTile::Geometry m_tensorGeometry;
    DimsMapping          m_nodeDimPerTensorDim;
    const Dim            m_indexSpaceDimForBroadcastedBatchDims;
};

class ConvTensorAccessPattern : public TensorAccessPattern
{
public:
    ConvTensorAccessPattern(const TensorPtr& tensor, DimsMapping nodeDimPerTensorDim, DimsMapping allRequiredDims);

    TensorTile getGranularity() const override;
    TensorTile getTensorTile(const NodeTile& nodeTile) const override;
    NodeTile   getNodeTile(const TensorTile& tensorTile, const NodeTile::Geometry& nodeResolution) const override;
    Dim        getIndexSpaceDim(Dim tensorDim) const override;

private:
    bool                 isAllRequiredDim(Dim nodeDim) const;
    const Dim            m_tensorRank;
    TensorTile::Geometry m_tensorGeometry;
    DimsMapping          m_nodeDimPerTensorDim;
    DimsMapping          m_allRequiredNodeDims;
};

// Access pattern for CONV nodes hierarchy - CONV FWD/BWD
class AccessPatternConvGenerator
{
public:
    static NodeAccessPatternPtr generate(const ConvBaseNode* convNode);
};

// Wrapper for access pattern provided by MME stack
class AccessPatternMmeNodeGenerator
{
public:
    static NodeAccessPatternPtr generate(const MmeNode* mmeNode);
};

}  // namespace gc::access_pattern
