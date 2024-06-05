#include "access_pattern_generator.h"
#include "access_pattern.h"
#include "conv_base_node.h"
#include "mme_desc_gen_utils.h"
#include "squeeze_node.h"
#include "tpc_node.h"
#include "mme_node.h"
#include "transpose_utils.h"
#include "tpc_slicing_blacklist.h"
#include "mme_brain_ifc.h"
#include "include/mme_access_pattern.h"

namespace gc::access_pattern
{
NodeAccessPatternPtr AccessPatternFromGlueCodeGenerator::generate(const TPCNode* tpcNode)
{
    HB_ASSERT_PTR(tpcNode);
    HB_ASSERT(tpcNode->isInstantiated(),
              "{} expects an instantiated node, but {} (GUID: {}) was not instantiated.",
              __func__,
              tpcNode->getNodeName(),
              tpcNode->getGUID());

    const auto& instance = tpcNode->getInstance();

    NodeTile::Geometry nodeGeometry(instance.indexSpaceGeometry, instance.indexSpaceGeometry + instance.indexSpaceRank);

    // Find dimensions we cannot slice because of shape inference problems
    DimSet blacklistedIndexSpaceDims = findBlacklistedIndexSpaceDims(tpcNode);

    // Add a dummy node dim at the end of the real node geometry with resolution of 1 -
    // will be used to map the all-required tensor dims that are not mapped to any index-space dim.
    nodeGeometry.push_back(1);

    auto ap = std::make_shared<NodeAccessPattern>(nodeGeometry.begin(), nodeGeometry.end());

    addAllTensorsToAccessPattern(ap, tpcNode->getInputs(), instance.inputTensorAccessPattern, blacklistedIndexSpaceDims);
    addAllTensorsToAccessPattern(ap, tpcNode->getOutputs(), instance.outputTensorAccessPattern, blacklistedIndexSpaceDims);
    addAllShapeTensorsToAccessPattern(ap, tpcNode);
    return ap;
}

void AccessPatternFromGlueCodeGenerator::addAllShapeTensorsToAccessPattern(NodeAccessPatternPtr& ap,
                                                                           const TPCNode*        tpcNode)
{
    // Shape Tensor #index is describing the output tensor #index, as described at "Dynamic Shapes" spec.
    const auto& instance = tpcNode->getInstance();

    TensorVector shapeTensors      = tpcNode->getInputODSTs();  // inputs that are OUTPUT_DESCRIBING_SHAPE_TENSOR
    unsigned     numOfShapeTensors = shapeTensors.size();
    unsigned     accessPatternIdx  = 0;
    unsigned     numOfOutputs      = tpcNode->getNumOutputs();
    for (unsigned outputIdx = 0; outputIdx < numOfOutputs; outputIdx++)
    {
        if (numOfShapeTensors <= outputIdx) break;
        // Shape tensor describing output, that's why it should get output instance
        TensorAccessPatternPtr tensorAP =
            createTensorAccessPattern(shapeTensors[outputIdx],
                                      instance.outputTensorAccessPattern + accessPatternIdx,
                                      ap->getNodeResolution().size() - 1,
                                      DimSet());
        ap->addTensorAccessPattern(shapeTensors[outputIdx], tensorAP);
        accessPatternIdx += TPCNode::numTensorGlueCodeAccessPatternEntries(tpcNode->getOutput(outputIdx));
    }
}

template<typename Container>
void AccessPatternFromGlueCodeGenerator::addAllTensorsToAccessPattern(NodeAccessPatternPtr&  ap,
                                                                      const Container&       tensors,
                                                                      const GlueCodeAP*      glueCodeAccessPatterns,
                                                                      DimSet                 blacklistedIndexSpaceDims)
{
    unsigned accessPatternIdx = 0;
    for (const TensorPtr& tensor : tensors)
    {
        // Shape tensors don't have access patterns filled by glue code
        // so looking at glueCodeAccessPatterns is not valid for them.
        // They are processed in a separate function (above).
        if (tensor->isShapeTensor()) continue;
        TensorAccessPatternPtr tensorAP = createTensorAccessPattern(tensor,
                                                                    glueCodeAccessPatterns + accessPatternIdx,
                                                                    ap->getNodeResolution().size() - 1,
                                                                    blacklistedIndexSpaceDims);
        ap->addTensorAccessPattern(tensor, tensorAP);
        accessPatternIdx += TPCNode::numTensorGlueCodeAccessPatternEntries(tensor);
    }
}

TensorAccessPatternPtr
AccessPatternFromGlueCodeGenerator::createTensorAccessPattern(const TensorPtr&  tensor,
                                                              const GlueCodeAP* glueCodeAccessPatterns,
                                                              Dim               indexSpaceDimForAllRequired,
                                                              DimSet            blacklistedIndexSpaceDims)
{
    if (tensor->isAuxTensor())
    {
        // Aux tensors don't appear in glueCodeAccessPatterns, so need to create
        // a tensor access pattern for them without it.
        return TensorAccessPatternPtr {new GlueCodeTensorAccessPattern(tensor, indexSpaceDimForAllRequired)};
    }
    return TensorAccessPatternPtr {
        new GlueCodeTensorAccessPattern(tensor, glueCodeAccessPatterns, indexSpaceDimForAllRequired, blacklistedIndexSpaceDims)};
}

DimSet
AccessPatternFromGlueCodeGenerator::findBlacklistedIndexSpaceDims(const TPCNode* tpcNode)
{
    if (tpcNode->isDynamicShape())
    {
        return getUnsliceableIndexSpaceDims(*tpcNode);
    }

    return DimSet();
}

struct AllRequiredDimMapping : public GlueCodeTensorAccessPattern::DimMapping
{
    AllRequiredDimMapping(TensorTile::Size tensorDimSize, Dim resolutionDim)
    : m_dimSize(tensorDimSize), m_resolutionDim(resolutionDim)
    {
    }

    DimTile getGranularity() const override { return DimTile {m_dimSize, 0}; }

    DimTile mapNodeRange(const NodeTile& nodeTile) const override { return getGranularity(); }

    Dim getIndexSpaceDim() const override { return m_resolutionDim; }

    const TensorTile::Size m_dimSize;
    const Dim              m_resolutionDim;
};

struct GlueCodeAccessPatternDimMapping : public GlueCodeTensorAccessPattern::DimMapping
{
    GlueCodeAccessPatternDimMapping(TensorTile::Size                         tensorDimSize,
                                    const tpc_lib_api::DimIndexSpaceMapping* dimAccessPattern)
    : m_dimSize(tensorDimSize),
      m_resolutionDim(dimAccessPattern->indexSpaceDim),
      m_a(dimAccessPattern->a),
      m_startB(dimAccessPattern->start_b),
      m_endB(dimAccessPattern->end_b)
    {
        HB_ASSERT(m_a >= 1, "Expected dim stride to be >= 1, but got: {}", m_a);
    }

    DimTile getGranularity() const override { return mapNodeRange(0, 1); }

    DimTile mapNodeRange(const NodeTile& nodeTile) const override
    {
        return mapNodeRange(nodeTile.offset[m_resolutionDim], nodeTile.geometry[m_resolutionDim]);
    }

    void updateNodeTile(NodeTile& nodeTile, DimTile tensorDimTile) const override
    {
        TensorTile::Coord tensorDimOffset = tensorDimTile.offset.front();
        TensorTile::Size  tensorDimSize   = tensorDimTile.geometry.front();
        TensorTile::Coord tensorEnd       = tensorDimOffset + tensorDimSize - 1;

        // tensorStart = nodeOffset * a + start_b
        double nodeStart = (tensorDimOffset - m_startB) / double(m_a);

        // tensorEnd = nodeEnd * a + end_b
        double nodeEnd = (tensorEnd - m_endB) / double(m_a);

        if (nodeStart != std::floor(nodeStart) || nodeEnd != std::floor(nodeEnd) || nodeStart < 0)
        {
            HB_ASSERT(tensorDimTile.geometry.front() == m_dimSize,
                      "Tile does not fit granularity and is not full tensor tile in the dimension");
            return;
        }
        HB_ASSERT(nodeEnd >= nodeStart, "Invalid node geometry (< 1), nodeEnd={}, nodeStart={}", nodeEnd, nodeStart);
        nodeTile.offset[m_resolutionDim]   = nodeStart;
        nodeTile.geometry[m_resolutionDim] = nodeEnd - nodeStart + 1;
    }

    Dim getIndexSpaceDim() const override { return m_resolutionDim; }

private:
    const TensorTile::Size m_dimSize;
    const Dim              m_resolutionDim;
    const float            m_a {0};
    const float            m_startB {0};
    const float            m_endB {0};

    DimTile mapNodeRange(NodeTile::Coord nodeTileOffset, NodeTile::Size nodeTileSize) const
    {
        NodeTile::Coord nodeTileEnd = nodeTileOffset + nodeTileSize - 1;

        double fTensorTileStart = (double)m_a * nodeTileOffset + (double)m_startB;
        double fTensorTileEnd   = (double)m_a * nodeTileEnd + (double)m_endB;
        if (fTensorTileStart > fTensorTileEnd) std::swap(fTensorTileStart, fTensorTileEnd);

        // It is legal for dim mapping not to map to an integer element index. In that case, the expectation is that the
        // tile will be rounded down at the start and up at the end.
        TensorTile::Coord tensorTileStart = std::floor(fTensorTileStart);
        TensorTile::Coord tensorTileEnd   = std::ceil(fTensorTileEnd);

        TensorTile::Size tensorTileSize = tensorTileEnd - tensorTileStart + 1;

        return DimTile {tensorTileSize, tensorTileStart};
    }
};

GlueCodeTensorAccessPattern::GlueCodeTensorAccessPattern(const TensorPtr&  tensor,
                                                         const GlueCodeAP* glueCodeAccessPatterns,
                                                         Dim               indexSpaceDimForAllRequired,
                                                         DimSet            blacklistedIndexSpaceDims)
{
    unsigned accPtrnIdx = 0;
    for (Dim dimBase = 0; dimBase < tensor->getDim(); dimBase += tpc_lib_api::MAX_INDEX_SPACE_DIM_SIZE)
    {
        for (Dim dimOffset = 0; dimOffset < tpc_lib_api::MAX_INDEX_SPACE_DIM_SIZE; dimOffset++)
        {
            if (dimBase + dimOffset >= tensor->getDim()) break;
            mapTensorDimension(tensor,
                               dimBase,
                               dimOffset,
                               glueCodeAccessPatterns[accPtrnIdx],
                               indexSpaceDimForAllRequired,
                               blacklistedIndexSpaceDims);
        }
        accPtrnIdx++;
    }
}

void GlueCodeTensorAccessPattern::mapTensorDimension(const TensorPtr&  tensor,
                                                     Dim               dimBase,
                                                     Dim               dimOffset,
                                                     const GlueCodeAP& glueCodeAccessPattern,
                                                     Dim               indexSpaceDimForAllRequired,
                                                     DimSet            blacklistedIndexSpaceDims)
{
    Dim dim = dimBase + dimOffset;
    if (dim >= tensor->getDim()) return;

    TensorTile::Size dimSize = tensor->getSizeInElements(dim);

    if (areAllDimsAllRequired(glueCodeAccessPattern) ||
        isDimAllRequired(glueCodeAccessPattern.mapping[dimOffset], dimSize) ||
        blacklistedIndexSpaceDims.contains(glueCodeAccessPattern.mapping[dimOffset].indexSpaceDim))
    {
        m_dimMappings.emplace(dim, new AllRequiredDimMapping {dimSize, indexSpaceDimForAllRequired});
    }
    else
    {
        m_dimMappings.emplace(dim,
                              new GlueCodeAccessPatternDimMapping {dimSize, glueCodeAccessPattern.mapping + dimOffset});
    }
}

bool GlueCodeTensorAccessPattern::areAllDimsAllRequired(const GlueCodeAP& glueCodeAP)
{
    return glueCodeAP.allRequired;
}

bool GlueCodeTensorAccessPattern::isDimAllRequired(const GlueCodeDimAP& glueCodeDimAP, TensorTile::Size dimSize)
{
    // A dimension is "all required" if:
    // Same size node tiles map to different size tensor tiles (except maybe edges)
    // Different node tiles map to one tensor tile
    // => In TPC access pattern: "a" is less than 1 (fractional steps may be rounded to the same tensor tile for several
    // different node tiles).
    // A single node resolution element is mapped to the entire tensor
    // => In TPC access pattern: the range from start_b to end_b is larger than the dimension size
    return (glueCodeDimAP.a < 1 || (glueCodeDimAP.end_b - glueCodeDimAP.start_b >= dimSize));
}

GlueCodeTensorAccessPattern::GlueCodeTensorAccessPattern(const TensorPtr& tensor, Dim indexSpaceDimForAllRequired)
{
    for (Dim dim = 0; dim < tensor->getDim(); dim++)
    {
        TensorTile::Size dimSize = tensor->getSizeInElements(dim);
        m_dimMappings.emplace(dim, new AllRequiredDimMapping {dimSize, indexSpaceDimForAllRequired});
    }
}

TensorTile GlueCodeTensorAccessPattern::getGranularity() const
{
    TensorTile minTile(m_dimMappings.size());
    for (const auto& dimAndMapping : m_dimMappings)
    {
        auto dimGranularity         = dimAndMapping.second->getGranularity();
        Dim  tensorDim              = dimAndMapping.first;
        minTile.offset[tensorDim]   = dimGranularity.offset.front();
        minTile.geometry[tensorDim] = dimGranularity.geometry.front();
    }
    return minTile;
}

TensorTile GlueCodeTensorAccessPattern::getTensorTile(const NodeTile& nodeTile) const
{
    TensorTile tile(m_dimMappings.size());
    for (const auto& dimAndMapping : m_dimMappings)
    {
        auto mappedDimTile       = dimAndMapping.second->mapNodeRange(nodeTile);
        Dim  tensorDim           = dimAndMapping.first;
        tile.offset[tensorDim]   = mappedDimTile.offset.front();
        tile.geometry[tensorDim] = mappedDimTile.geometry.front();
    }
    return tile;
}

NodeTile GlueCodeTensorAccessPattern::getNodeTile(const TensorTile&                    tensorTile,
                                                  const NodeAccessPattern::Resolution& nodeResolution) const
{
    NodeTile nodeTile(nodeResolution);

    for (const auto& dimAndMapping : m_dimMappings)
    {
        Dim tensorDim = dimAndMapping.first;
        DimMapping::DimTile tensorDimTile {tensorTile.geometry.at(tensorDim), tensorTile.offset.at(tensorDim)};
        dimAndMapping.second->updateNodeTile(nodeTile, tensorDimTile);
    }
    return nodeTile;
}

Dim GlueCodeTensorAccessPattern::getIndexSpaceDim(Dim tensorDim) const
{
    auto it = m_dimMappings.find(tensorDim);
    HB_ASSERT(it != m_dimMappings.end(), "Failed to find tensor dim mapping");
    return it->second->getIndexSpaceDim();
}

ReshapeTensorAccessPattern::ReshapeTensorAccessPattern(const TensorPtr& tensor,
                                                       Dim              mappedDim,
                                                       NodeTile::Size   numOfGranules)
: m_tensorRank(tensor->getDim()),
  m_mappedDim(mappedDim),
  m_tensorGeometry(m_tensorRank, 1),
  m_granularity(tensor->getSizeInElements(mappedDim) / numOfGranules)
{
    for (Dim dim = 0; dim < m_tensorRank; dim++)
    {
        m_tensorGeometry[dim] = tensor->getSizeInElements(dim);
    }
}

TensorTile ReshapeTensorAccessPattern::getGranularity() const
{
    TensorTile granularity(m_tensorGeometry);
    // allow slicing the node on the external dim to single elements
    granularity.geometry[m_mappedDim] = m_granularity;
    return granularity;
}

TensorTile ReshapeTensorAccessPattern::ReshapeTensorAccessPattern::getTensorTile(const NodeTile& nodeTile) const
{
    TensorTile tensorTile(m_tensorGeometry);
    // Currently only one dimension is "tracked" by the node resolution, so the mapped dim is necessarily mapped to
    // node dim 0.
    tensorTile.geometry[m_mappedDim] = nodeTile.geometry[0] * m_granularity;
    tensorTile.offset[m_mappedDim]   = nodeTile.offset[0] * m_granularity;
    return tensorTile;
}

NodeTile ReshapeTensorAccessPattern::getNodeTile(const TensorTile&         tensorTile,
                                                 const NodeTile::Geometry& nodeResolution) const
{
    NodeTile nodeTile(nodeResolution);
    // Currently only one dimension is "tracked" by the node resolution, so the mapped dim is necessarily mapped to
    // node dim 0.
    nodeTile.geometry[0] = tensorTile.geometry[m_mappedDim] / m_granularity;
    nodeTile.offset[0]   = tensorTile.offset[m_mappedDim] / m_granularity;
    return nodeTile;
}

Dim ReshapeTensorAccessPattern::getIndexSpaceDim(Dim tensorDim) const
{
    // Currently we have 2 dims in the node resolution: the first is used to "track" the mapped dim
    // (outer-most non-degenerate dim) and the second is used to map the rest of the dims
    // (all-required access pattern).
    if (tensorDim == m_mappedDim)
    {
        return 0;
    }
    return 1;
}

template<typename Container>
static Dim findLastNoneDegenerateDim(const Container& dimSizes)
{
    Dim curDim = dimSizes.size() - 1;
    while (curDim > 0)
    {
        if (dimSizes.at(curDim) > 1)
        {
            break;
        }
        curDim--;
    }
    return curDim;
}

bool AccessPatternReshapeGenerator::isValidOuterDimSize(Dim               outerDim,
                                                        const NSizeArray& sizes,
                                                        TensorTile::Size  minOuterDimSize,
                                                        TensorTile::Size  maxOuterDimSize)
{
    if (sizes[outerDim] % minOuterDimSize != 0)
    {
        // Reshape operands outer dims should be a multiple of one another to be valid for access pattern.
        return false;
    }
    TensorTile::Size mult = 1;
    for (int dim = outerDim; dim >= 0; dim--)
    {
        mult *= sizes[dim];
        if (mult >= maxOuterDimSize) break;
    }
    if (mult != maxOuterDimSize)
    {
        // Expect the bigger outer dim to be an aggregation of several dimensions from the other operand to be valid for
        // access pattern.
        return false;
    }
    return true;
}

NodeAccessPatternPtr AccessPatternReshapeGenerator::generate(const ReshapeNode* reshapeNode)
{
    HB_ASSERT_PTR(reshapeNode);

    const auto& input  = reshapeNode->getInput(0);
    const auto& output = reshapeNode->getOutput(0);

    Dim inputOuterDim  = findLastNoneDegenerateDim(input->getAllNSizesInElements());
    TensorTile::Size inputOuterDimSize  = input->getSizeInElements(inputOuterDim);
    Dim outputOuterDim = findLastNoneDegenerateDim(output->getAllNSizesInElements());
    TensorTile::Size outputOuterDimSize = output->getSizeInElements(outputOuterDim);

    auto [minOuterDimSize, maxOuterDimSize] = std::minmax(inputOuterDimSize, outputOuterDimSize);

    NodeTile::Size granularity = minOuterDimSize;

    if (!isValidOuterDimSize(inputOuterDim, input->getAllNSizesInElements(), minOuterDimSize, maxOuterDimSize) ||
        !isValidOuterDimSize(outputOuterDim, output->getAllNSizesInElements(), minOuterDimSize, maxOuterDimSize))
    {
        // Currently - only reshapes where the outer dimensions of the input and output are aggregation of each-other
        // are supported. Other cases will have 'all required' access pattern (until further improvement).
        granularity = 1;
    }

    // Current reshape access pattern allows non-all-required access only to the outer-most non-degenerate dim (assuming
    // it's the same size in input and output), so the node geometry reflects only the size of that dim.
    // Add a dummy node dim at the end of the real node geometry with resolution of 1 -
    // will be used to map the all-required tensor dims.
    NodeTile::Size granularityForAllRequiredDims = 1;
    auto           nodeGeometry                  = {granularity, granularityForAllRequiredDims};

    auto ap = std::make_shared<NodeAccessPattern>(nodeGeometry.begin(), nodeGeometry.end());

    ap->addTensorAccessPattern(
        reshapeNode->getInput(0),
        TensorAccessPatternPtr {new ReshapeTensorAccessPattern(reshapeNode->getInput(0), inputOuterDim, granularity)});

    if (reshapeNode->getNumInputs() > 1)
    {
        // Shape tensor input exist, with the output's geometry
        ap->addTensorAccessPattern(
            reshapeNode->getInput(1),
            TensorAccessPatternPtr {
                new ReshapeTensorAccessPattern(reshapeNode->getInput(1), outputOuterDim, granularity)});
    }

    ap->addTensorAccessPattern(
        reshapeNode->getOutput(0),
        TensorAccessPatternPtr {
            new ReshapeTensorAccessPattern(reshapeNode->getOutput(0), outputOuterDim, granularity)});

    return ap;
}

NodeAccessPatternPtr AccessPatternTransposeGenerator::generate(const Node*                      transposeNode,
                                                               const TransposePermutationArray& permutation)
{
    const TensorPtr& output = transposeNode->getOutput(0);
    HB_ASSERT_PTR(output);

    const auto& outputShape = output->getAllNSizesInElements();
    auto        ap =
        std::make_shared<NodeAccessPattern>(outputShape.begin(), std::next(outputShape.begin(), output->getDim()));

    ap->addTensorAccessPattern(output, TensorAccessPatternPtr(new IdentityTensorAccessPattern(output)));

    const TensorPtr& input = transposeNode->getInput(0);
    // The output tensor dims have 1:1 mapping to node dims.
    // In order to map the input tensor dims need to calculate the inverse permutation -
    // given an input dim find the matching output dim.
    const TransposePermutationArray& inverse = inversePermutation(permutation);
    DimsMapping nodeDimPerTensorDimInput(inverse.begin(), std::next(inverse.begin(), input->getDim()));
    ap->addTensorAccessPattern(
        input,
        TensorAccessPatternPtr(new TransposedTensorAccessPattern(input, permutation, nodeDimPerTensorDimInput)));

    return ap;
}

NodeAccessPatternPtr AccessPatternReductionGenerator::generate(const Node* node)
{
    HB_ASSERT(node->getNumOutputs() == 1,
              "Expected a single output for Reduction node {} but found: {}",
              node->getNodeName(),
              node->getNumOutputs());

    const TensorPtr& output = node->getOutput(0);
    HB_ASSERT_PTR(output);
    const auto& outputShape = output->getAllNSizesInElements();

    auto ap =
        std::make_shared<NodeAccessPattern>(outputShape.begin(), std::next(outputShape.begin(), output->getDim()));

    ap->addTensorAccessPattern(output, TensorAccessPatternPtr(new IdentityTensorAccessPattern(output)));

    for (const TensorPtr& in : node->getInputs())
    {
        ap->addTensorAccessPattern(in, TensorAccessPatternPtr(new IdentityTensorAccessPattern(in)));
    }

    return ap;
}

TensorTile IdentityTensorAccessPattern::getGranularity() const
{
    return TensorTile(m_rank);
}

TensorTile IdentityTensorAccessPattern::getTensorTile(const NodeTile& nodeTile) const
{
    return TensorTile(m_rank, nodeTile.geometry, nodeTile.offset);
}

Dim IdentityTensorAccessPattern::getIndexSpaceDim(Dim tensorDim) const
{
    HB_ASSERT(tensorDim < m_rank, "Invalid tensor dim {}", tensorDim);
    return tensorDim;  // 1:1 mapping
}

NodeTile IdentityTensorAccessPattern::getNodeTile(const TensorTile&         tensorTile,
                                                  const NodeTile::Geometry& nodeResolution) const
{
    return NodeTile(nodeResolution.size(), tensorTile.geometry, tensorTile.offset);
}

TensorTile TransposedTensorAccessPattern::getGranularity() const
{
    return TensorTile(m_rank);
}

TensorTile TransposedTensorAccessPattern::getTensorTile(const NodeTile& nodeTile) const
{
    TensorTile tensorTile(m_rank);
    for (Dim dim = 0; dim < m_rank; dim++)
    {
        tensorTile.geometry[m_permutation.at(dim)] = nodeTile.geometry.at(dim);
        tensorTile.offset[m_permutation.at(dim)]   = nodeTile.offset.at(dim);
    }
    return tensorTile;
}

NodeTile TransposedTensorAccessPattern::getNodeTile(const TensorTile&         tensorTile,
                                                    const NodeTile::Geometry& nodeResolution) const
{
    HB_ASSERT(nodeResolution.size() == m_rank,
              "Unexpected node resolution rank {} (expected: {})",
              nodeResolution.size(),
              m_rank);

    NodeTile nodeTile(nodeResolution.size());
    for (Dim dim = 0; dim < m_rank; dim++)
    {
        nodeTile.geometry[dim] = tensorTile.geometry.at(m_permutation.at(dim));
        nodeTile.offset[dim]   = tensorTile.offset.at(m_permutation.at(dim));
    }
    return nodeTile;
}

Dim TransposedTensorAccessPattern::getIndexSpaceDim(Dim tensorDim) const
{
    HB_ASSERT(tensorDim < m_nodeDimPerTensorDim.size(), "Invalid tensor dim {}", tensorDim);
    return m_nodeDimPerTensorDim.at(tensorDim);
}

NodeAccessPatternPtr AccessPatternExpandDimsGenerator::generate(const Node* expandDimsNode, Dim expandDim)
{
    const TensorPtr& output = expandDimsNode->getOutput(0);
    HB_ASSERT_PTR(output);

    const auto& outputShape = output->getAllNSizesInElements();
    auto        ap =
        std::make_shared<NodeAccessPattern>(outputShape.begin(), std::next(outputShape.begin(), output->getDim()));

    // The expand dims node shape is similar to the output tensor, with 1:1 mapping
    ap->addTensorAccessPattern(output, TensorAccessPatternPtr(new IdentityTensorAccessPattern(output)));

    const TensorPtr& input = expandDimsNode->getInput(0);
    // The expand dims input is missing the expanded dim from the node dims, and has 1:1 mapping for the rest of the
    // dims
    ap->addTensorAccessPattern(input, TensorAccessPatternPtr(new ExpandDimsTensorAccessPattern(input, expandDim)));

    return ap;
}

TensorTile ExpandDimsTensorAccessPattern::getGranularity() const
{
    return TensorTile(m_rank);
}

TensorTile ExpandDimsTensorAccessPattern::getTensorTile(const NodeTile& nodeTile) const
{
    // The expand dims input is missing the expanded dim from the node dims, and has 1:1 mapping for the rest of the
    // dims. Copy all dims but the expanded dim from the node and erase expand dim to shift back the higher dims
    TensorTile tensorTile(nodeTile.geometry.size(), nodeTile.geometry, nodeTile.offset);
    HB_ASSERT(m_rank + 1 == nodeTile.geometry.size(),
              "expand dims input rank {} must be smaller than node rank {} by 1",
              m_rank,
              nodeTile.geometry.size());
    tensorTile.geometry.erase(tensorTile.geometry.begin() + m_expandDim);
    tensorTile.offset.erase(tensorTile.offset.begin() + m_expandDim);
    return tensorTile;
}

NodeTile ExpandDimsTensorAccessPattern::getNodeTile(const TensorTile&         tensorTile,
                                                    const NodeTile::Geometry& nodeResolution) const
{
    // The expand dims input is missing the expanded dim from the node dims, and has 1:1 mapping for the rest of the
    // dims. Copy all dims from the tensor and insert the expanded dim and create the shift.
    NodeTile nodeTile(m_rank, tensorTile.geometry, tensorTile.offset);
    HB_ASSERT(m_rank + 1 == nodeResolution.size(),
              "expand dims input rank {} must be smaller than node rank {} by 1",
              m_rank,
              nodeResolution.size());
    // Using the insert function version of n elements of value, as the version
    // without n fails to compile for small vector (works fine on std::vector)
    nodeTile.geometry.insert(nodeTile.geometry.begin() + m_expandDim, 1, 1ULL);
    nodeTile.offset.insert(nodeTile.offset.begin() + m_expandDim, 1, 0ULL);
    return nodeTile;
}

Dim ExpandDimsTensorAccessPattern::getIndexSpaceDim(Dim tensorDim) const
{
    HB_ASSERT(tensorDim < m_rank, "Invalid tensor dim {}", tensorDim);
    // The expand dims input is missing the expanded dim from the node dims,
    // and has 1:1 mapping for the rest of the dims.
    return (tensorDim < m_expandDim) ? tensorDim : tensorDim + 1;
}

bool AccessPatternSqueezeGenerator::isSqueezedDim(DimVector squeezeDims, Dim dim)
{
    return std::find(squeezeDims.begin(), squeezeDims.end(), dim) != squeezeDims.end();
}

NodeAccessPatternPtr AccessPatternSqueezeGenerator::generate(const Node* squeezeNode, DimVector squeezeDims)
{
    const TensorPtr& input = squeezeNode->getInput(0);
    const TensorPtr& output = squeezeNode->getOutput(0);
    HB_ASSERT_PTR(input);
    HB_ASSERT_PTR(output);

    const auto& inputShape = input->getAllNSizesInElements();
    unsigned realDims = input->getDim() - squeezeDims.size(); // Represents dimensions in the input that have a mapping to a dim in the output
    unsigned paddingDims = output->getDim() - realDims; // Represents the number of 1 "padding" in the output (outer dims)
    unsigned geometrySize = realDims + paddingDims + squeezeDims.size();

    NodeTile::Geometry nodeGeometry(geometrySize, 1);
    DimsMapping        nodeDimPerTensorDimInput;
    DimsMapping        nodeDimPerTensorDimOutput;

    HB_ASSERT(output->getDim() >= realDims && output->getDim() <= input->getDim(),
              "{}'s output rank should be >= number of real dims and <= input's rank",
              squeezeNode->getNodeName());

    // The node geometry is 1 for all the squeezed dims and the independent output dims, so filling in just the non
    // squeezed dims as their size taken from the input
    unsigned inTensorDim = 0;
    for (; inTensorDim < input->getDim(); inTensorDim++)
    {
        unsigned nodeDim = inTensorDim;  // The node's first dims correspond to the input's dims with 1:1 mapping
        if (!isSqueezedDim(squeezeDims, nodeDim))
        {
            nodeDimPerTensorDimOutput.push_back(nodeDim);
        }
        nodeGeometry[nodeDim] = inputShape[nodeDim];
        nodeDimPerTensorDimInput.push_back(nodeDim);
    }

    for (unsigned outTensorDim = inTensorDim; outTensorDim < inTensorDim + paddingDims; outTensorDim++)
    {
        unsigned nodeDim = outTensorDim;
        nodeGeometry[nodeDim] = 1;
        nodeDimPerTensorDimOutput.push_back(nodeDim);
    }

    auto ap = std::make_shared<NodeAccessPattern>(nodeGeometry.begin(), nodeGeometry.end());
    ap->addTensorAccessPattern(input, TensorAccessPatternPtr(new SqueezeTensorAccessPattern(input, squeezeDims, nodeDimPerTensorDimInput)));
    ap->addTensorAccessPattern(output, TensorAccessPatternPtr(new SqueezeTensorAccessPattern(output, squeezeDims, nodeDimPerTensorDimOutput)));

    return ap;
}

TensorTile SqueezeTensorAccessPattern::getGranularity() const
{
    return TensorTile(m_rank);
}

bool SqueezeTensorAccessPattern::isSqueezedDim(Dim dim) const
{
    return std::find(m_squeezeDims.begin(), m_squeezeDims.end(), dim) != m_squeezeDims.end();
}

TensorTile SqueezeTensorAccessPattern::getTensorTile(const NodeTile& nodeTile) const
{
    TensorTile tensorTile(m_rank, 1, 0);
    for (Dim tensorDim = 0; tensorDim < m_rank; tensorDim++)
    {
        Dim nodeDim = m_nodeDimPerTensorDim[tensorDim];
        tensorTile.geometry[tensorDim] = nodeTile.geometry[nodeDim];
        tensorTile.offset[tensorDim] = nodeTile.offset[nodeDim];
    }

    return tensorTile;
}

NodeTile SqueezeTensorAccessPattern::getNodeTile(const TensorTile&         tensorTile,
                                                 const NodeTile::Geometry& nodeResolution) const
{
    NodeTile nodeTile(nodeResolution.size(), 1, 0);
    for (Dim tensorDim = 0; tensorDim < m_rank; tensorDim++)
    {
        Dim nodeDim = m_nodeDimPerTensorDim[tensorDim];
        nodeTile.geometry[nodeDim] = tensorTile.geometry[tensorDim];
        nodeTile.offset[nodeDim] = tensorTile.offset[tensorDim];
    }

    return nodeTile;
}

Dim SqueezeTensorAccessPattern::getIndexSpaceDim(Dim tensorDim) const
{
    HB_ASSERT((tensorDim < m_rank) && (tensorDim < m_nodeDimPerTensorDim.size()), "Invalid tensor dim {}", tensorDim);
    return m_nodeDimPerTensorDim[tensorDim];
}

GemmTensorAccessPattern::GemmTensorAccessPattern(const TensorPtr& tensor,
                                                 DimsMapping      nodeDimPerTensorDim,
                                                 bool             transposed,
                                                 Dim              indexSpaceDimForBroadcastedBatchDims)
: m_tensorRank(tensor->getDim()),
  m_tensorGeometry(m_tensorRank, 1),
  m_nodeDimPerTensorDim(std::move(nodeDimPerTensorDim)),
  m_indexSpaceDimForBroadcastedBatchDims(indexSpaceDimForBroadcastedBatchDims)
{
    for (Dim dim = 0; dim < m_tensorRank; dim++)
    {
        m_tensorGeometry[dim] = tensor->getSizeInElements(dim);
    }
    if (transposed)
    {
        std::swap(m_nodeDimPerTensorDim[0], m_nodeDimPerTensorDim[1]);
    }
}

TensorTile GemmTensorAccessPattern::getGranularity() const
{
    // Allow slicing the tensor to any size
    return TensorTile(m_tensorRank, 1, 0);
}

TensorTile GemmTensorAccessPattern::getTensorTile(const NodeTile& nodeTile) const
{
    TensorTile tensorTile(m_tensorRank, 1, 0);
    for (Dim tensorDim = 0; tensorDim < m_tensorRank; tensorDim++)
    {
        // For spatial dims, or batch dims which are not broadcasted - set 1:1 mapping
        if (!isBroadcastedBatchDim(tensorDim))
        {
            Dim nodeDim = m_nodeDimPerTensorDim.at(tensorDim);
            // 1:1 mapping
            tensorTile.geometry[tensorDim] = nodeTile.geometry[nodeDim];
            tensorTile.offset[tensorDim]   = nodeTile.offset[nodeDim];
        }
        // Else - the dim remains with the default geometry 1 offset 0
    }
    // The tensor tile is not clipped to the actual tensor size
    return tensorTile;
}

NodeTile GemmTensorAccessPattern::getNodeTile(const TensorTile&         tensorTile,
                                              const NodeTile::Geometry& nodeResolution) const
{
    NodeTile nodeTile(nodeResolution);
    for (Dim tensorDim = 0; tensorDim < tensorTile.offset.size(); tensorDim++)
    {
        HB_ASSERT(tensorTile.offset[tensorDim] >= 0, "Negative tensor tile offset not expected for gemm");
        // For spatial dims, or batch dims which are not broadcasted - set 1:1 mapping
        if (!isBroadcastedBatchDim(tensorDim))
        {
            Dim nodeDim                = m_nodeDimPerTensorDim.at(tensorDim);
            nodeTile.offset[nodeDim]   = tensorTile.offset[tensorDim];
            nodeTile.geometry[nodeDim] = tensorTile.geometry[tensorDim];
        }
        // Else - the dim is remains with the node geometry and offset
    }
    return nodeTile;
}

Dim GemmTensorAccessPattern::getIndexSpaceDim(Dim tensorDim) const
{
    HB_ASSERT(tensorDim < m_tensorRank, "Invalid tensor dim {}", tensorDim);
    if (isBroadcastedBatchDim(tensorDim))
    {
        return m_indexSpaceDimForBroadcastedBatchDims;
    }
    HB_ASSERT(tensorDim < m_nodeDimPerTensorDim.size(), "Missing mapping for tensor dim {}", tensorDim);
    return m_nodeDimPerTensorDim[tensorDim];
}

bool GemmTensorAccessPattern::isBroadcastedBatchDim(Dim tensorDim) const
{
    return (tensorDim >= DIM_GEMM_BATCH && m_tensorGeometry[tensorDim] == 1);
}

NodeAccessPatternPtr AccessPatternGemmGenerator::generate(const GEMMNode* gemmNode)
{
    HB_ASSERT_PTR(gemmNode);
    // For now supporting only FWD ops
    if (gemmNode->getNodeType() != Node::TYPE_GEMM && gemmNode->getNodeType() != Node::TYPE_BATCH_GEMM &&
        gemmNode->getNodeType() != Node::TYPE_MASKED_BATCH_GEMM)
    {
        return nullptr;
    }
    const auto& gemmParams = gemmNode->getGEMMParams();

    // Set node geometry dims (ignoring transposes, they are handled in the tensor AP)
    // GemmNodeDims is used to access the node geometry. DimsIndex enum is used to access the tensors dims.
    enum GemmNodeDims
    {
        NODE_DIM_OPERANDS_COMMON,      // Operands common dim (width A, height B)
        NODE_DIM_MASKS_COMMON,         // Masks common dim (width maskA, height maskB)
        NODE_DIM_OUT_WIDTH,            // Output width and B width
        NODE_DIM_OUT_HEIGHT,           // Output height  and A height
        NODE_DIM_NUM_IDENTICAL_MASKS,  // Masks internal batch dim
        NODE_DIM_DEGENERATED_BATCH,    // Broadcasted batch dims or dummies
        NODE_DIM_BATCH_BASE            // The rest of the dims are batch dims
    };

    // Geometry size = GemmNodeDims count + batch dims taken from output
    NodeTile::Geometry nodeGeometry(NODE_DIM_BATCH_BASE + (gemmNode->getOutput(0)->getDim() - DIM_GEMM_BATCH), 0);
    DimsMapping        nodeDimPerTensorDimOperandA   = {NODE_DIM_OPERANDS_COMMON, NODE_DIM_OUT_HEIGHT};
    DimsMapping        nodeDimPerTensorDimOperandB   = {NODE_DIM_OUT_WIDTH, NODE_DIM_OPERANDS_COMMON};
    DimsMapping        nodeDimPerTensorDimOutOperand = {NODE_DIM_OUT_WIDTH, NODE_DIM_OUT_HEIGHT};

    nodeGeometry[NODE_DIM_OPERANDS_COMMON] =
        gemmNode->getInput(TENSOR_IFM)->getSizeInElements(gemmParams.transpose_a ? DIM_W : DIM_C);
    nodeGeometry[NODE_DIM_OUT_WIDTH]  = gemmNode->getOutput(0)->getSizeInElements(DIM_C);
    nodeGeometry[NODE_DIM_OUT_HEIGHT] = gemmNode->getOutput(0)->getSizeInElements(DIM_W);
    nodeGeometry[NODE_DIM_DEGENERATED_BATCH] = 1;

    for (Dim dim = DIM_GEMM_BATCH; dim < gemmNode->getOutput(0)->getDim(); dim++)
    {
        Dim nodeDim = NODE_DIM_BATCH_BASE + (dim - DIM_GEMM_BATCH);
        nodeDimPerTensorDimOperandA.push_back(nodeDim);
        nodeDimPerTensorDimOperandB.push_back(nodeDim);
        nodeDimPerTensorDimOutOperand.push_back(nodeDim);
        nodeGeometry[nodeDim] = gemmNode->getOutput(0)->getSizeInElements(dim);
    }

    if (gemmNode->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
    {
        nodeGeometry[NODE_DIM_MASKS_COMMON] =
            gemmNode->getInput(TENSOR_AUX_BGEMM_MASK_A)->getSizeInElements(gemmParams.transpose_a ? DIM_W : DIM_C);
    }

    auto ap = std::make_shared<NodeAccessPattern>(nodeGeometry.begin(), nodeGeometry.end());

    ap->addTensorAccessPattern(gemmNode->getInput(TENSOR_IFM),
                               TensorAccessPatternPtr {new GemmTensorAccessPattern(gemmNode->getInput(TENSOR_IFM),
                                                                                   nodeDimPerTensorDimOperandA,
                                                                                   gemmParams.transpose_a,
                                                                                   NODE_DIM_DEGENERATED_BATCH)});
    ap->addTensorAccessPattern(gemmNode->getInput(TENSOR_WEIGHT),
                               TensorAccessPatternPtr {new GemmTensorAccessPattern(gemmNode->getInput(TENSOR_WEIGHT),
                                                                                   nodeDimPerTensorDimOperandB,
                                                                                   gemmParams.transpose_b,
                                                                                   NODE_DIM_DEGENERATED_BATCH)});
    ap->addTensorAccessPattern(gemmNode->getOutput(0),
                               TensorAccessPatternPtr {new GemmTensorAccessPattern(gemmNode->getOutput(0),
                                                                                   nodeDimPerTensorDimOutOperand,
                                                                                   false,
                                                                                   NODE_DIM_DEGENERATED_BATCH)});
    if ((gemmNode->getNodeType() == Node::TYPE_BATCH_GEMM || gemmNode->getNodeType() == Node::TYPE_GEMM) &&
        gemmNode->hasBias())
    {
        DimsMapping nodeDimPerTensorDimBiasOperand = {NODE_DIM_OUT_WIDTH};
        ap->addTensorAccessPattern(gemmNode->getInput(TENSOR_BIAS),
                                   TensorAccessPatternPtr {new GemmTensorAccessPattern(gemmNode->getInput(TENSOR_BIAS),
                                                                                       nodeDimPerTensorDimBiasOperand,
                                                                                       false,
                                                                                       NODE_DIM_DEGENERATED_BATCH)});
    }

    if (gemmNode->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
    {
        DimsMapping nodeDimPerTensorDimMaskA = {NODE_DIM_MASKS_COMMON,
                                                NODE_DIM_OUT_HEIGHT,
                                                NODE_DIM_NUM_IDENTICAL_MASKS};
        DimsMapping nodeDimPerTensorDimMaskB = {NODE_DIM_OUT_WIDTH,
                                                NODE_DIM_MASKS_COMMON,
                                                NODE_DIM_NUM_IDENTICAL_MASKS};
        // The first (internal) batch dim for the masks is num identical masks, the rest are the same as the output
        for (Dim dim = DIM_GEMM_BATCH + 1; dim < gemmNode->getOutput(0)->getDim(); dim++)
        {
            Dim nodeDim = NODE_DIM_BATCH_BASE + (dim - DIM_GEMM_BATCH);
            nodeDimPerTensorDimMaskA.push_back(nodeDim);
            nodeDimPerTensorDimMaskB.push_back(nodeDim);
        }

        ap->addTensorAccessPattern(
            gemmNode->getInput(TENSOR_AUX_BGEMM_MASK_A),
            TensorAccessPatternPtr {new GemmTensorAccessPattern(gemmNode->getInput(TENSOR_AUX_BGEMM_MASK_A),
                                                                nodeDimPerTensorDimMaskA,
                                                                gemmParams.transpose_a,
                                                                NODE_DIM_DEGENERATED_BATCH)});
        ap->addTensorAccessPattern(
            gemmNode->getInput(TENSOR_AUX_BGEMM_MASK_B),
            TensorAccessPatternPtr {new GemmTensorAccessPattern(gemmNode->getInput(TENSOR_AUX_BGEMM_MASK_B),
                                                                nodeDimPerTensorDimMaskB,
                                                                gemmParams.transpose_b,
                                                                NODE_DIM_DEGENERATED_BATCH)});
    }

    return ap;
}

ConvTensorAccessPattern::ConvTensorAccessPattern(const TensorPtr& tensor,
                                                 DimsMapping      nodeDimPerTensorDim,
                                                 DimsMapping      allRequiredNodeDims)
: m_tensorRank(tensor->getDim()),
  m_tensorGeometry(m_tensorRank, 1),
  m_nodeDimPerTensorDim(std::move(nodeDimPerTensorDim)),
  m_allRequiredNodeDims(allRequiredNodeDims)
{
    for (Dim dim = 0; dim < m_tensorRank; dim++)
    {
        m_tensorGeometry[dim] = tensor->getSizeInElements(dim);
    }
}

inline bool ConvTensorAccessPattern::isAllRequiredDim(Dim nodeDim) const
{
    return std::count(m_allRequiredNodeDims.begin(), m_allRequiredNodeDims.end(), nodeDim) != 0;
}

TensorTile ConvTensorAccessPattern::getGranularity() const
{
    TensorTile granularity(m_tensorGeometry);
    for (Dim tensorDim = 0; tensorDim < m_tensorRank; tensorDim++)
    {
        Dim nodeDim = m_nodeDimPerTensorDim[tensorDim];
        // if nodeDim is not an allRequired dim, then it can be sliced  (granularity is 1)
        if (!isAllRequiredDim(nodeDim))
        {
            granularity.geometry[tensorDim] = 1;
        }
    }
    return granularity;
}

TensorTile ConvTensorAccessPattern::getTensorTile(const NodeTile& nodeTile) const
{
    TensorTile tensorTile(m_tensorRank, 1, 0);
    for (Dim tensorDim = 0; tensorDim < m_tensorRank; tensorDim++)
    {
        Dim nodeDim = m_nodeDimPerTensorDim[tensorDim];
        if (isAllRequiredDim(nodeDim))
        {
            tensorTile.geometry[tensorDim] = m_tensorGeometry[tensorDim];
            HB_ASSERT(nodeTile.geometry[nodeDim] == 1,
                      "allRequired dim must have only 1 element in the nodeTile geometry");
            HB_ASSERT(nodeTile.offset[nodeDim] == 0,
                      "allRequired dim's offset must be 0 (because there is only one element)");
        }
        else
        {
            // Set 1:1 mapping for non allRequired dims
            tensorTile.geometry[tensorDim] = nodeTile.geometry[nodeDim];
            tensorTile.offset[tensorDim]   = nodeTile.offset[nodeDim];
        }
    }
    // The tensor tile is not clipped to the actual tensor size
    return tensorTile;
}

NodeTile ConvTensorAccessPattern::getNodeTile(const TensorTile&         tensorTile,
                                              const NodeTile::Geometry& nodeResolution) const
{
    NodeTile nodeTile(nodeResolution);
    for (Dim tensorDim = 0; tensorDim < m_tensorRank; tensorDim++)
    {
        Dim nodeDim = m_nodeDimPerTensorDim[tensorDim];
        if (isAllRequiredDim(nodeDim))
        {
            nodeTile.geometry[nodeDim] = 1;
            nodeTile.offset[nodeDim]   = 0;
            HB_ASSERT(tensorTile.geometry[tensorDim] >= m_tensorGeometry[tensorDim],
                      "allRequired dim size must be at least the full tensor dim size");
        }
        else
        {
            // Set 1:1 mapping for non allRequired dims
            HB_ASSERT(tensorTile.offset[tensorDim] >= 0, "Negative tensor tile offset not expected for sliceable dims");
            nodeTile.offset[nodeDim]   = tensorTile.offset[tensorDim];
            nodeTile.geometry[nodeDim] = tensorTile.geometry[tensorDim];
        }
    }
    return nodeTile;
}

Dim ConvTensorAccessPattern::getIndexSpaceDim(Dim tensorDim) const
{
    HB_ASSERT(tensorDim < m_nodeDimPerTensorDim.size(), "Invalid tensor dim {}", tensorDim);
    return m_nodeDimPerTensorDim[tensorDim];
}

NodeAccessPatternPtr AccessPatternConvGenerator::generate(const ConvBaseNode* convNode)
{
    HB_ASSERT_PTR(convNode);
    enum ConvNodeDims
    {
        NODE_DIM_B,
        NODE_DIM_D,
        NODE_DIM_H,
        NODE_DIM_W,
        NODE_DIM_C,
        NODE_DIM_K,
        NODE_DIM_Q,
        NODE_DIM_R,
        NODE_DIM_S,
        NODE_DIM_MAX
    };
    // Set dim-mapping between tensor dims and node dims
    NodeTile::Geometry nodeGeometry(NODE_DIM_MAX);
    DimsMapping        nodeDimPerTensorDimOperandX;
    DimsMapping        nodeDimPerTensorDimOperandW;
    DimsMapping        nodeDimPerTensorDimOperandY;
    if (convNode->is3DConvolution())
    {
        nodeDimPerTensorDimOperandX = {NODE_DIM_C, NODE_DIM_W, NODE_DIM_H, NODE_DIM_D, NODE_DIM_B};
        nodeDimPerTensorDimOperandW = {NODE_DIM_K, NODE_DIM_C, NODE_DIM_S, NODE_DIM_R, NODE_DIM_Q};
        nodeDimPerTensorDimOperandY = {NODE_DIM_K, NODE_DIM_W, NODE_DIM_H, NODE_DIM_D, NODE_DIM_B};
    }
    else
    {
        nodeDimPerTensorDimOperandX = {NODE_DIM_C, NODE_DIM_W, NODE_DIM_H, NODE_DIM_B};
        nodeDimPerTensorDimOperandW = {NODE_DIM_K, NODE_DIM_C, NODE_DIM_S, NODE_DIM_R};
        nodeDimPerTensorDimOperandY = {NODE_DIM_K, NODE_DIM_W, NODE_DIM_H, NODE_DIM_B};
    }
    if (convNode->getNodeType() == Node::TYPE_TRANSPOSED_DEDX)
    {
        std::swap(nodeDimPerTensorDimOperandW[0], nodeDimPerTensorDimOperandW[1]);
    }
    // Set node geometry
    // TODO [SW-76583] Extend conv access pattern to support spatial slicing
    nodeGeometry[NODE_DIM_B] = convNode->is3DConvolution()
                                   ? convNode->getYOperand()->getSizeInElements(DIM_B_FOR_5D_TENSOR)
                                   : convNode->getYOperand()->getSizeInElements(DIM_B);
    nodeGeometry[NODE_DIM_D] = 1;
    nodeGeometry[NODE_DIM_H] = 1;
    nodeGeometry[NODE_DIM_W] = 1;
    nodeGeometry[NODE_DIM_C] = convNode->getXOperand()->getSizeInElements(DIM_C);
    nodeGeometry[NODE_DIM_K] = convNode->getYOperand()->getSizeInElements(DIM_C);
    nodeGeometry[NODE_DIM_Q] = 1;
    nodeGeometry[NODE_DIM_R] = 1;
    nodeGeometry[NODE_DIM_S] = 1;

    // Initialize allRequiredDims according to nodeGeometry dim values
    std::vector<Dim> allRequiredDims;
    for (unsigned dim = 0; dim < NODE_DIM_MAX; dim++)
    {
        if (nodeGeometry[dim] == 1) allRequiredDims.push_back(dim);
    }

    auto ap = std::make_shared<NodeAccessPattern>(nodeGeometry.begin(), nodeGeometry.end());

    ap->addTensorAccessPattern(
        convNode->getXOperand(),
        TensorAccessPatternPtr {
            new ConvTensorAccessPattern(convNode->getXOperand(), nodeDimPerTensorDimOperandX, allRequiredDims)});
    ap->addTensorAccessPattern(
        convNode->getWOperand(),
        TensorAccessPatternPtr {
            new ConvTensorAccessPattern(convNode->getWOperand(), nodeDimPerTensorDimOperandW, allRequiredDims)});
    ap->addTensorAccessPattern(
        convNode->getYOperand(),
        TensorAccessPatternPtr {
            new ConvTensorAccessPattern(convNode->getYOperand(), nodeDimPerTensorDimOperandY, allRequiredDims)});

    auto shapeOperand = convNode->getShapeOperand();
    auto nodeType     = convNode->getNodeType();

    HB_ASSERT(shapeOperand == nullptr ||
                  (shapeOperand && (nodeType == Node::TYPE_DEDX || nodeType == Node::TYPE_TRANSPOSED_DEDX)),
              "Access pattern for Node {} doesn't support shape tensor",
              convNode->getNodeName());

    if (shapeOperand)
    {
        // dedx node (checked in the above assert )
        ap->addTensorAccessPattern(
            shapeOperand,
            TensorAccessPatternPtr {
                new ConvTensorAccessPattern(shapeOperand, nodeDimPerTensorDimOperandX, allRequiredDims)});
    }
    return ap;
}

class MmeTensorAccessPattern : public TensorAccessPattern
{
public:
    MmeTensorAccessPattern(const MmeCommon::AccessPattern::TensorAccessPattern& tap, size_t rank)
    : m_mmeTAP(tap), m_rank(rank)
    {
    }

    TensorTile getGranularity() const override
    {
        TensorTile granule(m_rank);
        for (Dim dim = 0; dim < m_rank; dim++)
        {
            granule.geometry.at(dim) = m_mmeTAP.dimsAccessPattern.at(dim).size;
        }
        return granule;
    }

    TensorTile getTensorTile(const NodeTile& nodeTile) const override
    {
        TensorTile tensorTile(m_rank);
        for (Dim dim = 0; dim < m_rank; dim++)
        {
            const auto& dimAP   = m_mmeTAP.dimsAccessPattern.at(dim);
            auto        nodeDim = dimAP.indexSpaceDim;

            auto nodeOffset           = nodeTile.offset.at(nodeDim);
            tensorTile.offset.at(dim) = dimAP.offset + dimAP.stride * nodeOffset;

            auto nodeSize               = nodeTile.geometry.at(nodeDim);
            tensorTile.geometry.at(dim) = dimAP.size * nodeSize;
        }
        return tensorTile;
    }

    NodeTile getNodeTile(const TensorTile& tensorTile, const NodeTile::Geometry& nodeResolution) const override
    {
        NodeTile nodeTile(nodeResolution);
        for (Dim dim = 0; dim < m_rank; dim++)
        {
            const auto& dimAP = m_mmeTAP.dimsAccessPattern.at(dim);
            if (dimAP.stride == 0) continue;  // not invertible.
            HB_ASSERT(dimAP.size != 0, "Unexpected 0 size access pattern to tensor dim {}", dim);

            auto nodeDim = dimAP.indexSpaceDim;

            auto tensorOffset = tensorTile.offset.at(dim);
            HB_ASSERT((tensorOffset - dimAP.offset) % dimAP.stride == 0,
                      "Tensor tile is not in an offset which is a multiple of the dimension stride (dim={}, stride={}, "
                      "offset={}, dim-base-offset={})",
                      dim,
                      dimAP.stride,
                      tensorOffset,
                      dimAP.offset);
            nodeTile.offset.at(nodeDim) = (tensorOffset - dimAP.offset) / dimAP.stride;

            auto tensorSize = tensorTile.geometry.at(dim);
            HB_ASSERT(tensorSize % dimAP.size == 0,
                      "Tensor tile size is not a multiple of the granularity (dim={}, granularity={}, size={})",
                      dim,
                      dimAP.size,
                      tensorSize);
            nodeTile.geometry.at(nodeDim) = tensorSize / dimAP.size;
        }
        return nodeTile;
    }

    Dim getIndexSpaceDim(Dim tensorDim) const override
    {
        return m_mmeTAP.dimsAccessPattern.at(tensorDim).indexSpaceDim;
    }

private:
    const MmeCommon::AccessPattern::TensorAccessPattern m_mmeTAP;
    const size_t                                        m_rank;
};

class MmeAccessPatternGeneratorImpl
{
public:
    explicit MmeAccessPatternGeneratorImpl(const MmeNode* mmeNode)
    : m_mmeNode(mmeNode), m_mmeAP(MmeBrainIfc::generateAccessPattern(mmeNode))
    {
    }

    NodeAccessPatternPtr generate()
    {
        validateNode();
        auto ap = createNodeAP();
        addOperandsAP(ap);
        return ap;
    }

private:
    const MmeNode*                 m_mmeNode;
    const MmeCommon::AccessPattern m_mmeAP;

    using Role = MmeCommon::OperandRole;

    void validateNode() const
    {
        HB_ASSERT(hasRoleAP(Role::X), "Missing X access pattern");
        HB_ASSERT(hasRoleAP(Role::W), "Missing W access pattern");
        HB_ASSERT(hasRoleAP(Role::Y), "Missing Y access pattern");

        HB_ASSERT(hasBias() || hasShape() || isMaskedBgemm() || !m_mmeNode->getInput(2),
                  "Unsupported input[2] in access pattern generation. MME Node: {}, Tensor: {}",
                  m_mmeNode->getNodeName(),
                  m_mmeNode->getInput(2)->getName());
        HB_ASSERT(isMaskedBgemm() || !m_mmeNode->getInput(3),
                  "Unsupported input[3] in access pattern generation. MME Node: {}, Tensor: {}",
                  m_mmeNode->getNodeName(),
                  m_mmeNode->getInput(3)->getName());
        for (unsigned idx = 4; idx <= 5; idx++)
        {
            HB_ASSERT(!m_mmeNode->getInput(idx) || m_mmeNode->getInput(idx)->isAuxTensor(),
                      "Unsupported input[{}}] in access pattern generation. MME Node: {}, Tensor: {}",
                      idx,
                      m_mmeNode->getNodeName(),
                      m_mmeNode->getInput(idx)->getName());
        }
    }

    NodeAccessPatternPtr createNodeAP() const
    {
        return std::make_shared<NodeAccessPattern>(m_mmeAP.indexSpace.begin(), m_mmeAP.indexSpace.end());
    }

    void addTensorAP(NodeAccessPatternPtr& ap, const TensorPtr& tensor, Role role) const
    {
        ap->addTensorAccessPattern(tensor,
                                   TensorAccessPatternPtr(new MmeTensorAccessPattern(roleAP(role), tensor->getDim())));
    }

    void addOperandsAP(NodeAccessPatternPtr& ap) const
    {
        // Using Gaudi3 since the only reasons for the chip type is hacks in gaudi1 and operations that are only
        // supported in Gaudi3. The access pattern is agnostic to these limitations and the tensor roles would be the
        // same either way.
        auto      chip   = MmeCommon::ChipType::e_mme_Gaudi3;
        auto      opType = getOperationTypeCommon(chip, *m_mmeNode);
        TensorPtr x, w, y, o;
        getTensorRolesCommon(*m_mmeNode, opType, x, w, y, o);

        addTensorAP(ap, x, Role::X);
        addTensorAP(ap, w, Role::W);
        addTensorAP(ap, y, Role::Y);

        if (hasBias()) addBias(ap);
        if (hasShape()) addShape(ap);
        if (isMaskedBgemm()) addMasks(ap);
        if (hasOutputCopy()) addOutputCopy(ap);
        if (hasAux()) addAux(ap);
    }

    bool hasBias() const { return m_mmeNode->hasBias(); }

    void addBias(NodeAccessPatternPtr& ap) const
    {
        // Hack to get over the fact the MME stack doesn't recognize or handle bias tensors
        const TensorPtr& bias = m_mmeNode->getInput(TENSOR_BIAS);
        validateBias(bias);
        addTensorAP(ap, bias, Role::BIAS);
    }

    void validateBias(const TensorPtr& bias) const
    {
        HB_ASSERT_PTR(bias);
        HB_ASSERT(bias->getDim() == 1, "Unexpected dimensionality of bias input: {} (expected 1)", bias->getDim());
        HB_ASSERT(bias->getSizeInElements(0) == m_mmeNode->getOutput(0)->getSizeInElements(0),
                  "Unexpected shape of bias tensor: [{}], expected: [{}]",
                  bias->getSizeInElements(0),
                  m_mmeNode->getOutput(0)->getSizeInElements(0));
    }

    bool isMaskedBgemm() const { return m_mmeNode->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM; }

    void addMasks(NodeAccessPatternPtr& ap) const
    {
        const TensorPtr& maskA = m_mmeNode->getInput(TENSOR_AUX_BGEMM_MASK_A);
        const TensorPtr& maskB = m_mmeNode->getInput(TENSOR_AUX_BGEMM_MASK_B);
        validateMasks(maskA, maskB);
        addTensorAP(ap, maskA, Role::MASK_A);
        addTensorAP(ap, maskB, Role::MASK_B);
    }

    void validateMasks(const TensorPtr& maskA, const TensorPtr& maskB) const
    {
        HB_ASSERT_PTR(maskA);
        HB_ASSERT_PTR(maskB);
    }

    bool hasShape() const { return (getConvNode() && getConvNode()->getShapeOperand() != nullptr); }

    void addShape(NodeAccessPatternPtr& ap) const
    {
        const TensorPtr& shapeOperand = getConvNode()->getShapeOperand();
        validateShape(shapeOperand);
        addTensorAP(ap, shapeOperand, Role::SHAPE);
    }

    void validateShape(const TensorPtr& shapeOperand) const
    {
        const TensorPtr& output = m_mmeNode->getOutput(0);
        HB_ASSERT(shapeOperand->isShapeTensor(), "Shape operand is not a shape tensor");
        HB_ASSERT(shapeOperand->getDim() == output->getDim() && shapeOperand->compareGeometry(*output),
                  "Unexpected: Shape tensor and output have different shapes");
    }

    mutable std::optional<const ConvBaseNode*> m_convNode {};  // cache for expansive dynamic cast of mmeNode to Conv
    const ConvBaseNode*                        getConvNode() const
    {
        if (!m_convNode.has_value())
        {
            m_convNode = dynamic_cast<const ConvBaseNode*>(m_mmeNode);
        }
        return m_convNode.value();
    }

    bool hasOutputCopy() const { return m_mmeNode->getOutput(TENSOR_SECONDARY_OFM) != nullptr; }

    void addOutputCopy(NodeAccessPatternPtr& ap) const
    {
        const TensorPtr& outputCopy = m_mmeNode->getOutput(TENSOR_SECONDARY_OFM);
        addTensorAP(ap, outputCopy, Role::OUTPUT_COPY);
    }

    bool hasAux() const
    {
        return m_mmeNode->getInput(TENSOR_AUX_CD_SCRATCHPAD) &&
               m_mmeNode->getInput(TENSOR_AUX_CD_SCRATCHPAD)->isAuxTensor() &&
               m_mmeNode->getInput(TENSOR_AUX_CD_REDUCTION) &&
               m_mmeNode->getInput(TENSOR_AUX_CD_REDUCTION)->isAuxTensor();
    }

    void addAux(NodeAccessPatternPtr& ap) const
    {
        const TensorPtr& scratchPad = m_mmeNode->getInput(TENSOR_AUX_CD_SCRATCHPAD);
        const TensorPtr& reduction  = m_mmeNode->getInput(TENSOR_AUX_CD_REDUCTION);
        addTensorAP(ap, scratchPad, Role::SCRATCH_PAD);
        addTensorAP(ap, reduction, Role::CONST);
    }

    bool hasRoleAP(Role r) const
    {
        return m_mmeAP.operandAccessPatterns.find(r) != m_mmeAP.operandAccessPatterns.end();
    }

    const MmeCommon::AccessPattern::TensorAccessPattern& roleAP(Role r) const
    {
        return m_mmeAP.operandAccessPatterns.at(r);
    }
};

NodeAccessPatternPtr AccessPatternMmeNodeGenerator::generate(const MmeNode* mmeNode)
{
    HB_ASSERT_PTR(mmeNode);
    return MmeAccessPatternGeneratorImpl(mmeNode).generate();
}

}  // namespace gc::access_pattern
