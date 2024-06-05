#include "access_pattern_test.h"
#include "access_pattern_generator.h"
#include "conv_base_node.h"
#include "gaudi2_graph.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "transpose_node.h"
#include "access_pattern_generator.cpp"
#include "brain/slicer/node_dcore_rois_setter.h"

using namespace gc::access_pattern;

using Range = DataRange<int>;

static TensorTile tensorTileFromTensor(const TensorPtr& tensor)
{
    Dim rank = tensor->getDim();
    return TensorTile(rank, tensor->getAllNSizesInElements(), TensorTile::Offset(rank, 0));
}

class GlueCodeAccessPatternTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<unsigned  // Num dimensions
                                       >
{
public:
    unsigned m_numDims = GetParam();
};

INSTANTIATE_TEST_SUITE_P(tensor_access_pattern, GlueCodeAccessPatternTest, testing::Range(1u, 9u));

TEST_P(GlueCodeAccessPatternTest, all_required_tensor_access_pattern_test)
{
    std::vector<TSize> sizes(m_numDims, 1024);

    auto tensor = std::make_shared<Tensor>(m_numDims, sizes.data(), syn_type_float);

    GlueCodeAP tensorAccessPatterns[2];
    tensorAccessPatterns[0].allRequired = true;
    tensorAccessPatterns[1].allRequired = true;

    auto                 tensorSize = tensor->getAllNSizesInElements();
    TensorTile::Geometry fullGeometry(tensorSize.begin(), std::next(tensorSize.begin(), tensor->getDim()));
    TensorTile::Offset   zeroOffset(tensor->getDim(), 0);

    Dim indexSpaceDimForAllRequired = 0;  // Dummy index-space dim to map tensor dims that are all required
    TensorAccessPatternPtr allReqAP(
        new GlueCodeTensorAccessPattern(tensor, tensorAccessPatterns, indexSpaceDimForAllRequired));

    for (auto tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
    {
        EXPECT_EQ(allReqAP->getIndexSpaceDim(tensorDim), indexSpaceDimForAllRequired);
    }

    TensorTile granularity = allReqAP->getGranularity();
    EXPECT_EQ(granularity.geometry, fullGeometry);
    EXPECT_EQ(granularity.offset, zeroOffset);

    NodeTile   nodeTile(NodeTile::Geometry {1, 2}, NodeTile::Offset {3, 4});
    TensorTile tensorTile = allReqAP->getTensorTile(nodeTile);
    EXPECT_EQ(tensorTile.geometry, fullGeometry);
    EXPECT_EQ(tensorTile.offset, zeroOffset);

    NodeTile::Geometry dummyNodeGeometry = {9, 8, 7, 6};
    NodeTile           mappedNodeTile    = allReqAP->getNodeTile(granularity, dummyNodeGeometry);
    EXPECT_EQ(mappedNodeTile.geometry, dummyNodeGeometry);
    EXPECT_EQ(mappedNodeTile.offset, NodeTile::Offset(dummyNodeGeometry.size(), 0));
}

TEST_P(GlueCodeAccessPatternTest, glue_code_tensor_sliceable_access_pattern_test)
{
    const TSize windowSize    = 128;
    const TSize tensorDimSize = 8 * windowSize;
    const int   dimOffset     = -10;

    std::vector<TSize> sizes(m_numDims, 8 * windowSize);

    auto tensor = std::make_shared<Tensor>(m_numDims, sizes.data(), syn_type_float);

    auto tensorSize = tensor->getAllNSizesInElements();

    for (Dim dim = 0; dim < m_numDims; dim++)
    {
        GlueCodeAP tensorAccessPatterns[2];
        tensorAccessPatterns[0].allRequired = true;
        tensorAccessPatterns[1].allRequired = true;
        auto* pTensorAccessPattern          = tensorAccessPatterns;

        Dim baseDim = 0;
        if (dim >= SYN_MAX_TENSOR_DIM)
        {
            baseDim = SYN_MAX_TENSOR_DIM;
            pTensorAccessPattern++;
        }

        pTensorAccessPattern->allRequired = false;
        for (Dim dimIdx = 0; dimIdx < SYN_MAX_TENSOR_DIM; dimIdx++)
        {
            pTensorAccessPattern->mapping[dimIdx].indexSpaceDim = dimIdx;
            pTensorAccessPattern->mapping[dimIdx].a             = 0;
            pTensorAccessPattern->mapping[dimIdx].start_b       = 0;
            pTensorAccessPattern->mapping[dimIdx].end_b         = tensorDimSize - 1;
        }

        pTensorAccessPattern->mapping[dim - baseDim].a       = windowSize;
        pTensorAccessPattern->mapping[dim - baseDim].start_b = dimOffset;
        pTensorAccessPattern->mapping[dim - baseDim].end_b   = windowSize + dimOffset - 1;

        Dim indexSpaceDimForAllRequired = 0;  // Dummy index-space dim to map tensor dims that are all required
        GlueCodeTensorAccessPattern tap(tensor, tensorAccessPatterns, indexSpaceDimForAllRequired);

        for (auto tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
        {
            EXPECT_EQ(tap.getIndexSpaceDim(tensorDim),
                      (dim == tensorDim) ? (tensorDim % SYN_MAX_TENSOR_DIM) : indexSpaceDimForAllRequired);
        }

        TensorTile granularity = tap.getGranularity();

        TensorTile::Geometry expTileGeometry(tensorSize.begin(), std::next(tensorSize.begin(), tensor->getDim()));
        expTileGeometry[dim] = windowSize;
        TensorTile::Offset expTileOffset(tensor->getDim(), 0);
        expTileOffset[dim] = dimOffset;

        EXPECT_EQ(granularity.geometry, expTileGeometry);
        EXPECT_EQ(granularity.offset, expTileOffset);

        // node tile size is 2x2x2x2x2 at offset 3x3x3x3x3, but that affects only the dimension with varied access
        // pattern (dim)
        NodeTile nodeTile(NodeTile::Geometry {2, 2, 2, 2, 2}, NodeTile::Offset {3, 3, 3, 3, 3});

        TensorTile tensorTile = tap.getTensorTile(nodeTile);
        expTileGeometry[dim]  = TensorTile::Size(2 * windowSize);
        expTileOffset[dim]    = TensorTile::Coord(3 * windowSize + dimOffset);
        EXPECT_EQ(tensorTile.geometry, expTileGeometry);
        EXPECT_EQ(tensorTile.offset, expTileOffset);

        // Now change the tensor tile to have 4 windows sized tile at 5 windows offset in the varied acces dimension.
        tensorTile.geometry[dim] = TensorTile::Size(4 * windowSize);
        tensorTile.offset[dim]   = TensorTile::Coord(5 * windowSize + dimOffset);

        // And given a 10x10x10x10x10 node geometry, when mapping the tensor tile to it,
        nodeTile = tap.getNodeTile(tensorTile, NodeTile::Geometry {10, 10, 10, 10, 10});

        // We expect to see size=4 and offset=5 at the dimension mapped to the varied access dimension and size=10,
        // offset=0 for the rest.
        NodeTile::Geometry expNodeTileGeometry {10, 10, 10, 10, 10};
        NodeTile::Offset   expNodeTileOffset {0, 0, 0, 0, 0};
        expNodeTileGeometry[dim - baseDim] = 4;
        expNodeTileOffset[dim - baseDim]   = 5;
        EXPECT_EQ(expNodeTileGeometry, nodeTile.geometry);
        EXPECT_EQ(expNodeTileOffset, nodeTile.offset);
    }
}

TEST_P(GlueCodeAccessPatternTest, glue_code_tensor_unsliceable_access_pattern_test)
{
    const TensorTile::Size       dimensionSize = 128;
    std::vector<TSize>           sizes(m_numDims, dimensionSize);
    tpc_lib_api::TensorAccessPattern tensorAccessPatterns[2] {};

    tensorAccessPatterns[0].allRequired = false;
    tensorAccessPatterns[1].allRequired = true;

    // 1st dimension window size depends on the offset
    tensorAccessPatterns[0].mapping[0].a = 0;

    // 2nd dimension has fractional access pattern
    tensorAccessPatterns[0].mapping[1].a       = 0.5;
    tensorAccessPatterns[0].mapping[1].start_b = 0;
    tensorAccessPatterns[0].mapping[1].end_b   = 4;

    // 3rd dimension has constant access pattern
    tensorAccessPatterns[0].mapping[2].a       = 0;
    tensorAccessPatterns[0].mapping[2].start_b = 10;
    tensorAccessPatterns[0].mapping[2].end_b   = 127;

    const auto indexSpaceDim = 5;
    for (Dim dimIdx = 0; dimIdx < SYN_MAX_TENSOR_DIM; dimIdx++)
    {
        tensorAccessPatterns[0].mapping[dimIdx].indexSpaceDim = indexSpaceDim;
    }

    auto tensor = std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);

    Dim indexSpaceDimForAllRequired = 0;  // Dummy index-space dim to map tensor dims that are all required
    GlueCodeTensorAccessPattern tap(tensor, tensorAccessPatterns, indexSpaceDimForAllRequired);

    for (auto tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
    {
        EXPECT_EQ(tap.getIndexSpaceDim(tensorDim), indexSpaceDimForAllRequired);
    }

    TensorTile           granularity = tap.getGranularity();
    TensorTile::Offset   expOffset(sizes.size(), 0);
    TensorTile::Geometry expGeometry(sizes.begin(), sizes.end());
    EXPECT_EQ(granularity.geometry, expGeometry);
    EXPECT_EQ(granularity.offset, expOffset);

    NodeTile   nodeTile(NodeTile::Geometry {1, 2, 3, 4}, NodeTile::Offset {6, 7, 8, 9});
    TensorTile mappedTensorTile = tap.getTensorTile(nodeTile);
    EXPECT_EQ(mappedTensorTile.geometry, expGeometry);
    EXPECT_EQ(mappedTensorTile.offset, expOffset);

    NodeTile mappedNodeTile = tap.getNodeTile(granularity, nodeTile.geometry);
    EXPECT_EQ(mappedNodeTile.geometry, nodeTile.geometry);
    EXPECT_EQ(mappedNodeTile.offset, NodeTile::Offset(nodeTile.offset.size(), 0));
}

class ParamsHolder
{
public:
    ParamsHolder() {}

    template<typename UserParamsType>
    ParamsHolder(const UserParamsType& userParams)
    {
        m_paramsBuffer.resize(sizeof(userParams));
        std::memcpy(m_paramsBuffer.data(), &userParams, sizeof(userParams));
    }

    void*  ptr() const { return m_paramsBuffer.data(); }
    size_t size() const { return m_paramsBuffer.size(); }

private:
    mutable std::vector<uint8_t> m_paramsBuffer;
};

struct AccessPatternGeneratorTestParams
{
    const char* guid;
    unsigned    numInputs;
    unsigned    numOutputs;
    unsigned    numDims;
    ParamsHolder userParams;
};

class TPCAccessPatternGeneratorTest
: public GraphOptimizerTest
, public testing::WithParamInterface<AccessPatternGeneratorTestParams>
{
protected:
    Gaudi2Graph  m_graph;
    NodePtr      m_node;
    TensorVector m_inputs;
    TensorVector m_outputs;

    TPCNode* createAndInstantiateNode()
    {
        std::vector<TSize> sizes(GetParam().numDims, 64);
        sizes[0] = 1024;

        for (int idx = 0; idx < GetParam().numInputs; idx++)
        {
            m_inputs.push_back(std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float));
        }
        for (int idx = 0; idx < GetParam().numOutputs; idx++)
        {
            m_outputs.push_back(std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float));
        }

        m_node          = NodeFactory::createNode(m_inputs,
                                         m_outputs,
                                         GetParam().userParams.ptr(),
                                         GetParam().userParams.size(),
                                         GetParam().guid,
                                         GetParam().guid);
        auto* asTPCNode = dynamic_cast<TPCNode*>(m_node.get());
        if (!asTPCNode) return nullptr;

        auto retVal = asTPCNode->init(tpc_lib_api::DEVICE_ID_GAUDI,
                                      &m_graph.getGraphAnnotation().cachedAuxiliaryTensors,
                                      m_graph.getNextTPCKernelUniqueId());
        if (retVal != tpc_lib_api::GLUE_SUCCESS) return nullptr;

        return asTPCNode;
    }

    void validateGranularity(const NodeAccessPatternPtr&                   accessPatternPtr,
                             const tpc_lib_api::HabanaKernelInstantiation& instance) const
    {
        validateTensorsGranularity(m_node->getInputs(), accessPatternPtr, instance.inputTensorAccessPattern);
        validateTensorsGranularity(m_node->getOutputs(), accessPatternPtr, instance.outputTensorAccessPattern);
    }

    void validateTensorsGranularity(const TensorVector&                     tensors,
                                    const NodeAccessPatternPtr&             accessPatternPtr,
                                    const tpc_lib_api::TensorAccessPattern* glueCodeTensorAPs) const
    {
        unsigned tensorIdx = 0;
        for (const TensorPtr& tensor : tensors)
        {
            TensorTile  granularity      = accessPatternPtr->getTensorGranularity(tensor);
            if (tensor->isAuxTensor())
            {
                LOG_DEBUG(GO_TEST, "Validating granularity of an aux tensor in node {}", m_node->getNodeName());
                TensorTile fullTile(tensor->getDim(),
                                    tensor->getAllNSizesInElements(),
                                    TensorTile::Offset(tensor->getDim(), 0));
                EXPECT_EQ(granularity.geometry, fullTile.geometry);
                EXPECT_EQ(granularity.offset, fullTile.offset);
                continue;
            }
            const auto& tensorGlueCodeAP = glueCodeTensorAPs[tensorIdx];
            for (Dim dim = 0; dim < tensor->getDim(); dim++)
            {
                const auto&       dimAccessPattern = tensorGlueCodeAP.mapping[dim];
                TensorTile::Size  tensorDimSize    = dimAccessPattern.end_b - dimAccessPattern.start_b + 1;
                TensorTile::Coord tensorDimOffset  = dimAccessPattern.start_b;
                EXPECT_EQ(granularity.geometry[dim], tensorDimSize);
                EXPECT_EQ(granularity.offset[dim], tensorDimOffset);
            }

            tensorIdx++;
        }
    }

    void validateOverlap(const NodeAccessPatternPtr&                   accessPatternPtr,
                         const tpc_lib_api::HabanaKernelInstantiation& instance) const
    {
        validateTensorsOverlap(m_node->getInputs(), accessPatternPtr, instance.inputTensorAccessPattern);
        validateTensorsOverlap(m_node->getOutputs(), accessPatternPtr, instance.outputTensorAccessPattern);
    }

    void validateTensorsOverlap(const TensorVector&                     tensors,
                                const NodeAccessPatternPtr&             accessPatternPtr,
                                const tpc_lib_api::TensorAccessPattern* glueCodeTensorAPs) const
    {
        for (int i = 0; i < tensors.size(); i++)
        {
            const TensorPtr& tensor        = tensors[i];
            IntersectionTile tensorOverlap = accessPatternPtr->getTensorOverlap(tensor);
            if (tensor->isAuxTensor())
            {
                validateAuxTensorOverlap(tensor, tensorOverlap);
            }
            else
            {
                validateTensorOverlap(tensor, tensorOverlap, glueCodeTensorAPs[i]);
            }
        }
    }

    void validateAuxTensorOverlap(const TensorPtr& tensor, const IntersectionTile& tensorOverlap) const
    {
        // Aux tensors have no access pattern in glue code and are
        // all-required, so they would have full overlap between adjacent
        // index space elements
        IntersectionTile expOverlap(tensor->getDim(),
                                    tensor->getAllNSizesInElements(),
                                    IntersectionTile::Offset(tensor->getDim(), 0));
        EXPECT_EQ(expOverlap.geometry, tensorOverlap.geometry);
        EXPECT_EQ(expOverlap.offset, tensorOverlap.offset);
    }

    void validateTensorOverlap(const TensorPtr&                        tensor,
                               const IntersectionTile&                 tensorOverlap,
                               const tpc_lib_api::TensorAccessPattern& tensorGlueCodeAP) const
    {
        for (Dim dim = 0; dim < tensor->getDim(); dim++)
        {
            auto        dimOverlap       = tensorOverlap.geometry[dim];
            const auto& dimAccessPattern = tensorGlueCodeAP.mapping[dim];
            validateTensorDimOverlap(dimOverlap, dimAccessPattern);
        }
    }

    void validateTensorDimOverlap(IntersectionTile::Size                   dimOverlap,
                                  const tpc_lib_api::DimIndexSpaceMapping& dimAccessPattern) const
    {
        // Overlap_size = index[0].end - index[1].start + 1
        // + 1 is because end is the last element index. If index[0] ends in element 127 and index[1] starts at element
        // 128, we want the overlap to be 0 (127 - 128 + 1). If index[0] ends in element 128 and index[1] starts in
        // element 128, i.e. they are sharing element 128, then we want the overlap to be 1 (128 - 128 + 1)
        auto                   index0End     = 0 * (double)dimAccessPattern.a + dimAccessPattern.end_b;
        auto                   index1Start   = 1 * (double)dimAccessPattern.a + dimAccessPattern.start_b;
        IntersectionTile::Size expDimOverlap = index0End - index1Start + 1;
        EXPECT_EQ(dimOverlap, expDimOverlap);
    }

    void validateTileMapping(const NodeTile&                               nodeTile,
                             const NodeAccessPatternPtr&                   accessPatternPtr,
                             const tpc_lib_api::HabanaKernelInstantiation& instance) const
    {
        validateAllTensorsTileMapping(nodeTile, m_inputs, accessPatternPtr, instance.inputTensorAccessPattern);
        validateAllTensorsTileMapping(nodeTile, m_outputs, accessPatternPtr, instance.outputTensorAccessPattern);
    }

    void validateAllTensorsTileMapping(const NodeTile&                         nodeTile,
                                       const TensorVector&                     tensors,
                                       const NodeAccessPatternPtr&             accessPatternPtr,
                                       const tpc_lib_api::TensorAccessPattern* glueCodeTensorAPs) const
    {
        unsigned tensorIdx = 0;
        for (const auto& tensor : tensors)
        {
            TensorTile tensorTile = accessPatternPtr->getTensorTile(tensor, nodeTile);
            ASSERT_EQ(tensorTile.geometry.size(), tensor->getDim());
            ASSERT_EQ(tensorTile.offset.size(), tensor->getDim());

            for (Dim dim = 0; dim < tensor->getDim(); dim++)
            {
                const auto&       dimAP          = glueCodeTensorAPs[tensorIdx].mapping[dim];
                Dim               nodeDim        = dimAP.indexSpaceDim;
                NodeTile::Coord   nodeStart      = nodeTile.offset[nodeDim];
                NodeTile::Coord   nodeEnd        = nodeTile.offset[nodeDim] + nodeTile.geometry[nodeDim] - 1;
                TensorTile::Coord tensorDimStart = std::floor((double)dimAP.a * nodeStart + dimAP.start_b);
                TensorTile::Coord tensorDimEnd   = std::ceil((double)dimAP.a * nodeEnd + dimAP.end_b);
                EXPECT_EQ(tensorTile.offset[dim], tensorDimStart);
                EXPECT_EQ(tensorTile.geometry[dim], tensorDimEnd - tensorDimStart + 1);
            }

            NodeTile mappedNodeTile = accessPatternPtr->getNodeTile(tensor, tensorTile);
            EXPECT_EQ(mappedNodeTile.geometry, nodeTile.geometry);
            EXPECT_EQ(mappedNodeTile.offset, nodeTile.offset);

            ++tensorIdx;
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    node_access_pattern,
    TPCAccessPatternGeneratorTest,
    testing::Values(
        //                               guid, numInputs, numOutputs, numDims
        AccessPatternGeneratorTestParams {"relu_fwd_f32", 1, 1, 4},
        AccessPatternGeneratorTestParams {"add_f32", 2, 1, 3},
        AccessPatternGeneratorTestParams {"sub_bwd_f32", 1, 2, 1},
        AccessPatternGeneratorTestParams {"max_bwd_f32", 3, 2, 2},
        AccessPatternGeneratorTestParams {"softmax_fwd_f32", 1, 1, 4, ParamsHolder(ns_Softmax::ParamsV3 {})}));

TEST_P(TPCAccessPatternGeneratorTest, generated_access_pattern_provide_granularity_and_overlap)
{
    auto* tpcNode = createAndInstantiateNode();
    ASSERT_NE(nullptr, tpcNode);

    auto accessPatternPtr = AccessPatternFromGlueCodeGenerator::generate(tpcNode);
    ASSERT_NE(nullptr, accessPatternPtr);

    // Expect each tensor granularity to match the region mapped from the first
    // index element.
    validateGranularity(accessPatternPtr, tpcNode->getInstance());
    validateOverlap(accessPatternPtr, tpcNode->getInstance());
}

TEST_P(TPCAccessPatternGeneratorTest, generated_access_pattern_provide_index_space_resolution)
{
    auto* tpcNode = createAndInstantiateNode();
    ASSERT_NE(nullptr, tpcNode);

    auto accessPatternPtr = AccessPatternFromGlueCodeGenerator::generate(tpcNode);
    ASSERT_NE(nullptr, accessPatternPtr);

    auto nodeResolution = accessPatternPtr->getNodeResolution();

    // Expect the node resolution to allow the work to be deviced like glue code
    // index space.
    const auto& instance = tpcNode->getInstance();
    const auto  numIndexSpaceElements =
        multiplyElements(instance.indexSpaceGeometry, instance.indexSpaceGeometry + instance.indexSpaceRank);
    const auto  numResolutionElements = multiplyElements(nodeResolution.begin(), nodeResolution.end());
    EXPECT_EQ(numIndexSpaceElements, numResolutionElements);
}

TEST_P(TPCAccessPatternGeneratorTest, generated_access_pattern_should_map_node_tile_to_tensor_tile_and_back)
{
    auto* tpcNode = createAndInstantiateNode();
    ASSERT_NE(nullptr, tpcNode);

    auto accessPatternPtr = AccessPatternFromGlueCodeGenerator::generate(tpcNode);
    ASSERT_NE(nullptr, accessPatternPtr);

    auto     nodeResolution = accessPatternPtr->getNodeResolution();
    NodeTile nodeTile(nodeResolution.size(), 1, 0);
    // Try 10 different node tile sizes
    for (unsigned test = 0; test < 10; test++)
    {
        for (Dim nodeDim = 0; nodeDim < nodeResolution.size(); nodeDim++)
        {
            if (nodeTile.geometry[nodeDim] + nodeTile.offset[nodeDim] < nodeResolution[nodeDim])
            {
                nodeTile.geometry[nodeDim]++;
                // Increase offset if possible
                if (nodeTile.geometry[nodeDim] + nodeTile.offset[nodeDim] < nodeResolution[nodeDim])
                {
                    nodeTile.offset[nodeDim]++;
                }
            }
        }
        validateTileMapping(nodeTile, accessPatternPtr, tpcNode->getInstance());
    }
}

void AccessPatternTest::validateGranularityAndOverlap(const TensorPtr& tensor) const
{
    NodeAccessPatternPtr accessPattern = m_node->getNodeAccessPattern();
    ASSERT_NE(nullptr, accessPattern);
    TensorTile       granularity = accessPattern->getTensorGranularity(tensor);
    IntersectionTile overlap     = accessPattern->getTensorOverlap(tensor);

    for (auto tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
    {
        if (isAllRequiredDim(tensorDim, tensor))
        {
            EXPECT_EQ(granularity.geometry[tensorDim], tensor->getSizeInElements(tensorDim));
            EXPECT_EQ(overlap.geometry[tensorDim], tensor->getSizeInElements(tensorDim));
        }
        else
        {
            EXPECT_EQ(granularity.geometry[tensorDim], 1);
            EXPECT_EQ(overlap.geometry[tensorDim], 0);
        }
    }
    validateNodeDimsMapping(tensor);
}

void AccessPatternTest::validateNodeDimsMapping(const TensorPtr& tensor) const
{
    NodeAccessPatternPtr accessPattern = m_node->getNodeAccessPattern();
    ASSERT_NE(nullptr, accessPattern);
    const NodeAccessPattern::Resolution& resolution = accessPattern->getNodeResolution();

    for (auto tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
    {
        if (isElementwiseDim(tensorDim, tensor))
        {
            auto nodeDim = accessPattern->getIndexSpaceDim(tensor, tensorDim);
            EXPECT_EQ(resolution[nodeDim], tensor->getSizeInElements(tensorDim));
        }
    }
}

void AccessPatternTest::validateTensorMapping(const TensorPtr& inTensor,
                                              const TensorPtr& outTensor,
                                              Dim              inputSlicingDim,
                                              Dim              outputSlicingDim) const
{
    NodeAccessPatternPtr accessPattern = m_node->getNodeAccessPattern();
    ASSERT_NE(nullptr, accessPattern);
    TensorTile::Size inputDimSize  = inTensor->getSizeInElements(inputSlicingDim);
    TensorTile::Size outputDimSize = outTensor->getSizeInElements(outputSlicingDim);

    TensorTile inputTile = accessPattern->getTensorGranularity(inTensor);
    ASSERT_EQ(inputTile.geometry[inputSlicingDim], 1)
        << "Expect input to be \"element-wise\" in dim " << inputSlicingDim;

    // Set input tile to cover the 3rd quarter of the slicing dimension
    inputTile.geometry[inputSlicingDim] = inputDimSize / 4;
    inputTile.offset[inputSlicingDim]   = inputDimSize / 2;

    NodeTile   nodeTile   = accessPattern->getNodeTile(inTensor, inputTile);
    TensorTile outputTile = accessPattern->getTensorTile(outTensor, nodeTile);
    EXPECT_EQ(outputTile.geometry[outputSlicingDim], outputDimSize / 4);
    EXPECT_EQ(outputTile.offset[outputSlicingDim], outputDimSize / 2);

    // Set the output tile to cover the 2nd quarter
    outputTile.offset[outputSlicingDim] = outputDimSize / 4;

    nodeTile  = accessPattern->getNodeTile(outTensor, outputTile);
    inputTile = accessPattern->getTensorTile(inTensor, nodeTile);
    EXPECT_EQ(inputTile.geometry[inputSlicingDim], inputDimSize / 4);
    EXPECT_EQ(inputTile.offset[inputSlicingDim], inputDimSize / 4);
}

std::vector<TSize> ReshapeAccessPatternTest::getOperandShape(const DimParams& dimParam)
{
    std::vector<TSize> shape;
    shape.reserve(dimParam.m_outerOnes + dimParam.m_reshapedDims.size() + dimParam.m_innerOnes + 1);
    shape.push_back(1024);
    shape.insert(shape.end(), dimParam.m_innerOnes, 1);
    shape.insert(shape.end(), dimParam.m_reshapedDims.begin(), dimParam.m_reshapedDims.end());
    shape.insert(shape.end(), dimParam.m_outerOnes, 1);

    return shape;
}

void ReshapeAccessPatternTest::createNode()
{
    m_inputs.push_back(createOperand(getOperandShape(GetParam().inputParams)));
    m_output = createOperand(getOperandShape(GetParam().outputParams));

    m_node = NodeFactory::createNode({m_inputs[0]}, {m_output}, nullptr, NodeFactory::reshapeNodeTypeName, "Reshape");
}

TensorPtr ReshapeAccessPatternTest::createOperand(std::vector<TSize> shape)
{
    return std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_float);
}

const ReshapeAccessPatternTestParams::DimParams& ReshapeAccessPatternTest::dimParams(const TensorPtr& tensor) const
{
    return tensor == m_inputs.front() ? GetParam().inputParams : GetParam().outputParams;
}

TensorTile::Size ReshapeAccessPatternTest::getExpectedFlattenedGranularity() const
{
    const auto& inputReshapedDims  = GetParam().inputParams.m_reshapedDims;
    const auto& outputReshapedDims = GetParam().outputParams.m_reshapedDims;
    if (inputReshapedDims.size() > outputReshapedDims.size())
    {
        // input is element-wise
        return outputReshapedDims.back() / inputReshapedDims.back();
    }
    else
    {
        // output is element-wise
        return inputReshapedDims.back() / outputReshapedDims.back();
    }
}

TensorTile::Size ReshapeAccessPatternTest::getExpectedOuterDimGranularity(const TensorPtr& tensor) const
{
    if (dimParams(tensor).m_reshapedDims.size() > 1)
    {
        // This is the element-wise tensor
        return 1;
    }
    else
    {
        // This is a reshaped (flattened) tensor
        return getExpectedFlattenedGranularity();
    }
}

void ReshapeAccessPatternTest::validateGranularityAndOverlap(const TensorPtr& tensor, const DimParams& dimParam) const
{
    NodeAccessPatternPtr accessPattern = m_node->getNodeAccessPattern();
    Dim                  outerDim      = getOuterDim(tensor, dimParam);
    TensorTile           granularity   = accessPattern->getTensorGranularity(tensor);
    IntersectionTile     overlap       = accessPattern->getTensorOverlap(tensor);

    // Except the outermost non-degenerate dimension, all granularity should be the full dimension size ATM.
    for (Dim dim = 0; dim < tensor->getDim(); dim++)
    {
        if (dim == outerDim) continue;
        EXPECT_EQ(granularity.geometry[dim], tensor->getSizeInElements(dim));
    }
    EXPECT_EQ(granularity.geometry[outerDim], getExpectedOuterDimGranularity(tensor));
    EXPECT_EQ(overlap.geometry[outerDim], 0);
}

std::tuple<TensorPtr, TensorPtr> ReshapeAccessPatternTest::getElementWiseAndReshapedTensors() const
{
    if (GetParam().inputParams.m_reshapedDims.size() > 1)
    {
        return {m_inputs.front(), m_output};
    }
    else
    {
        return {m_output, m_inputs.front()};
    }
}

bool ReshapeAccessPatternTest::isAllRequiredDim(unsigned dim, const TensorPtr& t) const
{
    bool isInput = m_node->getInput(0) == t;
    return dim != getOuterDim(t, isInput ? GetParam().inputParams : GetParam().outputParams);
}

Dim ReshapeAccessPatternTest::getOuterDim(const TensorPtr& tensor, const DimParams& dimParams) const
{
    return tensor->getDim() - dimParams.m_outerOnes - 1;
}

using DimParams = ReshapeAccessPatternTestParams::DimParams;
INSTANTIATE_TEST_SUITE_P(
    node_access_pattern,
    ReshapeAccessPatternTest,
    ::testing::Values(ReshapeAccessPatternTestParams {{0, 2, {128}},         {0, 0, {128}}},          // 128x1x1x... -> 128x...
                      ReshapeAccessPatternTestParams {{2, 0, {128}},         {0, 0, {128}}},          // 1x1x128x... -> 128x...
                      ReshapeAccessPatternTestParams {{1, 1, {128}},         {0, 0, {128}}},          // 1x128x1x... -> 128x...
                      ReshapeAccessPatternTestParams {{0, 0, {128}},         {0, 2, {128}}},          // 128x... -> 128x1x1x...
                      ReshapeAccessPatternTestParams {{0, 0, {128}},         {2, 0, {128}}},          // 128x... -> 1x1x128x...
                      ReshapeAccessPatternTestParams {{0, 0, {128}},         {1, 1, {128}}},          // 128x... -> 1x128x1x...
                      ReshapeAccessPatternTestParams {{0, 2, {128}},         {2, 0, {128}}},          // 128x1x1x... -> 1x1x128x...
                      ReshapeAccessPatternTestParams {{2, 0, {128}},         {0, 2, {128}}},          // 1x1x128x... -> 128x1x1x...
                      ReshapeAccessPatternTestParams {{0, 0, {32, 64}},      {0, 0, {32 * 64}}},      // 64x32x... -> 64*32x...
                      ReshapeAccessPatternTestParams {{0, 0, {6, 32, 64}},   {0, 0, {6 * 32 * 64}}},  // 64x32x6x... -> 64*32*6x...
                      ReshapeAccessPatternTestParams {{0, 0, {32 * 64}},     {0, 0, {32, 64}}},       // 64*32x... -> 64x32x...
                      ReshapeAccessPatternTestParams {{0, 0, {3 * 32 * 64}}, {0, 0, {3, 32, 64}}}     // 64*32*3x... -> 64x32x3...
                      ));

TEST_P(ReshapeAccessPatternTest, reshape_access_pattern_should_allow_slicing_outer_dims_with_same_size)
{
    createNode();
    ASSERT_NE(nullptr, m_node->getNodeAccessPattern());

    validateGranularityAndOverlap(m_inputs.front(), GetParam().inputParams);
    validateGranularityAndOverlap(m_output, GetParam().outputParams);

    auto [elementWise, reshaped] = getElementWiseAndReshapedTensors();
    validateTensorMapping(elementWise,
                          reshaped,
                          getOuterDim(elementWise, dimParams(elementWise)),
                          getOuterDim(reshaped, dimParams(reshaped)));
}

TEST_F(ReshapeAccessPatternTest, reshape_access_pattern_should_not_return_null_after_remove_contiguous_reshapes)
{
    // Currently we support 'separable' access pattern - i.e access pattern with some dimension that is not all required
    // - for reshapes where the input and output outer dimensions are an aggregation of each-other. Meaning, the bigger
    // outer dimension is a multiplication of one or more outer dimensions of the other operand. For example:
    // A,B,...->A*B,... or A*B*C,... -> A,B,C,...
    // The following reshape would not qualify for 'separable' access pattern: A,B,C,... -> A*B/4,4,C.. since this is
    // not an aggregation.
    // When unifying reshapes, the result may be 'inseparable' despite the united reshapes being each 'separable'. The
    // shapes below demonstrate this:
    // * input->intermediate - 32*32 -> 32,32 is an aggregation.
    // * intermediate->output - 32 -> 32 is a degenerate aggregation
    // * but input->output - 32*32 -> 32,32*32 is not an aggregation
    std::vector<TSize> inputShape        = {32 * 32, 32 * 32};
    std::vector<TSize> intermediateShape = {32 * 32, 32, 32};
    std::vector<TSize> outputShape       = {32, 32 * 32, 32};

    // This test checks that the reshape unifying pass is able to run on such a chain and that the resulting reshape
    // still provides an access pattern.

    // Graph: relu->reshape1->reshape2->relu
    Gaudi2Graph graph;

    TensorPtr in     = std::make_shared<Tensor>(inputShape.size(), inputShape.data(), syn_type_float);
    TensorPtr reluIn = std::make_shared<Tensor>(inputShape.size(), inputShape.data(), syn_type_float);
    NodePtr   relu1  = NodeFactory::createNode({in}, {reluIn}, nullptr, "relu_fwd_f32", "RELU_IN");
    ASSERT_TRUE(GraphEditor::addNode(graph, relu1));

    TensorPtr intermediate =
        std::make_shared<Tensor>(intermediateShape.size(), intermediateShape.data(), syn_type_float);
    NodePtr reshape1 =
        NodeFactory::createNode({reluIn}, {intermediate}, nullptr, NodeFactory::reshapeNodeTypeName, "RESHAPE_1");
    ASSERT_TRUE(GraphEditor::addNode(graph, reshape1));
    EXPECT_NE(nullptr, reshape1->getNodeAccessPattern());

    TensorPtr out = std::make_shared<Tensor>(outputShape.size(), outputShape.data(), syn_type_float);
    NodePtr   reshape2 =
        NodeFactory::createNode({intermediate}, {out}, nullptr, NodeFactory::reshapeNodeTypeName, "RESHAPE_2");
    ASSERT_TRUE(GraphEditor::addNode(graph, reshape2));
    EXPECT_NE(nullptr, reshape2->getNodeAccessPattern());

    TensorPtr reluOut = std::make_shared<Tensor>(outputShape.size(), outputShape.data(), syn_type_float);
    NodePtr   relu2   = NodeFactory::createNode({out}, {reluOut}, nullptr, "relu_fwd_f32", "RELU_OUT");
    ASSERT_TRUE(GraphEditor::addNode(graph, relu2));

    // Unite the reshapes into 1
    ASSERT_TRUE(removeContiguousReshapeNodes(graph));

    bool reshapeEncountered = false;
    for (const NodePtr& node : graph.getNodes())
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            EXPECT_FALSE(reshapeEncountered);  // test the test - make sure the graph contains a single reshape.
            reshapeEncountered = true;
            EXPECT_NE(nullptr, node->getNodeAccessPattern());
        }
    }
    EXPECT_TRUE(reshapeEncountered);  // test the test - make sure the graph contains a single reshape.
}

// This "iterator" generates all the permutations of a [0, range-1] and is only meant to be used in google
// testing::Range in order to produce tests for all possible permutations.
struct PermutationIterator
{
    bool                      end = false;
    TransposePermutationArray perm;

    PermutationIterator(Dim range) : perm(range)
    {
        for (Dim dim = 0; dim < range; dim++)
            perm[dim] = TransposePermutationDim(dim);
    }

    PermutationIterator operator+(int step)
    {
        end = !std::next_permutation(perm.begin(), perm.end());
        return PermutationIterator(*this);
    }

    bool operator<(const PermutationIterator& other) { return !end; }
};

class TransposeAccessPatternTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<PermutationIterator, const char*>>
{
protected:
    using Permutation = TransposePermutationArray;

    void createNode(const char* guid, const Permutation& permutation)
    {
        size_t rank = permutation.size();

        std::vector<TSize> inputShape(rank);
        std::vector<TSize> outputShape;

        TensorTile::Size dimSize = 100;

        synTransposeParams params;
        params.tensorDim = rank;
        auto* p          = params.permutation;

        // populate output shape with 100x200x300... and input shape with the corresponding sizes in the permutation
        // mapped dimensions
        for (Dim dim : permutation)
        {
            *p++            = TransposePermutationDim(dim);
            inputShape[dim] = dimSize;
            outputShape.push_back(dimSize);

            dimSize += 100;
        }

        LOG_TRACE(GO_TEST, "input shape: [{}]", toString(inputShape, ','));
        LOG_TRACE(GO_TEST, "output shape: [{}]", toString(outputShape, ','));

        m_input  = std::make_shared<Tensor>(inputShape.size(), inputShape.data(), syn_type_float);
        m_output = std::make_shared<Tensor>(outputShape.size(), outputShape.data(), syn_type_float);

        if (!guid)
        {
            m_node = NodeFactory::createInternalNode({m_input}, {m_output}, &permutation, NodeFactory::transposeMmeNodeTypeName, "MmeTranspose");
        }
        else if (std::string(guid) != NodeFactory::transposeNodeTypeName)
        {
            m_node = NodeFactory::createNode({m_input}, {m_output}, &permutation, guid, guid);
        }
        else
        {
            m_node = NodeFactory::createNode({m_input}, {m_output}, &params, sizeof(params), guid, guid);
        }
    }

    void validateGranularity(const NodeAccessPatternPtr& ap) const
    {
        TensorTile expGranularity(m_input->getDim(), 1, 0);

        TensorTile outputGranularity = ap->getTensorGranularity(m_output);
        EXPECT_EQ(outputGranularity.geometry, expGranularity.geometry);
        EXPECT_EQ(outputGranularity.offset, expGranularity.offset);

        TensorTile inputGranularity = ap->getTensorGranularity(m_input);
        EXPECT_EQ(inputGranularity.geometry, expGranularity.geometry);
        EXPECT_EQ(inputGranularity.offset, expGranularity.offset);
    }

    void validateMapping(const NodeAccessPatternPtr& ap, const TransposePermutationArray& permutation) const
    {
        // output shape is 100x200x300x..
        // create a tile with shape 50x100x150x... and offset 10x20x30x...
        TensorTile outputTile = tensorTileFromTensor(m_output);
        for (auto& size : outputTile.geometry)
        {
            size /= 2;
        }
        TensorTile::Coord newOffset = 10;
        for (auto& offset : outputTile.offset)
        {
            offset = newOffset;
            newOffset += 10;
        }

        NodeTile   nodeTile  = ap->getNodeTile(m_output, outputTile);
        TensorTile inputTile = ap->getTensorTile(m_input, nodeTile);

        Dim runningDim = 0;
        for (const auto& permutatedDim : permutation)
        {
            EXPECT_EQ(inputTile.geometry[permutatedDim], outputTile.geometry[runningDim]);
            EXPECT_EQ(inputTile.offset[permutatedDim], outputTile.offset[runningDim]);
            runningDim++;
        }

        NodeTile inputMappedNodeTile = ap->getNodeTile(m_input, inputTile);
        EXPECT_EQ(inputMappedNodeTile.geometry, nodeTile.geometry);
        EXPECT_EQ(inputMappedNodeTile.offset, nodeTile.offset);

        TensorTile mappedOutputTile = ap->getTensorTile(m_output, inputMappedNodeTile);
        EXPECT_EQ(mappedOutputTile.geometry, outputTile.geometry);
        EXPECT_EQ(mappedOutputTile.offset, outputTile.offset);
    }

    NodePtr   m_node;
    TensorPtr m_input;
    TensorPtr m_output;
};

TEST_P(TransposeAccessPatternTest, transpose_access_pattern_should_give_1x1_granularity)
{
    const Permutation& permutation = std::get<0>(GetParam()).perm;
    const char*        guid        = std::get<1>(GetParam());
    LOG_DEBUG(GO_TEST,
              "TransposeAccessPatternTest granularity test: GUID: {}, Permutation: {}",
              guid,
              toString(permutation, ','));

    createNode(guid, permutation);
    ASSERT_NE(nullptr, m_node);

    NodeAccessPatternPtr ap = m_node->getNodeAccessPattern();
    ASSERT_NE(nullptr, ap);

    validateGranularity(ap);
}

TEST_P(TransposeAccessPatternTest, transpose_access_pattern_should_map_output_to_input_according_to_the_permutation)
{
    const Permutation& permutation = std::get<0>(GetParam()).perm;
    const char*        guid        = std::get<1>(GetParam());
    LOG_DEBUG(GO_TEST,
              "TransposeAccessPatternTest granularity test: GUID: {}, Permutation: {}",
              guid,
              toString(permutation, ','));

    createNode(guid, permutation);
    ASSERT_NE(nullptr, m_node);

    NodeAccessPatternPtr ap = m_node->getNodeAccessPattern();
    ASSERT_NE(nullptr, ap);

    validateMapping(ap, permutation);
}

INSTANTIATE_TEST_SUITE_P(node_access_pattern,
                         TransposeAccessPatternTest,
                         ::testing::Combine(::testing::Range(PermutationIterator {4}, PermutationIterator {0}),
                                            ::testing::Values("transpose")));  // Not all guids support all permutations

//
// Correct behavior of the access pattern is checked in the suite above. These checks the availability in all transpose
// guids with minimal functionality since not all guids support all permutations.
//

TEST_F(TransposeAccessPatternTest, logical_transpose_should_provide_node_access_pattern)
{
    Permutation perm = {TransposePermutationDim(0),
                        TransposePermutationDim(2),
                        TransposePermutationDim(1),
                        TransposePermutationDim(3)};
    const char* guid = NodeFactory::transposeLogicNodeTypeName;
    createNode(guid, perm);

    NodeAccessPatternPtr ap = m_node->getNodeAccessPattern();
    EXPECT_NE(nullptr, ap);

    validateGranularity(ap);
    validateMapping(ap, perm);
}

TEST_F(TransposeAccessPatternTest, dma_transpose_should_provide_node_access_pattern)
{
    Permutation perm = {TransposePermutationDim(1),
                        TransposePermutationDim(0),
                        TransposePermutationDim(2),
                        TransposePermutationDim(3)};
    const char* guid = NodeFactory::transposeDmaNodeTypeName;
    createNode(guid, perm);

    NodeAccessPatternPtr ap = m_node->getNodeAccessPattern();
    EXPECT_NE(nullptr, ap);

    validateGranularity(ap);
    validateMapping(ap, perm);
}

TEST_F(TransposeAccessPatternTest, mme_transpose_should_provide_node_access_pattern)
{
    Permutation perm = {TransposePermutationDim(1),
                        TransposePermutationDim(0),
                        TransposePermutationDim(2),
                        TransposePermutationDim(3)};
    const char* guid = nullptr;  // MME transpose has no GUID
    createNode(guid, perm);

    NodeAccessPatternPtr ap = m_node->getNodeAccessPattern();
    EXPECT_NE(nullptr, ap);

    validateGranularity(ap);
    validateMapping(ap, perm);
}

void SqueezeAccessPatternTest::createNode()
{
    std::vector<TSize> inputShape = GetParam().inputShape;
    m_inputs.push_back(std::make_shared<Tensor>(inputShape.size(), inputShape.data(), syn_type_float));
    unsigned           outputSize = inputShape.size() - GetParam().squeezeDims.size() + GetParam().numOutputPaddingDims;
    std::vector<TSize> outputShape(outputSize, 1);
    unsigned outDim = 0;
    for (unsigned inDim = 0; inDim < inputShape.size(); inDim++)
    {
        if (inputShape[inDim] != 1)
        {
            outputShape[outDim] = inputShape[inDim];
            outDim++;
        }
    }
    m_output         = std::make_shared<Tensor>(outputShape.size(), outputShape.data(), syn_type_float);
    auto squeezeDims = GetParam().squeezeDims;

    m_node = NodeFactory::createNode(m_inputs, {m_output}, nullptr, NodeFactory::squeezeNodeTypeName, "squeeze");
}

bool SqueezeAccessPatternTest::isAllRequiredDim(unsigned dim, const TensorPtr& t) const
{
    return (t == m_inputs[0] && isSqueezedDim(dim)) ||
           (t == m_output && (dim >= m_inputs[0]->getDim() - GetParam().squeezeDims.size()));
}

bool SqueezeAccessPatternTest::isElementwiseDim(unsigned dim, const TensorPtr& t) const
{
    return !isAllRequiredDim(dim, t);
}

bool SqueezeAccessPatternTest::isSqueezedDim(unsigned dim) const
{
    return std::find(GetParam().squeezeDims.begin(), GetParam().squeezeDims.end(), dim) !=
                                 GetParam().squeezeDims.end();
}

// The non squeezed dims should have 1:1 mapping to the corresponding output dims.
// The dims before the squeezed dim match the dim index, the dims beyond it are shifted according to the number of
// squeezed dims before it.
Dim SqueezeAccessPatternTest::getMatchingOutDim(unsigned inDim) const
{
    unsigned outDimIdx = inDim;
    for (auto squeezeDim : GetParam().squeezeDims)
    {
        if (inDim < squeezeDim) break;
        outDimIdx--;
    }
    return outDimIdx;
}

TEST_P(SqueezeAccessPatternTest, squeeze_access_pattern_should_allow_any_slicing)
{
    createNode();

    validateGranularityAndOverlap(m_inputs[0]);
    validateGranularityAndOverlap(m_output);

    for (Dim inDim = 0; inDim < m_inputs[0]->getDim(); inDim++)
    {
        if (!isSqueezedDim(inDim))
        {
            Dim outDim = getMatchingOutDim(inDim);
            validateTensorMapping(m_inputs[0], m_output, inDim, outDim);
        }
        else
        {
            // There is no matching input dim for the squeezed dim
            MultiDims matchingDims =
                m_node->getNodeAccessPattern()->getTensorMatchingSlicedDims(m_output, m_inputs[0], inDim);
            ASSERT_TRUE(matchingDims.empty());
        }
    }
}

INSTANTIATE_TEST_SUITE_P(squeeze_access_pattern,
                         SqueezeAccessPatternTest,
                         ::testing::Values(SqueezeAccessPatternParams {{64, 1, 128, 24}, 0, {1}},
                                           SqueezeAccessPatternParams {{12, 1, 20}, 1, {1}},
                                           SqueezeAccessPatternParams {{12, 1, 28, 1, 100}, 0, {1, 3}},
                                           SqueezeAccessPatternParams {{123, 218, 1, 16}, 0, {2}},
                                           SqueezeAccessPatternParams {{50, 300, 70, 1, 1}, 0, {3, 4}},
                                           SqueezeAccessPatternParams {{1, 1, 11, 128}, 1, {0, 1}},
                                           SqueezeAccessPatternParams {{12, 1, 1, 20}, 1, {1, 2}},
                                           SqueezeAccessPatternParams {{50, 1, 16, 8, 1}, 2, {1, 4}},
                                           SqueezeAccessPatternParams {{50, 1, 1, 16}, 2, {1, 2}},
                                           SqueezeAccessPatternParams {{1, 128, 1, 1}, 0, {0, 2, 3}},
                                           SqueezeAccessPatternParams {{128, 1}, 0, {1}}));

void ExpandDimsAccessPatternTest::createNode()
{
    unsigned           expandDim(GetParam().expandDim);
    std::vector<TSize> inputShape = GetParam().inputShape;
    m_inputs.push_back(std::make_shared<Tensor>(inputShape.size(), inputShape.data(), syn_type_float));
    std::vector<TSize> outputShape = GetParam().inputShape;
    outputShape.insert(outputShape.begin() + expandDim, 1);
    m_output = std::make_shared<Tensor>(outputShape.size(), outputShape.data(), syn_type_float);

    m_node =
        NodeFactory::createNode(m_inputs, {m_output}, &expandDim, NodeFactory::expandDimsNodeTypeName, "ExpandDims");
}

bool ExpandDimsAccessPatternTest::isAllRequiredDim(unsigned dim, const TensorPtr& t) const
{
    if (t == m_output && dim == GetParam().expandDim) return true;
    return false;
}

bool ExpandDimsAccessPatternTest::isElementwiseDim(unsigned dim, const TensorPtr& t) const
{
    return !isAllRequiredDim(dim, t);
}

TEST_P(ExpandDimsAccessPatternTest, expand_dims_access_pattern_should_allow_any_slicing)
{
    createNode();

    validateGranularityAndOverlap(m_inputs[0]);
    validateGranularityAndOverlap(m_output);

    Dim expandDim(GetParam().expandDim);
    for (Dim outDim = 0; outDim < m_output->getDim(); outDim++)
    {
        if (outDim != expandDim)
        {
            // The non expanded dims should have 1:1 mapping to the corresponding input dims
            // The dims before the expanded dim match the dim index, the dims beyond it are shifted by 1
            Dim inDim = outDim < expandDim ? outDim : outDim - 1;
            validateTensorMapping(m_inputs[0], m_output, inDim, outDim);
        }
        else
        {
            // There is no matching input dim for the expand dim
            MultiDims matchingDims =
                m_node->getNodeAccessPattern()->getTensorMatchingSlicedDims(m_inputs[0], m_output, expandDim);
            ASSERT_TRUE(matchingDims.empty());
        }
    }
}

INSTANTIATE_TEST_SUITE_P(expand_dims_access_pattern,
                         ExpandDimsAccessPatternTest,
                         ::testing::Values(ExpandDimsAccessPatternParams {0, {64, 128, 24}},
                                           ExpandDimsAccessPatternParams {1, {12, 28, 100}},
                                           ExpandDimsAccessPatternParams {2, {123, 218, 16}},
                                           ExpandDimsAccessPatternParams {3, {50, 300, 70}},
                                           ExpandDimsAccessPatternParams {0, {11, 128}},
                                           ExpandDimsAccessPatternParams {1, {12, 20}},
                                           ExpandDimsAccessPatternParams {2, {50, 16}},
                                           ExpandDimsAccessPatternParams {0, {128}},
                                           ExpandDimsAccessPatternParams {1, {128}}));

TensorPtr GemmAccessPatternTest::createOperand(std::vector<TSize> shape, bool transposed, bool isMask)
{
    if (transposed)
    {
        std::swap(shape[0], shape[1]);
    }
    if (!GetParam().batchDims.empty())
    {
        if (isMask)
        {
            shape.push_back(1);                        // dim 2 of the mask is 1
            shape.push_back(GetParam().batchDims[1]);  // dim 3 of the mask is the external batch dim
        }
        else
        {
            shape.insert(shape.end(), GetParam().batchDims.begin(), GetParam().batchDims.end());
        }
    }

    return std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_float);
}

void GemmAccessPatternTest::createNode()
{
    std::string guid =
        (GetParam().batchDims.empty()) ? NodeFactory::gemmNodeTypeName : NodeFactory::batchGemmNodeTypeName;
    std::string name = (GetParam().batchDims.empty()) ? "Gemm" : "BGemm";

    synGEMMParams params(GetParam().transposeA, GetParam().transposeB);
    m_inputs.push_back(createOperand({GetParam().commonDim, GetParam().heightA}, params.transpose_a, false));
    m_inputs.push_back(createOperand({GetParam().widthB, GetParam().commonDim}, params.transpose_b, false));
    if (GetParam().masksCommonDim)  // marks this is masked bgemm node
    {
        ASSERT_EQ(GetParam().batchDims.size(), 2);  // validate the test params are correct

        guid = NodeFactory::maskedBatchGemmNodeTypeName;
        name = "MaskedBGemm";

        unsigned masksCommonDim = *(GetParam().masksCommonDim);
        m_inputs.push_back(createOperand({masksCommonDim, GetParam().heightA}, params.transpose_a, true));
        m_inputs.push_back(createOperand({GetParam().widthB, masksCommonDim}, params.transpose_b, true));
    }
    m_output = createOperand({GetParam().widthB, GetParam().heightA}, false, false);

    m_node = NodeFactory::createNode(m_inputs, {m_output}, &params, guid.c_str(), name.c_str());
}

bool GemmAccessPatternTest::isElementwiseDim(unsigned dim, const TensorPtr& t) const
{
    return true;
}

Dim GemmAccessPatternTest::getSpatialNonCommonDim(unsigned inputIdx) const
{
    if (inputIdx == 0)
    {
        return GetParam().transposeA ? DIM_C : DIM_W;
    }
    else
    {
        return GetParam().transposeB ? WEIGHT_DIM_C : WEIGHT_DIM_K;
    }
}

Dim GemmAccessPatternTest::getSpatialCommonDim(unsigned inputIdx) const
{
    return (1 - getSpatialNonCommonDim(inputIdx));
}

Dim GemmAccessPatternTest::getOutDimForInput(unsigned inputIdx, Dim inputDim) const
{
    if (inputDim < DIM_GEMM_BATCH)
    {
        return (inputIdx == 0) ? Dim(DIM_W) : Dim(WEIGHT_DIM_K);
    }
    // else - batch dim remains the same for output
    return inputDim;
}

TEST_P(GemmAccessPatternTest, mme_access_pattern_should_allow_any_slicing)
{
    createNode();

    validateGranularityAndOverlap(m_inputs[0]);
    validateGranularityAndOverlap(m_inputs[1]);
    validateGranularityAndOverlap(m_output);

    unsigned in0SlicingDim = getSpatialNonCommonDim(0);
    unsigned in1SlicingDim = getSpatialNonCommonDim(1);
    validateTensorMapping(m_inputs[0], m_output, in0SlicingDim, getOutDimForInput(0, in0SlicingDim));
    validateTensorMapping(m_inputs[1], m_output, in1SlicingDim, getOutDimForInput(1, in1SlicingDim));

    unsigned in0CommonDim = getSpatialCommonDim(0);
    unsigned in1CommonDim = getSpatialCommonDim(1);
    validateTensorMapping(m_inputs[0], m_inputs[1], in0CommonDim, in1CommonDim);
    for (Dim dim = DIM_GEMM_BATCH; dim < m_output->getDim(); dim++)
    {
        validateTensorMapping(m_inputs[0], m_output, dim, getOutDimForInput(0, dim));
        validateTensorMapping(m_inputs[1], m_output, dim, getOutDimForInput(1, dim));
    }
    if (GetParam().masksCommonDim)
    {
        validateTensorMapping(m_inputs[0], m_inputs[2], in0SlicingDim, in0SlicingDim);
        validateTensorMapping(m_inputs[1], m_inputs[3], in1SlicingDim, in1SlicingDim);
        validateTensorMapping(m_inputs[2], m_inputs[3], in0CommonDim, in1CommonDim);
        unsigned externalBatchDim = DIM_GEMM_BATCH + 1;
        validateTensorMapping(m_inputs[2], m_output, externalBatchDim, externalBatchDim);
        validateTensorMapping(m_inputs[3], m_output, externalBatchDim, externalBatchDim);
    }
}
TEST_P(BiasedGemmAccessPatternTest, mme_access_pattern_should_allow_any_slicing_with_bias_input)
{
    createNode();
    std::vector<TSize> biasSizes = {GetParam().widthB};
    auto               bias      = std::make_shared<Tensor>(biasSizes.size(), biasSizes.data(), syn_type_float);
    m_node->addInput(2, bias);
    m_inputs.push_back(bias);
    validateGranularityAndOverlap(m_inputs[0]);
    validateGranularityAndOverlap(m_inputs[1]);
    validateGranularityAndOverlap(m_inputs[2]);
    validateGranularityAndOverlap(m_output);

    unsigned in0SlicingDim = getSpatialNonCommonDim(0);
    unsigned in1SlicingDim = getSpatialNonCommonDim(1);
    validateTensorMapping(m_inputs[0], m_output, in0SlicingDim, getOutDimForInput(0, in0SlicingDim));
    validateTensorMapping(m_inputs[1], m_output, in1SlicingDim, getOutDimForInput(1, in1SlicingDim));
    validateTensorMapping(m_inputs[2], m_output, 0, getOutDimForInput(2, 0));

    unsigned in0CommonDim = getSpatialCommonDim(0);
    unsigned in1CommonDim = getSpatialCommonDim(1);
    validateTensorMapping(m_inputs[0], m_inputs[1], in0CommonDim, in1CommonDim);
    for (Dim dim = DIM_GEMM_BATCH; dim < m_output->getDim(); dim++)
    {
        validateTensorMapping(m_inputs[0], m_output, dim, getOutDimForInput(0, dim));
        validateTensorMapping(m_inputs[1], m_output, dim, getOutDimForInput(1, dim));
    }
}

INSTANTIATE_TEST_SUITE_P(gemm_access_pattern,
                         GemmAccessPatternTest,
                         ::testing::Values(GemmAccessPatternParams {64, 128, 24, {}, false, false, {}},
                                           GemmAccessPatternParams {12, 28, 100, {}, false, true, {}},
                                           GemmAccessPatternParams {123, 218, 16, {}, true, true, {}},
                                           GemmAccessPatternParams {50, 300, 70, {}, true, false, {}}));

INSTANTIATE_TEST_SUITE_P(bgemm_access_pattern,
                         GemmAccessPatternTest,
                         ::testing::Values(GemmAccessPatternParams {64, 128, 24, {12, 4}, false, false, {}},
                                           GemmAccessPatternParams {12, 28, 100, {3}, false, true, {}},
                                           GemmAccessPatternParams {123, 218, 16, {6, 12, 24}, true, true, {}},
                                           GemmAccessPatternParams {50, 300, 70, {5, 7}, true, false, {}}));
INSTANTIATE_TEST_SUITE_P(biased_gemm_access_pattern,
                         BiasedGemmAccessPatternTest,
                         ::testing::Values(GemmAccessPatternParams {64, 128, 24, {}, false, false, {}},
                                           GemmAccessPatternParams {12, 28, 100, {}, false, true, {}},
                                           GemmAccessPatternParams {123, 218, 16, {}, true, true, {}},
                                           GemmAccessPatternParams {50, 300, 70, {}, true, false, {}}));

INSTANTIATE_TEST_SUITE_P(biased_bgemm_access_pattern,
                         BiasedGemmAccessPatternTest,
                         ::testing::Values(GemmAccessPatternParams {64, 128, 24, {12, 4}, false, false, {}},
                                           GemmAccessPatternParams {12, 28, 100, {3}, false, true, {}},
                                           GemmAccessPatternParams {123, 218, 16, {6, 12, 24}, true, true, {}},
                                           GemmAccessPatternParams {50, 300, 70, {5, 7}, true, false, {}}));

INSTANTIATE_TEST_SUITE_P(masked_bgemm_access_pattern,
                         GemmAccessPatternTest,
                         ::testing::Values(GemmAccessPatternParams {64, 128, 24, {12, 4}, false, false, {13}},
                                           GemmAccessPatternParams {12, 28, 100, {3, 16}, false, true, {16}},
                                           GemmAccessPatternParams {123, 218, 16, {6, 24}, true, true, {8}},
                                           GemmAccessPatternParams {50, 300, 70, {5, 7}, true, false, {25}}));

void ConvAccessPatternTest::createNode()
{
    synConvolutionParams params;
    params.kH                = 3;
    params.kW                = 3;
    unsigned  ifmSpatialSize = 32;
    SizeArray xSize          = {GetParam().ifmC, ifmSpatialSize, ifmSpatialSize, GetParam().batchDim};
    SizeArray wSize          = {GetParam().ofmK, GetParam().ifmC, params.kW, params.kH};
    SizeArray ySize          = {GetParam().ofmK,
                       convOutputDimSize(xSize[1], params.kW, params.dW, params.padL + params.padR, params.dilW),
                       convOutputDimSize(xSize[2], params.kH, params.dH, params.padT + params.padB, params.dilH),
                       GetParam().batchDim};
    if (GetParam().guid == NodeFactory::transposedDeDxNodeTypeName)
    {
        // swap for transposed dedx
        std::swap(wSize[WEIGHT_DIM_K], wSize[WEIGHT_DIM_C]);
    }
    auto      xOperand       = std::make_shared<Tensor>(4, xSize.data(), syn_type_float);
    auto      wOperand       = std::make_shared<Tensor>(4, wSize.data(), syn_type_float);
    auto      yOperand       = std::make_shared<Tensor>(4, ySize.data(), syn_type_float);
    if (GetParam().guid == NodeFactory::convolutionNodeTypeName)
    {
        m_inputs.push_back(xOperand);
        m_inputs.push_back(wOperand);
        m_output = yOperand;
    }
    else if (GetParam().guid == NodeFactory::deDwNodeTypeName)
    {
        m_inputs.push_back(yOperand);
        m_inputs.push_back(xOperand);
        m_output = wOperand;
    }
    else if (GetParam().guid == NodeFactory::deDxNodeTypeName || NodeFactory::transposedDeDxNodeTypeName)
    {
        m_inputs.push_back(yOperand);
        m_inputs.push_back(wOperand);
        m_output = xOperand;
    }
    else
    {
        HB_ASSERT(false, "invalid node type for test");
    }
    m_node = std::dynamic_pointer_cast<ConvBaseNode>(
        NodeFactory::createNode(m_inputs, {m_output}, &params, GetParam().guid.c_str(), "conv"));
}

bool ConvAccessPatternTest::isAllRequiredDim(unsigned dim, const TensorPtr& t) const
{
    std::shared_ptr<ConvBaseNode> convNode = std::static_pointer_cast<ConvBaseNode>(m_node);
    if (convNode->getWOperand() == t)
        return (dim == WEIGHT_DIM_S || dim == WEIGHT_DIM_R || (convNode->is3DConvolution() && dim == WEIGHT_DIM_Q));
    return (dim == DIM_W || dim == DIM_H || (convNode->is3DConvolution() && dim == DIM_D_FOR_5D_TENSOR));
}

bool ConvAccessPatternTest::isElementwiseDim(unsigned dim, const TensorPtr& t) const
{
    return !isAllRequiredDim(dim, t);
}

Dim ConvAccessPatternTest::getNonCommonSlicableDim(unsigned inputIdx) const
{
    if (m_node->getNodeType() == Node::TYPE_CONVOLUTION)
    {
        if (inputIdx == 0) return DIM_B;
        return WEIGHT_DIM_K;
    }
    if (m_node->getNodeType() == Node::TYPE_DEDX)
    {
        if (inputIdx == 0) return DIM_B;
        return WEIGHT_DIM_C;
    }
    if (m_node->getNodeType() == Node::TYPE_TRANSPOSED_DEDX)
    {
        if (inputIdx == 0) return DIM_B;
        // input channels and output channels are swapped for this node
        return WEIGHT_DIM_K;
    }
    return DIM_C;
}

Dim ConvAccessPatternTest::getOutDimForInput(unsigned inputIdx, Dim inputDim) const
{
    if (Node::isDedxNode(m_node))
    {
        return (inputIdx == 1) ? Dim(DIM_C) : inputDim;
    }
    if (m_node->getNodeType() == Node::TYPE_DEDW)
    {
        return (inputIdx == 1) ? Dim(WEIGHT_DIM_C) : Dim(WEIGHT_DIM_K);
    }
    return inputDim;
}

Dim ConvAccessPatternTest::batchIdxSpcDim() const
{
    auto        ap     = m_node->getNodeAccessPattern();
    const auto& inputA = m_node->getInput(0);
    return ap->getIndexSpaceDim(inputA, inputA->getDim() - 1);
}

INSTANTIATE_TEST_SUITE_P(
    conv_fwd_access_pattern,
    ConvAccessPatternTest,
    ::testing::Values(ConvAccessPatternParams {64, 128, 24, NodeFactory::convolutionNodeTypeName},
                      ConvAccessPatternParams {12, 28, 100, NodeFactory::convolutionNodeTypeName},
                      ConvAccessPatternParams {123, 218, 16, NodeFactory::convolutionNodeTypeName},
                      ConvAccessPatternParams {50, 300, 70, NodeFactory::convolutionNodeTypeName}));

INSTANTIATE_TEST_SUITE_P(conv_dedw_access_pattern,
                         ConvAccessPatternTest,
                         ::testing::Values(ConvAccessPatternParams {64, 128, 24, NodeFactory::deDwNodeTypeName},
                                           ConvAccessPatternParams {12, 28, 100, NodeFactory::deDwNodeTypeName},
                                           ConvAccessPatternParams {123, 218, 16, NodeFactory::deDwNodeTypeName},
                                           ConvAccessPatternParams {50, 300, 70, NodeFactory::deDwNodeTypeName}));

INSTANTIATE_TEST_SUITE_P(conv_dedx_access_pattern,
                         ConvAccessPatternTest,
                         ::testing::Values(ConvAccessPatternParams {64, 128, 24, NodeFactory::deDxNodeTypeName},
                                           ConvAccessPatternParams {12, 28, 100, NodeFactory::deDxNodeTypeName},
                                           ConvAccessPatternParams {123, 218, 16, NodeFactory::deDxNodeTypeName},
                                           ConvAccessPatternParams {50, 300, 70, NodeFactory::deDxNodeTypeName}));

INSTANTIATE_TEST_SUITE_P(
    conv_transposed_dedx_access_pattern,
    ConvAccessPatternTest,
    ::testing::Values(ConvAccessPatternParams {64, 128, 24, NodeFactory::transposedDeDxNodeTypeName},
                      ConvAccessPatternParams {12, 28, 100, NodeFactory::transposedDeDxNodeTypeName},
                      ConvAccessPatternParams {123, 218, 16, NodeFactory::transposedDeDxNodeTypeName},
                      ConvAccessPatternParams {50, 300, 70, NodeFactory::transposedDeDxNodeTypeName}));

TEST_P(ConvAccessPatternTest, mme_access_pattern_should_block_slicing_for_all_required_dims)
{
    createNode();

    validateGranularityAndOverlap(m_inputs[0]);
    validateGranularityAndOverlap(m_inputs[1]);
    validateGranularityAndOverlap(m_output);

    unsigned in0SlicingDim = getNonCommonSlicableDim(0);
    unsigned in1SlicingDim = getNonCommonSlicableDim(1);

    validateTensorMapping(m_inputs[0], m_output, in0SlicingDim, getOutDimForInput(0, in0SlicingDim));
    validateTensorMapping(m_inputs[1], m_output, in1SlicingDim, getOutDimForInput(1, in1SlicingDim));
}

TEST_P(ConvAccessPatternTest, mme_access_pattern_should_support_concurrency)
{
    constexpr unsigned numDcores   = 4;
    constexpr TSize    granularity = 1;
    constexpr TSize    onesShape[] = {numDcores};
    constexpr bool     padWithNull = true;

    createNode();
    auto ap = m_node->getNodeAccessPattern();

    m_node->getNodeAnnotation().sliceROI = NodeTile(ap->getNodeResolution());
    gc::layered_brain::NodeDcoreROIsSetter perforator(m_node, numDcores);
    perforator.splitToDcoreROIs(batchIdxSpcDim(), granularity, 0);

    if (GetParam().guid == NodeFactory::deDwNodeTypeName && !m_node->getNodeAnnotation().m_dcoreROIs.empty())
    {
        // Add aux tensors
        auto scratchPad                  = m_output->clone(false, false, false);
        auto spShape                     = scratchPad->getAllNSizesInElements();
        spShape.at(scratchPad->getDim()) = numDcores;
        scratchPad->reshape(scratchPad->getDim() + 1, spShape.data());
        scratchPad->setAsAuxTensor(true);

        auto ones = std::make_shared<Tensor>(1, onesShape, syn_type_float);
        ones->setAsAuxTensor(false);

        m_node->addInput(TENSOR_AUX_CD_SCRATCHPAD, scratchPad, Node::TENSOR_TYPE_DATA, padWithNull);
        m_node->addInput(TENSOR_AUX_CD_REDUCTION, ones, Node::TENSOR_TYPE_DATA, padWithNull);

        // This triggers re-creation of the node's access pattern which is needed to add the aux tensors.
        // In normal flow, the re-creation will be triggered by the removal and addition of the node to the graph after
        // adding the aux tensors to it.
        m_node->updateCache();

        // Validate changes to the index space
        EXPECT_EQ(numDcores, ap->getNodeResolution().at(batchIdxSpcDim()));

        // Validate changes in tensors mapped to the batch index space.
        auto inputTile = ap->getTensorGranularity(m_inputs[0]);
        EXPECT_EQ(div_round_up(GetParam().batchDim, numDcores), inputTile.geometry.back());
        auto overlap = ap->getTensorOverlap(m_inputs[0]);
        EXPECT_EQ(0, overlap.geometry.back());  // No overlap ==> offset == geometry

        auto outputGranularity = ap->getTensorGranularity(m_output);
        auto outputOverlap     = ap->getTensorOverlap(m_output);
        auto spGranularity     = ap->getTensorGranularity(scratchPad);
        auto spOverlap         = ap->getTensorOverlap(scratchPad);
        // Scratchpad expected to have the same access pattern as the output, except for the last dim which isn't found
        // in the output and should be element-wise
        EXPECT_EQ(outputGranularity.geometry.size() + 1, spGranularity.geometry.size());
        for (Dim dim = 0; dim < outputGranularity.geometry.size(); ++dim)
        {
            EXPECT_EQ(ap->getIndexSpaceDim(m_output, dim), ap->getIndexSpaceDim(scratchPad, dim));
            EXPECT_EQ(outputGranularity.geometry.at(dim), spGranularity.geometry.at(dim));
            EXPECT_EQ(outputOverlap.geometry.at(dim), spOverlap.geometry.at(dim));
        }
        EXPECT_EQ(1, spGranularity.geometry.back());
        EXPECT_EQ(0, spOverlap.geometry.back());

        // Ones is expected to have an element-wise dimension access mapped to the batch index space dimension
        auto onesGranularity = ap->getTensorGranularity(ones);
        auto onesOverlap     = ap->getTensorOverlap(ones);
        EXPECT_EQ(1, onesGranularity.geometry.at(0));
        EXPECT_EQ(0, onesOverlap.geometry.at(0));
        EXPECT_EQ(batchIdxSpcDim(), ap->getIndexSpaceDim(ones, 0));
    }
    else
    {
        validateGranularityAndOverlap(m_inputs[0]);
        validateGranularityAndOverlap(m_inputs[1]);
        validateGranularityAndOverlap(m_output);

        unsigned in0SlicingDim = getNonCommonSlicableDim(0);
        unsigned in1SlicingDim = getNonCommonSlicableDim(1);

        validateTensorMapping(m_inputs[0], m_output, in0SlicingDim, getOutDimForInput(0, in0SlicingDim));
        validateTensorMapping(m_inputs[1], m_output, in1SlicingDim, getOutDimForInput(1, in1SlicingDim));
    }
}

// This test aims to check handling of calculation of negative offset for node tiles.
// In this test, the access pattern skips the first element of the tensor, so
// when mapping the full tensor to the node it may underflow out of bounds when
// given offset 0.
// In this case, the calculated nodeStart in the updateNodeTile method will be negative,
// to be assigned later to the NodeTile offset array which contains unsigned values, resulting in
// underflowed values.
// We would like to detect this cases and to be indifferent to this node tile dimension's geometry and offset.
TEST_F(GraphOptimizerTest, negative_node_tile_offset_glue_code_access_pattern)
{
    const Dim                resolutionDim = 2;
    const TensorTile::Size   dimSize       = 10;
    const NodeTile::Geometry nodeGeometry  = {1, 2, 8, 4, 1};
    const NodeTile::Offset   nodeOffset    = {0, 0, 0, 0, 0};

    tpc_lib_api::DimIndexSpaceMapping dimAccessPattern;
    dimAccessPattern.indexSpaceDim = resolutionDim;
    dimAccessPattern.a             = 1;
    dimAccessPattern.start_b       = 1;
    dimAccessPattern.end_b         = 1;

    const GlueCodeAccessPatternDimMapping                  accessPattern(dimSize, &dimAccessPattern);
    const GlueCodeTensorAccessPattern::DimMapping::DimTile tensorDimTile(dimSize, 0);

    NodeTile   nodeTile(nodeGeometry, nodeOffset);
    const auto nodeStart = nodeTile.offset[resolutionDim];
    accessPattern.updateNodeTile(nodeTile, tensorDimTile);

    ASSERT_EQ(nodeStart, nodeTile.offset[resolutionDim]);
}

TEST_F(GraphOptimizerTest, glue_code_access_pattern_large_tensor_dim_mapping)
{
    const TensorTile::Size tensorDimSize     = 117440512;
    const Dim              resolutionDim     = 0;
    const NodeTile::Size   resolutionDimSize = 1792;

    tpc_lib_api::DimIndexSpaceMapping dimAccessPattern;
    dimAccessPattern.indexSpaceDim = resolutionDim;
    dimAccessPattern.a             = 65536;
    dimAccessPattern.start_b       = 0;
    dimAccessPattern.end_b         = 65535;

    const GlueCodeAccessPatternDimMapping accessPattern(tensorDimSize, &dimAccessPattern);

    const NodeTile::Size     nodeTileSize = 256;
    const NodeTile::Geometry nodeGeometry = {nodeTileSize, 1, 1, 1, 1};
    NodeTile                 nodeTile(nodeGeometry);

    const TensorTile::Size expectedTensorDimSize   = 16777216;  // = 65536 * 256
    TensorTile::Coord      expectedTensorDimOffset = 0;
    for (auto nodeTileOffset = 0; nodeTileOffset < resolutionDimSize; nodeTileOffset += nodeTileSize)
    {
        const auto& tensorDimTile = accessPattern.mapNodeRange(nodeTile);
        ASSERT_EQ(tensorDimTile.geometry.at(resolutionDim), expectedTensorDimSize);
        ASSERT_EQ(tensorDimTile.offset.at(resolutionDim), expectedTensorDimOffset);
        nodeTile.offset.at(resolutionDim) += nodeTileSize;
        expectedTensorDimOffset += expectedTensorDimSize;
    }
}