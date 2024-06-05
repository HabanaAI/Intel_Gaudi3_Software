#include <memory>
#include <syn_singleton.hpp>
#include <type_traits>
#include "infra/cpu_calculator.h"
#include "synapse_common_types.h"
#include "utils.h"
#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "habana_graph.h"
#include "dedw_node.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "data_type_utils.h"
#include "types.h"

template<typename DType>
class SynTrainingGConvFwdBwdTest
: public SynTrainingTestInfra
, public testing::WithParamInterface<
      std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, bool>>
{
public:
    SynTrainingGConvFwdBwdTest() { setTestPackage(TEST_PACKAGE_CONVOLUTION); }

    void runGroupedConvolution(Node::eNodeType nodeType, bool check_filter2d_replaced = false);

    unsigned m_in1TensorIndex;
    unsigned m_in2TensorIndex;
    unsigned m_inShapeTensorIndex;
    unsigned m_outTensorIndex;
    unsigned m_nGroups;
    unsigned m_ofmDataSize;
    unsigned m_ifmDataSize;
    unsigned m_wghDataSize;
    bool m_isDynamic;
    std::vector<std::unique_ptr<DType[]>> m_refIfmBuffers;
    std::vector<std::unique_ptr<DType[]>> m_refWghBuffers;
    std::vector<std::unique_ptr<DType[]>> m_refOfmBuffers;
    std::unique_ptr<DType[]>              m_refConcatBuffer;

    //Add a grouped node of type <guid>, the inputs are randomized.
    //m_outTensorIndex is updated to hold the index of the output tensor.
    virtual void addGroupedNode(unsigned*                 in1DimSizes,
                        unsigned*                 in2DimSizes,
                        unsigned*                 outDimSizes,
                        std::unique_ptr<DType[]>& in1Buffer,
                        std::unique_ptr<DType[]>& in2Buffer,
                        synConvolutionParams      params,
                        const char*               guid,
                        unsigned*                 in1DimMinSizes,
                        unsigned*                 in2DimMinSizes,
                        unsigned*                 outDimMinSizes);

    // Initializes refBuffers, and sets to 0 all the m_refOfmBuffers
    void initializeRefs(std::vector<std::unique_ptr<DType[]>>& m_refOfmBuffers, unsigned refOfmDataSize);

    // Concats refOutBuffers with respect to the grouped node concat
    void concatRefOutputBuffer(const unsigned                         dataSize,
                               const unsigned                         jump,
                               std::vector<std::unique_ptr<DType[]>>& refOutBuffers);

    // Fill refBuffers with data from dataBuffers with respect to the grouped node split
    void fillRefs(std::vector<std::unique_ptr<DType[]>>& refBuffers,
                  std::unique_ptr<DType[]>&              dataBuffers,
                  unsigned                               dataSize,
                  const unsigned                         jumps);

    void densifyOutputBuffers(std::vector<std::unique_ptr<DType[]>>& refOutBuffers,
                              std::vector<std::unique_ptr<DType[]>>& refDenseOutBuffers,
                              const unsigned                         actualOutputSize);

    // Calculates the relevant operation (conv/dedx/dedw) and fills the refOutBuffers
    void calculateRefs(unsigned*                              refIn1DimSizes,
                       unsigned*                              refIn2DimSizes,
                       unsigned*                              refOutDimSizes,
                       std::vector<std::unique_ptr<DType[]>>& refIn1Buffers,
                       std::vector<std::unique_ptr<DType[]>>& refIn2Buffers,
                       std::vector<std::unique_ptr<DType[]>>& refOutBuffers,
                       Node::eNodeType                        nodeType);

    void validateFinalResult(unsigned dataSize);
};

template<typename DType>
void SynTrainingGConvFwdBwdTest<DType>::fillRefs(std::vector<std::unique_ptr<DType[]>>& refBuffers,
                                                 std::unique_ptr<DType[]>&              dataBuffers,
                                                 unsigned                               dataSize,
                                                 const unsigned                         jumps)
{
    std::unique_ptr<unsigned[]> refBuffItrs(new unsigned[m_nGroups]);
    for (unsigned i = 0; i < m_nGroups; i++)
    {
        refBuffItrs[i] = 0;
    }
    for (unsigned i = 0; i < dataSize; i++)
    {
        unsigned refBuffIdx = (i % (jumps)) / (jumps / m_nGroups);
        refBuffers[refBuffIdx][refBuffItrs[refBuffIdx]] = dataBuffers[i];
        refBuffItrs[refBuffIdx]++;
    }
}

template<typename DType>
void SynTrainingGConvFwdBwdTest<DType>::addGroupedNode(unsigned*                 in1DimSizes,
                                                       unsigned*                 in2DimSizes,
                                                       unsigned*                 outDimSizes,
                                                       std::unique_ptr<DType[]>& in1Buffer,
                                                       std::unique_ptr<DType[]>& in2Buffer,
                                                       synConvolutionParams      params,
                                                       const char*               guid,
                                                       unsigned*                 in1DimMinSizes,
                                                       unsigned*                 in2DimMinSizes,
                                                       unsigned*                 outDimMinSizes)
{
    synDataType dataDtype = dataTypeToSynType<DType>();

    unsigned in1DataSize = in1DimSizes[0] * in1DimSizes[1] * in1DimSizes[2] * in1DimSizes[3];
    unsigned in2DataSize = in2DimSizes[0] * in2DimSizes[1] * in2DimSizes[2] * in2DimSizes[3];

    m_in1TensorIndex = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           in1DimSizes,
                                           4,
                                           dataDtype,
                                           nullptr,
                                           "in1",
                                           0,
                                           0,
                                           nullptr,
                                           in1DimMinSizes);
    in1Buffer.reset(new DType[in1DataSize]);
    memcpy(in1Buffer.get(), m_hostBuffers[m_in1TensorIndex], in1DataSize * dataTypeSizeInBytes(dataDtype));

    m_in2TensorIndex = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           in2DimSizes,
                                           4,
                                           dataDtype,
                                           nullptr,
                                           "in2",
                                           0,
                                           0,
                                           nullptr,
                                           in2DimMinSizes);
    in2Buffer.reset(new DType[in2DataSize]);
    memcpy(in2Buffer.get(), m_hostBuffers[m_in2TensorIndex], in2DataSize * dataTypeSizeInBytes(dataDtype));

    m_outTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outDimSizes, 4, dataDtype, nullptr, "out", 0, 0, nullptr, outDimMinSizes);

    TensorIndices inputs = {m_in1TensorIndex, m_in2TensorIndex};
    if (std::strcmp(guid, NodeFactory::deDxNodeTypeName) == 0)
    {
        m_inShapeTensorIndex = createShapeTensor(INPUT_TENSOR, outDimSizes, outDimMinSizes, 4, dataDtype, "dx_shape");
        inputs.push_back(m_inShapeTensorIndex);
    }

    addNodeToGraph(guid, inputs, {m_outTensorIndex}, (void *) &params, sizeof(synConvolutionParams), guid);
}

template<typename DType>
void SynTrainingGConvFwdBwdTest<DType>::initializeRefs(std::vector<std::unique_ptr<DType[]>>& refOutBuffers,
                                                       unsigned                               refOutDataSize)
{
    // Calculate reference
    unsigned refIfmDataSize = m_ifmDataSize / m_nGroups;
    unsigned refWghDataSize = m_wghDataSize / m_nGroups;
    unsigned refOfmDataSize = m_ofmDataSize / m_nGroups;

    // Create reference tensors with sizes divided to groups
    for (int i = 0; i < m_nGroups; i++)
    {
        std::unique_ptr<DType[]> refIfmBuffer(new DType[refIfmDataSize]);
        m_refIfmBuffers.push_back(std::move(refIfmBuffer));

        std::unique_ptr<DType[]> refWghBuffer(new DType[refWghDataSize]);
        m_refWghBuffers.push_back(std::move(refWghBuffer));

        std::unique_ptr<DType[]> refOfmBuffer(new DType[refOfmDataSize]);
        m_refOfmBuffers.push_back(std::move(refOfmBuffer));
    }
    for (auto& refOutBuffer : refOutBuffers)
    {
        for (int i = 0; i < refOutDataSize; ++i)
        {
            if constexpr (std::is_same_v<DType, fp8_152_t> || std::is_same_v<DType, fp8_143_t>)
            {
                refOutBuffer.get()[i] = static_cast<DType>(0);
            }
            else
            {
                refOutBuffer.get()[i] = 0;
            }
        }
    }

}

template<typename DType>
void SynTrainingGConvFwdBwdTest<DType>::calculateRefs(unsigned*                              refIn1DimSizes,
                                                      unsigned*                              refIn2DimSizes,
                                                      unsigned*                              refOutDimSizes,
                                                      std::vector<std::unique_ptr<DType[]>>& refIn1Buffers,
                                                      std::vector<std::unique_ptr<DType[]>>& refIn2Buffers,
                                                      std::vector<std::unique_ptr<DType[]>>& refOutBuffers,
                                                      Node::eNodeType                        nodeType)
{
    synConvolutionParams convolutionParams;

    synDataType dataDtype = dataTypeToSynType<DType>();

    for (int i = 0; i < m_nGroups; i++)
    {
        unsigned refIn1Tensor = createTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             (float*)refIn1Buffers[i].get(),
                                             refIn1DimSizes,
                                             4,
                                             dataDtype);
        unsigned refIn2Tensor = createTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             (float*)refIn2Buffers[i].get(),
                                             refIn2DimSizes,
                                             4,
                                             dataDtype);
        unsigned refOutTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, refOutDimSizes, 4, dataDtype);
        switch (nodeType)
        {
            case Node::TYPE_CONVOLUTION:
                calculateFwdConvolution(m_tensorDescs[refIn1Tensor],
                                        (char*)refIn1Buffers[i].get(),
                                        m_tensorDescs[refIn2Tensor],
                                        (char*)refIn2Buffers[i].get(),
                                        m_tensorDescs[refOutTensor],
                                        (char*)refOutBuffers[i].get(),
                                        convolutionParams,
                                        m_deviceType);
                break;
            case Node::TYPE_DEDW:
                calculateDEDW(m_tensorDescs[refIn1Tensor],
                              (char*)refIn1Buffers[i].get(),
                              m_tensorDescs[refIn2Tensor],
                              (char*)refIn2Buffers[i].get(),
                              m_tensorDescs[refOutTensor],
                              (char*)refOutBuffers[i].get(),
                              convolutionParams,
                              m_deviceType);
                break;
            case Node::TYPE_DEDX:
                calculateDEDX(m_tensorDescs[refIn1Tensor],
                              (char*)refIn1Buffers[i].get(),
                              m_tensorDescs[refIn2Tensor],
                              (char*)refIn2Buffers[i].get(),
                              m_tensorDescs[refOutTensor],
                              (char*)refOutBuffers[i].get(),
                              convolutionParams,
                              m_deviceType);
                break;
            default: break;
        }
    }
}

template<typename DType>
void SynTrainingGConvFwdBwdTest<DType>::concatRefOutputBuffer(const unsigned                         dataSize,
                                                              const unsigned                         jump,
                                                              std::vector<std::unique_ptr<DType[]>>& refOutBuffers)
{
    std::unique_ptr<unsigned[]> refBuffItrs(new unsigned[m_nGroups]{0});
    m_refConcatBuffer.reset(new DType[dataSize]);
    for (unsigned i = 0; i < dataSize; i++)
    {
        unsigned refBuffIdx = (i % (jump)) / (jump / m_nGroups);
        m_refConcatBuffer[i] = refOutBuffers[refBuffIdx][refBuffItrs[refBuffIdx]];
        refBuffItrs[refBuffIdx]++;
    }
}

template<typename DType>
void SynTrainingGConvFwdBwdTest<DType>::validateFinalResult(unsigned dataSize)
{
    DType* pOutputBuffer = castHostOutBuffer<DType>(m_outTensorIndex);
    validateResult<DType, DType>(m_refConcatBuffer.get(), pOutputBuffer, dataSize);
}

template<typename DType>
void SynTrainingGConvFwdBwdTest<DType>::densifyOutputBuffers(std::vector<std::unique_ptr<DType[]>>& refOutBuffers,
                                                             std::vector<std::unique_ptr<DType[]>>& refDenseOutBuffers,
                                                             const unsigned                         actualOutputSize)
{
    unsigned actualSizePerGroup = actualOutputSize / m_nGroups;
    for (int i = 0; i < m_nGroups; i++)
    {
        std::unique_ptr<DType[]> denseOfmBuffer(new DType[actualSizePerGroup]);
        memcpy(denseOfmBuffer.get(), refOutBuffers[i].get(), actualSizePerGroup * sizeof(DType));
        refDenseOutBuffers.push_back(std::move(denseOfmBuffer));
    }
}

template<typename DType>
void SynTrainingGConvFwdBwdTest<DType>::runGroupedConvolution(Node::eNodeType nodeType, bool check_filter2d_replaced)
{
    GlobalConfTestSetter gConvVar("ENABLE_GCONV_PACKING", "1");
    synConvolutionParams params;
    const unsigned groups       = testing::get<0>(GetParam());
    const unsigned batch        = testing::get<1>(GetParam());
    const unsigned nFactorIfm   = testing::get<2>(GetParam()); //By what factor to multiply groups to get nIFM
    const unsigned nFactorOfm   = testing::get<3>(GetParam()); //By what factor to multiply groups to get nOFM
    const unsigned wOFM         = testing::get<4>(GetParam());
    const unsigned hOFM         = testing::get<5>(GetParam());
    const unsigned KH           = testing::get<6>(GetParam());
    const unsigned KW           = testing::get<7>(GetParam());
    m_isDynamic                 = testing::get<8>(GetParam());
    const unsigned nIFM         = groups * nFactorIfm;
    const unsigned nOFM         = groups * nFactorOfm;
    const unsigned minBatch     = batch;
    const unsigned actualBatch  = m_isDynamic ? std::max(1u, batch / 2) : batch;

    params.dH   = 1;
    params.dW   = 1;
    params.kH   = KH;
    params.kW   = KW;
    params.dilH = 1;
    params.dilW = 1;

    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    params.nGroups = groups;

    const unsigned wIFM = convInputDimSize(wOFM, KW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hIFM = convInputDimSize(hOFM, KH, params.dH, params.padT + params.padB, params.dilH);

    unsigned ifmDimSizes[]    = {nIFM, wIFM, hIFM, batch};
    unsigned wghDimSizes[]    = {nOFM, nIFM / groups, KW, KH};
    unsigned ofmDimSizes[]    = {nOFM, wOFM, hOFM, batch};
    unsigned ifmDimMinSizes[] = {nIFM, wIFM, hIFM, minBatch};
    unsigned ofmDimMinSizes[] = {nOFM, wOFM, hOFM, minBatch};

    unsigned refIfmDimSizes[] = {nIFM / groups, wIFM, hIFM, batch};
    unsigned refWghDimSizes[] = {nOFM / groups, nIFM / groups, KW, KH};
    unsigned refOfmDimSizes[] = {nOFM / groups, wOFM, hOFM, batch};

    m_ifmDataSize = ifmDimSizes[0] * ifmDimSizes[1] * ifmDimSizes[2] * ifmDimSizes[3];
    m_wghDataSize = wghDimSizes[0] * wghDimSizes[1] * wghDimSizes[2] * wghDimSizes[3];
    m_ofmDataSize = ofmDimSizes[0] * ofmDimSizes[1] * ofmDimSizes[2] * ofmDimSizes[3];
    m_nGroups = groups;

    std::unique_ptr<DType[]> ifmBuffer;
    std::unique_ptr<DType[]> wghBuffer;
    std::unique_ptr<DType[]> ofmBuffer;

    unsigned actualOutDataSize = ofmDimSizes[0] * ofmDimSizes[1] * ofmDimSizes[2] * actualBatch;
    std::array<unsigned, 4> in1ActualSizes;
    std::array<unsigned, 4> in2ActualSizes;
    std::array<unsigned, 4> inShapeActualSizes;
    std::array<unsigned, 4> outActualSizes;
    std::vector<std::unique_ptr<DType[]>> refDenseOutBuffers;
    switch (nodeType)
    {
        case Node::TYPE_CONVOLUTION:
            in1ActualSizes  = {nIFM, wIFM, hIFM, actualBatch};
            in2ActualSizes  = {nOFM, nIFM / groups, KW, KH};
            outActualSizes  = {nOFM, wOFM, hOFM, actualBatch};
            addGroupedNode(ifmDimSizes,
                           wghDimSizes,
                           ofmDimSizes,
                           ifmBuffer,
                           wghBuffer,
                           params,
                           "spatial_convolution",
                           ifmDimMinSizes,
                           wghDimSizes,
                           ofmDimMinSizes);
            initializeRefs(m_refOfmBuffers, m_ofmDataSize/groups);
            fillRefs(m_refIfmBuffers, ifmBuffer, m_ifmDataSize, nIFM);
            fillRefs(m_refWghBuffers, wghBuffer, m_wghDataSize, nOFM);
            calculateRefs(refIfmDimSizes, refWghDimSizes, refOfmDimSizes, m_refIfmBuffers, m_refWghBuffers, m_refOfmBuffers, nodeType);
            densifyOutputBuffers(m_refOfmBuffers, refDenseOutBuffers, actualOutDataSize);
            concatRefOutputBuffer(actualOutDataSize, nOFM, refDenseOutBuffers);
            break;
        case Node::TYPE_DEDW:
            in1ActualSizes  = {nOFM, wOFM, hOFM, actualBatch};
            in2ActualSizes  = {nIFM, wIFM, hIFM, actualBatch};
            outActualSizes  = {nOFM, nIFM / groups, KW, KH};
            actualOutDataSize = m_wghDataSize;
            addGroupedNode(ofmDimSizes,
                           ifmDimSizes,
                           wghDimSizes,
                           ofmBuffer,
                           ifmBuffer,
                           params,
                           "dedw",
                           ofmDimMinSizes,
                           ifmDimMinSizes,
                           wghDimSizes);
            initializeRefs(m_refWghBuffers, m_wghDataSize/groups);
            fillRefs(m_refIfmBuffers, ifmBuffer, m_ifmDataSize, nIFM);
            fillRefs(m_refOfmBuffers, ofmBuffer, m_ofmDataSize, nOFM);
            calculateRefs(refOfmDimSizes, refIfmDimSizes, refWghDimSizes, m_refOfmBuffers, m_refIfmBuffers, m_refWghBuffers, nodeType);
            concatRefOutputBuffer(m_wghDataSize, nOFM, m_refWghBuffers);
            break;
        case Node::TYPE_DEDX:
            in1ActualSizes  = {nOFM, wOFM, hOFM, actualBatch};
            in2ActualSizes  = {nOFM, nIFM / groups, KW, KH};
            inShapeActualSizes  = {nIFM, wIFM, hIFM, actualBatch};
            outActualSizes  = {nIFM, wIFM, hIFM, actualBatch};
            actualOutDataSize = ifmDimSizes[0] * ifmDimSizes[1] * ifmDimSizes[2] * actualBatch;
            addGroupedNode(ofmDimSizes,
                           wghDimSizes,
                           ifmDimSizes,
                           ofmBuffer,
                           wghBuffer,
                           params,
                           "dedx",
                           ofmDimMinSizes,
                           wghDimSizes,
                           ifmDimMinSizes);
            initializeRefs(m_refIfmBuffers, m_ifmDataSize/groups);
            fillRefs(m_refWghBuffers, wghBuffer, m_wghDataSize, nOFM);
            fillRefs(m_refOfmBuffers, ofmBuffer, m_ofmDataSize, nOFM);
            calculateRefs(refOfmDimSizes, refWghDimSizes, refIfmDimSizes, m_refOfmBuffers, m_refWghBuffers, m_refIfmBuffers, nodeType);
            densifyOutputBuffers(m_refIfmBuffers, refDenseOutBuffers, actualOutDataSize);
            concatRefOutputBuffer(m_ifmDataSize, nIFM, m_refIfmBuffers);
            break;
        default: break;
    }

    // before compilation, keep input & output tensors for comparison
    HabanaGraph* pGraph = synSingleton::getInstanceInternal()->getGraph(m_graphs[0].graphHandle);
    TensorPtr origIFM;
    TensorPtr origOFM;

    const NodeVector& origNodes = pGraph->getExeSortedNodes();
    for (const NodePtr& foundNode : origNodes)
    {
        switch (nodeType)
        {
            case Node::TYPE_CONVOLUTION:
            {
                std::shared_ptr<ConvolutionNode> convNode = std::dynamic_pointer_cast<ConvolutionNode>(foundNode);
                if (convNode != nullptr)
                {
                    origIFM = convNode->getInput(0);
                    origOFM = convNode->getOutput(0);
                }
                break;
            }
            default:
                check_filter2d_replaced = false;
        }
    }

    compileTopology();
    ASSERT_FALSE(HasFailure());

    setActualSizes(m_in1TensorIndex, in1ActualSizes.data());
    setActualSizes(m_in2TensorIndex, in2ActualSizes.data());
    setActualSizes(m_outTensorIndex, outActualSizes.data());
    if (nodeType == Node::TYPE_DEDX) setActualSizes(m_inShapeTensorIndex, inShapeActualSizes.data());

    runTopology(0, true);

    if (check_filter2d_replaced)
    {
        pGraph = synSingleton::getInstanceInternal()->getGraph(m_graphs[0].graphHandle);
        const NodeVector& compiledNodes = pGraph->getExeSortedNodes();
        ASSERT_EQ(compiledNodes.size(),1) << "Invalid num nodes (Created " << compiledNodes.size() << " , expecting 1)";

        for (const NodePtr& foundNode : compiledNodes)
        {
            ASSERT_FALSE(foundNode->getGUID() == "spatial_convolution") << "Convolution Node was not removed";
            TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(foundNode);
            if (tpcNode != nullptr)
            {
                if (tpcNode->isGuidPrefix("filter_2d"))
                {
                    ASSERT_EQ(tpcNode->getInput(TENSOR_IFM), origIFM) << "Invalid input tensor to filter";
                    ASSERT_EQ(tpcNode->getOutput(TENSOR_OFM), origOFM) << "Invalid output tensor to filter";
                }
            }
        }

    }
    validateFinalResult(actualOutDataSize);
}

class SynTrainingGConvPackingBf16Test : public SynTrainingGConvFwdBwdTest<bfloat16>
{
};

class SynTrainingGConvPackingFP8Test : public SynTrainingGConvFwdBwdTest<fp8_152_t>
{
};

class SynTrainingGConvPackingHFP8Test : public SynTrainingGConvFwdBwdTest<fp8_143_t>
{
};

class SynTrainingGConvPackingFloatTest : public SynTrainingGConvFwdBwdTest<float>
{
};

INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingGConvPackingFloatTest,
                         ::testing::Values(std::make_tuple(4, 1, 16, 16, 5, 5, 3, 3, false),
                                           std::make_tuple(3, 1, 1, 2, 2, 2, 3, 3, false),
                                           std::make_tuple(3, 1, 32, 42, 5, 5, 3, 3, false),
                                           std::make_tuple(12, 2, 5, 3, 8, 8, 3, 3, true),
                                           std::make_tuple(32, 4, 1, 1, 5, 5, 3, 3, true),
                                           std::make_tuple(80, 4, 4, 4, 56, 56, 3, 3, true),
                                           std::make_tuple(64, 4, 4, 4, 56, 56, 3, 3, true),
                                           std::make_tuple(64, 4, 4, 4, 4, 4, 1, 1, true),
                                           std::make_tuple(64, 4, 4, 4, 56, 56, 3, 3, false),
                                           std::make_tuple(32, 4, 4, 4, 56, 56, 3, 3, true),
                                           std::make_tuple(2, 4, 4, 4, 56, 56, 3, 3, true),
                                           std::make_tuple(16, 4, 2, 1, 128, 256, 3, 3, false)));
//resnext requirements
INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_L2,
    SynTrainingGConvPackingFloatTest,
    ::testing::Values(std::make_tuple(4, 1, 16, 16, 5, 5, 2, 2, false),            // 1 diag
                      std::make_tuple(12, 1, 16, 16, 5, 5, 3, 3, false),           // 3 diags
                      std::make_tuple(32, 1, 8, 8, 64, 64, 3, 3, false),           // 4 diags
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, false),         // 4 diags, resnext sizes
                      std::make_tuple((64 / 4) * 5, 4, 4, 4, 56, 56, 3, 3, true),  // 5 diags, resnext sizes
                      std::make_tuple(8, 4, 16, 16, 14, 14, 3, 3, true)));         // 2 diags, resnext sizes

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingFloatTest,
    ::testing::Values(std::make_tuple(4 + 2, 1, 16, 16, 5, 5, 2, 2, false),     // 1 diag (4 in diag) + 2 in remainder
                      std::make_tuple(32 + 1, 4, 16, 16, 14, 14, 3, 3, false),  // 2 diags (4 in diag) + 1 in remainder
                      std::make_tuple(12 + 3, 1, 16, 16, 5, 5, 3, 3, false),    // 3 diags (4 in diag) + 3 in remainder
                      std::make_tuple(32 + 6, 1, 8, 8, 64, 64, 3, 3, false),    // 4 diags (8 in diag) + 6 in remainder
                      std::make_tuple(80 + 11, 4, 4, 4, 56, 56, 3, 3, false)  // 5 diags (16 in diag) + 11 in remainder
                      ));

INSTANTIATE_TEST_SUITE_P(gconv_no_remainder_unaligned_L2,
                         SynTrainingGConvPackingFloatTest,  // c != k for all cases
                         ::testing::Values(std::make_tuple((64 / 10) * 1, 1, 16, 10, 5, 5, 3, 3, false),  // 1 diag
                                           std::make_tuple((64 / 10) * 1, 1, 7, 10, 5, 5, 3, 3, false),   // 1 diag
                                           std::make_tuple((64 / 12) * 2, 1, 10, 12, 5, 5, 3, 3, false),  // 2 diags
                                           std::make_tuple((64 / 6) * 2, 1, 10, 6, 5, 5, 3, 3, false),    // 2 diags
                                           std::make_tuple((64 / 11) * 3, 1, 3, 11, 5, 5, 3, 3, false),   // 3 diags
                                           std::make_tuple((64 / 6) * 4, 1, 4, 6, 5, 5, 3, 3, false))     // 4 diags
);

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_unaligned_ASIC_CI,
    SynTrainingGConvPackingFloatTest,
    ::testing::Values(std::make_tuple(4, 1, 16, 10, 5, 5, 3, 3, false),   // 1 diag - 4 in remainder (out of 6 in diag)
                      std::make_tuple(12, 1, 16, 6, 5, 5, 3, 3, false),   // 1 diags (10 in diag)  + 2 in remainder
                      std::make_tuple(15, 1, 16, 10, 5, 5, 3, 3, false),  // 2 diags (6 in diag)  + 3 in remainder
                      std::make_tuple(22, 1, 16, 6, 5, 5, 3, 3, false),   // 2 diags (10 in diag)   + 2 in remainder
                      std::make_tuple(16, 1, 16, 5, 5, 5, 3, 3, false),   // 1 diag  (12 in diag) + 4 in remainder
                      std::make_tuple(21, 1, 3, 10, 5, 5, 3, 3, false),   // 3 diags (6 in diag)   + 3 in remainder
                      std::make_tuple(27, 1, 4, 12, 5, 5, 3, 3, false))   // 5 diags (5 in diag)   + 2 in remainder
);
//resnext requirements

INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingBf16Test,
    ::testing::Values(std::make_tuple(32, 1, 8, 8, 64, 64, 3, 3, false),     // 4 diags
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, false),   // 4 diags, resnext sizes
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, true)));  // 4 diags, resnext sizes, DS

INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_L2,
    SynTrainingGConvPackingBf16Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 16, 5, 5, 2, 2, false),            // 1 diag
                      std::make_tuple(12, 1, 16, 16, 5, 5, 3, 3, false),           // 3 diags
                      std::make_tuple((64 / 4) * 5, 4, 4, 4, 56, 56, 3, 3, true),  // 5 diags, resnext sizes
                      std::make_tuple(8, 4, 16, 16, 14, 14, 3, 3, true)));         // 2 diags, resnext sizes

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingBf16Test,
    ::testing::Values(std::make_tuple(4 + 2, 1, 16, 16, 5, 5, 2, 2, false),    // 1 diag (4 in diag) + 2 in remainder
                      std::make_tuple(32 + 1, 4, 16, 16, 14, 14, 3, 3, true),  // 2 diags (4 in diag) + 1 in remainder
                      std::make_tuple(12 + 3, 1, 16, 16, 5, 5, 3, 3, false),   // 3 diags (4 in diag) + 3 in remainder
                      std::make_tuple(32 + 6, 1, 8, 8, 64, 64, 3, 3, false),   // 4 diags (8 in diag) + 6 in remainder
                      std::make_tuple(80 + 11, 4, 4, 4, 56, 56, 3, 3, true)    // 5 diags (16 in diag) + 11 in remainder
                      ));

INSTANTIATE_TEST_SUITE_P(gconv_no_remainder_unaligned_L2,
                         SynTrainingGConvPackingBf16Test,  // c != k for all cases
                         ::testing::Values(std::make_tuple((64 / 10) * 1, 1, 16, 10, 5, 5, 3, 3, false),  // 1 diag
                                           std::make_tuple((64 / 10) * 1, 1, 7, 10, 5, 5, 3, 3, false),   // 1 diag
                                           std::make_tuple((64 / 12) * 2, 1, 10, 12, 5, 5, 3, 3, false),  // 2 diags
                                           std::make_tuple((64 / 6) * 2, 1, 10, 6, 5, 5, 3, 3, false),    // 2 diags
                                           std::make_tuple((64 / 11) * 3, 1, 3, 11, 5, 5, 3, 3, false),   // 3 diags
                                           std::make_tuple((64 / 6) * 4, 1, 4, 6, 5, 5, 3, 3, false))     // 4 diags
);

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_unaligned_ASIC_CI,
    SynTrainingGConvPackingBf16Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 10, 5, 5, 3, 3, false),   // 1 diag - 4 in remainder (out of 6 in diag)
                      std::make_tuple(12, 1, 16, 6, 5, 5, 3, 3, false),   // 1 diags (10 in diag)  + 2 in remainder
                      std::make_tuple(15, 1, 16, 10, 5, 5, 3, 3, false),  // 2 diags (6 in diag)  + 3 in remainder
                      std::make_tuple(22, 1, 16, 6, 5, 5, 3, 3, false),   // 2 diags (10 in diag)   + 2 in remainder
                      std::make_tuple(16, 1, 16, 5, 5, 5, 3, 3, false),   // 1 diag  (12 in diag) + 4 in remainder
                      std::make_tuple(21, 1, 3, 10, 5, 5, 3, 3, false),   // 3 diags (6 in diag)   + 3 in remainder
                      std::make_tuple(27, 1, 4, 12, 5, 5, 3, 3, false))   // 5 diags (5 in diag)   + 2 in remainder
);

INSTANTIATE_TEST_SUITE_P(big_images_ASIC_CI,
                         SynTrainingGConvPackingBf16Test,
                         ::testing::Values(std::make_tuple(8, 1, 16, 16, 512, 512, 3, 3, false)));

TEST_P_GC(SynTrainingGConvPackingBf16Test, conv_group_test)
{
    runGroupedConvolution(Node::TYPE_CONVOLUTION);
}

TEST_P_GC(SynTrainingGConvPackingBf16Test, dedx_group_test)
{
    runGroupedConvolution(Node::TYPE_DEDX);
}

TEST_P_GC(SynTrainingGConvPackingBf16Test, dedw_group_test)
{
    runGroupedConvolution(Node::TYPE_DEDW);
}

INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingFP8Test,
    ::testing::Values(std::make_tuple(32, 1, 8, 8, 64, 64, 3, 3, false),     // 4 diags
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, false),   // 4 diags, resnext sizes
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, true)));  // 4 diags, resnext sizes, DS

INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_L2,
    SynTrainingGConvPackingFP8Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 16, 5, 5, 2, 2, false),            // 1 diag
                      std::make_tuple(12, 1, 16, 16, 5, 5, 3, 3, false),           // 3 diags
                      std::make_tuple((64 / 4) * 5, 4, 4, 4, 56, 56, 3, 3, true),  // 5 diags, resnext sizes
                      std::make_tuple(8, 4, 16, 16, 14, 14, 3, 3, true)));         // 2 diags, resnext sizes

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingFP8Test,
    ::testing::Values(std::make_tuple(4 + 2, 1, 16, 16, 5, 5, 2, 2, false),    // 1 diag (4 in diag) + 2 in remainder
                      std::make_tuple(32 + 1, 4, 16, 16, 14, 14, 3, 3, true),  // 2 diags (4 in diag) + 1 in remainder
                      std::make_tuple(12 + 3, 1, 16, 16, 5, 5, 3, 3, false),   // 3 diags (4 in diag) + 3 in remainder
                      std::make_tuple(32 + 6, 1, 8, 8, 64, 64, 3, 3, false),   // 4 diags (8 in diag) + 6 in remainder
                      std::make_tuple(80 + 11, 4, 4, 4, 56, 56, 3, 3, true)    // 5 diags (16 in diag) + 11 in remainder
                      ));

INSTANTIATE_TEST_SUITE_P(gconv_no_remainder_unaligned_L2,
                         SynTrainingGConvPackingFP8Test,  // c != k for all cases
                         ::testing::Values(std::make_tuple((64 / 10) * 1, 1, 16, 10, 5, 5, 3, 3, false),  // 1 diag
                                           std::make_tuple((64 / 10) * 1, 1, 7, 10, 5, 5, 3, 3, false),   // 1 diag
                                           std::make_tuple((64 / 12) * 2, 1, 10, 12, 5, 5, 3, 3, false),  // 2 diags
                                           std::make_tuple((64 / 6) * 2, 1, 10, 6, 5, 5, 3, 3, false),    // 2 diags
                                           std::make_tuple((64 / 11) * 3, 1, 3, 11, 5, 5, 3, 3, false),   // 3 diags
                                           std::make_tuple((64 / 6) * 4, 1, 4, 6, 5, 5, 3, 3, false))     // 4 diags
);

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_unaligned_ASIC_CI,
    SynTrainingGConvPackingFP8Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 10, 5, 5, 3, 3, false),   // 1 diag - 4 in remainder (out of 6 in diag)
                      std::make_tuple(12, 1, 16, 6, 5, 5, 3, 3, false),   // 1 diags (10 in diag)  + 2 in remainder
                      std::make_tuple(15, 1, 16, 10, 5, 5, 3, 3, false),  // 2 diags (6 in diag)  + 3 in remainder
                      std::make_tuple(22, 1, 16, 6, 5, 5, 3, 3, false),   // 2 diags (10 in diag)   + 2 in remainder
                      std::make_tuple(16, 1, 16, 5, 5, 5, 3, 3, false),   // 1 diag  (12 in diag) + 4 in remainder
                      std::make_tuple(21, 1, 3, 10, 5, 5, 3, 3, false),   // 3 diags (6 in diag)   + 3 in remainder
                      std::make_tuple(27, 1, 4, 12, 5, 5, 3, 3, false))   // 5 diags (5 in diag)   + 2 in remainder
);

INSTANTIATE_TEST_SUITE_P(DISABLED_big_images_ASIC_CI,
                         SynTrainingGConvPackingFP8Test,
                         ::testing::Values(std::make_tuple(8, 1, 16, 16, 512, 512, 3, 3, false)));

TEST_P_GC(SynTrainingGConvPackingFP8Test, conv_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    runGroupedConvolution(Node::TYPE_CONVOLUTION);
}

TEST_P_GC(SynTrainingGConvPackingFP8Test, dedx_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    runGroupedConvolution(Node::TYPE_DEDX);
}

TEST_P_GC(SynTrainingGConvPackingFP8Test, dedw_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    runGroupedConvolution(Node::TYPE_DEDW);
}

INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingHFP8Test,
    ::testing::Values(std::make_tuple(32, 1, 8, 8, 64, 64, 3, 3, false),     // 4 diags
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, false),   // 4 diags, resnext sizes
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, true)));  // 4 diags, resnext sizes, DS

INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_L2,
    SynTrainingGConvPackingHFP8Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 16, 5, 5, 2, 2, false),            // 1 diag
                      std::make_tuple(12, 1, 16, 16, 5, 5, 3, 3, false),           // 3 diags
                      std::make_tuple((64 / 4) * 5, 4, 4, 4, 56, 56, 3, 3, true),  // 5 diags, resnext sizes
                      std::make_tuple(8, 4, 16, 16, 14, 14, 3, 3, true)));         // 2 diags, resnext sizes

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingHFP8Test,
    ::testing::Values(std::make_tuple(4 + 2, 1, 16, 16, 5, 5, 2, 2, false),    // 1 diag (4 in diag) + 2 in remainder
                      std::make_tuple(32 + 1, 4, 16, 16, 14, 14, 3, 3, true),  // 2 diags (4 in diag) + 1 in remainder
                      std::make_tuple(12 + 3, 1, 16, 16, 5, 5, 3, 3, false),   // 3 diags (4 in diag) + 3 in remainder
                      std::make_tuple(32 + 6, 1, 8, 8, 64, 64, 3, 3, false),   // 4 diags (8 in diag) + 6 in remainder
                      std::make_tuple(80 + 11, 4, 4, 4, 56, 56, 3, 3, true)    // 5 diags (16 in diag) + 11 in remainder
                      ));

INSTANTIATE_TEST_SUITE_P(gconv_no_remainder_unaligned_L2,
                         SynTrainingGConvPackingHFP8Test,  // c != k for all cases
                         ::testing::Values(std::make_tuple((64 / 10) * 1, 1, 16, 10, 5, 5, 3, 3, false),  // 1 diag
                                           std::make_tuple((64 / 10) * 1, 1, 7, 10, 5, 5, 3, 3, false),   // 1 diag
                                           std::make_tuple((64 / 12) * 2, 1, 10, 12, 5, 5, 3, 3, false),  // 2 diags
                                           std::make_tuple((64 / 6) * 2, 1, 10, 6, 5, 5, 3, 3, false),    // 2 diags
                                           std::make_tuple((64 / 11) * 3, 1, 3, 11, 5, 5, 3, 3, false),   // 3 diags
                                           std::make_tuple((64 / 6) * 4, 1, 4, 6, 5, 5, 3, 3, false))     // 4 diags
);

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_unaligned_ASIC_CI,
    SynTrainingGConvPackingHFP8Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 10, 5, 5, 3, 3, false),   // 1 diag - 4 in remainder (out of 6 in diag)
                      std::make_tuple(12, 1, 16, 6, 5, 5, 3, 3, false),   // 1 diags (10 in diag)  + 2 in remainder
                      std::make_tuple(15, 1, 16, 10, 5, 5, 3, 3, false),  // 2 diags (6 in diag)  + 3 in remainder
                      std::make_tuple(22, 1, 16, 6, 5, 5, 3, 3, false),   // 2 diags (10 in diag)   + 2 in remainder
                      std::make_tuple(16, 1, 16, 5, 5, 5, 3, 3, false),   // 1 diag  (12 in diag) + 4 in remainder
                      std::make_tuple(21, 1, 3, 10, 5, 5, 3, 3, false),   // 3 diags (6 in diag)   + 3 in remainder
                      std::make_tuple(27, 1, 4, 12, 5, 5, 3, 3, false))   // 5 diags (5 in diag)   + 2 in remainder
);

INSTANTIATE_TEST_SUITE_P(DISABLED_big_images_ASIC_CI,
                         SynTrainingGConvPackingHFP8Test,
                         ::testing::Values(std::make_tuple(8, 1, 16, 16, 512, 512, 3, 3, false)));

TEST_P_GC(SynTrainingGConvPackingHFP8Test, conv_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    runGroupedConvolution(Node::TYPE_CONVOLUTION);
}

TEST_P_GC(SynTrainingGConvPackingHFP8Test, dedx_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    runGroupedConvolution(Node::TYPE_DEDX);
}

TEST_P_GC(SynTrainingGConvPackingHFP8Test, dedw_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    runGroupedConvolution(Node::TYPE_DEDW);
}

TEST_P_GC(SynTrainingGConvPackingFloatTest, conv_group_test)
{
    runGroupedConvolution(Node::TYPE_CONVOLUTION);
}

TEST_P_GC(SynTrainingGConvPackingFloatTest, dedw_group_test_ASIC_CI)
{
    runGroupedConvolution(Node::TYPE_DEDW);
}

TEST_P_GC(SynTrainingGConvPackingFloatTest, dedx_group_test)
{
    runGroupedConvolution(Node::TYPE_DEDX);
}

TEST_F_GC(SynTrainingGConvPackingFloatTest, group_fail_test)
{
    // Graph compilation should fail on nOFM size
    synConvolutionParams params;

    params.dH   = 1;
    params.dW   = 1;
    params.kH   = 2;
    params.kW   = 2;
    params.dilH = 1;
    params.dilW = 1;

    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    params.nGroups = 3;

    const unsigned batch = 1;
    const unsigned nIFM  = 12;
    const unsigned nOFM  = 61; // 61 % 3 != 0. will fail compilation
    const unsigned wOFM  = 4;
    const unsigned hOFM  = 4;

    const unsigned wIFM = convInputDimSize(wOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hIFM = convInputDimSize(hOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    unsigned dims = 4;
    unsigned ifmDimSizes[] = {nIFM, wIFM, hIFM, batch};
    unsigned wghDimSizes[] = {nOFM, nIFM / params.nGroups, params.kW, params.kH};
    unsigned ofmDimSizes[] = {nOFM, wOFM, hOFM, batch};

    const unsigned m_ifmDataSize = ifmDimSizes[0] * ifmDimSizes[1] * ifmDimSizes[2] * ifmDimSizes[3];
    const unsigned m_wghDataSize = wghDimSizes[0] * wghDimSizes[1] * wghDimSizes[2] * wghDimSizes[3];

    // Input
    float* ifmBuffer = new float[m_ifmDataSize];
    fillWithRandom(ifmBuffer, m_ifmDataSize, {0, 4});
    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, ifmBuffer, ifmDimSizes, dims, syn_type_single);

    // Weights
    float* wghBuffer = new float[m_wghDataSize];
    fillWithRandom(wghBuffer, m_wghDataSize, {0, 2});
    unsigned wTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, wghBuffer, wghDimSizes, dims, syn_type_single);

    // Output
    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_single);

    // Create grouped convolution node (conv node with nGroup > 1
    addNodeToGraph("spatial_convolution", {xTensorIndex, wTensorIndex}, {yTensorIndex}, (void *) &params,
                   sizeof(synConvolutionParams));

    GraphData& graphData = m_graphs[0];
    synStatus status = synGraphCompile(&graphData.recipeHandle,
                                       graphData.graphHandle,
                                       graphData.recipeName.c_str(),
                                       nullptr);
    ASSERT_NE(status, synSuccess) << "graph compilation should fail";

    // Clean up
    delete [] ifmBuffer;
    delete [] wghBuffer;
}

class SynTrainingFilter2dConvertTest : public SynTrainingGConvFwdBwdTest<float>
{
};

INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingFilter2dConvertTest,
                         ::testing::Values(std::make_tuple(2, 1, 1, 1, 32, 70, 5, 5, false)));

TEST_P_GC(SynTrainingFilter2dConvertTest, conv_filter_2d)
{
    auto is_pass_enabled = GCFG_GAUDI_ENABLE_GROUP_CONV_TO_FILTER_2D.value();
    GCFG_GAUDI_ENABLE_GROUP_CONV_TO_FILTER_2D.setValue(true);
    // Running group convolution with C = K = m_nGroups = 2, to validate conversion to filter2d.
    runGroupedConvolution(Node::TYPE_CONVOLUTION, true);
    // set the original config value back:
    GCFG_GAUDI_ENABLE_GROUP_CONV_TO_FILTER_2D.setValue(is_pass_enabled);
}

TEST_F_GC(SynTrainingTwoRunCompareTest, grouped_dedx_dedw_with_shared_input_ASIC_CI)
{
    unsigned sharedInSizes[] = {512, 14, 14, 256};
    unsigned sharedIn        = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "sharedIn",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sharedInSizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      sharedInSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned dedwInSizes[] = {512, 14, 14, 256};
    unsigned dedwIn        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "dedwIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    dedwInSizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    dedwInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned dedwOutSizes[] = {512, 16, 3, 3};
    unsigned dedwOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "dedwOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     dedwOutSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedwOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned char dedwParams[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0,  0,   1,   0,   0, 0,
                                  1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,  0, 56, 200, 1,   0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0,  0,   255, 127, 0, 0};
    addNodeToGraph("dedw", {sharedIn, dedwIn}, {dedwOut}, (void*)dedwParams, 72, "DEDW");

    unsigned dedxInSizes[] = {512, 16, 3, 3};
    unsigned dedxIn        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "dedxIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    dedxInSizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    dedxInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned dedxOutSizes[] = {512, 14, 14, 256};
    unsigned dedxOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "dedxOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     dedxOutSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedxOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned char dedxParams[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0,  0,   1,   0,   0, 0,
                                  1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,  0, 56, 200, 1,   0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0,  0,   255, 127, 0, 0};
    addNodeToGraph("dedx", {sharedIn, dedxIn}, {dedxOut}, (void*)dedxParams, 72, "DEDX");

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(FIRST_RUN, "ENABLE_GCONV_SPLIT_NODES_REUSE", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_GCONV_SPLIT_NODES_REUSE", "true");

    compareRunsResults({dedwOut, dedxOut});
}

TEST_F_GC(SynTrainingTwoRunCompareTest, grouped_dedx_dedw_with_shared_input_with_producers_ASIC_CI)
{
    unsigned sharedInSizes[] = {512, 14, 14, 256};

    unsigned sharedInProducer = createTensors(1,
                                              INPUT_TENSOR,
                                              true,
                                              "sharedInProducer",
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              sharedInSizes,
                                              4,
                                              syn_type_bf16,
                                              nullptr,
                                              0,
                                              0,
                                              nullptr,
                                              false,
                                              sharedInSizes,
                                              synTensorType::DATA_TENSOR)[0];

    unsigned sharedIn = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "sharedIn",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      sharedInSizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      sharedInSizes,
                                      synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_bf16", {sharedInProducer}, {sharedIn}, nullptr, 0, "relu_shared_in");

    unsigned dedwInSizes[] = {512, 14, 14, 256};

    unsigned dedwInProducer = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "dedwInProducer",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            dedwInSizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            dedwInSizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned dedwIn = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "dedwIn",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    dedwInSizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    dedwInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_bf16", {dedwInProducer}, {dedwIn}, nullptr, 0, "relu_dedw_in");

    unsigned dedwOutSizes[] = {512, 16, 3, 3};
    unsigned dedwOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "dedwOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     dedwOutSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedwOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned char dedwParams[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0,  0,   1,   0,   0, 0,
                                  1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,  0, 56, 200, 1,   0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0,  0,   255, 127, 0, 0};
    addNodeToGraph("dedw", {sharedIn, dedwIn}, {dedwOut}, (void*)dedwParams, 72, "DEDW");

    unsigned dedxInSizes[] = {512, 16, 3, 3};

    unsigned dedxInProducer = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "dedxInProducer",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            dedxInSizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            dedxInSizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned dedxIn = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "dedxIn",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    dedxInSizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    dedxInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_bf16", {dedxInProducer}, {dedxIn}, nullptr, 0, "relu_dedx_in");

    unsigned dedxOutSizes[] = {512, 14, 14, 256};
    unsigned dedxOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "dedxOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     dedxOutSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedxOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned char dedxParams[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0,  0,   1,   0,   0, 0,
                                  1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,  0, 56, 200, 1,   0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0,  0,   255, 127, 0, 0};
    addNodeToGraph("dedx", {sharedIn, dedxIn}, {dedxOut}, (void*)dedxParams, 72, "DEDX");

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(FIRST_RUN, "ENABLE_GCONV_SPLIT_NODES_REUSE", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_GCONV_SPLIT_NODES_REUSE", "true");

    compareRunsResults({dedwOut, dedxOut});
}

TEST_F_GC(SynTrainingTwoRunCompareTest, grouped_dedx_dedw_with_shared_input_with_control_dep_ASIC_CI)
{
    unsigned sharedInSizes[] = {512, 14, 14, 256};
    unsigned sharedIn        = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "sharedIn",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sharedInSizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      sharedInSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned dedxInSizes[] = {512, 16, 3, 3};
    unsigned dedxIn        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "dedxIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    dedxInSizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    dedxInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned dedxInShapeSizes[] = {512, 14, 14, 256};
    unsigned dedxInShape        = createTensors(1,
                                         INPUT_TENSOR,
                                         false,
                                         "dedxInShape",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         dedxInShapeSizes,
                                         4,
                                         syn_type_uint32,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         dedxInShapeSizes,
                                         synTensorType::SHAPE_TENSOR)[0];

    unsigned dedxOutSizes[] = {512, 14, 14, 256};
    unsigned dedxOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "dedxOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     dedxOutSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedxOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    synNodeId     dedxId;
    unsigned char dedxParams[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0,  0,  1,   0,   0, 0,
                                  1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,  0, 54, 23, 1,   0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0,  0,  253, 127, 0, 0};
    addNodeToGraph("dedx",
                   {sharedIn, dedxIn, dedxInShape},
                   {dedxOut},
                   (void*)dedxParams,
                   72,
                   "DEDX",
                   0 /*graphIndex*/,
                   &dedxId);

    unsigned dedwInSizes[] = {512, 14, 14, 256};
    unsigned dedwIn        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "dedwIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    dedwInSizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    dedwInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned dedwOutSizes[] = {512, 16, 3, 3};
    unsigned dedwOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "dedwOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     dedwOutSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedwOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    synNodeId     dedwId;
    unsigned char dedwParams[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0,  0,  1,   0,   0, 0,
                                  1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,  0, 54, 23, 1,   0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0,  0,  253, 127, 0, 0};
    addNodeToGraph("dedw", {sharedIn, dedwIn}, {dedwOut}, (void*)dedwParams, 72, "DEDW", 0 /*graphIndex*/, &dedwId);

    unsigned reluOutSizes[] = {512, 14, 14, 256};
    unsigned reluOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "reluOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     reluOutSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     reluOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    synNodeId reluId;
    addNodeToGraph("relu_bwd_bf16",
                   {dedxOut, dedwIn},
                   {reluOut},
                   nullptr,
                   0,
                   "RELU"
                   "0",
                   0 /*graphIndex*/,
                   &reluId);

    unsigned addInSizes[] = {512, 16, 3, 3};
    unsigned addIn        = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "addIn",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   addInSizes,
                                   4,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   addInSizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned addOutSizes[] = {512, 16, 3, 3};
    unsigned addOut        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    true,
                                    "addOut",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    addOutSizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addOutSizes,
                                    synTensorType::DATA_TENSOR)[0];

    synNodeId addId;
    addNodeToGraph("add_fwd_bf16", {addIn, dedwOut}, {addOut}, nullptr, 0, "ADD", 0 /*graphIndex*/, &addId);

    setNodeDependency(&dedwId, &reluId, 1, 1);
    setNodeDependency(&dedxId, &addId, 1, 1);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(FIRST_RUN, "ENABLE_GCONV_SPLIT_NODES_REUSE", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_GCONV_SPLIT_NODES_REUSE", "true");

    compareRunsResults({reluOut, addOut});
};

template<typename DType>
class SynTrainingGConvPackingGconvFwdConstantFoldingTest : public SynTrainingGConvFwdBwdTest<DType>
{
public:
    SynTrainingGConvPackingGconvFwdConstantFoldingTest() : SynTrainingGConvFwdBwdTest<DType>() {}

    void addGroupedNode(unsigned*                 in1DimSizes,
                        unsigned*                 in2DimSizes,
                        unsigned*                 outDimSizes,
                        std::unique_ptr<DType[]>& in1Buffer,
                        std::unique_ptr<DType[]>& in2Buffer,
                        synConvolutionParams      params,
                        const char*               guid,
                        unsigned*                 in1DimMinSizes,
                        unsigned*                 in2DimMinSizes,
                        unsigned*                 outDimMinSizes) override;
};

template<typename DType>
void SynTrainingGConvPackingGconvFwdConstantFoldingTest<DType>::addGroupedNode(unsigned*                 in1DimSizes,
                                                                               unsigned*                 in2DimSizes,
                                                                               unsigned*                 outDimSizes,
                                                                               std::unique_ptr<DType[]>& in1Buffer,
                                                                               std::unique_ptr<DType[]>& in2Buffer,
                                                                               synConvolutionParams      params,
                                                                               const char*               guid,
                                                                               unsigned*                 in1DimMinSizes,
                                                                               unsigned*                 in2DimMinSizes,
                                                                               unsigned*                 outDimMinSizes)
{
    synDataType dataDtype = dataTypeToSynType<DType>();

    unsigned in1DataSize = in1DimSizes[0] * in1DimSizes[1] * in1DimSizes[2] * in1DimSizes[3];
    unsigned in2DataSize = in2DimSizes[0] * in2DimSizes[1] * in2DimSizes[2] * in2DimSizes[3];

    this->m_in1TensorIndex = this->createPersistTensor(INPUT_TENSOR,
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       in1DimSizes,
                                                       4,
                                                       dataDtype,
                                                       nullptr,
                                                       "in1",
                                                       0,
                                                       0,
                                                       nullptr,
                                                       in1DimMinSizes);
    in1Buffer.reset(new DType[in1DataSize]);
    memcpy(in1Buffer.get(), this->m_hostBuffers[this->m_in1TensorIndex], in1DataSize * dataTypeSizeInBytes(dataDtype));

    this->m_in2TensorIndex      = this->createConstPersistTensor(INPUT_TENSOR,
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            in2DimSizes,
                                                            4,
                                                            dataDtype,
                                                            nullptr,
                                                            "in2");

    in2Buffer.reset(new DType[in2DataSize]);
    memcpy(in2Buffer.get(), this->m_hostBuffers[this->m_in2TensorIndex], in2DataSize * dataTypeSizeInBytes(dataDtype));

    this->m_outTensorIndex = this->createPersistTensor(OUTPUT_TENSOR,
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       outDimSizes,
                                                       4,
                                                       dataDtype,
                                                       nullptr,
                                                       "out",
                                                       0,
                                                       0,
                                                       nullptr,
                                                       outDimMinSizes);

    std::vector<unsigned> inputs = {this->m_in1TensorIndex, this->m_in2TensorIndex};
    if (std::strcmp(guid, NodeFactory::deDxNodeTypeName) == 0)
    {
        this->m_inShapeTensorIndex =
            this->createShapeTensor(INPUT_TENSOR, outDimSizes, outDimMinSizes, 4, dataDtype, "dx_shape");
        inputs.push_back(this->m_inShapeTensorIndex);
    }

    SynTrainingGConvFwdBwdTest<DType>::addNodeToGraph(guid,
                                                      inputs,
                                                      {this->m_outTensorIndex},
                                                      (void*)&params,
                                                      sizeof(synConvolutionParams),
                                                      guid);
};

class SynTrainingGConvPackingGconvFwdConstantFoldingBFloat16Test
: public SynTrainingGConvPackingGconvFwdConstantFoldingTest<bfloat16>
{
};

class SynTrainingGConvPackingGconvFwdConstantFoldingFP8Test
: public SynTrainingGConvPackingGconvFwdConstantFoldingTest<fp8_152_t>
{
};

class SynTrainingGConvPackingGconvFwdConstantFoldingHFP8Test
: public SynTrainingGConvPackingGconvFwdConstantFoldingTest<fp8_143_t>
{
};

class SynTrainingGConvPackingGconvFwdConstantFoldingFloatTest
: public SynTrainingGConvPackingGconvFwdConstantFoldingTest<float>
{
};

INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingGConvPackingGconvFwdConstantFoldingFloatTest,
                         ::testing::Values(std::make_tuple(4, 1, 16, 16, 5, 5, 3, 3, false),
                                           std::make_tuple(3, 1, 1, 2, 2, 2, 3, 3, false),
                                           std::make_tuple(3, 1, 32, 42, 5, 5, 3, 3, false),
                                           std::make_tuple(12, 2, 5, 3, 8, 8, 3, 3, true),
                                           std::make_tuple(32, 4, 1, 1, 5, 5, 3, 3, true),
                                           std::make_tuple(80, 4, 4, 4, 56, 56, 3, 3, true),
                                           std::make_tuple(64, 4, 4, 4, 56, 56, 3, 3, true),
                                           std::make_tuple(64, 4, 4, 4, 4, 4, 1, 1, true),
                                           std::make_tuple(64, 4, 4, 4, 56, 56, 3, 3, false),
                                           std::make_tuple(32, 4, 4, 4, 56, 56, 3, 3, true),
                                           std::make_tuple(2, 4, 4, 4, 56, 56, 3, 3, true),
                                           std::make_tuple(16, 4, 2, 1, 128, 256, 3, 3, false)));
// resnext requirements
INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_L2,
    SynTrainingGConvPackingGconvFwdConstantFoldingFloatTest,
    ::testing::Values(std::make_tuple(4, 1, 16, 16, 5, 5, 2, 2, false),            // 1 diag
                      std::make_tuple(12, 1, 16, 16, 5, 5, 3, 3, false),           // 3 diags
                      std::make_tuple(32, 1, 8, 8, 64, 64, 3, 3, false),           // 4 diags
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, false),         // 4 diags, resnext sizes
                      std::make_tuple((64 / 4) * 5, 4, 4, 4, 56, 56, 3, 3, true),  // 5 diags, resnext sizes
                      std::make_tuple(8, 4, 16, 16, 14, 14, 3, 3, true)            // 2 diags, resnext sizes
                      ));

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingGconvFwdConstantFoldingFloatTest,
    ::testing::Values(std::make_tuple(4 + 2, 1, 16, 16, 5, 5, 2, 2, false),     // 1 diag (4 in diag) + 2 in remainder
                      std::make_tuple(32 + 1, 4, 16, 16, 14, 14, 3, 3, false),  // 2 diags (4 in diag) + 1 in remainder
                      std::make_tuple(12 + 3, 1, 16, 16, 5, 5, 3, 3, false),    // 3 diags (4 in diag) + 3 in remainder
                      std::make_tuple(32 + 6, 1, 8, 8, 64, 64, 3, 3, false),    // 4 diags (8 in diag) + 6 in remainder
                      std::make_tuple(80 + 11, 4, 4, 4, 56, 56, 3, 3, false)  // 5 diags (16 in diag) + 11 in remainder
                      ));

INSTANTIATE_TEST_SUITE_P(gconv_no_remainder_unaligned_L2,
                         SynTrainingGConvPackingGconvFwdConstantFoldingFloatTest,  // c != k for all cases
                         ::testing::Values(std::make_tuple((64 / 10) * 1, 1, 16, 10, 5, 5, 3, 3, false),  // 1 diag
                                           std::make_tuple((64 / 10) * 1, 1, 7, 10, 5, 5, 3, 3, false),   // 1 diag
                                           std::make_tuple((64 / 12) * 2, 1, 10, 12, 5, 5, 3, 3, false),  // 2 diags
                                           std::make_tuple((64 / 6) * 2, 1, 10, 6, 5, 5, 3, 3, false),    // 2 diags
                                           std::make_tuple((64 / 11) * 3, 1, 3, 11, 5, 5, 3, 3, false),   // 3 diags
                                           std::make_tuple((64 / 6) * 4, 1, 4, 6, 5, 5, 3, 3, false))     // 4 diags
);

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_unaligned_ASIC_CI,
    SynTrainingGConvPackingGconvFwdConstantFoldingFloatTest,
    ::testing::Values(std::make_tuple(4, 1, 16, 10, 5, 5, 3, 3, false),   // 1 diag - 4 in remainder (out of 6 in diag)
                      std::make_tuple(12, 1, 16, 6, 5, 5, 3, 3, false),   // 1 diags (10 in diag)  + 2 in remainder
                      std::make_tuple(15, 1, 16, 10, 5, 5, 3, 3, false),  // 2 diags (6 in diag)  + 3 in remainder
                      std::make_tuple(22, 1, 16, 6, 5, 5, 3, 3, false),   // 2 diags (10 in diag)   + 2 in remainder
                      std::make_tuple(16, 1, 16, 5, 5, 5, 3, 3, false),   // 1 diag  (12 in diag) + 4 in remainder
                      std::make_tuple(21, 1, 3, 10, 5, 5, 3, 3, false),   // 3 diags (6 in diag)   + 3 in remainder
                      std::make_tuple(27, 1, 4, 12, 5, 5, 3, 3, false))   // 5 diags (5 in diag)   + 2 in remainder
);

TEST_P_GC(SynTrainingGConvPackingGconvFwdConstantFoldingFloatTest, conv_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    GCFG_ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING.setValue(true);
    runGroupedConvolution(Node::TYPE_CONVOLUTION);
}

TEST_P_GC(SynTrainingGConvPackingGconvFwdConstantFoldingFloatTest, dedx_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    GCFG_ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING.setValue(true);
    runGroupedConvolution(Node::TYPE_DEDX);
}

// resnext requirements
INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingGconvFwdConstantFoldingBFloat16Test,
    ::testing::Values(std::make_tuple(32, 1, 8, 8, 64, 64, 3, 3, false),    // 4 diags
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, false),  // 4 diags, resnext sizes
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, true)    // 4 diags, resnext sizes, DS
                      ));

INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_L2,
    SynTrainingGConvPackingGconvFwdConstantFoldingBFloat16Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 16, 5, 5, 2, 2, false),            // 1 diag
                      std::make_tuple(12, 1, 16, 16, 5, 5, 3, 3, false),           // 3 diags
                      std::make_tuple((64 / 4) * 5, 4, 4, 4, 56, 56, 3, 3, true),  // 5 diags, resnext sizes
                      std::make_tuple(8, 4, 16, 16, 14, 14, 3, 3, true)            // 2 diags, resnext sizes
                      ));

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingGconvFwdConstantFoldingBFloat16Test,
    ::testing::Values(std::make_tuple(4 + 2, 1, 16, 16, 5, 5, 2, 2, false),    // 1 diag (4 in diag) + 2 in remainder
                      std::make_tuple(32 + 1, 4, 16, 16, 14, 14, 3, 3, true),  // 2 diags (4 in diag) + 1 in remainder
                      std::make_tuple(12 + 3, 1, 16, 16, 5, 5, 3, 3, false),   // 3 diags (4 in diag) + 3 in remainder
                      std::make_tuple(32 + 6, 1, 8, 8, 64, 64, 3, 3, false),   // 4 diags (8 in diag) + 6 in remainder
                      std::make_tuple(80 + 11, 4, 4, 4, 56, 56, 3, 3, true)    // 5 diags (16 in diag) + 11 in remainder
                      ));

INSTANTIATE_TEST_SUITE_P(gconv_no_remainder_unaligned_L2,
                         SynTrainingGConvPackingGconvFwdConstantFoldingBFloat16Test,  // c != k for all cases
                         ::testing::Values(std::make_tuple((64 / 10) * 1, 1, 16, 10, 5, 5, 3, 3, false),  // 1 diag
                                           std::make_tuple((64 / 10) * 1, 1, 7, 10, 5, 5, 3, 3, false),   // 1 diag
                                           std::make_tuple((64 / 12) * 2, 1, 10, 12, 5, 5, 3, 3, false),  // 2 diags
                                           std::make_tuple((64 / 6) * 2, 1, 10, 6, 5, 5, 3, 3, false),    // 2 diags
                                           std::make_tuple((64 / 11) * 3, 1, 3, 11, 5, 5, 3, 3, false),   // 3 diags
                                           std::make_tuple((64 / 6) * 4, 1, 4, 6, 5, 5, 3, 3, false))     // 4 diags
);

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_unaligned_ASIC_CI,
    SynTrainingGConvPackingGconvFwdConstantFoldingBFloat16Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 10, 5, 5, 3, 3, false),   // 1 diag - 4 in remainder (out of 6 in diag)
                      std::make_tuple(12, 1, 16, 6, 5, 5, 3, 3, false),   // 1 diags (10 in diag)  + 2 in remainder
                      std::make_tuple(15, 1, 16, 10, 5, 5, 3, 3, false),  // 2 diags (6 in diag)  + 3 in remainder
                      std::make_tuple(22, 1, 16, 6, 5, 5, 3, 3, false),   // 2 diags (10 in diag)   + 2 in remainder
                      std::make_tuple(16, 1, 16, 5, 5, 5, 3, 3, false),   // 1 diag  (12 in diag) + 4 in remainder
                      std::make_tuple(21, 1, 3, 10, 5, 5, 3, 3, false),   // 3 diags (6 in diag)   + 3 in remainder
                      std::make_tuple(27, 1, 4, 12, 5, 5, 3, 3, false))   // 5 diags (5 in diag)   + 2 in remainder
);

INSTANTIATE_TEST_SUITE_P(big_images_ASIC_CI,
                         SynTrainingGConvPackingGconvFwdConstantFoldingBFloat16Test,
                         ::testing::Values(std::make_tuple(8, 1, 16, 16, 512, 512, 3, 3, false)));

TEST_P_GC(SynTrainingGConvPackingGconvFwdConstantFoldingBFloat16Test,
          conv_group_test,
          {synDeviceGaudi2, synDeviceGaudi3})
{
    GCFG_ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING.setValue(true);
    runGroupedConvolution(Node::TYPE_CONVOLUTION);
}

TEST_P_GC(SynTrainingGConvPackingGconvFwdConstantFoldingBFloat16Test,
          dedx_group_test,
          {synDeviceGaudi2, synDeviceGaudi3})
{
    GCFG_ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING.setValue(true);
    runGroupedConvolution(Node::TYPE_DEDX);
}

INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingGconvFwdConstantFoldingFP8Test,
    ::testing::Values(std::make_tuple(32, 1, 8, 8, 64, 64, 3, 3, false),    // 4 diags
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, false),  // 4 diags, resnext sizes
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, true)    // 4 diags, resnext sizes, DS
                      ));

INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_L2,
    SynTrainingGConvPackingGconvFwdConstantFoldingFP8Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 16, 5, 5, 2, 2, false),            // 1 diag
                      std::make_tuple(12, 1, 16, 16, 5, 5, 3, 3, false),           // 3 diags
                      std::make_tuple((64 / 4) * 5, 4, 4, 4, 56, 56, 3, 3, true),  // 5 diags, resnext sizes
                      std::make_tuple(8, 4, 16, 16, 14, 14, 3, 3, true)            // 2 diags, resnext sizes
                      ));

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingGconvFwdConstantFoldingFP8Test,
    ::testing::Values(std::make_tuple(4 + 2, 1, 16, 16, 5, 5, 2, 2, false),    // 1 diag (4 in diag) + 2 in remainder
                      std::make_tuple(32 + 1, 4, 16, 16, 14, 14, 3, 3, true),  // 2 diags (4 in diag) + 1 in remainder
                      std::make_tuple(12 + 3, 1, 16, 16, 5, 5, 3, 3, false),   // 3 diags (4 in diag) + 3 in remainder
                      std::make_tuple(32 + 6, 1, 8, 8, 64, 64, 3, 3, false),   // 4 diags (8 in diag) + 6 in remainder
                      std::make_tuple(80 + 11, 4, 4, 4, 56, 56, 3, 3, true)    // 5 diags (16 in diag) + 11 in remainder
                      ));

INSTANTIATE_TEST_SUITE_P(gconv_no_remainder_unaligned_L2,
                         SynTrainingGConvPackingGconvFwdConstantFoldingFP8Test,  // c != k for all cases
                         ::testing::Values(std::make_tuple((64 / 10) * 1, 1, 16, 10, 5, 5, 3, 3, false),  // 1 diag
                                           std::make_tuple((64 / 10) * 1, 1, 7, 10, 5, 5, 3, 3, false),   // 1 diag
                                           std::make_tuple((64 / 12) * 2, 1, 10, 12, 5, 5, 3, 3, false),  // 2 diags
                                           std::make_tuple((64 / 6) * 2, 1, 10, 6, 5, 5, 3, 3, false),    // 2 diags
                                           std::make_tuple((64 / 11) * 3, 1, 3, 11, 5, 5, 3, 3, false),   // 3 diags
                                           std::make_tuple((64 / 6) * 4, 1, 4, 6, 5, 5, 3, 3, false))     // 4 diags
);

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_unaligned_ASIC_CI,
    SynTrainingGConvPackingGconvFwdConstantFoldingFP8Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 10, 5, 5, 3, 3, false),   // 1 diag - 4 in remainder (out of 6 in diag)
                      std::make_tuple(12, 1, 16, 6, 5, 5, 3, 3, false),   // 1 diags (10 in diag)  + 2 in remainder
                      std::make_tuple(15, 1, 16, 10, 5, 5, 3, 3, false),  // 2 diags (6 in diag)  + 3 in remainder
                      std::make_tuple(22, 1, 16, 6, 5, 5, 3, 3, false),   // 2 diags (10 in diag)   + 2 in remainder
                      std::make_tuple(16, 1, 16, 5, 5, 5, 3, 3, false),   // 1 diag  (12 in diag) + 4 in remainder
                      std::make_tuple(21, 1, 3, 10, 5, 5, 3, 3, false),   // 3 diags (6 in diag)   + 3 in remainder
                      std::make_tuple(27, 1, 4, 12, 5, 5, 3, 3, false))   // 5 diags (5 in diag)   + 2 in remainder
);

INSTANTIATE_TEST_SUITE_P(DISABLED_big_images_ASIC_CI,
                         SynTrainingGConvPackingGconvFwdConstantFoldingFP8Test,
                         ::testing::Values(std::make_tuple(8, 1, 16, 16, 512, 512, 3, 3, false)));

TEST_P_GC(SynTrainingGConvPackingGconvFwdConstantFoldingFP8Test, conv_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    GCFG_ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING.setValue(true);
    runGroupedConvolution(Node::TYPE_CONVOLUTION);
}

TEST_P_GC(SynTrainingGConvPackingGconvFwdConstantFoldingFP8Test, dedx_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    GCFG_ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING.setValue(true);
    runGroupedConvolution(Node::TYPE_DEDX);
}

INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingGconvFwdConstantFoldingHFP8Test,
    ::testing::Values(std::make_tuple(32, 1, 8, 8, 64, 64, 3, 3, false),    // 4 diags
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, false),  // 4 diags, resnext sizes
                      std::make_tuple(32, 4, 8, 8, 128, 128, 3, 3, true)    // 4 diags, resnext sizes, DS
                      ));

INSTANTIATE_TEST_SUITE_P(
    gconv_no_remainder_aligned_L2,
    SynTrainingGConvPackingGconvFwdConstantFoldingHFP8Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 16, 5, 5, 2, 2, false),            // 1 diag
                      std::make_tuple(12, 1, 16, 16, 5, 5, 3, 3, false),           // 3 diags
                      std::make_tuple((64 / 4) * 5, 4, 4, 4, 56, 56, 3, 3, true),  // 5 diags, resnext sizes
                      std::make_tuple(8, 4, 16, 16, 14, 14, 3, 3, true)            // 2 diags, resnext sizes
                      ));

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_aligned_ASIC_CI,
    SynTrainingGConvPackingGconvFwdConstantFoldingHFP8Test,
    ::testing::Values(std::make_tuple(4 + 2, 1, 16, 16, 5, 5, 2, 2, false),    // 1 diag (4 in diag) + 2 in remainder
                      std::make_tuple(32 + 1, 4, 16, 16, 14, 14, 3, 3, true),  // 2 diags (4 in diag) + 1 in remainder
                      std::make_tuple(12 + 3, 1, 16, 16, 5, 5, 3, 3, false),   // 3 diags (4 in diag) + 3 in remainder
                      std::make_tuple(32 + 6, 1, 8, 8, 64, 64, 3, 3, false),   // 4 diags (8 in diag) + 6 in remainder
                      std::make_tuple(80 + 11, 4, 4, 4, 56, 56, 3, 3, true)    // 5 diags (16 in diag) + 11 in remainder
                      ));

INSTANTIATE_TEST_SUITE_P(gconv_no_remainder_unaligned_L2,
                         SynTrainingGConvPackingGconvFwdConstantFoldingHFP8Test,  // c != k for all cases
                         ::testing::Values(std::make_tuple((64 / 10) * 1, 1, 16, 10, 5, 5, 3, 3, false),  // 1 diag
                                           std::make_tuple((64 / 10) * 1, 1, 7, 10, 5, 5, 3, 3, false),   // 1 diag
                                           std::make_tuple((64 / 12) * 2, 1, 10, 12, 5, 5, 3, 3, false),  // 2 diags
                                           std::make_tuple((64 / 6) * 2, 1, 10, 6, 5, 5, 3, 3, false),    // 2 diags
                                           std::make_tuple((64 / 11) * 3, 1, 3, 11, 5, 5, 3, 3, false),   // 3 diags
                                           std::make_tuple((64 / 6) * 4, 1, 4, 6, 5, 5, 3, 3, false))     // 4 diags
);

INSTANTIATE_TEST_SUITE_P(
    gconv_remainder_unaligned_ASIC_CI,
    SynTrainingGConvPackingGconvFwdConstantFoldingHFP8Test,
    ::testing::Values(std::make_tuple(4, 1, 16, 10, 5, 5, 3, 3, false),   // 1 diag - 4 in remainder (out of 6 in diag)
                      std::make_tuple(12, 1, 16, 6, 5, 5, 3, 3, false),   // 1 diags (10 in diag)  + 2 in remainder
                      std::make_tuple(15, 1, 16, 10, 5, 5, 3, 3, false),  // 2 diags (6 in diag)  + 3 in remainder
                      std::make_tuple(22, 1, 16, 6, 5, 5, 3, 3, false),   // 2 diags (10 in diag)   + 2 in remainder
                      std::make_tuple(16, 1, 16, 5, 5, 5, 3, 3, false),   // 1 diag  (12 in diag) + 4 in remainder
                      std::make_tuple(21, 1, 3, 10, 5, 5, 3, 3, false),   // 3 diags (6 in diag)   + 3 in remainder
                      std::make_tuple(27, 1, 4, 12, 5, 5, 3, 3, false))   // 5 diags (5 in diag)   + 2 in remainder
);

INSTANTIATE_TEST_SUITE_P(DISABLED_big_images_ASIC_CI,
                         SynTrainingGConvPackingGconvFwdConstantFoldingHFP8Test,
                         ::testing::Values(std::make_tuple(8, 1, 16, 16, 512, 512, 3, 3, false)));

TEST_P_GC(SynTrainingGConvPackingGconvFwdConstantFoldingHFP8Test, conv_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    GCFG_ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING.setValue(true);
    runGroupedConvolution(Node::TYPE_CONVOLUTION);
}

TEST_P_GC(SynTrainingGConvPackingGconvFwdConstantFoldingHFP8Test, dedx_group_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    GCFG_ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING.setValue(true);
    runGroupedConvolution(Node::TYPE_DEDX);
}
