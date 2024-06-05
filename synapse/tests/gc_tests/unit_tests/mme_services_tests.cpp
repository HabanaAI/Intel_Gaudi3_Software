#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-param-util.h"
#include "graph_compiler/mme/mme_services.h"
#include "graph_optimizer_test.h"
#include "gaudi3_graph.h"
#include "node_factory.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "graph_compiler/mme/mme_brain_ifc.h"
#include "test_utils.h"

constexpr unsigned    rank  = 4;
constexpr synDataType dtype = syn_type_float;

class MmeServicesTest : public GraphOptimizerTest
{
public:
    MmeServicesTest() : m_graph(), m_setter(&m_graph) {}
    MMENodePtr createNode(const std::array<TSize, rank>& xSizes,
                          const std::array<TSize, rank>& wSizes,
                          const std::array<TSize, rank>& ySizes)
    {
        // create DEDW node with large CD
        TensorPtr            x = TensorPtr(new Tensor(rank, xSizes.data(), dtype));
        TensorPtr            w = TensorPtr(new Tensor(rank, wSizes.data(), dtype));
        TensorPtr            y = TensorPtr(new Tensor(rank, ySizes.data(), dtype));
        synConvolutionParams params;
        params.kW          = 3;
        auto       node    = NodeFactory::createNode({y, x}, {w}, &params, NodeFactory::deDwNodeTypeName, "dedw");
        MMENodePtr mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
        return mmeNode;
    }

protected:
    MmeCommon::MmeServices services;
    Gaudi3Graph m_graph;
    CompilationHalReaderSetter m_setter;
};

class AuxTensorTest
: public MmeServicesTest
, public testing::WithParamInterface<std::tuple<bool /*cd concurrency*/, bool /*deterministic*/>>
{
protected:
    MmeCommon::MmeStrategy generateStrategy(const MMENodePtr& mmeNode, bool cdConcurrency, bool deterministic)
    {
        auto mmeParams = mmeNode->getMmeBrainIfc()->getRecommendedMmeLayerParams();
        mmeParams.strategy.cdConcurrencyEn =
            cdConcurrency ? MmeCommon::BoolWithUndef::TurnedOn : MmeCommon::BoolWithUndef::TurnedOff;
        mmeParams.strategy.isDeterministic          = deterministic;
        return mmeParams.strategy;
    }

    void addAuxTensorsToNode(MMENodePtr& mmeNode, bool cdConcurrency, bool deterministic, unsigned perfDim, unsigned concurrencyLevel)
    {
        mmeNode->getNodeAnnotation().perforationDim = perfDim;
        auto strategy = generateStrategy(mmeNode, cdConcurrency, deterministic);
        auto pattern = services.matchPattern(mmeNode, strategy);
        ASSERT_EQ(pattern, MmeCommon::MmeServices::ePattern::CD_PARALLEL) << "expected CD_PARALLEL pattern";
        services.getAuxHandler().addAuxTensorsForCdParallel(mmeNode, concurrencyLevel);
    }

    MMENodePtr getNode()
    {
        const std::array<TSize, rank> xSizes = {512, 1024, 5, 5};
        const std::array<TSize, rank> wSizes = {1024, 512, 3, 1};
        const std::array<TSize, rank> ySizes = {1024, 1024, 5, 5};
        return createNode(xSizes, wSizes, ySizes);
    }

    MmeCommon::MmeServices::ePattern getPattern(MMENodePtr& mmeNode, unsigned perfDim)
    {
        bool concurrencyEn = std::get<0>(GetParam());
        bool deterministic = std::get<1>(GetParam());

        auto strategy = generateStrategy(mmeNode, concurrencyEn, deterministic);
        mmeNode->getNodeAnnotation().perforationDim = perfDim;
        return services.matchPattern(mmeNode, strategy);
    }
};

class AuxTensorPatternMatcherTest_no_match : public AuxTensorTest {};
class AuxTensorPatternMatcherTest_cd_parallel : public AuxTensorTest {};
class AuxTensorPatternMatcherTest_cd_concurrency : public AuxTensorTest {};

TEST_P(AuxTensorPatternMatcherTest_no_match, aux_tensor_pattern_matcher_non_added)
{
    setGlobalConfForTest(GCFG_ENABLE_CD_PARALLEL, "true");
    auto        mmeNode        = getNode();
    const auto& brainIfc       = mmeNode->getMmeBrainIfc();
    auto        indexSpaceRank = mmeNode->getNodeAccessPattern()->getNodeResolution().size();
    for (unsigned dim = 0; dim < indexSpaceRank; dim++)
    {
        if (!brainIfc->isCdDim(dim))
        {
            auto pattern = getPattern(mmeNode, dim);
            ASSERT_EQ(pattern, MmeCommon::MmeServices::ePattern::PATTERNS_NR)
                << "expected not to get a matching pattern";
        }
    }
}

TEST_P(AuxTensorPatternMatcherTest_cd_parallel, match_cd_parallel)
{
    setGlobalConfForTest(GCFG_ENABLE_CD_PARALLEL, "true");
    auto        mmeNode  = getNode();
    const auto& brainIfc = mmeNode->getMmeBrainIfc();
    for (unsigned dim : brainIfc->getCDDims())
    {
        auto pattern = getPattern(mmeNode, dim);
        ASSERT_EQ(pattern, MmeCommon::MmeServices::ePattern::CD_PARALLEL) << "expected to get a match on cd_parallel";
    }
}

TEST_P(AuxTensorPatternMatcherTest_cd_concurrency, match_cd_concurrency)
{
    setGlobalConfForTest(GCFG_ENABLE_CD_PARALLEL, "true");
    auto        mmeNode        = getNode();
    const auto& brainIfc       = mmeNode->getMmeBrainIfc();
    auto        indexSpaceRank = mmeNode->getNodeAccessPattern()->getNodeResolution().size();
    for (unsigned dim = 0; dim < indexSpaceRank; dim++)
    {
        if (!brainIfc->isCdDim(dim))
        {
            auto pattern = getPattern(mmeNode, dim);
            ASSERT_EQ(pattern, MmeCommon::MmeServices::ePattern::CD_CONCURRENCY)
                << "expected to get a match on cd_concurrency";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    no_match,
    AuxTensorPatternMatcherTest_no_match,
    ::testing::ValuesIn({
        std::tuple<bool, bool>
        // {true, false}, //TODO: uncomment once integration with non determinsitic cd concurrency [SW-174762] is done
        {false, true},
        {false, false},
    }));

INSTANTIATE_TEST_SUITE_P(CD_PARALLEL,
                         AuxTensorPatternMatcherTest_cd_parallel,
                         ::testing::ValuesIn({std::tuple<bool, bool> {false, false}}));

INSTANTIATE_TEST_SUITE_P(CD_CONCURRENCY,
                         AuxTensorPatternMatcherTest_cd_concurrency,
                         ::testing::ValuesIn({std::tuple<bool, bool> {true, true}}));

class AuxTensorTest_add_aux : public AuxTensorTest
{
public:
    void test(unsigned perfDim)
    {
        const std::array<TSize, rank> xSizes           = {512, 1024, 5, 5};
        const std::array<TSize, rank> wSizes           = {1024, 512, 3, 1};
        const std::array<TSize, rank> ySizes           = {1024, 1024, 5, 5};
        auto                          mmeNode          = createNode(xSizes, wSizes, ySizes);
        unsigned                      concurrencyLevel = CompilationHalReader::getHalReader()->getNumDcores();
        bool                          concurrencyEn    = std::get<0>(GetParam());
        bool                          deterministic    = std::get<1>(GetParam());
        addAuxTensorsToNode(mmeNode, concurrencyEn, deterministic, perfDim, concurrencyLevel);
        unsigned auxInputCount = 0;
        for (unsigned index = 0; index < mmeNode->getInputs().size(); index++)
        {
            auto& input = mmeNode->getInput(index);
            if (input && input->isAuxTensor())
            {
                ASSERT_TRUE(index == TENSOR_AUX_CD_SCRATCHPAD || index == TENSOR_AUX_CD_REDUCTION)
                    << "Aux index are wrong";
                if (index == TENSOR_AUX_CD_SCRATCHPAD)
                {
                    for (unsigned dim = 0; dim < rank; dim++)
                    {
                        ASSERT_EQ(input->getSizeInElements(dim), wSizes[dim]) << "scratchpad wrong size at dim" << dim;
                    }
                    ASSERT_EQ(input->getSizeInElements(rank), concurrencyLevel)
                        << "scratchpad - wrong size at last dim";
                }
                else if (index == TENSOR_AUX_CD_REDUCTION)
                {
                    ASSERT_EQ(input->getSizeInElements(0), concurrencyLevel);
                    ASSERT_EQ(input->getSizeInElements(1), 1);
                    ASSERT_EQ(input->getBufferSizeInBytes(), concurrencyLevel * getElementSizeInBytes(dtype));
                    float* dataBuffer = reinterpret_cast<float*>(input->getData());
                    for (unsigned idx = 0; idx < input->getTotalElements(); idx++)
                    {
                        ASSERT_EQ(dataBuffer[idx], 1.0f) << "wrong data in reduce aux tensor at idx : " << idx;
                    }
                }
                auxInputCount++;
            }
        }
        ASSERT_EQ(auxInputCount, 2) << "wrong number of aux inputs";
    }
};

INSTANTIATE_TEST_SUITE_P(
    AuxTensorTest,
    AuxTensorTest_add_aux,
    ::testing::ValuesIn({
        std::tuple<bool, bool> {false, false},
        // {true, false}, // TODO: uncomment once integration with non determinsitic cd concurrency [SW-174762] is done
    }));

TEST_P(AuxTensorTest_add_aux, add_aux_tensors_cd_parallel)
{
    setGlobalConfForTest(GCFG_ENABLE_CD_PARALLEL, "true");
    const std::array<TSize, rank> xSizes  = {512, 1024, 5, 5};
    const std::array<TSize, rank> wSizes  = {1024, 512, 3, 1};
    const std::array<TSize, rank> ySizes  = {1024, 1024, 5, 5};
    auto                          mmeNode = createNode(xSizes, wSizes, ySizes);
    for (unsigned commonDim : mmeNode->getMmeBrainIfc()->getCDDims())
    {
        test(commonDim);
    }
}