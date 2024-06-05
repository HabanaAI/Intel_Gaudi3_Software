#include "allocators_utils.h"
#include "graph_editor.h"
#include "graph_optimizer_test.h"
#include "habana_graph.h"
#include "liveness_analysis.h"
#include "synapse_common_types.h"
#include "node_factory.h"
#include "test_utils.h"
#include "graph_factory.h"

class FlashAttentionTestInfra
{
public:
    FlashAttentionTestInfra(unsigned      D           = 0,
                            unsigned      N           = 0,
                            unsigned      H           = 0,
                            unsigned      B           = 0,
                            float         dropoutProb = 0,
                            synDeviceType deviceType  = synDeviceGaudi2)
    : m_D(D),
      m_N(N),
      m_B(B),
      m_H(H),
      m_dropoutProb(dropoutProb),
      m_graph(GraphFactory::createGraph(deviceType, CompilationMode::Graph)) {};

    void            buildGraph();
    NodePtr         createSdpa(bool isFwd);
    ns_Sdpa::Params setParams();
    TensorPtr       createStatsTensor(synDataType dtype, bool setPersistent = false);
    TensorPtr       createTensor(const std::string& name);
    bool            maxTensorsExceedDram();

    const unsigned               m_D;
    const uint64_t               m_N;
    const unsigned               m_B;
    const unsigned               m_H;
    const float                  m_dropoutProb;
    const std::vector<TSize>     m_tensorSizes      = {m_D, m_N, m_H, m_B};
    const std::vector<TSize>     m_statsTensorSizes = {1, m_N, m_H, m_B};
    unsigned                     m_nodeIdx          = 0;
    unsigned                     m_tensorIdx        = 0;
    unsigned                     m_tensorCount      = 0;
    std::unique_ptr<HabanaGraph> m_graph;

protected:
    NodePtr createSdpaFwd();
    NodePtr createSdpaBwd();
};

class FlashAttentionTest : public GraphOptimizerTest
{
public:
    FlashAttentionTest(bool sliceCguid = false, bool reshapeSoftmax = false, bool gcSlicing = true)
    : m_sliceInCguid(sliceCguid), m_reshapeSoftmax(reshapeSoftmax), m_gcSlicing(gcSlicing) {};
    FlashAttentionTest(unsigned      D,
                       unsigned      N,
                       unsigned      H,
                       unsigned      B,
                       float         dropoutProb,
                       bool          sliceCguid,
                       bool          reshapeSoftmax,
                       bool          gcSlicing,
                       synDeviceType deviceType)
    : m_testBuilder(D, N, H, B, dropoutProb, deviceType),
      m_sliceInCguid(sliceCguid),
      m_reshapeSoftmax(reshapeSoftmax),
      m_gcSlicing(gcSlicing) {};

protected:
    void             SetUp() override;
    void             TearDown() override;
    virtual bool     shouldSkip();
    NodePtr          createSdpa(bool isFwd);
    bool             runTest();
    bool             isSoftmax(const NodePtr& n);
    ns_Sdpa::Params  setParams() { return m_testBuilder.setParams(); };
    TensorPtr        createStatsTensor(synDataType dtype, bool setPersistent = false)
    {
        return m_testBuilder.createStatsTensor(dtype, setPersistent);
    };
    TensorPtr               createTensor(const std::string& name) { return m_testBuilder.createTensor(name); };
    FlashAttentionTestInfra m_testBuilder;
    const char*             m_sdpaSlicingPrevCfg = nullptr;
    const char*             m_sdpaReshapePrevCfg = nullptr;
    const char*             m_gcSlicingPrevCfg = nullptr;
    const bool              m_sliceInCguid;
    const bool              m_reshapeSoftmax;
    const bool              m_gcSlicing;
};

using ParamTuple = std::tuple<unsigned,        // D
                              unsigned,        // N
                              unsigned,        // H
                              unsigned,        // B
                              bool,            // triagular attention mask used
                              float,           // dropout prob
                              bool,            // slice on B&H on cguid
                              bool,            // add reshape for softmax
                              bool,            // enable gc slicing
                              synDeviceType>;  // device type

class FlashAttentionParametrizedTest
: public FlashAttentionTest
, public testing::WithParamInterface<ParamTuple>
{
public:
    FlashAttentionParametrizedTest()
    : FlashAttentionTest(std::get<0>(GetParam()),
                         std::get<1>(GetParam()),
                         std::get<2>(GetParam()),
                         std::get<3>(GetParam()),
                         std::get<5>(GetParam()),
                         std::get<6>(GetParam()),
                         std::get<7>(GetParam()),
                         std::get<8>(GetParam()),
                         std::get<9>(GetParam())) {};
    void checkMemEfficientSchedule();
    struct PrintToStringParamName
    {
        template<class ParamType>
        std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const
        {
            std::stringstream ss;
            const unsigned    D           = std::get<0>(info.param);
            const unsigned    T           = std::get<1>(info.param);
            const unsigned    H           = std::get<2>(info.param);
            const unsigned    B           = std::get<3>(info.param);
            bool              doEnabled   = std::get<5>(info.param);
            bool              bhSlice     = std::get<6>(info.param);
            bool              reshapeSmax = std::get<7>(info.param);
            bool              gcSlicing   = std::get<8>(info.param);
            synDeviceType     deviceType  = std::get<9>(info.param);
            ss << std::to_string(B) << "x" << std::to_string(H) << "x" << std::to_string(T) << "x" << std::to_string(D)
               << "__Dropout_" << std::to_string(doEnabled) << "__bhSlice_" << std::to_string(bhSlice)
               << "__reshapeSmax_" << std::to_string(reshapeSmax) << "__gcSlicing_" << std::to_string(gcSlicing)
               << "__deviceType__" << toString(deviceType);
            return ss.str();
        }
    };
protected:
    void SetUp() override;
};

class FlashAttentionFwdSchedulingTest : public FlashAttentionParametrizedTest
{
};

class FlashAttentionFwdPipeliningTest : public FlashAttentionParametrizedTest
{
public:
    void checkPipelining();
    void checkSlicedSoftmax(const HabanaGraph& g);
};

class FlashAttentionBwdSchedulingTest : public FlashAttentionParametrizedTest
{
};

class FlashAttentionFwdBwdTest : public FlashAttentionParametrizedTest
{
protected:
    NodePtr  createSdpa();
};

class FlashAttentionFwdNegTest : public FlashAttentionFwdSchedulingTest
{
    bool shouldSkip() override { return false; }
};
