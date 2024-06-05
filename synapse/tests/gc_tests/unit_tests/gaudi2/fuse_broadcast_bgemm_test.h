#include "graph_optimizer_test.h"
#include "synapse_common_types.h"
#include "types.h"
#include "habana_graph.h"

using ParamTuple = std::tuple<SizeArray /*m_broadcastIn0Size*/,
                              SizeArray /*m_broadcastOutSize*/,
                              unsigned /*m_numOfTotalCastsNodes*/,
                              SizeArray /*m_reshapeSizes*/>;

class FuseBroadcastAndBGEMMTest : public GraphOptimizerTest
{
protected:
    void SetUp() override;
    void TearDown() override;
};

class FuseBroadcastAndBGEMMNegTest : public FuseBroadcastAndBGEMMTest
{
public:
    bool validateBroadcastWasntFused(const HabanaGraph& g, unsigned expectedNodesAmount);
};

class FuseBroadcastAndBGEMM
{
public:
    FuseBroadcastAndBGEMM(const SizeArray& broadcast0InSize,
                          const SizeArray& broadcastOutSize,
                          unsigned         numOfCasts,
                          const SizeArray& reshapeOutSizes)
    : m_broadcast0InSize(broadcast0InSize),
      m_broadcastOutSize(broadcastOutSize),
      m_numOfTotalCastsNodes(numOfCasts),
      m_reshapeOutSizes(reshapeOutSizes)
    {
    }

    FuseBroadcastAndBGEMM()
    : m_broadcast0InSize({0}),
      m_broadcastOutSize({0}),
      m_numOfTotalCastsNodes(0),
      m_reshapeOutSizes({0})
    {
    }

    virtual ~FuseBroadcastAndBGEMM() = default;

    void buildGraphWithBroadcastFusedPattern(HabanaGraph& g);

    bool checkThatInGraph(const HabanaGraph& g, const NodePtr& n);

    unsigned countLogicalReshapes(const HabanaGraph& g);

    NodePtr getBgemm()
    {
        return m_batchGemm;
    }

protected:
    TensorPtr createTensor(const std::string& name,
                           const SizeArray&   tensorSize,
                           const synDataType  dataType,
                           bool               isPersistent = false);
    void      setAsPersistent(TensorPtr& tensor, unsigned tensorsCount);
    NodePtr   createBroadcast(const SizeArray& inputSize, const SizeArray& outputSize, const synDataType dataType);
    NodePtr   createCast(const TensorPtr& castInput);
    NodePtr   createReshape(const TensorPtr& reshapeInput);

    SizeArray m_broadcast0InSize;
    SizeArray m_broadcastOutSize;
    unsigned  m_numOfTotalCastsNodes;
    SizeArray m_reshapeOutSizes;
    unsigned  m_castNodesCounter = 0;
    unsigned  m_tensorCount = 0;

    NodePtr m_broadcast;
    NodeVector m_casts;
    NodePtr m_reshape;
    NodePtr m_batchGemm;
};

class FuseBroadcastAndBGEMMParametrizedTest
: public FuseBroadcastAndBGEMM
, public GraphOptimizerTest
, public testing::WithParamInterface<ParamTuple>
{
public:
    FuseBroadcastAndBGEMMParametrizedTest()
    : FuseBroadcastAndBGEMM(std::get<0>(GetParam()),
                            std::get<1>(GetParam()),
                            std::get<2>(GetParam()),
                            std::get<3>(GetParam()))
    {
    }

    virtual ~FuseBroadcastAndBGEMMParametrizedTest() = default;

protected:
    void SetUp() override;
    void TearDown() override;
};