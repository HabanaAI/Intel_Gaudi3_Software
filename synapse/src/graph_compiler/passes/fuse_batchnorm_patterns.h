#pragma once
#include "graph.h"
#include "types.h"

// a mapping between node in pattern graph -> node in real graph
using PatternMatch = Graph::PatternMatch;

// general pattern fuser pass for BN2 nodes
class BnStage2PatternFuser
{
public:
    virtual ~BnStage2PatternFuser() = default;
    // return a temporary graph containing the searchable pattern
    virtual Graph getGraphPattern() const = 0;
    // given a pattern match map, return a list of nodes containing the fused pattern
    virtual NodeList fusePattern(const PatternMatch& match) const = 0;
    // check in the real graph if the given pattern match is valid for fusion
    virtual bool isValidPattern(const Graph& g, const PatternMatch& match) const;
    // get name
    virtual std::string_view getName() const = 0;

    enum BN2Dir
    {
        BN2_FWD,
        BN2_BWD
    };

    static std::string_view dir2Str(BN2Dir direction)
    {
        switch (direction)
        {
            case BN2_FWD:
                return "fwd";
            case BN2_BWD:
                return "bwd";
        }
        return "";
    }

protected:
    enum class BN2FwdInputs
    {
        IFM = 0,
        K,
        RUNNING_MEAN,
        RUNNING_VAR,
        MEAN_IN,
        GRAD_IN_1
    };

    enum class BN2FwdOutputs
    {
        OFM = 0,
        RUNNING_MEAN,
        RUNNING_VAR
    };

    enum class BN2BwdInputs
    {
        IFM = 0,
        MEAN_IN,
        OFM,
        GRAD_IN_1,
        GRAD_IN_2
    };

    enum class BN2BwdOutputs
    {
        SUM_DOT_P = 0,
        GRAD_OUT
    };

    enum class BN2Flavors
    {
        BN2_BWD,
        BN2_RELU_BWD,
        BN2_ADD_RELU_BWD,
        BN2_FWD,
        BN2_RELU_FWD,
        BN2_ADD_RELU_FWD
    };

    static std::string_view flavor2Str(BN2Flavors flavor)
    {
        switch (flavor)
        {
            case BN2Flavors::BN2_BWD:
                return "batch_norm_stage2_bwd";
            case BN2Flavors::BN2_RELU_BWD:
                return "batch_norm_stage2_relu_bwd";
            case BN2Flavors::BN2_ADD_RELU_BWD:
                return "batch_norm_stage2_add_relu_bwd";
            case BN2Flavors::BN2_FWD:
                return "batch_norm_stage2_fwd";
            case BN2Flavors::BN2_RELU_FWD:
                return "batch_norm_stage2_relu_fwd";
            case BN2Flavors::BN2_ADD_RELU_FWD:
                return "batch_norm_stage2_add_relu_fwd";
        }
        return "";
    }

    static std::string_view getDTypeStr() { return "bf16"; }

    explicit BnStage2PatternFuser(BN2Dir direction);

    // create a BN2 node
    static NodePtr createBN2Node(const TensorVector& inputs,
                                 const TensorVector& outputs,
                                 const NodePtr&      originalNode,
                                 BN2Flavors          flavor);
    // create temporary nodes for pattern graph utility functions
    static NodePtr createPatternBN2Node(BN2Dir direction);
    static NodePtr createPatternReLU(BN2Dir direction);
    static NodePtr createPatternAdd(BN2Dir direction);
    static NodePtr createPatternReshape(Node::eNodeType type);
    // node with same shape in inputs, or supported broadcast (1d broadcast)
    static bool isSupportedBroadcastedNode(const NodePtr& n);

    NodePtr m_bn;  // batch norm node in pattern graph
private:
    std::unordered_set<TensorPtr>                            getExternalTensors(const PatternMatch& match) const;
    std::tuple<bool, std::unordered_set<NodePtr>, TensorSet> getNodesBeforeFusion(const Graph&        g,
                                                                                  const PatternMatch& match) const;
};

using BnStage2PatternFuserPtr = std::shared_ptr<BnStage2PatternFuser>;

std::vector<BnStage2PatternFuserPtr> getAllBNFuserPatterns();
