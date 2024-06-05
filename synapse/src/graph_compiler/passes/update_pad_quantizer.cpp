#include "habana_pass.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "role_pattern.h"
#include "perf_lib_layer_params.h"
#include "quantizer.h"
#include "data_type_utils.h"

class PadMMERolePattern : public RolePattern
{
public:
    PadMMERolePattern()
        : RolePattern()
    {
        m_pattern = nullptr;
        m_checkGraphInput = false;
    }

    virtual ~PadMMERolePattern() {};

    void setPatternParams(GraphPtr pattern, bool checkGraphInput)
    {
        m_pattern         = pattern;
        m_checkGraphInput = checkGraphInput;
    }

    bool addMMERoleNode(TensorPtr& input, TensorPtr& output)
    {
        synConvolution3DParamsV2 convParams;
        TensorPtr                convW    = std::make_shared<Tensor>();
        NodePtr convNode = NodeFactory::createNode({input, convW, nullptr, nullptr},
                                                   {output}, &convParams, NodeFactory::convolutionNodeTypeName, "");
        if (convNode == nullptr)
        {
            LOG_ERR(QUANT, "PadMMERolePattern: error creating convolution node");
            return false;
        }
        return m_pattern->addNode(convNode);
    }

    bool addPadRoleNode(TensorPtr& input, TensorPtr& output)
    {
        ns_PadKernel::Params padParams;
        NodePtr padNode = NodeFactory::createGenericTPCNode({input}, {output}, &padParams, "pad", "");
        if (padNode == nullptr)
        {
            LOG_ERR(QUANT, "PadMMERolePattern: error creating pad node");
            return false;
        }

        return m_pattern->addNode(padNode);
    }

    static bool isMMENode(const NodePtr& node) { return node != nullptr && HabanaGraph::runsOnMME(node); }

    static bool isPadNode(const NodePtr& node)
    {
        if (node == nullptr) return false;
        TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
        if (tpcNode == nullptr) return false;
        return tpcNode->getGUIDWithoutDtype() == "pad";
    }

    virtual std::pair<int, int> numInputsRange(const pNode& n) const override
    {
        if (isMMENode(n))
        {
            return std::make_pair(2, 4);
        }
        return RolePattern::numInputsRange(n);
    }

    virtual bool rolesMatch(const HabanaGraph& g, const NodePtr& a, const NodePtr& b) const override
    {
        if (m_checkGraphInput)
        {
            return ((isMMENode(a) && isMMENode(b)) || (isPadNode(a) && isPadNode(b) && g.isInputTensor(a->getInput(0))));
        }

        return ((isMMENode(a) && isMMENode(b)) || (isPadNode(a) && isPadNode(b)));
    }

private:
    bool m_checkGraphInput;
};

void getMatches(HabanaGraph& g, NodesPatternVector& matches, PadMMERolePattern& pattern1, PadMMERolePattern& pattern2)
{
    pattern1.setPatternParams(std::make_shared<Graph>(), false);
    TensorPtr input1      = std::make_shared<Tensor>();
    TensorPtr mme1Output1 = std::make_shared<Tensor>();
    TensorPtr padOutput1  = std::make_shared<Tensor>();
    TensorPtr mme2Output1 = std::make_shared<Tensor>();

    pattern1.addMMERoleNode(input1, mme1Output1);
    pattern1.addPadRoleNode(mme1Output1, padOutput1);
    pattern1.addMMERoleNode(padOutput1, mme2Output1);

    pattern2.setPatternParams(std::make_shared<Graph>(), true);
    TensorPtr input2      = std::make_shared<Tensor>();
    TensorPtr padOutput2  = std::make_shared<Tensor>();
    TensorPtr mme2Output2 = std::make_shared<Tensor>();

    pattern2.addPadRoleNode(input2, padOutput2);
    pattern2.addMMERoleNode(padOutput2, mme2Output2);

    // the order of patterns is such that longer patterns will be matched first
    RolePatternVector patterns = {&pattern1, &pattern2};

    // match the graph for all possible patterns
    matches = matchMultiplePatternsWithSingleOutputNode(g, patterns);
}

bool updatePadQuantizer(HabanaGraph& g)
{
    if (!isInferenceQuantization(g))
    {
        LOG_DEBUG(QUANT, "Graph is not in inference mode, update pad quantizer won't run.");
        return true;
    }

    if (!GCFG_ENABLE_SYNAPSE_QUANTIZATION.value())
    {
        LOG_DEBUG(QUANT, "Quantization is disabled in synapse. Skip {} Pass", HLLOG_FUNC);
        return true;
    }

    // Update Pad nodes quantizer to backwardQuantizer if one of the following patterns appear:
    //       - Input->Pad->MME
    //       - MME->Pad->MME
    PadMMERolePattern pattern1;
    PadMMERolePattern pattern2;
    NodesPatternVector matches;
    getMatches(g, matches, pattern1, pattern2);

    for (const std::pair<NodeList, RolePattern*>& match : matches)
    {
        NodeList nodeList = match.first;
        auto iterPadNode = std::find_if(nodeList.begin(), nodeList.end(), PadMMERolePattern::isPadNode);
        HB_ASSERT(iterPadNode != nodeList.end(), "pad node not found");
        NodePtr padNode = *iterPadNode;

        auto iterMMENode = std::find_if(iterPadNode, nodeList.end(), PadMMERolePattern::isMMENode);
        HB_ASSERT(iterMMENode != nodeList.end(), "MME node not found");
        NodePtr mmeNode = *iterMMENode;

        TensorPtr padInput  = padNode->getInput(0);
        TensorPtr padOutput = padNode->getOutput(0);

        synDataType outputDtype = padOutput->getElementType();
        LOG_DEBUG(QUANT,
                  "update pad node {} input type according to output type {}",
                  padNode->getNodeName(),
                  getStringFromSynDataType(outputDtype));
        padInput->changeDefaultElementType(outputDtype, true);

        synDataType mmeNodePrecision = mmeNode->getNodePrecision();
        LOG_DEBUG(QUANT,
                  "update pad node {} precision to {}",
                  padNode->getNodeName(),
                  getStringFromSynDataType(mmeNodePrecision));
        padNode->setNodePrecision((mmeNodePrecision != syn_type_na ? mmeNodePrecision : syn_type_single));

        LOG_DEBUG(QUANT, "update quantizer of pad node {} to BackwardQuantizer", padNode->getNodeName());
        padNode->setQuantizer(std::make_shared<BackwardQuantizer>());
    }

    return true;
}
