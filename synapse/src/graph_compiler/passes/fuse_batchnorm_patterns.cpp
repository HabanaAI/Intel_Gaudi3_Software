#include "fuse_batchnorm_patterns.h"
#include "fcd_ops_utils.h"
#include "perf_lib_layer_params.h"
#include "node_factory.h"

BnStage2PatternFuser::BnStage2PatternFuser(BN2Dir direction)
{
    m_bn = createPatternBN2Node(direction);
}

// find all external tensors (tensors that appear after the fusion)
std::unordered_set<TensorPtr> BnStage2PatternFuser::getExternalTensors(const PatternMatch& match) const
{
    const auto                    fusedNodes = fusePattern(match);
    std::unordered_set<TensorPtr> externalTensors;
    for (const NodePtr& n : fusedNodes)
    {
        externalTensors.insert(n->getInputs().begin(), n->getInputs().end());
        externalTensors.insert(n->getOutputs().begin(), n->getOutputs().end());
    }
    return externalTensors;
}

std::tuple<bool, std::unordered_set<NodePtr>, TensorSet>
BnStage2PatternFuser::getNodesBeforeFusion(const Graph& g, const PatternMatch& match) const
{
    // collect all nodes and tensors before fusion
    std::unordered_set<NodePtr> preFusionNodes(match.size());
    TensorSet                   preFusionTensors;
    for (auto it : match)
    {
        const NodePtr& n = it.second;
        if (!g.containsNode(n))
        {
            LOG_TRACE(FUSE_BATCH_NORM, "{}: invalid. node {} no longer exists in graph", HLLOG_FUNC, n->getNodeName());
            return std::make_tuple(false, std::unordered_set<NodePtr>(), TensorSet());
        }
        preFusionNodes.insert(n);
        preFusionTensors.insert(n->getInputs().begin(), n->getInputs().end());
        preFusionTensors.insert(n->getOutputs().begin(), n->getOutputs().end());
    }
    return std::make_tuple(true, preFusionNodes, preFusionTensors);
}

// check for external consumers of internal tensors.
// if an internal (fused) tensor is consumed outside of the fusion pattern, the fusion is invalid
bool BnStage2PatternFuser::isValidPattern(const Graph& g, const PatternMatch& match) const
{
    // get the fused nodes from the pattern
    LOG_TRACE(FUSE_BATCH_NORM,
              "{}: checking pattern match {} on node: {}",
              HLLOG_FUNC,
              getName(),
              match.at(m_bn)->getNodeName());

    const auto externalTensors                            = getExternalTensors(match);
    const auto [result, preFusionNodes, preFusionTensors] = getNodesBeforeFusion(g, match);
    if (!result) return result;
    // for every tensor before the fusion, check if it is an internal tensor and consumed outside
    for (const TensorPtr& t : preFusionTensors)
    {
        if (!t) continue;
        if (externalTensors.find(t) != externalTensors.end()) continue;  // external tensor, can be used outside
        if (t->isUserManagedDram())
        {
            LOG_TRACE(FUSE_BATCH_NORM, "{}: invalid. internal tensor {} is persistent", HLLOG_FUNC, t->getName());
            return false;  // used in different graph
        }
        for (const NodePtr& n : g.getTensorConsumers(t))
        {
            if (preFusionNodes.find(n) == preFusionNodes.end())
            {
                LOG_TRACE(FUSE_BATCH_NORM,
                          "{}: invalid. internal tensor {} is consumed outside pattern",
                          HLLOG_FUNC,
                          t->getName());
                return false;  // a consumer outside the fusion exists
            }
        }
    }
    return true;
}

NodePtr BnStage2PatternFuser::createBN2Node(const TensorVector& inputs,
                                            const TensorVector& outputs,
                                            const NodePtr&      originalNode,
                                            BN2Flavors          flavor)
{
    auto             originalBn = std::dynamic_pointer_cast<TPCNode>(originalNode);
    std::string_view dtype      = originalBn ? originalBn->getDtypeFromGUID() : getDTypeStr();
    std::string      baseName   = originalBn ? originalBn->getNodeName() : "BN2_pattern";
    NodePtr          newBn      = NodeFactory::createNode(inputs,
                                            outputs,
                                            nullptr, /* Set below by calling to storeParamsInBuffer */
                                            fmt::format("{}_{}", flavor2Str(flavor), dtype),
                                            fmt::format("{}_fused", baseName));

    ns_BatchNormStage2Kernel::Params defaultParams;
    UserParams                       params     = originalBn ? originalBn->getParams() : &defaultParams;
    unsigned                         paramsSize = originalBn ? originalBn->getParamsSize() : sizeof(defaultParams);
    std::dynamic_pointer_cast<TPCNode>(newBn)->storeParamsInBuffer(params, paramsSize);
    return newBn;
}

NodePtr BnStage2PatternFuser::createPatternBN2Node(BN2Dir direction)
{
    unsigned     numInputs = direction == BN2_FWD ? 5 : 4;
    TensorVector inputs(numInputs);
    for (unsigned i = 0; i < numInputs; i++)
    {
        inputs[i] = std::make_shared<Tensor>();
    }
    TensorVector outputs = {std::make_shared<Tensor>()};
    return createBN2Node(inputs, outputs, nullptr, direction == BN2_FWD ? BN2Flavors::BN2_FWD : BN2Flavors::BN2_BWD);
}

NodePtr BnStage2PatternFuser::createPatternReLU(BN2Dir direction)
{
    TensorPtr    out    = std::make_shared<Tensor>();
    TensorVector inputs = {std::make_shared<Tensor>()};
    if (direction == BN2_BWD)
    {
        inputs.push_back(std::make_shared<Tensor>());
    }
    std::string reluGuid = fmt::format("relu_{}_{}", dir2Str(direction), getDTypeStr());
    return NodeFactory::createGenericTPCNode(inputs, {out}, nullptr, reluGuid, "PatternReLU");
}

NodePtr BnStage2PatternFuser::createPatternReshape(Node::eNodeType type)
{
    const char* guid = nullptr;
    switch (type)
    {
        case Node::TYPE_INTERNAL_RESHAPE:
            guid = NodeFactory::reshapeNodeTypeName;
            break;
        case Node::TYPE_STATIC_RESHAPE:
            guid = NodeFactory::staticReshapeNodeTypeName;
            break;
        default:
            break;
    };
    HB_ASSERT(guid != nullptr, "{}: unsupported reshape type {}", __FUNCTION__, type);
    TensorPtr out = std::make_shared<Tensor>();
    TensorPtr in  = std::make_shared<Tensor>();
    return NodeFactory::createNode({in}, {out}, nullptr, guid, "PatternReshape");
}

NodePtr BnStage2PatternFuser::createPatternAdd(BN2Dir direction)
{
    TensorVector inputs      = {std::make_shared<Tensor>(), std::make_shared<Tensor>()};
    TensorVector outputs     = {std::make_shared<Tensor>()};
    std::string  addNodeGuid = fmt::format("add_{}_{}", dir2Str(direction), getDTypeStr());
    std::string  addNodeName = fmt::format("PatternAdd_{}", dir2Str(direction), getDTypeStr());
    if (direction == BN2_BWD)
    {
        std::swap(inputs, outputs);
    }
    return NodeFactory::createGenericTPCNode(inputs, outputs, nullptr, addNodeGuid, addNodeName);
}

bool BnStage2PatternFuser::isSupportedBroadcastedNode(const NodePtr& n)
{
    const TensorPtr& out = n->getOutput(0);
    HB_ASSERT_PTR(out);
    for (const TensorPtr& in : n->getInputs())
    {
        if (!in->compareGeometry(*out) && in->getDim() != 1) return false;
    }
    return true;  // node with same shape in inputs, or supported broadcast (1d broadcast)
}

/*
  (0-4) +----------+(0)  (0)+------+(0)
 +----->+ BN2_FWD  +------->+ ReLU +------->
    In  +----------+   t1   +------+   t2

turns into:

  (0-4) +-----------+(0)
 +----->+bn_relu_fwd+------->
    In  +-----------+   t2

*/
class BnStage2FwdReluPatternFuser : public BnStage2PatternFuser
{
private:
    using BaseClass = BnStage2PatternFuser;

public:
    BnStage2FwdReluPatternFuser();
    Graph            getGraphPattern() const override;
    NodeList         fusePattern(const PatternMatch& match) const override;
    std::string_view getName() const override { return "BN2_FWD_RELU"; }

protected:
    NodePtr m_relu;
};

/*
   (0+4) +----------+(0)  (0)+-------+ (0)       (0)+------+(0)
  +----->+ BN2_FWD  +------->+Reshape+--------------> ReLU +------->
     In  +----------+   t1   +-------+      t2      +------+   t3

 turns into:

   (0+4) +-----------+(0)     (0)+-------+ (0)
  +----->+bn_relu_fwd+---------->+Reshape+---->
     In  +-----------+           +-------+   t3

*/
class BnStage2FwdReshapeReluPatternFuser : public BnStage2FwdReluPatternFuser
{
    using BaseClass = BnStage2FwdReluPatternFuser;

public:
    BnStage2FwdReshapeReluPatternFuser(Node::eNodeType type);
    std::string_view getName() const override { return "BN2_FWD_RESHAPE_RELU"; }
    Graph            getGraphPattern() const override;
    NodeList         fusePattern(const PatternMatch& match) const override;

protected:
    NodePtr m_reshape;
};

/*
                                             +
                                           t2|(1-idx)
                                             |
                                             v
  (0-4) +----------+(0)            (idx) +---+----+(0) (0)+------+(0)
 +----->+ BN2_FWD  +--------------------->+add_fwd+-------> ReLU +------->
    In  +----------+   t1                 +-------+  t3   +------+   t4

turns into:

  (0-4) +---------------+(0)
 +----->+bn_add_relu_fwd+---------->
    In  +---------------+   t4
               ^(5)
 +-------------+
       t2
*/
class BnStage2FwdAddReluPatternFuser : public BnStage2PatternFuser
{
private:
    using BaseClass = BnStage2PatternFuser;

public:
    // addInputIdx - the input index of the add node where BN2Fwd connects to.
    explicit BnStage2FwdAddReluPatternFuser(unsigned idx);
    Graph            getGraphPattern() const override;
    NodeList         fusePattern(const PatternMatch& match) const override;
    std::string_view getName() const override { return "BN2_FWD_ADD_RELU"; }

protected:
    NodePtr        m_relu;
    NodePtr        m_add;
    const unsigned m_idx;
};

/*
                                                         +
                                                       t2|(1+idx)
                                                         |
                                                         v
  (0+4) +----------+(0)    +-------+(0)        (idx) +---+----+(0) (0)+------+(0)
 +----->+ BN2_FWD  +-------+Reshape+---------------->+ add_fwd+-------> ReLU +------->
    In  +----------+   t1  +-------+   t5            +--------+  t3   +------+   t4

turns into:

  (0+4) +---------------+(0)  +-------+(0)
 +----->+bn_add_relu_fwd+---->+Reshape+---->
    In  +---------------+     +-------+ t4
                   (5)^
         +-------+    |
 +------>+Reshape+----+
    (t2) +-------+

*/
class BnStage2FwdReshapeAddReluPatternFuser : public BnStage2FwdAddReluPatternFuser
{
    using BaseClass = BnStage2FwdAddReluPatternFuser;

public:
    // addInputIdx - the input index of the add node where BN2Fwd connects to.
    BnStage2FwdReshapeAddReluPatternFuser(unsigned idx, Node::eNodeType type);
    Graph            getGraphPattern() const override;
    bool             isValidPattern(const Graph& g, const PatternMatch& match) const override;
    NodeList         fusePattern(const PatternMatch& match) const override;
    std::string_view getName() const override { return "BN2_FWD_RESHAPE_ADD_RELU"; }

protected:
    NodePtr m_reshape;
};

/*
                    (0-1)
 +----------+      +-------------+
    t0   (0)|         In         |
        +---v----+      (3)+-----v----+(0)
        |relu_bwd+---------> bn2_bwd  +------->
        +---^----+    t2   +----------+  t4
    t1   (1)|
 +----------+

turns into:

       t1
 +-------------+
            (2)|
   (0-1)+------v-----+              (0)  t4
+-------+bn2_relu_bwd+--------------------->
     In +------------+    |
            (3)^          `---------->
  +------------+            (1)   t2
       t0

*/
class BnStage2BwdReluPatternFuser : public BnStage2PatternFuser
{
private:
    using BaseClass = BnStage2PatternFuser;

public:
    BnStage2BwdReluPatternFuser();
    Graph            getGraphPattern() const override;
    NodeList         fusePattern(const PatternMatch& match) const override;
    std::string_view getName() const override { return "BN2_BWD_RELU"; }

protected:
    TensorVector getNewBnInputs(const PatternMatch& match) const;
    TensorVector getNewBnOutputs(const PatternMatch& match) const;
    NodePtr      m_relu;
};

/*
                                         (0-1)
    t5    (0)                           +-------------+
+-----------+                              In         |
        +---v---+    t0   (0)+--------+      (3)+-----v----+(0)
        |add_fwd+----------->+relu_bwd+---------> bn2_bwd  +------->
    t6  +---^---+            +---^----+    t2   +----------+  t4
+-----------+            t1   (1)|
          (1)         +----------+

turns into:

            t1
        +--------------+
                    (2)|
           (0-1)+----------------+              (0)  t4
        +-------+bn2_add_relu_bwd+--------------------->
             In +----------+-----+    |
                    (3)^   ^          `---------->
          +------------+   |            (1)   t2
              t5           |
                        (4)|
          +----------------+
              t6

*/
class BnStage2BwdReluAddPatternFuser : public BnStage2BwdReluPatternFuser
{
private:
    using BaseClass = BnStage2BwdReluPatternFuser;

public:
    BnStage2BwdReluAddPatternFuser();
    Graph            getGraphPattern() const override;
    NodeList         fusePattern(const PatternMatch& match) const override;
    std::string_view getName() const override { return "BN2_BWD_RELU_ADD"; }

protected:
    bool    isValidPattern(const Graph& g, const PatternMatch& match) const override;
    NodePtr m_addFwd;
};

/*
                                                     (0-1)
 +----------+                                       +-------------+
    t0   (0)|                                          In         |
        +---v----+(0)   (0)+-------+      (i)  t3        (3)+-----v----+(0)
        |relu_bwd+-------->+add_bwd+----+-------------------> bn2_bwd  +------->
        +---^----+  t2     +-------+    |                   +----------+  t5
    t1   (1)|                         t4|(1-i)
 +----------+                           v

turns into:

       t1
 +-------------+
            (2)|
   (0-1)+------------+        (0)  t5
+------>|bn2_relu_bwd|----------------->
     In +------------+  (1)|
            (3)^           |t3    +--------+ t4
  +------------+           +----->+identity+----->
       t0                         +--------+

*/
class BnStage2BwdAddReluPatternFuser : public BnStage2BwdReluPatternFuser
{
private:
    using BaseClass = BnStage2BwdReluPatternFuser;

public:
    explicit BnStage2BwdAddReluPatternFuser(unsigned idx);
    Graph            getGraphPattern() const override;
    NodeList         fusePattern(const PatternMatch& match) const override;
    std::string_view getName() const override { return "BN2_BWD_ADD_RELU"; }

protected:
    TensorVector getNewBnOutputs(const PatternMatch& match) const;
    NodePtr      createIdentityNode(const PatternMatch& match) const;
    NodePtr      m_add;
    unsigned     m_idx;
};

/*
                                                                        (0+1)
    t6    (0)                                                          +-------------+
+-----------+                                                             In         |
        +---v---+    t0   (0)+--------+    (0)+-------+      (i)  t3        (3)+-----v----+(0)
        |add_fwd+----------->+relu_bwd+------->add_bwd+----+-------------------> bn2_bwd  +------->
    t7  +---^---+            +---^----+  t2   +-------+    |                   +----------+  t5
+-----------+            t1   (1)|                       t4|(1-i)
          (1)         +----------+                         v

turns into:

                     t1
               +-------------+
                          (2)|
                 (0-1)+----------------+         (0)  t5
              +-------+bn2_add_relu_bwd+------------------->
                   In +----------+-----+   (1)|
                          (3)^   ^            |t3    +--------+ t4
                +------------+   |            +----->+identity+----->
                     t6          |                   +--------+
                              (4)|
               +-----------------+
                     t7

*/
class BnStage2BwdAddReluAddPatternFuser : public BnStage2BwdAddReluPatternFuser
{
private:
    using BaseClass = BnStage2BwdAddReluPatternFuser;

public:
    explicit BnStage2BwdAddReluAddPatternFuser(unsigned idx);
    Graph            getGraphPattern() const override;
    NodeList         fusePattern(const PatternMatch& match) const override;
    std::string_view getName() const override { return "BN2_BWD_ADD_RELU_ADD"; }

protected:
    bool    isValidPattern(const Graph& g, const PatternMatch& match) const override;
    NodePtr m_addFwd;
};

// ============================ Pattern Fusers Implementations ============================

BnStage2FwdReluPatternFuser::BnStage2FwdReluPatternFuser() : BaseClass(BN2_FWD)
{
    m_relu = createPatternReLU(BN2_FWD);
    m_relu->replaceInput(0, m_bn->getOutput(0));
};

Graph BnStage2FwdReluPatternFuser::getGraphPattern() const
{
    Graph g;
    g.addNode(m_bn);
    g.addNode(m_relu);
    return g;
}

NodeList BnStage2FwdReluPatternFuser::fusePattern(const PatternMatch& match) const
{
    const NodePtr& bn2Node                    = match.at(m_bn);
    TensorVector   fusedNodeInputs            = bn2Node->getInputs();
    TensorVector   fusedNodeOutputs           = bn2Node->getOutputs();
    fusedNodeOutputs[(int)BN2FwdOutputs::OFM] = match.at(m_relu)->getOutput(0);
    NodePtr stage2NodeRelu = createBN2Node(fusedNodeInputs, fusedNodeOutputs, bn2Node, BN2Flavors::BN2_RELU_FWD);
    return {stage2NodeRelu};
}

BnStage2FwdReshapeReluPatternFuser::BnStage2FwdReshapeReluPatternFuser(Node::eNodeType type) : BaseClass()
{
    m_reshape = createPatternReshape(type);
    m_reshape->replaceInput(0, m_bn->getOutput(0));
    m_relu->replaceInput(0, m_reshape->getOutput(0));
};

Graph BnStage2FwdReshapeReluPatternFuser::getGraphPattern() const
{
    Graph g = BaseClass::getGraphPattern();
    g.addNode(m_reshape);
    return g;
}

NodeList BnStage2FwdReshapeReluPatternFuser::fusePattern(const PatternMatch& match) const
{
    const NodePtr& reshapeNode        = match.at(m_reshape);
    const NodePtr& bn2Node            = match.at(m_bn);
    TensorVector   fusedNodeInputs    = bn2Node->getInputs();
    TensorVector   fusedNodeOutputs   = bn2Node->getOutputs();
    TensorPtr      newFusedNodeOutput = bn2Node->getOutput((int)BN2FwdOutputs::OFM)->clone(false, false, false);

    const NodePtr& newReshapeNode = reshapeNode->clone();
    newReshapeNode->replaceOutput(0, match.at(m_relu)->getOutput(0));
    newReshapeNode->replaceInput(0, newFusedNodeOutput);

    fusedNodeOutputs[(int)BN2FwdOutputs::OFM] = newFusedNodeOutput;
    NodePtr stage2NodeRelu = createBN2Node(fusedNodeInputs, fusedNodeOutputs, bn2Node, BN2Flavors::BN2_RELU_FWD);
    return {stage2NodeRelu, newReshapeNode};
}

BnStage2FwdAddReluPatternFuser::BnStage2FwdAddReluPatternFuser(unsigned idx) : BaseClass(BN2_FWD), m_idx(idx)
{
    HB_ASSERT(m_idx == 0 || m_idx == 1, "{}: illegal index: {}", HLLOG_FUNC, m_idx);
    BN2Dir direction = BN2_FWD;
    m_add            = createPatternAdd(direction);
    m_add->replaceInput(m_idx, m_bn->getOutput(0));
    m_relu = createPatternReLU(BN2_FWD);
    m_relu->replaceInput(0, m_add->getOutput(0));
};

Graph BnStage2FwdAddReluPatternFuser::getGraphPattern() const
{
    Graph g;
    g.addNode(m_bn);
    g.addNode(m_add);
    g.addNode(m_relu);
    return g;
}

NodeList BnStage2FwdAddReluPatternFuser::fusePattern(const PatternMatch& match) const
{
    const NodePtr& bn2Node                    = match.at(m_bn);
    TensorVector   fusedNodeInputs            = bn2Node->getInputs();
    TensorVector   fusedNodeOutputs           = bn2Node->getOutputs();
    fusedNodeOutputs[(int)BN2FwdOutputs::OFM] = match.at(m_relu)->getOutput(0);
    fusedNodeInputs.push_back(match.at(m_add)->getInput(1 - m_idx));
    NodePtr stage2NodeRelu = createBN2Node(fusedNodeInputs, fusedNodeOutputs, bn2Node, BN2Flavors::BN2_ADD_RELU_FWD);
    return {stage2NodeRelu};
}

BnStage2FwdReshapeAddReluPatternFuser::BnStage2FwdReshapeAddReluPatternFuser(unsigned idx, Node::eNodeType type)
: BaseClass(idx)
{
    m_reshape = createPatternReshape(type);
    m_reshape->replaceInput(0, m_bn->getOutput(0));
    m_add->replaceInput(m_idx, m_reshape->getOutput(0));
};

Graph BnStage2FwdReshapeAddReluPatternFuser::getGraphPattern() const
{
    Graph g = BaseClass::getGraphPattern();
    g.addNode(m_reshape);
    return g;
}

bool BnStage2FwdReshapeAddReluPatternFuser::isValidPattern(const Graph& g, const PatternMatch& match) const
{
    const NodePtr& addNode = match.at(m_add);
    if (addNode->isDynamicShape())
    {
        LOG_DEBUG(FUSE_BATCH_NORM, "{}: invalid fusion, {} is dynamic", __FUNCTION__, addNode->getNodeName());
        return false;
    }
    const TensorPtr& out = addNode->getOutput(0);
    for (const TensorPtr& in : addNode->getInputs())
    {
        if (!in->compareGeometry(*out))
        {
            LOG_DEBUG(FUSE_BATCH_NORM, "{}: invalid fusion, {} is broadcasted", __FUNCTION__, addNode->getNodeName());
            return false;
        }
    }
    return BaseClass::isValidPattern(g, match);
}

NodeList BnStage2FwdReshapeAddReluPatternFuser::fusePattern(const PatternMatch& match) const
{
    NodeList       ret;
    const NodePtr& reshapeNode      = match.at(m_reshape);
    const NodePtr& bn2Node          = match.at(m_bn);
    TensorVector   fusedNodeInputs  = bn2Node->getInputs();
    TensorVector   fusedNodeOutputs = bn2Node->getOutputs();

    TensorPtr      newFusedNodeOutput = bn2Node->getOutput((int)BN2FwdOutputs::OFM)->clone(false, false, false);
    const NodePtr& newReshapeOutNode  = reshapeNode->clone();
    newReshapeOutNode->replaceOutput(0, match.at(m_relu)->getOutput(0));
    newReshapeOutNode->replaceInput(0, newFusedNodeOutput);
    ret.push_back(newReshapeOutNode);

    TensorPtr      newFusedNodeInput = bn2Node->getOutput((int)BN2FwdOutputs::OFM)->clone(false, false, false);
    const NodePtr& newReshapeInNode  = reshapeNode->clone();
    newReshapeInNode->replaceOutput(0, newFusedNodeInput);
    newReshapeInNode->replaceInput(0, match.at(m_add)->getInput(1 - m_idx));
    if (newReshapeInNode->getInput(1) != nullptr)  // need new shape tensor
    {
        const TensorPtr& ifm          = bn2Node->getInput((int)BN2BwdInputs::IFM);
        NodePtr          extractShape = FcdOpsUtils::createExtractShapeNode(ifm);
        newReshapeInNode->replaceInput(1, extractShape->getOutput(0));
        ret.push_back(extractShape);
    }
    ret.push_back(newReshapeInNode);

    fusedNodeOutputs[(int)BN2FwdOutputs::OFM] = newFusedNodeOutput;
    fusedNodeInputs.push_back(newFusedNodeInput);
    NodePtr stage2NodeRelu = createBN2Node(fusedNodeInputs, fusedNodeOutputs, bn2Node, BN2Flavors::BN2_ADD_RELU_FWD);
    ret.push_back(stage2NodeRelu);
    return ret;
}

BnStage2BwdReluPatternFuser::BnStage2BwdReluPatternFuser() : BaseClass(BN2_BWD)
{
    m_relu = createPatternReLU(BN2_BWD);
    m_relu->replaceOutput(0, m_bn->getInput((int)BN2BwdInputs::GRAD_IN_1));
};

Graph BnStage2BwdReluPatternFuser::getGraphPattern() const
{
    Graph g;
    g.addNode(m_bn);
    g.addNode(m_relu);
    return g;
}

TensorVector BnStage2BwdReluPatternFuser::getNewBnInputs(const PatternMatch& match) const
{
    const NodePtr& bn2Node  = match.at(m_bn);
    const NodePtr& reluNode = match.at(m_relu);
    TensorVector   fusedNodeInputs(4);
    fusedNodeInputs[(int)BN2BwdInputs::IFM]       = bn2Node->getInput((int)BN2BwdInputs::IFM);
    fusedNodeInputs[(int)BN2BwdInputs::MEAN_IN]   = bn2Node->getInput((int)BN2BwdInputs::MEAN_IN);
    fusedNodeInputs[(int)BN2BwdInputs::OFM]       = reluNode->getInput(1);
    fusedNodeInputs[(int)BN2BwdInputs::GRAD_IN_1] = reluNode->getInput(0);
    return fusedNodeInputs;
}

TensorVector BnStage2BwdReluPatternFuser::getNewBnOutputs(const PatternMatch& match) const
{
    TensorVector outputs(2);
    outputs[(int)BN2BwdOutputs::SUM_DOT_P] = match.at(m_bn)->getOutput(0);
    outputs[(int)BN2BwdOutputs::GRAD_OUT]  = match.at(m_relu)->getOutput(0);
    return outputs;
}

NodeList BnStage2BwdReluPatternFuser::fusePattern(const PatternMatch& match) const
{
    return {createBN2Node(getNewBnInputs(match), getNewBnOutputs(match), match.at(m_bn), BN2Flavors::BN2_RELU_BWD)};
}

bool BnStage2BwdReluAddPatternFuser::isValidPattern(const Graph& g, const PatternMatch& match) const
{
    return BaseClass::isValidPattern(g, match) && isSupportedBroadcastedNode(match.at(m_addFwd));
}

BnStage2BwdReluAddPatternFuser::BnStage2BwdReluAddPatternFuser() : BaseClass()
{
    m_addFwd = createPatternAdd(BN2_FWD);
    m_addFwd->replaceOutput(0, m_relu->getInput(0));
};
Graph BnStage2BwdReluAddPatternFuser::getGraphPattern() const
{
    Graph g = BnStage2BwdReluPatternFuser::getGraphPattern();
    g.addNode(m_addFwd);
    return g;
}

NodeList BnStage2BwdReluAddPatternFuser::fusePattern(const PatternMatch& match) const
{
    const NodePtr& addFwd                = match.at(m_addFwd);
    TensorVector   inputs                = getNewBnInputs(match);
    inputs[(int)BN2BwdInputs::GRAD_IN_1] = addFwd->getInput(0);
    HB_ASSERT(inputs.size() == (int)BN2BwdInputs::GRAD_IN_2, "{}: unexpected num inputs", HLLOG_FUNC);
    inputs.push_back(addFwd->getInput(1));
    return {createBN2Node(inputs, getNewBnOutputs(match), match.at(m_bn), BN2Flavors::BN2_ADD_RELU_BWD)};
}

BnStage2BwdAddReluPatternFuser::BnStage2BwdAddReluPatternFuser(unsigned idx) : BaseClass(), m_idx(idx)
{
    HB_ASSERT(m_idx == 0 || m_idx == 1, "{}: illegal index: {}", HLLOG_FUNC, m_idx);
    m_add = createPatternAdd(BN2_BWD);
    m_add->replaceOutput(m_idx, m_bn->getInput((int)BN2BwdInputs::GRAD_IN_1));
    m_relu->replaceOutput(0, m_add->getInput(0));
    // removing the other output so that the pattern will have a single graph output
    m_add->replaceOutput(1 - m_idx, nullptr);
};

Graph BnStage2BwdAddReluPatternFuser::getGraphPattern() const
{
    Graph g = BnStage2BwdReluPatternFuser::getGraphPattern();
    g.addNode(m_add);
    return g;
}

TensorVector BnStage2BwdAddReluPatternFuser::getNewBnOutputs(const PatternMatch& match) const
{
    const NodePtr& bn2Node          = match.at(m_bn);
    const NodePtr& addNode          = match.at(m_add);
    TensorVector   fusedNodeOutputs = bn2Node->getOutputs();
    fusedNodeOutputs.push_back(addNode->getOutput(m_idx));
    return fusedNodeOutputs;
}

NodePtr BnStage2BwdAddReluPatternFuser::createIdentityNode(const PatternMatch& match) const
{
    const NodePtr&   addNode        = match.at(m_add);
    const TensorPtr& innerAddOutput = addNode->getOutput(m_idx);
    const TensorPtr& outerAddOutput = addNode->getOutput(1 - m_idx);
    return NodeFactory::createNode({innerAddOutput},
                                   {outerAddOutput},
                                   nullptr,
                                   NodeFactory::identityNodeTypeName,
                                   fmt::format("{}_planted_identity_", outerAddOutput->getName()));
}

NodeList BnStage2BwdAddReluPatternFuser::fusePattern(const PatternMatch& match) const
{
    const NodePtr& bn2Node  = match.at(m_bn);
    const NodePtr  identity = createIdentityNode(match);
    const NodePtr  newBn =
        createBN2Node(getNewBnInputs(match), getNewBnOutputs(match), bn2Node, BN2Flavors::BN2_RELU_BWD);
    return {newBn, identity};
}

bool BnStage2BwdAddReluAddPatternFuser::isValidPattern(const Graph& g, const PatternMatch& match) const
{
    return BaseClass::isValidPattern(g, match) && isSupportedBroadcastedNode(match.at(m_addFwd));
}

BnStage2BwdAddReluAddPatternFuser::BnStage2BwdAddReluAddPatternFuser(unsigned idx) : BaseClass(idx)
{
    m_addFwd = createPatternAdd(BN2_FWD);
    m_addFwd->replaceOutput(0, m_relu->getInput(0));
};

Graph BnStage2BwdAddReluAddPatternFuser::getGraphPattern() const
{
    Graph g = BaseClass::getGraphPattern();
    g.addNode(m_addFwd);
    return g;
}

NodeList BnStage2BwdAddReluAddPatternFuser::fusePattern(const PatternMatch& match) const
{
    const NodePtr& bn2Node               = match.at(m_bn);
    const NodePtr& addFwd                = match.at(m_addFwd);
    TensorVector   inputs                = getNewBnInputs(match);
    inputs[(int)BN2BwdInputs::GRAD_IN_1] = addFwd->getInput(0);
    HB_ASSERT(inputs.size() == (int)BN2BwdInputs::GRAD_IN_2, "{}: unexpected num inputs", HLLOG_FUNC);
    inputs.push_back(addFwd->getInput(1));
    NodePtr newBn    = createBN2Node(inputs, getNewBnOutputs(match), bn2Node, BN2Flavors::BN2_ADD_RELU_BWD);
    NodePtr identity = createIdentityNode(match);
    return {identity, newBn};
}

std::vector<BnStage2PatternFuserPtr> getAllBNFuserPatterns()
{
    std::vector<BnStage2PatternFuserPtr> patterns;
    patterns.push_back(std::make_shared<BnStage2BwdReluAddPatternFuser>());
    patterns.push_back(std::make_shared<BnStage2BwdReluPatternFuser>());

    patterns.push_back(std::make_shared<BnStage2BwdAddReluAddPatternFuser>(0));
    patterns.push_back(std::make_shared<BnStage2BwdAddReluAddPatternFuser>(1));
    patterns.push_back(std::make_shared<BnStage2BwdAddReluPatternFuser>(0));
    patterns.push_back(std::make_shared<BnStage2BwdAddReluPatternFuser>(1));

    patterns.push_back(std::make_shared<BnStage2FwdReluPatternFuser>());
    patterns.push_back(std::make_shared<BnStage2FwdReshapeReluPatternFuser>(Node::TYPE_INTERNAL_RESHAPE));
    patterns.push_back(std::make_shared<BnStage2FwdReshapeReluPatternFuser>(Node::TYPE_STATIC_RESHAPE));

    patterns.push_back(std::make_shared<BnStage2FwdAddReluPatternFuser>(1));
    patterns.push_back(std::make_shared<BnStage2FwdReshapeAddReluPatternFuser>(1, Node::TYPE_INTERNAL_RESHAPE));
    patterns.push_back(std::make_shared<BnStage2FwdReshapeAddReluPatternFuser>(1, Node::TYPE_STATIC_RESHAPE));
    patterns.push_back(std::make_shared<BnStage2FwdAddReluPatternFuser>(0));
    patterns.push_back(std::make_shared<BnStage2FwdReshapeAddReluPatternFuser>(0, Node::TYPE_INTERNAL_RESHAPE));
    patterns.push_back(std::make_shared<BnStage2FwdReshapeAddReluPatternFuser>(0, Node::TYPE_STATIC_RESHAPE));
    return patterns;
}
