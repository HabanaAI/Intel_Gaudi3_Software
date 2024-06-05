#include "passes.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "tpc_node.h"
#include "perf_lib_layer_params.h"

#include "graph_editor.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>

constexpr std::string_view padGuid     = "pad";
constexpr std::string_view maxpoolGuid = "maxpool_2d";
constexpr std::string_view avgpoolGuid = "avg_pool_2d";

// todo: add support for 3d pooling?

static void getPadConvPatterns(HabanaGraph& g, NodeSet& matchingNodes)
{
    auto padConvPattern1 = std::make_shared<Graph>();
    bool status;
    {
        ns_PadKernel::Params padParams;

        pTensor padIFM = std::make_shared<Tensor>();
        pTensor padOFM = std::make_shared<Tensor>();
        pNode   padNode = NodeFactory::createGenericTPCNode({padIFM}, {padOFM}, &padParams, padGuid, "pad_before_conv");

        synConvolution3DParamsV2 convolutionParams;

        pTensor convOFM  = std::make_shared<Tensor>();
        pTensor convWGH  = std::make_shared<Tensor>();
        pNode   convNode = NodeFactory::createNode({padOFM, convWGH, nullptr, nullptr},
                                                 {convOFM},
                                                 &convolutionParams,
                                                 NodeFactory::convolutionNodeTypeName,
                                                 "conv_after_pad");

        status = padConvPattern1->addNode(padNode);
        status = status && padConvPattern1->addNode(convNode);
    }

    if (status)
    {
        NodeSet p1MatchNodes = g.matchPatternWithSingleOutputNode(padConvPattern1.get(), NodeTypeMatchingFunc);
        matchingNodes.insert(p1MatchNodes.begin(), p1MatchNodes.end());
    }
    else
    {
        LOG_DEBUG(GC, "Pattern1 build failed for Pad-Conv fusing.");
        matchingNodes = NodeSet();
    }

    auto padConvPattern2 = std::make_shared<Graph>();
    {
        ns_PadKernel::Params padParams;

        pTensor padIFM = std::make_shared<Tensor>();
        pTensor padOFM = std::make_shared<Tensor>();
        pNode   padNode = NodeFactory::createGenericTPCNode({padIFM}, {padOFM}, &padParams, padGuid, "pad_before_conv");

        synConvolution3DParamsV2 convolutionParams;

        pTensor convOFM  = std::make_shared<Tensor>();
        pTensor convWGH  = std::make_shared<Tensor>();
        pNode   convNode = NodeFactory::createNode({padOFM, convWGH, nullptr, nullptr, nullptr},
                                                 {convOFM},
                                                 &convolutionParams,
                                                 NodeFactory::convolutionNodeTypeName,
                                                 "conv_after_pad");

        status = padConvPattern2->addNode(padNode);
        status = status && padConvPattern2->addNode(convNode);
    }

    if (status)
    {
        NodeSet p2MatchNodes = g.matchPatternWithSingleOutputNode(padConvPattern2.get(), NodeTypeMatchingFunc);
        matchingNodes.insert(p2MatchNodes.begin(), p2MatchNodes.end());
    }
    else
    {
        LOG_DEBUG(GC, "Pattern2 build failed for Pad-Conv fusing.");
    }
}

static void getPadPoolPatterns(HabanaGraph& g, NodeSet& matchingNodes)
{
    auto padMaxPoolPattern = std::make_shared<Graph>();
    bool pattern1Status;
    {
        ns_PadKernel::Params padParams;

        pTensor padIFM = std::make_shared<Tensor>();
        pTensor padOFM = std::make_shared<Tensor>();
        pNode   padNode =
            NodeFactory::createGenericTPCNode({padIFM}, {padOFM}, &padParams, padGuid, "pad_before_maxpool");

        ns_SpatialReduction::Params maxpoolParams;

        pTensor poolOFM  = std::make_shared<Tensor>();
        pNode   poolNode =
            NodeFactory::createGenericTPCNode({padOFM}, {poolOFM}, &maxpoolParams, maxpoolGuid, "maxpool_after_pad");

        pattern1Status = padMaxPoolPattern->addNode(padNode);
        pattern1Status = pattern1Status && padMaxPoolPattern->addNode(poolNode);
    }

    auto padAvgPoolPattern = std::make_shared<Graph>();
    bool pattern2Status;
    {
        pTensor padIFM = std::make_shared<Tensor>();
        pTensor padOFM = std::make_shared<Tensor>();
        pNode   padNode = NodeFactory::createGenericTPCNode({padIFM}, {padOFM}, nullptr, padGuid, "pad_before_avgpool");

        pTensor poolOFM = std::make_shared<Tensor>();
        pNode   poolNode =
            NodeFactory::createGenericTPCNode({padOFM}, {poolOFM}, nullptr, avgpoolGuid, "avgpool_after_pad");

        pattern2Status = padAvgPoolPattern->addNode(padNode);
        pattern2Status = pattern2Status && padAvgPoolPattern->addNode(poolNode);
    }

    matchingNodes = NodeSet();
    if (pattern1Status)
    {
        NodeSet p1Match = g.matchPatternWithSingleOutputNode(padMaxPoolPattern.get(), NodeTypeMatchingFunc);
        matchingNodes.insert(p1Match.begin(), p1Match.end());
    }
    else
    {
        LOG_DEBUG(GC, "Pattern build failed for Pad-MaxPool fusing.");
    }
    if (pattern2Status)
    {
        NodeSet p2Match = g.matchPatternWithSingleOutputNode(padAvgPoolPattern.get(), NodeTypeMatchingFunc);
        matchingNodes.insert(p2Match.begin(), p2Match.end());
    }
    else
    {
        LOG_DEBUG(GC, "Pattern build failed for Pad-AvgPool fusing.");
    }
}

static bool checkPadAndGetParams(const HabanaGraph& g, pNode pad, ns_PadKernel::Params& padParams)
{
    std::shared_ptr<TPCNode> tpc = std::dynamic_pointer_cast<TPCNode>(pad);
    HB_ASSERT(tpc != nullptr && tpc->isGuidPrefix(padGuid), "wrong node in pad pattern");

    if (!GraphEditor::canEliminateTensor(g, pad->getOutput(TENSOR_OFM)))
    {
        // can't fuse if the pad's output tensor is output / has more than 1 consumer
        return false;
    }
    padParams = *(ns_PadKernel::Params*)tpc->getParams();

    return true;
}

static bool getWHPads(const pNode&                padConsumer,
                      const ns_PadKernel::Params& padParams,
                      unsigned&                   wPadBegin,
                      unsigned&                   wPadEnd,
                      unsigned&                   hPadBegin,
                      unsigned&                   hPadEnd)
{
    const unsigned endPadsOffset = 4;

    unsigned nIndex = padConsumer->inputDimNameToIndex(TENSOR_IFM, 'N');
    unsigned hIndex = padConsumer->inputDimNameToIndex(TENSOR_IFM, 'H');
    unsigned wIndex = padConsumer->inputDimNameToIndex(TENSOR_IFM, 'W');
    unsigned cIndex = padConsumer->inputDimNameToIndex(TENSOR_IFM, 'C');

    HB_ASSERT(nIndex < Tensor::c_tensorMaxDim - 1 && hIndex < Tensor::c_tensorMaxDim - 1 &&
                  wIndex < Tensor::c_tensorMaxDim - 1 && cIndex < Tensor::c_tensorMaxDim - 1,
              "wrong layout for conv");

    if (padParams.pads[nIndex] != 0 || padParams.pads[endPadsOffset + nIndex] != 0 || padParams.pads[cIndex] != 0 ||
        padParams.pads[endPadsOffset + cIndex] != 0)
    {
        // conv and pooling only pad the spatial dimensions (W and H)
        return false;
    }

    wPadBegin = padParams.pads[wIndex];
    wPadEnd   = padParams.pads[endPadsOffset + wIndex];
    hPadBegin = padParams.pads[hIndex];
    hPadEnd   = padParams.pads[endPadsOffset + hIndex];

    return true;
}

// TODO handle paddingType here <===== PADDING_TYPE
static bool updateConvPadding(const pNode& n,
                              unsigned     wPadBegin,
                              unsigned     wPadEnd,
                              unsigned     hPadBegin,
                              unsigned     hPadEnd,
                              float        padVal)
{
    // currently convolution doesn't support asymmetric padding
    if (wPadBegin != wPadEnd || hPadBegin != hPadEnd)
    {
        return false;
    }

    std::shared_ptr<ConvolutionNode> conv = std::dynamic_pointer_cast<ConvolutionNode>(n);
    HB_ASSERT(conv != nullptr, "wrong node in pad-conv pattern");
    synConvolution3DParamsV2& convParams = conv->getConvolutionParams();

    unsigned convWPadBegin = convParams.padding[CONV_PAD_LEFT];
    unsigned convWPadEnd   = convParams.padding[CONV_PAD_RIGHT];
    unsigned convHPadBegin = convParams.padding[CONV_PAD_TOP];
    unsigned convHPadEnd   = convParams.padding[CONV_PAD_BOTTOM];

    bool alreadyPadded = convWPadBegin || convWPadEnd || convHPadBegin || convHPadEnd;
    if (alreadyPadded && padVal)
    {
        // since conv node with padding has 0 pad value, can not fuse different pad values.
        return false;
    }

    // add the padding from the pad node to the padding already in the convolution node
    convParams.padding[CONV_PAD_LEFT]   = convWPadBegin + wPadBegin;
    convParams.padding[CONV_PAD_RIGHT]  = convWPadEnd + wPadEnd;
    convParams.padding[CONV_PAD_TOP]    = convHPadBegin + hPadBegin;
    convParams.padding[CONV_PAD_BOTTOM] = convHPadEnd + hPadEnd;

    LOG_TRACE(GC,
              "updated conv padding in node {} to be padL={}, padR={}, padT={}, padB={}",
              n->getNodeName(),
              convParams.padding[CONV_PAD_LEFT],
              convParams.padding[CONV_PAD_RIGHT],
              convParams.padding[CONV_PAD_TOP],
              convParams.padding[CONV_PAD_BOTTOM]);

    n->setPaddingValue(n->getInput(TENSOR_IFM), padVal);

    return true;
}

static bool updatePoolPadding(const pNode& n,
                              unsigned     wPadBegin,
                              unsigned     wPadEnd,
                              unsigned     hPadBegin,
                              unsigned     hPadEnd,
                              float        padVal)
{
    std::shared_ptr<TPCNode> tpc = std::dynamic_pointer_cast<TPCNode>(n);
    HB_ASSERT_PTR(tpc);
    if (tpc->isGuidPrefix(maxpoolGuid))
    {
        if (padVal != -std::numeric_limits<float>::infinity() && padVal != -std::numeric_limits<int32_t>::infinity())
        {
            // max pool only supports padding with -infinity
            return false;
        }
    }
    else if (tpc->isGuidPrefix(avgpoolGuid))
    {
        if (padVal != 0)
        {
            // avg pool only supports padding with 0
            return false;
        }
    }
    else
    {
        HB_ASSERT(false, "wrong node in pad-pool pattern");
    }

    // the avg pool params struct inherits from the max pool params struct (ns_SpatialReduction)
    ns_SpatialReduction::Params* poolParams = (ns_SpatialReduction::Params*)tpc->getParams();

    // add the pad sizes from the pad node to the pad sizes already in the pooling node
    poolParams->pad_w_begin += wPadBegin;
    poolParams->pad_w_end += wPadEnd;
    poolParams->pad_h_begin += hPadBegin;
    poolParams->pad_h_end += hPadEnd;

    LOG_TRACE(GC,
              "updated pool padding in node {} to be pad_w_begin={}, pad_w_end={}, pad_h_begin={}, pad_h_end={}",
              n->getNodeName(),
              poolParams->pad_w_begin,
              poolParams->pad_w_end,
              poolParams->pad_h_begin,
              poolParams->pad_h_end);

    return true;
}

typedef bool (*updatePadding)(const pNode& n,
                              unsigned     wPadBegin,
                              unsigned     wPadEnd,
                              unsigned     hPadBegin,
                              unsigned     hPadEnd,
                              float        padVal);

static bool tryFusePad(HabanaGraph& g, pNode pad, updatePadding updatePaddingFunc)
{
    ns_PadKernel::Params padParams;
    memset(&padParams, 0, sizeof(ns_PadKernel::Params));
    unsigned wPadBegin = 0, wPadEnd = 0, hPadBegin = 0, hPadEnd = 0;

    if (!checkPadAndGetParams(g, pad, padParams))
    {
        return false;
    }

    NodeList consumers = g.getTensorConsumers(pad->getOutput(TENSOR_OFM));
    HB_ASSERT(consumers.size() == 1, "the number of consumers is {} but should be 1", consumers.size());
    pNode n = consumers.front();

    if (!getWHPads(n, padParams, wPadBegin, wPadEnd, hPadBegin, hPadEnd))
    {
        return false;
    }

    float    padVal = 0;
    TPCNodePtr tpcPad = std::dynamic_pointer_cast<TPCNode>(pad);
    padVal          = tpcPad->getDtypeFromGUID() == "i32" ? padParams.value.i : padParams.value.f;

    if (!updatePaddingFunc(n, wPadBegin, wPadEnd, hPadBegin, hPadEnd, padVal))
    {
        return false;
    }

    // after updating the padding in the node, remove the pad node and use its input as the new node input
    GraphEditor::removeNode(g, pad);
    GraphEditor::removeNode(g, n);
    n->replaceInput(TENSOR_IFM, pad->getInput(TENSOR_IFM));
    GraphEditor::addNode(g, n);

    LOG_INFO(GC, "Fused pad node {} into the subsequent node {}", pad->getNodeName(), n->getNodeName());

    return true;
}

bool fusePadIntoConvPool(HabanaGraph& g)
{
    return true; //TODO: [SW-137228]
    NodeSet padConvPatterns;
    getPadConvPatterns(g, padConvPatterns);

    for (const pNode& conv : padConvPatterns)
    {
        pTensor ifm = conv->getInput(TENSOR_IFM);
        pNode   pad = g.getTensorProducer(ifm);
        if (!tryFusePad(g, pad, &updateConvPadding))
        {
            LOG_TRACE(GC, "Couldn't fuse pad {} into conv {}", pad->getNodeName(), conv->getNodeName());
        }
    }

    NodeSet padPoolPatterns;
    getPadPoolPatterns(g, padPoolPatterns);

    for (const pNode& pool : padPoolPatterns)
    {
        pTensor ifm = pool->getInput(TENSOR_IFM);
        pNode   pad = g.getTensorProducer(ifm);
        if (!tryFusePad(g, pad, &updatePoolPadding))
        {
            LOG_TRACE(GC, "Couldn't fuse pad {} into pool {}", pad->getNodeName(), pool->getNodeName());
        }
    }

    return true;
}
