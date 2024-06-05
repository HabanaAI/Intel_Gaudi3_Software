#pragma once

#include <memory>
#include <gtest/gtest.h>
#include <math.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "habana_nodes.h"
#include "sim_graph.h"
#include "test_utils.h"
#include "transpose_permutation.h"
#include "transpose_nodes_creator.h"
#include <algorithm>
#include <iterator>
#include <perf_lib_layer_params.h>
#include "syn_logging.h"
#include "graph_optimizer_test.h"
#include "layout.h"
#include "quantization_data.h"

using namespace std;

static NodePtr createConvWithPadding(unsigned padH, unsigned padW,
                                     TensorPtr& OFM, unsigned& OFMsize, char*& ifm,
                                     char*& weights, synDataType type = syn_type_fixed)
{
    //Random ugly numbers below - but keep IFM at 64
    const TSize batch = 1;
    const TSize kW = 3;
    const TSize kH = 3;
    const TSize dW = 1;
    const TSize dH = 1;
    const TSize nOFM = 64;
    const TSize wOFM = 113;
    const TSize hOFM = 92;
    const TSize nIFM = 64;

    synConvolutionParams params;
    params.dH   = dH;
    params.dW   = dW;
    params.kH   = kH;
    params.kW   = kW;
    params.padT = padH;
    params.padB = padH;
    params.padL = padW;
    params.padR = padW;
    params.dilH = 1;
    params.dilW = 1;
    //o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * params.dW) + (params.kW - 1) * params.dilW + 1 - (params.padL + params.padR);
    const TSize hIFM = ((hOFM - 1) * params.dH) + (params.kH - 1) * params.dilH + 1 - (params.padT + params.padB);

    if (ifm == nullptr)
    {
        ifm = (char*)generateValuesArray(nIFM * wIFM * hIFM * batch, type, std::array<int, 2>({-2,2}));
    }
    if (weights == nullptr)
    {
        weights = (char*)generateValuesArray(nIFM * nOFM * params.kW * params.kH, type, std::array<int, 2>({-2,2}));
    }

    const TSize i_sizes[]        = { nIFM, wIFM, hIFM, batch };
    const TSize o_sizes[]        = { nOFM, wOFM, hOFM, batch };
    const TSize w_sizes[]        = { nOFM, nIFM, params.kW, params.kH };

    TensorPtr IFM = std::make_shared<Tensor>(4U, i_sizes, type, ifm);
    TensorPtr W   = std::make_shared<Tensor>(4U, w_sizes, type, weights);
    OFM         = std::make_shared<Tensor>(4U, o_sizes, syn_type_int32);

    OFMsize = nOFM * wOFM * hOFM * batch;

    NodePtr n = getConvNodeWithGoyaLayouts(IFM, W, nullptr, OFM, params, "");
    return n;
}

static NodePtr
createPadNode(uint32_t pads[8], NodePtr nextNode, char*& ifm,
              float padValue = 0, synDataType type = syn_type_fixed)
{
    TensorPtr padOutput = nextNode->getInput(TENSOR_IFM);
    TSize sizes[Tensor::c_tensorMaxDim] = {0};
    padOutput->getAllSizesInElements(sizes, Tensor::c_tensorMaxDim);
    unsigned inputSize = 1;
    for (int i = 0; i < 4; ++i)
    {
        sizes[i] = sizes[i] - pads[i] - pads[i + 4];
        inputSize *= sizes[i];
    }
    if (ifm == nullptr)
    {
        ifm     = new char[inputSize];
        std::generate(ifm, ifm + inputSize, Test_Random_Number_Creator (std::array<int, 2>({-2,2})));
    }
    TensorPtr padInput = std::make_shared<Tensor>(4U, sizes, type, ifm);

    const char* guid;
    switch (type)
    {
    case syn_type_int16:
    case syn_type_uint16:
        guid = "pad_i16";
        break;
    case syn_type_uint8:
    case syn_type_fixed:
    default:
        guid = "pad_i8";
        break;
    }

    ns_PadKernelEx::Params padParams;
    padParams.value.f = padValue;
    memcpy(padParams.pads, pads, sizeof(uint32_t) * 8);
    NodePtr  pad = NodeFactory::createNode({padInput}, {padOutput}, nullptr, guid, "pad_node");
    TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(pad);
    tpc->storeParamsInBuffer(&padParams, sizeof(ns_PadKernelEx::Params));
    return pad;
}

template <class GraphType>
class FUSE_PAD_CONV_PASSES : public GraphOptimizerTest
{
public:
    void fuse_pad_into_conv_symmetric()
    {
        GraphType g;
        /*
         * should be fused:
         * symmetric pad on the spatial dims (with pad value 0) followed by conv (that also does padding).
         * also compare the result to a reference where the padding is done entirely on the conv.
         */

        bool ret;

        unsigned totalHpad = 2;
        unsigned totalWpad = 2;

        SimGraph ref_g;

        TensorPtr OFMref;
        unsigned OFMrefSize;
        char *ifmRef = nullptr;
        char *weights = nullptr;
        NodePtr   convRef = createConvWithPadding(totalHpad, totalWpad, OFMref, OFMrefSize, ifmRef, weights);

        GraphEditor::addNode(ref_g, convRef);

        ret = ref_g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile reference graph";

        ret = ref_g.execute();
        ASSERT_EQ(ret, true) << "Failed to execute reference graph";


        g.setRecipeName("fuseSymmetricPad");

        TensorPtr OFM;
        unsigned OFMsize;
        char *ifm = nullptr;
        NodePtr   conv = createConvWithPadding(1, 1, OFM, OFMsize, ifm, weights); //conv with pads of size 1


        uint32_t pad_node_H = 1, pad_node_W = 1;
        uint32_t pads[8] = {0, pad_node_W, pad_node_H, 0, 0, pad_node_W, pad_node_H,
                            0}; //CWHN(begin)CWHN(end) - symmetric padding
        NodePtr  pad = createPadNode(pads, conv, ifmRef); // pads of size 1 - total of padding on conv+pad is size 2

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, conv);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph";

        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr)
            {
                ASSERT_FALSE(tpc->isGuidPrefix("pad")); // the pad wasn't fused into the conv
            }
            else
            {
                std::shared_ptr<ConvolutionNode> conv = std::dynamic_pointer_cast<ConvolutionNode>(n);
                if (conv != nullptr)
                {
                    synConvolution3DParams conv_params = conv->getConvolutionParams();
                    ASSERT_TRUE(conv_params.padding[CONV_PAD_TOP] == totalHpad &&
                                conv_params.padding[CONV_PAD_BOTTOM] == totalHpad &&
                                conv_params.padding[CONV_PAD_LEFT] == totalWpad &&
                                conv_params.padding[CONV_PAD_RIGHT] == totalWpad); // the pads were updated correctly
                }
            }

        }

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] weights;
        delete[] ifm;
        delete[] ifmRef;
    }

    void fuse_pad_into_conv_asymmetric()
    {
        /*
         * shouldn't be fused:
         * asymmetric pad on the spatial dims.
         */

        bool ret;

        GraphType g;
        g.setRecipeName("asymmetricPad");

        TensorPtr OFM;
        unsigned OFMsize;
        char *ifm = nullptr;
        char *weights = nullptr;
        NodePtr   conv = createConvWithPadding(1, 1, OFM, OFMsize, ifm, weights); //conv with pads of size 1

        uint32_t pads[8] = {0, 2, 0, 0, 0, 0, 2, 0}; //CWHN(begin)CWHN(end) - asymmetric padding
        char *padInput = nullptr;
        NodePtr  pad = createPadNode(pads, conv, padInput);

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, conv);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph with asymmetric pad";

        bool foundPad = false;
        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr && tpc->isGuidPrefix("pad"))
            {
                foundPad = true;
            }
        }
        ASSERT_TRUE(foundPad);

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] weights;
        delete[] ifm;
        delete[] padInput;
    }

    void fuse_non_zero_pad_into_padded_conv()
    {
        /*
         * shouldn't be fused:
         * pad value != 0 and convolution with padding.
         */

        bool ret;


        GraphType g;
        g.setRecipeName("nonZeroPadToPadded");

        TensorPtr OFM;
        unsigned OFMsize;
        char *ifm = nullptr;
        char *weights = nullptr;
        NodePtr   conv = createConvWithPadding(1, 1, OFM, OFMsize, ifm, weights); //conv with pads of size 1

        uint32_t pads[8] = {0, 1, 1, 0, 0, 1, 1, 0}; //CWHN(begin)CWHN(end) - symmetric padding
        char *padInput = nullptr;
        unsigned padValue = 1;
        NodePtr  pad = createPadNode(pads, conv, padInput, padValue);

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, conv);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph with non-zero pad";

        bool foundPad = false;
        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr && tpc->isGuidPrefix("pad"))
            {
                foundPad = true;
            }
        }
        ASSERT_TRUE(foundPad);

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] weights;
        delete[] ifm;
        delete[] padInput;
    }

    void fuse_non_zero_pad_into_int8_conv()
    {
        /*
         * should be fused:
         * pad value != 0 and conv node without padding.
         */

        bool ret;

        GraphType g;
        g.setRecipeName("nonZeroPadIntoInt8");

        TensorPtr OFM;
        unsigned OFMsize;
        char *ifm = nullptr;
        char *weights = nullptr;

        const synDataType type = syn_type_fixed;
        const double scale = 0.5;
        const double zp = -0.2;
        const float padValue = 2.0;
        const unsigned expectedQuantizedPadValue = 0X04040404;

        // create convolution node
        NodePtr               conv = createConvWithPadding(0, 0, OFM, OFMsize, ifm, weights, syn_type_fixed);
        synQuantizationParams quant;
        quant.m_qDataType = type;
        quant.m_scale = scale;
        quant.m_zp = zp;
        conv->getInput(TENSOR_IFM)->setQuantizationParams(quant);
        conv->setNodePrecision(syn_type_fixed);

        conv->getInput(TENSOR_WEIGHT)->setAsStaticParam();
        conv->getInput(TENSOR_WEIGHT)->setAsDataTypeMatchData();

        // create pad node
        uint32_t pads[8] = {0, 1, 1, 0, 0, 1, 1, 0}; //CWHN(begin)CWHN(end) - symmetric padding
        char *padInput = nullptr;
        NodePtr  pad = createPadNode(pads, conv, padInput, padValue, syn_type_fixed);

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, conv);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph with non-zero pad";

        bool foundPad = false;
        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr && tpc->isGuidPrefix("pad"))
            {
                foundPad = true;
            }
        }
        ASSERT_FALSE(foundPad);

        bool foundConv = false;
        uint32_t foundPaddingValue;
        for (NodePtr n : g.getExeSortedNodes())
        {
            std::shared_ptr<MmeNode> mme = std::dynamic_pointer_cast<MmeNode>(n);
            if (mme != nullptr )
            {
                foundConv = true;
                foundPaddingValue = mme->getPaddingValue(mme->getInput(TENSOR_IFM));
            }
        }
        ASSERT_TRUE(foundConv);
        ASSERT_EQ(foundPaddingValue, expectedQuantizedPadValue);

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] weights;
        delete[] ifm;
        delete[] padInput;
    }

    void fuse_non_zero_pad_into_int16_conv()
    {
        /*
         * should be fused:
         * pad value != 0 and conv node without padding.
         */

        bool ret;

        GraphType g;
        g.setRecipeName("nonZeroPadIntoInt16");

        TensorPtr OFM;
        unsigned OFMsize;
        char *ifm = nullptr;
        char *weights = nullptr;

        const synDataType type = syn_type_int16;
        const double scale = 0.5;
        const double zp = 0.0;
        const float padValue = 2.0;
        const unsigned expectedQuantizedPadValue = 0X00040004;

        // create convolution node
        NodePtr               conv = createConvWithPadding(0, 0, OFM, OFMsize, ifm, weights, syn_type_int16);
        synQuantizationParams quant;
        quant.m_qDataType = type;
        quant.m_scale = scale;
        quant.m_zp = zp;
        conv->getInput(TENSOR_IFM)->setQuantizationParams(quant);
        conv->getInput(TENSOR_IFM)->setInt16Limited(false);
        conv->setNodePrecision(syn_type_int16);

        // create pad node
        uint32_t pads[8] = {0, 1, 1, 0, 0, 1, 1, 0}; //CWHN(begin)CWHN(end) - symmetric padding
        char *padInput = nullptr;
        NodePtr  pad = createPadNode(pads, conv, padInput, padValue, syn_type_int16);

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, conv);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph with non-zero pad";

        bool foundPad = false;
        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr && tpc->isGuidPrefix("pad"))
            {
                foundPad = true;
            }
        }
        ASSERT_FALSE(foundPad);

        bool foundConv = false;
        uint32_t foundPaddingValue;
        for (NodePtr n : g.getExeSortedNodes())
        {
            std::shared_ptr<MmeNode> mme = std::dynamic_pointer_cast<MmeNode>(n);
            if (mme != nullptr )
            {
                foundConv = true;
                foundPaddingValue = mme->getPaddingValue(mme->getInput(TENSOR_IFM));
            }
        }
        ASSERT_TRUE(foundConv);
        ASSERT_EQ(foundPaddingValue, expectedQuantizedPadValue);

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] weights;
        delete[] ifm;
        delete[] padInput;
    }

    void fuse_pad_into_conv_non_spatial()
    {
        /*
         * shouldn't be fused:
         * pad on non-spatial dimensions (e.g. pad on C).
         */
        bool ret;

        GraphType g;
        g.setRecipeName("nonSpatialPad");

        TensorPtr OFM;
        unsigned OFMsize;
        char *ifm = nullptr;
        char *weights = nullptr;
        NodePtr   conv = createConvWithPadding(1, 1, OFM, OFMsize, ifm, weights); //conv with pads of size 1

        uint32_t pads[8] = {1, 1, 1, 0, 1, 1, 1, 0}; //CWHN(begin)CWHN(end) - symmetric padding
        char *padInput = nullptr;
        NodePtr  pad = createPadNode(pads, conv, padInput);

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, conv);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph with non-spatial pad";

        bool foundPad = false;
        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr && tpc->isGuidPrefix("pad"))
            {
                foundPad = true;
            }
        }
        ASSERT_TRUE(foundPad);

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] weights;
        delete[] ifm;
        delete[] padInput;
    }

    static NodePtr createPoolNodeWithPadding(unsigned pad_w_begin, unsigned pad_h_begin, unsigned pad_w_end, unsigned pad_h_end,
                                             bool maxPool, char*& input, char*& output, TSize& outputSize)
    {
        static const char* maxPoolKernelName     = "maxpool_2d_i8";
        static const char* averagePoolKernelName = "avg_pool_2d_i8";

        ns_AveragePooling::Params poolParams; // average pooling params include the params for max pool as well
        poolParams.pad_h_begin   = pad_h_begin;
        poolParams.pad_w_begin   = pad_w_begin;
        poolParams.pad_h_end     = pad_h_end;
        poolParams.pad_w_end     = pad_w_end;
        poolParams.kernel_w      = 3;
        poolParams.kernel_h      = 3;
        poolParams.stride_w      = 2;
        poolParams.stride_h      = 2;
        poolParams.dilation_w    = 1;
        poolParams.dilation_h    = 1;
        poolParams.pooling_convention = POOLING_CONVENTION_VALID;
        poolParams.includePadding = false; //only for average pooling

        const TSize inZ  = 64;
        const TSize outZ = inZ;
        const TSize outW = 29;
        const TSize outH = 29;
        const TSize inW = ((outW - 1) * poolParams.stride_w) + poolParams.kernel_w - poolParams.pad_w_begin - poolParams.pad_w_end;
        const TSize inH = ((outH - 1) * poolParams.stride_h) + poolParams.kernel_h - poolParams.pad_h_begin - poolParams.pad_h_end;
        const TSize inSizes[]  = { 1, inH, inW, inZ };
        const TSize outSizes[] = { 1, outH, outW, outZ };
        outputSize = outH * outW * outZ;

        if (input == nullptr)
        {
            input = new char[inH * inW * inZ];
        }
        if (output == nullptr)
        {
            output = new char[outputSize];
        }

        TensorPtr IFM = std::make_shared<Tensor>(4U, inSizes, syn_type_fixed, input);
        TensorPtr OFM = std::make_shared<Tensor>(4U, outSizes, syn_type_fixed, output);

        const char* guid = maxPool ? maxPoolKernelName : averagePoolKernelName;

        Node::NodeProperties p;
        p.inputLayouts = {gc::Layout("CWHN")};
        p.outputLayouts = {gc::Layout("CWHN")};
        NodePtr  pool = NodeFactory::createNode({IFM}, {OFM}, nullptr, guid, "pooling", p);
        TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(pool);
        tpc->storeParamsInBuffer(&poolParams, sizeof(ns_AveragePooling::Params));
        return pool;
    }

#define COMPARE_RESULTS 0 //disabled since executing the tpc nodes in a gc test takes a very long time

    void fuse_pad_into_max_pool_symmetric()
    {
        /*
         * max pool with symmetric padding and pad value -inf
         */

        bool ret;

        GraphType ref_g;
        ref_g.setRecipeName("maxpoolSymmetricRef");
        GraphType g;
        g.setRecipeName("maxpoolSymmetric");

        TSize outSize;
        char *input = nullptr;

#if COMPARE_RESULTS
        char *refOutput = nullptr;
        NodePtr refPool = createPoolNodeWithPadding(2, 2, 2, 2, true, input, refOutput, outSize); // pooling with padding of 2 on each side of W and H

        ref_GraphEditor::addNode(g, refPool);

        ret = ref_g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile reference graph of maxpoool with symmetric padding";

        ret = ref_g.execute();
        ASSERT_EQ(ret, true) << "Failed to execute reference graph of maxpoool with symmetric padding";
#endif

        char *output = nullptr;
        NodePtr pool = createPoolNodeWithPadding(1, 1, 1, 1, true, input, output, outSize); // pooling with padding of 1 on each side of W and H

        uint32_t pads[8] = {0, 1, 1, 0, 0, 1, 1, 0}; //CWHN(begin)CWHN(end) - symmetric padding of size 1
        NodePtr  pad = createPadNode(pads, pool, input, -INFINITY);

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, pool);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph of maxpoool with symmetric padding";

        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr)
            {
                ASSERT_FALSE(tpc->isGuidPrefix("pad")); // the pad wasn't fused into the pool
                ASSERT_TRUE(tpc->isGuidPrefix("maxpool"));
                ns_SpatialReduction::Params *params = (ns_SpatialReduction::Params *)tpc->getParams();
                ASSERT_TRUE(params->pad_w_begin == 2 &&
                            params->pad_h_begin == 2 &&
                            params->pad_w_end == 2 &&
                            params->pad_h_end == 2) << "Wrong padding fused into pool";
            }
        }

#if COMPARE_RESULTS
        ret = g.execute();
        ASSERT_EQ(ret, true) << "Failed to execute graph of maxpoool with symmetric padding";

        for (unsigned i = 0; i < outSize; ++i)
        {
            ASSERT_TRUE(output[i] == refOutput[i]) << "Wrong output at byte" << i << " Out: " << output[i] << " Ref: " << refOutput[i];
        }
#endif

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] input;
        delete[] output;
#if COMPARE_RESULTS
        delete[] refOutput;
#endif
    }

    void fuse_pad_into_max_pool_asymmetric()
    {
        /*
         * max pool with asymmetric padding and pad value -inf
         */
        bool ret;

        GraphType g;
        g.setRecipeName("maxpoolAsymmetric");

        TSize outSize;
        char *input = nullptr;
        char *output = nullptr;
        NodePtr  pool = createPoolNodeWithPadding(1, 1, 1, 1, true, input, output, outSize); // pooling with padding of 1 on each side of W and H

        uint32_t pads[8] = {0, 1, 0, 0, 0, 0, 1, 0}; //CWHN(begin)CWHN(end) - asymmetric padding of size 1 on w_begin and h_end
        NodePtr  pad = createPadNode(pads, pool, input, -INFINITY);

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, pool);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph of maxpoool with asymmetric padding";

        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr)
            {
                ASSERT_FALSE(tpc->isGuidPrefix("pad")); // the pad wasn't fused into the pool
                ASSERT_TRUE(tpc->isGuidPrefix("maxpool"));
                ns_SpatialReduction::Params *params = (ns_SpatialReduction::Params *)tpc->getParams();
                ASSERT_TRUE(params->pad_w_begin == 2 &&
                            params->pad_h_begin == 1 &&
                            params->pad_w_end == 1 &&
                            params->pad_h_end == 2) << "Wrong padding fused into pool";
            }
        }

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] input;
        delete[] output;
    }

    void fuse_pad_into_avg_pool_symmetric()
    {
        /*
         * avg pool with symmetric padding and pad value 0
         */
        bool ret;

        GraphType g;
        g.setRecipeName("avgpoolSymmetric");

        TSize outSize;
        char *input = nullptr;
        char *output = nullptr;
        NodePtr  pool = createPoolNodeWithPadding(1, 1, 1, 1, false, input, output, outSize); // pooling with padding of 1 on each side of W and H

        uint32_t pads[8] = {0, 1, 1, 0, 0, 1, 1, 0}; //CWHN(begin)CWHN(end) - symmetric padding of size 1
        NodePtr  pad = createPadNode(pads, pool, input);

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, pool);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph of avgpoool with symmetric padding";

        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr)
            {
                ASSERT_FALSE(tpc->isGuidPrefix("pad")); // the pad wasn't fused into the pool
                ASSERT_TRUE(tpc->isGuidPrefix("avg_pool"));
                ns_AveragePooling::Params *params = (ns_AveragePooling::Params *) tpc->getParams();
                ASSERT_TRUE(params->pad_w_begin == 2 &&
                            params->pad_h_begin == 2 &&
                            params->pad_w_end == 2 &&
                            params->pad_h_end == 2) << "Wrong padding fused into pool";
            }
        }

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] input;
        delete[] output;
    }

    void fuse_conv_pad_into_max_pool_remove_cast()
    {
        /*
         * avg pool with symmetric padding and pad value 0
         */
        bool ret;

        GraphType g;
        g.setRecipeName("convPadMaxPoolSymmetricRemoveCast");

        char *weights = nullptr;
        TensorPtr OFM_CONV;
        char *ifm = nullptr;

        const TSize batch = 1;
        const TSize nOFM = 64;
        const TSize wOFM = 11;
        const TSize hOFM = 3;
        const TSize nIFM = 64;

        synConvolutionParams params;
        params.dH   = 1;
        params.dW   = 1;
        params.kH   = 1;
        params.kW   = 1;
        params.padT = 1;
        params.padB = 1;
        params.padL = 1;
        params.padR = 1;
        params.dilH = 1;
        params.dilW = 1;
        const TSize wIFM = ((wOFM - 1) * params.dW) + (params.kW - 1) * params.dilW + 1 - (params.padL + params.padR);
        const TSize hIFM = ((hOFM - 1) * params.dH) + (params.kH - 1) * params.dilH + 1 - (params.padT + params.padB);

        if (ifm == nullptr)
        {
            ifm     = new char[nIFM * wIFM * hIFM * batch];
            std::generate(ifm, ifm + nIFM * wIFM * hIFM * batch, Test_Random_Number_Creator (std::array<int, 2>({-2,2})));
        }
        if (weights == nullptr)
        {
            weights = new char[nIFM * nOFM * params.kW * params.kH];
            std::generate(weights, weights + nIFM * nOFM * params.kW * params.kH, Test_Random_Number_Creator (std::array<int, 2>({-2,2})));
        }

        const TSize i_sizes[]        = { nIFM, wIFM, hIFM, batch };
        const TSize o_sizes[]        = { nOFM, wOFM, hOFM, batch };
        const TSize w_sizes[]        = { nOFM, nIFM, params.kW, params.kH };

        TensorPtr IFM_CONV = std::make_shared<Tensor>(4U, i_sizes, syn_type_fixed, ifm);
        TensorPtr W_CONV   = std::make_shared<Tensor>(4U, w_sizes, syn_type_fixed, weights);
        OFM_CONV         = std::make_shared<Tensor>(4U, o_sizes, syn_type_int16);

        NodePtr conv = getConvNodeWithGoyaLayouts(IFM_CONV, W_CONV, nullptr, OFM_CONV, params, "");
        GraphEditor::addNode(g, conv);
        unsigned outSize;

        ns_AveragePooling::Params poolParams; // average pooling params include the params for max pool as well
        poolParams.pad_h_begin   = 1;
        poolParams.pad_w_begin   = 1;
        poolParams.pad_h_end     = 1;
        poolParams.pad_w_end     = 1;
        poolParams.kernel_w      = 1;
        poolParams.kernel_h      = 1;
        poolParams.stride_w      = 1;
        poolParams.stride_h      = 1;
        poolParams.dilation_w    = 1;
        poolParams.dilation_h    = 1;
        poolParams.pooling_convention = POOLING_CONVENTION_VALID;

        const TSize inZ  = 64;
        const TSize outZ = 64;
        const TSize outW = 5;
        const TSize outH = 13;
        const TSize inW = ((outW - 1) * poolParams.stride_w) + poolParams.kernel_w - poolParams.pad_w_begin - poolParams.pad_w_end;
        const TSize inH = ((outH - 1) * poolParams.stride_h) + poolParams.kernel_h - poolParams.pad_h_begin - poolParams.pad_h_end;
        const TSize inSizes[]  = { inZ, inH, inW, 1 };
        const TSize outSizes[] = { outZ, outH, outW, 1 };
        outSize = outH * outW * outZ;

        char * input = new char[inH * inW * inZ];
        char * output = new char[outSize];

        TensorPtr IFM_POOL = std::make_shared<Tensor>(4U, inSizes, syn_type_fixed, input);
        TensorPtr OFM_POOL = std::make_shared<Tensor>(4U, outSizes, syn_type_fixed, output);

        Node::NodeProperties p;
        p.inputLayouts = {gc::Layout("CWHN")};
        p.outputLayouts = {gc::Layout("CWHN")};
        NodePtr pool = NodeFactory::createNode({IFM_POOL}, {OFM_POOL}, &poolParams, "maxpool_2d_i8", "pooling", p);

        uint32_t pads[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        TensorPtr padOutput = pool->getInput(TENSOR_IFM);
        TSize sizes[Tensor::c_tensorMaxDim] = {0};
        padOutput->getAllSizesInElements(sizes, Tensor::c_tensorMaxDim);
        unsigned inputSize = 1;
        for (int i = 0; i < 4; ++i)
        {
            sizes[i] = sizes[i] - pads[i] - pads[i + 4];
            inputSize *= sizes[i];
        }
        if (input == nullptr)
        {
            input     = new char[inputSize];
            std::generate(input, input + inputSize, Test_Random_Number_Creator (std::array<int, 2>({-2,2})));
        }

        ns_PadKernel::Params padParams;
        padParams.value.f = -std::numeric_limits<float>::infinity();
        memcpy(padParams.pads, pads, sizeof(uint32_t) * 8);
        NodePtr pad = NodeFactory::createNode({OFM_CONV}, {padOutput}, &padParams, "pad_i8", "pad_node");

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, pool);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph of maxpool with symmetric padding";

        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr)
            {
                ASSERT_FALSE(tpc->isGuidPrefix("pad")); // the pad wasn't fused into the maxpool
            }
        }

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] input;
        delete[] output;
        delete[] weights;
        delete[] ifm;
    }

/*
in the following test,  the output of the mme goes into both pad and another mme,
 and so the cast is inserted only on the direction of the pad, but the original tensor before the
 cast has to remain since it goes into the other mme
*/

    void fuse_conv_pad_into_max_pool_keep_cast()
    {
        /*
         * avg pool with symmetric padding and pad value 0
         */
        bool ret;

        GraphType g;
        g.setRecipeName("convPadMaxPoolSymmetricKeepCast");

        int16_t *weights = nullptr;
        TensorPtr OFM_CONV;
        int16_t *ifm = nullptr;

        const TSize batch = 1;
        const TSize nOFM = 64;
        const TSize wOFM = 11;
        const TSize hOFM = 3;
        const TSize nIFM = 64;

        synConvolutionParams params;
        params.dH   = 1;
        params.dW   = 1;
        params.kH   = 1;
        params.kW   = 1;
        params.padT = 1;
        params.padB = 1;
        params.padL = 1;
        params.padR = 1;
        params.dilH = 1;
        params.dilW = 1;
        const TSize wIFM = ((wOFM - 1) * params.dW) + (params.kW - 1) * params.dilW + 1 - (params.padL + params.padR);
        const TSize hIFM = ((hOFM - 1) * params.dH) + (params.kH - 1) * params.dilH + 1 - (params.padT + params.padB);

        if (ifm == nullptr)
        {
            ifm     = new int16_t[nIFM * wIFM * hIFM * batch];
            std::generate(ifm, ifm + nIFM * wIFM * hIFM * batch, Test_Random_Number_Creator (std::array<int, 2>({-2,2})));
        }
        if (weights == nullptr)
        {
            weights = new int16_t[nIFM * nOFM * params.kW * params.kH];
            std::generate(weights, weights + nIFM * nOFM * params.kW * params.kH, Test_Random_Number_Creator (std::array<int, 2>({-2,2})));
        }

        const TSize i_sizes[]        = { nIFM, wIFM, hIFM, batch };
        const TSize o_sizes[]        = { nOFM, wOFM, hOFM, batch };
        const TSize w_sizes[]        = { nOFM, nIFM, params.kW, params.kH };

        TensorPtr IFM_CONV = std::make_shared<Tensor>(4U, i_sizes, syn_type_int16, (char*)ifm);
        TensorPtr W_CONV   = std::make_shared<Tensor>(4U, w_sizes, syn_type_int16, (char*)weights);
        OFM_CONV         = std::make_shared<Tensor>(4U, o_sizes, syn_type_int16);

        NodePtr conv = getConvNodeWithGoyaLayouts(IFM_CONV, W_CONV, nullptr, OFM_CONV, params, "");
        GraphEditor::addNode(g, conv);
        TensorPtr OFM2_CONV         = std::make_shared<Tensor>(4U, o_sizes, syn_type_int16);

        Node::NodeProperties p2;
        p2.inputLayouts = {gc::Layout("CWHN"), gc::Layout("KCSR"), gc::Layout(), gc::Layout("CWHN")};
        p2.outputLayouts = {gc::Layout("CWHN")};
        params.padT = 0;
        params.padB = 0;
        params.padL = 0;
        params.padR = 0;
        NodePtr conv2 = getConvNodeWithGoyaLayouts(OFM_CONV, W_CONV, nullptr, OFM2_CONV, params, "");
        GraphEditor::addNode(g, conv2);
        TSize outSize;

        ns_AveragePooling::Params poolParams; // average pooling params include the params for max pool as well
        poolParams.pad_h_begin   = 1;
        poolParams.pad_w_begin   = 1;
        poolParams.pad_h_end     = 1;
        poolParams.pad_w_end     = 1;
        poolParams.kernel_w      = 1;
        poolParams.kernel_h      = 1;
        poolParams.stride_w      = 1;
        poolParams.stride_h      = 1;
        poolParams.dilation_w    = 1;
        poolParams.dilation_h    = 1;
        poolParams.pooling_convention = POOLING_CONVENTION_VALID;

        const TSize inZ  = 64;
        const TSize outZ = 64;
        const TSize outW = 5;
        const TSize outH = 13;
        const TSize inW = ((outW - 1) * poolParams.stride_w) + poolParams.kernel_w - poolParams.pad_w_begin - poolParams.pad_w_end;
        const TSize inH = ((outH - 1) * poolParams.stride_h) + poolParams.kernel_h - poolParams.pad_h_begin - poolParams.pad_h_end;
        const TSize inSizes[]  = { inZ, inH, inW, 1 };
        const TSize outSizes[] = { outZ, outH, outW, 1 };
        outSize = outH * outW * outZ;

        char * input = new char[inH * inW * inZ];
        char * output = new char[outSize];

        TensorPtr IFM_POOL = std::make_shared<Tensor>(4U, inSizes, syn_type_fixed, input);
        TensorPtr OFM_POOL = std::make_shared<Tensor>(4U, outSizes, syn_type_fixed, output);

        Node::NodeProperties p;
        p.inputLayouts = {gc::Layout("CWHN")};
        p.outputLayouts = {gc::Layout("CWHN")};
        NodePtr pool = NodeFactory::createNode({IFM_POOL}, {OFM_POOL}, &poolParams, "maxpool_2d_i8", "pooling", p);

        uint32_t pads[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        TensorPtr padOutput = pool->getInput(TENSOR_IFM);
        TSize sizes[Tensor::c_tensorMaxDim] = {0};
        padOutput->getAllSizesInElements(sizes, Tensor::c_tensorMaxDim);
        TSize inputSize = 1;
        for (int i = 0; i < 4; ++i)
        {
            sizes[i] = sizes[i] - pads[i] - pads[i + 4];
            inputSize *= sizes[i];
        }
        if (input == nullptr)
        {
            input     = new char[inputSize];
            std::generate(input, input + inputSize, Test_Random_Number_Creator (std::array<int, 2>({-2,2})));
        }

        ns_PadKernel::Params padParams;
        padParams.value.f = -std::numeric_limits<float>::infinity();
        memcpy(padParams.pads, pads, sizeof(uint32_t) * 8);
        NodePtr pad = NodeFactory::createNode({OFM_CONV}, {padOutput}, &padParams, "pad_i8", "pad_node");

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, pool);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph of maxpool with symmetric padding";

        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr)
            {
                TensorPtr input = n->getInput(0);
                NodePtr   producer = g.getTensorProducer(input);
                ASSERT_FALSE(tpc->isGuidPrefix("pad")); // the pad wasn't fused into the maxpool
                if (HabanaGraph::runsOnMME(producer))
                {
                    ASSERT_TRUE(tpc->isGuidPrefix("cast")); // the cast was removed (it shouldn't)
                }
            }
        }

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] input;
        delete[] output;
        delete[] weights;
        delete[] ifm;
    }

    void fuse_pad_into_avg_pool_asymmetric()
    {
        /*
        * avg pool with asymmetric padding and pad value 0
        */
        bool ret;

        GraphType g;
        g.setRecipeName("avgpoolAsymmetric");

        TSize outSize;
        char *input = nullptr;
        char *output = nullptr;
        NodePtr  pool = createPoolNodeWithPadding(1, 1, 1, 1, false, input, output, outSize); // pooling with padding of 1 on each side of W and H

        uint32_t pads[8] = {0, 1, 0, 0, 0, 0, 1, 0}; //CWHN(begin)CWHN(end) - asymmetric padding of size 1 on w_begin and h_end
        NodePtr  pad = createPadNode(pads, pool, input);

        GraphEditor::addNode(g, pad);
        GraphEditor::addNode(g, pool);

        ret = g.compile();
        ASSERT_EQ(ret, true) << "Failed to compile graph of avgpoool with asymmetric padding";

        for (NodePtr n : g.getExeSortedNodes())
        {
            TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
            if (tpc != nullptr)
            {
                ASSERT_FALSE(tpc->isGuidPrefix("pad")); // the pad wasn't fused into the pool
                ASSERT_TRUE(tpc->isGuidPrefix("avg_pool"));
                ns_AveragePooling::Params* params = (ns_AveragePooling::Params*)tpc->getParams();
                ASSERT_TRUE(params->pad_w_begin == 2 &&
                            params->pad_h_begin == 1 &&
                            params->pad_w_end == 1 &&
                            params->pad_h_end == 2) << "Wrong padding fused into pool";
            }
        }

        std::string gcfgUsedFileName = g.getRecipeName() + ".used";
        std::remove(gcfgUsedFileName.c_str());

        delete[] input;
        delete[] output;
    }

};
