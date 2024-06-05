#include <gtest/gtest.h>
#include "tensor.h"
#include "sim_graph.h"
#include "gaudi_graph.h"
#include "infra/global_conf_manager.h"
#include "graph_optimizer_test.h"
#include <graph_compiler/habana_nodes/node_factory.h>
#include <test_utils.h>
#include <perf_lib_layer_params.h>

using namespace gc;


bool isRequantNode(const pNode& node)
{
    return (node.get()->getNodeTypeStr().find("requant") != std::string::npos);
}

bool dTypetoRequantGuid(const synDataType type, std::string& str)
{
    switch (type)
    {
        case syn_type_int4:
        case syn_type_uint4:
            str = "requant_i4";
            return true;
        case syn_type_fixed:
        case syn_type_uint8:
            str = "requant_i8";
            return true;
        case syn_type_int16:
        case syn_type_uint16:
            str = "requant_i16";
            return true;
        default:
            return false;
    }
}

class RequantTest : public GraphOptimizerTest
{

protected:

    virtual void SetUp()
    {
        GraphOptimizerTest::SetUp();
        GCFG_ENABLE_SYNAPSE_QUANTIZATION.setValue(true);
    }

    virtual void TearDown()
    {
        GraphOptimizerTest::TearDown();
    }

    void create4DTensor(pTensor& t, const std::string name, const TSize sizes[4],
                        const synDataType type,  const double zp, const double scale,  float* data)
    {
        t = pTensor(new Tensor(4U, sizes, type, reinterpret_cast<char *>(data)));
        t->setName(name);
        QuantizationData quantInfo(type);
        quantInfo.setScale(scale);
        quantInfo.setZp(zp);
        t->setQuantizationParams(quantInfo);
        t->setMeasuredQuantizationParams();
    }

    void create_forward_conflicted_model(synDataType dType)
    {
        /*
              _S_         _C_
        t1   [  ] -----> [  ]    t4
        []-> [  ]        [  ] -> []
             [__] -----> [__]
        */

        GaudiGraph g;

        const TSize n = 1;
        const TSize w = 3;
        const TSize h = 3;
        const TSize batch = 1;
        float in1[n * w * h * batch] = {1, 3, 5, 7, 2, 4, 6, 8, 10};
        float in2[n * w * 2 * h * batch];
        float in3[n * w * 2 * h * batch];
        float in4[n * w * h * batch];
        const TSize sizes1[] = {n, w, h, batch};
        const TSize sizes2[] = {n, w * 2, h, batch};

        pTensor t1, t2, t3, t4;
        create4DTensor(t1, "t1", sizes2, dType, 0, 0.1, in1);
        create4DTensor(t2, "t2", sizes1, dType, 0, 0.2, in2);
        create4DTensor(t3, "t3", sizes1, dType, 0, 0.3, in3);
        create4DTensor(t4, "t4", sizes2, dType, 0, 0.4, in4);

        DynamicRange dynamicRange;
        dynamicRange.min   = 0;
        dynamicRange.max   = 1;
        dynamicRange.isSet = true;

        t1->setDynamicRange(dynamicRange);
        t2->setDynamicRange(dynamicRange);
        t3->setDynamicRange(dynamicRange);
        t4->setDynamicRange(dynamicRange);

        unsigned splitParams1 = 1;
        pNode split1Node = NodeFactory::createNode({t1}, {t2, t3}, &splitParams1, "split", "split1");
        unsigned concat1Params = 1;
        pNode concat1Node = NodeFactory::createNode({t2, t3}, {t4}, &concat1Params, "concat", "concat1");
        GraphEditor::addNode(g, split1Node);
        GraphEditor::addNode(g, concat1Node);
        ASSERT_TRUE(adjustScales(g)); // run pass
        ASSERT_TRUE(requantConflicts(g)); // run pass

        if (dType == syn_type_single)
        {
            const NodeVector& sortedNodes = g.getExeSortedNodes();
            auto it = std::find_if(sortedNodes.begin(), sortedNodes.end(), isRequantNode);
            ASSERT_EQ(it, sortedNodes.end()) << "Error: Requant Node was found in the graph!";
        }
        else
        {
            std::string requantGuid;
            ASSERT_TRUE( dTypetoRequantGuid(dType, requantGuid));

            pTensor requant0Output = concat1Node->getInput(0);
            pNode requant0Node = g.getTensorProducer(requant0Output);
            ASSERT_EQ(requant0Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
            ASSERT_EQ(requant0Output->getQuantizationParams().scale(0), 0.4)
                                        << "Error: Incorrect scale used for tensor " << requant0Output->getName();
            ASSERT_EQ(requant0Output->getQuantizationParams().zp(0), 0)
                                        << "Error: Incorrect zero point used for tensor " << requant0Output->getName();

            pTensor requant1Output = concat1Node->getInput(1);
            pNode requant1Node = g.getTensorProducer(requant1Output);
            ASSERT_EQ(requant1Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
            ASSERT_EQ(requant1Output->getQuantizationParams().scale(0), 0.4)
                                        << "Error: Incorrect scale used for tensor " << requant1Output->getName();
            ASSERT_EQ(requant1Output->getQuantizationParams().zp(0), 0)
                                        << "Error: Incorrect zero point used for tensor " << requant1Output->getName();
        }
    }

    void  create_backward_conflicted_model(synDataType dType)
    {
        /*
            t2
            []   _C_     _S_
               \[   ] _ [   ]/
            t1 /[___]   [___]\  _C_   t10
            []   _C_     _S_   [___]->[]
               \[   ] _ [   ]/
            t3 /[___]   [___]\
            []

       */

        GaudiGraph g;

        const TSize n     = 1;
        const TSize w     = 3;
        const TSize h     = 3;
        const TSize batch = 1;
        float in1[n * w * h * batch] = {1, 3, 5, 7, 2, 4, 6, 8, 10};
        float in2[n * w * 2 * h * batch];
        float in3[n * w * 2 * h * batch];
        float in4[n * w * h * batch];
        float in5[n * w * 2 * h * batch];
        float in6[n * w * h * batch];
        float in7[n * w * 2 * h * batch];
        float in8[n * w * h * batch];
        float in9[n * w * h * batch];
        float in10[n * w * h * batch];
        const TSize sizes1[] = {n, w, h, batch};
        const TSize sizes2[] = {n, w*2, h, batch};

        pTensor t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
        create4DTensor(t1, "t1", sizes1, dType, 0, 0.1, in1);
        create4DTensor(t2, "t2", sizes1, dType, 0, 0.2, in2);
        create4DTensor(t3, "t3", sizes1, dType, 0, 0.3, in3);
        create4DTensor(t4, "t4", sizes2, dType, 0, 0.4, in4);
        create4DTensor(t5, "t5", sizes2, dType, 0, 0.5, in5);
        create4DTensor(t6, "t6", sizes1, dType, 0, 0.6, in6);
        create4DTensor(t7, "t7", sizes1, dType, 0, 0.7, in7);
        create4DTensor(t8, "t8", sizes1, dType, 0, 0.8, in8);
        create4DTensor(t9, "t9", sizes1, dType, 0, 0.9, in9);
        create4DTensor(t10, "t10", sizes2, dType, 0, 1, in10);

        DynamicRange dynamicRange;
        dynamicRange.min   = 0;
        dynamicRange.max   = 1;
        dynamicRange.isSet = true;

        t1->setDynamicRange(dynamicRange);
        t2->setDynamicRange(dynamicRange);
        t3->setDynamicRange(dynamicRange);
        t4->setDynamicRange(dynamicRange);
        t5->setDynamicRange(dynamicRange);
        t6->setDynamicRange(dynamicRange);
        t7->setDynamicRange(dynamicRange);
        t8->setDynamicRange(dynamicRange);
        t9->setDynamicRange(dynamicRange);
        t10->setDynamicRange(dynamicRange);

        unsigned concat1Params = 1;
        pNode concat1Node = NodeFactory::createNode({t1, t2}, {t4}, &concat1Params, "concat", "concat1");
        unsigned concat2Params = 1;
        pNode concat2Node = NodeFactory::createNode({t1, t3}, {t5}, &concat2Params, "concat", "concat2");
        unsigned splitParams1 = 1;
        pNode split1Node = NodeFactory::createNode({t4}, {t6, t7}, &splitParams1, "split", "split1");
        unsigned split2Params = 1;
        pNode split2Node = NodeFactory::createNode({t5}, {t8, t9}, &split2Params, "split", "split2");
        unsigned concat3Params = 1;
        pNode concat3Node = NodeFactory::createNode({t7, t8}, {t10}, &concat3Params, "concat", "concat3");
        GraphEditor::addNode(g, concat1Node);
        GraphEditor::addNode(g, concat2Node);
        GraphEditor::addNode(g, split1Node);
        GraphEditor::addNode(g, split2Node);
        GraphEditor::addNode(g, concat3Node);

        ASSERT_TRUE(adjustScales(g)); // run pass
        ASSERT_TRUE(requantConflicts(g)); // run pass

        if (dType == syn_type_single)
        {
            const NodeVector& sortedNodes = g.getExeSortedNodes();
            auto it = std::find_if(sortedNodes.begin(), sortedNodes.end(), isRequantNode);
            ASSERT_EQ(it, sortedNodes.end()) << "Error: Requant Node was found in the graph!";
        }
        else
        {
            std::string requantGuid;
            ASSERT_TRUE( dTypetoRequantGuid(dType, requantGuid));
            pTensor requant0Output = concat3Node->getInput(0);
            pNode requant0Node = g.getTensorProducer(requant0Output);
            ASSERT_EQ(requant0Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
            ASSERT_EQ(requant0Output->getQuantizationParams().scale(0), 1)
                                        << "Error: Incorrect scale used for tensor " << requant0Output->getName();
            ASSERT_EQ(requant0Output->getQuantizationParams().zp(0), 0)
                                        << "Error: Incorrect zero point used for tensor " << requant0Output->getName();

            pTensor requant1Output = concat3Node->getInput(1);
            pNode requant1Node = g.getTensorProducer(requant1Output);
            ASSERT_EQ(requant1Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
            ASSERT_EQ(requant1Output->getQuantizationParams().scale(0), 1)
                                        << "Error: Incorrect scale used for tensor " << requant1Output->getName();
            ASSERT_EQ(requant1Output->getQuantizationParams().zp(0), 0)
                                        << "Error: Incorrect zero point used for tensor " << requant1Output->getName();
        }
    }

    void create_simple_backward_conflicted_model(synDataType dType)
    {
        /*
                    t0
                    |
                  (tanh)
                    |
            t1      t2      t3
             \    /   \    /
            (concat) (concat)
                |       |
                t4      t5
        */

        GaudiGraph g;

        const TSize n                      = 1;
        const TSize w                      = 3;
        const TSize h                      = 3;
        const TSize batch                  = 1;
        float       in0[n * w * h * batch] = {1, 3, 5, 7, 2, 4, 6, 8, 10};
        float       in1[n * w * h * batch];
        float       in2[n * w * 2 * h * batch];
        float       in3[n * w * 2 * h * batch];
        float       in4[n * w * h * batch];
        float       in5[n * w * 2 * h * batch];
        const TSize sizes1[] = {n, w, h, batch};
        const TSize sizes2[] = {n, w * 2, h, batch};

        pTensor t0, t1, t2, t3, t4, t5;
        create4DTensor(t0, "t0", sizes1, dType, 0, 0.05, in0);
        create4DTensor(t1, "t1", sizes1, dType, 0, 0.1, in1);
        create4DTensor(t2, "t2", sizes1, dType, 0, 0.2, in2);
        create4DTensor(t3, "t3", sizes1, dType, 0, 0.3, in3);
        create4DTensor(t4, "t4", sizes2, dType, 0, 0.4, in4);
        create4DTensor(t5, "t5", sizes2, dType, 0, 0.5, in5);

        DynamicRange dynamicRange;
        dynamicRange.min   = 0;
        dynamicRange.max   = 1;
        dynamicRange.isSet = true;

        t0->setDynamicRange(dynamicRange);
        t1->setDynamicRange(dynamicRange);
        t2->setDynamicRange(dynamicRange);
        t3->setDynamicRange(dynamicRange);
        t4->setDynamicRange(dynamicRange);
        t5->setDynamicRange(dynamicRange);

        pNode    tanh1Node     = NodeFactory::createNode({t0}, {t2}, nullptr, "tanh", "tanh1");
        unsigned concat1Params = 1;
        pNode    concat1Node   = NodeFactory::createNode({t1, t2}, {t4}, &concat1Params, "concat", "concat1");
        unsigned concat2Params = 1;
        pNode    concat2Node   = NodeFactory::createNode({t2, t3}, {t5}, &concat2Params, "concat", "concat2");

        GraphEditor::addNode(g, tanh1Node);
        GraphEditor::addNode(g, concat1Node);
        GraphEditor::addNode(g, concat2Node);

        ASSERT_TRUE(adjustScales(g));      // run pass
        ASSERT_TRUE(requantConflicts(g));  // run pass

        std::string requantGuid;
        ASSERT_TRUE(dTypetoRequantGuid(dType, requantGuid));
        pTensor requant0Output = concat1Node->getInput(1);
        pNode   requant0Node   = g.getTensorProducer(requant0Output);
        ASSERT_EQ(requant0Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
        ASSERT_EQ(requant0Output->getQuantizationParams().scale(0), 0.4)
            << "Error: Incorrect scale used for tensor " << requant0Output->getName();
        ASSERT_EQ(requant0Output->getQuantizationParams().zp(0), 0)
            << "Error: Incorrect zero point used for tensor " << requant0Output->getName();

        pTensor requant1Output = concat2Node->getInput(0);
        pNode   requant1Node   = g.getTensorProducer(requant1Output);
        ASSERT_EQ(requant1Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
        ASSERT_EQ(requant1Output->getQuantizationParams().scale(0), 0.5)
            << "Error: Incorrect scale used for tensor " << requant1Output->getName();
        ASSERT_EQ(requant1Output->getQuantizationParams().zp(0), 0)
            << "Error: Incorrect zero point used for tensor " << requant1Output->getName();
    }

    void  create_ancestor_input_model(synDataType dType)
    {
        /*
        data    conv1     conv2     concat
        []---->[    ]--->[    ]--->[     ]->
                    \____________/
        */

        const TSize kW          = 5;
        const TSize kH          = 5;
        const TSize dW          = 1;
        const TSize dH          = 1;
        const TSize nOFM        = 1;
        const TSize wOFM        = 5;
        const TSize hOFM        = 5;
        const TSize nIFM        = 1;
        const TSize wIFM        = ((wOFM - 1) * dW) + kW;
        const TSize hIFM        = ((hOFM - 1) * dH) + kH;
        const TSize i_sizes[4]  = { nIFM, wIFM, hIFM, 1 };
        const TSize o_sizes[4]  = { nOFM, wOFM, hOFM, 1 };
        const TSize c_sizes[4]  = { nOFM, wOFM*2, hOFM, 1 };
        const TSize w_sizes[4]  = { nOFM, nIFM, kW, kH };
        const TSize b_sizes[1]  = { nOFM };
        std::shared_ptr<Tensor> IFM, OFM1, OFM2, W1, W2, B1, B2, COUT;
        synConvolutionParams    params1, params2;

        params1.dH = dH;
        params1.dW = dW;
        params1.kH = kH;
        params1.kW = kW;
        params1.padT = 0;
        params1.padB = 0;
        params1.padL = 0;
        params1.padR = 0;
        params1.dilH = 1;
        params1.dilW = 1;

        params2.dH = dH;
        params2.dW = dW;
        params2.kH = kH;
        params2.kW = kW;
        params2.padT = 0;
        params2.padB = 0;
        params2.padL = 0;
        params2.padR = 0;
        params2.dilH = 1;
        params2.dilW = 1;

        float ifm[nIFM * wIFM * hIFM];
        float weights1[nIFM * nOFM * kW * kH];
        float bias1[nOFM];
        float weights2[nIFM * nOFM * kW * kH];
        float bias2[nOFM];

        create4DTensor(IFM, "IFM", i_sizes, dType, 0, 0.1, ifm);
        create4DTensor(OFM1, "OFM1", o_sizes, dType, 0, 0.2, nullptr);
        W1    = std::shared_ptr<Tensor>(new Tensor(4U, w_sizes, dType, reinterpret_cast<char*>(weights1)));
        B1    = std::shared_ptr<Tensor>(new Tensor(1U, b_sizes, dType, reinterpret_cast<char*>(bias1)));
        create4DTensor(OFM2, "OFM2", o_sizes, dType, 0, 0.3, nullptr);
        W2    = std::shared_ptr<Tensor>(new Tensor(4U, w_sizes, dType, reinterpret_cast<char*>(weights2)));
        B2    = std::shared_ptr<Tensor>(new Tensor(1U, b_sizes, dType, reinterpret_cast<char*>(bias2)));
        create4DTensor(COUT, "COUT", c_sizes, dType, 0, 0.4, nullptr);

        DynamicRange dynamicRange;
        dynamicRange.min   = 0;
        dynamicRange.max   = 1;
        dynamicRange.isSet = true;

        IFM->setDynamicRange(dynamicRange);
        OFM1->setDynamicRange(dynamicRange);
        OFM2->setDynamicRange(dynamicRange);
        COUT->setDynamicRange(dynamicRange);

        GaudiGraph g;
        pNode convNode1 = getConvNodeWithGoyaLayouts(IFM, W1, B1, OFM1, params1, "conv1");
        GraphEditor::addNode(g, convNode1);
        pNode convNode2 = getConvNodeWithGoyaLayouts(OFM1, W2, B2, OFM2, params2, "conv2");
        GraphEditor::addNode(g, convNode2);
        unsigned concatParams = 1;
        pNode concatNode = NodeFactory::createNode({OFM1, OFM2}, {COUT}, &concatParams, "concat", "concat");
        GraphEditor::addNode(g, concatNode);

        ASSERT_TRUE(adjustScales(g)); // run pass
        ASSERT_TRUE(requantConflicts(g)); // run pass

        std::string requantGuid;
        ASSERT_TRUE(dTypetoRequantGuid(dType, requantGuid));
        pTensor requant0Output = concatNode->getInput(0);
        pNode requant0Node = g.getTensorProducer(requant0Output);
        ASSERT_EQ(requant0Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
        ASSERT_EQ(requant0Output->getQuantizationParams().scale(0), 0.4)
                                    << "Error: Incorrect scale used for tensor " << requant0Output->getName();
        ASSERT_EQ(requant0Output->getQuantizationParams().zp(0), 0)
                                    << "Error: Incorrect zero point used for tensor " << requant0Output->getName();
    }

    void  create_ancestor_dontcare_input_model(synDataType dType)
    {
        /*
        data    conv1     maxpool   concat
        []---->[    ]--->[    ]--->[     ]->
                    \____________/
        */

        const TSize kW          = 5;
        const TSize kH          = 5;
        const TSize dW          = 1;
        const TSize dH          = 1;
        const TSize nOFM        = 1;
        const TSize wOFM        = 5;
        const TSize hOFM        = 5;
        const TSize nIFM        = 1;
        const TSize wIFM        = ((wOFM - 1) * dW) + kW;
        const TSize hIFM        = ((hOFM - 1) * dH) + kH;
        const TSize i_sizes[4]  = { nIFM, wIFM, hIFM, 1 };
        const TSize o_sizes[4]  = { nOFM, wOFM, hOFM, 1 };
        const TSize m_sizes[4]  = { nOFM, wOFM, hOFM, 1 };
        const TSize c_sizes[4]  = { nOFM, wOFM*2, hOFM, 1 };
        const TSize w_sizes[4]  = { nOFM, nIFM, kW, kH };
        const TSize b_sizes[1]  = { nOFM };
        std::shared_ptr<Tensor> IFM, OFM, W, B, MOUT, COUT;
        synConvolutionParams    params;

        params.dH = dH;
        params.dW = dW;
        params.kH = kH;
        params.kW = kW;
        params.padT = 0;
        params.padB = 0;
        params.padL = 0;
        params.padR = 0;
        params.dilH = 1;
        params.dilW = 1;

        float ifm[nIFM * wIFM * hIFM];
        float weights[nIFM * nOFM * kW * kH];
        float bias[nOFM];

        create4DTensor(IFM, "IFM", i_sizes, dType, 0, 0.1, ifm);
        create4DTensor(OFM, "OFM1", o_sizes, dType, 0, 0.2, nullptr);
        W = std::shared_ptr<Tensor>(new Tensor(4U, w_sizes, dType, reinterpret_cast<char*>(weights)));
        B = std::shared_ptr<Tensor>(new Tensor(1U, b_sizes, dType, reinterpret_cast<char*>(bias)));
        create4DTensor(MOUT, "MOUT", m_sizes, dType, 0, 0.3, nullptr);
        create4DTensor(COUT, "COUT", c_sizes, dType, 0, 0.4, nullptr);

        DynamicRange dynamicRange;
        dynamicRange.min   = 0;
        dynamicRange.max   = 1;
        dynamicRange.isSet = true;

        IFM->setDynamicRange(dynamicRange);
        OFM->setDynamicRange(dynamicRange);
        MOUT->setDynamicRange(dynamicRange);
        COUT->setDynamicRange(dynamicRange);

        GaudiGraph g;
        pNode convNode = getConvNodeWithGoyaLayouts(IFM, W, B, OFM, params, "conv1");
        GraphEditor::addNode(g, convNode);

        ns_SpatialReduction::Params maxpoolParams;
        maxpoolParams.pooling_convention = POOLING_CONVENTION_VALID;
        maxpoolParams.kernel_w = 3;
        maxpoolParams.kernel_h = 3;
        maxpoolParams.stride_w = 2;
        maxpoolParams.stride_h = 2;
        maxpoolParams.dilation_w = 1;
        maxpoolParams.dilation_h = 1;
        maxpoolParams.pad_w_begin = 1;
        maxpoolParams.pad_w_end = 1;
        maxpoolParams.pad_h_begin = 1;
        maxpoolParams.pad_h_end = 1;

        pNode maxPoolNode = NodeFactory::createGenericTPCNode(TensorVector(1, OFM),
                                                              TensorVector(1, MOUT),
                                                              &maxpoolParams,
                                                              "maxpool_2d_i8",
                                                              "maxpool");
        GraphEditor::addNode(g, maxPoolNode);

        unsigned concatParams = 1;
        pNode concatNode = NodeFactory::createNode({OFM, MOUT}, {COUT}, &concatParams, "concat", "concat");
        GraphEditor::addNode(g, concatNode);

        ASSERT_TRUE(adjustScales(g)); // run pass
        ASSERT_TRUE(requantConflicts(g)); // run pass

        std::string requantGuid;
        ASSERT_TRUE(dTypetoRequantGuid(dType, requantGuid));
        pTensor requant0Output = concatNode->getInput(0);
        pNode requant0Node = g.getTensorProducer(requant0Output);
        ASSERT_EQ(requant0Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
        ASSERT_EQ(requant0Output->getQuantizationParams().scale(0), 0.4)
                                    << "Error: Incorrect scale used for tensor " << requant0Output->getName();
        ASSERT_EQ(requant0Output->getQuantizationParams().zp(0), 0)
                                    << "Error: Incorrect zero point used for tensor " << requant0Output->getName();
    }

    void  create_requant_conflicts_same_scale_backward_one_different_model(synDataType dType)
    {
        GaudiGraph g;

        const TSize n     = 1;
        const TSize w     = 3;
        const TSize h     = 3;
        const TSize batch = 1;
        float in1[n * w * h * batch] = {1, 3, 5, 7, 2, 4, 6, 8, 10};
        float in2[n * w * 2 * h * batch];
        float in3[n * w * 2 * h * batch];
        float in4[n * w * h * batch];
        float in5[n * w * 2 * h * batch];
        float in6[n * w * h * batch];
        float in7[n * w * 2 * h * batch];
        float in8[n * w * h * batch];
        float in9[n * w * h * batch];
        float in10[n * w * h * batch];
        float in11[n * w * h * batch];
        float in12[n * w * h * batch];
        float in13[n * w * h * batch];

        const TSize sizes1[] = {n, w, h, batch};
        const TSize sizes2[] = {n, w*2, h, batch};

        pTensor t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13;
        create4DTensor(t1, "t1", sizes1, dType, 0, 0.3, in1);
        create4DTensor(t2, "t2", sizes1, dType, 0, 0.3, in2);
        create4DTensor(t3, "t3", sizes1, dType, 0, 0.3, in3);
        create4DTensor(t4, "t4", sizes1, dType, 0, 0.3, in4);
        create4DTensor(t5, "t5", sizes2, dType, 0, 0.3, in5);
        create4DTensor(t6, "t6", sizes2, dType, 0, 0.5, in6);
        create4DTensor(t7, "t7", sizes2, dType, 0, 0.3, in7);
        create4DTensor(t8, "t8", sizes1, dType, 0, 0.3, in8);
        create4DTensor(t9, "t9", sizes1, dType, 0, 0.3, in9);
        create4DTensor(t10, "t10", sizes1, dType, 0, 0.3, in10);
        create4DTensor(t11, "t11", sizes1, dType, 0, 0.3, in11);
        create4DTensor(t12, "t12", sizes1, dType, 0, 0.3, in12);
        create4DTensor(t13, "t13", sizes1, dType, 0, 0.3, in13);

        DynamicRange dynamicRange;
        dynamicRange.min   = 0;
        dynamicRange.max   = 1;
        dynamicRange.isSet = true;

        t1->setDynamicRange(dynamicRange);
        t2->setDynamicRange(dynamicRange);
        t3->setDynamicRange(dynamicRange);
        t4->setDynamicRange(dynamicRange);
        t5->setDynamicRange(dynamicRange);
        t6->setDynamicRange(dynamicRange);
        t7->setDynamicRange(dynamicRange);
        t8->setDynamicRange(dynamicRange);
        t9->setDynamicRange(dynamicRange);
        t10->setDynamicRange(dynamicRange);
        t11->setDynamicRange(dynamicRange);
        t12->setDynamicRange(dynamicRange);
        t13->setDynamicRange(dynamicRange);

        unsigned concat1Params = 1;
        pNode concat1Node = NodeFactory::createNode({t1, t2}, {t5}, &concat1Params, "concat", "concat1");
        unsigned concat2Params = 1;
        pNode concat2Node = NodeFactory::createNode({t1, t3}, {t6}, &concat2Params, "concat", "concat2");
        unsigned concat3Params = 1;
        pNode concat3Node = NodeFactory::createNode({t1, t4}, {t7}, &concat3Params, "concat", "concat3");

        unsigned splitParams1 = 1;
        pNode split1Node = NodeFactory::createNode({t5}, {t8, t9}, &splitParams1, "split", "split1");
        unsigned split2Params = 1;
        pNode split2Node = NodeFactory::createNode({t6}, {t10, t11}, &split2Params, "split", "split2");
        unsigned split3Params = 1;
        pNode split3Node = NodeFactory::createNode({t7}, {t12, t13}, &split3Params, "split", "split3");

        GraphEditor::addNode(g, concat1Node);
        GraphEditor::addNode(g, concat2Node);
        GraphEditor::addNode(g, concat3Node);
        GraphEditor::addNode(g, split1Node);
        GraphEditor::addNode(g, split2Node);
        GraphEditor::addNode(g, split3Node);

        ASSERT_TRUE(adjustScales(g)); // run pass
        ASSERT_TRUE(requantConflicts(g)); // run pass

        std::string requantGuid;
        ASSERT_TRUE(dTypetoRequantGuid(dType, requantGuid));
        pTensor requant0Output = concat2Node->getInput(0);
        pNode   requant0Node   = g.getTensorProducer(requant0Output);
        ASSERT_EQ(requant0Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
        ASSERT_EQ(requant0Output->getQuantizationParams().scale(0), 0.5)
            << "Error: Incorrect scale used for tensor " << requant0Output->getName();
        ASSERT_EQ(requant0Output->getQuantizationParams().zp(0), 0)
            << "Error: Incorrect zero point used for tensor " << requant0Output->getName();

        pTensor requant1Output = concat3Node->getInput(0);
        pNode   requant1Node   = g.getTensorProducer(requant1Output);
        ASSERT_EQ(requant1Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
        ASSERT_EQ(requant1Output->getQuantizationParams().scale(0), 0.3)
            << "Error: Incorrect scale used for tensor " << requant1Output->getName();
        ASSERT_EQ(requant1Output->getQuantizationParams().zp(0), 0)
            << "Error: Incorrect zero point used for tensor " << requant1Output->getName();
    }

    void  create_backward_forward_conflicted_model(synDataType dType, bool sameScale = false)
    {
        /*
              t2[]  _C_     _S_
                  \[   ] _ [   ]/
        t1   _S_  /[___]   [___]\  _C_   t12
        []->[   ]-  _C_     _S_   [___]->[]
            [   ] \[   ] _ [   ]/
                  /[___]   [___]\
              t3[]

        */

        GaudiGraph g;

        const TSize n     = 1;
        const TSize w     = 3;
        const TSize h     = 3;
        const TSize batch = 1;
        float in1[n * w * h * batch] = {1, 3, 5, 7, 2, 4, 6, 8, 10};
        float in2[n * w * 2 * h * batch];
        float in3[n * w * 2 * h * batch];
        float in4[n * w * h * batch];
        float in5[n * w * 2 * h * batch];
        float in6[n * w * h * batch];
        float in7[n * w * 2 * h * batch];
        float in8[n * w * h * batch];
        float in9[n * w * h * batch];
        float in10[n * w * h * batch];
        float in11[n * w * h * batch];
        float in12[n * w * 2 * h * batch];

        const TSize sizes1[] = {n, w, h, batch};
        const TSize sizes2[] = {n, w*2, h, batch};

        pTensor t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;
        create4DTensor(t1, "t1", sizes2, dType,  0, (sameScale? 1 : 0.1), in1);
        create4DTensor(t2, "t2", sizes1, dType,  0, (sameScale? 1 : 0.2), in2);
        create4DTensor(t3, "t3", sizes1, dType,  0, (sameScale? 1 : 0.3), in3);
        create4DTensor(t4, "t4", sizes1, dType,  0, (sameScale? 1 : 0.4), in4);
        create4DTensor(t5, "t5", sizes1, dType,  0, (sameScale? 1 : 0.5), in5);
        create4DTensor(t6, "t6", sizes2, dType,  0, (sameScale? 1 : 0.6), in6);
        create4DTensor(t7, "t7", sizes2, dType,  0, (sameScale? 1 : 0.7), in7);
        create4DTensor(t8, "t8", sizes1, dType,  0, (sameScale? 1 : 0.8), in8);
        create4DTensor(t9, "t9", sizes1, dType,  0, (sameScale? 1 : 0.9), in9);
        create4DTensor(t10, "t10", sizes1, dType,  0, 1, in10);
        create4DTensor(t11, "t11", sizes1, dType,  0,(sameScale? 1 : 1.1), in11);
        create4DTensor(t12, "t12", sizes2, dType,  0, (sameScale? 1 : 1.2), in12);

        DynamicRange dynamicRange;
        dynamicRange.min   = 0;
        dynamicRange.max   = 1;
        dynamicRange.isSet = true;

        t1->setDynamicRange(dynamicRange);
        t2->setDynamicRange(dynamicRange);
        t3->setDynamicRange(dynamicRange);
        t4->setDynamicRange(dynamicRange);
        t5->setDynamicRange(dynamicRange);
        t6->setDynamicRange(dynamicRange);
        t7->setDynamicRange(dynamicRange);
        t8->setDynamicRange(dynamicRange);
        t9->setDynamicRange(dynamicRange);
        t10->setDynamicRange(dynamicRange);
        t11->setDynamicRange(dynamicRange);
        t12->setDynamicRange(dynamicRange);

        unsigned splitParamsTop = 1;
        pNode splitTopNode = NodeFactory::createNode({t1}, {t4, t5}, &splitParamsTop, "split", "split_top");

        unsigned concat1Params = 1;
        pNode concat1Node = NodeFactory::createNode({t4, t2}, {t6}, &concat1Params, "concat", "concat1");
        unsigned concat2Params = 1;
        pNode concat2Node = NodeFactory::createNode({t5, t3}, {t7}, &concat2Params, "concat", "concat2");

        unsigned splitParams1 = 1;
        pNode split1Node = NodeFactory::createNode({t6}, {t8, t9}, &splitParams1, "split", "split1");
        unsigned split2Params = 1;
        pNode split2Node = NodeFactory::createNode({t7}, {t10, t11}, &split2Params, "split", "split2");

        unsigned concat3Params = 1;
        pNode concat3Node = NodeFactory::createNode({t8, t10}, {t12}, &concat3Params, "concat", "concat3");

        GraphEditor::addNode(g, splitTopNode);
        GraphEditor::addNode(g, concat1Node);
        GraphEditor::addNode(g, concat2Node);
        GraphEditor::addNode(g, concat3Node);
        GraphEditor::addNode(g, split1Node);
        GraphEditor::addNode(g, split2Node);

        ASSERT_TRUE(adjustScales(g)); // run pass
        ASSERT_TRUE(requantConflicts(g)); // run pass

        if (sameScale)
        {
            const NodeVector& sortedNodes = g.getExeSortedNodes();
            auto it = std::find_if(sortedNodes.begin(), sortedNodes.end(), isRequantNode);
            ASSERT_EQ(it, sortedNodes.end()) << "Error: Requant Node was found in the graph!";
        }
        else
        {
            std::string requantGuid;
            ASSERT_TRUE(dTypetoRequantGuid(dType, requantGuid));
            pTensor requant0Output = concat1Node->getInput(0);
            pNode requant0Node = g.getTensorProducer(requant0Output);
            ASSERT_EQ(requant0Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
            ASSERT_EQ(requant0Output->getQuantizationParams().scale(0), 0.6)
                                        << "Error: Incorrect scale used for tensor " << requant0Output->getName();
            ASSERT_EQ(requant0Output->getQuantizationParams().zp(0), 0)
                                        << "Error: Incorrect zero point used for tensor " << requant0Output->getName();

            pTensor requant1Output = concat2Node->getInput(0);
            pNode requant1Node = g.getTensorProducer(requant1Output);
            ASSERT_EQ(requant1Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
            ASSERT_EQ(requant1Output->getQuantizationParams().scale(0), 0.7)
                                        << "Error: Incorrect scale used for tensor " << requant1Output->getName();
            ASSERT_EQ(requant1Output->getQuantizationParams().zp(0), 0)
                                        << "Error: Incorrect zero point used for tensor " << requant1Output->getName();

            pTensor requant2Output = concat3Node->getInput(0);
            pNode requant2Node = g.getTensorProducer(requant2Output);
            ASSERT_EQ(requant2Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
            ASSERT_EQ(requant2Output->getQuantizationParams().scale(0), 1.2)
                                        << "Error: Incorrect scale used for tensor " << requant2Output->getName();
            ASSERT_EQ(requant2Output->getQuantizationParams().zp(0), 0)
                                        << "Error: Incorrect zero point used for tensor " << requant2Output->getName();

            pTensor requant3Output = concat3Node->getInput(1);
            pNode requant3Node = g.getTensorProducer(requant3Output);
            ASSERT_EQ(requant3Node->getGUID(), requantGuid) << "Error: Missing " << requantGuid << " node";
            ASSERT_EQ(requant3Output->getQuantizationParams().scale(0), 1.2)
                                        << "Error: Incorrect scale used for tensor " << requant3Output->getName();
            ASSERT_EQ(requant3Output->getQuantizationParams().zp(0), 0)
                                        << "Error: Incorrect zero point used for tensor " << requant3Output->getName();
        }
    }
};

TEST_F(RequantTest, DISABLED_lockAncestorsForRequant)
{
    /**
     * relu1 -> relu2 -> add
     *     \----------/
     * first relu output tensor should be locked after the pass
     */
    GaudiGraph g;
    pTensor   input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor   relu1  = std::make_shared<Tensor>(syn_type_int8);
    pTensor   relu2  = std::make_shared<Tensor>(syn_type_int8);
    pTensor   output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {relu1},
                                          nullptr, NodeFactory::reluNodeTypeName, "relu1");

    pNode node2 = NodeFactory::createNode({relu1}, {relu2},
                                          nullptr, NodeFactory::reluNodeTypeName, "relu2");

    pNode node3 = NodeFactory::createNode({relu2, relu1}, {output},
                                          nullptr, NodeFactory::addNodeTypeName, "add");

    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    GraphEditor::addNode(g, node3);
    lockAncestorsForRequant(g);

    ASSERT_TRUE(relu1->isLocked());
    ASSERT_FALSE(relu2->isLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(input->isLocked());
}

TEST_F(RequantTest, DISABLED_test_forward_conflict)
{
    create_forward_conflicted_model(syn_type_int16);
}

TEST_F(RequantTest, DISABLED_test_forward_conflict_fp32)
{
    create_forward_conflicted_model(syn_type_single);
}

TEST_F(RequantTest, DISABLED_test_backward_conflict)
{
    create_backward_conflicted_model(syn_type_int16);
}

TEST_F(RequantTest, DISABLED_test_simple_backward_conflict)
{
    create_simple_backward_conflicted_model(syn_type_int16);
}

TEST_F(RequantTest, DISABLED_test_backward_conflict_fp32)
{
    create_backward_conflicted_model(syn_type_single);
}

TEST_F(RequantTest, DISABLED_test_requant_conflicts_same_scale_backward_one_different)
{
    create_requant_conflicts_same_scale_backward_one_different_model(syn_type_int16);
}

TEST_F(RequantTest, DISABLED_test_requant_backward_forward_conflict)
{
    create_backward_forward_conflicted_model(syn_type_int16, false);
}

TEST_F(RequantTest, DISABLED_test_requant_conflicts_same_scale)
{
    create_backward_forward_conflicted_model(syn_type_int16, true);
}

TEST_F(RequantTest, DISABLED_test_requant_ancestor)
{
    create_ancestor_input_model(syn_type_int16);
}

TEST_F(RequantTest, DISABLED_test_requant_ancestor_dontcare)
{
    create_ancestor_dontcare_input_model(syn_type_int8);
}
