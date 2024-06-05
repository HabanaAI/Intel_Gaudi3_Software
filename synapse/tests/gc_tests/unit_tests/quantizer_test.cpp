#include <gtest/gtest.h>
#include "utils.h"
#include "quantizer.h"
#include "graph_optimizer_test.h"
#include "gaudi_graph.h"
#include "synapse_common_types.h"
#include "node_factory.h"
#include "quantization_data.h"
#include <map>
class QuantizerTestSubject : public Quantizer
{
public:
    bool run_isConflictedWith(pTensor tensor, QuantizationMap& newQuant)
    {
        return isConflictedWith(tensor, newQuant);
    }

    bool run_isQuantMapEmpty(QuantizationMap& quantMap)
    {
        return isQuantMapEmpty(quantMap);
    }

    bool run_isAllLocked(TensorVector& tensors)
    {
        return isAllLocked(tensors);
    }

    QuantizationMap run_getSingleScaleFromTensors(TensorVector& tensors, int index=Quantizer::index_na)
    {
        return getSingleScaleFromTensors(tensors, index);
    }

    void run_lockTensors(pNode node, TensorVector& tensors)
    {
        lockTensors(node, tensors);
    }

    void run_setInputScale(HabanaGraph& g, pNode node, QuantizationMap& quantInfo,
                           std::vector<uint32_t> numSuccessorsPerInput, int index=Quantizer::index_na)
    {
        setInputScale(g, node, quantInfo, numSuccessorsPerInput, index);
    }

    void run_setOutputScale(pNode node, QuantizationMap& quantInfo, int index=Quantizer::index_na)
    {
        setOutputScale(node, quantInfo, index);
    }
};

class QuantizerTest : public GraphOptimizerTest
{
protected:
    virtual void SetUp()
    {
        GraphOptimizerTest::SetUp();
    }

    virtual void TearDown()
    {
        GraphOptimizerTest::TearDown();
    }
};

TEST_F(QuantizerTest, quantConflicts_NoConflict)
{
    QuantizerTestSubject qts;
    QuantizationMap a, b;
    a[quant_type_int8] = QuantizationData(quant_type_int8);
    a[quant_type_int16] = QuantizationData(quant_type_int16);
    b[quant_type_int8] = QuantizationData(quant_type_int8);
    b[quant_type_int16] = QuantizationData(quant_type_int16);
    pTensor tensor      = std::make_shared<Tensor>(syn_type_int8);
    tensor->setAllQuantizationParams(a);

    ASSERT_FALSE(qts.run_isConflictedWith(tensor, b));
}

TEST_F(QuantizerTest, quantConflicts_DifferentLength)
{
    QuantizerTestSubject qts;
    QuantizationMap a, b;
    a[quant_type_int8] = QuantizationData(quant_type_int8);
    a[quant_type_int16] = QuantizationData(quant_type_int16);
    b[quant_type_int8] = QuantizationData(quant_type_int8);
    pTensor tensor      = std::make_shared<Tensor>(syn_type_int8);
    tensor->setAllQuantizationParams(a);

    ASSERT_TRUE(qts.run_isConflictedWith(tensor, b));
}

TEST_F(QuantizerTest, quantConflicts_DifferentQuantData)
{
    QuantizerTestSubject qts;
    QuantizationMap a, b;
    a[quant_type_int8] = QuantizationData(quant_type_int8);
    a[quant_type_int16] = QuantizationData(quant_type_int16);
    b[quant_type_int8] = QuantizationData(quant_type_int8);
    b[quant_type_int16] = QuantizationData(quant_type_int16);
    b[quant_type_int16].setScale(0.5);
    pTensor tensor = std::make_shared<Tensor>(syn_type_int8);
    tensor->setAllQuantizationParams(a);

    ASSERT_TRUE(qts.run_isConflictedWith(tensor, b));
}

TEST_F(QuantizerTest, isQuantMapEmpty_Positive)
{
    QuantizerTestSubject qts;
    QuantizationMap a;
    ASSERT_TRUE(qts.run_isQuantMapEmpty(a));
}

TEST_F(QuantizerTest, isQuantMapEmpty_Negative)
{
    QuantizerTestSubject qts;
    QuantizationMap a;
    a[quant_type_int16].setScale(0.5);
    ASSERT_FALSE(qts.run_isQuantMapEmpty(a));
}

TEST_F(QuantizerTest, isAllLocked_Positive)
{
    GaudiGraph           g;
    QuantizerTestSubject qts;
    pTensor              a    = std::make_shared<Tensor>(syn_type_int8);
    pTensor              b    = std::make_shared<Tensor>(syn_type_int8);
    pNode node = NodeFactory::createNode({a}, {b},
                                         nullptr, NodeFactory::reluNodeTypeName, "node");
    a->lockQuantization(node);
    b->lockQuantization(node);
    TensorVector tv = {a, b};
    ASSERT_TRUE(qts.run_isAllLocked(tv));
}

TEST_F(QuantizerTest, isAllLocked_Negative1)
{
    GaudiGraph           g;
    QuantizerTestSubject qts;
    pTensor              a    = std::make_shared<Tensor>(syn_type_int8);
    pTensor              b    = std::make_shared<Tensor>(syn_type_int8);
    pNode node = NodeFactory::createNode({a}, {b},
                                         nullptr, NodeFactory::reluNodeTypeName, "node");
    a->lockQuantization(node);
    TensorVector tv = {a, b};
    ASSERT_FALSE(qts.run_isAllLocked(tv));
}

TEST_F(QuantizerTest, isAllLocked_Negative2)
{
    QuantizerTestSubject qts;
    pTensor              a  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              b  = std::make_shared<Tensor>(syn_type_int8);
    TensorVector tv = {a, b};
    ASSERT_FALSE(qts.run_isAllLocked(tv));
}

TEST_F(QuantizerTest, getSingleScaleFromTensors_NoIndex_Positive1)
{
    QuantizerTestSubject qts;
    QuantizationMap qa;
    qa[quant_type_uint8] = QuantizationData(quant_type_uint8);
    qa[quant_type_uint8].setScale(0.5);
    qa[quant_type_uint8].m_isUserQuantInfo = true;
    QuantizationMap qb;
    qb[quant_type_int16] = QuantizationData(quant_type_int16);
    qb[quant_type_int16].setScale(0.2);
    qb[quant_type_int16].m_isUserQuantInfo = true;
    pTensor a                              = std::make_shared<Tensor>(syn_type_int8);
    a->setElementType(syn_type_uint8);
    a->setAllQuantizationParams(qa);
    pTensor b = std::make_shared<Tensor>(syn_type_int8);
    b->setElementType(syn_type_int16);
    b->setAllQuantizationParams(qb);
    TensorVector tv = {a, b};
    QuantizationMap singleScale = qts.run_getSingleScaleFromTensors(tv);
    ASSERT_TRUE(singleScale[quant_type_uint8] == qa[quant_type_uint8]);
    ASSERT_TRUE(singleScale[quant_type_int16] == qb[quant_type_int16]);
}

TEST_F(QuantizerTest, getSingleScaleFromTensors_NoIndex_Positive2)
{
    QuantizerTestSubject qts;
    QuantizationMap qa;
    qa[quant_type_uint8] = QuantizationData(quant_type_uint8);
    qa[quant_type_uint8].setScale(0.5);
    qa[quant_type_uint8].m_isUserQuantInfo = true;
    QuantizationMap qb;
    qb[quant_type_uint8] = QuantizationData(quant_type_uint8);
    qb[quant_type_uint8].setScale(0.5);
    qb[quant_type_uint8].m_isUserQuantInfo = true;
    pTensor a                              = std::make_shared<Tensor>(syn_type_int8);
    a->setElementType(syn_type_uint8);
    a->setAllQuantizationParams(qa);
    pTensor b = std::make_shared<Tensor>(syn_type_int8);
    b->setElementType(syn_type_uint8);
    b->setAllQuantizationParams(qb);
    TensorVector tv = {a, b};
    QuantizationMap singleScale = qts.run_getSingleScaleFromTensors(tv);
    ASSERT_TRUE(singleScale[quant_type_uint8] == qa[quant_type_uint8]);
    ASSERT_TRUE(singleScale[quant_type_uint8] == qb[quant_type_uint8]);
}

TEST_F(QuantizerTest, getSingleScaleFromTensors_NoIndex_Override)
{
    QuantizerTestSubject qts;
    QuantizationMap qa;
    qa[quant_type_uint8] = QuantizationData(quant_type_uint8);
    qa[quant_type_uint8].setScale(0.5);
    qa[quant_type_uint8].m_isUserQuantInfo = true;
    QuantizationMap qb;
    qb[quant_type_uint8] = QuantizationData(quant_type_uint8);
    qb[quant_type_uint8].setScale(0.2);
    qb[quant_type_uint8].m_isUserQuantInfo = true;
    pTensor a                              = std::make_shared<Tensor>(syn_type_int8);
    a->setElementType(syn_type_uint8);
    a->setAllQuantizationParams(qa);
    pTensor b = std::make_shared<Tensor>(syn_type_int8);
    b->setElementType(syn_type_uint8);
    b->setAllQuantizationParams(qb);
    TensorVector tv = {a, b};
    QuantizationMap singleScale = qts.run_getSingleScaleFromTensors(tv);
    ASSERT_TRUE(singleScale[quant_type_uint8] == qb[quant_type_uint8]);
}

TEST_F(QuantizerTest, getSingleScaleFromTensors_Index0)
{
    QuantizerTestSubject qts;
    QuantizationMap qa;
    qa[quant_type_uint8] = QuantizationData(quant_type_uint8);
    qa[quant_type_uint8].setScale(0.5);
    qa[quant_type_uint8].m_isUserQuantInfo = true;
    QuantizationMap qb;
    qb[quant_type_uint8] = QuantizationData(quant_type_uint8);
    qb[quant_type_uint8].setScale(0.2);
    qb[quant_type_uint8].m_isUserQuantInfo = true;
    pTensor a                              = std::make_shared<Tensor>(syn_type_int8);
    a->setElementType(syn_type_uint8);
    a->setAllQuantizationParams(qa);
    pTensor b = std::make_shared<Tensor>(syn_type_int8);
    b->setElementType(syn_type_uint8);
    b->setAllQuantizationParams(qb);
    TensorVector tv = {a, b};
    QuantizationMap singleScale = qts.run_getSingleScaleFromTensors(tv, 0);
    ASSERT_TRUE(singleScale[quant_type_uint8] == qa[quant_type_uint8]);
}

TEST_F(QuantizerTest, getSingleScaleFromTensors_Index1)
{
    QuantizerTestSubject qts;
    QuantizationMap qa;
    qa[quant_type_uint8] = QuantizationData(quant_type_uint8);
    qa[quant_type_uint8].setScale(0.5);
    qa[quant_type_uint8].m_isUserQuantInfo = true;
    QuantizationMap qb;
    qb[quant_type_uint8] = QuantizationData(quant_type_uint8);
    qb[quant_type_uint8].setScale(0.2);
    qb[quant_type_uint8].m_isUserQuantInfo = true;
    pTensor a                              = std::make_shared<Tensor>(syn_type_int8);
    a->setElementType(syn_type_uint8);
    a->setAllQuantizationParams(qa);
    pTensor b = std::make_shared<Tensor>(syn_type_int8);
    b->setElementType(syn_type_uint8);
    b->setAllQuantizationParams(qb);
    TensorVector tv = {a, b};
    QuantizationMap singleScale = qts.run_getSingleScaleFromTensors(tv, 1);
    ASSERT_TRUE(singleScale[quant_type_uint8] == qb[quant_type_uint8]);
}

TEST_F(QuantizerTest, lockTensors)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              a    = std::make_shared<Tensor>(syn_type_int8);
    pTensor              b    = std::make_shared<Tensor>(syn_type_int8);
    pNode node = NodeFactory::createNode({a}, {b},
                                         nullptr, NodeFactory::reluNodeTypeName, "node");
    TensorVector tv = {a, b};
    qts.run_lockTensors(node, tv);
    ASSERT_TRUE(a->isLocked());
    ASSERT_TRUE(b->isLocked());
}

TEST_F(QuantizerTest, setInputScale_SingleInputNoConflict)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    q[quant_type_int8] = QuantizationData(quant_type_int8);
    q[quant_type_int8].setScale(0.5);
    qts.run_setInputScale(g, node2, q, {1});
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, q));
}

TEST_F(QuantizerTest, setInputScale_SingleInputConflict_InputGraph)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle}, nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output}, nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    input->setMeasuredQuantizationParams(mq);
    input->lockQuantization(node1);
    QuantizationMap q;
    q[quant_type_int8] = QuantizationData(quant_type_int8);
    q[quant_type_int8].setScale(0.5);
    qts.run_setInputScale(g, node1, q, {1});
    ASSERT_TRUE(input->isLocked());
    ASSERT_TRUE(input->isRequantLocked());
    ASSERT_FALSE(middle->isLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(input, mq));
}

TEST_F(QuantizerTest, setInputScale_SingleInputConflict)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    middle->lockQuantization(node1);
    QuantizationMap q;
    q[quant_type_int8] = QuantizationData(quant_type_int8);
    q[quant_type_int8].setScale(0.5);
    qts.run_setInputScale(g, node2, q, {1});
    ASSERT_TRUE(middle->isLocked());
    ASSERT_TRUE(middle->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, mq));
}

TEST_F(QuantizerTest, setInputScale_TwoInputsSuccessorsRequant)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              input1 = std::make_shared<Tensor>(syn_type_int8);
    pTensor              input2 = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);

    pNode relu = NodeFactory::createNode({input1}, {input2},
                                          nullptr, NodeFactory::reluNodeTypeName, "relu");

    pNode node1 = NodeFactory::createNode({input2}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle, input2}, {output},
                                          nullptr, NodeFactory::addNodeTypeName, "node2");
    GraphEditor::addNode(g, relu);
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    input2->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    q[quant_type_int8] = QuantizationData(quant_type_int8);
    q[quant_type_int8].setScale(0.5);
    qts.run_setInputScale(g, node2, q, {1, 2});
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(input2->isLocked());
    ASSERT_TRUE(input2->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, q));
    ASSERT_FALSE(qts.run_isConflictedWith(input2, mq));
}

TEST_F(QuantizerTest, setInputScale_TwoInputsNoRequant)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              input   = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle2 = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output  = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle, middle2}, {output},
                                          nullptr, NodeFactory::addNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    middle2->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    q[quant_type_int8] = QuantizationData(quant_type_int8);
    q[quant_type_int8].setScale(0.5);
    qts.run_setInputScale(g, node2, q, {1, 1});
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(middle2->isLocked());
    ASSERT_FALSE(middle2->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, q));
    ASSERT_FALSE(qts.run_isConflictedWith(middle2, q));
}

TEST_F(QuantizerTest, setInputScale_TwoInputsIndex1)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              input   = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle2 = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output  = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle, middle2}, {output},
                                          nullptr, NodeFactory::addNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    middle2->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    q[quant_type_int8] = QuantizationData(quant_type_int8);
    q[quant_type_int8].setScale(0.5);
    qts.run_setInputScale(g, node2, q, {1, 1}, 1);
    ASSERT_FALSE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(middle2->isLocked());
    ASSERT_FALSE(middle2->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, mq));
    ASSERT_FALSE(qts.run_isConflictedWith(middle2, q));
}

TEST_F(QuantizerTest, setInputScale_TwoInputsEmptyMap)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              input   = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle2 = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output  = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle, middle2}, {output},
                                          nullptr, NodeFactory::addNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    middle2->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    qts.run_setInputScale(g, node2, q, {1, 1});
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(middle2->isLocked());
    ASSERT_FALSE(middle2->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, mq));
    ASSERT_FALSE(qts.run_isConflictedWith(middle2, mq));
}

TEST_F(QuantizerTest, setOutputScale_SingleOutputNoConflict)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    output->setMeasuredQuantizationParams(mq);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    q[quant_type_int8] = QuantizationData(quant_type_int8);
    q[quant_type_int8].setScale(0.5);
    qts.run_setOutputScale(node2, q);
    ASSERT_FALSE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_TRUE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(output, q));
}

TEST_F(QuantizerTest, setOutputScale_SingleOutputLockedConflict)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    output->setMeasuredQuantizationParams(mq);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    q[quant_type_int8] = QuantizationData(quant_type_int8);
    q[quant_type_int8].setScale(0.5);
    middle->lockQuantization(node1);
    qts.run_setOutputScale(node1, q);
    auto conflicts = middle->getConflictingQuants();
    ASSERT_TRUE(conflicts[node1->getId()] == mq);
    ASSERT_TRUE(middle->isLocked());
    ASSERT_TRUE(middle->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, q));
}

TEST_F(QuantizerTest, setOutputScale_SingleOutputRequantLockedConflict)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    output->setMeasuredQuantizationParams(mq);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    q[quant_type_int8] = QuantizationData(quant_type_int8);
    q[quant_type_int8].setScale(0.5);
    middle->requantLock(node1);
    qts.run_setOutputScale(node1, q);
    ASSERT_TRUE(middle->isLocked());
    ASSERT_TRUE(middle->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, q));
}

TEST_F(QuantizerTest, setOutputScale_SingleOutputEmptyMap)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    output->setMeasuredQuantizationParams(mq);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    qts.run_setOutputScale(node1, q);
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, mq));
}

TEST_F(QuantizerTest, setOutputScale_TwoOutputsIndex1)
{
    QuantizerTestSubject qts;
    GaudiGraph           g;
    unsigned dim = 0;
    TSize sizes1[] = {2, 1, 1, 1, 1};
    TSize sizes2[] = {1, 1, 1, 1, 1};
    pTensor input = std::make_shared<Tensor>(2, sizes1, syn_type_int8);
    pTensor middle = std::make_shared<Tensor>(2, sizes2, syn_type_int8);
    pTensor middle2 = std::make_shared<Tensor>(2, sizes2, syn_type_int8);
    pTensor output = std::make_shared<Tensor>(2, sizes2, syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle, middle2},
                                          &dim, NodeFactory::splitNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    output->setMeasuredQuantizationParams(mq);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    q[quant_type_int8] = QuantizationData(quant_type_int8);
    q[quant_type_int8].setScale(0.5);
    middle2->requantLock(node1);
    qts.run_setOutputScale(node1, q, 1);
    ASSERT_TRUE(middle2->isLocked());
    ASSERT_TRUE(middle2->isRequantLocked());
    ASSERT_FALSE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, mq));
    ASSERT_FALSE(qts.run_isConflictedWith(middle2, q));
}

TEST_F(QuantizerTest, Quantizer_scales_b)
{
    QuantizerTestSubject qts;
    Quantizer rqts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0};
    rqts.adjustScales(g, node1, false, successors);
    ASSERT_FALSE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
}

TEST_F(QuantizerTest, Quantizer_scales_f)
{
    QuantizerTestSubject qts;
    Quantizer rqts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0};
    rqts.adjustScales(g, node1, true, successors);
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
}

TEST_F(QuantizerTest, backwardQuantizer_scales_b)
{
    QuantizerTestSubject qts;
    BackwardQuantizer rqts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    mq[quant_type_int8].m_isUserQuantInfo = true;
    middle->setMeasuredQuantizationParams(mq);
    middle->setElementType(syn_type_int8);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0};
    rqts.adjustScales(g, node1, false, successors);
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(input, mq));
}

TEST_F(QuantizerTest, backwardQuantizer_scales_f)
{
    QuantizerTestSubject qts;
    BackwardQuantizer rqts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0};
    rqts.adjustScales(g, node1, true, successors);
    ASSERT_FALSE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_TRUE(qts.run_isConflictedWith(input, mq));
}

TEST_F(QuantizerTest, forwardQuantizer_scales_b)
{
    QuantizerTestSubject qts;
    ForwardQuantizer rqts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    input->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0};
    rqts.adjustScales(g, node1, false, successors);
    ASSERT_FALSE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_TRUE(qts.run_isConflictedWith(middle, mq));
}

TEST_F(QuantizerTest, forwardQuantizer_scales_f)
{
    QuantizerTestSubject qts;
    ForwardQuantizer rqts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    mq[quant_type_int8].m_isUserQuantInfo = true;
    input->setMeasuredQuantizationParams(mq);
    input->setElementType(syn_type_int8);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0};
    rqts.adjustScales(g, node1, true, successors);
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, mq));
}

TEST_F(QuantizerTest, forwardQuantizer_scales_specific_f)
{
    QuantizerTestSubject qts;
    ForwardQuantizer rqts(0);
    GaudiGraph           g;
    pTensor              input1 = std::make_shared<Tensor>(syn_type_int8);
    pTensor              input2 = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input1, input2}, {middle},
                                          nullptr, NodeFactory::addNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    mq[quant_type_int8].m_isUserQuantInfo = true;
    input1->setMeasuredQuantizationParams(mq);
    input1->setElementType(syn_type_int8);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0, 0};
    rqts.adjustScales(g, node1, true, successors);
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(input1->isLocked());
    ASSERT_FALSE(input1->isRequantLocked());
    ASSERT_TRUE(input2->isLocked());
    ASSERT_FALSE(input2->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(middle, mq));
    ASSERT_TRUE(qts.run_isConflictedWith(input2, mq));
}

TEST_F(QuantizerTest, backwardDontCareQuantizer_scales_b)
{
    QuantizerTestSubject qts;
    BackwardDontCareQuantizer rqts;
    GaudiGraph                g;
    pTensor                   input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor                   middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor                   output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0};
    rqts.adjustScales(g, node1, false, successors);
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
}

TEST_F(QuantizerTest, backwardDontCareQuantizer_scales_f)
{
    QuantizerTestSubject qts;
    BackwardDontCareQuantizer rqts;
    GaudiGraph                g;
    pTensor                   input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor                   middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor                   output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0};
    rqts.adjustScales(g, node1, true, successors);
    ASSERT_FALSE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
}

TEST_F(QuantizerTest, SelectiveBackwardQuantizer_scales_f)
{
    QuantizerTestSubject qts;
    std::map<uint32_t, uint32_t> m;
    m[0] = 1;
    SelectiveBackwardQuantizer rqts(m);
    GaudiGraph                 g;
    pTensor                    input1 = std::make_shared<Tensor>(syn_type_int8);
    pTensor                    input2 = std::make_shared<Tensor>(syn_type_int8);
    pTensor                    middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor                    output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input1, input2}, {middle},
                                          nullptr, NodeFactory::addNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0, 0};
    rqts.adjustScales(g, node1, true, successors);
    ASSERT_FALSE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(input1->isLocked());
    ASSERT_FALSE(input1->isRequantLocked());
    ASSERT_TRUE(input2->isLocked());
    ASSERT_FALSE(input2->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_TRUE(qts.run_isConflictedWith(input1, mq));
    ASSERT_TRUE(qts.run_isConflictedWith(input2, mq));
}

TEST_F(QuantizerTest, SelectiveBackwardQuantizer_scales_b)
{
    QuantizerTestSubject qts;
    std::map<uint32_t, uint32_t> m;
    m[0] = 1;
    SelectiveBackwardQuantizer rqts(m);
    GaudiGraph                 g;
    pTensor                    input1 = std::make_shared<Tensor>(syn_type_int8);
    pTensor                    input2 = std::make_shared<Tensor>(syn_type_int8);
    pTensor                    middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor                    output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input1, input2}, {middle},
                                          nullptr, NodeFactory::addNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    mq[quant_type_int8].m_isUserQuantInfo = true;
    middle->setMeasuredQuantizationParams(mq);
    middle->setElementType(syn_type_int8);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0, 0};
    rqts.adjustScales(g, node1, false, successors);
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(input2->isLocked());
    ASSERT_FALSE(input2->isRequantLocked());
    ASSERT_FALSE(input1->isLocked());
    ASSERT_FALSE(input1->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_TRUE(qts.run_isConflictedWith(input1, mq));
    ASSERT_FALSE(qts.run_isConflictedWith(input2, mq));
}

TEST_F(QuantizerTest, AlignInputsQuantizer_scales_f)
{
    QuantizerTestSubject qts;
    AlignInputsQuantizer rqts;
    GaudiGraph           g;
    pTensor              input1 = std::make_shared<Tensor>(syn_type_int8);
    pTensor              input2 = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input1, input2}, {middle},
                                          nullptr, NodeFactory::addNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq1, mq2;
    mq1[quant_type_int8] = QuantizationData(quant_type_int8);
    mq1[quant_type_int8].setScale(0.2);
    input1->setMeasuredQuantizationParams(mq1);
    mq2[quant_type_int8] = QuantizationData(quant_type_int8);
    mq2[quant_type_int8].setScale(0.5);
    input2->setMeasuredQuantizationParams(mq2);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0, 0};
    rqts.adjustScales(g, node1, true, successors);
    ASSERT_FALSE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_FALSE(input1->isLocked());
    ASSERT_FALSE(input1->isRequantLocked());
    ASSERT_FALSE(input2->isLocked());
    ASSERT_FALSE(input2->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_FALSE(qts.run_isConflictedWith(input1, mq1));
    ASSERT_FALSE(qts.run_isConflictedWith(input2, mq2));
}

TEST_F(QuantizerTest, AlignInputsQuantizer_scales_b)
{
    QuantizerTestSubject qts;
    AlignInputsQuantizer rqts;
    GaudiGraph           g;
    pTensor              input1 = std::make_shared<Tensor>(syn_type_int8);
    pTensor              input2 = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input1, input2}, {middle},
                                          nullptr, NodeFactory::addNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    QuantizationMap mq1, mq2;
    mq1[quant_type_int8] = QuantizationData(quant_type_int8);
    mq1[quant_type_int8].setScale(0.2);
    input1->setMeasuredQuantizationParams(mq1);
    mq2[quant_type_int8] = QuantizationData(quant_type_int8);
    mq2[quant_type_int8].setScale(0.5);
    input2->setMeasuredQuantizationParams(mq2);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0, 0};
    rqts.adjustScales(g, node1, false, successors);
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(input2->isLocked());
    ASSERT_FALSE(input2->isRequantLocked());
    ASSERT_TRUE(input1->isLocked());
    ASSERT_FALSE(input1->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
    ASSERT_TRUE(qts.run_isConflictedWith(middle, mq2));
    ASSERT_FALSE(qts.run_isConflictedWith(input1, mq2));
    ASSERT_FALSE(qts.run_isConflictedWith(input2, mq2));
}

TEST_F(QuantizerTest, dontCareQuantizer_scales_b)
{
    QuantizerTestSubject qts;
    DontCareQuantizer rqts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0};
    rqts.adjustScales(g, node1, false, successors);
    ASSERT_FALSE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_FALSE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
}

TEST_F(QuantizerTest, dontCareQuantizer_scales_f)
{
    QuantizerTestSubject qts;
    DontCareQuantizer rqts;
    GaudiGraph           g;
    pTensor              input  = std::make_shared<Tensor>(syn_type_int8);
    pTensor              middle = std::make_shared<Tensor>(syn_type_int8);
    pTensor              output = std::make_shared<Tensor>(syn_type_int8);
    pNode node1 = NodeFactory::createNode({input}, {middle},
                                          nullptr, NodeFactory::reluNodeTypeName, "node1");

    pNode node2 = NodeFactory::createNode({middle}, {output},
                                          nullptr, NodeFactory::reluNodeTypeName, "node2");
    GraphEditor::addNode(g, node1);
    GraphEditor::addNode(g, node2);
    QuantizationMap mq;
    mq[quant_type_int8] = QuantizationData(quant_type_int8);
    mq[quant_type_int8].setScale(0.2);
    middle->setMeasuredQuantizationParams(mq);
    QuantizationMap q;
    std::vector<uint32_t> successors = {0};
    rqts.adjustScales(g, node1, true, successors);
    ASSERT_TRUE(middle->isLocked());
    ASSERT_FALSE(middle->isRequantLocked());
    ASSERT_TRUE(input->isLocked());
    ASSERT_FALSE(input->isRequantLocked());
    ASSERT_FALSE(output->isLocked());
    ASSERT_FALSE(output->isRequantLocked());
}
