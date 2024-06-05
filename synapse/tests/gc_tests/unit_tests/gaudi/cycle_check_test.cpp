#include "gtest/gtest.h"
#include "gc_tests/unit_tests/graph_optimizer_test.h"
#include "habana_pass.h"
#include "node.h"
#include "node_factory.h"
#include "gaudi_graph.h"
#include "perf_lib_layer_params.h"
#include "habana_nodes.h"
#include "graph_editor.h"
#include "test_utils.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"

class GaudiCheckCyclesTest : public GraphOptimizerTest
{
protected:
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    }
};

TEST_F(GaudiCheckCyclesTest, gaudi_check_cycles_data_test_no_cycle)
{
    GaudiGraph g;

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setTensorInSram();
    IN1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    OUT1->setTensorInSram();
    OUT1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");
    OUT2->setTensorInSram();
    OUT2->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setTensorInSram();
    OUT3->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    OUT4->setTensorInSram();
    OUT4->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");
    OUT5->setTensorInSram();
    OUT5->setMemoryDescriptor(persistentMemoryDesc);

    pNode neg1 = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, "neg_fwd_f32", "neg1");
    GraphEditor::addNode(g, neg1);
    pNode neg2 = NodeFactory::createGenericTPCNode({OUT1}, {OUT2}, nullptr, "neg_fwd_f32", "neg2");
    GraphEditor::addNode(g, neg2);
    pNode neg3 = NodeFactory::createGenericTPCNode({OUT2}, {OUT3}, nullptr, "neg_fwd_f32", "neg3");
    GraphEditor::addNode(g, neg3);
    pNode neg4 = NodeFactory::createGenericTPCNode({OUT3}, {OUT4}, nullptr, "neg_fwd_f32", "neg4");
    GraphEditor::addNode(g, neg4);
    pNode neg5 = NodeFactory::createGenericTPCNode({OUT4}, {OUT5}, nullptr, "neg_fwd_f32", "neg5");
    GraphEditor::addNode(g, neg5);

    ASSERT_FALSE(g.printGraphCycles());
}

TEST_F(GaudiCheckCyclesTest, gaudi_check_cycles_data_test_one_cycle)
{
    GaudiGraph g;

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setTensorInSram();
    IN1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    OUT1->setTensorInSram();
    OUT1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");
    OUT2->setTensorInSram();
    OUT2->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setTensorInSram();
    OUT3->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    OUT4->setTensorInSram();
    OUT4->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");
    OUT5->setTensorInSram();
    OUT5->setMemoryDescriptor(persistentMemoryDesc);

    pNode neg1 = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, "neg_fwd_f32", "neg1");
    GraphEditor::addNode(g, neg1);
    pNode neg2 = NodeFactory::createGenericTPCNode({OUT1}, {OUT2}, nullptr, "neg_fwd_f32", "neg2");
    GraphEditor::addNode(g, neg2);
    pNode add_c = NodeFactory::createGenericTPCNode({OUT2, OUT4}, {OUT3}, nullptr, "add_fwd_f32", "add_c");
    GraphEditor::addNode(g, add_c);
    pNode neg4 = NodeFactory::createGenericTPCNode({OUT3}, {OUT4}, nullptr, "neg_fwd_f32", "neg4");
    GraphEditor::addNode(g, neg4);
    pNode neg5 = NodeFactory::createGenericTPCNode({OUT4}, {OUT5}, nullptr, "neg_fwd_f32", "neg5");
    GraphEditor::addNode(g, neg5);

    ASSERT_TRUE(g.printGraphCycles());
}

TEST_F(GaudiCheckCyclesTest, gaudi_check_cycles_data_test_long_cycle)
{
    GaudiGraph g;

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setTensorInSram();
    IN1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    OUT1->setTensorInSram();
    OUT1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");
    OUT2->setTensorInSram();
    OUT2->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setTensorInSram();
    OUT3->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    OUT4->setTensorInSram();
    OUT4->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");
    OUT5->setTensorInSram();
    OUT5->setMemoryDescriptor(persistentMemoryDesc);

    pNode neg1 = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, "neg_fwd_f32", "neg1");
    GraphEditor::addNode(g, neg1);
    pNode add_c = NodeFactory::createGenericTPCNode({OUT1, OUT4}, {OUT2}, nullptr, "add_fwd_f32", "add_c");
    GraphEditor::addNode(g, add_c);
    pNode neg2 = NodeFactory::createGenericTPCNode({OUT2}, {OUT3}, nullptr, "neg_fwd_f32", "neg2");
    GraphEditor::addNode(g, neg2);
    pNode neg3 = NodeFactory::createGenericTPCNode({OUT3}, {OUT4}, nullptr, "neg_fwd_f32", "neg3");
    GraphEditor::addNode(g, neg3);
    pNode neg4 = NodeFactory::createGenericTPCNode({OUT4}, {OUT5}, nullptr, "neg_fwd_f32", "neg4");
    GraphEditor::addNode(g, neg4);

    ASSERT_TRUE(g.printGraphCycles());
}

TEST_F(GaudiCheckCyclesTest, gaudi_check_cycles_data_test_two_cycles)
{
    GaudiGraph g;

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setTensorInSram();
    IN1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    OUT1->setTensorInSram();
    OUT1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");
    OUT2->setTensorInSram();
    OUT2->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setTensorInSram();
    OUT3->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    OUT4->setTensorInSram();
    OUT4->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");
    OUT5->setTensorInSram();
    OUT5->setMemoryDescriptor(persistentMemoryDesc);

    pNode add_c1 = NodeFactory::createGenericTPCNode({IN1, OUT2}, {OUT1}, nullptr, "add_fwd_f32", "add_c1");
    GraphEditor::addNode(g, add_c1);
    pNode neg1 = NodeFactory::createGenericTPCNode({OUT1}, {OUT2}, nullptr, "neg_fwd_f32", "neg1");
    GraphEditor::addNode(g, neg1);
    pNode add_c2 = NodeFactory::createGenericTPCNode({OUT2, OUT4}, {OUT3}, nullptr, "add_fwd_f32", "add_c2");
    GraphEditor::addNode(g, add_c2);
    pNode neg2 = NodeFactory::createGenericTPCNode({OUT3}, {OUT4}, nullptr, "neg_fwd_f32", "neg2");
    GraphEditor::addNode(g, neg2);
    pNode neg3 = NodeFactory::createGenericTPCNode({OUT4}, {OUT5}, nullptr, "neg_fwd_f32", "neg3");
    GraphEditor::addNode(g, neg3);

    ASSERT_TRUE(g.printGraphCycles());
}

TEST_F(GaudiCheckCyclesTest, gaudi_check_cycles_data_test_ctrl_dep_cycle)
{
    GaudiGraph g;

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setTensorInSram();
    IN1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    OUT1->setTensorInSram();
    OUT1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");
    OUT2->setTensorInSram();
    OUT2->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setTensorInSram();
    OUT3->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    OUT4->setTensorInSram();
    OUT4->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");
    OUT5->setTensorInSram();
    OUT5->setMemoryDescriptor(persistentMemoryDesc);

    pNode neg1 = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, "neg_fwd_f32", "neg1");
    GraphEditor::addNode(g, neg1);
    pNode neg2 = NodeFactory::createGenericTPCNode({OUT1}, {OUT2}, nullptr, "neg_fwd_f32", "neg2");
    GraphEditor::addNode(g, neg2);
    pNode neg3 = NodeFactory::createGenericTPCNode({OUT2}, {OUT3}, nullptr, "neg_fwd_f32", "neg3");
    GraphEditor::addNode(g, neg3);
    pNode neg4 = NodeFactory::createGenericTPCNode({OUT3}, {OUT4}, nullptr, "neg_fwd_f32", "neg4");
    GraphEditor::addNode(g, neg4);
    pNode neg5 = NodeFactory::createGenericTPCNode({OUT4}, {OUT5}, nullptr, "neg_fwd_f32", "neg5");
    GraphEditor::addNode(g, neg5);

    NodeSet blocking;
    NodeSet blocked;

    blocking.insert(neg4);
    blocked.insert(neg2);
    g.addControlDependency(blocking, blocked);
    blocking.clear();
    blocked.clear();

    ASSERT_TRUE(g.printGraphCycles());
}