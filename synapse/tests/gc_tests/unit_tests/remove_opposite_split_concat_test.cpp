#include "compilation_hal_reader.h"
#include "gaudi_graph.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "graph_visualization.h"
#include "node_factory.h"

class RemoveSplitConcatTest : public GraphOptimizerTest
{
protected:
    unsigned numOfNodes(Node::eNodeType type)
    {
        unsigned counter = 0;
        for (const auto& n : m_graph.getNodes())
        {
            if (n && n->getNodeType() == type) ++counter;
        }
        return counter;
    }

    GaudiGraph m_graph;
};

TEST_F(RemoveSplitConcatTest, base_test)
{
    const unsigned numOfTensors = 10;

    NSizeArray smallSizes = {1};
    NSizeArray BigSizes   = {numOfTensors};

    synAxisParams aggDim = {.axis = 0};

    TensorVector inputs;
    TensorVector outputs;

    for (unsigned i = 0; i < numOfTensors; ++i)
    {
        TensorPtr in(new Tensor(1, smallSizes.data(), syn_type_float));
        TensorPtr out(new Tensor(1, smallSizes.data(), syn_type_float));

        synMemoryDescriptor inMemDesc(true);
        synMemoryDescriptor outMemDesc(true);
        in->setMemoryDescriptor(inMemDesc);
        out->setMemoryDescriptor(outMemDesc);

        inputs.push_back(in);
        outputs.push_back(out);
    }

    TensorPtr shared(new Tensor(1, BigSizes.data(), syn_type_float));
    NodePtr   concat =
        NodeFactory::createNode(inputs, {shared}, &aggDim, NodeFactory::concatenateNodeInternalTypeName, "concat");
    NodePtr split =
        NodeFactory::createNode({shared}, outputs, &aggDim, NodeFactory::splitNodeInternalTypeName, "split");

    GraphEditor::addNodes(m_graph, {split, concat});

    GraphVisualization::graphVisualizationOnDemand(m_graph, "PreGraph");
    removeOppositeConcatSplitSequence(m_graph);
    GraphVisualization::graphVisualizationOnDemand(m_graph, "PostGraph");

    HB_ASSERT(numOfNodes(Node::TYPE_INTERNAL_CONCAT) + numOfNodes(Node::TYPE_INTERNAL_SPLIT) == 0,
              "Unexpected split/concat nodes");
    HB_ASSERT(numOfNodes(Node::TYPE_IDENTITY) == numOfTensors, "Unexpected nodes");
}

TEST_F(RemoveSplitConcatTest, number_of_identities)
{
    const unsigned numOfTensors = 10;

    NSizeArray smallSizes = {1};
    NSizeArray BigSizes   = {numOfTensors};

    synAxisParams aggDim = {.axis = 0};

    TensorVector inputs;
    TensorVector outputs;

    for (unsigned i = 0; i < numOfTensors; ++i)
    {
        TensorPtr in(new Tensor(1, smallSizes.data(), syn_type_float));
        TensorPtr in_(new Tensor(1, smallSizes.data(), syn_type_float));
        TensorPtr out(new Tensor(1, smallSizes.data(), syn_type_float));
        TensorPtr out_(new Tensor(1, smallSizes.data(), syn_type_float));

        if (i % 2 == 0)
        {
            synMemoryDescriptor inMemDesc(true);
            in_->setMemoryDescriptor(inMemDesc);
        }
        if (i % 3 == 0)
        {
            synMemoryDescriptor outMemDesc(true);
            out_->setMemoryDescriptor(outMemDesc);
        }

        NodePtr neg1 = NodeFactory::createNode({in}, {in_}, nullptr, "neg_fwd_f32", "neg_in_" + std::to_string(i));
        NodePtr neg2 = NodeFactory::createNode({out_}, {out}, nullptr, "neg_fwd_f32", "neg_out_" + std::to_string(i));

        GraphEditor::addNodes(m_graph, {neg1, neg2});
        inputs.push_back(in_);
        outputs.push_back(out_);
    }

    TensorPtr shared(new Tensor(1, BigSizes.data(), syn_type_float));
    NodePtr   concat =
        NodeFactory::createNode(inputs, {shared}, &aggDim, NodeFactory::concatenateNodeInternalTypeName, "concat");
    NodePtr split =
        NodeFactory::createNode({shared}, outputs, &aggDim, NodeFactory::splitNodeInternalTypeName, "split");

    GraphEditor::addNodes(m_graph, {split, concat});

    GraphVisualization::graphVisualizationOnDemand(m_graph, "PreGraph");
    removeOppositeConcatSplitSequence(m_graph);
    GraphVisualization::graphVisualizationOnDemand(m_graph, "MidGraph");
    HB_ASSERT(numOfNodes(Node::TYPE_IDENTITY) == 7, "Unexpected nodes");
    removeRedundantLogicalNodes(m_graph);
    GraphVisualization::graphVisualizationOnDemand(m_graph, "PostGraph");

    HB_ASSERT(numOfNodes(Node::TYPE_INTERNAL_CONCAT) + numOfNodes(Node::TYPE_INTERNAL_SPLIT) == 0,
              "Unexpected split/concat nodes");
    HB_ASSERT(numOfNodes(Node::TYPE_IDENTITY) == 2, "Unexpected nodes");
}

TEST_F(RemoveSplitConcatTest, no_remove)
{
    const unsigned numOfTensors = 10;

    NSizeArray smallSizes = {1};
    NSizeArray midSizes   = {2};
    NSizeArray BigSizes   = {numOfTensors};

    synAxisParams aggDim = {.axis = 0};

    TensorVector inputs;
    TensorVector outputs;

    for (unsigned i = 0; i < numOfTensors; ++i)
    {
        TensorPtr           in(new Tensor(1, smallSizes.data(), syn_type_float));
        synMemoryDescriptor inMemDesc(true);
        in->setMemoryDescriptor(inMemDesc);
        inputs.push_back(in);
        if (i != 0)
        {
            TensorPtr           out(new Tensor(1, i == 1 ? midSizes.data() : smallSizes.data(), syn_type_float));
            synMemoryDescriptor outMemDesc(true);
            out->setMemoryDescriptor(outMemDesc);
            outputs.push_back(out);
        }
    }

    TensorPtr shared(new Tensor(1, BigSizes.data(), syn_type_float));
    NodePtr   concat =
        NodeFactory::createNode(inputs, {shared}, &aggDim, NodeFactory::concatenateNodeInternalTypeName, "concat");
    NodePtr split =
        NodeFactory::createNode({shared}, outputs, &aggDim, NodeFactory::splitNodeInternalTypeName, "split");

    GraphEditor::addNodes(m_graph, {split, concat});

    GraphVisualization::graphVisualizationOnDemand(m_graph, "PreGraph");
    removeOppositeConcatSplitSequence(m_graph);
    GraphVisualization::graphVisualizationOnDemand(m_graph, "PostGraph");

    HB_ASSERT(numOfNodes(Node::TYPE_INTERNAL_CONCAT) + numOfNodes(Node::TYPE_INTERNAL_SPLIT) == 2,
              "Unexpected split/concat nodes");
    HB_ASSERT(numOfNodes(Node::TYPE_IDENTITY) == 0, "Unexpected nodes");
}

TEST_F(RemoveSplitConcatTest, depth_2_symmetric)
{
    const unsigned numOfTensors = 3;

    NSizeArray smallSizes = {8, 1};
    NSizeArray midSizes   = {8, numOfTensors};
    NSizeArray BigSizes   = {8, numOfTensors * numOfTensors};

    synAxisParams aggDim = {.axis = 1};

    TensorVector inputs;
    TensorVector outputs;

    for (unsigned i = 0; i < numOfTensors; ++i)
    {
        TensorVector inputs_;
        TensorVector outputs_;
        TensorPtr    in_(new Tensor(2, midSizes.data(), syn_type_float));
        TensorPtr    out_(new Tensor(2, midSizes.data(), syn_type_float));
        inputs.push_back(in_);
        outputs.push_back(out_);

        for (unsigned j = 0; j < numOfTensors; ++j)
        {
            TensorPtr           in(new Tensor(2, smallSizes.data(), syn_type_float));
            TensorPtr           out(new Tensor(2, smallSizes.data(), syn_type_float));
            synMemoryDescriptor inMemDesc(true);
            synMemoryDescriptor outMemDesc(true);
            in->setMemoryDescriptor(inMemDesc);
            out->setMemoryDescriptor(outMemDesc);
            inputs_.push_back(in);
            outputs_.push_back(out);
        }
        NodePtr concat = NodeFactory::createNode(inputs_,
                                                 {in_},
                                                 &aggDim,
                                                 NodeFactory::concatenateNodeInternalTypeName,
                                                 "concat_" + std::to_string(i));
        NodePtr split  = NodeFactory::createNode({out_},
                                                outputs_,
                                                &aggDim,
                                                NodeFactory::splitNodeInternalTypeName,
                                                "split_" + std::to_string(i));

        GraphEditor::addNodes(m_graph, {split, concat});
    }

    TensorPtr shared(new Tensor(2, BigSizes.data(), syn_type_float));
    NodePtr   concat =
        NodeFactory::createNode(inputs, {shared}, &aggDim, NodeFactory::concatenateNodeInternalTypeName, "concat");
    NodePtr split =
        NodeFactory::createNode({shared}, outputs, &aggDim, NodeFactory::splitNodeInternalTypeName, "split");

    GraphEditor::addNodes(m_graph, {split, concat});

    GraphVisualization::graphVisualizationOnDemand(m_graph, "PreGraph");
    removeOppositeConcatSplitSequence(m_graph);
    GraphVisualization::graphVisualizationOnDemand(m_graph, "PostGraph");

    HB_ASSERT(numOfNodes(Node::TYPE_INTERNAL_CONCAT) + numOfNodes(Node::TYPE_INTERNAL_SPLIT) == 0,
              "Unexpected split/concat nodes");
    HB_ASSERT(numOfNodes(Node::TYPE_IDENTITY) == numOfTensors * numOfTensors, "Unexpected nodes");
}

TEST_F(RemoveSplitConcatTest, depth_2_keep_persistent)
{
    const unsigned numOfTensors = 3;

    NSizeArray smallSizes = {8, 1};
    NSizeArray midSizes   = {8, numOfTensors};
    NSizeArray BigSizes   = {8, numOfTensors * numOfTensors};

    synAxisParams aggDim = {.axis = 1};

    TensorVector inputs;
    TensorVector outputs;

    for (unsigned i = 0; i < numOfTensors; ++i)
    {
        TensorVector        inputs_;
        TensorVector        outputs_;
        TensorPtr           in_(new Tensor(2, midSizes.data(), syn_type_float));
        TensorPtr           out_(new Tensor(2, midSizes.data(), syn_type_float));
        synMemoryDescriptor inMemDesc(true);
        synMemoryDescriptor outMemDesc(true);
        in_->setMemoryDescriptor(inMemDesc);
        out_->setMemoryDescriptor(outMemDesc);
        inputs.push_back(in_);
        outputs.push_back(out_);

        for (unsigned j = 0; j < numOfTensors; ++j)
        {
            TensorPtr in(new Tensor(2, smallSizes.data(), syn_type_float));
            TensorPtr out(new Tensor(2, smallSizes.data(), syn_type_float));
            inputs_.push_back(in);
            outputs_.push_back(out);
        }
        NodePtr concat = NodeFactory::createNode(inputs_,
                                                 {in_},
                                                 &aggDim,
                                                 NodeFactory::concatenateNodeInternalTypeName,
                                                 "concat_" + std::to_string(i));
        NodePtr split  = NodeFactory::createNode({out_},
                                                outputs_,
                                                &aggDim,
                                                NodeFactory::splitNodeInternalTypeName,
                                                "split_" + std::to_string(i));

        GraphEditor::addNodes(m_graph, {split, concat});
    }

    TensorPtr shared(new Tensor(2, BigSizes.data(), syn_type_float));
    NodePtr   concat =
        NodeFactory::createNode(inputs, {shared}, &aggDim, NodeFactory::concatenateNodeInternalTypeName, "concat");
    NodePtr split =
        NodeFactory::createNode({shared}, outputs, &aggDim, NodeFactory::splitNodeInternalTypeName, "split");

    GraphEditor::addNodes(m_graph, {split, concat});

    GraphVisualization::graphVisualizationOnDemand(m_graph, "PreGraph");
    removeOppositeConcatSplitSequence(m_graph);
    GraphVisualization::graphVisualizationOnDemand(m_graph, "PostGraph");

    HB_ASSERT(numOfNodes(Node::TYPE_INTERNAL_CONCAT) == numOfTensors, "Unexpected split/concat nodes");
    HB_ASSERT(numOfNodes(Node::TYPE_IDENTITY) == numOfTensors, "Unexpected nodes");
}

TEST_F(RemoveSplitConcatTest, depth_2_different_agg_dims_symmetric)
{
    const unsigned numOfTensors = 3;

    NSizeArray smallSizes = {8, 1};
    NSizeArray midSizes   = {8, numOfTensors};
    NSizeArray BigSizes   = {8 * numOfTensors, numOfTensors};

    synAxisParams aggDim = {.axis = 0};

    TensorVector inputs;
    TensorVector outputs;

    for (unsigned i = 0; i < numOfTensors; ++i)
    {
        TensorVector inputs_;
        TensorVector outputs_;
        TensorPtr    in_(new Tensor(2, midSizes.data(), syn_type_float));
        TensorPtr    out_(new Tensor(2, midSizes.data(), syn_type_float));
        inputs.push_back(in_);
        outputs.push_back(out_);

        for (unsigned j = 0; j < numOfTensors; ++j)
        {
            TensorPtr           in(new Tensor(2, smallSizes.data(), syn_type_float));
            TensorPtr           out(new Tensor(2, smallSizes.data(), syn_type_float));
            synMemoryDescriptor inMemDesc(true);
            synMemoryDescriptor outMemDesc(true);
            in->setMemoryDescriptor(inMemDesc);
            out->setMemoryDescriptor(outMemDesc);
            inputs_.push_back(in);
            outputs_.push_back(out);
        }
        synAxisParams aggDim1 = {.axis = 1};
        NodePtr       concat  = NodeFactory::createNode(inputs_,
                                                 {in_},
                                                 &aggDim1,
                                                 NodeFactory::concatenateNodeInternalTypeName,
                                                 "concat_" + std::to_string(i));
        NodePtr       split   = NodeFactory::createNode({out_},
                                                outputs_,
                                                &aggDim1,
                                                NodeFactory::splitNodeInternalTypeName,
                                                "split_" + std::to_string(i));

        GraphEditor::addNodes(m_graph, {split, concat});
    }

    TensorPtr shared(new Tensor(2, BigSizes.data(), syn_type_float));
    NodePtr   concat =
        NodeFactory::createNode(inputs, {shared}, &aggDim, NodeFactory::concatenateNodeInternalTypeName, "concat");
    NodePtr split =
        NodeFactory::createNode({shared}, outputs, &aggDim, NodeFactory::splitNodeInternalTypeName, "split");

    GraphEditor::addNodes(m_graph, {split, concat});

    GraphVisualization::graphVisualizationOnDemand(m_graph, "PreGraph");
    removeOppositeConcatSplitSequence(m_graph);
    GraphVisualization::graphVisualizationOnDemand(m_graph, "PostGraph");

    HB_ASSERT(numOfNodes(Node::TYPE_INTERNAL_CONCAT) + numOfNodes(Node::TYPE_INTERNAL_SPLIT) == 0,
              "Unexpected split/concat nodes");
    HB_ASSERT(numOfNodes(Node::TYPE_IDENTITY) == numOfTensors * numOfTensors, "Unexpected nodes");
}

TEST_F(RemoveSplitConcatTest, depth_2_remove_partial)
{
    const unsigned numOfTensors = 3;

    NSizeArray smallSizes = {8, 1};
    NSizeArray midSizes   = {8, numOfTensors};
    NSizeArray BigSizes   = {8 * numOfTensors, numOfTensors};

    synAxisParams aggDim0 = {.axis = 0};
    synAxisParams aggDim1 = {.axis = 1};

    TensorVector inputs1;
    TensorVector inputs2;
    TensorVector outputs;

    for (unsigned i = 0; i < numOfTensors; ++i)
    {
        TensorPtr           in1(new Tensor(2, smallSizes.data(), syn_type_float));
        TensorPtr           in2(new Tensor(2, midSizes.data(), syn_type_float));
        TensorPtr           out(new Tensor(2, midSizes.data(), syn_type_float));
        synMemoryDescriptor inMemDesc(true);
        synMemoryDescriptor outMemDesc(true);
        in1->setMemoryDescriptor(inMemDesc);
        if (i != 0) in2->setMemoryDescriptor(inMemDesc);
        out->setMemoryDescriptor(outMemDesc);
        inputs1.push_back(in1);
        inputs2.push_back(in2);
        outputs.push_back(out);
    }

    TensorPtr shared(new Tensor(2, BigSizes.data(), syn_type_float));

    NodePtr concat1 = NodeFactory::createNode(inputs1,
                                              {inputs2[0]},
                                              &aggDim1,
                                              NodeFactory::concatenateNodeInternalTypeName,
                                              "concat_1");
    NodePtr concat2 =
        NodeFactory::createNode(inputs2, {shared}, &aggDim0, NodeFactory::concatenateNodeInternalTypeName, "concat_2");
    NodePtr split =
        NodeFactory::createNode({shared}, outputs, &aggDim0, NodeFactory::splitNodeInternalTypeName, "split");

    GraphEditor::addNodes(m_graph, {split, concat1, concat2});

    GraphVisualization::graphVisualizationOnDemand(m_graph, "PreGraph");
    removeOppositeConcatSplitSequence(m_graph);
    GraphVisualization::graphVisualizationOnDemand(m_graph, "PostGraph");

    HB_ASSERT(numOfNodes(Node::TYPE_INTERNAL_SPLIT) == 0, "Unexpected split/concat nodes");
    HB_ASSERT(numOfNodes(Node::TYPE_INTERNAL_CONCAT) == 1, "Unexpected split/concat nodes");
    HB_ASSERT(numOfNodes(Node::TYPE_IDENTITY) == numOfTensors, "Unexpected nodes");
}

TEST_F(RemoveSplitConcatTest, huge_tensors)
{
    const unsigned   tensor_dim   = 4;
    const NSizeArray sizes        = {1, 1, 1, 4UL * 1024UL * 1024UL * 1024UL};
    TensorPtr        i            = TensorPtr(new Tensor(tensor_dim, sizes.data(), syn_type_single));
    TensorPtr        o1           = TensorPtr(new Tensor(tensor_dim, sizes.data(), syn_type_single));
    TensorPtr        o2           = TensorPtr(new Tensor(tensor_dim, sizes.data(), syn_type_single));

    synMemoryDescriptor memDesc(true);  // persistent

    i->setMemoryDescriptor(memDesc);
    i->setMemorySectionID(1);
    o2->setMemoryDescriptor(memDesc);
    o2->setMemorySectionID(3);

    NodePtr n1 = NodeFactory::createNode({i}, {o1}, nullptr, "neg_fwd_f32", "neg_f32_1");
    GraphEditor::addNode(m_graph, n1);
    NodePtr n2 = NodeFactory::createNode({o1}, {o2}, nullptr, "neg_fwd_f32", "neg_f32_2");
    GraphEditor::addNode(m_graph, n2);

    ASSERT_TRUE(m_graph.compile()) << "failed to compile graph";

    for (const NodePtr& node : m_graph.getNodes())
    {
        // TPC kernel nodes should only have one consumer or producer of type concat/split
        if (!HabanaGraph::runsOnTPC(node)) continue;

        size_t numOfSplitConcatNodes = 0;
        for (const NodePtr& consumer : m_graph.getNodeConsumers(node))
        {
            if (consumer->getNodeType() == Node::TYPE_INTERNAL_CONCAT ||
                consumer->getNodeType() == Node::TYPE_INTERNAL_SPLIT)
            {
                ++numOfSplitConcatNodes;
            }
        }
        for (const NodePtr& producer : m_graph.getNodeProducers(node))
        {
            if (producer->getNodeType() == Node::TYPE_INTERNAL_CONCAT ||
                producer->getNodeType() == Node::TYPE_INTERNAL_SPLIT)
            {
                ++numOfSplitConcatNodes;
            }
        }
        HB_ASSERT(numOfSplitConcatNodes == 1 || numOfSplitConcatNodes == 0,
                  "Unexpected split/concat nodes (#{}) for node: {}",
                  numOfSplitConcatNodes,
                  node->getNodeName());
    }
}