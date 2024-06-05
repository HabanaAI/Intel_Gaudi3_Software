#include <gtest/gtest.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "graph_optimizer_test.h"

class Split : public GraphOptimizerTest {};

TEST_F(Split, outer_dim)
{
    const TSize tensorDim   = 3;
    const TSize splitDim    = tensorDim - 1;
    const TSize inSize[]    = { 8, 128, 10 };
    const TSize outSize[]   = { inSize[0], inSize[1], 1 };
    const TSize nOuts       = inSize[splitDim];
    const TSize outByteSize = outSize[2] * outSize[1] * outSize[0] * sizeof(char);
    TensorPtr      in = TensorPtr(new Tensor(tensorDim, inSize, syn_type_fixed));
    TensorVector outs;
    for (unsigned i = 0; i < nOuts; ++i)
    {
        outs.push_back(TensorPtr(new Tensor(tensorDim, outSize, syn_type_fixed)));
    }

    NodePtr split = NodeFactory::createNode({in}, outs, &splitDim, NodeFactory::splitNodeTypeName, "");

    unsigned baseAddr = 0x1000;
    unsigned offset   = baseAddr;
    in->setSramOffset(baseAddr);
    split->runLogicalOperation();
    for (TensorPtr t : outs)
    {
        ASSERT_EQ(t->getSramOffset(), offset);
        offset += outByteSize;
    }
}

TEST_F(Split, outer_dim_squeeze)
{
    const TSize tensorDim   = 3;
    const TSize splitDim    = tensorDim - 1;
    const TSize inSize[]    = { 8, 128, 10 };
    const TSize outSize[]   = { inSize[0], inSize[1]};
    const TSize nOuts       = inSize[splitDim];
    const TSize outByteSize = outSize[1] * outSize[0] * sizeof(char);
    TensorPtr      in = TensorPtr(new Tensor(tensorDim, inSize, syn_type_fixed));
    TensorVector outs;
    for (unsigned i = 0; i < nOuts; ++i)
    {
        outs.push_back(TensorPtr(new Tensor(tensorDim - 1, outSize, syn_type_fixed)));
    }

    NodePtr split = NodeFactory::createNode({in}, outs, &splitDim, NodeFactory::splitNodeTypeName, "");

    unsigned baseAddr = 0x1000;
    unsigned offset   = baseAddr;
    in->setSramOffset(baseAddr);
    split->runLogicalOperation();
    for (TensorPtr t : outs)
    {
        ASSERT_EQ(t->getSramOffset(), offset);
        offset += outByteSize;
    }
}
