#include "graph_optimizer_test.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "graph_compiler/passes/handle_memory_reuse.h"
#include "gaudi_graph.h"
#include "gtest/gtest.h"

class StridedTensorOverlapTest : public GraphOptimizerTest
{
};

TEST_F(StridedTensorOverlapTest, test1)
{
    GaudiGraph g;
    {
        // original shape [64, 128, 2], strides [1, 64, 64*128]
        std::vector<TSize>    shapeA   = {64, 64, 2};
        std::vector<uint64_t> stridesA = {1, 128, 64 * 128, 64 * 128};
        std::vector<TSize>    shapeB   = {64, 64, 2};
        std::vector<uint64_t> stridesB = {1, 128, 64 * 128, 64 * 128};
        uint64_t              baseA    = 0;
        uint64_t              baseB    = 64;

        auto t_real = std::make_shared<Tensor>();
        auto t1     = std::make_shared<Tensor>(shapeA.size(), shapeA.data(), syn_type_int8, nullptr, stridesA.data());
        auto t2     = std::make_shared<Tensor>(shapeB.size(), shapeB.data(), syn_type_int8, nullptr, stridesB.data());
        t1->setAsAliasSubTensor(t_real, baseA);
        t2->setAsAliasSubTensor(t_real, baseB);

        EXPECT_FALSE(MemoryReuseHandler::isStridedOverlap(t1, t2));

        baseB = 32;
        t2->resetAliasing();
        t2->setAsAliasSubTensor(t_real, baseB);
        EXPECT_TRUE(MemoryReuseHandler::isStridedOverlap(t1, t2));
    }

    {
        // original shape [4, 6, 8, 10], strides [1, 6, 4*6, 4*6*8]
        std::vector<TSize>    shapeA   = {4 * 6, 4, 10};
        std::vector<uint64_t> stridesA = {1, 4 * 6 * 2, 4 * 6 * 8, 4 * 6 * 8};
        std::vector<TSize>    shapeB   = {4, 6, 4, 10};
        std::vector<uint64_t> stridesB = {1, 4, 4 * 6 * 2, 4 * 6 * 8, 4 * 6 * 8};
        uint64_t              baseA    = 4 * 6;
        uint64_t              baseB    = 0;
        auto                  t_real   = std::make_shared<Tensor>();
        auto t1 = std::make_shared<Tensor>(shapeA.size(), shapeA.data(), syn_type_int8, nullptr, stridesA.data());
        auto t2 = std::make_shared<Tensor>(shapeB.size(), shapeB.data(), syn_type_int8, nullptr, stridesB.data());
        t1->setAsAliasSubTensor(t_real, baseA);
        t2->setAsAliasSubTensor(t_real, baseB);
        EXPECT_FALSE(MemoryReuseHandler::isStridedOverlap(t1, t2));
    }

    {
        std::vector<TSize>    shapeA   = {4, 1, 10};
        std::vector<uint64_t> stridesA = {1, 2, 10, 10};
        std::vector<TSize>    shapeB   = {5, 1, 4};
        std::vector<uint64_t> stridesB = {1, 2, 10, 10};
        uint64_t              baseA    = 8;
        uint64_t              baseB    = 0;
        auto                  t_real   = std::make_shared<Tensor>();
        auto t1 = std::make_shared<Tensor>(shapeA.size(), shapeA.data(), syn_type_int8, nullptr, stridesA.data());
        auto t2 = std::make_shared<Tensor>(shapeB.size(), shapeB.data(), syn_type_int8, nullptr, stridesB.data());
        t1->setAsAliasSubTensor(t_real, baseA);
        t2->setAsAliasSubTensor(t_real, baseB);
        EXPECT_TRUE(MemoryReuseHandler::isStridedOverlap(t1, t2));
    }
}