#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <utils/dense_coord_iter.h>
#include "graph_optimizer_test.h"

class DenseCoordIterTests : public GraphOptimizerTest {};

TEST_F(DenseCoordIterTests, normal_iteration)
{
    std::array<unsigned, 3> bounds{2, 3, 4};
    DenseCoordIter<unsigned, 3> it(bounds);
    ASSERT_EQ(it.size(), 3);
    ASSERT_FALSE(it.end());

    ASSERT_EQ(it[0], 0);
    ASSERT_EQ(it[1], 0);
    ASSERT_EQ(it[2], 0);

    it.next();

    ASSERT_EQ(it[0], 1);
    ASSERT_EQ(it[1], 0);
    ASSERT_EQ(it[2], 0);

    it.next();

    ASSERT_EQ(it[0], 0);
    ASSERT_EQ(it[1], 1);
    ASSERT_EQ(it[2], 0);

    it.next();

    ASSERT_EQ(it[0], 1);
    ASSERT_EQ(it[1], 1);
    ASSERT_EQ(it[2], 0);

    it.next();

    ASSERT_EQ(it[0], 0);
    ASSERT_EQ(it[1], 2);
    ASSERT_EQ(it[2], 0);

    it.advance(1);

    ASSERT_EQ(it[0], 1);
    ASSERT_EQ(it[1], 2);
    ASSERT_EQ(it[2], 0);

    it.advance(2);

    ASSERT_EQ(it[0], 1);
    ASSERT_EQ(it[1], 0);
    ASSERT_EQ(it[2], 1);

    it.advance(2*3*4 - 7 - 1); // go to the last coord

    ASSERT_EQ(it[0], 1);
    ASSERT_EQ(it[1], 2);
    ASSERT_EQ(it[2], 3);

    ASSERT_FALSE(it.end());
    it.next();
    ASSERT_TRUE(it.end());
}

TEST_F(DenseCoordIterTests, smallest_3d_tensor)
{
    std::array<unsigned, 3> bounds{1, 1, 1};
    DenseCoordIter<unsigned, 3> it(bounds);

    ASSERT_FALSE(it.end());
    it.next();
    ASSERT_TRUE(it.end());
}

TEST_F(DenseCoordIterTests, smallest_1d_tensor)
{
    std::array<unsigned, 1> bounds{1};
    DenseCoordIter<unsigned, 1> it(bounds);

    ASSERT_FALSE(it.end());
    it.next();
    ASSERT_TRUE(it.end());
}

TEST_F(DenseCoordIterTests, frozen_dims)
{
    std::array<unsigned, 4> bounds{2, 3, 4, 5};
    DenseCoordIter<unsigned, 4, DCI::freeze_dims(0, 2)> it(bounds);
    ASSERT_EQ(it.size(), 4);
    ASSERT_FALSE(it.end());

    ASSERT_EQ(it[0], 0);
    ASSERT_EQ(it[1], 0);
    ASSERT_EQ(it[2], 0);
    ASSERT_EQ(it[3], 0);

    it.next();

    ASSERT_EQ(it[0], 0);
    ASSERT_EQ(it[1], 1);
    ASSERT_EQ(it[2], 0);
    ASSERT_EQ(it[3], 0);

    it.next();

    ASSERT_EQ(it[0], 0);
    ASSERT_EQ(it[1], 2);
    ASSERT_EQ(it[2], 0);
    ASSERT_EQ(it[3], 0);

    it.next();

    ASSERT_EQ(it[0], 0);
    ASSERT_EQ(it[1], 0);
    ASSERT_EQ(it[2], 0);
    ASSERT_EQ(it[3], 1);

    it.next();

    ASSERT_EQ(it[0], 0);
    ASSERT_EQ(it[1], 1);
    ASSERT_EQ(it[2], 0);
    ASSERT_EQ(it[3], 1);

    it.advance(2);

    ASSERT_EQ(it[0], 0);
    ASSERT_EQ(it[1], 0);
    ASSERT_EQ(it[2], 0);
    ASSERT_EQ(it[3], 2);

    it.advance(3*5 - 6 - 1); // go to the last coord

    ASSERT_EQ(it[0], 0);
    ASSERT_EQ(it[1], 2);
    ASSERT_EQ(it[2], 0);
    ASSERT_EQ(it[3], 4);

    ASSERT_FALSE(it.end());
    it.next();
    ASSERT_TRUE(it.end());
}

TEST_F(DenseCoordIterTests, overflow_test)
{
    std::array<unsigned, 20> bounds;
    for (int i = 0; i < bounds.size(); ++i)
    {
        bounds[i] = 16;
    }
    DenseCoordIter<unsigned, 20> it(bounds);
    it.advance(0xDEADBEEFDEADBEEF);
    // we should get the same thing in hex in reverse
    ASSERT_EQ(it[19], 0);
    ASSERT_EQ(it[18], 0);
    ASSERT_EQ(it[17], 0);
    ASSERT_EQ(it[16], 0);
    ASSERT_EQ(it[15], 0xD);
    ASSERT_EQ(it[14], 0xE);
    ASSERT_EQ(it[13], 0xA);
    ASSERT_EQ(it[12], 0xD);
    ASSERT_EQ(it[11], 0xB);
    ASSERT_EQ(it[10], 0xE);
    ASSERT_EQ(it[9], 0xE);
    ASSERT_EQ(it[8], 0xF);

    ASSERT_EQ(it[7], 0xD);
    ASSERT_EQ(it[6], 0xE);
    ASSERT_EQ(it[5], 0xA);
    ASSERT_EQ(it[4], 0xD);
    ASSERT_EQ(it[3], 0xB);
    ASSERT_EQ(it[2], 0xE);
    ASSERT_EQ(it[1], 0xE);
    ASSERT_EQ(it[0], 0xF);
}
