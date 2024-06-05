#include "graph_optimizer_test.h"

#include "brain/slicer/strategy_filter.h"

using namespace gc::layered_brain::slicer;

class ConflictingPerforationDetectionTest : public GraphOptimizerTest
{
};

TEST_F(ConflictingPerforationDetectionTest, no_conflict__with_no_nodes)
{
    ConflictingPerforationDetector cpd {};

    EXPECT_FALSE(cpd.hasConflict());
}

TEST_F(ConflictingPerforationDetectionTest, conflict__with_perforation_not_in_node_bvds)
{
    ConflictingPerforationDetector cpd {};

    cpd.addNodeBVDs(3, {1, 2, 4});

    EXPECT_TRUE(cpd.hasConflict());
}

TEST_F(ConflictingPerforationDetectionTest, no_conflict__without_perforation)
{
    ConflictingPerforationDetector cpd {};

    cpd.addNodeBVDs(std::nullopt, {1, 2, 4});

    EXPECT_FALSE(cpd.hasConflict());
}

TEST_F(ConflictingPerforationDetectionTest, no_conflict__same_perforations)
{
    ConflictingPerforationDetector cpd {};

    cpd.addNodeBVDs(7, {7, 8});
    cpd.addNodeBVDs(7, {5, 6, 7});
    cpd.addNodeBVDs(7, {7});

    EXPECT_FALSE(cpd.hasConflict());
}

TEST_F(ConflictingPerforationDetectionTest, conflict__non_same_non_disjoint_perforations_resolveable_second)
{
    ConflictingPerforationDetector cpd {};

    // When the perforation is different and not disjoint (one of the nodes is perforated on a BVD that is shared with
    // another node that is perforated differently).
    cpd.addNodeBVDs(7, {7, 8});
    cpd.addNodeBVDs(6, {5, 6, 7});  // This node could be perforated on BVD-7 like the previous one.

    // Expect conflict detection
    EXPECT_TRUE(cpd.hasConflict());
}

TEST_F(ConflictingPerforationDetectionTest, conflict__non_same_non_disjoint_perforations_resolveable_first)
{
    ConflictingPerforationDetector cpd {};

    // same as the previous scenario, but this time the first node added can change to have same perforation as the 2nd
    cpd.addNodeBVDs(7, {7, 8});  // This node could be perforated on BVD-8 like the next one.
    cpd.addNodeBVDs(8, {8});

    // Expect conflict detection
    EXPECT_TRUE(cpd.hasConflict());
}

TEST_F(ConflictingPerforationDetectionTest, no_conflict__disjoint_perforations)
{
    ConflictingPerforationDetector cpd {};

    // When the perforation is different and disjoint (each node perforation bvd doesn't appear in any other node's
    // bvds).
    cpd.addNodeBVDs(8, {6, 7, 8});
    cpd.addNodeBVDs(5, {5, 6, 7});
    cpd.addNodeBVDs(4, {4, 6, 7});

    // Expect no conflict detection
    EXPECT_FALSE(cpd.hasConflict());
}

TEST_F(ConflictingPerforationDetectionTest, no_conflict__disjoint_groups)
{
    ConflictingPerforationDetector cpd {};

    // When the perforation is partially different and disjoint (each node perforation bvd doesn't appear in any other
    // node's bvds, or they are equal).
    cpd.addNodeBVDs(1, {1});
    cpd.addNodeBVDs(2, {5, 2});
    cpd.addNodeBVDs(2, {4, 2});

    // Expect no conflict detection
    EXPECT_FALSE(cpd.hasConflict());
}

TEST_F(ConflictingPerforationDetectionTest, no_conflict__multiple_disjoint_groups)
{
    ConflictingPerforationDetector cpd {};

    // When the perforation is partially different and disjoint (each node perforation bvd doesn't appear in any other
    // node's bvds, or they are equal).
    cpd.addNodeBVDs(2, {1, 2, 3});
    cpd.addNodeBVDs(5, {4, 5, 6});
    cpd.addNodeBVDs(8, {7, 8, 9});
    cpd.addNodeBVDs(2, {2, 10, 11});
    cpd.addNodeBVDs(5, {6, 5, 12});
    cpd.addNodeBVDs(5, {5});

    // Expect no conflict detection
    EXPECT_FALSE(cpd.hasConflict());
}

TEST_F(ConflictingPerforationDetectionTest, no_conflict__unresolveable_bridge_between_disjoint_groups__select_first)
{
    ConflictingPerforationDetector cpd {};

    cpd.addNodeBVDs(1, {1, 2});
    cpd.addNodeBVDs(3, {3, 4});

    // This node can be perforated same as the first or the second, but not both (bridges the groups). So it is not
    // resolveable.
    cpd.addNodeBVDs(1, {1, 3});

    // Expect no conflict detection
    EXPECT_FALSE(cpd.hasConflict());
}

TEST_F(ConflictingPerforationDetectionTest, no_conflict__unresolveable_bridge_between_disjoint_groups__select_second)
{
    ConflictingPerforationDetector cpd {};

    cpd.addNodeBVDs(1, {1, 2});
    cpd.addNodeBVDs(3, {3, 4});

    // This node can be perforated same as the first or the second, but not both (bridges the groups). So it is not
    // resolveable.
    cpd.addNodeBVDs(3, {1, 3});

    // Expect no conflict detection
    EXPECT_FALSE(cpd.hasConflict());
}