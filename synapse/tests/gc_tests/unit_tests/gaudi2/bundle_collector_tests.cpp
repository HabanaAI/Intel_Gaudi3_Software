#include "layered_brain_test.h"
#include "bundle_collector.h"

using namespace gc::layered_brain;

class BundleCollectorTest : public LayeredBrainTest
{
};

TEST_F(BundleCollectorTest, bundle_collector_should_return_empty_list_when_nothing_is_bundled)
{
    createGraph(10);

    BundleCollector bc(m_graph);

    EXPECT_TRUE(bc.getAllBundles().empty());
}

TEST_F(BundleCollectorTest, bundle_collector_should_collect_bundle_nodes)
{
    createGraph(3);
    bundleNodes(33, 0, 3);

    BundleCollector bc(m_graph);
    Bundles         bundles = bc.getAllBundles();

    ASSERT_EQ(1, bundles.size()) << "Expecting a single bundle";
    auto [id, nodes] = *bundles.begin();
    EXPECT_EQ(33, id);
    EXPECT_EQ(nodes.size(), 3);
}

TEST_F(BundleCollectorTest, bundle_collector_should_collect_multiple_bundles)
{
    createGraph(5);
    bundleNodes(12, 0, 3);
    bundleNodes(7, 3, 5);

    BundleCollector bc(m_graph);
    Bundles         bundles = bc.getAllBundles();

    ASSERT_EQ(2, bundles.size());

    auto bundle12 = bundles.find(12);
    ASSERT_NE(bundle12, bundles.end());
    EXPECT_EQ(3, bundle12->second.size());

    auto bundle7 = bundles.find(7);
    ASSERT_NE(bundle7, bundles.end());
    EXPECT_EQ(2, bundle7->second.size());
}

TEST_F(BundleCollectorTest, bundle_collector_should_collect_bundle_nodes_in_execution_order)
{
    // Create 5+4+3 nodes chain. Bundle the first 5 and the last 3.
    createGraph(5 + 4 + 3);
    BundleNodes bundle93 = bundleNodes(93, 0, 5);
    BundleNodes bundle19 = bundleNodes(19, 5 + 4, 5 + 4 + 3);

    BundleCollector bc(m_graph);
    Bundles         bundles = bc.getAllBundles();

    EXPECT_EQ(2, bundles.size());

    ASSERT_NO_THROW(bundles.at(93));
    EXPECT_EQ(bundles.at(93), bundle93);

    ASSERT_NO_THROW(bundles.at(19));
    EXPECT_EQ(bundles.at(19), bundle19);
}