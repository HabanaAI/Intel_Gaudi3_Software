#include "gaudi_base_regs_cache_test_common.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"

class Gaudi3BaseRegsCacheTest : public GaudiBaseRegsCacheTestCommon<Gaudi3Graph>
{
};

TEST_F(Gaudi3BaseRegsCacheTest, DISABLED_basic)  // TODO enable, dma1_node not suppopretd yet
{
    basic();
}

TEST_F(Gaudi3BaseRegsCacheTest, single_logical_engine_graph)
{
    single_logical_engine_graph();
}

TEST_F(Gaudi3BaseRegsCacheTest, saturating_node)
{
    saturating_node();
}

TEST_F(Gaudi3BaseRegsCacheTest, multi_logical_engines_graph)
{
    multi_logical_engines_graph();
}
