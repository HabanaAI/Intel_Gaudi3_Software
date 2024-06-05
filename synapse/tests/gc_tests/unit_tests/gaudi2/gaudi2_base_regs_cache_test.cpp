#include "gaudi_base_regs_cache_test_common.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "scoped_configuration_change.h"

class Gaudi2BaseRegsCacheTest : public GaudiBaseRegsCacheTestCommon<Gaudi2Graph>
{
};

TEST_F(Gaudi2BaseRegsCacheTest, basic)
{
    basic();
}

TEST_F(Gaudi2BaseRegsCacheTest, single_logical_engine_graph)
{
    ScopedConfigurationChange disableGCOpValidation("ENABLE_GC_NODES_VALIDATION_BY_OPS_DB", "false");
    single_logical_engine_graph();
}

TEST_F(Gaudi2BaseRegsCacheTest, saturating_node)
{
    ScopedConfigurationChange disableGCOpValidation("ENABLE_GC_NODES_VALIDATION_BY_OPS_DB", "false");
    saturating_node();
}

TEST_F(Gaudi2BaseRegsCacheTest, multi_logical_engines_graph)
{
    multi_logical_engines_graph();
}
