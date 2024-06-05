#include "node_info_defs.h"
#include "habana_global_conf.h"

using hl_gcfg::MakePrivate;
using hl_gcfg::MakePublic;

GlobalConfInt64 GCFG_MAX_NODES_IN_EAGER_GRAPH(
    "MAX_NODES_IN_EAGER_GRAPH",
    "Restrict number of nodes in eager graph to a limit. Exceeding the limit cause falling-back to graph mode",
    eager_mode::defaultMaxNodesPerGraph,
    MakePrivate);
