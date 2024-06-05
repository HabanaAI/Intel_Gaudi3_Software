#pragma once

#include "gtest/gtest.h"
#include  "habana_graph.h"
#include "graph_optimizer_test.h"

class UniqueHash : public GraphOptimizerTest
{
public:
    void blob_equality(HabanaGraph *g);
    void blob_inequality(HabanaGraph *g);
};