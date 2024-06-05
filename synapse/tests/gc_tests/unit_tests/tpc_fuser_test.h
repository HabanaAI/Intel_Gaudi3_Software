#pragma once

#include "gc_tests/unit_tests/graph_optimizer_test.h"
#include "graph_compiler/passes/tpc_fuser.h"
#include "node_factory.h"
#include "kernel_db.h"

class TPCFuserTest : public GraphOptimizerTest
{

protected:
    virtual void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        prev_CFG_RUN_TPC_FUSER = GCFG_RUN_TPC_FUSER.value();
        pre_GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER = GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER.value();
        GCFG_RUN_TPC_FUSER.setValue(true);
        GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER.setValue(1);
    }

    virtual void TearDown() override
    {
        GCFG_RUN_TPC_FUSER.setValue(prev_CFG_RUN_TPC_FUSER);
        GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER.setValue(pre_GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER);
        GraphOptimizerTest::TearDown();
    }

    bool prev_CFG_RUN_TPC_FUSER;
    unsigned pre_GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER;
};

extern optimizedGraphStatus replaceOptimizedCluster(HabanaGraph&                              graph,
                                                    const std::shared_ptr<GCTPCFuserWrapper>& fuser);

gcapi::FuserRetVal_t
fuserGraphFuncOptReturnSameGraph(const FuserGraphTypeV4* graphIn, FuserGraphTypeV4* graphOut, bool debug);

// Testing TPCFuser class
// Test case: Returned fusedGraph does not contain all external tensor.
//            two nodes where intermediate tensor is marked as persistent, is missing from fused graph
// Expecting no fusion.
//
gcapi::GlueCodeReturn_t fuserGraphGetEmptyPreGraph(const FuserNodeTypeV4* nodeIn, FuserGraphTypeV4** graphOut);

gcapi::FuserRetVal_t
fuserGraphFuncOptimizedGraphUnchanged(const FuserGraphTypeV4* graphIn, FuserGraphTypeV4* graphOut, bool debug);