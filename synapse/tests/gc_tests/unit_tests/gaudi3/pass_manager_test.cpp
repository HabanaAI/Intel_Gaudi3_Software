#include "../pass_manager_test.h"
#include "gaudi3_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "graph_compiler/pass_dependencies/training/passes_dependencies.h"
#include "pass_dependencies/training/be/be_gc_brain.h"

TEST_F(PassManagerTest, print_pass_names_gaudi3)
{
    GraphForPassPrinting<Gaudi3Graph> g;
    g.printAllPasses();
}
class LayeredBrainPassManagerTest : public GraphOptimizerTest
{
};

/**
 * @brief Pass template that clones current graph and its pass manager
 *        then sets pass manager to cloned graph and proceeds
 *        executing passes up to LastPass (including)
 */
template<PassId LastPass>
static bool runPassesUntil(HabanaGraph& g)
{
    auto pmClone   = g.clonePassManager();
    auto tempGraph = g.clone();
    tempGraph->setPassManager(pmClone);
    return tempGraph->runPartialPasses(LastPass);
}

TEST_F(LayeredBrainPassManagerTest, run_partial)
{
    Gaudi3Graph g;
    // mimics layered brain iteration in which all passes up to final layered brain pass
    // execute on the temporary graph then it is finally evaluated
    auto lbSimulationTestPass = std::make_shared<HabanaPass<runPassesUntil<PASS_ID_BUNDLE_MEMORY_MANAGEMENT>>>(
        "lbSimulationPass",
        PASS_ID_RUN_LAYERED_BRAIN,
        PassIDSet {RUN_LAYERED_BRAIN_DEPENDENCY_SET});
    // replace layered brain pass with simulation pass
    g.replaceCompilationPass(lbSimulationTestPass);
    auto success = g.compile();
    ASSERT_TRUE(success);
}