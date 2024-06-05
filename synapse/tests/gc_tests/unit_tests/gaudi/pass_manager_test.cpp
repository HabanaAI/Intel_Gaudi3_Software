#include "../pass_manager_test.h"
#include "gaudi_graph.h"

TEST_F(PassManagerTest, print_pass_names_gaudi)
{
    GraphForPassPrinting<GaudiGraph> g;
    g.printAllPasses();
}
