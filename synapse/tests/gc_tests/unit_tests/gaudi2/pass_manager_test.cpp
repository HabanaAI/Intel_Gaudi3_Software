#include "../pass_manager_test.h"
#include "gaudi2_graph.h"

TEST_F(PassManagerTest, print_pass_names_gaudi2)
{
    GraphForPassPrinting<Gaudi2Graph> g;
    g.printAllPasses();
}
