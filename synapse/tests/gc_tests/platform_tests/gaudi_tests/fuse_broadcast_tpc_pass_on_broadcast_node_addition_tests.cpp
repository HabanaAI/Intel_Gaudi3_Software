#include "gc_gaudi_test_infra.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "node_factory.h"

TEST_F_GC(SynTrainingTestInfra, fuse_broadcast_tpc_triggered_on_broadcast_node_creation_test)
{
    TestNSizes     sizes    = {60, 1};
    TestNSizes     newSizes = {60, 3};
    constexpr auto dim = 2, concatAxis = 1;

    const auto tensor0 =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes.data(), dim, syn_type_single);

    const auto tensor1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, newSizes.data(), dim, syn_type_single);

    synConcatenateParams param = {.axis = concatAxis};
    addNodeToGraph(NodeFactory::concatenateNodeTypeName,
                   {tensor0, tensor0, tensor0},
                   {tensor1},
                   static_cast<void*>(&param),
                   sizeof(param),
                   "concat");

    const auto tensor2 =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, newSizes.data(), dim, syn_type_single);

    const auto tensor3 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, newSizes.data(), dim, syn_type_single);

    addNodeToGraph("add_f32", {tensor1, tensor2}, {tensor3}, nullptr, 0, "add");

    compileTopology("fuseBroadcastTpc_pass_on_BroadcastNode_addition_1", 0);
    const auto currRecipe = getRecipeHandle()->basicRecipeHandle.recipe;
    ASSERT_EQ(currRecipe->node_nr, 1) << "The amount of nodes is not as expected";
}