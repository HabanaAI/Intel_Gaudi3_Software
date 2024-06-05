#include <gtest/gtest.h>
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "runtime/common/recipe/recipe_manager.hpp"
#include "habana_graph_mock.hpp"

class UTRecipeManagerTest : public ::testing::Test
{
};

TEST_F(UTRecipeManagerTest, checkCompileGetRemove)
{
    RecipeManager                  rMng;
    HabanaGraphMock                graph(synDeviceGaudi2, {0x100, 0x200, 0x400});
    const std::string              fileName("myRecipe.txt");
    InternalRecipeHandle*          pRecipeHndl = nullptr;
    bool                           status;

    status = rMng.addRecipeHandleAndCompileGraph(&graph, false, nullptr, 0, fileName.c_str(), nullptr, pRecipeHndl);
    ASSERT_EQ(synSuccess, status);
    ASSERT_NE(nullptr, pRecipeHndl);
    status = rMng.removeRecipeHandle(pRecipeHndl);
    ASSERT_EQ(true, status);
}

TEST_F(UTRecipeManagerTest, checkCompileGetRemoveAll)
{
    RecipeManager         rMng;
    HabanaGraphMock       graph(synDeviceGaudi2, {0x100, 0x200, 0x400});
    const std::string     fileName("myRecipe.txt");
    InternalRecipeHandle* pRecipeHndl = nullptr;
    bool                  status;

    status = rMng.addRecipeHandleAndCompileGraph(&graph, false, nullptr, 0, fileName.c_str(), nullptr, pRecipeHndl);
    ASSERT_EQ(synSuccess, status);
    ASSERT_NE(nullptr, pRecipeHndl);
    status = rMng.removeAllRecipeHandle();
    ASSERT_EQ(true, status);
}
