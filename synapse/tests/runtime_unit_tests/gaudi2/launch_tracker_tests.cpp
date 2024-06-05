#include <gtest/gtest.h>

#include "hpp/syn_context.hpp"

#include "runtime/scal/common/recipe_launcher/launch_tracker.hpp"
#include "runtime/scal/common/recipe_launcher/recipe_launcher.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"

class RecipeLauncherMock : public RecipeLauncherInterface
{
public:
    RecipeLauncherMock(uint32_t id, uint32_t busyCnt) : m_recipeHandle {}, m_id(id), m_busyCnt(busyCnt) {}

    bool isCopyNotCompleted() const override { return true; }

    synStatus checkCompletionCopy(uint64_t timeout) override { return checkCompletionBusy(); }

    synStatus checkCompletionCompute(uint64_t timeout) override { return checkCompletionBusy(); }

    std::string getDescription() const override { return "id " + std::to_string(m_id); }

    bool dfaLogDescription(bool               oldestRecipeOnly,
                           uint64_t           currentLongSo,
                           bool               dumpRecipe,
                           const std::string& callerMsg,
                           bool               forTools) const override
    {
        return true;
    }

    virtual const InternalRecipeHandle& getInternalRecipeHandle() const override { return m_recipeHandle; }

private:
    synStatus checkCompletionBusy()
    {
        if (m_busyCnt > 0)
        {
            LOG_TRACE(SYN_RT_TEST, "checkCompletionBusy {} busy {}", getDescription(), m_busyCnt);
            m_busyCnt--;
            return synBusy;
        }
        else
        {
            LOG_TRACE(SYN_RT_TEST, "checkCompletionBusy {} success", getDescription());
            return synSuccess;
        }
    }

    const InternalRecipeHandle m_recipeHandle;
    const uint32_t             m_id;
    uint32_t                   m_busyCnt;
};

class UTGaudi2LaunchTrackerTests : public ::testing::Test
{
};

TEST(UTGaudi2LaunchTrackerTests, basic)
{
    syn::Context  context;
    LaunchTracker lt(3);  // 3 -> check 3 every time

    // add one, return success
    {
        std::unique_ptr<RecipeLauncherMock> rl1(new RecipeLauncherMock(1, 0));
        lt.add(std::move(rl1));

        uint64_t numOfCompletionsCopy, numOfCompletionsCompute;
        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 1);
    }

    // add one, return fail
    {
        std::unique_ptr<RecipeLauncherMock> rl2(new RecipeLauncherMock(2, 1));
        lt.add(std::move(rl2));

        uint64_t numOfCompletionsCopy, numOfCompletionsCompute;
        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 1);
        ASSERT_EQ(numOfCompletionsCompute, 0);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 1);
    }

    // add two, first OK, second is busy
    {
        std::unique_ptr<RecipeLauncherMock> rl3(new RecipeLauncherMock(3, 0));
        std::unique_ptr<RecipeLauncherMock> rl4(new RecipeLauncherMock(4, 3));
        lt.add(std::move(rl3));
        lt.add(std::move(rl4));

        uint64_t numOfCompletionsCopy, numOfCompletionsCompute;
        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);  // 3 is OK, 4 is busy
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 1);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);  // 4 is busy again
        ASSERT_EQ(numOfCompletionsCopy, 1);
        ASSERT_EQ(numOfCompletionsCompute, 0);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 1);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 0);
    }

    // add 4, all are OK
    {
        std::unique_ptr<RecipeLauncherMock> rl5(new RecipeLauncherMock(5, 0));
        std::unique_ptr<RecipeLauncherMock> rl6(new RecipeLauncherMock(6, 0));
        std::unique_ptr<RecipeLauncherMock> rl7(new RecipeLauncherMock(7, 0));
        std::unique_ptr<RecipeLauncherMock> rl8(new RecipeLauncherMock(8, 1));

        lt.add(std::move(rl5));
        lt.add(std::move(rl6));
        lt.add(std::move(rl7));
        lt.add(std::move(rl8));

        uint64_t numOfCompletionsCopy, numOfCompletionsCompute;
        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);  // 5,6,7 cleared
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 3);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);  // 8 is busy again
        ASSERT_EQ(numOfCompletionsCopy, 1);
        ASSERT_EQ(numOfCompletionsCompute, 0);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);  // 8 is busy
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 1);
    }

    // check checkForCompletionNeedResource - add 1, keep it busy for some time, make sure we wait
    {
        // Sanity
        synStatus status = lt.waitForCompletionCopy();
        ASSERT_EQ(status, synUnavailable);

        std::unique_ptr<RecipeLauncherMock> rl9(new RecipeLauncherMock(9, 10));
        lt.add(std::move(rl9));

        uint64_t numOfCompletionsCopy, numOfCompletionsCompute;
        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);  // 9 is busy
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 0);

        status = lt.waitForCompletionCopy();  // roll the rest of the 9 busy
        ASSERT_EQ(status, synSuccess);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);  // 9 is cleared
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 1);
    }

    // check checkForCompletionAll - add 4, verify all are cleared
    {
        std::unique_ptr<RecipeLauncherMock> rl10(new RecipeLauncherMock(10, 0));
        std::unique_ptr<RecipeLauncherMock> rl11(new RecipeLauncherMock(11, 3));
        std::unique_ptr<RecipeLauncherMock> rl12(new RecipeLauncherMock(12, 3));
        std::unique_ptr<RecipeLauncherMock> rl13(new RecipeLauncherMock(13, 0));

        lt.add(std::move(rl10));
        lt.add(std::move(rl11));
        lt.add(std::move(rl12));
        lt.add(std::move(rl13));

        synStatus status = lt.checkForCompletionAll();
        ASSERT_EQ(status, synSuccess);

        status = lt.checkForCompletionAll();
        ASSERT_EQ(status, synSuccess);

        uint64_t numOfCompletionsCopy, numOfCompletionsCompute;
        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 0);
    }

    // check checkForCompletionAll - add 4, verify all are cleared by copy->compute order
    {
        std::unique_ptr<RecipeLauncherMock> rl14(new RecipeLauncherMock(14, 1));
        std::unique_ptr<RecipeLauncherMock> rl15(new RecipeLauncherMock(15, 2));
        std::unique_ptr<RecipeLauncherMock> rl16(new RecipeLauncherMock(16, 3));
        std::unique_ptr<RecipeLauncherMock> rl17(new RecipeLauncherMock(17, 3));

        lt.add(std::move(rl14));
        lt.add(std::move(rl15));
        lt.add(std::move(rl16));
        lt.add(std::move(rl17));

        uint64_t numOfCompletionsCopy, numOfCompletionsCompute;
        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 1);
        ASSERT_EQ(numOfCompletionsCompute, 0);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 1);
        ASSERT_EQ(numOfCompletionsCompute, 1);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 1);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 1);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 1);
        ASSERT_EQ(numOfCompletionsCompute, 0);

        lt.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
        ASSERT_EQ(numOfCompletionsCopy, 0);
        ASSERT_EQ(numOfCompletionsCompute, 1);
    }
}
