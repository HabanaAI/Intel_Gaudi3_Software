#include "launch_tracker.hpp"

#include "global_statistics.hpp"
#include "log_manager.h"
#include "mem_mgrs.hpp"

#include "runtime/scal/common/infra/scal_includes.hpp"
#include "log_manager.h"
#include "runtime/scal/common/infra/scal_utils.hpp"

#include "runtime/scal/common/recipe_launcher/recipe_launcher.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"

const uint64_t TIMEOUT_100_USEC = 100;
const uint64_t TIMEOUT_1_SEC    = 1000000;

/*
 ***************************************************************************************************
 *   @brief LaunchTracker() - constructor. recipeLauncher is needed to unset memory after a recipe is done
 *                            scalStream is needed to check/wait for a recipe to end
 *
 *   @param  recipeLauncher, scalStream
 *   @return None
 *
 ***************************************************************************************************
 */
LaunchTracker::LaunchTracker(uint64_t numOfCompletionsToChk)
: m_numOfCompletionsToChk(numOfCompletionsToChk),
  m_isCompareAfterLaunch(ScalUtils::isCompareAfterDownloadPost() || ScalUtils::isCompareAfterLaunch())
{
}

/*
 ***************************************************************************************************
 *   @brief add() - this function is called when a recipe starts execution. It saves needed information
 *                  to check later if the launch has finished
 *
 *   @param  targetVal - a number that we can wait/check for the launch
 *   @param  ids       - recipe + running id
 *   @return None
 *
 ***************************************************************************************************
 */
void LaunchTracker::add(std::unique_ptr<RecipeLauncherInterface> recipeLauncher)
{
    // verify target value
    LOG_TRACE(SYN_WORK_COMPL, "LaunchTracker: adding {}", recipeLauncher->getDescription());

    m_launches.push_back(std::move(recipeLauncher));
}

/*
 ***************************************************************************************************
 *   @brief checkForCompletion() - check if the X (x=2) oldest launches has finished
 *
 *   @param  None
 *   @return synStatus
 *
 ***************************************************************************************************
 */
void LaunchTracker::checkForCompletion(uint64_t& rNumOfCompletionsCopy, uint64_t& rNumOfCompletionsCompute)
{
    rNumOfCompletionsCopy = rNumOfCompletionsCompute = 0;
    const uint64_t timeout                           = m_isCompareAfterLaunch ? SCAL_FOREVER : 0;
    const uint64_t numOfLaunches                     = m_launches.size();
    STAT_GLBL_COLLECT(numOfLaunches, inFlightLaunchReqStart);

    const uint64_t numOfCompletionsToChk = std::min(m_numOfCompletionsToChk, numOfLaunches);
    uint64_t       loop                  = 0;
    for (; loop < numOfCompletionsToChk; loop++)
    {
        const synStatus status = checkOldestCompletionCompute(timeout);
        if (status != synSuccess)
        {
            break;
        }

        rNumOfCompletionsCompute++;
    }
    STAT_GLBL_COLLECT(m_launches.size(), inFlightLaunchReqLeft);

    RecipeLauncherDeque::iterator iter = m_launches.begin();

    for (; loop < numOfCompletionsToChk; loop++)
    {
        RecipeLauncherInterface* pRecipeLauncher = iter->get();
        const synStatus          status          = pRecipeLauncher->checkCompletionCopy(timeout);

        if (status != synSuccess)
        {
            break;
        }

        rNumOfCompletionsCopy++;
        ++iter;
    }
}

/*
 ***************************************************************************************************
 *   @brief waitForCompletionCopy() - wait for the oldest launch has finished, This is called when we are
 *                                             out of resources and must wait for a recipe to finish and release
 *resources
 *
 *   @param  None
 *   @return synStatus
 *
 ***************************************************************************************************
 */
synStatus LaunchTracker::waitForCompletionCopy()
{
    RecipeLauncherDeque::iterator iter = m_launches.begin();
    for (; iter != m_launches.end(); ++iter)
    {
        RecipeLauncherInterface* pRecipeLauncher = iter->get();
        if (pRecipeLauncher->isCopyNotCompleted())
        {
            break;
        }
    }

    if (iter == m_launches.end())
    {
        LOG_ERR(SYN_WORK_COMPL, "waitForCompletionCopy could not find any incomplete copy");
        return synUnavailable;
    }

    RecipeLauncherInterface* pRecipeLauncher = iter->get();
    synStatus status;

    do
    {
        status = pRecipeLauncher->checkCompletionCopy(TIMEOUT_100_USEC);
        LOG_DEBUG(SYN_WORK_COMPL, "checkCompletionCopy finished with status {}", status);
    } while (status == synBusy);

    return status;
}

/*
 ***************************************************************************************************
 *   @brief checkForCompletionAll() - wait until all launches are done (used before closing the stream)
 *
 *   @param  None
 *   @return synStatus
 *
 ***************************************************************************************************
 */
synStatus LaunchTracker::checkForCompletionAll()
{
    LOG_TRACE(SYN_WORK_COMPL, "Wait for all, in q {:x}", m_launches.size());
    while (!m_launches.empty())
    {
        // one second, so we can log every second
        synStatus status = checkOldestCompletionCompute(TIMEOUT_1_SEC);

        if (status == synSuccess)
        {
            LOG_TRACE(SYN_WORK_COMPL, "finished one, in q {:x}", m_launches.size());
            continue;
        }

        if (status == synBusy)
        {
            LOG_TRACE(SYN_WORK_COMPL, "busy, still {:x}", m_launches.size());
            continue;
        }

        LOG_ERR(SYN_WORK_COMPL, "in wait for all, got an error {}", status);
        return status;
    }
    return synSuccess;
}

synStatus LaunchTracker::checkRecipeCompletion(const InternalRecipeHandle& rRecipeHandle)
{
    unsigned lastRecipeIndex = std::numeric_limits<unsigned>::max();
    for (unsigned iter = 0; iter < m_launches.size(); iter++)
    {
        const unsigned index = m_launches.size() - iter - 1;
        if (m_launches[index]->getInternalRecipeHandle().recipeSeqNum == rRecipeHandle.recipeSeqNum)
        {
            lastRecipeIndex = index;
            break;
        }
    }

    if (lastRecipeIndex != std::numeric_limits<unsigned>::max())
    {
        const unsigned completionRequired = lastRecipeIndex + 1;
        unsigned       completionCounter  = 0;
        while (completionCounter != completionRequired)
        {
            // one second, so we can log every second
            const synStatus status = checkOldestCompletionCompute(TIMEOUT_1_SEC);

            switch (status)
            {
                case synSuccess:
                {
                    LOG_TRACE(SYN_WORK_COMPL,
                                   "finished one recipeSeqNum {}, in q {:x} ",
                                   rRecipeHandle.recipeSeqNum,
                                   m_launches.size());
                    completionCounter++;
                    break;
                }
                case synBusy:
                {
                    LOG_TRACE(SYN_WORK_COMPL,
                                   "busy recipeSeqNum {}, still {:x}",
                                   rRecipeHandle.recipeSeqNum,
                                   m_launches.size());
                    continue;
                }
                default:
                {
                    LOG_ERR(SYN_WORK_COMPL,
                                 "in wait for recipeSeqNum {}, got an error {:x}",
                                 rRecipeHandle.recipeSeqNum,
                                 status);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    // if we are here, then most likely the device is dead. We sleep to avoid too many
                    // logs, event-fd should trigger, dfa and termination on another thread
                    continue;
                }
            }
        }
    }
    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief checkOldestCompletionCompute() - checks if the oldest synLaunch has finished. Timeout is given
 *                                    as a param from the user
 *
 *   @param  timeout - max time to wait for completion
 *   @return synStatus
 *
 ***************************************************************************************************
 */
synStatus LaunchTracker::checkOldestCompletionCompute(uint64_t timeout)
{
    if (m_launches.empty())
    {
        LOG_TRACE(SYN_WORK_COMPL, "non-inflight, timeout {}", timeout);
        return synUnavailable;
    }

    RecipeLauncherInterface* pRecipeLauncher = m_launches.front().get();
    synStatus                status          = pRecipeLauncher->checkCompletionCompute(timeout);
    if (status != synSuccess)
    {
        return status;
    }

    m_launches.pop_front();
    return synSuccess;
}

/**
 * Gets descriptions the recipes.
 * Each vector of strings is per recipe.
 * @return A vector of vectors (one vector per recipe)
 */
void LaunchTracker::dfaLogRecipesDesc(bool               oldestRecipeOnly,
                                      uint64_t           currentLongSo,
                                      bool               dumpRecipe,
                                      const std::string& callerMsg,
                                      bool               forTools)
{
    if (m_launches.empty())
    {
        LOG_INFO(SYN_DEV_FAIL, "No recipes found");
        return;
    }

    if (forTools)  // for tools, show one line of recipe info
    {
        for (const auto& iter : m_launches)
        {
            iter->dfaLogDescription(oldestRecipeOnly, currentLongSo, dumpRecipe, callerMsg, true);
        }
        return;
    }

    for (const auto& iter : m_launches)
    {
        bool logged = iter->dfaLogDescription(oldestRecipeOnly, currentLongSo, dumpRecipe, callerMsg, false);
        if (oldestRecipeOnly && logged)
        {
            break;
        }
    }
}
