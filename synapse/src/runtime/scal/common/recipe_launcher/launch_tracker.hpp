#pragma once

#include "synapse_common_types.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

class RecipeLauncherInterface;
struct InternalRecipeHandle;

/*********************************************************************************
 * This class is used to track recipes that are being executed on the device
 *********************************************************************************/

// Todo remove this class - the recipe launcher should not be aware to it
class LaunchTrackerInterface
{
public:
    virtual synStatus waitForCompletionCopy() = 0;
};

class LaunchTracker : public LaunchTrackerInterface
{
public:
    LaunchTracker(uint64_t numOfCompletionsToChk);

    void   add(std::unique_ptr<RecipeLauncherInterface> recipeLauncher);
    size_t size() { return m_launches.size(); }

    void        checkForCompletion(uint64_t& rNumOfCompletionsCopy, uint64_t& rNumOfCompletionsCompute);
    synStatus   waitForCompletionCopy() override;
    synStatus   checkForCompletionAll();
    synStatus   checkRecipeCompletion(const InternalRecipeHandle& rRecipeHandle);
    void        dfaLogRecipesDesc(bool               oldestRecipeOnly,
                                  uint64_t           currentLongSo,
                                  bool               dumpRecipe,
                                  const std::string& callerMsg,
                                  bool               forTools);

private:
    using RecipeLauncherDeque = std::deque<std::unique_ptr<RecipeLauncherInterface>>;

    synStatus checkOldestCompletionCompute(uint64_t timeout);

    const uint64_t      m_numOfCompletionsToChk;
    const bool          m_isCompareAfterLaunch;
    RecipeLauncherDeque m_launches;
};
