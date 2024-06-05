#pragma once

#include "synapse_common_types.h"

struct InternalRecipeHandle;

class RecipeLauncherInterface
{
public:
    virtual ~RecipeLauncherInterface() = default;
    virtual bool        isCopyNotCompleted() const                           = 0;
    virtual synStatus   checkCompletionCopy(uint64_t timeout)                = 0;
    virtual synStatus   checkCompletionCompute(uint64_t timeout)             = 0;
    virtual std::string getDescription() const                 = 0;
    virtual bool        dfaLogDescription(bool               oldestRecipeOnly,
                                          uint64_t           currentLongSo,
                                          bool               dumpRecipe,
                                          const std::string& callerMsg,
                                          bool               forTools) const = 0;

    virtual const InternalRecipeHandle& getInternalRecipeHandle() const = 0;
};
