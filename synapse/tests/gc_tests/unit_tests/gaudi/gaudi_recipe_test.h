#pragma once

#include "queue_command.h"
#include "command_queue.h"
#include "recipe_test_base.h"

class GaudiRecipeTest: public RecipeTestBase
{
    virtual QueueCommand* getWriteReg(unsigned valueCmd) const override;
    virtual QueueCommand* getMonArm() const override;
    virtual CommandQueue* getTpcQueue(unsigned engineId,
                                      unsigned engineIndex,
                                      unsigned stream,
                                      bool sendSyncEvents) const override;
};

class GaudiRecipeTestStagedSubmission: public GaudiRecipeTest
{
    protected:
        virtual void SetUp() override;
};

