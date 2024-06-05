#pragma once
#include "pass_registrator.h"

class TrainingPassRegistrator : public PassRegistrator
{
public:
    virtual void registerGroups(HabanaGraph& graph) const override;
};
