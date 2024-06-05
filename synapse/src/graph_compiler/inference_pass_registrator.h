#pragma once
#include "pass_registrator.h"

class InferencePassRegistrator : public PassRegistrator
{
public:
    virtual void registerGroups(HabanaGraph& graph) const override;
};
