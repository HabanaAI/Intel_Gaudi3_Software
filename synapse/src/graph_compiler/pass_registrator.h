#pragma once

class HabanaGraph;

class PassRegistrator
{
public:
    virtual void registerGroups(HabanaGraph& graph) const = 0;
};
