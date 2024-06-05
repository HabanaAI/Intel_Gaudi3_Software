#pragma once

#include "graph_compiler/recipe_generator.h"

namespace gaudi2
{

class Gaudi2RecipeGenerator : public RecipeGenerator
{
public:
    Gaudi2RecipeGenerator(const HabanaGraph* g);

    virtual ~Gaudi2RecipeGenerator() = default;

    static uint32_t getGaudi2VersionMinor();

    virtual std::string getEngineStr(unsigned id) const override;
    virtual void        validateQueue(ConstCommandQueuePtr queue, bool isSetup) const override;
    virtual bool        shouldCreateECBs() const override { return true; }
    virtual uint32_t    getVersionMinor() const override;
    virtual void        setBlobFlags(RecipeBlob* blob, QueueCommand* cmd) const override;
    virtual bool        isSFGInitCommand(QueueCommand* cmd) const override;
    virtual unsigned    getInitSfgValue(QueueCommand* cmd) const override;
};

}