#pragma once

#include "graph_compiler/recipe_generator.h"

namespace gaudi3
{
class Gaudi3RecipeGenerator : public RecipeGenerator
{
public:
    Gaudi3RecipeGenerator(const HabanaGraph* g);

    virtual ~Gaudi3RecipeGenerator() = default;

    static uint32_t getGaudi3VersionMinor();

    virtual std::string getEngineStr(unsigned id) const override;
    virtual void        validateQueue(ConstCommandQueuePtr queue, bool isSetup) const override;
    virtual bool        shouldCreateECBs() const override;
    virtual uint32_t    getVersionMinor() const override;
    virtual void        setBlobFlags(RecipeBlob* blob, QueueCommand* cmd) const override;
    virtual bool        isMMEDmaNode(const NodePtr& n) const override;
};

}  // namespace gaudi3
