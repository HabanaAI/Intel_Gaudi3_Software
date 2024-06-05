#pragma once

#include "graph_compiler/recipe_generator.h"

namespace gaudi
{

class GaudiRecipeGenerator : public RecipeGenerator
{
public:
    GaudiRecipeGenerator(const HabanaGraph* g);

    virtual ~GaudiRecipeGenerator();

    virtual std::string getEngineStr(unsigned id) const override;
    virtual void validateQueue(ConstCommandQueuePtr queue, bool isSetup) const override;
    virtual void
    inspectRecipePackets(const void* buffer, unsigned bufferSize, std::string_view bufferName) const override;
    virtual void serializeSyncSchemeDebugInfo(debug_sync_scheme_t* syncSchemeInfo) const override;

private:
    void collectNodeSyncInfo(std::vector<NodeSyncInfo>& allNodesSyncInfo) const override;

};

}