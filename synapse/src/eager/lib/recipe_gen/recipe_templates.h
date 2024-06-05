#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_templates_defs.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/types.h"

namespace eager_mode
{
// A singleton to create and query all recipe templates
class RecipeTemplates
{
public:
    static RecipeTemplates& getInstance();
    void                    createAllTemplates();
    const TemplateOfEngine& getTemplate(ChipType chip, EngineType engine, TensorsNrType tensorsNr) const;

    const KernelInfo& getNOPKernelInfo(ChipType chip) const;

private:
    void initSpecialBlobInfo(TemplateOfEngine& templateOfEngine);
    void initNOPNodes();
    void initAndExtractKernelInfo(tpc_lib_api::DeviceId device, const TensorPtr& inTensor, uint32_t kernelUniqueId);

private:
    bool                m_isTemplatesCreated = false;  // Flag to guarantee creating templates one time
    TemplatesOfAllChips m_templates;                   // Templates of all chips and all engine
    NOPKernelsofAllChips m_nopKernels;
};

}  // namespace eager_mode
