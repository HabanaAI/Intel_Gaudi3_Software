#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_templates_defs.h"

namespace eager_mode::gaudi2_spec_info
{
struct TemplatesCreator final : public TemplatesCreatorBase
{
    void create(TemplatesOfChip& templatesOfChip) override;

private:
    static void createMmeTemplates(TemplatesOfChip& templatesOfChip);
    static void createTpcTemplates(TemplatesOfChip& templatesOfChip);
    static void createDmaTemplates(TemplatesOfChip& templatesOfChip);
};

}  // namespace eager_mode::gaudi2_spec_info