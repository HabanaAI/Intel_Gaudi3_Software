#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_gen_hal.h"
#include "utils/general_defs.h"

namespace eager_mode::gaudi3_spec_info
{
class DescGeneratorHal final : public eager_mode::DescGeneratorHal
{
public:
    unsigned deviceTypeToLogicalQueue(EngineType engineType, const Node& node) const override;
    unsigned safeIncrement(unsigned logicalId, unsigned sigVal, unsigned incAmount) const override;
};

}  // namespace eager_mode::gaudi3_spec_info