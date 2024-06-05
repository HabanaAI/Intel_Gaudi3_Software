#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/sync_scheme_manager_base.h"

namespace eager_mode::gaudi3_spec_info
{

class SyncSchemeManager final : public SyncSchemeManagerBase
{
public:
    explicit constexpr SyncSchemeManager(const DescGeneratorHal& descGenHal) : SyncSchemeManagerBase(descGenHal) {}
    void generateWorkDistributionContexts(Node2DescContainer& multiNode2Desc) const override;
};

}  // namespace eager_mode::gaudi3_spec_info