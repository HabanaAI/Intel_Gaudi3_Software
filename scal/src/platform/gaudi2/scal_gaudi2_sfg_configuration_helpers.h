#pragma once
#include "common/scal_sfg_configuration_helpers.h"
#include "infra/monitor.hpp"
#include "infra/sob.hpp"

struct Gaudi2SobTypes
{
    using MonitorX       = MonitorG2;
    using SobX           = SobG2;
};

using sfgMonitorsHierarchyMetaDataG2 = SfgMonitorsHierarchyMetaData<Gaudi2SobTypes>;

static inline int getMmeEnginesMultiplicationFactor()
{
    return 1;
}
static inline int getNbMmeCompletionSignals()
{
    return 2;
}
