#pragma once
#include "common/scal_sfg_configuration_helpers.h"
#include "infra/monitor.hpp"
#include "infra/sob.hpp"


struct Gaudi3SobTypes
{
    using MonitorX       = MonitorG3;
    using SobX           = SobG3;
};

using sfgMonitorsHierarchyMetaDataG3 = SfgMonitorsHierarchyMetaData<Gaudi3SobTypes>;

static inline int getMmeEnginesMultiplicationFactor()
{
    return 2;
}

static inline int getNbMmeCompletionSignals()
{
    return 2;
}
