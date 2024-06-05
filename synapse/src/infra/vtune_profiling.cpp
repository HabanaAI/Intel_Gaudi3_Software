#ifdef VTUNE_ENABLED
#include "vtune_profiling.h"


const __itt_domain* VTuneProfiler::HABANA_DOMAIN = __itt_domain_create("Habana");

#endif