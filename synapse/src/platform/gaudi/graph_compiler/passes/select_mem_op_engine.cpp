#include "memcpy_engine_manager.h"
#include "gaudi_graph.h"

namespace gaudi
{
    bool selectMemcpyEngine(GaudiGraph& g)
    {
        return MemcpyEngineManager().selectEngine(g);
    }
}
