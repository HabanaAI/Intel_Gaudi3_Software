#include "memcpy_engine_manager.h"
#include "gaudi2_graph.h"

namespace gaudi2
{
    bool selectMemcpyEngine(Gaudi2Graph& g)
    {
        return MemcpyEngineManager().selectEngine(g);
    }
}
