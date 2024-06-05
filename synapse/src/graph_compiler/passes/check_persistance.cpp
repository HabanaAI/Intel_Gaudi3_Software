#include "passes.h"
#include "habana_graph.h"

bool checkInputPersistence(HabanaGraph& g)
{
    std::list<pTensor> inputs = g.getGraphInputs();
    for (auto& t : inputs)
    {
        CHECK_RET_FALSE(
            g.isUserManagedDram(t) || t->isStaticParam() || t->isShapeTensor() || t->isHost2DeviceTensor(),
            "All Inputs should be declared as persistent, User section ,constant, or non-device shape tensors! ({})",
            t->getName());
    }
    return true;
}
