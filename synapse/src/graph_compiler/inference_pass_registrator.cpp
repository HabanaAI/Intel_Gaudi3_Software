#include "inference_pass_registrator.h"
#include "habana_graph.h"
#include "pass_dependencies/inference/passes_dependencies.h"

#define GRAPH_REGISTER_GROUP(name_, id_, groupMembers_, depSet_)                                                       \
    graph.addPass(pPass(new PassGroup(#name_, (id_), groupMembers_, depSet_)))

void InferencePassRegistrator::registerGroups(HabanaGraph& graph) const
{
    PassIDSet noGroupMembers;
    for (PassId i = PASS_ID_INVALID_ID; i < PASS_ID_MAX_ID; i = PassId(i+1))
    {
        noGroupMembers.insert(i);
    }
    noGroupMembers.erase(GROUP_ID_NO_GROUP);
    GRAPH_REGISTER_GROUP(PassesMainGroup, GROUP_ID_NO_GROUP, noGroupMembers, {});
}
