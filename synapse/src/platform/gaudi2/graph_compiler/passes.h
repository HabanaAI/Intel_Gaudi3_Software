#pragma once

#include <string>
#include <memory>
#include "types.h"
#include "habana_pass.h"
#include "gaudi2_graph.h"

class HabanaGraph;

namespace gaudi2
{
template<bool (*applyFunc)(Gaudi2Graph&)>
class Gaudi2Pass : public Pass
{
public:
    Gaudi2Pass(std::string_view name, PassId id, PassIDSet dependencySet)
    : Pass(name, id, PASS_DEF_PRIO, {}, dependencySet)
    {}

    pPass create() const override { return pPass(new Gaudi2Pass<applyFunc>(*this)); }
    bool Apply(HabanaGraph& graph) const override
    {
        try
        {
            return applyFunc(dynamic_cast<Gaudi2Graph&>(graph));
        }
        catch (std::bad_cast& ex)
        {
            HB_ASSERT(0, "Did not get gaudi2 graph");
            return false;
        }
    }

    // Passes that allow predicates should define specialization to override the default.
    bool canRunMultipleTimes() override { return Pass::canRunMultipleTimes(); }
};

#define REGISTER_GAUDI2_PASS(func_, id_, depSet_) addPass(pPass(new Gaudi2Pass<func_>(#func_, (id_), depSet_)))

#define SET_GAUDI2_PASS_CAN_RUN_MULTIPLE_TIMES(pass_) \
    template<> inline bool Gaudi2Pass<pass_>::canRunMultipleTimes() { return true; }

bool splitToLogicalROIs(Gaudi2Graph& g);
bool allocateSyncs(Gaudi2Graph& g);
bool createDMADispatchers(Gaudi2Graph& g);
bool allocateTpcKernels(Gaudi2Graph& g);
bool scheduleFlashAttentionNodes(Gaudi2Graph& g);
bool calculateTensorROIsLinearRanges(Gaudi2Graph& g);
bool manageBaseRegsCache(Gaudi2Graph &g);
bool generateMmeDescriptors(Gaudi2Graph& g);
bool signalOutFromGraph(Gaudi2Graph& g);
bool setGraphNodesPrecision(Gaudi2Graph& g);

bool loadTpcKernels(Gaudi2Graph& g);
SET_GAUDI2_PASS_CAN_RUN_MULTIPLE_TIMES(loadTpcKernels)

bool selectMemcpyEngine(Gaudi2Graph&);
SET_GAUDI2_PASS_CAN_RUN_MULTIPLE_TIMES(selectMemcpyEngine)

} // namespace gaudi2
