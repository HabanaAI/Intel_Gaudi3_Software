#pragma once

#include <string>
#include <memory>
#include "types.h"
#include "habana_pass.h"
#include "gaudi3_graph.h"
#include "infra/defs.h"

class HabanaGraph;

namespace gaudi3
{
template<bool (*applyFunc)(Gaudi3Graph&)>
class Gaudi3Pass : public Pass
{
public:
    Gaudi3Pass(std::string_view name, PassId id, PassIDSet dependencySet)
    : Pass(name, id, PASS_DEF_PRIO, {}, dependencySet)
    {
    }

    pPass create() const override { return pPass(new Gaudi3Pass<applyFunc>(*this)); }
    bool  Apply(HabanaGraph& graph) const override
    {
        try
        {
            return applyFunc(dynamic_cast<Gaudi3Graph&>(graph));
        }
        catch (std::bad_cast& ex)
        {
            HB_ASSERT(0, "Did not get gaudi3 graph");
            return false;
        }
    }

    // Passes that allow predicates should define specialization to override the default.
    bool canRunMultipleTimes() override { return Pass::canRunMultipleTimes(); }
};

#define REGISTER_GAUDI3_PASS(func_, id_, depSet_) addPass(pPass(new Gaudi3Pass<func_>(#func_, (id_), depSet_)))

#define SET_GAUDI3_PASS_CAN_RUN_MULTIPLE_TIMES(pass_)                                                                  \
    template<>                                                                                                         \
    inline bool Gaudi3Pass<pass_>::canRunMultipleTimes()                                                               \
    {                                                                                                                  \
        return true;                                                                                                   \
    }

bool splitToLogicalROIs(Gaudi3Graph& g);
bool splitToDcoreROIs(Gaudi3Graph& g);
bool allocateSyncs(Gaudi3Graph& g);
bool calculateTensorROIsLinearRanges(Gaudi3Graph& g);
bool generateCacheMaitenanceTasks(Gaudi3Graph& g);
bool manageBaseRegsCache(Gaudi3Graph& g);
bool allocateTpcKernels(Gaudi3Graph& g);
bool generateMmeDescriptors(Gaudi3Graph& g);
bool patchMmeDescriptors(Gaudi3Graph& g);
bool patchMmeMcids(Gaudi3Graph& g);
bool setTensorInHbm(Gaudi3Graph& g);
bool alignTransposeViaGemmOutput(Gaudi3Graph& g);
bool signalOutFromGraph(Gaudi3Graph& g);

bool loadTpcKernels(Gaudi3Graph& g);
SET_GAUDI3_PASS_CAN_RUN_MULTIPLE_TIMES(loadTpcKernels)

bool selectMemcpyEngine(Gaudi3Graph&);
SET_GAUDI3_PASS_CAN_RUN_MULTIPLE_TIMES(selectMemcpyEngine)

}  // namespace gaudi3