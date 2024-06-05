#pragma once

#include <string>
#include <memory>
#include "infra/defs.h"
#include "types.h"
#include "habana_pass.h"
#include "gaudi_graph.h"

class HabanaGraph;

namespace gaudi
{
template<bool (*applyFunc)(GaudiGraph&)>
class GaudiPass : public Pass
{
public:
    GaudiPass(std::string_view name, PassId id, PassIDSet dependencySet)
    : Pass(name, id, PASS_DEF_PRIO, {}, dependencySet)
    {
    }

    pPass create() const override { return pPass(new GaudiPass<applyFunc>(*this)); }
    bool  Apply(HabanaGraph& graph) const override
    {
        try
        {
            return applyFunc(dynamic_cast<GaudiGraph&>(graph));
        }
        catch (std::bad_cast& ex)
        {
            LOG_ERR(GC, "failed getting gaudi graph. pass information: name:[{}], id[{}]", getName(), getId());
            return false;
        }
    }

    // Passes that allow predicates should define specialization to override the default.
    bool canRunMultipleTimes() override { return Pass::canRunMultipleTimes(); }
};

#define REGISTER_GAUDI_PASS(func_, id_, depSet_) addPass(pPass(new GaudiPass<func_>(#func_, (id_), depSet_)))

#define SET_GAUDI_PASS_CAN_RUN_MULTIPLE_TIMES(pass_)                                                                   \
    template<>                                                                                                         \
    inline bool GaudiPass<pass_>::canRunMultipleTimes()                                                                \
    {                                                                                                                  \
        return true;                                                                                                   \
    }

bool allocateSyncs(GaudiGraph&);
bool splitToLogicalROIs(GaudiGraph&);
bool createDMADispatchers(GaudiGraph&);
bool calculateTensorROIsLinearRanges(GaudiGraph&);

bool fuseTransposeGemm(GaudiGraph&);
bool fuseTransposeMME(GaudiGraph&);
bool allocateTpcKernels(GaudiGraph&);
bool signalOutFromGraph(GaudiGraph& g);

bool addH2DOp(GaudiGraph& g);

bool loadTpcKernels(GaudiGraph&);
SET_GAUDI_PASS_CAN_RUN_MULTIPLE_TIMES(loadTpcKernels)

bool selectMemcpyEngine(GaudiGraph&);
SET_GAUDI_PASS_CAN_RUN_MULTIPLE_TIMES(selectMemcpyEngine)

}  // namespace gaudi
