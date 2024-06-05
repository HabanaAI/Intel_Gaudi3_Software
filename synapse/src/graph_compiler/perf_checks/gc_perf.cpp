#include "gc_perf.h"
#include "small_fcd_perf_check.h"
#include "strided_view_insert_perf_check.h"
#include "types.h"
#include "habana_graph.h"

GcPerf::GcPerf(const HabanaGraph& g) : m_graph(g)
{
    // Register all checks here, the registration function argument is set of devices to perform the checks (empty=all)
    registerCheck(SmallFcdPerfCheck::createPerfCheck);
    registerCheck(StridedOpPerfCheck::createPerfCheck);
}

void GcPerf::execute() const
{
    for (const NodePtr& node : m_graph.getNodes())
    {
        for (const auto& perfCheck : m_checks)
        {
            perfCheck->run(node);
        }
    }
}

bool gcPerfChecks(HabanaGraph& g)
{
    GcPerf(g).execute();
    return true;
}

void GcPerf::registerCheck(const std::function<GcPerfCheckPtr(const HabanaGraph&)> func,
                           const std::set<synDeviceType>&                          devices)
{
    if (devices.empty() || devices.count(m_graph.getDeviceType()))
    {
        m_checks.push_back(func(m_graph));
    }
}