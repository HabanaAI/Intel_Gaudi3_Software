#include "graph_factory.h"
#include "sim_graph.h"
#include "eager/eager_interface.h"
#include "infra/global_conf_manager.h"

extern HabanaGraphPtr instantiateGaudiGraph();
extern HabanaGraphPtr instantiateGaudi2Graph();
extern HabanaGraphPtr instantiateGaudi3Graph();

HabanaGraphPtr GraphFactory::createGraph(synDeviceType deviceType, CompilationMode compilationMode)
{
    compilationMode = GCFG_FORCE_EAGER.value() == 1
                          ? CompilationMode::Eager
                          : (GCFG_FORCE_EAGER.value() == 0 ? CompilationMode::Graph : compilationMode);

    GlobalConfManager::instance().setDeviceType(deviceType);

    if (compilationMode == CompilationMode::Eager)
    {
        HB_ASSERT(eager_mode::isValidForEager(deviceType), "Eager mode compilation is not supported for the device");
        return createEagerGraph(deviceType);
    }

    switch (deviceType)
    {
        case synDeviceGaudi:
            return createGaudiGraph();

        case synDeviceGaudi2:
            return createGaudi2Graph();

        case synDeviceGaudi3:
            return createGaudi3Graph();

        case synDeviceEmulator:
        case synDeviceTypeInvalid:
        case synDeviceTypeSize:
        default:
            return createSimGraph();
    }

    // Cannot get here
    return createSimGraph();
}

HabanaGraphPtr GraphFactory::createSimGraph()
{
    return std::make_unique<SimGraph>();
}

HabanaGraphPtr GraphFactory::createGaudiGraph()
{
    return instantiateGaudiGraph();
}

HabanaGraphPtr GraphFactory::createGaudi2Graph()
{
    return instantiateGaudi2Graph();
}

HabanaGraphPtr GraphFactory::createGaudi3Graph()
{
    return instantiateGaudi3Graph();
}

HabanaGraphPtr GraphFactory::createEagerGraph(synDeviceType deviceType)
{
    return HabanaGraphPtr {eager_mode::createEagerGraph(deviceType)};
}