#pragma once

#include "synapse_common_types.h"
#include <functional>
#include <memory>
#include <set>
#include <vector>

class Node;
class HabanaGraph;

class GcPerf
{
public:
    enum class LogLevel
    {
        LOW  = 0,
        HIGH = 1
    };

    GcPerf(const HabanaGraph& g);
    class GcPerfCheck
    {
    public:
        virtual void run(const std::shared_ptr<Node>& node) const = 0;
        virtual ~GcPerfCheck()                                    = default;

    protected:
        virtual std::string_view name() const = 0;
        GcPerfCheck(const HabanaGraph& g) : m_graph(g) {}

        const HabanaGraph& m_graph;
    };
    using GcPerfCheckPtr = std::unique_ptr<const GcPerfCheck>;

    void execute() const;

private:
    void registerCheck(const std::function<GcPerfCheckPtr(const HabanaGraph&)> func,
                       const std::set<synDeviceType>&                          devices = {});

    const HabanaGraph&          m_graph;
    std::vector<GcPerfCheckPtr> m_checks;
};

// This macro can only be used within classes that inherit from GcPerfCheck
#define PERF_REPORT(level, msg, ...)                                                                                   \
    HLLOG_WARN(GC_PERF, "[{}][{}] " msg, level == GcPerf::LogLevel::LOW ? "LOW " : "HIGH", name(), ##__VA_ARGS__)
