#pragma once

#include <memory>
#include <cstring>
#include "synapse_common_types.h"
#include "compiler_types.h"
#include "tpc_kernel_lib_interface.h"

class HalReader;
namespace gc
{
class EngineSelector;
}

class GraphTraits
{
public:
    GraphTraits(synDeviceType device, CompilationMode mode = CompilationMode::Graph);

    bool trainingGraph() const { return m_isTraining; }

    bool inferenceGraph() const { return !m_isTraining; }

    bool isQuantizationEnabled() const { return m_isQuantizationEnabled; }

    double backoffFactor() const { return m_backoffFactor; }

    const std::shared_ptr<HalReader>& getHalReader() const { return m_halReader; }

    CompilationMode getCompilationMode() const { return m_compilationMode; }

    tpc_lib_api::DeviceId getDeviceId() const { return m_deviceId; };

    void setTrainingGraph(bool mode);

    void setQuantizationEnabled(bool enabled);

    void setBackoffFactor(double boFactor);

private:
    std::shared_ptr<HalReader> m_halReader;

    bool              m_isTraining;
    bool              m_isQuantizationEnabled;
    double            m_backoffFactor;
    CompilationMode   m_compilationMode;
    tpc_lib_api::DeviceId m_deviceId;
};
