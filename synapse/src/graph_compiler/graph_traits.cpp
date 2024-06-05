#include "graph_traits.h"

#include "hal_reader/hal_reader.h"
#include "infra/defs.h"
#include "platform/gaudi/graph_compiler/engine_selector.h"
#include "utils.h"

extern HalReaderPtr instantiateGrecoHalReader();
extern HalReaderPtr instantiateGaudiHalReader();
extern HalReaderPtr instantiateGaudi2HalReader();
extern HalReaderPtr instantiateGaudi3HalReader();

GraphTraits::GraphTraits(synDeviceType device, CompilationMode mode)
: m_isTraining(false),
  m_isQuantizationEnabled(false),
  m_backoffFactor(1.0),
  m_compilationMode(mode),
  m_deviceId(deviceTypeToDeviceID(device))
{
    switch (device)
    {
        case synDeviceGaudi:
            m_isTraining = true;
            m_halReader  = instantiateGaudiHalReader();
            break;
        case synDeviceGaudi2:
            m_isTraining = true;
            m_halReader  = instantiateGaudi2HalReader();
            break;
        case synDeviceGaudi3:
            m_isTraining = true;
            m_halReader  = instantiateGaudi3HalReader();
            break;
        default:
            HB_ASSERT(false, "Device has no traits");
    }
}

void GraphTraits::setTrainingGraph(bool mode)
{
    LOG_TRACE(GC, "Setting GraphTraits training mode to {}", mode ? "true" : "false");
    m_isTraining = mode;
}

void GraphTraits::setQuantizationEnabled(bool enable)
{
    LOG_TRACE(GC, "Setting GraphTraits quantization enabled to {}", enable ? "true" : "false");
    m_isQuantizationEnabled = enable;
}

void GraphTraits::setBackoffFactor(double boFactor)
{
    LOG_TRACE(GC, "Setting GraphTraits backoff factor to {}", boFactor);
    m_backoffFactor = boFactor;
}