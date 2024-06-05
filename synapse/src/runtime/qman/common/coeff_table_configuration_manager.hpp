#pragma once

#include "synapse_types.h"
#include "log_manager.h"
#include "runtime/qman/common/master_qmans_definition_interface.hpp"

class DeviceGaudi;
namespace generic
{
class CommandBufferPktGenerator;
}
class HalReader;

class CoeffTableConfManager
{
public:
    CoeffTableConfManager(DeviceGaudi* device);
    virtual ~CoeffTableConfManager() = default;

    synStatus submitCoeffTableConfiguration(const synDeviceType deviceType);

protected:
    virtual uint64_t getSpecialFuncCoeffTableAllocatedSize() = 0;
    virtual uint64_t getSpecialFuncCoeffTableSize()          = 0;
    virtual void*    getSpecialFuncCoeffTableData()          = 0;

    generic::CommandBufferPktGenerator* m_cmdBuffPktGenerator        = nullptr;
    std::shared_ptr<HalReader>          m_halReader                  = nullptr;
    QmanDefinitionInterface*            m_qmanDefs                   = nullptr;
    bool                                m_isSyncWithExternalRequired = false;
    bool                                m_isConfigOnInternal         = false;

private:
    synStatus generateCoeffTableConfigurationPackets(char*& pPackets, uint64_t& packetsSize, uint64_t tableBaseAddr);

    DeviceGaudi* m_device;
};

std::unique_ptr<CoeffTableConfManager> createCoeffTableConfManager(DeviceGaudi* device);