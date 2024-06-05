#include "device_mock.hpp"

#include "runtime/qman/gaudi/master_qmans_definition.hpp"
#include "runtime/qman/gaudi/command_buffer_packet_generator.hpp"

QmanDefinitionInterface* getDeviceQmansDefinition(synDeviceType devType)
{
    QmanDefinitionInterface* pQmansDefinition = nullptr;

    switch (devType)
    {
        case synDeviceGaudi:
        {
            pQmansDefinition = gaudi::QmansDefinition::getInstance();
            break;
        }
        default:
        {
            HB_ASSERT(false, "invalid device type {}", devType);
        }
    };

    return pQmansDefinition;
}

generic::CommandBufferPktGenerator* getDeviceCommandBufferPktGenerator(synDeviceType devType)
{
    generic::CommandBufferPktGenerator* pCmdBuffPktGenerator = nullptr;

    switch (devType)
    {
        case synDeviceGaudi:
        {
            pCmdBuffPktGenerator = gaudi::CommandBufferPktGenerator::getInstance();
            break;
        }
        default:
        {
            HB_ASSERT(false, "invalid device type {}", devType);
        }
    };

    return pCmdBuffPktGenerator;
}