#include "master_qmans_definition_interface.hpp"

#include "defs.h"

#include "runtime/qman/gaudi/master_qmans_definition.hpp"

QmanDefinitionInterface* getQmansDefinition(synDeviceType deviceType)
{
    QmanDefinitionInterface* qmansDef = nullptr;
    switch (deviceType)
    {
        case synDeviceGaudi:
        {
            qmansDef = gaudi::QmansDefinition::getInstance();
            break;
        }
        default:
        {
            HB_ASSERT(false, "Illegal device type {}", deviceType);
        }
    }
    return qmansDef;
}
