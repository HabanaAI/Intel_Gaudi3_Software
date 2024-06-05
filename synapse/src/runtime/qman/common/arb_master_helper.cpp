#include "arb_master_helper.hpp"
#include "runtime/qman/gaudi/master_qmans_definition.hpp"
#include "defs.h"

ArbMasterHelper::ArbMasterHelper(synDeviceType deviceType) : m_deviceType(deviceType)
{
    switch (m_deviceType)
    {
        case synDeviceGaudi:
        {
            m_qmansDef = gaudi::QmansDefinition::getInstance();
            break;
        }
        default:
        {
            HB_ASSERT(false, "Illegal device type {}", m_deviceType);
        }
    }
}

uint64_t ArbMasterHelper::getArbMasterQmanId(synDeviceType deviceType, QmanDefinitionInterface* qmansDef)
{
    uint64_t arbMasterBaseQmanId = 0;

    switch (deviceType)
    {

        case synDeviceGaudi:
        {
            arbMasterBaseQmanId = qmansDef->getArbitratorMasterQueueIdForCompute();
            break;
        }
        default:
        {
            HB_ASSERT(false, "Illegal device type");
        }
    }
    return arbMasterBaseQmanId;
}