#pragma once

#include "synapse_common_types.h"

class QmanDefinitionInterface;

class ArbMasterHelper
{
public:
    ArbMasterHelper(synDeviceType deviceType);

    virtual ~ArbMasterHelper() = default;

    uint64_t getArbMasterQmanId() const { return getArbMasterQmanId(m_deviceType, m_qmansDef); }

private:
    static uint64_t getArbMasterQmanId(synDeviceType deviceType, QmanDefinitionInterface* qmansDef);

    const synDeviceType m_deviceType;

    QmanDefinitionInterface* m_qmansDef {nullptr};
};