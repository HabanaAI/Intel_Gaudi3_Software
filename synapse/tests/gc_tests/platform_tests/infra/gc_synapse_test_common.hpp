#pragma once

#include "infra/gc_synapse_test.h"

class SynTestCommon : public SynTest
{
public:
    SynTestCommon() {}
    SynTestCommon(synDeviceType deviceType) { m_deviceType = deviceType; }
};
