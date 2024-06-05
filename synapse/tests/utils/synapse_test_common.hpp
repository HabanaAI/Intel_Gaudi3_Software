#pragma once

#include "synapse_test.hpp"

class SynTestCommon : public SynTest
{
public:
    SynTestCommon() {}
    SynTestCommon(synDeviceType deviceType) { m_deviceType = deviceType; }
};
