#pragma once

#include "syn_base_test.hpp"
#include "habana_global_conf.h"

class DsdRecipeTestBase : public SynBaseTest
{
public:
    DsdRecipeTestBase() : SynBaseTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }
    bool m_gaudi3DSD;

public:
    void SetUp()
    {
        SynBaseTest::SetUp();
    }

    void TearDown()
    {
        SynBaseTest::TearDown();
    }
};
