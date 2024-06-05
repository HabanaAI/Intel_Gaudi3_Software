#pragma once

#include "../gaudi_tests/gc_resnet_demo_test.h"

class SynGaudi2ResNetFloat8Test : public SynTrainingResNetTest
{
protected:
    SynGaudi2ResNetFloat8Test()
    {
        if (m_deviceType == synDeviceTypeInvalid)
        {
            LOG_WARN(SYN_TEST,
                     "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi2");
            m_deviceType = synDeviceGaudi2;
        }
        setSupportedDevices({synDeviceGaudi2});
    }
    void SetUpTest() override { SynTrainingResNetTest::SetUpTest(); }
};