#include "gaudi2_test_infra.h"
#include "synapse_api.h"
#include "tensor_validator.inl"

#include "global_conf_manager.h"

#include "syn_singleton.hpp"

const SynGaudi2TestInfra::TENSOR_INDEX SynGaudi2TestInfra::INVALID_TENSOR_INDEX = -1;
const std::string                      SynGaudi2TestInfra::INVALID_RECIPE_FILE_NAME("");
const std::string                      SynGaudi2TestInfra::FILE_NAME_SUFFIX("_");

SynGaudi2TestInfra::SynGaudi2TestInfra()
: m_streamHandleDownload(0), m_streamHandleCompute(0), m_streamHandleUpload(0), m_eventHandle(0)
{
    if (m_deviceType == synDeviceTypeInvalid)
    {
        LOG_WARN(SYN_TEST,
                 "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi2");
        m_deviceType = synDeviceGaudi2;
    }
    setSupportedDevices({synDeviceGaudi2});
};
