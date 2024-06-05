#include "global_conf_test_setter.h"

#include "synapse_api.h"
#include "log_manager.h"

GlobalConfTestSetter::GlobalConfTestSetter(const std::string& gConf, const std::string& value)
: m_gConf(gConf)
{
    auto status = synConfigurationGet(m_gConf.c_str(), m_oldValue, valueSize);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "synConfigurationGet failed to get configuration {}", gConf);
    }
    status = synConfigurationSet(m_gConf.c_str(), value.c_str());
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "synConfigurationSet failed to set configuration {} to {}", gConf, value);
    }
}

GlobalConfTestSetter::~GlobalConfTestSetter()
{
    auto status = synConfigurationSet(m_gConf.c_str(), m_oldValue);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "synConfigurationSet failed to set configuration to old value {} to {}", m_gConf, m_oldValue);
    }
}
