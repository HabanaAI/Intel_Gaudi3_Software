#pragma once

#include <string>

/**
 * Set global conf to a test wanted value
 * Return the global conf to old value on destruction
 */
class GlobalConfTestSetter
{
public:
    explicit GlobalConfTestSetter(const std::string& gConf, const std::string& value);

    virtual ~GlobalConfTestSetter();

private:
    static constexpr uint64_t valueSize = 1024;
    char m_oldValue[valueSize];
    const std::string m_gConf;
};
