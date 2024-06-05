#pragma once
#include <algorithm>
#include <string>
#include <cstdlib>

#ifdef SWTOOLS_DEP
#include <hl_gcfg/hlgcfg_item.hpp>

using hl_gcfg::MakePrivate;
using hl_gcfg::MakePublic;

using GlobalConfBool = hl_gcfg::GcfgItemBool;
using GlobalConfString = hl_gcfg::GcfgItemString;

#else

template<class T>
class MmeGlobalConfItem
{
public:
    MmeGlobalConfItem(const std::string& name,
                      const std::string& description,
                      const T& defaultValue,
                      bool isPublic = false)
    : m_name(name), m_value(defaultValue)
    {
        setValue(name);
    }

    const T& value() const { return m_value; }

private:
    template<typename U = T>
    typename std::enable_if<std::is_same<U, std::string>::value, void>::type setValue(const std::string& name)
    {
        const char* value = getenv(name.c_str());
        if (value)
        {
            m_value = std::string(value);
        }
    }

    template<typename U = T>
    typename std::enable_if<std::is_same<U, bool>::value, void>::type setValue(const std::string& name)
    {
        const char* value = getenv(name.c_str());
        if (!value) return;
        std::string str(value);
        std::string loweredStr(str);
        std::transform(loweredStr.begin(), loweredStr.end(), loweredStr.begin(), tolower);

        if (loweredStr == "false" || str == "0")
        {
            m_value = false;
        }
        else if (loweredStr == "true" || str == "1")
        {
            m_value = true;
        }
    }

    std::string m_name;
    T m_value;
};

constexpr bool MakePublic = true;
constexpr bool MakePrivate = !MakePublic;

using GlobalConfBool = MmeGlobalConfItem<bool>;
using GlobalConfString = MmeGlobalConfItem<std::string>;
#endif
