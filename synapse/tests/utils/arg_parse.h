#pragma once

#include "infra/log_manager.h"
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

class ArgParser
{
    enum ArgType
    {
        FLAG,
        SINGLE,
        MULTI
    };

    template<typename T>
    static T fromString(const std::string& str)
    {
        std::stringstream ss(str);
        T                 value;
        ss >> value;
        return value;
    }

public:
    struct Arg
    {
        Arg(ArgType t, std::string sn, std::string ln, std::string dv, std::string h, std::set<std::string> ov, bool r)
        : type(t), shortName(sn), longName(ln), help(h), defaultValue(dv), optionalValues(ov), required(r)
        {
        }

        ArgType                  type;
        std::string              shortName;
        std::string              longName;
        std::string              help;
        std::string              defaultValue;
        std::set<std::string>    optionalValues;
        std::vector<std::string> values;
        bool                     required;

        bool hasValue() { return !values.empty(); }

        template<typename T>
        T getValue()
        {
            if (values.empty())
            {
                throw std::runtime_error(fmt::format("try to get value from empty values list, arg: {}", longName));
            }
            return fromString<T>(values.front());
        }

        template<typename T>
        T getValueOrDefault(const T& defaultValue)
        {
            if (values.empty()) return defaultValue;
            return fromString<T>(values.front());
        }

        template<typename T>
        std::vector<T> getValues()
        {
            std::vector<T> ret;
            ret.reserve(values.size());
            for (const auto& v : values)
            {
                ret.emplace_back(fromString<T>(v));
            }
            return ret;
        }
    };

    template<typename T>
    class ArgValue
    {
    public:
        ArgValue(const std::shared_ptr<Arg>& arg) : m_arg(arg) {}

        bool           hasValue() { return m_arg->hasValue(); }
        T              getValue() { return m_arg->getValue<T>(); }
        T              getValueOrDefault(const T& defaultValue) { return m_arg->getValueOrDefault<T>(defaultValue); }
        std::vector<T> getValues() { return m_arg->getValues<T>(); }
        explicit       operator bool() const { return !m_arg->values.empty(); }

    private:
        std::shared_ptr<Arg> m_arg;
    };

    ArgParser(const std::string& activationKey = "")
    : m_help(add("-h", "--help", false, "show help menu")), m_activationKey(activationKey)
    {
    }

    std::string toString(const std::set<std::string>& strings)
    {
        return strings.empty() ? "" : fmt::format("{{ {} }}", fmt::join(strings.begin(), strings.end(), ", "));
    }

    bool helpRequested() { return m_help.getValue(); }

    std::string getHelp()
    {
        std::stringstream arguments;
        arguments << "arguments:" << std::endl;
        uint32_t snLength = 0;
        uint32_t lnLength = 0;
        uint32_t ovLength = 0;
        uint32_t hLength  = 0;
        uint32_t dLength  = 0;

        for (const auto& a : m_longNames)
        {
            auto arg     = a.second;
            auto optVals = toString(arg->optionalValues);
            if (optVals.size() > ovLength) ovLength = optVals.size();
            if (arg->shortName.size() > snLength) snLength = arg->shortName.size();
            if (arg->longName.size() > lnLength) lnLength = arg->longName.size();
            if (arg->help.size() > hLength) hLength = arg->help.size();
            if (arg->defaultValue.size() > dLength) dLength = arg->defaultValue.size();
        }
        for (const auto& a : m_longNames)
        {
            auto arg     = a.second;
            auto optVals = toString(arg->optionalValues);
            arguments << std::left << "     ";
            arguments << std::setw(snLength + 4) << arg->shortName;
            arguments << std::setw(lnLength + 4) << arg->longName;
            arguments << std::setw(ovLength + 4) << optVals;
            arguments << std::setw(hLength + 4) << arg->help;
            if (!arg->defaultValue.empty())
                arguments << std::setw(dLength + 4) << "(default: " << arg->defaultValue + ")";
            arguments << std::endl;
        }
        return arguments.str();
    }

    void parse(int argc, char* argv[])
    {
        std::map<std::string, std::shared_ptr<Arg>> argMap;
        argMap.insert(m_shortNames.begin(), m_shortNames.end());
        argMap.insert(m_longNames.begin(), m_longNames.end());
        argMap.erase("");
        std::shared_ptr<Arg> lastArg = nullptr;
        if (!m_activationKey.empty())
        {
            if (std::string(*++argv) != m_activationKey) return;
            --argc;
        }
        while (++argv, --argc)
        {
            std::string curr = std::string(*argv);
            if (argMap.find(curr) != argMap.end())
            {
                auto currArg = argMap.at(curr);
                if (lastArg && (lastArg->type == SINGLE || (lastArg->type == MULTI && lastArg->values.empty())))
                {
                    throw std::runtime_error("missing value for argument: " + lastArg->longName);
                }
                currArg->values.clear();
                if (currArg->type == FLAG)
                {
                    currArg->values.push_back(currArg->defaultValue == "0" ? "1" : "0");
                    lastArg = nullptr;
                }
                else
                {
                    lastArg = currArg;
                }
                continue;
            }
            else if (lastArg != nullptr)
            {
                if (!lastArg->optionalValues.empty() &&
                    lastArg->optionalValues.find(curr) == lastArg->optionalValues.end())
                {
                    throw std::runtime_error("argument: " + curr + ", is not an optional value");
                }

                lastArg->values.push_back(curr);
                if (lastArg->type == SINGLE) lastArg = nullptr;
                continue;
            }
            throw std::runtime_error("invalid argument: " + curr);
        }
        m_parsed = true;
        if (helpRequested()) return;
        for (const auto& e : argMap)
        {
            if (e.second->required && e.second->values.empty())
            {
                throw std::runtime_error("missing required argument: " + e.first);
            }
        }
    }

    ArgValue<bool>
    add(const std::string& shortName, const std::string& longName, bool defaultValue, const std::string& help)
    {
        return add<bool>(FLAG, shortName, longName, defaultValue ? "1" : "0", help, {}, false);
    }

    template<typename T>
    ArgValue<T> add(const std::string&    shortName,
                    const std::string&    longName,
                    const std::string&    defaultValue,
                    const std::string&    help,
                    std::set<std::string> optionalValues,
                    bool                  required = false)
    {
        return add<T>(SINGLE, shortName, longName, defaultValue, help, optionalValues, required);
    }

    template<typename T>
    ArgValue<T> addMulti(const std::string&    shortName,
                         const std::string&    longName,
                         const std::string&    defaultValue,
                         const std::string&    help,
                         std::set<std::string> optionalValues,
                         bool                  required = false)
    {
        return add<T>(MULTI, shortName, longName, defaultValue, help, optionalValues, required);
    }

    bool contains(const std::string& longName) const { return m_longNames.find(longName) != m_longNames.end(); }

    template<typename T>
    ArgValue<T> getArg(const std::string& longName) const
    {
        if (!contains(longName))
        {
            throw std::runtime_error(fmt::format("argument: {} doesn't exist", longName));
        }
        return ArgValue<T>(m_longNames.at(longName));
    }

    template<typename T>
    T getValue(const std::string& longName) const
    {
        if (!contains(longName))
        {
            throw std::runtime_error(fmt::format("argument: {} doesn't exist", longName));
        }
        return ArgValue<T>(m_longNames.at(longName)).getValue();
    }

    template<typename T>
    T getValueOrDefault(const std::string& longName, const T& defaultVal) const
    {
        if (contains(longName))
        {
            return ArgValue<T>(m_longNames.at(longName)).getValueOrDefault(defaultVal);
        }
        return defaultVal;
    }

    template<typename T>
    std::vector<T> getValues(const std::string& longName) const
    {
        if (contains(longName))
        {
            return ArgValue<T>(m_longNames.at(longName)).getValues();
        }
        return {};
    }

    explicit operator bool() const { return m_parsed; }

private:
    template<typename T>
    ArgValue<T> add(ArgType               type,
                    const std::string&    shortName,
                    const std::string&    longName,
                    const std::string&    defaultValue,
                    const std::string&    help,
                    std::set<std::string> optionalValues,
                    bool                  required)
    {
        if (m_longNames.count(longName) != 0)
        {
            throw std::runtime_error("long name already used: " + longName);
        }
        if (!shortName.empty() && m_shortNames.count(shortName) != 0)
        {
            throw std::runtime_error("short name already used: " + shortName);
        }
        auto arg = std::make_shared<Arg>(type, shortName, longName, defaultValue, help, optionalValues, required);

        m_shortNames.insert(std::pair<std::string, std::shared_ptr<Arg>>(shortName, arg));
        m_longNames.insert(std::pair<std::string, std::shared_ptr<Arg>>(longName, arg));
        auto ret = ArgValue<T>(m_longNames.at(longName));
        if (!defaultValue.empty())
        {
            m_longNames.at(longName)->values.push_back(defaultValue);
        }
        return ret;
    }

    std::unordered_map<std::string, std::shared_ptr<Arg>> m_shortNames;
    std::unordered_map<std::string, std::shared_ptr<Arg>> m_longNames;
    ArgValue<bool>                                        m_help;
    std::string                                           m_activationKey;
    bool                                                  m_parsed = false;
};