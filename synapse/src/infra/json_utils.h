#pragma once

#include "json.hpp"
#include <optional>

namespace json_utils
{
using Json = nlohmann_hcl::json;

Json jsonFromFile(const std::string& filePath);

void jsonToFile(const Json& jsonObject, const std::string& filePath, unsigned indent = -1);

std::string toString(const json_utils::Json& j);

// for mandatory fields
const Json& get(const Json& j, const std::string& f);

// for non mandatory fields
template<class T>
T get(const Json& j, const std::string& f, T defaultValue)
{
    return j.count(f) != 0 ? j.at(f).get<T>() : defaultValue;
}
// for non mandatory fields
template<class T>
T get_or_default(const Json& j, const std::string& f)
{
    return j.count(f) != 0 ? j.at(f).get<T>() : T {};
}
// for non mandatory fields
template<class T>
std::optional<T> get_opt(const Json& j, const std::string& f)
{
    return j.count(f) != 0 ? std::optional<T>{j.at(f).get<T>()} : std::optional<T>{};
}

template<class T>
std::optional<T> get(const Json& data, const std::vector<std::string>& keys)
{
    if (keys.empty()) return std::optional<T>();
    Json ret = data;
    for (auto it = keys.begin(); it != keys.end(); ++it)
    {
        auto next = ret.find(*it);
        if (next == ret.end()) return std::optional<T>();
        ret = *next;
    }
    return ret;
}

inline bool isEmpty(const Json& data)
{
    return (data.size() == 0) || data.is_null();
}

}  // namespace json_utils
