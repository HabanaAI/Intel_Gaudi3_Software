#pragma once

#include <string>
#include <algorithm>
#include "size_param.hpp"

namespace hl_gcfg
{
HLGCFG_INLINE_NAMESPACE{

template<class T>
std::string toString(T const & value)
{
    return std::to_string(value);
}

inline std::string toString(std::string const & value)
{
    return value;
}

inline std::string toString(std::string && value)
{
    return std::move(value);
}

template<class T>
T fromString(const std::string& str);

template<>
inline hl_gcfg::SizeParam fromString(const std::string& str)
{
    // read the size with optional units
    // -- default is bytes
    // -- 1K == 1024 (not 1000)
    // -- example valid inputs: 64K, 512kb, 32mb, 16g, 40GB
    hl_gcfg::SizeParam res(str);
    if (!res.isValid())
    {
        throw std::invalid_argument(str);
    }
    return res;
}

template<>
inline uint64_t fromString(const std::string& str)
{
    std::string::size_type nextCharIdx = 0;
    uint64_t res = std::stoull(str, &nextCharIdx, 0);
    // stoull will succeed even if only the FIRST part of the string is valid (i.e "777NG"), but we'll be more strict -
    // unless the WHOLE string represents a number - fail
    if (nextCharIdx != str.length())
    {
        throw std::invalid_argument(str);
    }
    return res;
}

template<>
inline int64_t fromString(const std::string& str)
{
    std::string::size_type nextCharIdx = 0;
    int64_t res = std::stoll(str, &nextCharIdx, 0);
    // stoll will succeed even if only the FIRST part of the string is valid (i.e "777NG"), but we'll be more strict -
    // unless the WHOLE string represents a number - fail
    if (nextCharIdx != str.length())
    {
        throw std::invalid_argument(str);
    }
    return res;
}

template<>
inline float fromString(const std::string& str)
{
    std::string::size_type nextCharIdx = 0;
    float res = std::stof(str, &nextCharIdx);
    // stof will succeed even if only the FIRST part of the string is valid (i.e "7.77NG"), but we'll be more strict -
    // unless the WHOLE string represents a number - fail
    if (nextCharIdx != str.length())
    {
        throw std::invalid_argument(str);
    }
    return res;
}

template<>
inline bool fromString(const std::string& str)
{
    std::string loweredStr(str);
    std::transform(loweredStr.begin(), loweredStr.end(), loweredStr.begin(), tolower);

    if (loweredStr == "false" || str == "0")
    {
        return false;
    }
    else if (loweredStr == "true" || str == "1")
    {
        return true;
    }

    throw std::invalid_argument(str);
}

template<>
inline std::string fromString(const std::string& str)
{
    return str;
}

}}
