#pragma once
#include <cassert>
#include <exception>
#include <string>
#include "json.hpp"

// helper macro for in-struct enum serialization declaration
// the same as in json.hpp tonly difference - 'friend' instaed of 'inline'
#define scaljson_JSON_SERIALIZE_ENUM_INTRUSIVE(ENUM_TYPE, ...)                                  \
    template<typename BasicJsonType>                                                            \
    friend void to_json(BasicJsonType& j, const ENUM_TYPE& e)                                   \
    {                                                                                           \
        static_assert(std::is_enum<ENUM_TYPE>::value, #ENUM_TYPE " must be an enum!");          \
        static const std::pair<ENUM_TYPE, BasicJsonType> m[] = __VA_ARGS__;                     \
        auto it = std::find_if(std::begin(m), std::end(m),                                      \
                               [e](const std::pair<ENUM_TYPE, BasicJsonType>& ej_pair) -> bool  \
        {                                                                                       \
            return ej_pair.first == e;                                                          \
        });                                                                                     \
        j = ((it != std::end(m)) ? it : std::begin(m))->second;                                 \
    }                                                                                           \
    template<typename BasicJsonType>                                                            \
    friend void from_json(const BasicJsonType& j, ENUM_TYPE& e)                                 \
    {                                                                                           \
        static_assert(std::is_enum<ENUM_TYPE>::value, #ENUM_TYPE " must be an enum!");          \
        static const std::pair<ENUM_TYPE, BasicJsonType> m[] = __VA_ARGS__;                     \
        auto it = std::find_if(std::begin(m), std::end(m),                                      \
                               [&j](const std::pair<ENUM_TYPE, BasicJsonType>& ej_pair) -> bool \
        {                                                                                       \
            return ej_pair.second == j;                                                         \
        });                                                                                     \
        e = ((it != std::end(m)) ? it : std::begin(m))->first;                                  \
    }

class ExceptionWithCode : public std::exception
{
public:
    ExceptionWithCode(int errCode, std::string desc)
    : m_errCode(errCode)
    , m_desc(std::move(desc))
    {}
    int code() const { return m_errCode; }
    const char* what() const noexcept override
    {
        return m_desc.c_str();
    }
private:
    int m_errCode;
    std::string m_desc;
};

struct JsonDisgnosticsAccess : scaljson::detail::exception
{
    using scaljson::detail::exception::diagnostics;
};


#define THROW_CONFIG_ERROR(errorCode, jsonNode, msgFmt, ...) \
    throw ExceptionWithCode(errorCode, fmt::format(FMT_COMPILE("{}({}): json:{} " msgFmt), \
                            __FUNCTION__, __LINE__, JsonDisgnosticsAccess::diagnostics(jsonNode), ##__VA_ARGS__));

#define THROW_INVALID_CONFIG(jsonNode, msgFmt, ...)          \
    throw ExceptionWithCode(SCAL_INVALID_CONFIG, fmt::format(FMT_COMPILE("{}({}): json:{} " msgFmt), \
                            __FUNCTION__, __LINE__, JsonDisgnosticsAccess::diagnostics(jsonNode), ##__VA_ARGS__));

#define VALIDATE_JSON_NODE_EXISTS(jsonParentNode, nodeName)                                                                  \
    if (jsonParentNode.find(nodeName) == jsonParentNode.end())                                                               \
    {                                                                                                                        \
        THROW_INVALID_CONFIG(jsonParentNode, "error parsing {} not found", nodeName);                                        \
    }

#define VALIDATE_JSON_NODE_DOESNT_EXIST(jsonParentNode, nodeName)                                                            \
    if (jsonParentNode.find(nodeName) != jsonParentNode.end())                                                               \
    {                                                                                                                        \
        THROW_INVALID_CONFIG(jsonParentNode, "error parsing {} exist but it shouldn't", nodeName);                           \
    }

#define VALIDATE_JSON_NODE_IS_ARRAY(jsonParentNode, nodeName)                                                                \
    VALIDATE_JSON_NODE_EXISTS(jsonParentNode, nodeName)                                                                      \
    if (!jsonParentNode[nodeName].is_array())                                                                                \
    {                                                                                                                        \
        THROW_INVALID_CONFIG(jsonParentNode, "error parsing {} not an array", nodeName);                                     \
    }

#define VALIDATE_JSON_NODE_IS_AN_EMPTY_ARRAY(jsonParentNode, nodeName)                                                       \
    VALIDATE_JSON_NODE_IS_ARRAY(jsonParentNode, nodeName)                                                                    \
    if (jsonParentNode[nodeName].size() != 0)                                                                                \
    {                                                                                                                        \
        THROW_INVALID_CONFIG(jsonParentNode, "error parsing {} not an empty array", nodeName);                               \
    }

#define VALIDATE_JSON_NODE_IS_NOT_EMPTY_ARRAY(jsonParentNode, nodeName)                                                      \
    VALIDATE_JSON_NODE_IS_ARRAY(jsonParentNode, nodeName)                                                                    \
    if (jsonParentNode[nodeName].size() == 0)                                                                                \
    {                                                                                                                        \
        THROW_INVALID_CONFIG(jsonParentNode[nodeName], "error parsing array must be not an empty");                          \
    }

#define VALIDATE_JSON_NODE_IS_OBJECT(jsonParentNode, nodeName)                                                               \
    VALIDATE_JSON_NODE_EXISTS(jsonParentNode, nodeName)                                                                      \
    if (!jsonParentNode[nodeName].is_object())                                                                               \
    {                                                                                                                        \
        THROW_INVALID_CONFIG(jsonParentNode, "error parsing {} not an object", nodeName);                                    \
    }

#define CATCH_JSON()                                                                                                         \
    catch(scaljson::detail::exception const & e)                                                                             \
    {                                                                                                                        \
       throw ExceptionWithCode(SCAL_INVALID_CONFIG, fmt::format(FMT_COMPILE("{}({}): fd={} {}"), __FUNCTION__, __LINE__, m_fd, e.what())); \
    }                                                                                                                        \
    catch(...)                                                                                                               \
    {                                                                                                                        \
        throw;                                                                                                               \
    }
