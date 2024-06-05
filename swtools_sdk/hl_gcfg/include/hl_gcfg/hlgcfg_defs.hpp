#pragma once
#include <utility>
#include <optional>
#include <ostream>
#include <limits>

#define HLGCFG_API __attribute__((visibility("default")))

#define HLGCFG_VER 1
#define HLGCFG_INLINE_VER 1_4

#define HLGCFG_CONCAT_(a, b) a##b
#define HLGCFG_CONCAT(a, b) HLGCFG_CONCAT_(a,b)


#define HLGCFG_NAMESPACE        inline namespace HLGCFG_CONCAT(ver_, HLGCFG_VER)
#define HLGCFG_INLINE_NAMESPACE inline namespace HLGCFG_CONCAT(inline_ver_, HLGCFG_INLINE_VER)

namespace hl_gcfg{
HLGCFG_NAMESPACE{

const uint32_t InvalidDeviceType = std::numeric_limits<uint32_t>::max();
enum class ErrorCode
{
    success,
    cannotOpenFile,
    configRegistryWasDestroyed,
    configNameNotFoundInRegistry,
    configNameAlreadyRegistered,
    aliasNotFound,
    primaryNameNotFoundForAlias,
    primaryKeyNotSet,
    privateConfigAccess,
    observersUpdateFailed,
    valueWasAlreadySetFromEnv,
    invalidString,
    envAliasesConflict, // conf env aliases are used in inconsistent way (several aliases are set at the same time)

};
// boost::outcome for poor
template <class T>
struct Outcome
{
    Outcome(T value)
    : m_value(std::move(value))
    , m_errorCode{ErrorCode::success}
    {}
    Outcome(ErrorCode errorCode, std::string errorDesc)
    : m_errorCode{errorCode}
    , m_errorDesc{std::move(errorDesc)}
    {}

    explicit operator bool() const { return m_value.has_value(); }
    bool has_value() const { return m_value.has_value(); }
    bool has_error() const { return !has_value(); }

    const T &           value()    const { return *m_value; }
    ErrorCode           errorCode() const { return m_errorCode; }
    const std::string & errorDesc() const { return *m_errorDesc; }
private:
    std::optional<T>           m_value;
    ErrorCode                  m_errorCode;
    std::optional<std::string> m_errorDesc;
};

template <>
struct Outcome<void>{
    Outcome()
    : m_errorCode{ErrorCode::success}
    {}
    Outcome(ErrorCode errorCode, std::string errorDesc)
    : m_errorCode{errorCode}
    , m_errorDesc{std::move(errorDesc)}
    {}

    explicit operator bool() const { return !m_errorDesc.has_value(); }
    bool has_error() const { return m_errorDesc.has_value(); }

    ErrorCode           errorCode() const { return m_errorCode; }
    const std::string & errorDesc() const { return *m_errorDesc; }
    friend std::ostream &  operator <<(std::ostream & os, Outcome<void> const & rv)
    {
        os << "ErrCode: " << (unsigned) rv.m_errorCode;
        if (rv.has_error())
        {
            os << " desc: " << rv.errorDesc();
        }
        return os;
    }
private:
    ErrorCode                  m_errorCode;
    std::optional<std::string> m_errorDesc;
};

using VoidOutcome = Outcome<void>;

#define HLGCFG_RETURN_WARN(errorCode, fmtMsg, ...)                                                                     \
    do{                                                                                                                \
        HLGCFG_LOG_WARN(fmtMsg, ##__VA_ARGS__);                                                                        \
        return {hl_gcfg::ErrorCode::errorCode, fmt::format(FMT_COMPILE("{}: " fmtMsg), __func__, ##__VA_ARGS__)};      \
    }while(false)

#define HLGCFG_RETURN_ERR(errorCode, fmtMsg, ...)                                                                      \
    do{                                                                                                                \
        HLGCFG_LOG_ERR(fmtMsg, ##__VA_ARGS__);                                                                         \
        return {hl_gcfg::ErrorCode::errorCode, fmt::format(FMT_COMPILE("{}: " fmtMsg), __func__, ##__VA_ARGS__)};      \
    }while(false)

}}