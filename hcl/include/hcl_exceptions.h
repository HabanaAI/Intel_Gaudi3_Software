#pragma once

#include <cstdio>
#include <exception>
#include <string>
#include <sstream>

namespace hcl
{
class HclException : public std::exception
{
    // TODO: call-stack may be printed as part of the object constructor
public:
    template<typename T, typename... Args>
    HclException(T&& a_value, Args&&... a_args)
    {
        std::ostringstream s;
        s << std::forward<T>(a_value);
        addToStream(s, std::forward<Args>(a_args)...);
        m_exceptionString = s.str();
    }
    const char* what() const noexcept { return m_exceptionString.c_str(); }

private:
    template<typename T, typename... Args>
    void addToStream(std::ostringstream& a_stream, T&& a_value, Args&&... a_args)
    {
        a_stream << std::forward<T>(a_value);
        addToStream(a_stream, std::forward<Args>(a_args)...);
    }

    void addToStream(std::ostringstream&) {}

    std::string m_exceptionString;
};

class NotImplementedException : public HclException
{
public:
    NotImplementedException(const std::string& msg = "") : HclException(msg) {}
};

class VerifyException : public HclException
{
public:
    VerifyException(const std::string& msg = "") : HclException(msg) {}
};

}  // namespace hcl