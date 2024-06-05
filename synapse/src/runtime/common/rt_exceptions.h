#pragma once

class RTException : public std::exception
{
    std::string _msg;

public:
    RTException(const std::string& msg) : _msg(msg) {}

    virtual const char* what() const noexcept override { return _msg.c_str(); }
};

class RTNotImplementedException : public std::exception
{
};