#pragma once

#include "synapse_common_types.h"

#include <stdexcept>
#include <string>

namespace syn
{
class Exception : public std::runtime_error
{
public:
    Exception(synStatus status, const std::string& message) : std::runtime_error(message), m_status(status) {}
    virtual ~Exception() = default;

    synStatus status() const { return m_status; }

private:
    synStatus m_status;
};
}  // namespace syn