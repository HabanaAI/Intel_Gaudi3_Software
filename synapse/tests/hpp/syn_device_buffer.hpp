#pragma once

#include "syn_object.hpp"

namespace syn
{
class DeviceBuffer : public SynObject<uint64_t>
{
public:
    DeviceBuffer() : m_size(0) {}

    uint64_t getAddress() const { return handle(); }
    uint64_t getSize() const { return m_size; }

private:
    DeviceBuffer(const std::shared_ptr<uint64_t>& handle, uint64_t size) : SynObject(handle), m_size(size) {}

    uint64_t m_size;

    friend class Device;  // Device class requires access to DeviceBuffer private constructor
};
}  // namespace syn