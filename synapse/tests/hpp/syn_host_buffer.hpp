#pragma once

#include "syn_object.hpp"

namespace syn
{
class HostBuffer : public SynObject<void>
{
public:
    HostBuffer() : m_size(0) {};

    void* get() const { return handlePtr(); }

    template<class T>
    T* getAs() const
    {
        return static_cast<T*>(handlePtr());
    }

    uint64_t getAddress() const { return reinterpret_cast<uint64_t>(handlePtr()); }
    uint64_t getSize() const { return m_size; }

private:
    HostBuffer(const std::shared_ptr<void>& handle, uint64_t size) : SynObject(handle), m_size(size) {}

    uint64_t m_size;

    friend class Device;  // Device class requires access to HostBuffer private constructor
    friend class Tensor;  // Tensor class requires access to HostBuffer private constructor
};
}  // namespace syn