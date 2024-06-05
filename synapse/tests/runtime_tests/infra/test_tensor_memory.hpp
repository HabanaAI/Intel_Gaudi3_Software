#pragma once

#include <stdint.h>
#include <vector>
#include "test_host_buffer_malloc.hpp"
#include "test_device_buffer_alloc.hpp"

class TensorMemory
{
public:
    TensorMemory();
    TensorMemory(TestHostBufferMalloc&& bufferHost, TestDeviceBufferAlloc&& bufferDevice)
    : m_bufferHost(std::move(bufferHost)), m_bufferDevice(std::move(bufferDevice))
    {
    }
    TensorMemory(const TensorMemory&) = delete;
    TensorMemory(TensorMemory&& other)
    : m_bufferHost(std::move(other.m_bufferHost)), m_bufferDevice(std::move(other.m_bufferDevice))
    {
    }
    ~TensorMemory() = default;

    const TestHostBufferMalloc&  getTestHostBuffer() const { return m_bufferHost; }
    TestHostBufferMalloc&        getTestHostBuffer() { return m_bufferHost; }
    const TestDeviceBufferAlloc& getTestDeviceBuffer() const { return m_bufferDevice; }

private:
    TestHostBufferMalloc  m_bufferHost;
    TestDeviceBufferAlloc m_bufferDevice;
};

typedef std::vector<TensorMemory> TensorMemoryVec;

struct LaunchSectionMemory
{
    TestHostBufferMalloc host;
    TestDeviceBufferAlloc dev;
};

struct LaunchTensorMemory
{
    TensorMemoryVec m_tensorInfoVecInputs;
    TensorMemoryVec m_tensorInfoVecOutputs;

    std::map<uint32_t, LaunchSectionMemory>  m_sectionsData;

    template<class TData>
    const TData* getInputHostBuffer(unsigned inputIdx) const
    {
        return reinterpret_cast<const TData*>(m_tensorInfoVecInputs[inputIdx].getTestHostBuffer().getBuffer());
    }
    template<class TData>
    const TData* getOutputHostBuffer(unsigned outputIdx) const
    {
        return reinterpret_cast<const TData*>(m_tensorInfoVecOutputs[outputIdx].getTestHostBuffer().getBuffer());
    }
};
