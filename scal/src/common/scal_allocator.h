#pragma once

#include <map>
#include <string>
#include <mutex>
#include <type_traits>

#include "scal_base.h"
#include "logger.h"

class ScalHeapAllocator : public Scal::Allocator
{
public:
    explicit ScalHeapAllocator(const std::string& name);
    virtual ~ScalHeapAllocator() = default;

    virtual void setSize(uint64_t memorySize) override;
    virtual uint64_t alloc(uint64_t size, uint64_t alignment = c_cl_size) override;
    virtual void free(uint64_t ptr) override;
    virtual void getInfo(uint64_t& totalSize, uint64_t& freeSize) override;
protected:
    std::string m_name;
    std::map<uint64_t, uint64_t> m_slabs; // slabs per size
    std::mutex m_mutex;
    uint64_t   m_totalSize = 0;
    uint64_t   m_freeSize  = 0;
};
