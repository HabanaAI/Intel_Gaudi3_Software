#pragma once
#include <stdint.h>

#include "fs_assert.h"

namespace Gaudi2
{
namespace Mme
{
template <typename TYPE, int SIZE>
class DataQ
{
   public:
    DataQ() : m_wrPtr(0), m_rdPtr(0) { std::memset(m_data, 0xdd, sizeof(m_data)); }

    virtual ~DataQ() {}

    bool isFull() { return (m_wrPtr - m_rdPtr == SIZE); }

    bool isEmpty() { return m_wrPtr == m_rdPtr; }

    unsigned getLevel() { return (m_wrPtr - m_rdPtr); }

    void push(const TYPE& dataIn)
    {
        FS_ASSERT_MSG(!isFull(), "Push to FIFO Full");
        m_data[m_wrPtr % SIZE] = dataIn;
        m_wrPtr++;
    }

    TYPE pop()
    {
        FS_ASSERT_MSG(!isEmpty(), "Pop from empty FIFO");
        uint64_t ptr = m_rdPtr % SIZE;
        m_rdPtr++;
        return m_data[ptr];
    }

    TYPE get(const uint64_t ptr_offset) { return m_data[(m_rdPtr + ptr_offset) % SIZE]; }

    void set(const uint8_t ptr_offset, const TYPE wr_data) { m_data[(m_rdPtr + ptr_offset) % SIZE] = wr_data; }

    void flush()
    {
        m_wrPtr = 0;
        m_rdPtr = 0;
    }

   private:
    TYPE     m_data[SIZE];
    uint64_t m_wrPtr;
    uint64_t m_rdPtr;
};

} // namespace Mme
} // namespace Gaudi2
