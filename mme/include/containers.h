#ifndef MME__CONTAINERS_H
#define MME__CONTAINERS_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

template<typename T, std::size_t N>
class SmallCircularFIFOCache
{
public:
    // Reset to initial empty state
    void clear()
    {
        m_head = 0;
        m_maxVal = 0;
    }

    // Insert value if not presnet; Return if was already present.
    // When the cache is full (N items) the oldest inserted item is replaced.
    bool insert(const T& v)
    {
        if (contains(v))
        {
            return true;
        }
        m_data[m_head++] = v;
        m_maxVal = std::max(m_maxVal, m_head);
        m_head = m_head == N ? 0 : m_head;
        return false;
    }

private:
    bool contains(const T& v) const
    {
        return std::find(std::begin(m_data), std::begin(m_data) + m_maxVal, v) != std::begin(m_data) + m_maxVal;
    }

    using IndexType = uint_fast8_t;
    static_assert(N - 1 <= std::numeric_limits<IndexType>::max(), "");

    std::array<T, N> m_data;  // Initially unintialized
    IndexType m_head = 0;  // Next location to insert data; wraps around

    // m_data has valid values between indexes [0, m_maxVal).
    // maxVal starts at 0 for an empty cache and increases until the cache is
    // filled and the whole [0, m_maxVal=N) range is valid.
    // Note: If the cache has low hit ratio and there's a valid guaranteed
    //       unused value like -1, it makes sense to std::fill it in the c'tor
    //       and throw away m_maxVal (making insert cheaper), always searching
    //       the whole container.
    IndexType m_maxVal = 0;  // m_data start
};

#endif //MME__CONTAINERS_H
