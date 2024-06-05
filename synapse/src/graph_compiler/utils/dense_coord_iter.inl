#pragma once

#include <cassert>
#include "defs.h"

template<typename T, unsigned int dim, uint32_t frozen_dims_bitmask>
template<typename ArrayLike>
std::array<T, dim> DenseCoordIter<T, dim, frozen_dims_bitmask>::
    mkBounds(const ArrayLike arr)
{
    std::array<T, dim> result;
    for (int i = 0; i < dim; ++i)
    {
        HB_ASSERT(arr[i] > 0, "Invalid dimension bounds");
        result[i] = arr[i];
    }
    return result;
}

template<typename T, unsigned int dim, uint32_t frozen_dims_bitmask>
template<typename ArrayLike>
DenseCoordIter<T, dim, frozen_dims_bitmask>::DenseCoordIter(const ArrayLike bounds)
: m_bounds(mkBounds(bounds)),
  m_pos{},
  m_end(false)
{
    static_assert(dim < 32, "Unsupported dim");
    static_assert((1 << dim) > frozen_dims_bitmask,
            "Freeze dimensions outside of range.");
}

template<typename T, unsigned int dim, uint32_t frozen_dims_bitmask>
void DenseCoordIter<T, dim, frozen_dims_bitmask>::next()
{
    // Add one with carry on unfrozen dimensions
    int carry = 1;
    for (unsigned i = 0; i < dim; i++)
    {
        if (1 & (frozen_dims_bitmask >> i)) continue;
        m_pos[i] += carry;
        if (m_pos[i] == m_bounds[i])
        {
            m_pos[i] = 0;
        }
        else
        {
            return;
        }
    }
    m_end = true;
}

template<typename T, unsigned int dim, uint32_t frozen_dims_bitmask>
void DenseCoordIter<T, dim, frozen_dims_bitmask>::advance(uint64_t n)
{
    // Add 'n' with carry on unfrozen dimensions
    for (unsigned i = 0; n && i < dim; i++)
    {
        if (1 & (frozen_dims_bitmask >> i)) continue;
        n += m_pos[i];
        uint64_t tmp = n / m_bounds[i];
        m_pos[i] = n - tmp * m_bounds[i];
        n = tmp;
    }
    m_end = m_end || n != 0;
}
