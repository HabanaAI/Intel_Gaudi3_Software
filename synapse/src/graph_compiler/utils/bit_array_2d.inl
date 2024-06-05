#pragma once

#include <vector>
#include <algorithm>
#include <cstring>

BitArray2D::BitArray2D(uint32_t numRows) : m_n(numRows), m_storage(getRowSize(numRows) * numRows, 0) {}

bool BitArray2D::getBit(uint32_t row, uint32_t col) const
{
    uint32_t colIndex     = col / BITS_IN_UINT64;
    uint32_t elementIndex = row * getRowSize(m_n) + colIndex;
    uint32_t bitOffset    = col % BITS_IN_UINT64;
    return (m_storage[elementIndex] >> bitOffset) & 1;
}

void BitArray2D::setBit(uint32_t row, uint32_t col, bool value)
{
    uint32_t colIndex     = col / BITS_IN_UINT64;
    uint32_t elementIndex = row * getRowSize(m_n) + colIndex;
    uint32_t bitOffset    = col % BITS_IN_UINT64;
    uint64_t mask         = 1ULL << bitOffset;
    if (value)
    {
        m_storage[elementIndex] |= mask;
    }
    else
    {
        m_storage[elementIndex] &= ~mask;
    }
}

void BitArray2D::copyRow(uint32_t srcRow, uint32_t destRow)
{
    size_t elementsPerRow = getRowSize(m_n);
    std::memcpy(&m_storage[destRow * elementsPerRow],
                &m_storage[srcRow * elementsPerRow],
                elementsPerRow * sizeof(uint64_t));
}

void BitArray2D::bitwiseOr(uint32_t row1, uint32_t row2, uint32_t destRow)
{
    size_t elementsPerRow = getRowSize(m_n);
    for (uint32_t i = 0; i < elementsPerRow; ++i)
    {
        m_storage[destRow * elementsPerRow + i] =
            m_storage[row1 * elementsPerRow + i] | m_storage[row2 * elementsPerRow + i];
    }
}

void BitArray2D::bitwiseAnd(uint32_t row1, uint32_t row2, uint32_t destRow)
{
    size_t elementsPerRow = getRowSize(m_n);
    for (uint32_t i = 0; i < elementsPerRow; ++i)
    {
        m_storage[destRow * elementsPerRow + i] =
            m_storage[row1 * elementsPerRow + i] & m_storage[row2 * elementsPerRow + i];
    }
}

void BitArray2D::setDiagonal()
{
    size_t elementsPerRow = getRowSize(m_n);
    for (uint32_t i = 0; i < m_n; ++i)
    {
        uint32_t colIndex     = i / BITS_IN_UINT64;
        uint32_t elementIndex = i * elementsPerRow + colIndex;
        uint32_t bitOffset    = i % BITS_IN_UINT64;
        m_storage[elementIndex] |= (1ULL << bitOffset);
    }
}
