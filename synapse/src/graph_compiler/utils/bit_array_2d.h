#pragma once

#include "utils.h"
#include <cstdint>
#include <vector>

/*
    2d NxN bit array that supports efficient row-wise operations.
*/

class BitArray2D
{
public:
    explicit BitArray2D(uint32_t numRows);

    bool getBit(uint32_t row, uint32_t col) const;
    void setBit(uint32_t row, uint32_t col, bool value);
    void copyRow(uint32_t srcRow, uint32_t destRow);
    void bitwiseOr(uint32_t row1, uint32_t row2, uint32_t destRow);
    void bitwiseAnd(uint32_t row1, uint32_t row2, uint32_t destRow);
    void setDiagonal();

private:
    uint32_t              m_n;
    std::vector<uint64_t> m_storage;

    static size_t getRowSize(uint32_t n) { return (n + BITS_IN_UINT64 - 1) / BITS_IN_UINT64; }
};

#include "bit_array_2d.inl"