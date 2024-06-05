#pragma once

#include "defs.h"
#include <cstdint>
#include "limits_4bit.h"


inline void addSign4BitTo8Bit(int8_t& value)
{
    // check bit 3 sign, if negative append leading 1s in bits 4-7, since we lost them when the number was condensed.
    if (value & 0x8U) value |= 0xf0U;
}


inline void clip4BitTo8Bit(int8_t& value)
{
    if (value > INT4_MAX_VAL) value = INT4_MAX_VAL;
    else if (value < INT4_MIN_VAL) value = INT4_MIN_VAL;
}


inline void clip4BitTo8Bit(uint8_t& value)
{
    if (value > UINT4_MAX_VAL) value = UINT4_MAX_VAL;
}


inline std::pair<int8_t, int8_t> expand4Bit(int8_t inCondensed)
{
    int8_t outLow = inCondensed & 0xfU; // take bits 0-3
    addSign4BitTo8Bit(outLow);

    // Cast the condensed element to uint in order to safely shift. Shift of a negative integer is Undefined Behaviour.
    uint8_t inCondensedBits = *reinterpret_cast<uint8_t*>(&inCondensed);
    uint8_t outHighBits     = (inCondensedBits & 0xf0U) >> 4U; // take bits 4-7 and shift right
    int8_t outHigh          = *reinterpret_cast<int8_t*>(&outHighBits);
    addSign4BitTo8Bit(outHigh);

    auto out = std::pair<int8_t , int8_t>{outLow, outHigh};
    return out;
}


inline std::pair<uint8_t, uint8_t> expand4Bit(uint8_t inCondensed)
{
    uint8_t outLow  = inCondensed & 0xf; // take bits 0-3
    uint8_t outHigh = (inCondensed & 0xf0) >> 4U; // take bits 4-7 and shift right to 0-3 bits
    auto out = std::pair<uint8_t , uint8_t>{outLow, outHigh};
    return out;
}


inline int8_t condense4Bit(int8_t inLow, int8_t inHigh)
{
    clip4BitTo8Bit(inLow);
    clip4BitTo8Bit(inHigh);

    // Cast the high element to uint in order to safely shift left. Shift a signed integer is Undefined Behaviour.
    uint8_t inHighBits  = *reinterpret_cast<uint8_t*>(&inHigh);
    uint8_t outHighBits = (inHighBits & 0xfU) << 4U; // take bits 0-3 and shift left to 4-7 bits
    int8_t outHigh      = *reinterpret_cast<int8_t*>(&outHighBits);

    int8_t out = (inLow & 0xfU) | outHigh; // take bits 0-3 and condense the two elements in one byte

    return out;
}


inline uint8_t condense4Bit(uint8_t inLow, uint8_t inHigh)
{
    clip4BitTo8Bit(inLow);
    clip4BitTo8Bit(inHigh);

    uint8_t outHigh = (inHigh & 0xfU) << 4U; // take bits 0-3 and shift left to 4-7 bits
    uint8_t out     = (inLow & 0xfU) | outHigh; // take bits 0-3 and condense the two elements in one byte

    return out;
}


template <typename T>
void condense8BitBufferTo4BitBuffer(T* int8Buffer, unsigned elementsNum, T* int4Buffer)
{
    HB_ASSERT(elementsNum % 2 == 0, "Buffer elements number must be even when condensing to 4 bits");
    for (unsigned i = 0; i < elementsNum / 2; i++)
    {
        int4Buffer[i] = condense4Bit(int8Buffer[2*i], int8Buffer[2*i + 1]);
    }
}

template <typename T>
void expand4BitBufferTo8BitBuffer(T* int4Buffer, unsigned elementsNum, T* int8Buffer)
{
    HB_ASSERT(elementsNum % 2 == 0, "Buffer elements number must be even when expanding from 4 bits");
    for (unsigned i = 0; i < elementsNum / 2; i++)
    {
        std::pair<T, T> expandedElements = expand4Bit(int4Buffer[i]);
        int8Buffer[i * 2]     = expandedElements.first;
        int8Buffer[i * 2 + 1] = expandedElements.second;
    }
}
