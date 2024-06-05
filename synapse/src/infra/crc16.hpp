#pragma once
#include <cstdint>

uint16_t crc_16(uint16_t crcStart, const void* ptr, uint32_t num_bytes);

template<class T>
inline uint16_t crc16(uint16_t crc, T const& value)
{
    return crc_16(crc, &value, sizeof(value));
}