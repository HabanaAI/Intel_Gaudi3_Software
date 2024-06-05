#include "crc16.hpp"

#include <array>

static std::array<uint16_t, 256> init_crc16_tab(void)
{
    const uint16_t            CRC_POLY_16 = 0xA001;
    std::array<uint16_t, 256> crc_tab16;
    for (uint16_t i = 0; i < 256; i++)
    {
        uint16_t crc = 0;
        uint16_t c   = i;

        for (uint16_t j = 0; j < 8; j++)
        {
            if ((crc ^ c) & 0x0001) crc = (crc >> 1) ^ CRC_POLY_16;
            else
                crc = crc >> 1;

            c = c >> 1;
        }

        crc_tab16[i] = crc;
    }

    // [CID: 40788] Intentional - crc_tab16 is initialized right after declaration
    return crc_tab16;
} /* init_crc16_tab */

static const std::array<uint16_t, 256> crc_tab16 = init_crc16_tab();

uint16_t crc_16(uint16_t crcStart, const void* void_ptr, uint32_t num_bytes)
{
    uint16_t       crc = crcStart;
    const uint8_t* ptr = (const uint8_t*)void_ptr;
    if (ptr != nullptr)
    {
        for (uint32_t a = 0; a < num_bytes; a++)
        {
            uint16_t short_c = *ptr;
            uint16_t tmp     = crc ^ short_c;
            crc              = (crc >> 8) ^ crc_tab16[tmp & 0xff];

            ptr++;
        }
    }
    return crc;
}