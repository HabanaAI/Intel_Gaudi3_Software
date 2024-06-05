#pragma once

namespace gaudi
{
static const uint64_t SPECIAL_FUNCS_INTERVAL256_BASE_ADDR = 0;
static const uint64_t SPECIAL_FUNCS_INTERVAL128_BASE_ADDR = 425984;
static const uint64_t SPECIAL_FUNCS_INTERVAL64_BASE_ADDR  = 638976;
static const uint64_t SPECIAL_FUNCS_INTERVAL32_BASE_ADDR  = 753664;
static const uint64_t SPECIAL_FUNCS_TABLE_SIZE            = 819200;
}

extern uint8_t coefficientsTableH3[gaudi::SPECIAL_FUNCS_TABLE_SIZE];
