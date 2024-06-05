#pragma once

#include <limits>
#include <stdint.h>

static const uint32_t MAX_COMP_SYNC_GROUP_COUNT = std::numeric_limits<uint32_t>::max();

#define MAX_NUM_OF_RECIPE_BASE_ADDRESSES         (4)
#define RECIPE_BASE_ADDRESSES_INDICES_MASK       (0xF)
#define RECIPE_BASE_ADDRESSES_INDICES_FIELD_SIZE (4)
