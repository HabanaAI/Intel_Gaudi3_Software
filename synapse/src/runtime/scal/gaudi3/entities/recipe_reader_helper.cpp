#include "recipe_reader_helper.hpp"

// XXX_COMPUTE_ECB_LIST_BUFF_SIZE
#include "gaudi3_arc_eng_packets.h"

#include <stdint.h>

using namespace gaudi3;

uint32_t RecipeReaderHelper::getDynamicEcbListBufferSize() const
{
    return DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;
}

uint32_t RecipeReaderHelper::getStaticEcbListBufferSize() const
{
    return STATIC_COMPUTE_ECB_LIST_BUFF_SIZE;
}