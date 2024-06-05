#pragma once

#include <stdint.h>

namespace common
{
class RecipeReaderHelper
{
public:
    RecipeReaderHelper()          = default;
    virtual ~RecipeReaderHelper() = default;

    virtual uint32_t getDynamicEcbListBufferSize() const = 0;
    virtual uint32_t getStaticEcbListBufferSize() const  = 0;
};
}  // namespace common
