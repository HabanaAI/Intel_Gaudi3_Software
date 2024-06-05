#pragma once

#include "runtime/common/recipe/recipe_reader_helper.hpp"

#include <stdint.h>

namespace gaudi3
{
class RecipeReaderHelper : public common::RecipeReaderHelper
{
public:
    static const common::RecipeReaderHelper* getInstance()
    {
        static const gaudi3::RecipeReaderHelper recipeReaderInstance;

        return &recipeReaderInstance;
    }

    virtual ~RecipeReaderHelper() = default;

    virtual uint32_t getDynamicEcbListBufferSize() const override;
    virtual uint32_t getStaticEcbListBufferSize() const override;

private:
    RecipeReaderHelper() = default;
};
}  // namespace gaudi3
