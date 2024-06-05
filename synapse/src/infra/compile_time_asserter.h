#pragma once

#define NUM_OF_ELEMENTS(array) sizeof(array) / sizeof(array[0])

#define COMPILE_TIME_ASSERT_VERIFY_ARRAY_SIZE(array, numOfElements)                                                    \
    static_assert(NUM_OF_ELEMENTS(array) == numOfElements, "array size mismatch")
