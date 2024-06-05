//A macro for not using parameters but not getting warned about it.
#ifndef MME__UTILS_H
#define MME__UTILS_H

#include <cstdint>

#define UNUSED(x) ((void)x)

typedef union
{
    void* p;
    uint64_t u64;
    uint32_t u32[2];
} ptrToInt;

#endif  // MME__UTILS_H
