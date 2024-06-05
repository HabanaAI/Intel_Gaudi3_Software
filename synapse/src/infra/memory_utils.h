#pragma once

#include <string.h>

inline int memcmpSafe (const void *s1, const void *s2, size_t n)
{
    if (n == 0) return 0;
    return memcmp(s1, s2, n);
}
