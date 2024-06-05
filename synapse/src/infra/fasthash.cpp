#include "fasthash.h"


uint64_t fasthash(const std::string& s)
{
    return std::hash<std::string>{}(s);
}

uint64_t fasthash(const void* ptr, size_t len)
{
    return std::hash<std::string>{}(std::string{reinterpret_cast<const char*>(ptr), reinterpret_cast<const char*>(ptr) + len});
}
