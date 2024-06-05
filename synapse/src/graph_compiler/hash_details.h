#pragma once

#include <functional>
#include <tuple>
#include <cstdint>

namespace gc::hash_details
{
template<typename T>
static uint64_t int_hash(const std::enable_if_t<std::is_integral_v<T>, T>& key)
{
    // https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    uint64_t hash = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9LLU;
    hash          = (hash ^ (hash >> 27)) * 0x94d049bb133111ebLLU;
    hash          = hash ^ (hash >> 31);
    return hash;
}

// Prevent using std::hash implementation for integer types (identity)
const auto generic_hash = [](const auto& v) {
    if constexpr (std::is_integral_v<std::decay_t<decltype(v)>>)
    {
        return int_hash<std::decay_t<decltype(v)>>(v);
    }
    else
    {
        return std::hash<std::decay_t<decltype(v)>> {}(v);
    }
};

const auto hasher = [](std::size_t hash, auto&&... values) {
    const auto hashCombiner = [&hash](auto&& val) {
        // https://stackoverflow.com/questions/4948780/magic-number-in-boosthash-combine
        hash ^= generic_hash(val) + 0x9e3779b97f4a7c15LLU + (hash << 12) + (hash >> 4);
    };

    (hashCombiner(std::forward<decltype(values)>(values)), ...);
    return hash;
};

struct pair_hash
{
    template<typename T0, typename T1>
    std::size_t operator()(const std::pair<T0, T1>& p) const
    {
        std::size_t hash = 0;
        std::apply([&hash](auto&&... values) { hash = gc::hash_details::hasher(hash, values...); }, p);
        return hash;
    }
};

struct tuple_hash
{
    template<typename... TT>
    std::size_t operator()(const std::tuple<TT...>& t) const
    {
        std::size_t hash = 0;
        std::apply([&hash](auto&&... values) { hash = gc::hash_details::hasher(hash, values...); }, t);
        return hash;
    }
};

}  // namespace gc::hash_details
