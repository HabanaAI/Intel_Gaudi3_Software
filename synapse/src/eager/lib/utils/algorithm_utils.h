#pragma once

// synapse api (relative to include/)
#include "graph_compiler/types.h"

// std includes
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace eager_mode
{
template<typename T, typename BinaryOp>
constexpr bool vector_all_of_pairwise(const std::vector<T>& v, BinaryOp op)
{
    if (v.size() >= 2)
    {
        const size_t maxIdx = v.size() - 1;
        for (size_t i = 0; i < maxIdx; ++i)
        {
            if (op(v[i], v[i + 1]) == false) return false;
        }
    }
    return true;
}

// Append value if not present in the container and return if appended
template<typename C, typename V>
constexpr bool appendIfUniq(C& container, V&& value)
{
    if (std::find(std::begin(container), std::end(container), value) == std::end(container))
    {
        container.push_back(std::forward<V>(value));
        return true;
    }
    return false;
}

// This function is restricted to finding the maximum batch value in a shape dimensions range.
// 'from' is the first inclusive index of the range, 'to' is the last exclusive one.
// It's legal to pass 'from' > 'to', in this case the result is 1 (default batch value).
constexpr unsigned getMaxBatchDim(const SizeArray& dims, unsigned from, unsigned to)
{
    if (from > to) return 1;
    if (from == to) return dims[from];
    unsigned maxDim = dims[from];
    for (unsigned i = from + 1; i <= to; ++i)
    {
        if (maxDim < dims[i])
        {
            maxDim = dims[i];
        }
    }
    return maxDim;
}

// Merge values before specified indices (m) with values from a container (c).
//
// ex.
//  std::vector<int> c = {10, 20, 30, 40, 50};  // original container
//
//  // before the original index 0 of c, insert -1
//  // before the original index 1 of c, insert -2 and -3 (preserving order)
//  // before the original index 5 of c, insert -4 (this is one past the last item in c)
//  std::vector<std::pair<size_t, int>> m = {{0, -1}, {1, -2}, {1, -3}, {5, -4}};  // pairs of indices and values
//
//  std::vector<int> ref = {-1, 10, -2, -3, 20, 30, 40, 50, -4};
//  assert(moveIntoContainerAtIndices(c, std::move(m)) == ref);
template<class VecC, class VecM>
inline void moveIntoContainerAtIndices(/*IN,OUT*/ VecC& c, VecM& m)
{
    if (m.empty()) return;

    assert(std::is_sorted(m.begin(),
                          m.end(),
                          // The default std::less<> would also compare T which we do not desire
                          [](const auto& a, const auto& b) { return a.first < b.first; }) &&
           m.back().first <= c.size());

    const std::size_t cNum = c.size();
    const std::size_t mNum = m.size();
    const std::size_t oNum = cNum + mNum;

    c.resize(oNum);

    // Iterate in reverse order to avoid overwrites
    std::size_t left   = mNum;  // Num of elements left to insert
    std::size_t outIdx = oNum - 1;
    while (left > 0)  // When there's nothing left to add, the rest are already in place
    {
        c[outIdx] = std::move(m[left - 1].first == outIdx - (left - 1) ? m[--left].second : c[outIdx - left]);
        --outIdx;
    }

    m.clear();
}

}  // namespace eager_mode
