#pragma once

#include <cstdint>
#include <vector>
#include "include/sync/data_range.h"

template<typename T>
using BasicSegment = DataRange<T>;

using Segment = BasicSegment<uint32_t>;

using Segments = std::vector<Segment>;


template<size_t Alignment>
Segments alignSegments(const Segments& segments)
{
    Segments alignedSegments;
    alignedSegments.reserve(segments.size());
    for (auto& it : segments)
    {
        alignedSegments.emplace_back(it.alignBeginning<Alignment>());
    }

    return alignedSegments;
}

template<size_t Alignment>
void alignSegmentsInplace(Segments& segments)
{
    for (auto& it : segments)
    {
        it.start(it.start() - it.start() % Alignment);
    }
}

template<typename Pred>
Segments joinAdjecentSegments(const Segments& segments, Pred canJoin)
{
    Segments joinedAdjectent;
    if (segments.empty()) return joinedAdjectent;
    joinedAdjectent.reserve(segments.size());
    auto start = segments.begin(), prev = segments.begin();
    for (auto i = std::next(segments.begin()); i != segments.end(); prev = i++)
    {
        if (i->start() != prev->end() || !canJoin(*prev, *i))
        {
            joinedAdjectent.emplace_back(start->start(), prev->end());
            start = i;
        }
    }

    joinedAdjectent.emplace_back(start->start(), prev->end());
    return joinedAdjectent;
}


template<typename Predicate>
Segments trueSegments(uint32_t size, Predicate pred, uint32_t start = 0)
{
    Segments segs;
    unsigned segStart = -1;
    for (unsigned i = start; i < start + size; ++i)
    {
        if (pred(i))
        {
            // check if we are already within true segment
            if (segStart == -1)
            {
                segStart = i;
            }
        }
        else if (segStart != -1)
        {
            // pred false, terminating segment.
            segs.emplace_back(segStart, i);
            segStart = -1;
        }
    }
    if (segStart != -1)
    {
        // Terminating the end segment.
        segs.emplace_back(segStart, start + size);
    }
    return segs;
}



template<typename KeyFunc, typename KeyType = decltype(std::declval<KeyFunc>()(0))>
Segments groupby(uint32_t size, KeyFunc key, uint32_t start = 0, KeyType ignoreKey = 0)
{
    Segments segs;
    if (size == 0) return segs; // ensuring at least 1 element
    unsigned segStart = 0;
    auto targetKey = key(start);
    for (unsigned i = start + 1; i < start + size; ++i)
    {
        auto currKey = key(i);
        if (currKey == targetKey)
        {
            continue;
        }
        else
        {
            if (targetKey != ignoreKey) segs.emplace_back(segStart, i);
            segStart = i;
            targetKey = currKey;
        }
    }
    // Terminating the end segment.
    if (targetKey != ignoreKey) segs.emplace_back(segStart, start + size);
    return segs;
}
