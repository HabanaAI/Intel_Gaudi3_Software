#pragma once

#include <list>
#include "types.h"

struct Range
{
    uint64_t base;
    uint64_t size;
};

void allocateRange(const Range &reqRange, const std::list<Range>::iterator &rangesIter,
                   std::list<Range> &freeRanges);
bool rangeIntersectsWith(const Range &range, const Range &otherRange, Range &intersection);
bool rangeContainedIn(const Range &range, const Range &otherRange);
bool rangeContains(const Range &range, const Range &otherRange);
