#include "range.h"
#include "infra/defs.h"

bool rangeIntersectsWith(const Range &range, const Range &otherRange, Range &intersection)
{
    deviceAddrOffset end1 = range.base + range.size;
    deviceAddrOffset end2 = otherRange.base + otherRange.size;

    deviceAddrOffset maxBase = std::max<deviceAddrOffset>(range.base, otherRange.base);
    deviceAddrOffset minEnd  = std::min<deviceAddrOffset>(end1, end2);

    if (maxBase > minEnd) return false;  // no intersection

    intersection.base = maxBase;
    intersection.size = minEnd - maxBase;
    return true;
}

bool rangeContains(const Range &range, const Range &otherRange)
{
    if (otherRange.base < range.base) return false;
    deviceAddrOffset end1 = range.base + range.size;
    deviceAddrOffset end2 = otherRange.base + otherRange.size;
    return (end2 <= end1);
}


bool rangeContainedIn(const Range &range, const Range &otherRange)
{
    return rangeContains(otherRange, range);
}

void allocateRange(const Range &reqRange, const std::list<Range>::iterator &rangesIter,
                   std::list<Range> &freeRanges)
{
    Range& fromRange = *rangesIter;
    HB_ASSERT(fromRange.size >= reqRange.size, "requsted range is larger than free range");
    HB_ASSERT(reqRange.base >= fromRange.base, "requsted base address isn't part of passed range");

    // Update SramImage (Allocate)
    if (fromRange.size == reqRange.size)
    {
        freeRanges.erase(rangesIter);
    }
    else
    {
        if (fromRange.base == reqRange.base)
        {
            fromRange.base += reqRange.size;
            fromRange.size -= reqRange.size;
        }
        else if ((fromRange.base + fromRange.size) == (reqRange.base + reqRange.size))
        {
            fromRange.size -= reqRange.size;
        }
        else
        {
            Range newRange;
            newRange.base = reqRange.base + reqRange.size;
            newRange.size = fromRange.base + fromRange.size - newRange.base;
            fromRange.size = reqRange.base - fromRange.base;
            freeRanges.insert(std::next(rangesIter), newRange);
        }
    }
}
