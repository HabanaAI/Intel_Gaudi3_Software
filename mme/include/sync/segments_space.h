#ifndef MME__SEGMENTS_SPACE_H
#define MME__SEGMENTS_SPACE_H

#include <map>

template <class SEG>
struct SegSplit
{
    static void split(SEG & left, SEG & right, const uint64_t offset) {}
};

template <class SEG>
struct SegMerge
{
    static bool merge(SEG & left, const SEG & right, const uint64_t sizeRight) {return left == right;}
};


template <class SEG, class SEG_SPLIT = SegSplit<SEG>, class SEG_MERGE = SegMerge<SEG>>
class SegmentsSpace
{
public:
    struct SegRecord
    {
        uint64_t size;
        bool valid;
        SEG seg;
    };

    typedef typename std::map<uint64_t, SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::SegRecord>::const_iterator const_iterator;

    SegmentsSpace();
    bool getSegments(const uint64_t base, const uint64_t size, std::vector<SEG*> &segs);
    bool getSegments(const uint64_t base, const uint64_t size, std::vector<std::tuple<uint64_t, uint64_t, SEG*>>& segs);
    void addSegment(const uint64_t base, const uint64_t size, const SEG &seg, bool overwrite=true);

    const_iterator cbegin() const { return m_segs.cbegin();}
    const_iterator cend() const { return m_segs.cend();}
    void mergeSegments();
    bool isSegmentCovered(const uint64_t base, const uint64_t size) const;
    bool isSegmentInvalid(const uint64_t base, const uint64_t size) const;
    void erase(const uint64_t base, const uint64_t size);

    void getCoveredSegments(
            const uint64_t base,
            const uint64_t size,
            const_iterator &cbegin,
            const_iterator &cend);

private:

    typedef typename std::map<uint64_t, SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::SegRecord>::iterator iterator;

    void findCoveredSegments(
        const uint64_t base,
        const uint64_t size,
        iterator &begin,
        iterator &end);

    std::map<uint64_t, SegRecord> m_segs;
};


template <class SEG, class SEG_SPLIT, class SEG_MERGE>
SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::SegmentsSpace()
{
    SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::SegRecord defaultRecord = { 0 };
    defaultRecord.size =  UINT64_MAX;
    defaultRecord.valid = false;
    m_segs[0] = defaultRecord;
}

template <class SEG, class SEG_SPLIT, class SEG_MERGE>
void SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::findCoveredSegments(
    const uint64_t base,
    const uint64_t size,
    iterator &begin,
    iterator &end)
{
    auto startIt = m_segs.lower_bound(base);

    if ((startIt == m_segs.end()) || (startIt->first != base))
    {
        if (startIt != m_segs.begin())
        {
            startIt--;
        }

        uint64_t leftSegSize = base - startIt->first;
        uint64_t rightSegSize = startIt->second.size - leftSegSize;

        auto & leftSegRecord = startIt->second;
        startIt = m_segs.emplace_hint(startIt, base, leftSegRecord);
        auto & rightSegRecord = startIt->second;

        leftSegRecord.size = leftSegSize;
        rightSegRecord.size = rightSegSize;

        if (startIt->second.valid)
        {
            SEG_SPLIT::split(leftSegRecord.seg, rightSegRecord.seg, leftSegSize);
        }
    }

    uint64_t segEnd = base + size;
    auto endIt = m_segs.lower_bound(segEnd);

    if (((endIt == m_segs.end()) || (endIt->first != segEnd)) &&
        (segEnd != m_segs.rbegin()->first + m_segs.rbegin()->second.size))
    {
        if (endIt != m_segs.begin())
        {
            endIt--;
        }

        uint64_t leftSegSize = segEnd - endIt->first;
        uint64_t rightSegSize = endIt->second.size - leftSegSize;

        auto & leftSegRecord = endIt->second;
        endIt = m_segs.emplace_hint(endIt, segEnd, leftSegRecord);
        auto & rightSegRecord = endIt->second;

        leftSegRecord.size = leftSegSize;
        rightSegRecord.size = rightSegSize;

        if (endIt->second.valid)
        {
            SEG_SPLIT::split(leftSegRecord.seg, rightSegRecord.seg, leftSegSize);
        }
    }

    begin = startIt;
    end = endIt;
}

template<typename SEG, class SEG_SPLIT, class SEG_MERGE>
void SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::addSegment(
    const uint64_t base,
    const uint64_t size,
    const SEG &seg,
    bool overwrite)
{
    iterator begin;
    iterator end;
    findCoveredSegments(base, size, begin, end);
    if (overwrite)
    {
        begin->second.size = size;
        begin->second.valid = true;
        begin->second.seg = seg;
        begin++;
        m_segs.erase(begin, end);
    }
    else
    {
        auto prevNonValidIt = end;
        for (auto it = begin; it != end; /* nothing */)
        {
            auto currIt = it++;

            if (currIt->second.valid)
            {
                prevNonValidIt = end;
            }
            else if (prevNonValidIt == end)
            {
                prevNonValidIt = currIt;
                prevNonValidIt->second.valid = true;
                prevNonValidIt->second.seg = seg;
            }
            else
            {
                prevNonValidIt->second.size += currIt->first;
                m_segs.erase(currIt);
            }
        }
    }
}

template<typename SEG, class SEG_SPLIT, class SEG_MERGE>
bool SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::getSegments(
    const uint64_t base,
    const uint64_t size,
    std::vector<SEG*> &segs)
{
    bool allValid = true;

    iterator begin;
    iterator end;
    findCoveredSegments(base, size, begin, end);
    for (auto it = begin; it != end; ++it)
    {
        if (it->second.valid)
        {
            segs.push_back(&it->second.seg);
        }
        else
        {
            allValid = false;
        }
    }

    return allValid;
}

template<typename SEG, class SEG_SPLIT, class SEG_MERGE>
bool SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::getSegments(const uint64_t base,
                                                           const uint64_t size,
                                                           std::vector<std::tuple<uint64_t, uint64_t, SEG*>>& segs)
{
    bool allValid = true;

    iterator begin;
    iterator end;
    findCoveredSegments(base, size, begin, end);
    for (auto it = begin; it != end; ++it)
    {
        if (it->second.valid)
        {
            uint64_t startAddr = it->first;
            auto nextIt = std::next(it);
            uint64_t endAddr = (nextIt == end) ? base + size : nextIt->first;
            segs.emplace_back(std::make_tuple(startAddr, endAddr, &it->second.seg));
        }
        else
        {
            allValid = false;
        }
    }

    return allValid;
}

template <class SEG, class SEG_SPLIT, class SEG_MERGE>
void SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::getCoveredSegments(
        const uint64_t base,
        const uint64_t size,
        const_iterator &cbegin,
        const_iterator &cend)
{
    iterator begin;
    iterator end;

    findCoveredSegments(base, size, begin, end);
    cbegin = begin;
    cend = end;
}

template <class SEG, class SEG_SPLIT, class SEG_MERGE>
void SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::mergeSegments()
{
    auto start = m_segs.begin();
    while (start != m_segs.end())
    {
        auto end = start;
        end++;
        while (end != m_segs.end())
        {
            if (start->second.valid != end->second.valid)
            {
                break;
            }
            else if ((start->second.valid) && (!SEG_MERGE::merge(start->second.seg, end->second.seg, end->second.size)))
            {
                break;
            }
            start->second.size += end->second.size;
            end = m_segs.erase(end);
        }
        start = end;
    }
}

template <class SEG, class SEG_SPLIT, class SEG_MERGE>
bool SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::isSegmentCovered(const uint64_t base, const uint64_t size) const
{
    auto startIt = m_segs.lower_bound(base);
    if ((startIt == m_segs.end()) || (startIt->first != base))
    {
        startIt--;
    }

    const auto endIt = m_segs.lower_bound(base + size);

    while (startIt != endIt)
    {
        if (!startIt->second.valid)
        {
            return false;
        }
        startIt++;
    }
    return true;
}

template <class SEG, class SEG_SPLIT, class SEG_MERGE>
bool SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::isSegmentInvalid(const uint64_t base, const uint64_t size) const
{
    auto startIt = m_segs.lower_bound(base);
    if ((startIt == m_segs.end()) || (startIt->first != base))
    {
        startIt--;
    }

    const auto endIt = m_segs.lower_bound(base + size);

    while (startIt != endIt)
    {
        if (startIt->second.valid)
        {
            return false;
        }
        startIt++;
    }
    return true;
}

template <class SEG, class SEG_SPLIT, class SEG_MERGE>
void SegmentsSpace<SEG, SEG_SPLIT, SEG_MERGE>::erase(const uint64_t base, const uint64_t size)
{
    iterator begin;
    iterator end;
    findCoveredSegments(base, size, begin, end);
    begin->second.size = size;
    begin->second.valid = false;
    begin->second.seg = SEG();
    begin++;
    m_segs.erase(begin, end);
}

#endif //MME__SEGMENTS_SPACE_H
