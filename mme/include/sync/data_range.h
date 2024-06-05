#ifndef MME__DATA_RANGE_H
#define MME__DATA_RANGE_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

template<class T>
class DataRange
{
public:
    explicit DataRange(const T& start = T(), const T& end = T()) : m_start(start), m_end(end) {}

    bool operator<(const DataRange& rhs) const
    {
        return m_start != rhs.m_start ? m_start < rhs.m_start : m_end < rhs.m_end;
    }

    bool isOverlap(const DataRange& rhs) const { return (m_start < rhs.m_end && m_end > rhs.m_start); }

    const T size() const { return m_end - m_start; }

    const T& start() const { return m_start; }

    const T& end() const { return m_end; }

    void start(const T& start) { m_start = start; }

    void end(const T& end) { m_end = end; }

    void size(const T& size) { m_end = m_start + size; }

    void addStart(const T& addend) { m_start += addend; }

    void addEnd(const T& addend) { m_end += addend; }

    void shiftRange(const T& shift)
    {
        addStart(shift);
        addEnd(shift);
    }

    std::vector<DataRange<T>> splitAt(const T& position)
    {
        assert(start() <= position && position <= end() && "Position must be inside the data range");
        if (start() == position || position == end())
        {
            return {*this};
        }
        return {DataRange<T> {start(), position}, DataRange<T> {position, end()}};
    }

    /**
     * Merge range with a given range
     * Return true if merge was successful
     */
    bool merge(const DataRange& rhs)
    {
        if (isOverlap(rhs) || m_start == rhs.end() || m_end == rhs.start())
        {
            start(std::min(m_start, rhs.start()));
            end(std::max(m_end, rhs.end()));
            return true;
        }
        else
        {
            return false;
        }
    }

    template<size_t Alignment>
    DataRange<T> alignBeginning() const
    {
        return DataRange<T> {start() - start() % Alignment, end()};
    }
    // Will join non-adjacent ranges (e.g. 10-20 and 30-40 to 10-40)
    DataRange<T> join(const DataRange& other) const
    {
        return DataRange<T>(std::min(m_start, other.m_start), std::max(m_end, other.m_end));
    }

    template<typename DataRangeContainer>
    void splitToChunks(DataRangeContainer& out, T chunkSize) const
    {
        out.reserve(round_to_multiple(size(), chunkSize));
        T position = start();
        while (end() - position > chunkSize)
        {
            out.emplace_back(position, position + chunkSize);
            position += chunkSize;
        }

        if (end() - position != 0)
        {
            out.emplace_back(position, end());
        }
    }

    template<typename DataRangeContainer>
    void splitEvenlyAsBestAsPossibleWithPreffered(DataRangeContainer& out,
                                                  size_t groupCount,
                                                  T chunkSize,
                                                  uint32_t prefferedMax = 0) const
    {
        splitEvenlyAsBestAsPossible(out, groupCount, chunkSize);
        std::sort(out.begin(), out.end());
        if (prefferedMax != 0)
        {
            std::rotate(out.begin(), out.begin() + out.size() - prefferedMax, out.end());
        }
    }

    template<typename DataRangeContainer>
    void splitEvenlyAsBestAsPossible(DataRangeContainer& out, size_t groupCount, T chunkSize) const
    {
        if (groupCount == 1)
        {
            out.push_back(*this);
            return;
        }
        T position = start();
        out.reserve(groupCount);
        for (size_t i = 0; i < groupCount && position != end(); i++)
        {
            auto elementsLeft = (end() - position);
            auto groupsLeft = groupCount - i;
            auto avgElementsPerGroupLeft = std::max<T>(elementsLeft / groupsLeft, 1);
            auto rounded = round_to_multiple(avgElementsPerGroupLeft, chunkSize);
            if (position + rounded > end())
            {
                rounded = end() - position;
            }
            out.emplace_back(position, position + rounded);
            position += rounded;
        }
    }

    // temporary, should be removed once mme and synapse new code get promoted
    // currently kept for backward compatabillity with pre-existing synapse code.
    std::vector<DataRange<T>> splitToChunks(T chunkSize) const
    {
        std::vector<DataRange<T>> result;
        splitToChunks(result, chunkSize);
        return result;
    }

    std::vector<DataRange<T>>
    splitEvenlyAsBestAsPossibleWithPreffered(size_t groupCount, T chunkSize, uint32_t prefferedMax = 0) const
    {
        std::vector<DataRange<T>> result;
        splitEvenlyAsBestAsPossibleWithPreffered(result, groupCount, chunkSize, prefferedMax);
        return result;
    }

    std::vector<DataRange<T>> splitEvenlyAsBestAsPossible(size_t groupCount, T chunkSize) const
    {
        std::vector<DataRange<T>> result;
        splitEvenlyAsBestAsPossible(result, groupCount, chunkSize);
        return result;
    }

    bool operator==(const DataRange<T>& rhs) const { return (m_start == rhs.m_start) && (m_end == rhs.m_end); }

private:
    T m_start;
    T m_end;
};

class CyclicDataRange
{
public:
    explicit CyclicDataRange(uint64_t start, uint64_t end, uint32_t stride) : m_stride(stride)
    {
        assert(stride != 0 && "got zero stride!!");
        assert(end - start <= m_stride);
        m_start = modN(start);
        m_end = modN(end);
        m_size = (start + m_stride == end) ? m_stride : modN(m_end - m_start);
    }

    // for infinite cyclic ranges with the same stride
    bool isOverlap(const CyclicDataRange& rhs) const
    {
        if (m_stride != rhs.m_stride)
        {
            int32_t lcm = (m_stride / gcd(m_stride, rhs.m_stride)) * rhs.m_stride;
            return isOverlap(rhs, 0, lcm);
        }
        return ((m_size != 0) && (modN(m_start - rhs.m_start) < rhs.m_size)) ||
               ((rhs.m_size != 0) && (modN(rhs.m_start - m_start) < m_size));
    }

    // for checking is a linear range overlaps a cyclic range
    bool isOverlap(uint64_t start, uint64_t end) const
    {
        if (start == end || m_size == 0) return false;  // size 0, no overlap
        if ((end - start) >= m_stride) return true;  // size larger than stride - always overlap
        return isOverlap(CyclicDataRange(start, end, m_stride));
    }

    // check overlap with other cyclic range in a finite interval
    bool isOverlap(const CyclicDataRange& rhs, uint64_t start, uint64_t end) const
    {
        if ((m_size == 0) || (rhs.m_size == 0) || (start == end)) return false;  // size 0, no overlap
        if ((m_stride == rhs.m_stride) && ((end - start) >= m_stride))  // treat like infinite cyclic ranges
        {
            return isOverlap(rhs);
        }

        // different stride overlap - worst case, the amount of linear ranges will be O(lcm/stride)
        int32_t lcm = (m_stride / gcd(m_stride, rhs.m_stride)) * rhs.m_stride;
        uint64_t newEnd = std::min(end, start + lcm);
        return isOverlap(*this, rhs, start, newEnd);
    }

    bool pointInRange(uint64_t x) const { return modN(x - m_start) < m_size; }

    bool operator==(const CyclicDataRange& rhs) const
    {
        return (m_stride == rhs.m_stride) && (m_start == rhs.m_start) && (m_end == rhs.m_end);
    }

    void shift(uint64_t offset)
    {
        m_start = modN((int64_t) m_start + offset);
        m_end = modN((int64_t) m_end + offset);
    }

    int32_t size() const { return m_size; }

    int32_t start() const { return m_start; }

    uint32_t stride() const { return m_stride; }

    int32_t end() const { return m_end; }

private:
    // "real" mathematical modulo operation
    int32_t modN(int64_t x) const
    {
        int32_t res = x % m_stride;
        return res >= 0 ? res : m_stride + res;
    }

    // check for overlap between 2 cyclic ranges in a finite interval
    static bool isOverlap(const CyclicDataRange& a, const CyclicDataRange& b, uint64_t ustart, uint64_t uend)
    {
        assert(uend < std::numeric_limits<int64_t>::max());
        int64_t end = uend;
        int64_t start = ustart;
        int64_t aPtr = a.alignStart(start);
        int64_t bPtr = b.alignStart(start);
        while (aPtr < end && bPtr < end)
        {
            if (std::max({aPtr, bPtr, start}) < std::min({aPtr + a.size(), bPtr + b.size(), end}))
            {
                return true;
            }
            if (aPtr < bPtr || (aPtr == bPtr && a.size() < b.size()))
            {
                aPtr += a.m_stride;
            }
            else
            {
                bPtr += b.m_stride;
            }
        }
        return false;
    }

    static int32_t gcd(int32_t a, int32_t b)
    {
        while (b != 0)
        {
            int32_t tmp = b;
            b = a % b;
            a = tmp;
        }
        return a;
    }

    int64_t alignStart(int64_t x) const
    {
        int64_t xMod = modN(x);
        int64_t res = x + m_start - xMod;
        if (m_start < m_end)
        {
            return (m_end <= xMod) ? res + m_stride : res;
        }
        return (m_end > xMod) ? res - m_stride : res;
    }

    int32_t m_start;
    int32_t m_end;
    uint32_t m_stride;
    int32_t m_size;  // saved for caching
};

#endif //MME__DATA_RANGE_H
