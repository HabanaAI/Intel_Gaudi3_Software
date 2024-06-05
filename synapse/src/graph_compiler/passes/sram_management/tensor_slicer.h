#pragma once

#include <map>
#include "bundle.h"

// implement the main "lru like" logic of the TensorSlicer in order to support double buffering
template<class T>
class TensorSlicerCache
{
public:
    TensorSlicerCache<T>(const Bundle::Solution::pSlicedOperand& slicedOperand, const std::string& nameForLog);

    T getCachedSlice(const CoordArray& coord, bool log = true);

    void addNewSlice(const CoordArray& coord, const T& newSlice);

    void destroySlice(const CoordArray& coord);

    void logCacheStatus();

private:
    const Bundle::Solution::pSlicedOperand m_sliceOperand;
    const std::string m_tensorNameForLog;

    std::map<CoordArray, T> m_sliceCache;
    std::list<CoordArray> m_cachedCoord;
    std::set<CoordArray> m_createdCoords;
};


/**
 * Giving the slice operand with the original tensor and a single slice chunk
 * TensorSlicer generate the slice according to the coordinate
 * and share the slice between requesters, unless a new slice is requested (forceCreate)
 * or disposing the slice
 */
class TensorSlicer
{
public:
    TensorSlicer(const Bundle::Solution::pSlicedOperand& slicedOperand, uint32_t bundleIdx = 0, const Settable<uint64_t>& multiBufferId = Settable<uint64_t>());

    pTensor getSlice(const HalReader& halReader, const CoordArray& coord, bool forceCreate = false);

    void destroySlice(const CoordArray& coord);

    static void setSliceMemory(pTensor& slice, bool inSram);

    SizeVector getSliceOffsets(const CoordArray& coord);

    bool shouldUseTensorView() { return m_sliceOperand->requiresTensorView; }

private:
    bool shouldAlignToCacheLine();

    /**
     * align tensor to cache line (optimization for MME)
     */
    bool alignTensorToCacheLine(const pTensor& tensor, const HalReader& halReader);

    std::string getSliceName(const CoordArray& coord);

    void setMultiBufferId(pTensor slice);

    Bundle::Solution::pSlicedOperand m_sliceOperand;
    const std::string m_baseName;
    SizeArray m_originalTensorSize;
    std::map<CoordArray, uint32_t> m_coordCount;
    uint32_t m_bundleIdx;
    TensorSlicerCache<pTensor> m_cache;

    const Settable<uint64_t> m_multiBufferId;

};


template<class T>
TensorSlicerCache<T>::TensorSlicerCache(const Bundle::Solution::pSlicedOperand& slicedOperand,
                                        const std::string& nameForLog)
        : m_sliceOperand(slicedOperand),
          m_tensorNameForLog(nameForLog) {}

template<class T>
T TensorSlicerCache<T>::getCachedSlice(const CoordArray& coord, bool log /*= true*/)
{
    auto cachedSlice = m_sliceCache.find(coord);
    if (cachedSlice != m_sliceCache.end())
    {
        if (log)
        {
            LOG_DEBUG(BE_SLICER, "Use cached slice for tensor {}, coordinate [{}]",
                      m_tensorNameForLog, toString(coord, ','));
        }
        return cachedSlice->second;
    }
    return {};
}

template<class T>
void TensorSlicerCache<T>::addNewSlice(const CoordArray& coord, const T& newSlice)
{
    if (m_sliceOperand->resideInSRAM)
    {
        if (m_cachedCoord.size() == m_sliceOperand->numOfBuffers)
        {
            destroySlice(m_cachedCoord.back());
            m_cachedCoord.pop_back();
        }
        m_cachedCoord.push_front(coord);
    }

    // If the coord was already created once, don't cache it again
    if (m_createdCoords.insert(coord).second)
    {
        m_sliceCache[coord] = newSlice;
    }
}

template<class T>
void TensorSlicerCache<T>::destroySlice(const CoordArray& coord)
{
    m_sliceCache.erase(coord);
}

template<class T>
void TensorSlicerCache<T>::logCacheStatus()
{
    std::stringstream cacheString;
    for (auto c: m_cachedCoord)
    {
        cacheString << '[' << toString(c, ',') << "] ";
    }
    LOG_TRACE(BE_SLICER, "{} cached: {}", m_tensorNameForLog, cacheString.str());
}
