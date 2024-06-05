#ifndef SCAL_SCAL_DATA_H
#define SCAL_SCAL_DATA_H
#include "scal_base.h"

template <typename EngineInfoGx, size_t N>
static inline bool qmanId2DccmAddr(const unsigned qid, uint64_t& dccmAddr, const std::array<EngineInfoGx, N> engineInfoArr)
{
    for (const EngineInfoGx& info : engineInfoArr)
    {
        if (qid == info.queueId)
        {
            dccmAddr = info.dccmAddr;
            return true;
        }
    }

    return false;
}

template <typename EngineInfoGx, size_t N>
static bool arcName2QueueId(const std::string & arcName, unsigned &qid, const std::array<EngineInfoGx, N> engineInfoArr)
{
    qid = (unsigned)-1;
    for (const EngineInfoGx & info : engineInfoArr)
    {
        if (arcName == info.name)
        {
            qid = info.queueId;
            break;
        }
    }

    return (qid != (unsigned)-1);
}


template <typename EngineInfoGx, size_t N>
inline bool arcName2DccmAddr(const std::string & arcName, uint64_t &dccmAddr, const std::array<EngineInfoGx, N> engineInfoArr)
{
    for (const EngineInfoGx & info : engineInfoArr)
    {
        if (arcName == info.name)
        {
            dccmAddr = info.dccmAddr;
            return true;
        }
    }

    return false;
}

template <typename EngineInfoGx, size_t N>
inline bool arcName2CoreType(const std::string & arcName, CoreType &coreType, const std::array<EngineInfoGx, N> engineInfoArr)
{
    coreType = NUM_OF_CORE_TYPES;       // coreType may be uninitialized
    for (const EngineInfoGx & info : engineInfoArr)
    {
        if (arcName == info.name)
        {
            coreType = info.coreType;
            return true;
        }
    }
    return false;
}

template <typename EngineInfoGx, size_t N>
inline bool arcName2CpuId(const std::string & arcName, unsigned &cpuId, const std::array<EngineInfoGx, N> engineInfoArr)
{
    for (const EngineInfoGx & info : engineInfoArr)
    {
        if (arcName == info.name)
        {
            cpuId = info.cpuId;
            return true;
        }
    }

    return false;
}


inline bool isOverlap(unsigned start, unsigned end, unsigned otherStart, unsigned otherEnd)
{
    return  ((start >= otherStart) && (start < otherEnd))
            || ((end >= otherStart) && (end < otherEnd))
            || ((start <= otherStart) && (end >= otherEnd));
}



#endif //SCAL_SCAL_DATA_H
