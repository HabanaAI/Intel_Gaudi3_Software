#pragma once

#include <stack>
#include <deque>

#include "log_manager.h"

class SegmentAlloc
{
public:
    SegmentAlloc() {}
    virtual ~SegmentAlloc()
    {
        // check sanity before release
        isSanityChkOk(false);
    }

    void init(uint16_t numSegments)
    {
        m_numSegments = numSegments;

        m_freeSegments.resize(numSegments);
        for (int i = 0; i < numSegments; i++)
        {
            m_freeSegments[i] = i;
        }
    }

    size_t size() { return m_freeSegments.size(); }

    std::vector<uint16_t> getSegments(int numSegments)
    {
        if (m_freeSegments.size() < numSegments)
        {
            return {};
        }

        std::vector<uint16_t> allocSegments(numSegments);

        for (int i = 0; i < numSegments; i++)
        {
            allocSegments[i] = m_freeSegments.back();
            m_freeSegments.pop_back();
        }
        return allocSegments;
    }

    void releaseSegments(const std::vector<uint16_t>& usedSegments)
    {
        for (int seg : usedSegments)
        {
            m_freeSegments.push_back(seg);
        }
    }

    bool isSanityChkOk(bool expectAll)
    {
        bool ok = true;

        if (expectAll)
        {
            if (m_freeSegments.size() != m_numSegments)
            {
                LOG_ERR_T(SYN_MEM_ALLOC, "Not all segments are free {} != {}", m_numSegments, m_freeSegments.size());
                ok = false;
            }
        }

        std::vector<bool> inFreeDb(m_numSegments);

        for (int i = 0; i < m_freeSegments.size(); i++)
        {
            auto segment = m_freeSegments[i];
            if (inFreeDb[segment] == true)  // already in free list
            {
                LOG_ERR_T(SYN_MEM_ALLOC, "segment {:x} is at least twice in free db", segment);
                ok = false;
            }
            inFreeDb[segment] = true;
        }

        if (expectAll)  // log also the missing ones
        {
            for (int i = 0; i < m_freeSegments.size(); i++)
            {
                if (inFreeDb[i] == false)
                {
                    LOG_ERR_T(SYN_MEM_ALLOC, "segment {:x} was not found", i);
                }
            }
        }
        return ok;
    }

private:
    std::vector<uint16_t> m_freeSegments;
    uint16_t              m_numSegments;
};
