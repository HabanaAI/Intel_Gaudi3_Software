#pragma once

#include "mapped_mem_mgr.hpp"
#include "hbm_global_mem_mgr.hpp"
#include "arc_hbm_mem_mgr.hpp"

struct MemMgrs
{
    MemMgrs(std::string mmmName, DevMemoryAllocInterface& devMemoryAllocInterface)
    : mappedMemMgr(mmmName, devMemoryAllocInterface)
    {
    }

    HbmGlblMemMgr hbmGlblMemMgr;
    MappedMemMgr  mappedMemMgr;
    ArcHbmMemMgr  arcHbmMemMgr;
};
