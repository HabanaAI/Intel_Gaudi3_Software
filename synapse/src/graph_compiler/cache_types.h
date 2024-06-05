#pragma once

#include <cstdint>
#include "habana_global_conf.h"
#include "llvm/small_vector.h"
#include "sync/sync_types.h"

static constexpr int32_t MAX_NUM_DCORES = 4;

using LogicalMcid  = uint64_t;
using PhysicalMcid = uint16_t;

typedef enum
{
    MEMORY_TYPE_SRAM,
    MEMORY_TYPE_DRAM,
} MemoryType;

typedef enum
: uint8_t
{
    NotSupported,
    SkipCache,
    NoAllocate,
    HomeAllocate,
    DcoreAllocate,
    SharedAllocate,
} CacheDirective;

typedef enum
: uint8_t
{
    Low      = 0x0,
    Normal   = 0x1,
    High     = 0x2,
    Top      = 0x3,
} CacheClass;

typedef enum
{
    NOP     = 0,
    DEGRADE = 1,
    DISCARD = 2,

    // Must be last
    LAST_CM_ACTION = DISCARD
} CacheMaintenanceAction;
#define MAX_CM_ACTION_TYPES (CacheMaintenanceAction::LAST_CM_ACTION + 1)

struct DcoreROI
{
    TOffset baseOffset[HABANA_DIM_MAX] = {0};
    TSize   size[HABANA_DIM_MAX]       = {0}; // in index space
};
typedef std::vector<DcoreROI> DcoreRoisVec;

struct CacheMetaData
{
    CacheDirective         cacheDirective = (CacheDirective)GCFG_DEFAULT_CACHE_DIRECTIVE.value();
    CacheMaintenanceAction cmAction       = NOP;
    LogicalMcid            mcid           = 0;
    CacheClass             cacheClass     = Normal;

    bool operator==(const CacheMetaData& rhs) const
    {
        return (cacheDirective == rhs.cacheDirective && cmAction == rhs.cmAction && mcid == rhs.mcid &&
                cacheClass == rhs.cacheClass);
    }

    std::string print_directive() const
    {
        switch (cacheDirective)
        {
            case NotSupported:
                return "notSupported";
            case SkipCache:
                return "skipCache";
            case NoAllocate:
                return "noAlloc";
            case HomeAllocate:
                return "allocH";
            case DcoreAllocate:
                return "allocD";
            case SharedAllocate:
                return "allocDH";
        }

        return "";
    }

    std::string print_class() const
    {
        switch (cacheClass)
        {
            case Low:
                return "Low";
            case Normal:
                return "Normal";
            case High:
                return "High";
            case Top:
                return "Top";
        }

        return "";
    }

    std::string print_action() const
    {
        switch (cmAction)
        {
            case NOP:
                return "Nop";
            case DEGRADE:
                return "Degrade";
            case DISCARD:
                return "Discard";
        }

        return "";
    }
};

struct CmCmd
{
    LogicalMcid            mcid = 0;
    CacheMaintenanceAction op = NOP;
    DependencyMap          deps;
};

struct CmeRollover
{
    bool                   doRollover = false;
    unsigned               rolloverId = 0;
    uint8_t                rolloverEngineBitmap = 0; // bit 0 for mme, bit 1 for rot
};

struct CmeSobReset
{
    unsigned               sobResetTotalNumEngs = 0; // if > 0 then reset is needed
    unsigned               sobResetId = 0;
};

struct CmeTasks
{
    std::vector<CmCmd> cmCmds;
    CmeRollover        rollover;
    CmeSobReset        sobReset;
};
