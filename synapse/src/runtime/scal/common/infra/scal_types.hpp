#pragma once

#include "scal.h"

#include "define_synapse_common.hpp"

#include <array>
#include <cstdint>
#include <chrono>

class ScalStreamCopyInterface;

struct ComputeCompoundResources
{
    ScalStreamCopyInterface* m_pRxCommandsStream      = nullptr;
    ScalStreamCopyInterface* m_pTxCommandsStream      = nullptr;
    ScalStreamCopyInterface* m_pDev2DevCommandsStream = nullptr;

    unsigned m_streamIndex = std::numeric_limits<unsigned>::max();
};

enum LaunchDebugType
{
    COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD = 0x00000001L,
    // NOTE!!!: these have a lot of limitations. They assume
    // recipe is still available (not destroyed), it assumes
    // recipe wasn't stepped over in HBM and more. Use
    // wisely
    COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD_POST = 0x00000002L,
    COMPARE_RECIPE_ON_DEVICE_AFTER_LAUNCH        = 0x00000004L,
    SEND_EACH_PACKET                             = 0x00000008L,
    PROTECT_MAPPED_MEM                           = 0x00000010L,
};

struct ScalLongSyncObject
{
    uint64_t m_index;
    uint64_t m_targetValue;

    bool operator!=(const ScalLongSyncObject& o) const;
};

const ScalLongSyncObject LongSoEmpty {0, 0};

struct PoolMemoryStatus
{
    uint64_t free;
    uint64_t total;
    uint64_t devBaseAddr;
    uint64_t totalSize;
};

struct RecipeSeqId
{
    explicit RecipeSeqId(uint64_t in) { val = in; }
    uint64_t val;
};

struct EntryIds
{
    RecipeSeqId recipeId;
    uint64_t    runningId;
};

typedef uint64_t MonitorAddressesType[10];
typedef uint32_t MonitorValuesType[10];

typedef unsigned MonitorIdType;
typedef uint8_t  FenceIdType;

struct EngineGrpArr
{
    uint8_t numEngineGroups;
    uint8_t eng[4];
};

struct ResourceInformation
{
    EngineGrpArr engineGrpArr;
    uint32_t     targetVal;
};

struct McidInfo
{
    uint16_t mcidDegradeCount {0};  // number of mcid degarde per recipe
    uint16_t mcidDiscardCount {0};  // number of mcid discard per recipe
};

enum class ResourceStreamType : uint8_t
{
    USER_DMA_UP,
    FIRST      = USER_DMA_UP,
    PDMA_FIRST = USER_DMA_UP,
    USER_DMA_DOWN,
    USER_DEV_TO_DEV,
    SYNAPSE_DMA_UP,
    SYNAPSE_DMA_DOWN,
    SYNAPSE_DEV_TO_DEV,
    PDMA_LAST = SYNAPSE_DEV_TO_DEV,
    COMPUTE,
    AMOUNT
};

struct DevStreamInfo
{
    std::array<uint32_t, SCAL_NUMBER_OF_GROUPS> clusterTypeCompletionsAmountDB;

    std::array<ResourceInformation, (uint8_t)ResourceStreamType::AMOUNT> resourcesInfo;
};

struct ScalDevSpecificInfo
{
    uint64_t dramBaseAddr {0};
    uint64_t dramEndAddr {0};
};

struct CgTdrInfo
{
    uint64_t                                           prevCompleted;
    bool                                               armed;
    std::chrono::time_point<std::chrono::steady_clock> armTime;
};

enum class TdrType
{
    CHECK,   // check for timeout
    DFA,     // show current status
    CLOSE_STREAM // when closing the stream
};

struct TdrRtn
{
    bool        failed = false;
    std::string msg;
};

enum class PdmaDir
{
    DEVICE_TO_HOST,
    HOST_TO_DEVICE,
    INVALID
};

enum class LinPdmaBarrierMode
{
    DISABLED,
    INTERNAL,
    EXTERNAL
};
