#include "sync_conventions.h"
#include "infra/compile_time_asserter.h"
#include "infra/defs.h"

namespace gaudi
{

SyncConventions& SyncConventions::instance()
{
    // Singleton implementation - static variable is created only once.
    static SyncConventions onlyOneInstance;
    return onlyOneInstance;
}

std::string SyncConventions::getSyncObjName(unsigned int objId) const
{
    std::string SyncObjName;
    static const char* speSyncObjName[] = {"SYNC_OBJ_DMA_DOWN_FEEDBACK",
                                           "SYNC_OBJ_DMA_UP_FEEDBACK",
                                           "SYNC_OBJ_DMA_STATIC_DRAM_SRAM_FEEDBACK",
                                           "SYNC_OBJ_DMA_SRAM_DRAM_FEEDBACK",
                                           "SYNC_OBJ_FIRST_COMPUTE_FINISH",
                                           "SYNC_OBJ_HOST_DRAM_DONE",
                                           "SYNC_OBJ_DBG_CTR_DEPRECATED",
                                           "SYNC_OBJ_DMA_ACTIVATIONS_DRAM_SRAM_FEEDBACK",
                                           "SYNC_OBJ_ENGINE_SEM_MME_0",
                                           "SYNC_OBJ_ENGINE_SEM_MME_1",
                                           "SYNC_OBJ_ENGINE_SEM_TPC_0",
                                           "SYNC_OBJ_ENGINE_SEM_TPC_1",
                                           "SYNC_OBJ_ENGINE_SEM_TPC_2",
                                           "SYNC_OBJ_ENGINE_SEM_TPC_3",
                                           "SYNC_OBJ_ENGINE_SEM_TPC_4",
                                           "SYNC_OBJ_ENGINE_SEM_TPC_5",
                                           "SYNC_OBJ_ENGINE_SEM_TPC_6",
                                           "SYNC_OBJ_ENGINE_SEM_TPC_7",
                                           "SYNC_OBJ_ENGINE_SEM_DMA_1",
                                           "SYNC_OBJ_ENGINE_SEM_DMA_2",
                                           "SYNC_OBJ_ENGINE_SEM_DMA_3",
                                           "SYNC_OBJ_ENGINE_SEM_DMA_4",
                                           "SYNC_OBJ_ENGINE_SEM_DMA_5",
                                           "SYNC_OBJ_ENGINE_SEM_DMA_6",
                                           "SYNC_OBJ_ENGINE_SEM_DMA_7",
                                           "SYNC_OBJ_DBG_CTR_0",
                                           "SYNC_OBJ_DBG_CTR_1",  // 26
                                           "",
                                           "",
                                           "",
                                           "",
                                           "",
                                           "SYNC_OBJ_SIGNAL_OUT_GROUP_0",
                                           "SYNC_OBJ_SIGNAL_OUT_GROUP_1",
                                           "SYNC_OBJ_SIGNAL_OUT_GROUP_2",
                                           "SYNC_OBJ_SIGNAL_OUT_GROUP_3"

    };

    COMPILE_TIME_ASSERT_VERIFY_ARRAY_SIZE(speSyncObjName, SYNC_OBJ_MAX_SAVED_ID - SYNC_OBJ_MIN_SAVED_ID);

    if ((objId >= SYNC_OBJ_MIN_SAVED_ID) && (objId < SYNC_OBJ_MAX_SAVED_ID))
    {
        SyncObjName = speSyncObjName[objId - SYNC_OBJ_MIN_SAVED_ID];
    }

    return SyncObjName;
}

unsigned SyncConventions::getLowerQueueID(HabanaDeviceType deviceType, unsigned engineId) const
{
    unsigned baseIdx = SYNC_OBJ_ENGINE_SEM_BASE;
    unsigned semIdx;

    switch (deviceType)
    {
        case DEVICE_MME:
            semIdx = SYNC_OBJ_ENGINE_SEM_MME_0 + engineId;
            break;
        case DEVICE_TPC:
            semIdx = SYNC_OBJ_ENGINE_SEM_TPC_0 + engineId;
            break;
        case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
            semIdx = SYNC_OBJ_ENGINE_SEM_DMA_2 + engineId;
            break;
        default:
            HB_ASSERT(false, "not supported");
            return 0;
    }
    return semIdx - baseIdx;
}

} // namespace gaudi