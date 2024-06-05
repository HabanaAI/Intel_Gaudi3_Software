#pragma once

#include "graph_compiler/sync/sync_conventions.h"

namespace gaudi
{

#define GAUDI_SYNC_GROUP_SIZE (8)

enum
{
    SYNC_OBJ_MIN_SAVED_ID = 0,  // MUST BE FIRST

    SYNC_OBJ_DMA_DOWN_FEEDBACK =
        0,                         // On finish, incremented by one when write complete flag is on (DMA 1) - Hard coded
    SYNC_OBJ_DMA_UP_FEEDBACK = 1,  // On finish, incremented by one when write complete flag is on (DMA 2) - Hard coded
    SYNC_OBJ_DMA_STATIC_DRAM_SRAM_FEEDBACK =
        2,  // On finish, incremented by one when write complete flag is on (DMA 3) - Hard coded
    SYNC_OBJ_DMA_SRAM_DRAM_FEEDBACK =
        3,  // On finish, incremented by one when write complete flag is on (DMA 4) - Hard coded
    SYNC_OBJ_FIRST_COMPUTE_FINISH = 4,
    SYNC_OBJ_HOST_DRAM_DONE       = 5,
    SYNC_OBJ_DBG_CTR_DEPRECATED   = 6,
    SYNC_OBJ_DMA_ACTIVATIONS_DRAM_SRAM_FEEDBACK =
        7,  // On finish, incremented by one when write complete flag is on (DMA 0) - Hard coded
    SYNC_OBJ_ENGINE_SEM_BASE  = 8,
    SYNC_OBJ_ENGINE_SEM_MME_0 = 8,
    SYNC_OBJ_ENGINE_SEM_MME_1 = 9,
    SYNC_OBJ_ENGINE_SEM_TPC_0 = 10,
    SYNC_OBJ_ENGINE_SEM_TPC_1 = 11,
    SYNC_OBJ_ENGINE_SEM_TPC_2 = 12,
    SYNC_OBJ_ENGINE_SEM_TPC_3 = 13,
    SYNC_OBJ_ENGINE_SEM_TPC_4 = 14,
    SYNC_OBJ_ENGINE_SEM_TPC_5 = 15,
    SYNC_OBJ_ENGINE_SEM_TPC_6 = 16,
    SYNC_OBJ_ENGINE_SEM_TPC_7 = 17,
    SYNC_OBJ_ENGINE_SEM_DMA_1 = 18,
    SYNC_OBJ_ENGINE_SEM_DMA_2 = 19,
    SYNC_OBJ_ENGINE_SEM_DMA_3 = 20,
    SYNC_OBJ_ENGINE_SEM_DMA_4 = 21,
    SYNC_OBJ_ENGINE_SEM_DMA_5 = 22,
    SYNC_OBJ_ENGINE_SEM_DMA_6 = 23,
    SYNC_OBJ_ENGINE_SEM_DMA_7 = 24,
    SYNC_OBJ_ENGINE_SEM_END   = 24,
    SYNC_OBJ_DBG_CTR_0        = 25,
    SYNC_OBJ_DBG_CTR_1        = 26,  // must follow DBG_CTR_0

    SYNC_OBJ_SIGNAL_OUT_GROUP_0 = 32,  // Must be multiple of 8 (for grouping)
    SYNC_OBJ_SIGNAL_OUT_GROUP_1 = 33,
    SYNC_OBJ_SIGNAL_OUT_GROUP_2 = 34,
    SYNC_OBJ_SIGNAL_OUT_GROUP_3 = 35,

    SYNC_OBJ_MAX_SAVED_ID  // MUST BE LAST
};

enum {
    MON_OBJ_MIN_SAVED_ID               = 0, // MUST BE FIRST

    MON_OBJ_ENGINE_SEM_BASE            = 0,
    MON_OBJ_ENGINE_SEM_MME_0           = 0,
    MON_OBJ_ENGINE_SEM_MME_1           = 1,
    MON_OBJ_ENGINE_SEM_TPC_0           = 2,
    MON_OBJ_ENGINE_SEM_TPC_1           = 3,
    MON_OBJ_ENGINE_SEM_TPC_2           = 4,
    MON_OBJ_ENGINE_SEM_TPC_3           = 5,
    MON_OBJ_ENGINE_SEM_TPC_4           = 6,
    MON_OBJ_ENGINE_SEM_TPC_5           = 7,
    MON_OBJ_ENGINE_SEM_TPC_6           = 8,
    MON_OBJ_ENGINE_SEM_TPC_7           = 9,
    MON_OBJ_ENGINE_SEM_DMA_1           = 10,
    MON_OBJ_ENGINE_SEM_DMA_2           = 11,
    MON_OBJ_ENGINE_SEM_DMA_3           = 12,
    MON_OBJ_ENGINE_SEM_DMA_4           = 13,
    MON_OBJ_ENGINE_SEM_DMA_5           = 14,
    MON_OBJ_ENGINE_SEM_DMA_6           = 15,
    MON_OBJ_ENGINE_SEM_DMA_7           = 16,
    MON_OBJ_ENGINE_SEM_END             = 16,

    MON_OBJ_MAX_SAVED_ID  // MUST BE LAST
};

class SyncConventions : public ::SyncConventions
{
public:
    static SyncConventions& instance();
    std::string getSyncObjName(unsigned int objId) const override;
    unsigned getSyncObjMinSavedId() const override                     {return SYNC_OBJ_MIN_SAVED_ID;}
    unsigned getSyncObjMaxSavedId() const override                     {return SYNC_OBJ_MAX_SAVED_ID;}
    unsigned getSyncObjDmaDownFeedback() const override                {return SYNC_OBJ_DMA_DOWN_FEEDBACK;}
    unsigned getSyncObjDmaUpFeedback() const override                  {return SYNC_OBJ_DMA_UP_FEEDBACK;}
    unsigned getSyncObjDmaStaticDramSramFeedback() const override      {return SYNC_OBJ_DMA_STATIC_DRAM_SRAM_FEEDBACK;}
    unsigned getSyncObjDmaSramDramFeedback() const override            {return SYNC_OBJ_DMA_SRAM_DRAM_FEEDBACK;}
    unsigned getSyncObjFirstComputeFinish() const override             {return SYNC_OBJ_FIRST_COMPUTE_FINISH;}
    unsigned getSyncObjHostDramDone() const override                   {return SYNC_OBJ_HOST_DRAM_DONE;}
    unsigned getSyncObjDbgCtr() const override                         {return SYNC_OBJ_DBG_CTR_0;}
    unsigned getSyncObjDmaActivationsDramSramFeedback() const override {return SYNC_OBJ_DMA_ACTIVATIONS_DRAM_SRAM_FEEDBACK;}
    unsigned getSyncObjEngineSem(unsigned engineIdx) const override    {return SYNC_OBJ_ENGINE_SEM_BASE + engineIdx;}
    unsigned getMonObjMinSavedId() const override                      {return MON_OBJ_MIN_SAVED_ID;}
    unsigned getMonObjMaxSavedId() const override                      {return MON_OBJ_MAX_SAVED_ID;}
    unsigned getMonObjEngineSemBase() const override                   {return MON_OBJ_ENGINE_SEM_BASE;}
    unsigned getMonObjDmaDownFeedbackReset() const override            {return 0;}
    unsigned getMonObjDmaUpFeedbackReset() const override              {return 0;}
    int getSyncAddOp() const  override                                 {return SYNC_OP_ADD;}
    int getSyncSetOp() const  override                                 {return SYNC_OP_SET;}
    unsigned getGroupSize() const override                             {return GAUDI_SYNC_GROUP_SIZE;}

    unsigned getLowerQueueID(HabanaDeviceType deviceType, unsigned engineId) const override;

    unsigned getSignalOutGroup() const override { return SYNC_OBJ_SIGNAL_OUT_GROUP_0; }

    bool isSignalOutGroupSupported() const override { return true; }
    unsigned getNumOfSignalGroups() const override
    {
        return SYNC_OBJ_SIGNAL_OUT_GROUP_3 - SYNC_OBJ_SIGNAL_OUT_GROUP_0 + 1;
    }
};

} // namespace gaudi
