#pragma once

#include "synapse_common_types.h"
#include "recipe_package_types.hpp"
#include "settable.h"

struct DeviceAgnosticRecipeStaticInfo
{
    DeviceAgnosticRecipeStaticInfo();

    bool getProgramCommandsChunksAmount(eExecutionStage stage, uint64_t& rProgramCommandsChunksAmount) const;

    void setProgramCommandsChunksAmount(eExecutionStage stage, uint64_t programCommandsChunksAmount);

    uint64_t getMaxProgramCommandsChunksAmount() const;

    bool getDcSizeCommand(uint64_t& chunkSize) const;

    void setDcSizeCommand(uint64_t chunkSize);

    void setWorkCompletionProgramIndex(eExecutionStage stage, uint64_t workCompletionProgramIndex);

    bool getWorkCompletionProgramIndex(eExecutionStage stage, uint64_t& rWorkCompletionProgramIndex) const;

    Settable<uint64_t> m_programCommandsChunksAmount[EXECUTION_STAGE_LAST];

    uint64_t m_maxProgramCommandsChunksAmount;

    Settable<uint64_t> m_dcSizeCommand;

    Settable<uint64_t> m_workCompletionProgramIndex[EXECUTION_STAGE_LAST];  // where GC sends the work completion blob

    uint64_t m_patchingBlobsChunksSize;

    uint64_t m_patchingBlobsChunksDataChunksAmount;
};
