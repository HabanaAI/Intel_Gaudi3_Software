#include "device_agnostic_recipe_static_info.hpp"
#include "defs.h"
#include "log_manager.h"

DeviceAgnosticRecipeStaticInfo::DeviceAgnosticRecipeStaticInfo()
: m_maxProgramCommandsChunksAmount(0), m_patchingBlobsChunksSize(0), m_patchingBlobsChunksDataChunksAmount(0)
{
}

bool DeviceAgnosticRecipeStaticInfo::getProgramCommandsChunksAmount(eExecutionStage stage,
                                                                    uint64_t&       rProgramCommandsChunksAmount) const
{
    HB_ASSERT(stage != EXECUTION_STAGE_LAST, "Illegal execution stage");

    if (!m_programCommandsChunksAmount[stage].is_set())
    {
        return false;
    }

    rProgramCommandsChunksAmount = m_programCommandsChunksAmount[stage].value();
    return true;
}

void DeviceAgnosticRecipeStaticInfo::setProgramCommandsChunksAmount(eExecutionStage stage,
                                                                    uint64_t        programCommandsChunksAmount)
{
    HB_ASSERT(stage != EXECUTION_STAGE_LAST, "Illegal execution stage");
    m_programCommandsChunksAmount[stage] = programCommandsChunksAmount;
    m_maxProgramCommandsChunksAmount     = std::max(m_maxProgramCommandsChunksAmount, programCommandsChunksAmount);
}

uint64_t DeviceAgnosticRecipeStaticInfo::getMaxProgramCommandsChunksAmount() const
{
    return m_maxProgramCommandsChunksAmount;
}

bool DeviceAgnosticRecipeStaticInfo::getDcSizeCommand(uint64_t& chunkSize) const
{
    if (!m_dcSizeCommand.is_set())
    {
        return false;
    }
    chunkSize = m_dcSizeCommand.value();
    return true;
}

void DeviceAgnosticRecipeStaticInfo::setDcSizeCommand(uint64_t chunkSize)
{
    m_dcSizeCommand.set(chunkSize);
}

void DeviceAgnosticRecipeStaticInfo::setWorkCompletionProgramIndex(eExecutionStage stage,
                                                                   uint64_t        workCompletionProgramIndex)
{
    HB_ASSERT(stage != EXECUTION_STAGE_LAST, "Illegal execution stage");
    m_workCompletionProgramIndex[stage] = workCompletionProgramIndex;
}

bool DeviceAgnosticRecipeStaticInfo::getWorkCompletionProgramIndex(eExecutionStage stage,
                                                                   uint64_t&       rWorkCompletionProgramIndex) const
{
    HB_ASSERT(stage != EXECUTION_STAGE_LAST, "Illegal execution stage");

    if (!m_workCompletionProgramIndex[stage].is_set())
    {
        return false;
    }

    rWorkCompletionProgramIndex = m_workCompletionProgramIndex[stage].value();
    return true;
}
