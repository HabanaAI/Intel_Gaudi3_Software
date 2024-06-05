#include "section_id_generator.h"

#include "define_synapse_common.hpp"
#include "types_exception.h"
#include "utils.h"

SectionIDGenerator::SectionIDGenerator()
{
    initAllGenerators();
}

void SectionIDGenerator::initAllGenerators()
{
    nextIDs[USER_ALLOCATED_SECTIONS] = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
    // max memory section ID is reserved for SRAM
    maxIDs[USER_ALLOCATED_SECTIONS] =
            static_cast<SectionIDGenerator::SectionIDType>(getMaxMemorySectionID());

    nextIDs[GC_ALLOCATED_SECTIONS] = 0;
    maxIDs[GC_ALLOCATED_SECTIONS] = std::numeric_limits<typeof(nextIDs[0])>::max();
}

SectionIDGenerator::SectionIDType SectionIDGenerator::nextSectionId(AllocationManagementType allocType)
{
    HB_ASSERT(nextIDs[allocType] < maxIDs[allocType], "memory section ID overflow");
    if (allocType >= NOF_ALLOCATION_TYPES)
    {
        throw SynapseException(fmt::format("Unexpected section allocation type: {}", allocType));
    }
    return nextIDs[allocType]++;
}

SectionIDGenerator::SectionIDType
SectionIDGenerator::getNumberOfMemorySections(AllocationManagementType allocType) const
{
    return nextIDs[allocType] + 1;
}
