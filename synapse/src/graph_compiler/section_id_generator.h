#pragma once

#include <array>
#include <cstdint>
class SectionIDGenerator
{
public:
    using SectionIDType = uint32_t;

    enum AllocationManagementType
    {
        // The user allocates the base address for these sections.
        // For example: persistent tensors or the workspace/program/program-data slabs
        USER_ALLOCATED_SECTIONS,

        // The user provides grouping and an offset from the group base, but the compiler provides the base itself.
        // For example: RMW sections and SRAM/DRAM multi-buffers.
        GC_ALLOCATED_SECTIONS,

        // Must be last
        NOF_ALLOCATION_TYPES
    };

    SectionIDGenerator();

    SectionIDType nextSectionId(AllocationManagementType allocType);

    SectionIDType getNumberOfMemorySections(AllocationManagementType allocType) const;

private:
    std::array<SectionIDType, NOF_ALLOCATION_TYPES> nextIDs{};
    std::array<SectionIDType, NOF_ALLOCATION_TYPES> maxIDs{};

    void initAllGenerators();
};
