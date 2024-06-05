#pragma once

#include <stdint.h>
#include "define.hpp"

struct patch_point_t;

namespace patching
{
class HostAddressPatchingInformation  // describes that patching that was done on the data-chunks
{
public:
    HostAddressPatchingInformation();

    virtual ~HostAddressPatchingInformation();

    virtual bool initialize(uint64_t maxSectionId, uint64_t numOfSectionsToPatch);

    virtual bool isInitialized() const { return m_isInitialized; };

    bool isFirstPatching() const { return m_dbgFirstPatching; };

    const uint64_t* getSectionsToHostAddressDB() const;

    uint64_t getSectionsDbSize() const;

    bool markConstZeroSizeSection(uint64_t sectionId);

    bool setSectionHostAddress(uint64_t sectionId,
                               uint64_t sectionTypeId,
                               uint64_t sectionHostAddress,
                               bool     isZeroSizeSection,
                               bool     forceUpdate         = false,
                               bool     isScratchpadSection = false);

    void markSectionTypeForPatching(uint64_t sectionTypeId);

    bool hasNewSectionAddress() const;

    void patchingCompletion();
    void patchingAbort();

    bool validateAllSectionsAddressSet() const;

    SectionTypesToPatch& getSectionTypesQueuedForPatching();

    void dump() const;

protected:
    bool m_isInitialized;

    // DBs with size of m_sectionIdDbSize for o(1) access
    uint64_t* m_sectionToHostAddressDb;  // sec -> host addr
    uint32_t* m_sectionPatchingIndexDb;  // sec patching index
    uint64_t  m_sectionIdDbSize;         // maxID + 1, section Id db (size of the arrays above)

    uint32_t m_numberOfSectionsChecked;  // checked
    uint32_t m_numberOfSectionsUpdated;  // updated
    uint32_t m_currentPatchingIndex;     // patch index

    uint64_t m_numOfSections;  // num sections

    // Number of sections-groups that their address had not been changed, but are marked as if they did
    uint32_t m_dbgNumOfSectionsGroupForecfullyUpdated;

    SectionTypesToPatch m_sectionTypesToPatch;  // set<uint64_t> (section types to patch?)
    const bool          m_shouldLog;

    const bool     m_enforceNewScatchpadSectionAddr;
    const uint64_t m_enforceNumOfAdditionalNewPtsSectionAddr;

    bool m_dbgFirstPatching;
};
}  // namespace patching
