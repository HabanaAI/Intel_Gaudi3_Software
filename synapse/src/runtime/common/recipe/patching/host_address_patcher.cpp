#include "host_address_patcher.hpp"

#include "define.hpp"

#include "defenders.h"
#include "habana_global_conf.h"
#include "habana_global_conf_runtime.h"
#include "log_manager.h"
#include "recipe.h"
#include "synapse_runtime_logging.h"
#include "utils.h"

#include "define_synapse_common.hpp"

#include <limits>

using namespace patching;

HostAddressPatchingInformation::HostAddressPatchingInformation()
: m_sectionToHostAddressDb(nullptr),
  m_sectionPatchingIndexDb(nullptr),
  m_sectionIdDbSize(0),
  m_numberOfSectionsChecked(0),
  m_numberOfSectionsUpdated(0),
  m_currentPatchingIndex(1),
  m_numOfSections(0),
  m_dbgNumOfSectionsGroupForecfullyUpdated(0),
  m_shouldLog(LOG_LEVEL_AT_LEAST_TRACE(SYN_PATCH_INFO)),
  m_enforceNewScatchpadSectionAddr(GCFG_DBG_ENFORCE_NEW_SCRATCHPAD_SECTION_ADDRESS.value()),
  m_enforceNumOfAdditionalNewPtsSectionAddr(GCFG_DBG_ENFORCE_NUM_OF_NEW_SECTIONS_GROUP_ADDRESSES.value()),
  m_dbgFirstPatching(true)
{
}

HostAddressPatchingInformation::~HostAddressPatchingInformation()
{
    delete[] m_sectionToHostAddressDb;
    delete[] m_sectionPatchingIndexDb;
}

bool HostAddressPatchingInformation::initialize(uint64_t maxSectionId, uint64_t numOfSectionsToPatch)
{
    LOG_TRACE(SYN_PATCHING,
              "HostAddressPatchingInformation::initialize: maxSectionId {} numOfSectionsToPatch {}",
              maxSectionId,
              numOfSectionsToPatch);

    if (maxSectionId == 0)
    {
        LOG_ERR(SYN_PATCHING, "{}: Invalid initialization size (maxSectionId {})", HLLOG_FUNC, maxSectionId);

        return false;
    }

    if (m_sectionIdDbSize != 0)
    {
        if (m_sectionIdDbSize != (maxSectionId + 1))
        {
            LOG_ERR(SYN_PATCHING,
                    "{}: Already initialized with number of valid PP-IDs {} while max PP-ID is ({})",
                    HLLOG_FUNC,
                    m_sectionIdDbSize,
                    maxSectionId);

            return false;
        }

        return true;
    }

    uint64_t sectionIdDbSize = maxSectionId + 1;

    m_sectionToHostAddressDb = new uint64_t[sectionIdDbSize];
    m_sectionPatchingIndexDb = new uint32_t[sectionIdDbSize];

    std::memset(m_sectionToHostAddressDb, 0, sectionIdDbSize * sizeof(uint64_t));
    std::memset(m_sectionPatchingIndexDb, 0, sectionIdDbSize * sizeof(uint32_t));

    m_sectionIdDbSize = sectionIdDbSize;
    m_numOfSections   = numOfSectionsToPatch;

    m_isInitialized = true;

    return true;
}

bool HostAddressPatchingInformation::hasNewSectionAddress() const
{
    HB_ASSERT(isInitialized(), "HostAddressPatchingInformation is not initialized");

    return (m_numberOfSectionsUpdated != 0);
}

bool HostAddressPatchingInformation::validateAllSectionsAddressSet() const
{
    if (m_numberOfSectionsChecked == m_numOfSections)
    {
        return true;
    }

    LOG_ERR(SYN_PATCHING,
            "{}: Not all sections checked (numberOfSectionsChecked {} numOfSections {})",
            HLLOG_FUNC,
            m_numberOfSectionsChecked,
            m_numOfSections);

    uint32_t* pCurrentSectionPatchingIndex = m_sectionPatchingIndexDb;
    for (uint64_t sectionId = 0; sectionId < m_sectionIdDbSize; sectionId++, pCurrentSectionPatchingIndex++)
    {
        if (*pCurrentSectionPatchingIndex != m_currentPatchingIndex)
        {
            LOG_ERR(SYN_PATCHING,
                    "sectionId {} has old patching-info-index {}",
                    sectionId,
                    *pCurrentSectionPatchingIndex);
        }
    }

    return false;
}

const uint64_t* HostAddressPatchingInformation::getSectionsToHostAddressDB() const
{
    HB_ASSERT(isInitialized(), "HostAddressPatchingInformation is not initialized");

    return m_sectionToHostAddressDb;
}

uint64_t HostAddressPatchingInformation::getSectionsDbSize() const
{
    HB_ASSERT(isInitialized(), "HostAddressPatchingInformation is not initialized");

    return m_sectionIdDbSize;
}

void HostAddressPatchingInformation::markSectionTypeForPatching(uint64_t sectionTypeId)
{
    m_sectionTypesToPatch.insert(sectionTypeId);
}

bool HostAddressPatchingInformation::markConstZeroSizeSection(uint64_t sectionId)
{
    HB_ASSERT((sectionId < m_sectionIdDbSize), "sectionId {} was not found", sectionId);
    uint32_t& sectionCurrentPatchingIndex = m_sectionPatchingIndexDb[sectionId];

    // section already resolved already by tensorInfo
    if (sectionCurrentPatchingIndex == m_currentPatchingIndex)
    {
        return true;
    }

    if (m_shouldLog)
    {
        LOG_TRACE(SYN_PATCH_INFO,
                  "{}: sectionId {} is zero sized, and not needed to be patched",
                  HLLOG_FUNC,
                  sectionId);
    }

    sectionCurrentPatchingIndex = m_currentPatchingIndex;
    m_numberOfSectionsChecked++;
    return true;
}

bool HostAddressPatchingInformation::setSectionHostAddress(uint64_t sectionId,
                                                           uint64_t sectionType,
                                                           uint64_t hostAddress,
                                                           bool     isZeroSizeSection,
                                                           bool     forceUpdate,
                                                           bool     isScratchpadSection)
{
    HB_ASSERT((sectionId < m_sectionIdDbSize), "sectionId {} was not found", sectionId);
    uint64_t& sectionCurrentAddress       = m_sectionToHostAddressDb[sectionId];
    uint32_t& sectionCurrentPatchingIndex = m_sectionPatchingIndexDb[sectionId];

    if (sectionCurrentPatchingIndex == m_currentPatchingIndex)
    {
        if (sectionCurrentAddress == hostAddress)
        {
            if (m_shouldLog)
            {
                LOG_TRACE(SYN_PATCHING,
                          "host address was already set for section {} patching index {}",
                          sectionId,
                          m_currentPatchingIndex);
            }
            return true;
        }
        else
        {
            if (hostAddress == 0)
            {
                return true;
            }

            if (sectionCurrentAddress == 0)
            {
                // try to resolve the zero section address
                // if have another tensor in the section which the address is not 0
                if (m_shouldLog)
                {
                    LOG_TRACE(SYN_PATCH_INFO,
                              "{}: {} address 0x{:x} set for section-id {} section-type {}",
                              HLLOG_FUNC,
                              (sectionId <= MEMORY_ID_RESERVED_FOR_PROGRAM) ? "Workspace" : "Section",
                              hostAddress,
                              sectionId,
                              sectionType);
                }

                LOG_TRACE(SYN_API,
                          "Updated sectionId {} hostAddress {:#x} (originaly zero-address)",
                          sectionId,
                          hostAddress);
                sectionCurrentAddress = hostAddress;
                m_numberOfSectionsChecked++;

                return true;
            }
            else
            {
                LOG_ERR(SYN_PATCHING,
                        "host address for section {} with patching index {} has an adrress conflict: 0x{:x} vs 0x{:x}",
                        sectionId,
                        m_currentPatchingIndex,
                        sectionCurrentAddress,
                        hostAddress);
                return false;
            }
        }
    }

    bool shouldUpdateSection                     = (sectionCurrentAddress != hostAddress) || (forceUpdate);
    bool isForcefullyUpdatedNonScratchpadSection = false;
    if (isScratchpadSection)
    {
        if (m_enforceNewScatchpadSectionAddr)
        {
            shouldUpdateSection = true;
        }
    }
    else
    {
        if (m_dbgNumOfSectionsGroupForecfullyUpdated < m_enforceNumOfAdditionalNewPtsSectionAddr)
        {
            shouldUpdateSection                     = true;
            isForcefullyUpdatedNonScratchpadSection = true;
        }
    }

    if (shouldUpdateSection)
    {
        // section has changed, update DB to reflect patching is required
        m_numberOfSectionsUpdated++;
        auto insertionStatus = m_sectionTypesToPatch.insert(sectionType);
        if (isForcefullyUpdatedNonScratchpadSection && insertionStatus.second)
        {
            m_dbgNumOfSectionsGroupForecfullyUpdated++;
        }

        if (m_shouldLog)
        {
            LOG_TRACE(SYN_PATCH_INFO,
                      "{}: {} address 0x{:x} set for section-id {} section-type {}",
                      HLLOG_FUNC,
                      (sectionId <= MEMORY_ID_RESERVED_FOR_PROGRAM) ? "Workspace" : "Section",
                      hostAddress,
                      sectionId,
                      sectionType);
        }

        sectionCurrentAddress = hostAddress;
    }

    if ((hostAddress != 0) || isZeroSizeSection)
    {
        LOG_TRACE(SYN_API, "Updated sectionId {} hostAddress {:#x}", sectionId, hostAddress);
        sectionCurrentPatchingIndex = m_currentPatchingIndex;
        m_numberOfSectionsChecked++;
    }

    return true;
}

void HostAddressPatchingInformation::patchingCompletion()
{
    m_currentPatchingIndex++;
    m_numberOfSectionsUpdated = 0;
    m_numberOfSectionsChecked = 0;

    m_dbgNumOfSectionsGroupForecfullyUpdated = 0;

    m_dbgFirstPatching = false;

    m_sectionTypesToPatch.clear();
}

void HostAddressPatchingInformation::patchingAbort()
{
    std::memset(m_sectionToHostAddressDb, 0, m_sectionIdDbSize * sizeof(uint64_t));
    std::memset(m_sectionPatchingIndexDb, 0, m_sectionIdDbSize * sizeof(uint32_t));

    m_currentPatchingIndex    = 1;
    m_numberOfSectionsUpdated = 0;
    m_numberOfSectionsChecked = 0;

    m_dbgNumOfSectionsGroupForecfullyUpdated = 0;

    m_sectionTypesToPatch.clear();
}

SectionTypesToPatch& HostAddressPatchingInformation::getSectionTypesQueuedForPatching()
{
    return m_sectionTypesToPatch;
}

void HostAddressPatchingInformation::dump() const
{
    for (auto sectionTypesToPatch : m_sectionTypesToPatch)
    {
        LOG_DEBUG(SYN_PATCHING, "sectionTypesToPatch {}", sectionTypesToPatch);
    }
}