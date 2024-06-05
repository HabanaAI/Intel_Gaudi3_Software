#include "qman_cp_info.hpp"

#include "define.hpp"
#include "fence_state_machine.hpp"
#include "sync_manager_info.hpp"

#include "defenders.h"
#include "synapse_runtime_logging.h"

#include <sstream>

using namespace common;

const std::unordered_map<QmanCpInfo::eMonitorSetupPhase, std::string> QmanCpInfo::m_monitorSetupPhaseDesc = {
    {MONITOR_SETUP_PHASE_NOT_SET, "Not set"},
    {MONITOR_SETUP_PHASE_LOW, "low-address"},
    {MONITOR_SETUP_PHASE_HIGH, "high-address"},
    {MONITOR_SETUP_PHASE_DATA, "value (payload-data)"}};

bool QmanCpInfo::setNewInfo(uint64_t                 handle,
                            uint64_t                 hostAddress,
                            uint64_t                 bufferSize,
                            uint32_t                 cpIndex,
                            uint16_t                 predValue,
                            SyncManagerInfoDatabase* pSyncManagerInfoDb)
{
    CHECK_POINTER(SYN_CS_PARSER, pSyncManagerInfoDb, "SYNC-Manager-Info", false);

    m_handle             = handle;
    m_hostAddress        = hostAddress;
    m_bufferSize         = bufferSize;
    m_cpIndex            = cpIndex;
    m_predValue          = predValue;
    m_pSyncManagerInfoDb = pSyncManagerInfoDb;

    uint32_t fenceIndex = m_cpIndex * getFenceIdsPerCp();
    for (unsigned i = 0; i < getFenceIdsPerCp(); i++, fenceIndex++)
    {
        m_cpFenceSm[fenceIndex].init(fenceIndex);
    }

    m_isInitialized = true;

    return true;
}

void QmanCpInfo::reset(bool shouldResetSyncManager, uint64_t cpDmaPacketIndex)
{
    m_packetIndex = 0;
    if (shouldResetSyncManager)
    {
        m_monitorSetupInfo.clear();
    }

    m_isInitialized = false;
}

bool QmanCpInfo::finalize()
{
    for (auto singleFenceSm : m_cpFenceSm)
    {
        if (!singleFenceSm.second.finalize())
        {
            return false;
        }
    }

    return true;
}

void QmanCpInfo::printStartParsing(std::string_view qmanDescription)
{
    LOG_GCP_VERBOSE("{}: handle 0x{:x} hostAddress 0x{:x} size 0x{:x}",
                    qmanDescription,
                    m_handle,
                    m_hostAddress,
                    m_bufferSize);
}

void QmanCpInfo::updateNextBuffer(uint32_t packetSize)
{
    uint32_t sizeInBytes = packetSize * sizeof(uint32_t);
    m_hostAddress += sizeInBytes;
    m_bufferSize -= sizeInBytes;
}

bool QmanCpInfo::validatePacketSize(uint32_t packetSize)
{
    if (packetSize * sizeof(uint32_t) > m_bufferSize)
    {
        LOG_GCP_FAILURE("{}: Invalid size packetSize 0x{:x} buffer-Size left 0x{:x}",
                        HLLOG_FUNC,
                        packetSize * sizeof(uint32_t),
                        m_bufferSize);

        return false;
    }

    return true;
}

std::string QmanCpInfo::_getMonitorSetupPhaseDescription(eMonitorSetupPhase phaseId)
{
    auto descriptionIter = m_monitorSetupPhaseDesc.find(phaseId);
    if (descriptionIter == m_monitorSetupPhaseDesc.end())
    {
        return "Invalid";
    }

    return descriptionIter->second;
}