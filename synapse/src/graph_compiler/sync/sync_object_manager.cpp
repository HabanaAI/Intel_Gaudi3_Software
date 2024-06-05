#include "sync_object_manager.h"

#include "defs.h"

#include "types_exception.h"
#include "types.h"

SyncObjectManager::SyncObjectManager(unsigned numSyncObjects, unsigned numMonitors, SyncConventions& syncConventions)
: m_numSyncObjects(numSyncObjects), m_numMonitorObjects(numMonitors), m_syncConventions(syncConventions)
{
    unsigned lastSyncObjId = m_numSyncObjects + m_syncConventions.getSyncObjMinId();

    for (unsigned id = m_syncConventions.getSyncObjMinId(); id < lastSyncObjId; id += m_syncConventions.getGroupSize())
    {
        bool reserved = false;
        for (unsigned j = 0 ; j < m_syncConventions.getGroupSize() ; ++j)
        {
            if (isReservedSyncId(id + j))
            {
                reserved = true;
            }
        }
        if (reserved)
        {
            for (unsigned j = 0 ; j < m_syncConventions.getGroupSize() ; ++j)
            {
                if (!isReservedSyncId(id + j))
                {
                    m_syncObjectIDs.push_back(id + j);
                }
            }
        }
        else
        {
            m_groupSyncObjectIDs.push_back(id / m_syncConventions.getGroupSize());
        }
    }

    unsigned lastMonitorObjId = m_numMonitorObjects + m_syncConventions.getMonObjMinId();

    for (unsigned id = m_syncConventions.getMonObjMinId(); id < lastMonitorObjId; ++id)
    {
        if (!isReservedMonitorId(id))
        {
            m_monitorObjectIDs.push_back(id);
        }
    }
}

SyncObjectManager::SyncId SyncObjectManager::getFreeSyncId()
{
    if (m_syncObjectIDs.empty())
    {
        // Check if can use a group id
        if (m_groupSyncObjectIDs.empty())
        {
            LOG_ERR(SYNC_SCHEME, "No free sync objects to allocate, try reduce MAX_CHUNKS_PER_DMA_TRANSFER compilation attribute");
            throw SyncValueOverflow();
        }
        else
        {
            // Break up a group id
            SyncObjectManager::SyncId groupId = m_groupSyncObjectIDs.front();
            m_groupSyncObjectIDs.pop_front();
            for (unsigned i = 0 ; i < m_syncConventions.getGroupSize() ; ++i)
            {
                m_syncObjectIDs.push_back(groupId * m_syncConventions.getGroupSize() + i);
            }
        }
    }

    SyncObjectManager::SyncId ret = m_syncObjectIDs.front();
    m_syncObjectIDs.pop_front();
    return ret;
}

SyncObjectManager::SyncId SyncObjectManager::getFreeGroupId()
{
    // Check if can use a group id
    if (m_groupSyncObjectIDs.empty())
    {
        LOG_ERR(SYNC_SCHEME, "No free group sync objects to allocate");
        return -1;
    }

    SyncObjectManager::SyncId ret = m_groupSyncObjectIDs.front();

    m_groupSyncObjectIDs.pop_front();
    return ret;
}

SyncObjectManager::MonitorId SyncObjectManager::getFreeMonitorId()
{
    if (m_monitorObjectIDs.empty())
    {
        LOG_ERR(SYNC_SCHEME, "No free monitors objects to allocate");
        return -1;
    }

    MonitorId ret = m_monitorObjectIDs.front();
    m_monitorObjectIDs.pop_front();
    return ret;
}

void SyncObjectManager::releaseSyncObject(SyncObjectManager::SyncId id)
{
    if (! isReservedSyncId(id))
    {
        HB_ASSERT(m_syncObjectIDs.size() < m_numSyncObjects, "Sync objects underrun");
        m_syncObjectIDs.push_back(id);
    }
}

void SyncObjectManager::releaseMonObject(MonitorId monId)
{
    if (! isReservedMonitorId(monId))
    {
        HB_ASSERT(m_monitorObjectIDs.size() < m_numMonitorObjects, "Monitor objects underrun");
        m_monitorObjectIDs.push_back(monId);
    }
}

bool SyncObjectManager::areAllSyncObjectsAreFree() const
{
    return (m_syncObjectIDs.size() + m_syncConventions.getGroupSize() * m_groupSyncObjectIDs.size()) ==
           (m_numSyncObjects - numReservedSyncObjects());
}

bool SyncObjectManager::isReservedSyncId(SyncId syncId) const
{
    return (syncId >= m_syncConventions.getSyncObjMinSavedId() && syncId < m_syncConventions.getSyncObjMaxSavedId());
}

unsigned int SyncObjectManager::numReservedSyncObjects() const
{
    return m_syncConventions.getSyncObjMaxSavedId() - m_syncConventions.getSyncObjMinSavedId();
}

bool SyncObjectManager::isReservedMonitorId(MonitorId monId) const
{
    return (monId >= m_syncConventions.getMonObjMinSavedId() && monId < m_syncConventions.getMonObjMaxSavedId());
}

// This is due to [H3-2116] and currently only relevant for gaudi.
SyncObjectManager::SyncId SyncObjectManager::getDummySyncId()
{
    if(!m_dummySyncId)
    {
        m_dummySyncId = std::make_shared<SyncId>(getFreeSyncId());
    }
    return *m_dummySyncId;
}
