#ifndef _SYNC_OBJECT_MANAGER_H_
#define _SYNC_OBJECT_MANAGER_H_

#include <deque>
#include <memory>
#include "sync_conventions.h"

//A class for managing SRAM. Needs to be considerably more complex and interact with SramBlobs
//For now a simple slab allocator
class SyncObjectManager
{
public:
    //typedef this because we may change it in the future to be a pair of ID and set
    typedef unsigned SyncId;
    typedef unsigned MonitorId;

    SyncObjectManager(unsigned numSyncObjects, unsigned numMonitors, SyncConventions& syncConventions);

    //Acquire an available object.
    SyncId getFreeSyncId();
    SyncId getFreeGroupId();

    MonitorId getFreeMonitorId();

    //Release a previously-acquired object
    void releaseSyncObject(SyncId id);
    void releaseMonObject(MonitorId monId);  // currently unused

    bool areAllSyncObjectsAreFree() const;

    bool isReservedSyncId(SyncId syncId) const;
    SyncId getDummySyncId();

    const SyncConventions& getSyncConventions() const { return m_syncConventions; }

private:

    unsigned int numReservedSyncObjects() const;
    bool isReservedMonitorId(MonitorId monId) const;

    std::deque<SyncId> m_syncObjectIDs;
    std::deque<SyncId> m_groupSyncObjectIDs;
    std::deque<MonitorId> m_monitorObjectIDs;

    const unsigned m_numSyncObjects;
    const unsigned m_numMonitorObjects;

    SyncConventions& m_syncConventions;
    std::shared_ptr<SyncId> m_dummySyncId;
};
#endif
