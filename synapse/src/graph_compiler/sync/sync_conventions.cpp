#include "sync_conventions.h"
#include "infra/defs.h"

unsigned SyncConventions::getLowerQueueID(HabanaDeviceType deviceType, unsigned engineId) const
{
    HB_ASSERT(false, "was not implemented");
    return 0;
}
unsigned SyncConventions::getSignalOutGroup() const
{
    HB_ASSERT(false, "Unimplemented");
    return 0;
};

bool SyncConventions::isSignalOutGroupSupported() const
{
    return false;
}