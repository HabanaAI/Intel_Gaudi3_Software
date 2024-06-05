#include <synapse_api_types.h>
#include "infra/containers/slot_map_alloc.hpp"
#include "section_handle.hpp"

// When running GraphCompiler_tests, we don't have syn_singleton, so we don't have the mapping between
// section handle and section pointer. Adding it here to support the tests

ConcurrentSlotMapAlloc<InternalSectionHandle> sectionHndlSlopMap;

SlotMapItemSptr<InternalSectionHandle> getSectionPtrFromHandle(synSectionHandle handle)
{
    return sectionHndlSlopMap[(SMHandle)handle];
}
