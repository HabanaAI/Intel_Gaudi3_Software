#include "address_range_mapper.hpp"

#include "defs.h"
#include "synapse_runtime_logging.h"

#include <sstream>

bool AddressRangeMapper::addMapping(uint64_t                 handle,
                                    uint64_t                 mappedAddress,
                                    uint64_t                 size,
                                    eAddressRangeMappingType mappingType)
{
    AddressRangeEntriesDbIterator entryIter;

    eAddressRangeMappingStatus mappingStatus = _findEntry(handle, size, entryIter);

    if (mappingStatus == ARMS_MAPPING_STATUS_FAILURE)
    {
        return false;
    }

    if (mappingStatus == ARMS_MAPPING_STATUS_FOUND)
    {  // In case found, ensure that it is the exact entry
        if ((entryIter->first != handle) || (entryIter->second.mappedAddress != mappedAddress) ||
            (entryIter->second.size != size) || (entryIter->second.mappingType != mappingType))
        {
            LOG_ERR(SYN_MEM_MAP,
                    "{}: Mismatch between mapping-request {} and entry {}",
                    HLLOG_FUNC,
                    _getMappingDescription(handle, mappedAddress, size, mappingType),
                    _getMappingDescription(entryIter->first,
                                           entryIter->second.mappedAddress,
                                           entryIter->second.size,
                                           entryIter->second.mappingType));

            return false;
        }
    }
    else if (mappingStatus == ARMS_MAPPING_STATUS_NOT_FOUND)
    {
        m_mappingDB[handle] = {.mappedAddress = mappedAddress, .size = size, .mappingType = mappingType};

        LOG_TRACE(SYN_MEM_MAP,
                  "Mapping handle 0x{:x} to address-range 0x{:x} (size {} mappingType {})",
                  handle,
                  mappedAddress,
                  size,
                  mappingType);

        return true;
    }

    return true;
}

bool AddressRangeMapper::removeMapping(uint64_t handle)
{
    AddressRangeEntriesDbIterator entryIter;

    // The handle must be exact (size is irrelevant)
    const uint64_t anySize = 1;

    eAddressRangeMappingStatus mappingStatus = _findEntry(handle, anySize, entryIter);

    if (mappingStatus == ARMS_MAPPING_STATUS_FAILURE)
    {
        return false;
    }

    if (mappingStatus == ARMS_MAPPING_STATUS_NOT_FOUND)
    {
        LOG_ERR(SYN_MEM_MAP, "{}: Handle 0x{:x} not found", HLLOG_FUNC, handle);
        return false;
    }

    if (handle != entryIter->first)
    {
        LOG_ERR(SYN_MEM_MAP,
                "{}: Handle 0x{:x} is internal to entry [{}, {})",
                HLLOG_FUNC,
                handle,
                entryIter->first,
                entryIter->first + entryIter->second.size - 1);

        return false;
    }

    LOG_TRACE(SYN_MEM_MAP,
              "{}: Removing entry of handle 0x{:x}: mapped base-address 0x{:x} size {} {}",
              HLLOG_FUNC,
              handle,
              entryIter->second.mappedAddress,
              entryIter->second.size,
              _getMappingType(entryIter->second.mappingType));

    m_mappingDB.erase(entryIter);
    return true;
}

uint64_t AddressRangeMapper::getMappedAddress(uint64_t& size, uint64_t handle)
{
    AddressRangeEntriesDbIterator entryIter;

    uint64_t mappedAddress = 0;

    // Find the entry related to the handle (size is irrelevant)
    const uint64_t anySize = 1;

    eAddressRangeMappingStatus mappingStatus = _findEntry(handle, anySize, entryIter);

    if (mappingStatus == ARMS_MAPPING_STATUS_FAILURE)
    {
        return 0;
    }

    if (mappingStatus == ARMS_MAPPING_STATUS_NOT_FOUND)
    {
        LOG_ERR(SYN_MEM_MAP, "{}: Handle 0x{:x} not found", HLLOG_FUNC, handle);
        return 0;
    }

    uint64_t handleOffset = (handle - entryIter->first);

    size          = entryIter->second.size - handleOffset;
    mappedAddress = entryIter->second.mappedAddress + handleOffset;

    return mappedAddress;
}

bool AddressRangeMapper::updateMappingsOf(AddressRangeMapper& otherMapper) const
{
    for (auto currMapping : m_mappingDB)
    {
        if (!otherMapper.addMapping(currMapping.first,
                                    currMapping.second.mappedAddress,
                                    currMapping.second.size,
                                    currMapping.second.mappingType))
        {
            return false;
        }
    }

    return true;
}

void AddressRangeMapper::clear()
{
    m_mappingDB.clear();
}

AddressRangeMapper::eAddressRangeMappingStatus
AddressRangeMapper::_findEntry(uint64_t handle, uint64_t bufferSize, AddressRangeEntriesDbIterator& entryIter)
{
    if (m_mappingDB.size() == 0)
    {
        return ARMS_MAPPING_STATUS_NOT_FOUND;
    }

    // mapIter - the first entry that its key is >= mappedAddress
    entryIter = m_mappingDB.lower_bound(handle);

    // Lower-bound returns the entry that its key is >= (greater or equal) the searched key
    //
    // 1) In case entry-found is not "end()", verify is current entry match (key equals base value)
    //
    // else
    //
    // 2) In case entry-found is not "begin()", then the last entry is the suspected entry
    if ((entryIter != m_mappingDB.end()) && (entryIter->first == handle))
    {
        return ARMS_MAPPING_STATUS_FOUND;
    }
    //
    if (entryIter != m_mappingDB.begin())
    {
        entryIter--;
    }

    return _isValidMapping(entryIter, handle, bufferSize);
}

AddressRangeMapper::eAddressRangeMappingStatus
AddressRangeMapper::_isValidMapping(const AddressRangeEntriesDbIterator& entryIter,
                                    uint64_t                             handle,
                                    uint64_t                             bufferSize)
{
    uint64_t entryHandleBase = entryIter->first;
    uint64_t entryBufferSize = entryIter->second.size;

    uint64_t entryHandleLast = entryHandleBase + entryBufferSize - 1;
    uint64_t handleLast      = handle + bufferSize - 1;

    LOG_TRACE(SYN_MEM_MAP,
              "{}: handle - new [0x{:x} 0x{:x}] entry [0x{:x} 0x{:x}] ({})",
              HLLOG_FUNC,
              handle,
              handleLast,
              entryHandleBase,
              entryHandleLast,
              _getMappingType(entryIter->second.mappingType));

    // The handle is Out-Of-Entry's-Bounds
    if ((handle > entryHandleLast) || (handle < entryHandleBase))
    {
        return ARMS_MAPPING_STATUS_NOT_FOUND;
    }

    // In case the entry has tag-mapping, the handle must match the entry's one
    if ((entryIter->second.mappingType == ARM_MAPPING_TYPE_TAG) && (entryHandleBase != handle))
    {
        LOG_ERR(SYN_MEM_MAP,
                "{}: Illegal tag-mapping for handle - new 0x{:x} entry 0x{:x}",
                HLLOG_FUNC,
                handle,
                entryHandleBase);

        return ARMS_MAPPING_STATUS_FAILURE;
    }

    // The Entry's Address-Range (AR) and the handle AR does not match
    if (handleLast > entryHandleLast)
    {
        LOG_ERR(SYN_MEM_MAP,
                "{}: Illegal range-mapping for handle - new [0x{:x} 0x{:x}] entry [0x{:x} 0x{:x}]",
                HLLOG_FUNC,
                handle,
                handleLast,
                entryHandleBase,
                entryHandleLast);

        return ARMS_MAPPING_STATUS_FAILURE;
    }

    return ARMS_MAPPING_STATUS_FOUND;
}

std::string AddressRangeMapper::_getMappingType(eAddressRangeMappingType mappingType)
{
    switch (mappingType)
    {
        case ARM_MAPPING_TYPE_RANGE:
            return "range";

        case ARM_MAPPING_TYPE_TAG:
            return "tag";
    }

    HB_ASSERT(false, "Not expected to reach here");
    return "";
}

std::string AddressRangeMapper::_getMappingDescription(uint64_t                 handle,
                                                       uint64_t                 mappedAddress,
                                                       uint64_t                 size,
                                                       eAddressRangeMappingType mappingType)
{
    std::stringstream description;

    description << "[Handle 0x" << std::hex << handle << "Mapped 0x" << std::hex << mappedAddress << "Size "
                << std::fixed << size << _getMappingType(mappingType);

    return description.str();
}