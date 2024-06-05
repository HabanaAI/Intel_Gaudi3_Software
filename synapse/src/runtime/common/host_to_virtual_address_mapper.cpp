#include "host_to_virtual_address_mapper.hpp"

#include "syn_logging.h"

#include "utils.h"

#include "synapse_runtime_logging.h"

#define VERIFY_IS_NULL_POINTER_RET(logname, pointer, name, retVal)                                                     \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_ERR(logname, "{}: Got null pointer for {} ", HLLOG_FUNC, name);                                            \
        return retVal;                                                                                                 \
    }

typedef eHostAddrToVirtualAddrMappingStatus eMappingStatus;

eMappingStatus HostAddrToVirtualAddrMapper::getDeviceVirtualAddress(void*     hostAddress,
                                                                    uint64_t  bufferSize,
                                                                    uint64_t* pDeviceVA,
                                                                    bool*     pIsExactKeyFound) const
{
    VERIFY_IS_NULL_POINTER_RET(SYN_MEM_MAP, pDeviceVA, "device-VA", HATVA_MAPPING_STATUS_FAILURE);

    bool      isExactKeyFound  = true;
    void*     hostAddressFound = nullptr;
    uint64_t  deviceVAFound    = 0;

    HostAddrToVirtualAddrIterator mapIter;

    std::unique_lock<std::mutex> lck(m_mutex);

    eMappingStatus mappingStatus = _findEntry(hostAddress, bufferSize, mapIter);
    if (mappingStatus == HATVA_MAPPING_STATUS_FOUND)
    {
        hostAddressFound = mapIter->first;
        deviceVAFound    = mapIter->second.deviceVA;

        *pDeviceVA = deviceVAFound + ((uint64_t)hostAddress - (uint64_t)hostAddressFound);

        isExactKeyFound = (hostAddressFound == hostAddress);

        LOG_TRACE(SYN_MEM_MAP,
                  "{} {}: Found VA {} for host-address {} is-exact-key {} mappingDesc {}",
                  HLLOG_FUNC,
                  m_name,
                  *pDeviceVA,
                  hostAddress,
                  isExactKeyFound,
                  mapIter->second.name);

        if (pIsExactKeyFound != nullptr)
        {
            *pIsExactKeyFound = isExactKeyFound;
        }
    }
    else
    {
        // Might be due to validation whether buffer is mapped -> debug level
        LOG_DEBUG(SYN_MEM_MAP, "{} {}: Did not find device VA for host-address {}", HLLOG_FUNC, m_name, hostAddress);
    }

    return mappingStatus;
}

bool HostAddrToVirtualAddrMapper::setMapping(void*              hostAddress,
                                             uint64_t           deviceVA,
                                             uint64_t           bufferSize,
                                             bool               isUserRequest,
                                             const std::string& mappingDesc)
{
    bool status = true;

    std::unique_lock<std::mutex> lck(m_mutex);

    HostAddrToVirtualAddrIterator mapIter;

    eMappingStatus mappingStatus = _findEntry(hostAddress, bufferSize, mapIter);
    if (mappingStatus == HATVA_MAPPING_STATUS_NOT_FOUND)
    {
        LOG_DEBUG(SYN_MEM_MAP,
                  "{} {}: Added mapping ({}) between host-address {} and device VA {:x} (bufferSize 0x{:x})",
                  HLLOG_FUNC,
                  m_name,
                  mappingDesc,
                  hostAddress,
                  deviceVA,
                  bufferSize);
        m_addressMapper[hostAddress].deviceVA      = deviceVA;
        m_addressMapper[hostAddress].bufferSize    = bufferSize;
        m_addressMapper[hostAddress].name          = mappingDesc;
        m_addressMapper[hostAddress].isUserRequest = isUserRequest;
    }
    else if ((mapIter->second.deviceVA == deviceVA) && (mapIter->second.bufferSize == bufferSize))
    {
        LOG_DEBUG(SYN_MEM_MAP,
                  "{} {}: Re-mapping ({}) between host-address {} and device VA {} (bufferSize 0x{:x})",
                  HLLOG_FUNC,
                  m_name,
                  mappingDesc,
                  hostAddress,
                  deviceVA,
                  bufferSize);
        status = true;
    }
    else
    {
        LOG_ERR(SYN_MEM_MAP,
                "{} {}: Failed to set mapping ({}) between host-address {} and device VA {} (bufferSize 0x{:x})",
                HLLOG_FUNC,
                m_name,
                mappingDesc,
                hostAddress,
                deviceVA,
                bufferSize);
        status = false;
    }

    return status;
}

bool HostAddrToVirtualAddrMapper::clearMapping(void* hostAddress)
{
    std::unique_lock<std::mutex> lck(m_mutex);

    HostAddrToVirtualAddrIterator mapIter;

    // No checking size when clearing
    // The caller is responsible
    static const uint64_t anySize       = 1;
    eMappingStatus        mappingStatus = _findEntry(hostAddress, anySize, mapIter);
    if (mappingStatus == HATVA_MAPPING_STATUS_FOUND)
    {
        LOG_DEBUG(SYN_MEM_MAP,
                  "{} {}: Clearing mapping (Entry {}) between host-address 0x{:x} and device VA 0x{:x}",
                  HLLOG_FUNC,
                  m_name,
                  mapIter->second.name,
                  (uint64_t)hostAddress,
                  (uint64_t)mapIter->second.deviceVA);
        return (m_addressMapper.erase(hostAddress) != 0);
    }
    else
    {
        return false;
    }
}

uint64_t HostAddrToVirtualAddrMapper::size()
{
    return m_addressMapper.size();
}

void HostAddrToVirtualAddrMapper::dfaLogMapper() const
{
    LOG_TRACE(SYN_DEV_FAIL, "--- Mapper name: {} --- has {} Entries.", m_name, m_addressMapper.size());

    size_t cnt = 0;
    for (const auto& x : m_addressMapper)
    {
        LOG_TRACE(SYN_DEV_FAIL, "{:6}: {:16x} -> {:16x} {:8x} ({})",
                  cnt++, TO64(x.first), x.second.deviceVA, x.second.bufferSize, x.second.name);
    }
    LOG_TRACE(SYN_DEV_FAIL, "--- Done ---");
}


HostAddrToVirtualAddrIterator HostAddrToVirtualAddrMapper::begin()
{
    std::unique_lock<std::mutex> lck(m_mutex);

    return m_addressMapper.begin();
}

HostAddrToVirtualAddrIterator HostAddrToVirtualAddrMapper::end()
{
    std::unique_lock<std::mutex> lck(m_mutex);

    return m_addressMapper.end();
}

eMappingStatus HostAddrToVirtualAddrMapper::_findEntry(void*                          hostAddress,
                                                       uint64_t                       bufferSize,
                                                       HostAddrToVirtualAddrIterator& mapIter) const
{
    eMappingStatus mappingStatus = HATVA_MAPPING_STATUS_NOT_FOUND;

    if (m_addressMapper.size() != 0)
    {
        // mapIter - the first entry that its key is >= hostAddress
        mapIter = m_addressMapper.lower_bound(hostAddress);

        // Lower-bound returns the entry that its key is >= searched parameter
        //
        // 1) In case entry-found is "end()", we need to CHECK (*) whether the
        //    last (valid) entry includes the searched buffer (as there are entries,
        //    there must be one before it)
        //
        // 2) In case the entry has the exact key, just need to also "localy-check" its size
        //
        // 3) In case it is not the exact key, we want to CHECK (*) its preceding entry
        //
        // 4) In case that entry is the first one, there is no "preceding entry", and hence
        //    we can conclude that it was not found
        //
        // * CHECK -> _validateContainedInMapEntry
        if (mapIter == m_addressMapper.end())
        {
            mapIter--;
        }
        else if (mapIter->first == hostAddress)
        {
            if (bufferSize > mapIter->second.bufferSize)
            {
                LOG_ERR(SYN_MEM_MAP,
                        "{} {}: illegal size hostAddress 0x{:x} bufferSize {} bufferSizeMapped {} mappingDesc {}",
                        HLLOG_FUNC,
                        m_name,
                        (uint64_t)hostAddress,
                        bufferSize,
                        mapIter->second.bufferSize,
                        mapIter->second.name);
                return HATVA_MAPPING_STATUS_INVALID_SIZE;
            }
            else
            {
                return HATVA_MAPPING_STATUS_FOUND;
            }
        }
        else if (mapIter != m_addressMapper.begin())
        {
            mapIter--;
        }
        else
        {
            return HATVA_MAPPING_STATUS_NOT_FOUND;
        }

        return _validateContainedInMapEntry(mapIter, hostAddress, bufferSize);
    }

    return mappingStatus;
}

eMappingStatus HostAddrToVirtualAddrMapper::_validateContainedInMapEntry(HostAddrToVirtualAddrIterator mapIter,
                                                                         void*                         hostAddress,
                                                                         uint64_t                      bufferSize) const
{
    uint64_t entryHostAddress = (uint64_t)mapIter->first;
    uint64_t entryBufferSize  = mapIter->second.bufferSize;

    uint64_t entryLastHostAddress   = (uint64_t)entryHostAddress + entryBufferSize - 1;
    uint64_t requestLastHostAddress = (uint64_t)hostAddress + bufferSize - 1;

    LOG_TRACE(SYN_MEM_MAP,
              "{} {}: (Entry: {}) hostAddress 0x{:x} entryHostAddress 0x{:x} entryLastHostAddress 0x{:x} "
              "requestLastHostAddress 0x{:x} bufferSize 0x{:x}",
              HLLOG_FUNC,
              m_name,
              mapIter->second.name,
              (uint64_t)hostAddress,
              entryHostAddress,
              entryLastHostAddress,
              requestLastHostAddress,
              bufferSize);

    if (((uint64_t)hostAddress > entryLastHostAddress) || ((uint64_t)hostAddress < entryHostAddress))
    {
        return HATVA_MAPPING_STATUS_NOT_FOUND;
    }

    if (requestLastHostAddress <= entryLastHostAddress)
    {
        return HATVA_MAPPING_STATUS_FOUND;
    }
    else
    {
        LOG_ERR(SYN_MEM_MAP,
                "{} {}: (Entry: {}) illegal Address requestLastHostAddress 0x{:x} entryLastHostAddress 0x{:x} "
                "hostAddress 0x{:x} entryHostAddress 0x{:x} bufferSize 0x{:x}",
                HLLOG_FUNC,
                m_name,
                mapIter->second.name,
                requestLastHostAddress,
                entryLastHostAddress,
                (uint64_t)hostAddress,
                entryHostAddress,
                bufferSize);
        return HATVA_MAPPING_STATUS_INVALID_SIZE;
    }
}