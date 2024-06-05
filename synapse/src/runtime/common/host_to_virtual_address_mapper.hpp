/*
 *
 */
#pragma once

#include <map>
#include <mutex>
#include <cstdint>
#include <string>

struct VirtualMappingData
{
    uint64_t    deviceVA;
    uint64_t    bufferSize;
    std::string name;
    bool        isUserRequest;
};

typedef std::map<void*, VirtualMappingData> HostAddrToVirtualAddrMap;
typedef HostAddrToVirtualAddrMap::const_iterator HostAddrToVirtualAddrIterator;

enum eHostAddrToVirtualAddrMappingStatus
{
    HATVA_MAPPING_STATUS_FOUND,         // An entry contains requested buffer & size
    HATVA_MAPPING_STATUS_NOT_FOUND,     // No entry containing requested buffer & size had been found
    HATVA_MAPPING_STATUS_INVALID_SIZE,  // Part of requested buffer contained in an entry, but size mismatch
    HATVA_MAPPING_STATUS_FAILURE        // Any other (non-mapping) failure
};

typedef eHostAddrToVirtualAddrMappingStatus eMappingStatus;

class HostAddrToVirtualAddrMapper
{
public:
    HostAddrToVirtualAddrMapper(const std::string& rName) : m_name(rName) {};

    virtual ~HostAddrToVirtualAddrMapper() = default;

    eHostAddrToVirtualAddrMappingStatus getDeviceVirtualAddress(void*     hostAddress,
                                                                uint64_t  bufferSize,
                                                                uint64_t* pDeviceVA,
                                                                bool*     pIsExactKeyFound = nullptr) const;

    bool setMapping(void*              hostAddress,
                    uint64_t           deviceVA,
                    uint64_t           bufferSize,
                    bool               isUserRequest,
                    const std::string& mappingDesc);

    bool clearMapping(void* hostAddress);

    void dfaLogMapper() const;

    uint64_t                      size();
    HostAddrToVirtualAddrIterator begin();
    HostAddrToVirtualAddrIterator end();

private:
    eHostAddrToVirtualAddrMappingStatus
    _findEntry(void* hostAddress, uint64_t bufferSize, HostAddrToVirtualAddrIterator& mapIter) const;

    eHostAddrToVirtualAddrMappingStatus
    _validateContainedInMapEntry(HostAddrToVirtualAddrIterator mapIter, void* hostAddress, uint64_t bufferSize) const;

    const std::string m_name;

    mutable std::mutex m_mutex;

    HostAddrToVirtualAddrMap m_addressMapper;
};
