#pragma once

#include <map>
#include <cstdint>
#include <string>

// This class is NOT thread-safe
class AddressRangeMapper
{
public:
    // Defining an enum, for possible future enhancment
    enum eAddressRangeMappingType
    {
        ARM_MAPPING_TYPE_RANGE,  // Mapping between one range to another (address-plane)
        ARM_MAPPING_TYPE_TAG     // Mapping between a single handle and an address range
    };

    AddressRangeMapper() {};

    ~AddressRangeMapper() {};

    bool addMapping(uint64_t handle, uint64_t mappedAddress, uint64_t size, eAddressRangeMappingType mappingType);

    bool removeMapping(uint64_t handle);

    // Will return zero in case handle was not found
    uint64_t getMappedAddress(uint64_t& size, uint64_t handle);

    // Updates the otherMapper with current's mappings
    bool updateMappingsOf(AddressRangeMapper& otherMapper) const;

    void clear();

private:
    enum eAddressRangeMappingStatus
    {
        ARMS_MAPPING_STATUS_FOUND,      // An entry contains requested buffer & size
        ARMS_MAPPING_STATUS_NOT_FOUND,  // Entry was not found
        ARMS_MAPPING_STATUS_FAILURE     // Other
    };

    struct AddressRangEntry
    {
        uint64_t                 mappedAddress;
        uint64_t                 size;
        eAddressRangeMappingType mappingType;
    };

    using AddressRangeEntriesDB         = std::map<uint64_t, AddressRangEntry>;
    using AddressRangeEntriesDbIterator = AddressRangeEntriesDB::iterator;

    // Finds an entry that contains [handle, handle + bufferSize) mapping
    // In case that entry is TAG-mapping, then the entry's key be that handle
    eAddressRangeMappingStatus
    _findEntry(uint64_t handle, uint64_t bufferSize, AddressRangeEntriesDbIterator& entryIter);

    eAddressRangeMappingStatus
    _isValidMapping(const AddressRangeEntriesDbIterator& entryIter, uint64_t handle, uint64_t bufferSize);

    static std::string _getMappingType(eAddressRangeMappingType mappingType);

    static std::string _getMappingDescription(uint64_t                 handle,
                                              uint64_t                 mappedAddress,
                                              uint64_t                 size,
                                              eAddressRangeMappingType mappingType);

    AddressRangeEntriesDB m_mappingDB;
};
