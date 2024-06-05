#pragma once

#include "address_range_mapper.hpp"

#include <unordered_map>
#include <vector>

class InflightCsParserHelper
{
public:
    // As parsing may require different structure of a "CS",
    // and may include breaking of the "Recipe_CS" into several "Stage-CS",
    // we would like that the parser will get ordered content per engine
    struct ParserCsEntry
    {
        uint64_t m_bufferAddress;
        uint64_t m_bufferSize;
    };

    enum eParsingType
    {
        PARSING_TYPE_REGULAR_ENQUEUE,
        PARSING_TYPE_ACTIVATE,
        PARSING_TYPE_ACTIVATE_THEN_ENQUEUE,
    };

    typedef std::vector<ParserCsEntry>                            ParserHwStreamCsEntries;
    typedef std::unordered_map<uint32_t, ParserHwStreamCsEntries> ParserCsPqEntries;

    // We can pass num-of-engines to reserve the DB size, but as this is not a critical-performance module...
    InflightCsParserHelper(AddressRangeMapper* pAddressRangeMap, uint64_t streamPhysicalOffset, bool isActivatePreNQ);
    virtual ~InflightCsParserHelper() = default;

    const ParserCsPqEntries &getParserCsPqEntries();

    void addParserPqEntry(uint32_t hwStreamId, uint64_t bufferAddress, uint64_t bufferSize);

    uint64_t getHostAddress(uint64_t& bufferSize, uint64_t handle);

    uint64_t getStreamPhysicalOffset() { return m_queuePhysicalOffset; };

    eParsingType getParsingType() { return m_parsingType; }

private:
    // HwStreamId to ParserCsEntries Data base
    ParserCsPqEntries m_pqEntriesPerStreamIdDb;

    AddressRangeMapper* m_pAddressRangeMap;
    uint64_t            m_queuePhysicalOffset;

    eParsingType m_parsingType;
};

class InflightCsParser
{
public:
    InflightCsParser() {};
    virtual ~InflightCsParser() = default;

    virtual bool parse(InflightCsParserHelper& parserHelper) = 0;
};