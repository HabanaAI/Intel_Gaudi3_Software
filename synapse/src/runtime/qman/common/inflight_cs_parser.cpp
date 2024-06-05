#include "inflight_cs_parser.hpp"

using ParserCsPqEntries = InflightCsParserHelper::ParserCsPqEntries;

InflightCsParserHelper::InflightCsParserHelper(AddressRangeMapper* pAddressRangeMap,
                                               uint64_t            streamPhysicalOffset,
                                               bool                isActivatePreNQ)
: m_pAddressRangeMap(pAddressRangeMap),
  m_queuePhysicalOffset(streamPhysicalOffset),
  m_parsingType(isActivatePreNQ ? PARSING_TYPE_ACTIVATE : PARSING_TYPE_REGULAR_ENQUEUE)
{
}

const ParserCsPqEntries & InflightCsParserHelper::getParserCsPqEntries()
{
    return m_pqEntriesPerStreamIdDb;
}

void InflightCsParserHelper::addParserPqEntry(uint32_t hwStreamId, uint64_t bufferAddress, uint64_t bufferSize)
{
    m_pqEntriesPerStreamIdDb[hwStreamId].push_back({.m_bufferAddress = bufferAddress, .m_bufferSize = bufferSize});
}

uint64_t InflightCsParserHelper::getHostAddress(uint64_t& bufferSize, uint64_t handle)
{
    return m_pAddressRangeMap->getMappedAddress(bufferSize, handle);
}