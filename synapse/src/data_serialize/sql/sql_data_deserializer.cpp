#include "sql_data_deserializer.h"
#include <memory>

namespace data_serialize
{
SqlGraphDataDeserializer::SqlGraphDataDeserializer(const std::string& filePath, const GraphInfo& graphInfo)
: m_filePath(filePath), m_sql(SqlDeserializer::createSqlDeserializer(filePath)), m_graphInfo(graphInfo)
{
}

std::vector<std::string> SqlGraphDataDeserializer::getTensorsNames() const
{
    return m_sql->getTensorsNames(m_graphInfo);
}

uint64_t SqlGraphDataDeserializer::getDataSize(const std::string& tensorName, uint64_t iteration) const
{
    return m_sql->getDataSize(m_graphInfo, tensorName, iteration);
}

std::vector<uint8_t> SqlGraphDataDeserializer::getData(const std::string& tensorName, uint64_t iteration) const
{
    return m_sql->getData(m_graphInfo, tensorName, iteration);
}

void SqlGraphDataDeserializer::getData(const std::string& tensorName,
                                       uint64_t           iteration,
                                       uint8_t*           data,
                                       TSize              dataSize) const
{
    m_sql->getData(m_graphInfo, tensorName, iteration, data, dataSize);
}

std::vector<TSize> SqlGraphDataDeserializer::getShape(const std::string& tensorName, uint64_t iteration) const
{
    return m_sql->getShape(m_graphInfo, tensorName, iteration);
}

std::vector<uint8_t> SqlGraphDataDeserializer::getPermutation(const std::string& tensorName, uint64_t iteration) const
{
    return m_sql->getPermutation(m_graphInfo, tensorName, iteration);
}

synDataType SqlGraphDataDeserializer::getDataType(const std::string& tensorName, uint64_t iteration) const
{
    return m_sql->getDataType(m_graphInfo, tensorName, iteration);
}

std::set<uint64_t> SqlGraphDataDeserializer::getDataIterations() const
{
    return m_sql->getDataIterations(m_graphInfo);
}

std::set<uint64_t> SqlGraphDataDeserializer::getNonDataIterations() const
{
    return m_sql->getNonDataIterations(m_graphInfo);
}

SqlDataDeserializer::SqlDataDeserializer(const std::string& filePath)
: m_filePath(filePath), m_sql(SqlDeserializer::createSqlDeserializer(filePath))
{
}

std::vector<uint64_t> SqlDataDeserializer::getGroups() const
{
    return m_sql->getGroups();
}

std::vector<GraphInfo> SqlDataDeserializer::getGraphsInfos() const
{
    return m_sql->getGraphsInfos();
}

std::vector<GraphInfo> SqlDataDeserializer::getGraphsInfos(uint64_t group) const
{
    return m_sql->getGraphsInfos(group);
}

GraphDataDeserializerPtr SqlDataDeserializer::getGraph(const GraphInfo& graphInfo) const
{
    return std::make_shared<SqlGraphDataDeserializer>(m_filePath, graphInfo);
}
}  // namespace data_serialize
