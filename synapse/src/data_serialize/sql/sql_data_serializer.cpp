#include "sql_data_serializer.h"

namespace data_serialize
{
SqlDataSerializer::SqlDataSerializer(const std::string& filePath, const GraphInfo& graphInfo)
: m_sql(std::make_unique<SqlSerializer>(filePath)), m_graphInfo(graphInfo)
{
}

const GraphInfo& SqlDataSerializer::getInfo()
{
    return m_graphInfo;
}

uint64_t SqlDataSerializer::getIterationCount() const
{
    return m_sql->getIterationCount(m_graphInfo);
}

size_t SqlDataSerializer::serialize(TensorMetadata& tmd)
{
    return m_sql->insert(m_graphInfo, tmd);
}

void SqlDataSerializer::updateData(TensorMetadata& tmd)
{
    m_sql->updateData(m_graphInfo, tmd);
}

void SqlDataSerializer::updateRecipeId(uint16_t recipeId, size_t index)
{
    m_sql->updateRecipeId(recipeId, index);
}

void SqlDataSerializer::removePrevIterations(size_t numIterationsToKeep)
{
    auto iterations = m_sql->getIterations(m_graphInfo);
    if (iterations.size() <= numIterationsToKeep) return;
    auto numOfIterationsToRemove = iterations.size() - numIterationsToKeep;
    iterations.erase(iterations.begin() + numOfIterationsToRemove, iterations.end());
    for (const auto& i : iterations)
    {
        m_sql->removeDataId(m_graphInfo, i);
    }
    auto dataIdsToRemove = m_sql->getDataIdsToRemove(m_graphInfo);
    for (const auto& i : dataIdsToRemove)
    {
        m_sql->removeData(i);
    }
}
}  // namespace data_serialize
