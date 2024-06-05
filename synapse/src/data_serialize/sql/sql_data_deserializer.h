#pragma once

#include "include/data_serializer/data_serializer.h"
#include "sql_db_deserializer.h"

namespace data_serialize
{
class SqlGraphDataDeserializer : public GraphDataDeserializer
{
public:
    SqlGraphDataDeserializer(const std::string& filePath, const GraphInfo& graphInfo);
    virtual ~SqlGraphDataDeserializer() = default;

    std::vector<std::string> getTensorsNames() const override;
    std::vector<TSize>       getShape(const std::string& tensorName, uint64_t iteration) const override;
    std::vector<uint8_t>     getPermutation(const std::string& tensorName, uint64_t iteration) const override;
    std::set<uint64_t>       getDataIterations() const override;
    std::set<uint64_t>       getNonDataIterations() const override;
    synDataType              getDataType(const std::string& tensorName, uint64_t iteration) const override;
    uint64_t                 getDataSize(const std::string& tensorName, uint64_t iteration) const override;
    std::vector<uint8_t>     getData(const std::string& tensorName, uint64_t iteration) const override;
    void getData(const std::string& tensorName, uint64_t iteration, uint8_t* data, TSize dataSize) const override;

private:
    std::string                      m_filePath;
    std::unique_ptr<SqlDeserializer> m_sql;
    std::map<std::string, uint64_t>  m_tensors;
    GraphInfo                        m_graphInfo;
};

class SqlDataDeserializer : public DataDeserializer
{
public:
    SqlDataDeserializer(const std::string& filePath);
    virtual ~SqlDataDeserializer() = default;

    std::vector<GraphInfo>   getGraphsInfos() const override;
    std::vector<GraphInfo>   getGraphsInfos(uint64_t group) const override;
    std::vector<uint64_t>    getGroups() const override;
    GraphDataDeserializerPtr getGraph(const GraphInfo& graphInfo) const override;

private:
    std::string                      m_filePath;
    std::unique_ptr<SqlDeserializer> m_sql;
};
}  // namespace data_serialize
