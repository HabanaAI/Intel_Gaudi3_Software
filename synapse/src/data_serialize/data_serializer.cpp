#include "data_serializer/data_serializer.h"
#include "sql/sql_data_serializer.h"
#include "sql/sql_data_deserializer.h"

namespace data_serialize
{
DataSerializerPtr DataSerializer::create(const std::string& filePath, const GraphInfo& graphInfo)
{
    return std::make_shared<SqlDataSerializer>(filePath, graphInfo);
}

DataDeserializerPtr DataDeserializer::create(const std::string& filePath)
{
    return std::make_shared<SqlDataDeserializer>(filePath);
}
}  // namespace data_serialize
