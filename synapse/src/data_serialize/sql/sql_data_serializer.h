#pragma once

#include "data_serializer/data_serializer.h"
#include "sql_db_serializer.h"

namespace data_serialize
{
class SqlDataSerializer : public DataSerializer
{
public:
    SqlDataSerializer(const std::string& filePath, const GraphInfo& graphInfo);

    virtual ~SqlDataSerializer() = default;

    virtual size_t serialize(TensorMetadata& tensorsData) override;
    virtual void   updateData(TensorMetadata& tensorsData) override;
    virtual void   updateRecipeId(uint16_t recipeId, size_t index) override;

    virtual uint64_t         getIterationCount() const override;
    virtual const GraphInfo& getInfo() override;
    virtual void             removePrevIterations(size_t numIterationsToKeep) override;

private:
    std::unique_ptr<SqlSerializer> m_sql;
    const GraphInfo                m_graphInfo;
};
}  // namespace data_serialize
