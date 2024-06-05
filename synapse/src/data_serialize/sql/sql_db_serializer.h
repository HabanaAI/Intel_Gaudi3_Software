#pragma once

#include "data_serializer/ds_types.h"
#include "sql_db.h"

struct sqlite3;

namespace data_serialize
{
class SqlSerializer : public Sql
{
public:
    SqlSerializer(const std::string& dbName);

    size_t   insert(const GraphInfo& graphInfo, TensorMetadata& tmd);
    void     updateData(const GraphInfo& graphInfo, TensorMetadata& tmd);
    void     updateRecipeId(uint16_t recipeId, size_t index);
    void     insertData(const TensorMetadata& tmd, int64_t dataHash);
    size_t   insertTensor(const GraphInfo& graphInfo, const TensorMetadata& tmd, int64_t dataHash);
    uint64_t getIterationCount(const GraphInfo& graphInfo) const;
    uint64_t getTensorIterationCount(const GraphInfo& graphInfo, uint64_t tensorId) const;
    unsigned getIndexForInsertion(Lock& lock) const;

    // Remove prev data iterations.
    std::vector<uint64_t> getIterations(const GraphInfo& graphInfo) const;
    std::vector<int64_t>  getDataIdsToRemove(const GraphInfo& graphInfo) const;
    void                  removeDataId(const GraphInfo& graphInfo, size_t iteration) const;
    void                  removeData(int64_t dataId) const;

private:
    void create();
    void createTable(const std::string& graphId);
    bool tableExist(const std::string& tableName) const;
};
}  // namespace data_serialize