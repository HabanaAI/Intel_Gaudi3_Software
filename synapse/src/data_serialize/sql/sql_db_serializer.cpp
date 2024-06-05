#include "sql_db_serializer.h"

#include "sql_db.h"
#include "synapse_api.h"
#include "types_exception.h"

#include <cstdint>
#include <lz4/lz4hc.h>
#include <memory>
#include <sqlite/sqlite3.h>

#include <functional>
#include <string_view>

namespace data_serialize
{
static const int FILE_VERSION = 8;

SqlSerializer::SqlSerializer(const std::string& dbName) : Sql(dbName, false)
{
    create();
}

static void tryCompress(TensorMetadata& tmd)
{
    // Avoid compression in case of failure.
    if (tmd.dataSize > std::numeric_limits<int>::max() || tmd.data == nullptr)
    {
        tmd.compression = Compression::NO_COMP;
        return;
    }
    uint8_t*  data       = reinterpret_cast<uint8_t*>(tmd.data.get());
    const int maxDstSize = LZ4_compressBound(tmd.dataSize);
    tmd.compressedData   = std::vector<uint8_t>(maxDstSize);

    int compressedDataDize = 0;
    switch (tmd.compression)
    {
        case Compression::NO_COMP:
            break;
        case Compression::LZ4:
            compressedDataDize = LZ4_compress_default(reinterpret_cast<char*>(data),
                                                      reinterpret_cast<char*>(tmd.compressedData.data()),
                                                      tmd.dataSize,
                                                      maxDstSize);
            break;
        case Compression::LZ4_HC:
            compressedDataDize = LZ4_compress_HC(reinterpret_cast<char*>(data),
                                                 reinterpret_cast<char*>(tmd.compressedData.data()),
                                                 tmd.dataSize,
                                                 maxDstSize,
                                                 LZ4HC_CLEVEL_MAX);
            break;
    }

    // Avoid compression in case of failure.
    if (compressedDataDize <= 0 || tmd.dataSize < compressedDataDize)
    {
        tmd.compression    = Compression::NO_COMP;
        tmd.compressedData = {};
    }
    else
    {
        tmd.compressedData.resize(compressedDataDize);
    }
}

static int64_t getHash(const TensorMetadata& tmd)
{
    if (tmd.data)
    {
        return bit_cast<int64_t>(
            std::hash<std::string_view> {}({reinterpret_cast<char*>(tmd.data.get()), tmd.dataSize}));
    }
    return 0;
}

bool SqlSerializer::tableExist(const std::string& tableName) const
{
    std::string tensorTableCountCmd = fmt::format("select count(name) from sqlite_schema where name = '{}'", tableName);
    uint32_t    tableCount          = 0;
    exe(tensorTableCountCmd, &tableCount, singleIntCallback<uint32_t>);

    return tableCount > 0;
}

void SqlSerializer::createTable(const std::string& tableName)
{
    if (!tableExist(tableName))
    {
        std::string table = fmt::format("create table if not exists '{}' ("
                                        "ROW_INDEX      int     not NULL,"
                                        "GROUP_ID       int     not NULL,"
                                        "LAUNCH_INDEX   int     not NULL,"
                                        "GRAPH_NAME     text    not NULL,"
                                        "RECIPE_ID      int     not NULL,"
                                        "NAME           text    not NULL,"
                                        "ID             int     not NULL,"
                                        "ITERATION      int     not NULL,"
                                        "TYPE           int     not NULL,"
                                        "DATA_TYPE      int     not NULL,"
                                        "VALIDATION     int     not NULL,"
                                        "CONST_TENSOR   int     not NULL,"
                                        "SHAPE          blob,"
                                        "PERMUTATION    blob,"
                                        "DATA_ID        int     not NULL,"
                                        "CONSTRAINT tensor_pk PRIMARY KEY (RECIPE_ID,ROW_INDEX,ID,ITERATION),"
                                        "FOREIGN KEY(DATA_ID) references DATA(ID));",
                                        tableName);

        exe(table, nullptr, nullptr);
    }
}

void SqlSerializer::create()
{
    exe("PRAGMA foreign_keys = ON", nullptr, nullptr);

    std::string version = "create table if not exists VERSION("
                          "SYNAPSE     text    not NULL,"
                          "FILE        int     not NULL,"
                          "CONSTRAINT version_pk PRIMARY KEY (FILE));";

    std::string data = "create table if not exists DATA("
                       "ID          int not NULL,"
                       "COMPRESSION int     not NULL,"
                       "DATA        blob,"
                       "CONSTRAINT data_pk PRIMARY KEY (ID));";

    exe(version, nullptr, nullptr);
    exe(data, nullptr, nullptr);

    const int maxLen = 256;
    char      synVersion[maxLen];
    synDriverGetVersion(synVersion, maxLen);

    std::string cmd = fmt::format("insert or replace into VERSION (SYNAPSE,FILE) "
                                  "VALUES ('{}',{});",
                                  synVersion,
                                  FILE_VERSION);
    exe(cmd, nullptr, nullptr);

    createTable(TENSORS_TABLE);
}

void SqlSerializer::insertData(const TensorMetadata& tmd, int64_t dataHash)
{
    Lock lk(m_dbFile, m_readOnly);

    uint64_t       dataSize = tmd.compressedData.empty() ? tmd.dataSize : tmd.compressedData.size();
    const uint8_t* data =
        tmd.compressedData.empty() ? reinterpret_cast<uint8_t*>(tmd.data.get()) : tmd.compressedData.data();

    sqlite3_stmt* stmt = nullptr;

    std::string insert = fmt::format("INSERT OR REPLACE into DATA (ID,COMPRESSION,DATA) "
                                     "VALUES ({},{},?);",
                                     dataHash,
                                     int(tmd.compression));

    HB_ASSERT(sqlite3_prepare_v2(lk.db(), insert.c_str(), -1, &stmt, nullptr) == SQLITE_OK,
              "prepare failed, sql: {}, error: {}",
              insert,
              sqlite3_errmsg(lk.db()));

    HB_ASSERT(sqlite3_bind_blob(stmt, 1, data ? data : nullptr, dataSize, SQLITE_STATIC) == SQLITE_OK,
              "bind blob failed: {}",
              sqlite3_errmsg(lk.db()));

    HB_ASSERT(sqlite3_step(stmt) == SQLITE_DONE, "step failed: {}", sqlite3_errmsg(lk.db()));

    HB_ASSERT(sqlite3_finalize(stmt) == SQLITE_OK, "finalize failed: {}", sqlite3_errmsg(lk.db()));
}

size_t SqlSerializer::insertTensor(const GraphInfo& graphInfo, const TensorMetadata& tmd, int64_t dataHash)
{
    constexpr int invalidDataIteration = -1;
    uint64_t      iteration            = 0;
    if (tmd.validation == TensorValidation::VALID)
    {
        iteration = getTensorIterationCount(graphInfo, tmd.id);
    }

    Lock lk(m_dbFile, m_readOnly);

    sqlite3_stmt* stmt  = nullptr;
    unsigned      index = getIndexForInsertion(lk);
    std::string   insert =
        fmt::format("INSERT OR REPLACE into '{}' "
                    "(ROW_INDEX,GROUP_ID,LAUNCH_INDEX,GRAPH_NAME,RECIPE_ID,NAME,ID,ITERATION,TYPE,DATA_TYPE,VALIDATION,"
                    "CONST_TENSOR,SHAPE,PERMUTATION,DATA_ID) "
                    "VALUES ({}, {}, {}, '{}', {}, '{}', {}, {}, {}, {}, {}, {}, ?, ?, {});",
                    TENSORS_TABLE,
                    index,
                    graphInfo.group,
                    tmd.launchIndex,
                    graphInfo.id,
                    int(graphInfo.recipeId),
                    tmd.name,
                    tmd.id,
                    tmd.validation == TensorValidation::VALID ? iteration : invalidDataIteration,
                    int(tmd.type),
                    int(tmd.dataType),
                    int(tmd.validation),
                    int(tmd.constTensor),
                    dataHash);

    HB_ASSERT(sqlite3_prepare_v2(lk.db(), insert.c_str(), -1, &stmt, nullptr) == SQLITE_OK,
              "prepare failed, sql: {}, error: {}",
              insert,
              sqlite3_errmsg(lk.db()));

    HB_ASSERT(sqlite3_bind_blob(stmt, 1, tmd.shape.data(), tmd.shape.size() * sizeof(tmd.shape[0]), SQLITE_STATIC) ==
                  SQLITE_OK,
              "bind blob failed: {}",
              sqlite3_errmsg(lk.db()));

    HB_ASSERT(sqlite3_bind_blob(stmt,
                                2,
                                tmd.permutation.empty() ? nullptr : tmd.permutation.data(),
                                tmd.permutation.size() * sizeof(tmd.permutation[0]),
                                SQLITE_STATIC) == SQLITE_OK,
              "bind blob failed: {}",
              sqlite3_errmsg(lk.db()));

    HB_ASSERT(sqlite3_step(stmt) == SQLITE_DONE, "step failed: {}", sqlite3_errmsg(lk.db()));

    HB_ASSERT(sqlite3_finalize(stmt) == SQLITE_OK, "finalize failed: {}", sqlite3_errmsg(lk.db()));

    return index;
}

size_t SqlSerializer::insert(const GraphInfo& graphInfo, TensorMetadata& tmd)
{
    int64_t     dataHash     = getHash(tmd);
    std::string dataCountCmd = fmt::format("select count(*) from DATA where ID = {}", dataHash);
    uint64_t    dataCount    = 0;
    exe(dataCountCmd, &dataCount, singleIntCallback<uint64_t>);

    if (dataCount == 0)
    {
        tryCompress(tmd);
        insertData(tmd, dataHash);
    }

    auto index = insertTensor(graphInfo, tmd, dataHash);
    tmd.index  = index;
    return index;
}

void SqlSerializer::updateData(const GraphInfo& graphInfo, TensorMetadata& tmd)
{
    HB_ASSERT(tmd.index.has_value(),
              "can't update tensor data for tensor: {} recipe ID: {}, missing tensor index",
              tmd.name,
              graphInfo.recipeId);

    int64_t     dataHash     = getHash(tmd);
    std::string dataCountCmd = fmt::format("select count(*) from DATA where ID = {}", dataHash);
    uint64_t    dataCount    = 0;
    exe(dataCountCmd, &dataCount, singleIntCallback<uint64_t>);

    if (dataCount == 0)
    {
        tryCompress(tmd);
        insertData(tmd, dataHash);
    }

    std::string setData =
        fmt::format("update {} set DATA_ID = {} where ROW_INDEX = {};", TENSORS_TABLE, dataHash, tmd.index.value());

    Lock lk(m_dbFile, m_readOnly);

    exe(setData, nullptr, nullptr);
}

void SqlSerializer::updateRecipeId(uint16_t recipeId, size_t index)
{
    std::string setRecipeId =
        fmt::format("update {} set RECIPE_ID = {} where ROW_INDEX = {};", TENSORS_TABLE, int(recipeId), index);

    Lock lk(m_dbFile, m_readOnly);

    exe(setRecipeId, nullptr, nullptr);
}

uint64_t SqlSerializer::getIterationCount(const GraphInfo& graphInfo) const
{
    // any entry of same graph ID and same tensor name is considered as iteration
    if (!tableExist(TENSORS_TABLE)) return 0;

    std::string countCmd =
        fmt::format("select count(*) from {} where RECIPE_ID = {} and VALIDATION = {} and CONST_TENSOR = 0",
                    TENSORS_TABLE,
                    int(graphInfo.recipeId),
                    int(TensorValidation::VALID));

    uint64_t count = 0;
    exe(countCmd, &count, singleIntCallback<uint64_t>);

    if (count == 0) return 0;

    std::string iterationsCmd =
        fmt::format("select MAX(ITERATION) from {} where RECIPE_ID = {} and VALIDATION = {} and CONST_TENSOR = 0",
                    // fmt::format("select MAX(ITERATION) from {}",
                    TENSORS_TABLE,
                    int(graphInfo.recipeId),
                    int(TensorValidation::VALID));

    uint64_t iteration = 0;
    exe(iterationsCmd, &iteration, singleIntCallback<uint64_t>);
    if (static_cast<int>(iteration) == INVALID_CALLBACK_RESULT)
    {
        throw SynapseException(fmt::format("Failed to read iteration count, graph ID: {}, recipe ID: {}",
                                           graphInfo.id,
                                           graphInfo.recipeId));
    }
    return iteration + 1;
}

uint64_t SqlSerializer::getTensorIterationCount(const GraphInfo& graphInfo, uint64_t tensorId) const
{
    // any entry of same graph ID and same tensor name is considered as iteration
    if (!tableExist(TENSORS_TABLE)) return 0;

    std::string count = fmt::format(
        "select count(*) from '{}' where RECIPE_ID = {} and ID = {} and VALIDATION = {} and CONST_TENSOR = 0",
        TENSORS_TABLE,
        graphInfo.recipeId,
        tensorId,
        int(TensorValidation::VALID));
    uint64_t size = 0;
    exe(count, &size, singleIntCallback<uint64_t>);
    if (static_cast<int>(size) == INVALID_CALLBACK_RESULT)
    {
        throw SynapseException(fmt::format("Failed to read iteration count, graph ID: {}, group: {}, tensor id: {}",
                                           graphInfo.id,
                                           graphInfo.group,
                                           tensorId));
    }
    return size;
}

unsigned SqlSerializer::getIndexForInsertion(Lock& lock) const
{
    std::string query = fmt::format("select count(*) from '{}'", TENSORS_TABLE);
    uint64_t    size  = 0;
    exeCmd(lock.db(), query, &size, singleIntCallback<uint64_t>);
    if (static_cast<int>(size) == INVALID_CALLBACK_RESULT)
    {
        throw SynapseException(fmt::format("Failed to read rows count"));
    }
    return size;
}

std::vector<uint64_t> SqlSerializer::getIterations(const GraphInfo& graphInfo) const
{
    using SetType = std::set<uint64_t>;

    SetType iterations;

    const auto query = fmt::format("select DISTINCT ITERATION from '{}' where RECIPE_ID = {} and DATA_ID != 0",
                                   TENSORS_TABLE,
                                   graphInfo.recipeId);

    exe(query, &iterations, [](void* data, int argc, char** argv, char** colNames) {
        auto& ret = *static_cast<SetType*>(data);
        ret.insert(std::stoull(argv[0]));
        return 0;
    });

    std::vector<uint64_t> ret(iterations.begin(), iterations.end());
    std::sort(ret.begin(), ret.end());
    return ret;
}

std::vector<int64_t> SqlSerializer::getDataIdsToRemove(const GraphInfo& graphInfo) const
{
    using SetType = std::set<int64_t>;

    SetType tensorsDataIds;

    const auto queryTensors =
        fmt::format("select DISTINCT DATA_ID from '{}' where DATA_ID != 0", TENSORS_TABLE, graphInfo.recipeId);

    exe(queryTensors, &tensorsDataIds, [](void* data, int argc, char** argv, char** colNames) {
        auto& ret = *static_cast<SetType*>(data);
        ret.insert(std::stoll(argv[0]));
        return 0;
    });

    SetType dataIds;

    const auto queryData = fmt::format("select DISTINCT ID from '{}' where ID != 0", "DATA", graphInfo.recipeId);

    exe(queryData, &dataIds, [](void* data, int argc, char** argv, char** colNames) {
        auto& ret = *static_cast<SetType*>(data);
        ret.insert(std::stoll(argv[0]));
        return 0;
    });

    std::vector<int64_t> ret;
    for (const auto& dataId : dataIds)
    {
        if (tensorsDataIds.find(dataId) == tensorsDataIds.end())
        {
            ret.push_back(dataId);
        }
    }
    return ret;
}

void SqlSerializer::removeDataId(const GraphInfo& graphInfo, size_t iteration) const
{
    std::string cmd = fmt::format("update {} set DATA_ID = 0 where RECIPE_ID = {} and ITERATION = {};",
                                  TENSORS_TABLE,
                                  graphInfo.recipeId,
                                  iteration);

    Lock lk(m_dbFile, m_readOnly);

    exe(cmd, nullptr, nullptr);
}

void SqlSerializer::removeData(int64_t dataId) const
{
    const auto idsToDelete = fmt::format("delete from '{}' where ID = {}", "DATA", dataId);

    Lock lk(m_dbFile, m_readOnly);

    exe(idsToDelete, nullptr, nullptr);
}
}  // namespace data_serialize