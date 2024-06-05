#include "sql_db_deserializer.h"

#include "data_serializer/ds_types.h"
#include "defs.h"
#include "log_manager.h"
#include "synapse_common_types.h"
#include "type_utils.h"
#include "types_exception.h"

#include "lz4/lz4.h"
#include "spdlog/fmt/bundled/core.h"

#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

namespace data_serialize
{
std::unique_ptr<SqlDeserializer> SqlDeserializer::createSqlDeserializer(const std::string& dbName)
{
    HB_ASSERT(sqlite3_threadsafe() == 1, "sqlite wasn't compile in thread safe mode");

    uint32_t version = 0;
    {
        Sql::SqlDb sqlDb(dbName, true);

        std::string count      = "select count(name) from sqlite_schema where name = 'VERSION'";
        uint32_t    tableCount = 0;
        exeCmd(sqlDb.db(), count, &tableCount, singleIntCallback<uint32_t>);
        if (tableCount == 0)
        {
            return std::make_unique<SqlDeserializerV0>(dbName);
        }

        std::string select = "select FILE from VERSION";
        exeCmd(sqlDb.db(), select, &version, singleIntCallback<uint32_t>);
    }

    switch (version)
    {
        case 0:
            return std::make_unique<SqlDeserializerV0>(dbName);
        case 1:
            return std::make_unique<SqlDeserializerV1>(dbName);
        case 2:
            return std::make_unique<SqlDeserializerV2>(dbName);
        case 3:
            return std::make_unique<SqlDeserializerV3>(dbName);
        case 4:
            return std::make_unique<SqlDeserializerV4>(dbName);
        case 5:
            return std::make_unique<SqlDeserializerV5>(dbName);
        case 6:
            return std::make_unique<SqlDeserializerV6>(dbName);
        case 7:
            return std::make_unique<SqlDeserializerV7>(dbName);
        case 8:
            return std::make_unique<SqlDeserializerV8>(dbName);
        default:
            throw SynapseException(fmt::format("Invalid DB file version: {}", version));
    }

    return nullptr;
}

SqlDeserializer::SqlDeserializer(const std::string& dbName) : Sql(dbName, true) {}

void SqlDeserializer::decompress(const char* src, int srcSize, char* dst, int dstSize)
{
    const int decompressedSize = LZ4_decompress_safe(src, dst, srcSize, dstSize);
    HB_ASSERT(decompressedSize == dstSize,
              "decompressed size ({}) is different than expected size({})",
              decompressedSize,
              dstSize);
}

int SqlDeserializer::tensorMetadataCallback(void* data, int argc, char** argv, char** colNames)
{
    TensorTableColumn& tmd = *static_cast<TensorTableColumn*>(data);
    for (size_t i = 0; i < argc; i++)
    {
        std::string colName = colNames[i];
        auto&       value   = argv[i];
        if (colName == "TYPE")
        {
            tmd.dataType = synDataType(std::stoi(value));
        }
        if (colName == "DATA_TYPE")
        {
            tmd.dataType = synDataType(std::stoi(value));
        }
        if (colName == "SHAPE")
        {
            for (auto& d : tmd.shape)
            {
                std::memcpy(&d, value, tmd.dimSizeInBytes);
                value += tmd.dimSizeInBytes;
            }
        }
        if (colName == "PERMUTATION")
        {
            std::memcpy(tmd.permutation.data(), value, tmd.permutation.size() * sizeof(tmd.permutation[0]));
        }
        if (colName == "ITERATION")
        {
            tmd.dataIterations.insert(std::stoi(value));
        }
        if (colName == "DATA")
        {
            if (value == nullptr)
            {
                tmd.invalidData = true;
                continue;
            }

            if (tmd.compression == Compression::NO_COMP)
            {
                std::memcpy(tmd.data, value, tmd.dataSize);
            }
            else
            {
                decompress(value, tmd.compressedDataSize, reinterpret_cast<char*>(tmd.data), tmd.dataSize);
            }
        }
    }
    return 0;
}

std::vector<std::string> SqlDeserializer::getGraphsNames() const
{
    struct GraphsNames
    {
        std::vector<std::string> names;
    };

    std::string list = getGraphsNamesQuery();
    GraphsNames ret;
    exe(list, &ret, [](void* data, int argc, char** argv, char** colNames) {
        auto graphsNames = static_cast<GraphsNames*>(data);
        graphsNames->names.push_back(argv[0]);
        return 0;
    });
    return ret.names;
}

std::vector<uint64_t> SqlDeserializer::getGroups() const
{
    return {};
}

std::vector<GraphInfo> SqlDeserializer::getGraphsInfos() const
{
    return {};
}

std::vector<GraphInfo> SqlDeserializer::getGraphsInfos(uint64_t group) const
{
    return {};
}

std::vector<std::string> SqlDeserializer::getTensorsNames(const GraphInfo& graphInfo) const
{
    struct TensorsNames
    {
        std::vector<std::string> names;
    };

    TensorsNames tensorsNames;
    std::string  list = getTensorsNamesQuery(graphInfo);
    exe(list, &tensorsNames, [](void* data, int argc, char** argv, char** colNames) {
        auto tensorsNames = static_cast<TensorsNames*>(data);
        tensorsNames->names.push_back(argv[0]);
        return 0;
    });
    return tensorsNames.names;
}

synDataType SqlDeserializer::getDataType(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    std::string       select = getDataTypeQuery(graphInfo, name, iteration);
    TensorTableColumn tmd;
    exe(select, &tmd, tensorMetadataCallback);
    return tmd.dataType;
}

uint8_t SqlDeserializer::getDimSizeInBytes() const
{
    return sizeof(uint32_t);
}

uint64_t SqlDeserializer::getShapeSize(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    std::string select = getShapeSizeQuery(graphInfo, name, iteration);
    uint64_t    size   = 0;
    auto        sts    = exe(select, &size, singleIntCallback<uint64_t>);
    HB_ASSERT(sts == SQLITE_OK,
              "Failed to read shape size, graph ID: {}, tensor name: {}, iteration: {}",
              graphInfo.id,
              name,
              iteration);
    return size / getDimSizeInBytes();
}

uint64_t
SqlDeserializer::getPermutationSize(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    std::string select = getPermutationSizeQuery(graphInfo, name, iteration);
    uint64_t    size   = 0;
    auto        sts    = exe(select, &size, singleIntCallback<uint64_t>);
    HB_ASSERT(sts == SQLITE_OK,
              "Failed to read permutation size, graph ID: {}, tensor name: {}, iteration: {}",
              graphInfo.id,
              name,
              iteration);
    return size;
}

Compression
SqlDeserializer::getCompressionType(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    std::string select = getCompressionTypeQuery(graphInfo, name, iteration);

    Compression comp;
    auto        sts = exe(select, &comp, singleIntCallback<int>);
    HB_ASSERT(sts == SQLITE_OK,
              "Failed to read compression type, graph ID: {}, tensor name: {}, iteration: {}",
              graphInfo.id,
              name,
              iteration);
    return comp;
}

Compression SqlDeserializer::getCompressionType(int64_t dataId) const
{
    std::string select = getCompressionTypeQuery(dataId);

    Compression comp;
    auto        sts = exe(select, &comp, singleIntCallback<int>);
    HB_ASSERT(sts == SQLITE_OK, "Failed to read compression type, data ID: {}", dataId);
    return comp;
}

uint64_t SqlDeserializer::getDataSize(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    synDataType        dataType = getDataType(graphInfo, name, iteration);
    std::vector<TSize> shape    = getShape(graphInfo, name, iteration);
    return getActualTensorSize<TSize>(shape.size(), shape.data(), dataType);
}

int64_t SqlDeserializer::getDataId(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    std::string select = getDataIdQuery(graphInfo, name, iteration);
    int64_t     id     = 0;
    auto        sts    = exe(select, &id, singleIntCallback<int64_t>);
    HB_ASSERT(sts == SQLITE_OK,
              "Failed to read data ID, graph ID: {}, tensor name: {}, iteration: {}",
              graphInfo.id,
              name,
              iteration);
    return id;
}

std::vector<TSize>
SqlDeserializer::getShape(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    auto shapeSize = getShapeSize(graphInfo, name, iteration);
    if (shapeSize == 0) return {};
    std::string       select = getShapeQuery(graphInfo, name, iteration);
    TensorTableColumn tmd;
    tmd.shape          = std::vector<TSize>(shapeSize);
    tmd.dimSizeInBytes = getDimSizeInBytes();
    exe(select, &tmd, tensorMetadataCallback);
    return tmd.shape;
}

std::vector<uint8_t>
SqlDeserializer::getPermutation(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    auto permutationSize = getPermutationSize(graphInfo, name, iteration);
    if (permutationSize == 0) return {};
    std::string       select = getPermutationQuery(graphInfo, name, iteration);
    TensorTableColumn tmd;
    tmd.permutation = std::vector<uint8_t>(permutationSize);
    exe(select, &tmd, tensorMetadataCallback, false);
    return tmd.permutation;
}

std::vector<uint8_t>
SqlDeserializer::getData(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    uint64_t             dataSize = getDataSize(graphInfo, name, iteration);
    std::vector<uint8_t> data(dataSize);
    getData(graphInfo, name, iteration, data.data(), dataSize);
    return data;
}

void SqlDeserializer::getData(const GraphInfo&   graphInfo,
                              const std::string& name,
                              uint64_t           iteration,
                              uint8_t*           data,
                              TSize              dataSize) const
{
    std::string select = getDataQuery(graphInfo, name, iteration);

    TensorTableColumn tmd;
    tmd.data               = data;
    tmd.invalidData        = false;
    tmd.dataSize           = dataSize;
    tmd.compressedDataSize = getStoredDataSize(graphInfo, name, iteration);
    tmd.compression        = getCompressionType(graphInfo, name, iteration);
    exe(select, &tmd, tensorMetadataCallback);
}

uint64_t
SqlDeserializer::getStoredDataSize(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    std::string select = getStoredDataSizeQuery(graphInfo, name, iteration);
    uint64_t    size   = 0;
    auto        sts    = exe(select, &size, singleIntCallback<uint64_t>);
    if (sts != SQLITE_OK)
    {
        LOG_WARN(SYNREC,
                 "Failed to read stored data size, graph ID: {}, tensor name: {}, iteration: {}",
                 graphInfo.id,
                 name,
                 iteration);
    }
    return size;
}

uint64_t SqlDeserializer::getStoredDataSize(int64_t dataId) const
{
    if (dataId == 0) return 0;

    std::string select = getStoredDataSizeQuery(dataId);
    uint64_t    size   = 0;
    auto        sts    = exe(select, &size, singleIntCallback<uint64_t>);
    HB_ASSERT(sts == SQLITE_OK, "Failed to read stored data size, data ID: {}", dataId);
    return size;
}

std::set<uint64_t> SqlDeserializer::getDataIterations(const GraphInfo& graphInfo) const
{
    std::string       iters = getDataIterationsQuery(graphInfo);
    TensorTableColumn tmd {};
    exe(iters, &tmd, tensorMetadataCallback);
    return tmd.dataIterations;
}

std::set<uint64_t> SqlDeserializer::getNonDataIterations(const GraphInfo& graphInfo) const
{
    std::string       iters = getNonDataIterationsQuery(graphInfo);
    TensorTableColumn tmd {};
    exe(iters, &tmd, tensorMetadataCallback);
    // tmd.dataIterations contains iterations that have DATA_ID = 0 in the db,
    // but there may be intersection with iterations that have DATA_ID != 0 (getDataIterations).

    const auto dataIters = getDataIterations(graphInfo);
    for (const auto& dataIter : dataIters)
    {
        tmd.dataIterations.erase(dataIter);
    }
    return tmd.dataIterations;
}

std::string SqlDeserializer::getGraphsNamesQuery() const
{
    return "select name from sqlite_schema where name NOT LIKE '%VERSION%' and name not like '%DATA%' and "
           "name not like '%sqlite%'";
}

std::string SqlDeserializer::getTensorsNamesQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select DISTINCT NAME from TENSORS where GRAPH_ID = '{}'", graphInfo.id);
}

std::string
SqlDeserializer::getDataTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select DATA_TYPE from TENSORS where NAME = '{}' and GRAPH_ID = '{}' and ITERATION = '{}'",
                       name,
                       graphInfo.id,
                       iteration);
}

std::string
SqlDeserializer::getShapeSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select length(SHAPE) from TENSORS where NAME = '{}' and GRAPH_ID = '{}' and ITERATION = '{}'",
                       name,
                       graphInfo.id,
                       iteration);
}

std::string
SqlDeserializer::getPermutationSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format(
        "select length(PERMUTATION) from TENSORS where NAME = '{}' and GRAPH_ID = '{}' and ITERATION = '{}'",
        name,
        graphInfo.id,
        iteration);
}

std::string
SqlDeserializer::getCompressionTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select COMPRESSION from TENSORS where NAME = '{}' and GRAPH_ID = '{}' and ITERATION = '{}'",
                       name,
                       graphInfo.id,
                       iteration);
}

std::string SqlDeserializer::getCompressionTypeQuery(int64_t dataId) const
{
    return fmt::format("select COMPRESSION from DATA where ID = {}", dataId);
}

std::string
SqlDeserializer::getDataIdQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select DATA_ID from TENSORS where NAME = '{}' and GRAPH_ID = '{}' and ITERATION = '{}'",
                       name,
                       graphInfo.id,
                       iteration);
}

std::string
SqlDeserializer::getShapeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select SHAPE from TENSORS where NAME = '{}' and GRAPH_ID = '{}' and ITERATION = '{}'",
                       name,
                       graphInfo.id,
                       iteration);
}

std::string
SqlDeserializer::getPermutationQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select PERMUTATION from TENSORS where NAME = '{}' and GRAPH_ID = '{}' and ITERATION = '{}'",
                       name,
                       graphInfo.id,
                       iteration);
}

std::string SqlDeserializer::getDataQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select DATA from TENSORS where NAME = '{}' and GRAPH_ID = '{}' and ITERATION = '{}'",
                       name,
                       graphInfo.id,
                       iteration);
}

std::string SqlDeserializer::getDataQuery(int64_t dataId) const
{
    return fmt::format("select DATA from DATA where ID = {}", dataId);
}

std::string SqlDeserializer::getStoredDataSizeQuery(int64_t dataId) const
{
    return fmt::format("select length(DATA) from DATA where ID = {}", dataId);
}

std::string
SqlDeserializer::getStoredDataSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select length(DATA) from TENSORS where NAME = '{}' and GRAPH_ID = '{}' and ITERATION = '{}'",
                       name,
                       graphInfo.id,
                       iteration);
}

std::string SqlDeserializer::getDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from TENSORS where GRAPH_ID = '{}' and DATA != 'NULL'", graphInfo.id);
}

std::string SqlDeserializer::getNonDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from TENSORS where GRAPH_ID = '{}' and DATA == 'NULL'", graphInfo.id);
}

SqlDeserializerV0::SqlDeserializerV0(const std::string& dbName) : SqlDeserializer(dbName) {}

std::string
SqlDeserializerV0::getDataQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select DATA from TENSORS where NAME = '{}' and GRAPH_ID = '{}' and ITERATION = '{}'",
                       name,
                       graphInfo.id,
                       iteration);
}

std::string
SqlDeserializerV0::getStoredDataSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select length(DATA) from TENSORS where NAME = '{}' and GRAPH_ID = '{}' and ITERATION = '{}'",
                       name,
                       graphInfo.id,
                       iteration);
}

std::string SqlDeserializerV0::getDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from TENSORS where GRAPH_ID = '{}' and DATA != 'NULL'", graphInfo.id);
}

std::string SqlDeserializerV0::getNonDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from TENSORS where GRAPH_ID = '{}' and DATA == 'NULL'", graphInfo.id);
}

SqlDeserializerV1::SqlDeserializerV1(const std::string& dbName) : SqlDeserializer(dbName) {}

std::string
SqlDeserializerV1::getDataQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    auto dataId = getDataId(graphInfo, name, iteration);
    return fmt::format("select DATA from DATA where DATA_ID = {}", dataId);
}

std::string
SqlDeserializerV1::getStoredDataSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    auto dataId = getDataId(graphInfo, name, iteration);
    return fmt::format("select length(DATA) from DATA where DATA_ID = {}", dataId);
}

std::string SqlDeserializerV1::getDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from TENSORS where GRAPH_ID = '{}' and DATA_ID != 0", graphInfo.id);
}

std::string SqlDeserializerV1::getNonDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from TENSORS where GRAPH_ID = '{}' and DATA_ID == 0", graphInfo.id);
}

SqlDeserializerV2::SqlDeserializerV2(const std::string& dbName) : SqlDeserializerV1(dbName) {}

std::string SqlDeserializerV2::getTensorsNamesQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select DISTINCT NAME from '{}'", graphInfo.id);
}

std::string
SqlDeserializerV2::getDataTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select DATA_TYPE from '{}' where NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       name,
                       iteration);
}

std::string
SqlDeserializerV2::getShapeSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select length(SHAPE) from '{}' where NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       name,
                       iteration);
}

std::string SqlDeserializerV2::getPermutationSizeQuery(const GraphInfo&   graphInfo,
                                                       const std::string& name,
                                                       uint64_t           iteration) const
{
    return fmt::format("select length(PERMUTATION) from '{}' where NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       name,
                       iteration);
}

std::string SqlDeserializerV2::getCompressionTypeQuery(const GraphInfo&   graphInfo,
                                                       const std::string& name,
                                                       uint64_t           iteration) const
{
    return fmt::format("select COMPRESSION from '{}' where NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       name,
                       iteration);
}

std::string
SqlDeserializerV2::getDataIdQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select DATA_ID from '{}' where NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       name,
                       iteration);
}

std::string
SqlDeserializerV2::getShapeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select SHAPE from '{}' where NAME = '{}' and ITERATION = '{}'", graphInfo.id, name, iteration);
}

std::string
SqlDeserializerV2::getPermutationQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select PERMUTATION from '{}' where NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       name,
                       iteration);
}

std::string
SqlDeserializerV2::getDataQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    auto dataId = getDataId(graphInfo, name, iteration);
    return fmt::format("select DATA from DATA where ID = {}", dataId);
}

std::string
SqlDeserializerV2::getStoredDataSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    auto dataId = getDataId(graphInfo, name, iteration);
    return fmt::format("select length(DATA) from DATA where ID = {}", dataId);
}

std::string SqlDeserializerV2::getDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from '{}' where DATA_ID != 0", graphInfo.id);
}

std::string SqlDeserializerV2::getNonDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from '{}' where DATA_ID == 0", graphInfo.id);
}

SqlDeserializerV3::SqlDeserializerV3(const std::string& dbName) : SqlDeserializerV2(dbName) {}

uint8_t SqlDeserializerV3::getDimSizeInBytes() const
{
    return sizeof(TSize);
}

SqlDeserializerV4::SqlDeserializerV4(const std::string& dbName) : SqlDeserializerV3(dbName) {}

std::string SqlDeserializerV4::getTensorsNamesQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select DISTINCT NAME from '{}' where GRAPH_GROUP = '{}'", graphInfo.id, graphInfo.group);
}

std::string
SqlDeserializerV4::getDataTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select DATA_TYPE from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       graphInfo.group,
                       name,
                       iteration);
}

std::string
SqlDeserializerV4::getShapeSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select length(SHAPE) from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       graphInfo.group,
                       name,
                       iteration);
}

std::string SqlDeserializerV4::getPermutationSizeQuery(const GraphInfo&   graphInfo,
                                                       const std::string& name,
                                                       uint64_t           iteration) const
{
    return fmt::format(
        "select length(PERMUTATION) from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}'",
        graphInfo.id,
        graphInfo.group,
        name,
        iteration);
}

std::string SqlDeserializerV4::getCompressionTypeQuery(const GraphInfo&   graphInfo,
                                                       const std::string& name,
                                                       uint64_t           iteration) const
{
    return fmt::format("select COMPRESSION from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       graphInfo.group,
                       name,
                       iteration);
}

std::string
SqlDeserializerV4::getDataIdQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select DATA_ID from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       graphInfo.group,
                       name,
                       iteration);
}

std::string
SqlDeserializerV4::getShapeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select SHAPE from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       graphInfo.group,
                       name,
                       iteration);
}

std::string
SqlDeserializerV4::getPermutationQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select PERMUTATION from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}'",
                       graphInfo.id,
                       graphInfo.group,
                       name,
                       iteration);
}

std::string SqlDeserializerV4::getDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from '{}' where GRAPH_GROUP = '{}' and DATA_TYPE != 0 and DATA_ID != 0",
                       graphInfo.id,
                       graphInfo.group);
}

std::string SqlDeserializerV4::getNonDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from '{}' where GRAPH_GROUP = '{}' and DATA_TYPE != 0 and DATA_ID == 0",
                       graphInfo.id,
                       graphInfo.group);
}

std::set<uint64_t> SqlDeserializerV4::getTableGroups(const std::string& tableName) const
{
    std::set<uint64_t> groups;
    const auto         query = fmt::format("select DISTINCT GRAPH_GROUP from '{}'", tableName);

    exe(query, &groups, [](void* data, int argc, char** argv, char** colNames) {
        auto& ret = *static_cast<std::set<uint64_t>*>(data);
        ret.insert(std::stoull(argv[0]));
        return 0;
    });

    return groups;
}

std::vector<uint64_t> SqlDeserializerV4::getGroups() const
{
    const auto         graphs = getGraphsNames();
    std::set<uint64_t> groups;
    for (const auto& graph : graphs)
    {
        auto curr = getTableGroups(graph);
        groups.insert(curr.begin(), curr.end());
    }
    std::vector<uint64_t> ret(groups.begin(), groups.end());
    return ret;
}

std::vector<GraphInfo> SqlDeserializerV4::getGraphsInfos() const
{
    const auto graphs = getGraphsNames();

    using InfosSet = std::set<std::pair<std::string, uint64_t>>;

    InfosSet infosSet;

    for (const auto& graph : graphs)
    {
        const auto query = fmt::format("select DISTINCT GRAPH_GROUP from '{}'", graph);

        auto curr = getTableGroups(graph);
        for (const auto& group : curr)
        {
            infosSet.insert({graph, group});
        }
    }

    std::vector<GraphInfo> ret;
    for (const auto& i : infosSet)
    {
        ret.push_back({std::get<0>(i), uint16_t(-1), std::get<1>(i)});
    }
    return ret;
}

std::vector<GraphInfo> SqlDeserializerV4::getGraphsInfos(uint64_t group) const
{
    auto                   allGraphs = getGraphsInfos();
    std::vector<GraphInfo> ret;
    for (const auto& g : allGraphs)
    {
        if (g.group == group)
        {
            ret.push_back(g);
        }
    }
    return ret;
}

SqlDeserializerV5::SqlDeserializerV5(const std::string& dbName) : SqlDeserializerV4(dbName) {}

std::string SqlDeserializerV5::getTensorsNamesQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select NAME from '{}' where GRAPH_GROUP = '{}' and VALIDATION = {}",
                       graphInfo.id,
                       graphInfo.group,
                       int(TensorValidation::VALID));
}

std::string
SqlDeserializerV5::getDataTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format(
        "select DATA_TYPE from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}' and VALIDATION = {}",
        graphInfo.id,
        graphInfo.group,
        name,
        iteration,
        int(TensorValidation::VALID));
}

std::string
SqlDeserializerV5::getShapeSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select length(SHAPE) from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}' "
                       "and VALIDATION = {}",
                       graphInfo.id,
                       graphInfo.group,
                       name,
                       iteration,
                       int(TensorValidation::VALID));
}

std::string SqlDeserializerV5::getPermutationSizeQuery(const GraphInfo&   graphInfo,
                                                       const std::string& name,
                                                       uint64_t           iteration) const
{
    return fmt::format("select length(PERMUTATION) from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = "
                       "'{}' and VALIDATION = {}",
                       graphInfo.id,
                       graphInfo.group,
                       name,
                       iteration,
                       int(TensorValidation::VALID));
}

std::string SqlDeserializerV5::getCompressionTypeQuery(const GraphInfo&   graphInfo,
                                                       const std::string& name,
                                                       uint64_t           iteration) const
{
    return fmt::format("select COMPRESSION from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}' and "
                       "VALIDATION = {}",
                       graphInfo.id,
                       graphInfo.group,
                       name,
                       iteration,
                       int(TensorValidation::VALID));
}

std::string
SqlDeserializerV5::getDataIdQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format(
        "select DATA_ID from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}' and VALIDATION = {}",
        graphInfo.id,
        graphInfo.group,
        name,
        iteration,
        int(TensorValidation::VALID));
}

std::string
SqlDeserializerV5::getShapeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format(
        "select SHAPE from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}' and VALIDATION = {}",
        graphInfo.id,
        graphInfo.group,
        name,
        iteration,
        int(TensorValidation::VALID));
}

std::string
SqlDeserializerV5::getPermutationQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select PERMUTATION from '{}' where GRAPH_GROUP = '{}' and NAME = '{}' and ITERATION = '{}' and "
                       "VALIDATION = {}",
                       graphInfo.id,
                       graphInfo.group,
                       name,
                       iteration,
                       int(TensorValidation::VALID));
}

std::string SqlDeserializerV5::getDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format(
        "select ITERATION from '{}' where GRAPH_GROUP = '{}' and DATA_TYPE != 0 and DATA_ID != 0 and VALIDATION = {}",
        graphInfo.id,
        graphInfo.group,
        int(TensorValidation::VALID));
}

std::string SqlDeserializerV5::getNonDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format(
        "select ITERATION from '{}' where GRAPH_GROUP = '{}' and DATA_TYPE != 0 and DATA_ID == 0 and VALIDATION = {}",
        graphInfo.id,
        graphInfo.group,
        int(TensorValidation::VALID));
}

std::set<uint64_t> SqlDeserializerV5::getTableGroups(const std::string& tableName) const
{
    std::set<uint64_t> groups;
    const auto         query = fmt::format("select DISTINCT GRAPH_GROUP from '{}' where VALIDATION = {}",
                                   tableName,
                                   int(TensorValidation::VALID));

    exe(query, &groups, [](void* data, int argc, char** argv, char** colNames) {
        auto& ret = *static_cast<std::set<uint64_t>*>(data);
        ret.insert(std::stoull(argv[0]));
        return 0;
    });

    return groups;
}

SqlDeserializerV6::SqlDeserializerV6(const std::string& dbName) : SqlDeserializerV5(dbName) {}

std::string SqlDeserializerV6::getGraphsNamesQuery() const
{
    return fmt::format("select distinct GRAPH_NAME from '{}' where VALIDATION = {}",
                       TENSORS_TABLE,
                       int(TensorValidation::VALID));
}

std::string SqlDeserializerV6::getTensorsNamesQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select distinct NAME from '{}' where RECIPE_ID = {} and VALIDATION = {}",
                       TENSORS_TABLE,
                       graphInfo.recipeId,
                       int(TensorValidation::VALID));
}

std::string
SqlDeserializerV6::getDataTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select DATA_TYPE from '{}' where RECIPE_ID = {} and NAME = '{}' and "
                       "ITERATION = '{}' and VALIDATION = {}",
                       TENSORS_TABLE,
                       graphInfo.recipeId,
                       name,
                       iteration,
                       int(TensorValidation::VALID));
}

std::string
SqlDeserializerV6::getShapeSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select length(SHAPE) from '{}' where RECIPE_ID = {} and NAME = '{}' and "
                       "ITERATION = '{}' "
                       "and VALIDATION = {}",
                       TENSORS_TABLE,
                       graphInfo.recipeId,
                       name,
                       iteration,
                       int(TensorValidation::VALID));
}

std::string SqlDeserializerV6::getPermutationSizeQuery(const GraphInfo&   graphInfo,
                                                       const std::string& name,
                                                       uint64_t           iteration) const
{
    return fmt::format("select length(PERMUTATION) from '{}' where RECIPE_ID = {} and NAME = "
                       "'{}' and ITERATION = "
                       "'{}' and VALIDATION = {}",
                       TENSORS_TABLE,
                       graphInfo.recipeId,
                       name,
                       iteration,
                       int(TensorValidation::VALID));
}

std::string SqlDeserializerV6::getCompressionTypeQuery(const GraphInfo&   graphInfo,
                                                       const std::string& name,
                                                       uint64_t           iteration) const
{
    return fmt::format("select COMPRESSION from '{}' where RECIPE_ID = {} and NAME = '{}' and "
                       "ITERATION = '{}' and "
                       "VALIDATION = {}",
                       TENSORS_TABLE,
                       graphInfo.recipeId,
                       name,
                       iteration,
                       int(TensorValidation::VALID));
}

std::string
SqlDeserializerV6::getDataIdQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select DATA_ID from '{}' where RECIPE_ID = {} and NAME = '{}' and "
                       "ITERATION = '{}' and VALIDATION = {}",
                       TENSORS_TABLE,
                       graphInfo.recipeId,
                       name,
                       iteration,
                       int(TensorValidation::VALID));
}

std::string
SqlDeserializerV6::getShapeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select SHAPE from '{}' where RECIPE_ID = {} and NAME = '{}' and "
                       "ITERATION = '{}' and VALIDATION = {}",
                       TENSORS_TABLE,
                       graphInfo.recipeId,
                       name,
                       iteration,
                       int(TensorValidation::VALID));
}

std::string
SqlDeserializerV6::getPermutationQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const
{
    return fmt::format("select PERMUTATION from '{}' where RECIPE_ID = {} and NAME = '{}' and "
                       "ITERATION = '{}' and "
                       "VALIDATION = {}",
                       TENSORS_TABLE,
                       graphInfo.recipeId,
                       name,
                       iteration,
                       int(TensorValidation::VALID));
}

std::string SqlDeserializerV6::getDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from '{}' where RECIPE_ID = {} and DATA_TYPE != 0 and "
                       "DATA_ID != 0 and VALIDATION = {}",
                       TENSORS_TABLE,
                       graphInfo.recipeId,
                       int(TensorValidation::VALID));
}

std::string SqlDeserializerV6::getNonDataIterationsQuery(const GraphInfo& graphInfo) const
{
    return fmt::format("select ITERATION from '{}' where RECIPE_ID = {} and DATA_TYPE != 0 and "
                       "DATA_ID == 0 and VALIDATION = {}",
                       TENSORS_TABLE,
                       graphInfo.recipeId,
                       int(TensorValidation::VALID));
}

std::vector<uint64_t> SqlDeserializerV6::getGroups() const
{
    return {};
}

std::vector<GraphInfo> SqlDeserializerV6::getGraphsInfos() const
{
    using InfosSet = std::set<std::tuple<std::string, uint16_t, uint64_t>>;

    std::string list = fmt::format("select GRAPH_NAME,RECIPE_ID from {}", TENSORS_TABLE);

    InfosSet infosSet;

    exe(list, &infosSet, [](void* data, int argc, char** argv, char** colNames) {
        auto   infos = static_cast<InfosSet*>(data);
        size_t lines = argc / 2;
        for (size_t i = 0; i < lines; ++i)
        {
            infos->insert({argv[i + 0], static_cast<uint16_t>(std::stoul(argv[i + 1])), static_cast<uint64_t>(-1)});
        }
        return 0;
    });

    std::vector<GraphInfo> ret;
    for (const auto& i : infosSet)
    {
        ret.push_back({std::get<0>(i), std::get<1>(i), std::get<2>(i)});
    }
    return ret;
}

std::vector<GraphInfo> SqlDeserializerV6::getGraphsInfos(uint64_t group) const
{
    return {};
}

SqlDeserializerV7::SqlDeserializerV7(const std::string& dbName) : SqlDeserializerV6(dbName) {}

uint64_t SqlDeserializerV7::getStoredDataSize(int64_t dataId) const
{
    if (dataId == 0) return 0;

    std::string select = SqlDeserializer::getStoredDataSizeQuery(dataId);
    uint64_t    size   = 0;
    auto        sts    = exe(select, &size, singleIntCallback<uint64_t>);
    HB_ASSERT(sts == SQLITE_OK, "Failed to read stored data size, data ID: {}", dataId);
    return size;
}

void SqlDeserializerV7::getData(const GraphInfo&   graphInfo,
                                const std::string& name,
                                uint64_t           iteration,
                                uint8_t*           data,
                                TSize              dataSize) const
{
    auto dataId = getDataId(graphInfo, name, iteration);

    std::string select = SqlDeserializer::getDataQuery(dataId);

    TensorTableColumn tmd;
    tmd.data               = data;
    tmd.invalidData        = false;
    tmd.dataSize           = dataSize;
    tmd.compressedDataSize = getStoredDataSize(dataId);
    tmd.compression        = getCompressionType(dataId);
    exe(select, &tmd, tensorMetadataCallback);
}

SqlDeserializerV8::SqlDeserializerV8(const std::string& dbName) : SqlDeserializerV7(dbName) {}

std::vector<uint64_t> SqlDeserializerV8::getGroups() const
{
    std::vector<uint64_t> ret;

    const auto query = fmt::format("select DISTINCT GROUP_ID from '{}' where VALIDATION = {}",
                                   TENSORS_TABLE,
                                   int(TensorValidation::VALID));

    exe(query, &ret, [](void* data, int argc, char** argv, char** colNames) {
        auto& groups = *static_cast<std::vector<uint64_t>*>(data);
        for (int i = 0; i < argc; i++)
        {
            groups.push_back(std::stoull(argv[i]));
        }
        return 0;
    });

    return ret;
}

std::vector<GraphInfo> SqlDeserializerV8::getGraphsInfos() const
{
    using InfosSet = std::set<std::tuple<std::string, uint16_t, uint64_t>>;

    std::string list = fmt::format("select GRAPH_NAME,RECIPE_ID,GROUP_ID from {}", TENSORS_TABLE);

    InfosSet infosSet;

    int                    res = exe(list, &infosSet, [](void* data, int argc, char** argv, char** colNames) {
        auto   infos = static_cast<InfosSet*>(data);
        size_t lines = argc / 3;
        for (size_t i = 0; i < lines; ++i)
        {
            infos->insert({argv[i + 0], static_cast<uint16_t>(std::stoul(argv[i + 1])), std::stoull(argv[i + 2])});
        }
        return 0;
    });
    std::vector<GraphInfo> ret;
    for (const auto& i : infosSet)
    {
        ret.push_back({std::get<0>(i), std::get<1>(i), std::get<2>(i)});
    }
    return ret;
}

std::vector<GraphInfo> SqlDeserializerV8::getGraphsInfos(uint64_t group) const
{
    using InfosSet = std::set<std::pair<std::string, uint16_t>>;

    std::string list = fmt::format("select GRAPH_NAME,RECIPE_ID from {} where GROUP_ID = {}", TENSORS_TABLE, group);

    InfosSet infosSet;

    int res = exe(list, &infosSet, [](void* data, int argc, char** argv, char** colNames) {
        auto   infos = static_cast<InfosSet*>(data);
        size_t lines = argc / 2;
        for (size_t i = 0; i < lines; ++i)
        {
            infos->insert({argv[i + 0], static_cast<uint16_t>(std::stoul(argv[i + 1]))});
        }
        return 0;
    });

    std::vector<GraphInfo> ret;
    for (const auto& i : infosSet)
    {
        ret.push_back({i.first, i.second, group});
    }
    return ret;
}
}  // namespace data_serialize
