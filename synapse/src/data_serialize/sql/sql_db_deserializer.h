#pragma once

#include "data_serializer/ds_types.h"
#include "sql_db.h"
#include "synapse_common_types.h"
#include <memory>
#include <set>

struct sqlite3;

namespace data_serialize
{
class SqlDeserializer : public Sql
{
public:
    static std::unique_ptr<SqlDeserializer> createSqlDeserializer(const std::string& dbName);

    virtual synDataType getDataType(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual std::vector<TSize> getShape(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual std::vector<uint8_t>
                     getPermutation(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual uint64_t getDataSize(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;

    virtual std::vector<uint64_t>  getGroups() const;
    virtual std::vector<GraphInfo> getGraphsInfos() const;
    virtual std::vector<GraphInfo> getGraphsInfos(uint64_t group) const;

    virtual std::vector<std::string> getTensorsNames(const GraphInfo& graphInfo) const;
    virtual std::set<uint64_t>       getDataIterations(const GraphInfo& graphInfo) const;
    virtual std::set<uint64_t>       getNonDataIterations(const GraphInfo& graphInfo) const;

    virtual std::vector<uint8_t> getData(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual void                 getData(const GraphInfo&   graphInfo,
                                         const std::string& name,
                                         uint64_t           iteration,
                                         uint8_t*           data,
                                         TSize              dataSize) const;

protected:
    SqlDeserializer(const std::string& dbName);
    std::vector<std::string> getGraphsNames() const;

    static void decompress(const char* src, int srcSize, char* dst, int dstSize);
    static int  tensorMetadataCallback(void* data, int argc, char** argv, char** colNames);

    virtual uint64_t getStoredDataSize(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual uint64_t getStoredDataSize(int64_t dataId) const;
    virtual int64_t  getDataId(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual uint64_t getShapeSize(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual uint8_t  getDimSizeInBytes() const;
    virtual uint64_t getPermutationSize(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;

    virtual Compression
    getCompressionType(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual Compression getCompressionType(int64_t dataId) const;

    virtual std::string getGraphsNamesQuery() const;
    virtual std::string getTensorsNamesQuery(const GraphInfo& graphInfo) const;
    virtual std::string getDataTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual std::string
    getShapeSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual std::string
    getPermutationSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual std::string
    getCompressionTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual std::string getCompressionTypeQuery(int64_t dataId) const;
    virtual std::string getDataIdQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual std::string getShapeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual std::string
    getPermutationQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual std::string getDataQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual std::string getDataQuery(int64_t dataId) const;
    virtual std::string
    getStoredDataSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const;
    virtual std::string getStoredDataSizeQuery(int64_t dataId) const;
    virtual std::string getDataIterationsQuery(const GraphInfo& graphInfo) const;
    virtual std::string getNonDataIterationsQuery(const GraphInfo& graphInfo) const;
};

class SqlDeserializerV0 : public SqlDeserializer
{
public:
    SqlDeserializerV0(const std::string& dbName);

protected:
    std::string getDataQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getStoredDataSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getDataIterationsQuery(const GraphInfo& graphInfo) const override;
    std::string getNonDataIterationsQuery(const GraphInfo& graphInfo) const override;
};

class SqlDeserializerV1 : public SqlDeserializer
{
public:
    SqlDeserializerV1(const std::string& dbName);

protected:
    std::string getDataQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getStoredDataSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getDataIterationsQuery(const GraphInfo& graphInfo) const override;
    std::string getNonDataIterationsQuery(const GraphInfo& graphInfo) const override;
};

class SqlDeserializerV2 : public SqlDeserializerV1
{
public:
    SqlDeserializerV2(const std::string& dbName);

protected:
    std::string getTensorsNamesQuery(const GraphInfo& graphInfo) const override;
    std::string
    getDataTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getShapeSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getPermutationSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getCompressionTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getDataIdQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getShapeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getPermutationQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getDataQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getStoredDataSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getDataIterationsQuery(const GraphInfo& graphInfo) const override;
    std::string getNonDataIterationsQuery(const GraphInfo& graphInfo) const override;
};

class SqlDeserializerV3 : public SqlDeserializerV2
{
public:
    SqlDeserializerV3(const std::string& dbName);

protected:
    uint8_t getDimSizeInBytes() const override;
};

class SqlDeserializerV4 : public SqlDeserializerV3
{
public:
    SqlDeserializerV4(const std::string& dbName);

    std::string getTensorsNamesQuery(const GraphInfo& graphInfo) const override;
    std::string
    getDataTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getShapeSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getPermutationSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getCompressionTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getDataIdQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getShapeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getPermutationQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getDataIterationsQuery(const GraphInfo& graphInfo) const override;
    std::string getNonDataIterationsQuery(const GraphInfo& graphInfo) const override;

    std::vector<uint64_t>  getGroups() const override;
    std::vector<GraphInfo> getGraphsInfos() const override;
    std::vector<GraphInfo> getGraphsInfos(uint64_t group) const override;

protected:
    std::set<uint64_t> getTableGroups(const std::string& tableName) const;
};

class SqlDeserializerV5 : public SqlDeserializerV4
{
public:
    SqlDeserializerV5(const std::string& dbName);

    std::string getTensorsNamesQuery(const GraphInfo& graphInfo) const override;
    std::string
    getDataTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getShapeSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getPermutationSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getCompressionTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getDataIdQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getShapeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getPermutationQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getDataIterationsQuery(const GraphInfo& graphInfo) const override;
    std::string getNonDataIterationsQuery(const GraphInfo& graphInfo) const override;

protected:
    std::set<uint64_t> getTableGroups(const std::string& tableName) const;
};

class SqlDeserializerV6 : public SqlDeserializerV5
{
public:
    SqlDeserializerV6(const std::string& dbName);

    std::string getGraphsNamesQuery() const override;
    std::string getTensorsNamesQuery(const GraphInfo& graphInfo) const override;
    std::string
    getDataTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getShapeSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getPermutationSizeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getCompressionTypeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getDataIdQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string getShapeQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string
    getPermutationQuery(const GraphInfo& graphInfo, const std::string& name, uint64_t iteration) const override;
    std::string            getDataIterationsQuery(const GraphInfo& graphInfo) const override;
    std::string            getNonDataIterationsQuery(const GraphInfo& graphInfo) const override;
    std::vector<uint64_t>  getGroups() const override;
    std::vector<GraphInfo> getGraphsInfos() const override;
    std::vector<GraphInfo> getGraphsInfos(uint64_t group) const override;
};

class SqlDeserializerV7 : public SqlDeserializerV6
{
public:
    SqlDeserializerV7(const std::string& dbName);

    uint64_t getStoredDataSize(int64_t dataId) const override;
    void     getData(const GraphInfo&   graphInfo,
                     const std::string& name,
                     uint64_t           iteration,
                     uint8_t*           data,
                     TSize              dataSize) const override;
};

class SqlDeserializerV8 : public SqlDeserializerV7
{
public:
    SqlDeserializerV8(const std::string& dbName);

    std::vector<uint64_t>  getGroups() const override;
    std::vector<GraphInfo> getGraphsInfos() const override;
    std::vector<GraphInfo> getGraphsInfos(uint64_t group) const override;
};
}  // namespace data_serialize