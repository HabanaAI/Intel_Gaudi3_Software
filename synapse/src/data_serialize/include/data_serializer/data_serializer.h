#pragma once

#include "data_serializer/ds_types.h"
#include <map>
#include <set>
#include <string>

namespace data_serialize
{
class DataSerializer;
class DataDeserializer;
class GraphDataDeserializer;

using DataSerializerPtr        = std::shared_ptr<data_serialize::DataSerializer>;
using DataDeserializerPtr      = std::shared_ptr<data_serialize::DataDeserializer>;
using GraphDataDeserializerPtr = std::shared_ptr<data_serialize::GraphDataDeserializer>;

class DataSerializer
{
public:
    static DataSerializerPtr create(const std::string& filePath, const GraphInfo& graphInfo);

    virtual ~DataSerializer() = default;

    virtual size_t           serialize(TensorMetadata& tensorsData)           = 0;
    virtual void             updateData(TensorMetadata& tensorsData)          = 0;
    virtual void             updateRecipeId(uint16_t recipeId, size_t index)  = 0;
    virtual uint64_t         getIterationCount() const                        = 0;
    virtual const GraphInfo& getInfo()                                        = 0;
    virtual void             removePrevIterations(size_t numIterationsToKeep) = 0;
};

class GraphDataDeserializer
{
public:
    virtual ~GraphDataDeserializer() = default;

    virtual std::vector<std::string> getTensorsNames() const                                                     = 0;
    virtual std::vector<TSize>       getShape(const std::string& tensorName, uint64_t iteration) const           = 0;
    virtual std::vector<uint8_t>     getPermutation(const std::string& tensorName, uint64_t iteration) const     = 0;
    virtual std::set<uint64_t>       getDataIterations() const                                                   = 0;
    virtual std::set<uint64_t>       getNonDataIterations() const                                                = 0;
    virtual synDataType              getDataType(const std::string& tensorName, uint64_t iteration) const        = 0;
    virtual uint64_t                 getDataSize(const std::string& tensorName, uint64_t iteration) const        = 0;
    virtual std::vector<uint8_t>     getData(const std::string& tensorName, uint64_t iteration) const            = 0;
    virtual void getData(const std::string& tensorName, uint64_t iteration, uint8_t* data, TSize dataSize) const = 0;
};

class DataDeserializer
{
public:
    static DataDeserializerPtr create(const std::string& filePath);

    virtual ~DataDeserializer() = default;

    virtual std::vector<GraphInfo>   getGraphsInfos() const                     = 0;
    virtual std::vector<GraphInfo>   getGraphsInfos(uint64_t group) const       = 0;
    virtual std::vector<uint64_t>    getGroups() const                          = 0;
    virtual GraphDataDeserializerPtr getGraph(const GraphInfo& graphInfo) const = 0;
};
}  // namespace data_serialize
