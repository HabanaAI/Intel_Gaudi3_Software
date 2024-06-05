#pragma once

#include "data_container.h"
#include "data_serializer/data_serializer.h"
#include "defs.h"
#include "hpp/syn_graph.hpp"
#include "hpp/syn_host_buffer.hpp"
#include "hpp/syn_tensor.hpp"
#include "mme_reference/data_types/non_standard_dtypes.h"
#include "synapse_common_types.h"
#include "test_types.hpp"
#include "test_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class ConstantDataGenerator
{
public:
    ConstantDataGenerator(float value) : m_value(value) {}
    template<typename T>
    T getVal()
    {
        return static_cast<T>(m_value);
    }

private:
    float m_value;
};

class RandomDataGenerator
{
public:
    RandomDataGenerator(const float minVal, const float maxVal, size_t seed) : m_dist(minVal, maxVal)
    {
        m_eng.seed(seed);
    }

    template<typename T>
    T getVal()
    {
        return static_cast<T>(m_dist(m_eng));
    }

private:
    std::default_random_engine            m_eng {};
    std::uniform_real_distribution<float> m_dist;
};

class DataProvider : public DataContainer
{
public:
    virtual ~DataProvider() = default;

    virtual void               copyBuffer(const std::string& tensorName, syn::HostBuffer buffer) const = 0;
    virtual uint64_t           getBufferSize(const std::string& tensorName) const                      = 0;
    virtual std::set<uint64_t> getDataIterations() const                                               = 0;
    virtual std::set<uint64_t> getNonDataIterations() const                                            = 0;
    virtual void               setDataIteration(uint64_t iteration)                                    = 0;

    virtual std::set<uint64_t> getAllDataIterations() const
    {
        auto ret = getDataIterations();
        ret.merge(getNonDataIterations());
        return ret;
    }

    template<typename T, typename UnaryOp>
    void generatBuffer(const size_t elementsCount, UnaryOp valGen, uint8_t* buffer, TSize bufferSize) const
    {
        TSize minBufferSize = elementsCount * sizeof(T);
        HB_ASSERT(bufferSize >= elementsCount * sizeof(T),
                  "failed to generate buffer, insufficient buffer size, required: {}, actual: {}",
                  minBufferSize,
                  bufferSize);
        for (size_t i = 0; i < elementsCount; ++i)
        {
            T v = valGen.template getVal<T>();
            memcpy(buffer + i * sizeof(T), &v, sizeof(T));
        }
    }

    template<typename T, typename UnaryOp>
    std::vector<uint8_t> generatBuffer(const size_t elementsCount, UnaryOp valGen) const
    {
        std::vector<uint8_t> res(elementsCount * sizeof(T));
        generatBuffer<T>(elementsCount, valGen, res.data(), res.size());
        return res;
    }

    template<typename T>
    std::vector<uint8_t> generatBuffer(uint64_t elementsCount, synDataType dataType, T getValue) const
    {
        switch (dataType)
        {
            case syn_type_bf16:
                return generatBuffer<bfloat16>(elementsCount, getValue);
            case syn_type_single:
                return generatBuffer<float>(elementsCount, getValue);
            case syn_type_int8:
                return generatBuffer<int8_t>(elementsCount, getValue);
            case syn_type_int16:
                return generatBuffer<int16_t>(elementsCount, getValue);
            case syn_type_int32:
                return generatBuffer<int32_t>(elementsCount, getValue);
            case syn_type_uint8:
                return generatBuffer<uint8_t>(elementsCount, getValue);
            case syn_type_uint16:
                return generatBuffer<uint16_t>(elementsCount, getValue);
            case syn_type_uint32:
                return generatBuffer<uint32_t>(elementsCount, getValue);
            case syn_type_int64:
                return generatBuffer<int64_t>(elementsCount, getValue);
            case syn_type_uint64:
                return generatBuffer<uint64_t>(elementsCount, getValue);
            case syn_type_fp16:
                return generatBuffer<fp16_t>(elementsCount, getValue);
            case syn_type_na:
            case syn_type_fp8_152:
            case syn_type_int4:
            case syn_type_uint4:
            case syn_type_tf32:
            case syn_type_hb_float:
            case syn_type_fp8_143:
            case syn_type_ufp16:
            case syn_type_max:
                break;
        }
        throw std::runtime_error("DataProvider, unsupported data type");
    }

    template<typename T>
    void
    generatBuffer(uint64_t elementsCount, synDataType dataType, T getValue, uint8_t* buffer, TSize bufferSize) const
    {
        switch (dataType)
        {
            case syn_type_bf16:
                generatBuffer<bfloat16>(elementsCount, getValue, buffer, bufferSize);
                break;
            case syn_type_single:
                generatBuffer<float>(elementsCount, getValue, buffer, bufferSize);
                break;
            case syn_type_int8:
                generatBuffer<int8_t>(elementsCount, getValue, buffer, bufferSize);
                break;
            case syn_type_int16:
                generatBuffer<int16_t>(elementsCount, getValue, buffer, bufferSize);
                break;
            case syn_type_int32:
                generatBuffer<int32_t>(elementsCount, getValue, buffer, bufferSize);
                break;
            case syn_type_uint8:
                generatBuffer<uint8_t>(elementsCount, getValue, buffer, bufferSize);
                break;
            case syn_type_uint16:
                generatBuffer<uint16_t>(elementsCount, getValue, buffer, bufferSize);
                break;
            case syn_type_uint32:
                generatBuffer<uint32_t>(elementsCount, getValue, buffer, bufferSize);
                break;
            case syn_type_int64:
                generatBuffer<int64_t>(elementsCount, getValue, buffer, bufferSize);
                break;
            case syn_type_uint64:
                generatBuffer<uint64_t>(elementsCount, getValue, buffer, bufferSize);
                break;
            case syn_type_fp16:
                generatBuffer<fp16_t>(elementsCount, getValue, buffer, bufferSize);
                break;
            case syn_type_na:
            case syn_type_fp8_152:
            case syn_type_int4:
            case syn_type_uint4:
            case syn_type_tf32:
            case syn_type_hb_float:
            case syn_type_fp8_143:
            case syn_type_ufp16:
            case syn_type_max:
                throw std::runtime_error("DataProvider, unsupported data type");
        }
    }
};

class SyntheticDataProvider : public DataProvider
{
public:
    std::vector<uint8_t> getBuffer(const std::string& tensorName) const override
    {
        syn::Tensor tensor   = m_tensors.at(tensorName);
        auto        elements = tensor.getSizeInElements();
        auto        dataType = tensor.getDataType();
        return generateData(elements, dataType, tensorName);
    }

    uint64_t getBufferSize(const std::string& tensorName) const override
    {
        syn::Tensor tensor = m_tensors.at(tensorName);
        return tensor.getSizeInBytes();
    }

    void copyBuffer(const std::string& tensorName, syn::HostBuffer buffer) const override
    {
        syn::Tensor tensor   = m_tensors.at(tensorName);
        auto        elements = tensor.getSizeInElements();
        auto        dataType = tensor.getDataType();
        generateData(elements, dataType, tensorName, buffer.getAs<uint8_t>(), buffer.getSize());
    }

    std::vector<TSize> getShape(const std::string& tensorName) const override
    {
        auto tensor   = m_tensors.at(tensorName);
        auto geometry = tensor.getGeometryExt(synGeometryMaxSizes);
        return std::vector<TSize>(geometry.sizes, geometry.sizes + geometry.dims);
    }

    std::vector<uint8_t> getPermutation(const std::string& tensorName) const override
    {
        auto tensor      = m_tensors.at(tensorName);
        auto permutation = tensor.getPermutation();
        return std::vector<uint8_t>(permutation.permutation, permutation.permutation + permutation.dims);
    }

    synDataType getDataType(const std::string& tensorName) const override
    {
        auto tensor = m_tensors.at(tensorName);
        return tensor.getDataType();
    }

    std::vector<std::string> getTensorsNames() const override
    {
        std::vector<std::string> tensorsNames;
        tensorsNames.reserve(m_tensors.size());

        for (const auto& t : m_tensors)
        {
            tensorsNames.emplace_back(t.first);
        }
        return tensorsNames;
    }

    std::set<uint64_t> getDataIterations() const override { return {}; }

    std::set<uint64_t> getNonDataIterations() const override { return {0}; }

    void setDataIteration(uint64_t iteration) override {}

    virtual std::vector<uint8_t>
    generateData(uint64_t elementsCount, synDataType dataType, std::string_view bufferName) const = 0;

    virtual void generateData(uint64_t         elementsCount,
                              synDataType      dataType,
                              std::string_view bufferName,
                              uint8_t*         buffer,
                              TSize            bufferSize) const = 0;

protected:
    SyntheticDataProvider(const syn::Tensors& tensors)
    {
        for (const auto& t : tensors)
        {
            m_tensors.emplace(t.getName(), t);
        }
    }

    SyntheticDataProvider(const std::map<std::string, syn::Tensor>& tensors) : m_tensors(tensors) {}

    std::map<std::string, syn::Tensor> m_tensors;
};

class UniformDataProvider : public SyntheticDataProvider
{
public:
    UniformDataProvider(const float value, const std::map<std::string, syn::Tensor>& tensors)
    : SyntheticDataProvider(tensors), m_dataGenerator(value)
    {
    }

    UniformDataProvider(const float value, const syn::Tensors& tensors)
    : SyntheticDataProvider(tensors), m_dataGenerator(value)
    {
    }

    virtual std::vector<uint8_t>
    generateData(uint64_t elementsCount, synDataType dataType, std::string_view bufferName) const
    {
        return generatBuffer(elementsCount, dataType, m_dataGenerator);
    }

    virtual void generateData(uint64_t         elementsCount,
                              synDataType      dataType,
                              std::string_view bufferName,
                              uint8_t*         buffer,
                              TSize            bufferSize) const
    {
        generatBuffer(elementsCount, dataType, m_dataGenerator, buffer, bufferSize);
    }

private:
    ConstantDataGenerator m_dataGenerator;
};

class RandDataProvider : public SyntheticDataProvider
{
public:
    RandDataProvider(const float minVal, const float maxVal, const std::map<std::string, syn::Tensor>& tensors)
    : SyntheticDataProvider(tensors), m_minVal(minVal), m_maxVal(maxVal)
    {
    }

    RandDataProvider(const float minVal, const float maxVal, const syn::Tensors& tensors)
    : SyntheticDataProvider(tensors), m_minVal(minVal), m_maxVal(maxVal)
    {
    }

    virtual std::vector<uint8_t>
    generateData(uint64_t elementsCount, synDataType dataType, std::string_view bufferName) const
    {
        RandomDataGenerator dataGenerator(m_minVal, m_maxVal, std::hash<std::string_view>()(bufferName));
        return generatBuffer(elementsCount, dataType, dataGenerator);
    }

    virtual void generateData(uint64_t         elementsCount,
                              synDataType      dataType,
                              std::string_view bufferName,
                              uint8_t*         buffer,
                              TSize            bufferSize) const
    {
        RandomDataGenerator dataGenerator(m_minVal, m_maxVal, std::hash<std::string_view>()(bufferName));
        generatBuffer(elementsCount, dataType, dataGenerator, buffer, bufferSize);
    }

private:
    const float m_minVal;
    const float m_maxVal;
};

class CapturedDataProvider : public DataProvider
{
public:
    CapturedDataProvider(const std::string& filePath,
                         const std::string& graphName,
                         uint16_t           recipeId,
                         uint64_t           group,
                         uint64_t           iteration = 0)
    : m_dataDeserializer(data_serialize::DataDeserializer::create(filePath)->getGraph({graphName, recipeId, group})),
      m_iteration(iteration),
      m_dataIterations(m_dataDeserializer->getDataIterations()),
      m_nonDataIterations(m_dataDeserializer->getNonDataIterations())
    {
        if (m_dataIterations.empty() && m_nonDataIterations.empty())
        {
            throw std::runtime_error(
                fmt::format("Missing graph data, file: {}, graph name: {}, recipe ID: {}, group: {}, iteration: {}",
                            filePath,
                            graphName,
                            recipeId,
                            group,
                            iteration));
        }
    }

    std::vector<std::string> getTensorsNames() const override { return m_dataDeserializer->getTensorsNames(); }

    std::vector<uint8_t> getBuffer(const std::string& tensorName) const override
    {
        return m_dataDeserializer->getData(tensorName, m_iteration);
    }

    uint64_t getBufferSize(const std::string& tensorName) const override
    {
        return m_dataDeserializer->getDataSize(tensorName, m_iteration);
    }

    void copyBuffer(const std::string& tensorName, syn::HostBuffer buffer) const override
    {
        m_dataDeserializer->getData(tensorName, m_iteration, buffer.getAs<uint8_t>(), buffer.getSize());
    }

    std::vector<TSize> getShape(const std::string& tensorName) const override
    {
        return m_dataDeserializer->getShape(tensorName, m_iteration);
    }

    std::vector<uint8_t> getPermutation(const std::string& tensorName) const override
    {
        std::vector<uint8_t> permutation = m_dataDeserializer->getPermutation(tensorName, m_iteration);
        std::vector<TSize>   shape       = getShape(tensorName);

        size_t index = permutation.size();
        permutation.resize(shape.size());
        for (; index < permutation.size(); ++index)
        {
            permutation[index] = index;
        }
        return permutation;
    }

    synDataType getDataType(const std::string& tensorName) const override
    {
        return m_dataDeserializer->getDataType(tensorName, m_iteration);
    }

    std::set<uint64_t> getDataIterations() const override { return m_dataIterations; }

    std::set<uint64_t> getNonDataIterations() const override { return m_nonDataIterations; }

    void setDataIteration(uint64_t iteration) override { m_iteration = iteration; }

private:
    data_serialize::GraphDataDeserializerPtr m_dataDeserializer;
    uint64_t                                 m_iteration;
    std::set<uint64_t>                       m_dataIterations;
    std::set<uint64_t>                       m_nonDataIterations;
};

class ManualDataProvider : public DataProvider
{
public:
    ManualDataProvider() {}

    ManualDataProvider(const syn::Tensors& tensors)
    {
        for (const auto& t : tensors)
        {
            m_tensors.emplace(t.getName(), t);
        }
    }

    void setTensor(const syn::Tensor& tensor, MemInitType memInitType, const std::vector<uint8_t>& buffer = {})
    {
        m_tensors.emplace(tensor.getName(), tensor);

        switch (memInitType)
        {
            case MEM_INIT_RANDOM_WITH_NEGATIVE:
            {
                RandomDataGenerator dataGenerator(-2.0, 2.0, std::hash<std::string_view>()(tensor.getName()));
                uint64_t elementCount = tensor.getSizeInBytes() / tensor.getElementSizeInBytes(tensor.getDataType());
                m_tensorsBuffers[tensor.getName()] = generatBuffer(elementCount, tensor.getDataType(), dataGenerator);
                break;
            }
            case MEM_INIT_RANDOM_POSITIVE:
            {
                RandomDataGenerator dataGenerator(0, 2.0, std::hash<std::string_view>()(tensor.getName()));
                uint64_t elementCount = tensor.getSizeInBytes() / tensor.getElementSizeInBytes(tensor.getDataType());
                m_tensorsBuffers[tensor.getName()] = generatBuffer(elementCount, tensor.getDataType(), dataGenerator);
                break;
            }
            case MEM_INIT_ALL_ONES:
            {
                ConstantDataGenerator dataGenerator(1);
                uint64_t elementCount = tensor.getSizeInBytes() / tensor.getElementSizeInBytes(tensor.getDataType());
                m_tensorsBuffers[tensor.getName()] = generatBuffer(elementCount, tensor.getDataType(), dataGenerator);
                break;
            }
            case MEM_INIT_ALL_ZERO:
            {
                ConstantDataGenerator dataGenerator(0);
                uint64_t elementCount = tensor.getSizeInBytes() / tensor.getElementSizeInBytes(tensor.getDataType());
                m_tensorsBuffers[tensor.getName()] = generatBuffer(elementCount, tensor.getDataType(), dataGenerator);
                break;
            }
            case MEM_INIT_FROM_INITIALIZER:
            case MEM_INIT_FROM_INITIALIZER_NO_CAST:
                setBuffer(tensor, buffer);
                break;
            case MEM_INIT_NONE:
                m_tensorsBuffers[tensor.getName()] = {};
                break;

            case MEM_INIT_RANDOM_WITH_NEGATIVE_ONLY:
            case MEM_INIT_COMPILATION_ONLY:
                HB_ASSERT(false, "Not supported");
                break;
        }
    }

    template<typename T>
    void setBuffer(const syn::Tensor& tensor, const std::vector<T>& buffer)
    {
        std::vector<uint8_t> copy(tensor.getSizeInBytes());
        std::memcpy(copy.data(), buffer.data(), std::min(copy.size(), buffer.size() * sizeof(T)));
        m_tensorsBuffers[tensor.getName()] = copy;
    }

    std::vector<uint8_t> getBuffer(const std::string& tensorName) const override
    {
        return getTensorData(tensorName, m_tensorsBuffers);
    }

    uint64_t getBufferSize(const std::string& tensorName) const override
    {
        return getTensorData(tensorName, m_tensorsBuffers).size();
    }

    void copyBuffer(const std::string& tensorName, syn::HostBuffer hostBuffer) const override
    {
        const auto& buffer = getTensorData(tensorName, m_tensorsBuffers);
        std::memcpy(hostBuffer.get(), buffer.data(), hostBuffer.getSize());
    }

    void setShape(const std::string& tensorName, const std::vector<TSize>& shape)
    {
        m_tensorsShapes[tensorName] = shape;
    }

    std::vector<TSize> getShape(const std::string& tensorName) const override
    {
        if (tensorDataExists(tensorName, m_tensorsShapes))
        {
            return getTensorData(tensorName, m_tensorsShapes);
        }
        auto tensor   = getTensorData(tensorName, m_tensors);
        auto geometry = tensor.getGeometryExt(synGeometryMaxSizes);
        return std::vector<TSize>(geometry.sizes, geometry.sizes + geometry.dims);
    }

    void setPermutation(const std::string& tensorName, const std::vector<uint8_t>& permutation)
    {
        m_tensorsPermutations[tensorName] = permutation;
    }

    std::vector<uint8_t> getPermutation(const std::string& tensorName) const override
    {
        if (tensorDataExists(tensorName, m_tensorsPermutations))
        {
            return getTensorData(tensorName, m_tensorsPermutations);
        }
        auto tensor      = getTensorData(tensorName, m_tensors);
        auto permutation = tensor.getPermutation();
        return std::vector<uint8_t>(permutation.permutation, permutation.permutation + permutation.dims);
    }

    synDataType getDataType(const std::string& tensorName) const override
    {
        auto tensor = getTensorData(tensorName, m_tensors);
        return tensor.getDataType();
    }

    std::vector<std::string> getTensorsNames() const override
    {
        std::vector<std::string> tensorsNames;
        tensorsNames.reserve(m_tensors.size());

        for (const auto& t : m_tensors)
        {
            tensorsNames.emplace_back(t.first);
        }
        return tensorsNames;
    }

    std::set<uint64_t> getDataIterations() const override { return {0}; }
    std::set<uint64_t> getNonDataIterations() const override { return {0}; }

    void setDataIteration(uint64_t iteration) override {}

private:
    template<class T>
    const T& getTensorData(const std::string& tensorName, const std::map<std::string, T> map) const
    {
        auto it = map.find(tensorName);
        if (it == map.end())
        {
            throw std::runtime_error(fmt::format("ManualDataProvider, missing data for tensor: {}", tensorName));
        }
        return it->second;
    }

    template<class T>
    bool tensorDataExists(const std::string& tensorName, const std::map<std::string, T> map) const
    {
        return map.find(tensorName) != map.end();
    }

    uint64_t getElementsCount(const std::string& tensorName) const
    {
        auto tensor = getTensorData(tensorName, m_tensors);
        return tensor.getSizeInElements();
    }

    std::map<std::string, syn::Tensor>          m_tensors;
    std::map<std::string, std::vector<uint8_t>> m_tensorsBuffers;
    std::map<std::string, std::vector<TSize>>   m_tensorsShapes;
    std::map<std::string, std::vector<uint8_t>> m_tensorsPermutations;
};
