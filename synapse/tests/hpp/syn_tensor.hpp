#pragma once

#include "syn_object.hpp"
#include "syn_section.hpp"
#include "syn_host_buffer.hpp"
#include "syn_quantization.hpp"
#include "synapse_common_types.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

namespace syn
{
class Tensor;
using Tensors = std::vector<Tensor>;

class Tensor : public SynObject<synTensor>
{
public:
    Tensor() = default;

    static uint32_t getElementSizeInBytes(synDataType type)
    {
        switch (type)
        {
            case syn_type_int8:
            case syn_type_uint8:
            case syn_type_fp8_143:
            case syn_type_fp8_152:
                return sizeof(char);
            case syn_type_bf16:
            case syn_type_int16:
            case syn_type_uint16:
            case syn_type_fp16:
            case syn_type_ufp16:
                return sizeof(int16_t);
            case syn_type_float:
            case syn_type_hb_float:
                return sizeof(float);
            case syn_type_int32:
            case syn_type_uint32:
                return sizeof(int32_t);
            case syn_type_int64:
            case syn_type_uint64:
                return sizeof(int64_t);
            case syn_type_int4:
            case syn_type_uint4:
                return 1;
            case syn_type_na:
            case syn_type_tf32:
            case syn_type_max:
                SYN_CHECK(synInvalidArgument);
        }
        SYN_CHECK(synInvalidArgument);
        return 0;
    }

    template<class T>
    static uint64_t getSizeInElements(const T shape[HABANA_DIM_MAX], uint32_t dims)
    {
        uint64_t tensorSize = 1;
        for (size_t d = 0; d < dims; ++d)
        {
            tensorSize *= shape[d];
        }
        return tensorSize;
    }

    template<class T>
    static uint64_t getSizeInElements(const std::vector<T>& shape)
    {
        uint64_t tensorSize = 1;
        for (size_t d = 0; d < shape.size(); ++d)
        {
            tensorSize *= shape[d];
        }
        return tensorSize;
    }

    static uint64_t getMaxSizeInElements(const synRetrievedLaunchTensorInfo& info)
    {
        return getSizeInElements(info.tensorMaxSize, info.tensorDims);
    }

    static uint64_t getMinSizeInElements(const synRetrievedLaunchTensorInfo& info)
    {
        return getSizeInElements(info.tensorMinSize, info.tensorDims);
    }

    static uint64_t getMaxSizeInElements(const synRetrievedLaunchTensorInfoExt& info)
    {
        return getSizeInElements(info.tensorMaxSize, info.tensorDims);
    }

    static uint64_t getMinSizeInElements(const synRetrievedLaunchTensorInfoExt& info)
    {
        return getSizeInElements(info.tensorMinSize, info.tensorDims);
    }

    static uint64_t getSizeInBytes(const synTensorGeometry&         geometry,
                                   const synTensorDeviceFullLayout& layout,
                                   const synDataType                dataType)
    {
        synTensorGeometryExt geometryExt;
        geometryExt.dims = geometry.dims;
        std::copy(geometry.sizes, geometry.sizes + HABANA_DIM_MAX, geometryExt.sizes);
        return getSizeInBytes(geometryExt, layout, dataType);
    }

    // TODO: is prodValue needed? for now, preserving existing behavior
    static uint64_t getSizeInBytes(const synTensorGeometryExt&      geometry,
                                   const synTensorDeviceFullLayout& layout,
                                   const synDataType                dataType)
    {
        uint64_t maxStride = 0;                          // Track max-size in case of slices
        uint64_t prodValue = geometry.dims > 0 ? 1 : 0;  // Track dims-prod in case of all 0 layouts
        for (uint32_t i = 0; i < geometry.dims; ++i)
        {
            const TSize dim = geometry.sizes[i];
            if (dim == 0) return 0;  // In case of ZeroSizedTensor

            maxStride = std::max(maxStride, layout.strides[i] * dim);
            prodValue *= dim;
        }

        return maxStride != 0 ? maxStride : prodValue * getElementSizeInBytes(dataType);
    }

    static uint64_t getMaxSizeInBytes(const synRetrievedLaunchTensorInfo& info)
    {
        return getMaxSizeInElements(info) * getElementSizeInBytes(synDataType(info.tensorDataType));
    }

    static uint64_t getMinSizeInBytes(const synRetrievedLaunchTensorInfo& info)
    {
        return getMinSizeInElements(info) * getElementSizeInBytes(synDataType(info.tensorDataType));
    }

    static uint64_t getMaxSizeInBytes(const synRetrievedLaunchTensorInfoExt& info)
    {
        return getMaxSizeInElements(info) * getElementSizeInBytes(synDataType(info.tensorDataType));
    }

    static uint64_t getMinSizeInBytes(const synRetrievedLaunchTensorInfoExt& info)
    {
        return getMinSizeInElements(info) * getElementSizeInBytes(synDataType(info.tensorDataType));
    }

    void assignToSection(Section Section, uint64_t byteOffset)
    {
        SYN_CHECK(synTensorAssignToSection(handle(), Section.handle(), byteOffset));
        m_section = Section;
    }

    void setQuantizationData(synQuantizationProperty prop, void* propVal, uint64_t propSize)
    {
        SYN_CHECK(synTensorSetQuantizationData(handle(), prop, propVal, propSize));
    }

    template<class T>
    void setHostPtr(std::vector<T>& buffer, bool copyBuffer) const
    {
        auto layout = getDeviceFullLayout();
        setHostPtr(buffer, layout.deviceDataType, copyBuffer);
    }

    template<class T>
    void setHostPtr(std::vector<T>& buffer, const synDataType dataType, bool copyBuffer) const
    {
        SYN_CHECK(synTensorSetHostPtr(handle(),
                                      reinterpret_cast<void*>(buffer.data()),
                                      buffer.size() * sizeof(T),
                                      dataType,
                                      copyBuffer));
    }

    uint64_t getByteOffsetInSection() const
    {
        synSectionHandle sectionHandle;
        uint64_t         byteOffset = 0;
        SYN_CHECK(synTensorGetSection(handle(), &sectionHandle, &byteOffset));
        return byteOffset;
    }

    // For tensors without a section, an empty section object is returned.
    Section getSection() const
    {
        synSectionHandle sectionHandle;
        uint64_t         byteOffset = 0;
        SYN_CHECK(synTensorGetSection(handle(), &sectionHandle, &byteOffset));
        return sectionHandle == nullptr ? Section() : Section(std::make_shared<synSectionHandle>(sectionHandle));
    }

    void setGeometry(const synTensorGeometry& geometry, const synGeometryType geometryType) const
    {
        SYN_CHECK(synTensorSetGeometry(handle(), &geometry, geometryType));
    }

    void setGeometry(const synTensorGeometryExt& geometry, const synGeometryType geometryType) const
    {
        SYN_CHECK(synTensorSetGeometryExt(handle(), &geometry, geometryType));
    }

    synTensorGeometry getGeometry(const synGeometryType geometryType) const
    {
        synTensorGeometry geometry = {};
        SYN_CHECK(synTensorGetGeometry(handle(), &geometry, geometryType));
        return geometry;
    }

    synTensorGeometryExt getGeometryExt(const synGeometryType geometryType) const
    {
        synTensorGeometryExt geometry = {};
        SYN_CHECK(synTensorGetGeometryExt(handle(), &geometry, geometryType));
        return geometry;
    }



    void setGeometry(const std::vector<TSize>& geometry, const synGeometryType geometryType) const
    {
        SYN_THROW_IF(geometry.size() > HABANA_DIM_MAX, synInvalidArgument);
        synTensorGeometryExt geo = {};
        memcpy(&geo.sizes, geometry.data(), geometry.size() * sizeof(TSize));
        geo.dims = static_cast<TSize>(geometry.size());
        setGeometry(geo, geometryType);
    }

    void setDeviceLayout(const synTensorDeviceFullLayout& layout) const
    {
        SYN_CHECK(synTensorSetDeviceFullLayout(handle(), &layout));
    }

    void setDeviceLayout(const std::vector<uint32_t>& strides, synDataType dataType) const
    {
        SYN_THROW_IF(strides.size() > HABANA_DIM_MAX, synInvalidArgument);
        synTensorDeviceFullLayout layout;
        layout.deviceDataType = dataType;
        memset(layout.strides, 0, sizeof(layout.strides));
        if (strides.size() > 0)
        {
            memcpy(layout.strides, strides.data(), strides.size() * sizeof(uint32_t));
        }
        setDeviceLayout(layout);
    }

    void setDeviceDataType(const synDataType dataType) const
    {
        SYN_CHECK(synTensorSetDeviceDataType(handle(), dataType));
    }

    template<class T>
    void setQuantizationData(const synQuantizationProperty property, std::vector<T>& propVal) const
    {
        SYN_CHECK(synTensorSetQuantizationData(handle(),
                                               property,
                                               reinterpret_cast<void*>(propVal.data()),
                                               propVal.size() * sizeof(T)));
    }

    synTensorDeviceFullLayout getDeviceFullLayout() const
    {
        synTensorDeviceFullLayout layout = {};
        SYN_CHECK(synTensorGetDeviceFullLayout(handle(), &layout));
        return layout;
    }

    void setExternal(bool isExternal) const { SYN_CHECK(synTensorSetExternal(handle(), isExternal)); }
    bool isExternal() const
    {
        bool isExternal;
        SYN_CHECK(synTensorGetExternal(handle(), &isExternal));
        return isExternal;
    }

    void setAllowPermutation(bool allowPermutation) const
    {
        SYN_CHECK(synTensorSetAllowPermutation(handle(), allowPermutation));
    }
    bool getAllowPermutation() const
    {
        int8_t allowPermutation;
        SYN_CHECK(synTensorGetAllowPermutation(handle(), &allowPermutation));
        return allowPermutation;
    }

    void setPermutation(const std::vector<uint8_t>& permutation) const
    {
        SYN_THROW_IF(permutation.size() > HABANA_DIM_MAX, synInvalidArgument);
        synTensorPermutation p = {};

        p.dims = permutation.size();
        std::memcpy(p.permutation, permutation.data(), permutation.size());

        setPermutation(p);
    }

    void setPermutation(const synTensorPermutation& permutation) const
    {
        SYN_CHECK(synTensorSetPermutation(handle(), &permutation));
    }

    synTensorPermutation getPermutation() const
    {
        synTensorPermutation permutation = {};
        SYN_CHECK(synTensorGetPermutation(handle(), &permutation));
        return permutation;
    }

    std::string getName() const
    {
        char name[maxStringLength];
        SYN_CHECK(synTensorGetName(handle(), maxStringLength, name));
        return name;
    }

    HostBuffer getHostBuffer()
    {
        void*       buffer;
        uint64_t    size;
        synDataType dataType;
        SYN_CHECK(synTensorGetHostPtr(handle(), &buffer, &size, &dataType));
        std::shared_ptr<void> hostBufferHandlePtr(buffer, [](void*) {});  // prevent deleting the data before the tensor
        return HostBuffer(hostBufferHandlePtr, size);
    }

    template<class T>
    void getQuantizationData(Quantization<T> quant) const
    {
        SYN_CHECK(synTensorGetQuantizationData(handle(), quant.getProperty(), &quant.get(), &quant.size()));
    }

    uint64_t getSizeInBytes() const
    {
        synTensorDeviceFullLayout layout   = getDeviceFullLayout();
        synTensorGeometryExt      geometry = getGeometryExt(synGeometryMaxSizes);
        return getSizeInBytes(geometry, layout, layout.deviceDataType);
    }

    TSize getSizeInElements() const
    {
        TSize                tensorSize = 1;
        synTensorGeometryExt geometry   = getGeometryExt(synGeometryMaxSizes);
        for (unsigned i = 0; i < geometry.dims; ++i)
        {
            tensorSize *= geometry.sizes[i];
        }
        return tensorSize;
    }

    synTensorType getType() const { return m_type; }
    synDataType   getDataType() const { return getDeviceFullLayout().deviceDataType; }

protected:
    Tensor(std::shared_ptr<synTensor> handle, const synTensorType type) : SynObject(handle), m_type(type) {}

    Section       m_section;
    synTensorType m_type = TENSOR_TYPE_MAX;

    friend class GraphBase;  // GraphBase class requires access to Tensor private constructor
};
}  // namespace syn
