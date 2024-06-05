#pragma once

#include <cstdint>
#include <cstring>
#include <array>
#include <random>
#include <functional>
#include <unordered_map>
#include <ostream>
#include "include/mme_common/mme_common_enum.h"
#include "include/general_utils.h"

using byte = char;

class DataBuffer
{
public:
    DataBuffer() = default;
    DataBuffer(byte* data, uint64_t sizeInBytes, const bool shouldCopyData = true);
    explicit DataBuffer(uint64_t sizeInBytes, bool initializeFirst = false, int alignment = 0);
    ~DataBuffer();
    DataBuffer(const DataBuffer& other);
    DataBuffer& operator=(const DataBuffer& other);
    DataBuffer(DataBuffer&& other);
    DataBuffer& operator=(DataBuffer&& other);
    byte* operator[](uint64_t idx) const;
    byte* get() { return m_data; }
    const byte* get() const { return m_data; }
    uint64_t sizeInBytes() const { return m_bufferSize; }
    void setShoudFree(bool value);
    void reset();
    template<typename T>
    explicit operator T*() const
    {
        return (T*) get();
    }
    template<typename T>
    explicit operator T*()
    {
        return (T*) get();
    }

private:
    byte* m_data = nullptr;
    bool m_shouldFree = false;
    uint64_t m_bufferSize = 0;
};


// base class for simulating an mme tensor.
// platform-specific features can be added in the derived classes.
// the slowest changing dimension is in the highest index in the array.
// Indices always start at 'm_sizes' array index 0 and end according to the order of the tensor.
// e.g. sizes=[9,9, 0,0] represents a 9*9 matrix.
class MMESimTensorBase
{
public:
    virtual ~MMESimTensorBase() = default;
    static const unsigned int c_tensorMaxDim = 5;

    uint64_t getMemorySize() const;
    uint64_t getSizeInElements() const;
    uint64_t getElementsCount() const { return getSizeInElements(); }
    MMESimTensorBase& operator=(const MMESimTensorBase& other);

    void setDeviceAddress(byte* addr, bool checkInBuffer = true);
    void setName(std::string name) { m_name = name; }
    std::string getName() const { return m_name; }
    // todo AlonG: check if alignment is needed at all, and if its default value should be 128 as in Gaudi
    void setData(uint64_t tensorSizeInBytes, byte* data, int alignment = 0, const bool shouldCopyData = true);
    void setData(const DataBuffer& dataBuf);
    byte* data() const { return m_data; }
    void copySizes(int* sizes) const;
    void copySizes(MmeCommon::SizeArray sizes) const;
    const MmeCommon::SizeArray& getSizes() const { return m_sizes; };
    MmeCommon::SizeArray& getSizes() { return m_sizes; };
    uint64_t getSize(int dim) const { return m_sizes[dim]; }
    void copyStrides(int* strides) const;
    void copyStrides(MmeCommon::SizeArray strides) const;
    const MmeCommon::SizeArray& getStrides() const { return m_strides; }
    uint64_t getStride(int dim) const { return m_strides[dim]; }
    int getDim() const { return m_dim; }
    int getElementSize() const { return m_size; }
    MmeCommon::EMmeDataType getElementType() const { return m_type; }
    unsigned getFpBias() const { return m_fpBias; }
    void setFpBias(unsigned bias) { m_fpBias = bias; }
    MmeCommon::InfNanMode getInfNanMode() const { return m_infNanMode; }
    byte* getElementAt(const int* offsets) const;
    MmeCommon::SizeArray getOffsetOfIndex(const unsigned element) const;
    bool isContiguous() const;
    byte* getElementAt(const MmeCommon::SizeArray offsets) const;
    void fill(const byte* value) { memset(m_buffer.get(), *value, m_buffer.sizeInBytes()); }
    bool is4Bits() const;
    bool isInt() const;
    bool isSigned() const;
    uint64_t setStridesAndGetTensorSize(const MmeCommon::SizeArray* strides, int dim);

    static bool isType4Bits(MmeCommon::EMmeDataType type)
    {
        return type == MmeCommon::e_type_int4 || type == MmeCommon::e_type_uint4;
    }
    bool advanceOneElement(MmeCommon::SizeArray& idx) const;
    char* getElementAtIndex(const unsigned index) const;
    int32_t getIntValueAt(const int* offsets);
    int32_t getIntValueAt(const MmeCommon::SizeArray offsets);
    template<typename T>
    void setFourBitsValueAt(const int index, int8_t val)
    {
        int offset = index / 2;
        T* ptr = (T*) m_data + offset;
        if (0 == index % 2) ptr->i0 = val;
        else
            ptr->i1 = val;
    }

protected:
    // cannot build this class directly - must use derived classes.
    MMESimTensorBase() = default;
    MMESimTensorBase(const MmeCommon::SizeArray& sizes,
                     int dim,
                     MmeCommon::EMmeDataType type,
                     unsigned fpBias,
                     MmeCommon::InfNanMode infNanMode,
                     byte* data = nullptr,
                     const MmeCommon::SizeArray* strides = nullptr);
    MMESimTensorBase(const MMESimTensorBase* other, int* offsets, int dim, const MmeCommon::SizeArray* sizes);
    // backward compatibility constructor
    MMESimTensorBase(const int* sizes,
                     int dim,
                     MmeCommon::EMmeDataType type,
                     byte* data = nullptr,
                     int* strides = nullptr);

    MMESimTensorBase(const MMESimTensorBase* other, int* offsets, int dim, const int* sizes);
    MMESimTensorBase(const MMESimTensorBase* other);
    MMESimTensorBase(const MMESimTensorBase* other, char* data);

    void setSize(int size) { m_size = size; }
    void setSize(int dim, int size) { m_sizes[dim] = size; }
    void setStride(int dim, int stride) { m_strides[dim] = stride; }

private:
    MmeCommon::EMmeDataType m_type = MmeCommon::EMmeDataType::e_type_bf16;;
    MmeCommon::InfNanMode m_infNanMode = MmeCommon::e_mme_full_inf_nan;
    int m_fpBias = 0;
    int m_size = 0;  // size of element
    byte* m_data = nullptr;
    DataBuffer m_buffer;
    int m_dim = 0;
#ifdef VERIF_COMP
    MmeCommon::SizeArray m_sizes = {{1}};  // size in elements of each dimension
    MmeCommon::SizeArray m_strides = {{1}};  // stride in elements on each dimension
#else
    MmeCommon::SizeArray m_sizes = {1};  // size in elements of each dimension
    MmeCommon::SizeArray m_strides = {1};  // stride in elements on each dimension
#endif
    std::string m_name = "";
};
