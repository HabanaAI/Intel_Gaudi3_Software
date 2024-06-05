#include "sim_tensor_base.h"
#include <cstdint>
#include <functional>  // std::multiplies
#include <numeric>  // std::accumulate
#include "mme_assert.h"

using namespace MmeCommon;

DataBuffer::DataBuffer(byte* data, uint64_t sizeInBytes, const bool shouldCopyData)
{
    if (shouldCopyData)
    {
        m_data = new byte[sizeInBytes];
        memcpy(m_data, data, sizeInBytes);
        m_shouldFree = true;
    }
    else
    {
        m_data = data;
        m_shouldFree = false;
    }
    m_bufferSize = sizeInBytes;
}

DataBuffer::DataBuffer(uint64_t sizeInBytes, bool initializeFirst, int alignment)
{
    // todo AlonG: verify that alignment is not needed in gaudi
    MME_ASSERT(alignment == 0, "Data should be aligned");

    m_data = new byte[sizeInBytes + alignment];
    if (initializeFirst) memset(m_data, 0, sizeInBytes);
    m_shouldFree = true;
    m_bufferSize = sizeInBytes;
}

DataBuffer::~DataBuffer()
{
    if (m_shouldFree) delete[] m_data;
}

byte* DataBuffer::operator[](uint64_t idx) const
{
    MME_ASSERT(idx < m_bufferSize, "index is larger than allocated size");
    return &m_data[idx];
}

DataBuffer& DataBuffer::operator=(const DataBuffer& other)
{
    if (&other != this)
    {
        reset();
        m_bufferSize = other.m_bufferSize;
        m_data = new byte[m_bufferSize];
        memcpy(m_data, other.m_data, m_bufferSize);
        m_shouldFree = true;
    }
    return *this;
}

void DataBuffer::reset()
{
    if (m_shouldFree) delete[] m_data;
    m_data = nullptr;
    m_bufferSize = 0;
    m_shouldFree = false;
}

void DataBuffer::setShoudFree(bool value)
{
    m_shouldFree = value;
}

DataBuffer::DataBuffer(const DataBuffer& other)
{
    m_bufferSize = other.m_bufferSize;
    m_data = new byte[m_bufferSize];
    memcpy(m_data, other.m_data, m_bufferSize);
    m_shouldFree = true;
}

DataBuffer::DataBuffer(DataBuffer&& other)
{
    m_data = std::move(other.m_data);
    m_shouldFree = other.m_shouldFree;
    m_bufferSize = other.m_bufferSize;
    other.m_shouldFree = false;
    other.m_data = nullptr;
    other.m_bufferSize = 0;
}

DataBuffer& DataBuffer::operator=(DataBuffer&& other)
{
    if (this != &other)
    {
        reset();
        m_data = std::move(other.m_data);
        m_shouldFree = other.m_shouldFree;
        m_bufferSize = other.m_bufferSize;
        other.m_shouldFree = false;
        other.m_data = nullptr;
        other.m_bufferSize = 0;
    }
    return *this;
}

MMESimTensorBase::MMESimTensorBase(const int* sizes, int dim, EMmeDataType type, byte* data, int* strides)
: m_type(type), m_dim(dim)
{
    m_sizes.fill(1);
    for (unsigned i = 0; i < dim; ++i)
    {
        m_sizes[i] = sizes[i];
    }
    // Derived class should set m_size, and check for type compatibility as it is
    // platform specific. and then call those functions -
    //     unsigned tensorSize = setStridesAndGetTensorSize(&stridesArray, dim);
    //     setData(tensorSize, data, alignment);
}

MMESimTensorBase::MMESimTensorBase(const MMESimTensorBase* other, int* offsets, int dim, const int* sizes)
{
    m_dim = dim;
    m_type = other->m_type;
    m_size = other->m_size;
    m_strides = other->m_strides;
    m_sizes.fill(1);
    for (unsigned i = 0; i < dim; ++i)
    {
        m_sizes[i] = sizes[i];
    }
    m_buffer.reset();
    m_buffer = other->m_buffer;
    m_data = getElementAt(offsets);
}

MMESimTensorBase::MMESimTensorBase(const SizeArray& sizes,
                                   int dim,
                                   EMmeDataType type,
                                   unsigned fpBias,
                                   InfNanMode infNanMode,
                                   byte* data,
                                   const SizeArray* strides)
: m_type(type), m_fpBias(fpBias), m_infNanMode(infNanMode), m_dim(dim)
{
    m_sizes.fill(1);
    for (unsigned i = 0; i < dim; ++i)
    {
        m_sizes[i] = sizes[i];
    }
    // Derived class should set m_size, and check for type compatibility as it is
    // platform specific. and then call those functions - unsigned tensorSize =
    // setStridesAndGetTensorSize(strides, dim); setData(tensorSize, data);
}

MMESimTensorBase::MMESimTensorBase(const MMESimTensorBase* other)
{
    m_dim = other->m_dim;
    m_type = other->m_type;
    m_infNanMode = other->m_infNanMode;
    m_fpBias = other->m_fpBias;
    m_size = other->m_size;
    m_strides = other->m_strides;
    m_sizes = other->m_sizes;
    m_buffer.reset();
    m_buffer = other->m_buffer;
    m_data = other->m_data;
}

MMESimTensorBase::MMESimTensorBase(const MMESimTensorBase* other, int* offsets, int dim, const SizeArray* sizes)
: MMESimTensorBase(other)
{
    m_dim = dim;
    m_sizes = *sizes;
    m_data = getElementAt(offsets);
}

uint64_t MMESimTensorBase::getSizeInElements() const
{
    uint64_t ret = 0;
    for (int i = 0; i < m_dim; i++)
    {
        if (m_sizes[i] == 0)
        {
            return 0;
        }
        uint64_t dimSize = m_sizes[i] * (uint64_t) m_strides[i];
        if (dimSize > ret)
        {
            ret = dimSize;
        }
    }

    return ret;
}

uint64_t MMESimTensorBase::getMemorySize() const
{
    return getSizeInElements() * m_size;
}

bool MMESimTensorBase::is4Bits() const
{
    return ((m_type == e_type_int4) || (m_type == e_type_uint4));
}
bool MMESimTensorBase::isInt() const
{
    return (m_type < e_type_last_int_type);
}
bool MMESimTensorBase::isSigned() const
{
    return ((m_type != e_type_uint4) && (m_type != e_type_uint8) && (m_type != e_type_uint16));
}
int32_t MMESimTensorBase::getIntValueAt(const SizeArray offsets)
{
    const char* ptr = getElementAt(offsets);
    if (sizeof(uint32_t) == m_size)
    {
        return *(uint32_t*) ptr;
    }
    else if (sizeof(uint16_t) == m_size)
    {
        return *(uint16_t*) ptr;
    }
    else if (sizeof(uint8_t) == m_size)
    {
        return *(uint8_t*) ptr;
    }
    else
    {
        MME_ASSERT(0, "invalid data type");
    }
    return -1;
}

int32_t MMESimTensorBase::getIntValueAt(const int* offsets)
{
    const char* ptr = getElementAt(offsets);
    if (sizeof(uint32_t) == m_size)
    {
        return *(uint32_t*) ptr;
    }
    else if (sizeof(uint16_t) == m_size)
    {
        return *(uint16_t*) ptr;
    }
    else if (sizeof(uint8_t) == m_size)
    {
        return *(uint8_t*) ptr;
    }
    else
    {
        MME_ASSERT(0, "invalid data type");
    }
    return -1;
}
char* MMESimTensorBase::getElementAtIndex(const unsigned index) const
{
    SizeArray offsets = getOffsetOfIndex(index);
    return getElementAt(offsets);
}
bool MMESimTensorBase::advanceOneElement(SizeArray& idx) const
{
    for (int k = 0; k < m_dim; k++)
    {
        idx[k]++;
        if (idx[k] == m_sizes[k])
        {
            idx[k] = 0;
        }
        else
        {
            {
                return true;
            }
        }
    }
    return false;
}

MMESimTensorBase& MMESimTensorBase::operator=(const MMESimTensorBase& other)
{
    // handle self assignment
    if (this == &other)
    {
        return *this;
    }
    m_type = other.m_type;
    m_size = other.m_size;
    m_sizes = other.m_sizes;
    m_strides = other.m_strides;
    m_dim = other.m_dim;
    m_buffer = other.m_buffer;
    uint64_t diff = other.m_data - other.m_buffer.get();
    m_data = m_buffer.get() + diff;
    return *this;
}

void MMESimTensorBase::setDeviceAddress(byte* addr, bool checkAddrInBuffer)
{
    // check out of range
    MME_ASSERT(!checkAddrInBuffer ||
           (addr > m_buffer.get() && addr < m_buffer.get() + m_buffer.sizeInBytes()), "Out of range address");
    m_data = addr;
}

void MMESimTensorBase::copySizes(int* sizes) const
{
    memcpy(sizes, m_sizes.data(), m_dim * sizeof(int));
}

void MMESimTensorBase::copySizes(SizeArray sizes) const
{
    memcpy(sizes.data(), m_sizes.data(), m_dim * sizeof(int));
}

void MMESimTensorBase::copyStrides(int* strides) const
{
    memcpy(strides, m_strides.data(), m_dim * sizeof(int));
}

void MMESimTensorBase::copyStrides(SizeArray strides) const
{
    memcpy(strides.data(), m_strides.data(), m_dim * sizeof(int));
}

byte* MMESimTensorBase::getElementAt(const int* offsets) const
{
    int offset;
    offset = offsets[0];

    for (int i = 1; i < m_dim; i++)
    {
        offset += m_strides[i] * offsets[i];
    }
    return m_data + (offset * m_size);
}

byte* MMESimTensorBase::getElementAt(SizeArray offsets) const
{
    int offset;
    offset = offsets[0];

    for (int i = 1; i < m_dim; i++)
    {
        offset += m_strides[i] * offsets[i];
    }
    return m_data + (offset * m_size);
}

// sets the strides array and calculate the full tensor size in bytes.
uint64_t MMESimTensorBase::setStridesAndGetTensorSize(const SizeArray* strides, int dim)
{
    uint64_t tensorSize = 1;
    m_strides.fill(0);
    if (strides)
    {
        m_strides = *strides;
        for (unsigned i = 0; i < strides->size(); i++)
        {
            uint64_t stride = m_strides[i];
            uint64_t size = stride * m_sizes[i];
            if (size > tensorSize) tensorSize = size;
        }
    }
    else
    {
        for (unsigned i = 0; i < dim; i++)
        {
            m_strides[i] = tensorSize;
            tensorSize *= m_sizes[i];
        }
    }
    return tensorSize * getElementSize();
}

bool MMESimTensorBase::isContiguous() const
{
    if (m_strides[0] != 1)
    {
        return false;
    }

    for (unsigned i = 1; i < m_dim; i++)
    {
        if (m_strides[i] != (m_strides[i - 1] * m_sizes[i - 1]))
        {
            return false;
        }
    }

    return true;
};

void MMESimTensorBase::setData(uint64_t tensorSizeInBytes, byte* data, int alignment, const bool shouldCopyData)
{
    if (data != nullptr)
    {
        m_buffer = DataBuffer(data, tensorSizeInBytes, shouldCopyData);
        m_data = m_buffer.get();
    }
    else
    {
        m_buffer = DataBuffer(tensorSizeInBytes, false);
        m_data = m_buffer.get();
    }
}
void MMESimTensorBase::setData(const DataBuffer& dataBuf)
{
    m_buffer = dataBuf;
    m_data = m_buffer.get();
}

SizeArray MMESimTensorBase::getOffsetOfIndex(const unsigned element) const
{
    SizeArray index = {0};
    uint64_t currentSize = 1;
    unsigned dim = 0;
    MME_ASSERT(element < getSizeInElements(), "element out of bounds");
    for (dim = 0; dim < getDim(); ++dim)
    {
        if (element < currentSize * m_sizes[dim])
        {
            break;
        }
        currentSize *= m_sizes[dim];
    }

    unsigned currentIndex = element;
    for (int i = (int)dim; i >= 0; --i)
    {
        index[i] = currentIndex / currentSize;
        currentIndex %= currentSize;
        currentSize /= (i > 0) ? m_sizes[i - 1] : 1;
    }
    return index;
}
