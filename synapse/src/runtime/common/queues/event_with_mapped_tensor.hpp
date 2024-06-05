#pragma once
#include "event_interface.hpp"

struct InternalRecipeHandle;

const uint32_t SEQUENCE_OFFSET_NOT_USED = UINT32_MAX;

class EventWithMappedTensor : public EventInterface
{
public:
    virtual void setMappedTensor(uint64_t              tensorOffset,
                                 uint32_t              tensorId,
                                 const char*           tensorName,
                                 InternalRecipeHandle* pInternalRecipeHandle)
    {
        m_tensorMapping = TensorMapping(tensorOffset, tensorId, tensorName, pInternalRecipeHandle);
    }

    void clearTensorMapping() { m_tensorMapping = TensorMapping(); }

    virtual void clearState() { clearTensorMapping(); }

    InternalRecipeHandle* getInternalRecipeHandle() const { return m_tensorMapping.m_pInternalRecipeHandle; }

    const char* getTensorName() const { return m_tensorMapping.m_mappedTensorName; }

    uint32_t getTensorIdx() const { return m_tensorMapping.m_mappedTensorIdx; }

    uint64_t getSequenceOffset() const { return m_tensorMapping.m_sequenceIdOffset; }

    bool isInternalSignalingEvent() const { return getTensorIdx() != UINT32_MAX; }

    static const uint64_t INVALID_SEQ_ID = std::numeric_limits<uint64_t>::max();

private:
    struct TensorMapping
    {
        TensorMapping() : TensorMapping(SEQUENCE_OFFSET_NOT_USED, UINT32_MAX, "", 0) {}

        TensorMapping(uint64_t              sequenceOffset,
                      uint32_t              tensorIdx,
                      const char*           tensorName,
                      InternalRecipeHandle* pInternalRecipeHandle)
        : m_mappedTensorIdx(tensorIdx),
          m_mappedTensorName(tensorName),
          m_sequenceIdOffset(sequenceOffset),
          m_pInternalRecipeHandle(pInternalRecipeHandle)
        {
        }

        uint32_t    m_mappedTensorIdx;  // tensor idx in recipe
        const char* m_mappedTensorName;
        uint64_t    m_sequenceIdOffset;
        InternalRecipeHandle* m_pInternalRecipeHandle;
    };

    TensorMapping m_tensorMapping;
};