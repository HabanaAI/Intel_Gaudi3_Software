#pragma once

#include <memory>
#include <cstdint>
#include <optional>
#include <vector>
#include "address_fields_container_info.h"
#include "defs.h"
#include "cache_types.h"

// Estimation of maximum descriptors per node
static constexpr int32_t MAX_ESTIMATE_AMOUNT_DESC_PER_NODE = 16;

// DescriptorToFwCtx<Desc_t>::type will return the type of the ARC firmware context matching the descriptor type
// previously coupled using this macro. For example of how coupling is done see the usage of this macro in
// gaudi2_types.h. If no coupling was done (like in non-ARC platforms), the default DescriptorToFwCtx will be in
// effect which means a struct with dummy byte will be returned as the type of the firmware context.
#define COUPLE_DESCRIPTOR_TO_FW_CTX(Desc_t, Ctx_t) \
    template<>                                     \
    struct DescriptorToFwCtx<Desc_t>               \
    {                                              \
        using type = Ctx_t;                        \
    }

struct ExecutionInfo
{
    unsigned pipelineLevel;
    unsigned engineIndex;
    unsigned predicate; // support selective execution using HW predicate mechanism
};

#define MASK_OFFSET(block, reg) (offsetof(block, reg) / sizeof(uint32_t))
#define MASK_SIZE(block) (sizeof(block) / sizeof(uint32_t))
#define SET_MASK_REG_ON(mask, offset) mask[offset] = true
#define SET_MASK_BULK(maskIterator, startOffset, endOffset, flag) std::fill(maskIterator + startOffset, maskIterator + endOffset, flag)
#define SET_MASK_BULK_ON(maskIterator, startOffset, endOffset) SET_MASK_BULK(maskIterator, startOffset, endOffset, true)

template<class Desctype>  // DMA / TPC / ROT / MME descriptor
using ValidityMask = std::array<bool, MASK_SIZE(Desctype)>;

template<class Desctype>
using DescAndMask = std::pair<Desctype, ValidityMask<Desctype>>;

template<typename T>
struct DescriptorToFwCtx
{
    struct type { uint8_t dummy; };
};

// A class that wraps a Descriptor and any information classes related to it
template<class Descriptor, typename FwCtx = typename DescriptorToFwCtx<Descriptor>::type>
class DescriptorWrapper
{
public:
    using OptionalValidityMask = std::optional<ValidityMask<Descriptor>>;

    enum class BlockT
    {
        DESCRIPTOR = 0,
        WD_CTX = 1
    };

    DescriptorWrapper() = default;

    DescriptorWrapper(const Descriptor& descriptor, const OptionalValidityMask& mask)
    : m_descriptor(descriptor), m_mask(mask)
    {
    }

    DescriptorWrapper(const Descriptor& descriptor)
    : DescriptorWrapper(descriptor, OptionalValidityMask())
    {
    }

    const BasicFieldsContainerInfo& getBasicFieldsContainerInfo() const { return m_basicFieldsContainerInfo; }
    const ExecutionInfo&            getExecutionInfo() const { return m_executionInfo; }
    Descriptor&                     getDescriptor() { return m_descriptor; }
    const Descriptor&               getDescriptor() const { return m_descriptor; }
    BasicFieldsContainerInfo&       getBasicFieldsContainerInfo() { return m_basicFieldsContainerInfo; }
    ExecutionInfo&                  getExecutionInfo() { return m_executionInfo; }
    OptionalValidityMask&           getMask() { return m_mask; }
    const OptionalValidityMask&     getMask() const { return m_mask; }
    BasicFieldsContainerInfo&       getBasicFieldsContainerInfoForCtx() { return m_basicFieldsContainerInfoForCtx; }
    const BasicFieldsContainerInfo& getBasicFieldsContainerInfoForCtx() const { return m_basicFieldsContainerInfoForCtx; }

    void setFwCtx(const FwCtx& ctx)
    {
        m_fwCtxs[0] = ctx;
        m_fwCtxsCount = 1;
    }
    void addFwCtx(const FwCtx& ctx)
    {
        HB_ASSERT(m_fwCtxsCount < MAX_NUM_DCORES, "array overflow");
        m_fwCtxs[m_fwCtxsCount++] = ctx;
    }
    FwCtx& getFwCtx(unsigned i = 0)
    {
        HB_ASSERT(i < MAX_NUM_DCORES, "index out of bound");
        return m_fwCtxs[i];
    }
    const FwCtx& getFwCtx(unsigned i = 0) const
    {
        HB_ASSERT(i < MAX_NUM_DCORES, "index out of bound");
        return m_fwCtxs[i];
    }
    unsigned getFwCtxCount() const
    {
        return m_fwCtxsCount;
    }

    BasicFieldsContainerInfo& getBasicFieldsContainerInfoForBlock(BlockT blockType)
    {
        const DescriptorWrapper& w = *this;
        return const_cast<BasicFieldsContainerInfo&>(getBasicFieldsContainerInfoForBlock(blockType));
    }
    const BasicFieldsContainerInfo& getBasicFieldsContainerInfoForBlock(BlockT blockType) const
    {
        switch (blockType)
        {
            case BlockT::DESCRIPTOR: return getBasicFieldsContainerInfo();
            case BlockT::WD_CTX:     return getBasicFieldsContainerInfoForCtx();
        }
        HB_ASSERT(false, "Invalid block type {}", uint32_t(blockType));
    }

private:
    Descriptor                 m_descriptor;
    BasicFieldsContainerInfo   m_basicFieldsContainerInfo;
    BasicFieldsContainerInfo   m_basicFieldsContainerInfoForCtx;
    ExecutionInfo              m_executionInfo = {0, 0, 0};
    OptionalValidityMask       m_mask;
    FwCtx                      m_fwCtxs[MAX_NUM_DCORES] = {0}; // multiple contexts for descriptors with dcore locality
    unsigned                   m_fwCtxsCount = 0;
};
