#include <gaudi/asic_reg_structs/sync_object_regs.h>
#include "platform/gaudi/graph_compiler/queue_command_factory.h"
#include "platform/gaudi/graph_compiler/command_queue.h"
#include "platform/gaudi/graph_compiler/hal_conventions.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "gaudi_graph.h"
#include "gaudi_types.h"
#include "include/gaudi/mme_descriptor_generator.h"
#include "node_annotation.h"
#include "queue_command.h"
#include "sync/sync_conventions.h"
#include "habana_global_conf.h"


//---------------------------------------------------------
//------------------- CompletionQueue -------------------
//---------------------------------------------------------

gaudi::CompletionQueue::CompletionQueue() :
    ::CommandQueue(QueueCommandFactory::instance(),
                   (unsigned)getQueueID(DEVICE_COMPLETION_QUEUE, 0),
                   DEVICE_COMPLETION_QUEUE)
{
    m_maxStreams = CompilationHalReader::getHalReader()->getNumEngineStreams();
}

//---------------------------------------------------------
//---------------------- MmeQueue -------------------------
//---------------------------------------------------------

gaudi::MmeQueue::MmeQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents) :
    ::MmeQueue<gaudi::MmeDesc>(QueueCommandFactory::instance(), (unsigned)getQueueID(DEVICE_MME, engineId))
{
    m_engineId       = engineId;
    m_engineIndex    = engineIndex;
    m_maxStreams     = CompilationHalReader::getHalReader()->getNumEngineStreams();
    m_sendSyncEvents = sendSyncEvents;
}

void gaudi::MmeQueue::setDescriptorSignaling(gaudi::MmeDesc& desc, const std::shared_ptr<SyncObject>& sync)
{
    if (sync != nullptr)
    {
        gaudi::patchSyncObject(desc,
                               Mme::e_mme_local,
                               gaudi::getSyncObjectAddress(sync->id),
                               sync->value,
                               sync->operation == SYNC_OP_ADD,
                               m_sendSyncEvents /*perf event*/);

        gaudi::patchSyncObject(desc,
                               Mme::e_mme_remote,
                               gaudi::getSyncObjectAddress(sync->id + 1),
                               sync->value,
                               sync->operation == SYNC_OP_ADD,
                               m_sendSyncEvents /*perf event*/);
    }
}

//---------------------------------------------------------
//---------------------- TpcQueue -------------------------
//---------------------------------------------------------

gaudi::TpcQueue::TpcQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents, bool graphHasDynamicity)
: ::TpcQueue<gaudi::TpcDesc>(QueueCommandFactory::instance(), (unsigned) getQueueID(DEVICE_TPC, engineId)),
  m_graphHasDynamicity(graphHasDynamicity)
{
    m_engineId       = engineId;
    m_engineIndex    = engineIndex;
    m_maxStreams     = CompilationHalReader::getHalReader()->getNumEngineStreams();
    m_sendSyncEvents = sendSyncEvents;
}

std::vector<DescSection> gaudi::TpcQueue::getUnpredicatedSections(pNode n, const gaudi::TpcDesc& desc) const
{
    if (predicateSectionEnabled(n))
    {
        return {DescSection {&desc,
                             reinterpret_cast<const char*>(&desc.m_desc.tid_base_dim_0) -
                                 reinterpret_cast<const char*>(&desc),
                             0},
                DescSection {&desc.m_desc.kernel_config,
                             sizeof(desc) - (reinterpret_cast<const char*>(&desc.m_desc.kernel_config) -
                                             reinterpret_cast<const char*>(&desc)),
                             reinterpret_cast<const char*>(&desc.m_desc.kernel_config) -
                                 reinterpret_cast<const char*>(&desc)}};
    }
    else if (separateSizeSectionEnabled(n))
    {
        return {DescSection {&desc,
                             reinterpret_cast<const char*>(&desc.m_so) - reinterpret_cast<const char*>(&desc),
                             0,
                             true},
                DescSection {&desc.m_so,
                             sizeof(desc) -
                                 (reinterpret_cast<const char*>(&desc.m_so) - reinterpret_cast<const char*>(&desc)),
                             reinterpret_cast<const char*>(&desc.m_so) - reinterpret_cast<const char*>(&desc)}};
    }
    return {DescSection(desc)};
}

bool gaudi::TpcQueue::predicateSectionEnabled(pNode n) const
{
    // There is no point to do duplicate-and-predicate if we are not compressing blobs
    // or use single blob per engine. Also, the feature must be turned on.
    // In dynamic shapes the compression is set off (even if the GCFG is on), so in order to keep consistency,
    // we also have to disable the predicate mechanism
    if (!GCFG_ENABLE_TPC_PREDICATED_CMD.value()) return false;
    if (!GCFG_COMPRESS_BLOBS.value()) return false;
    if (m_graphHasDynamicity) return false;
    return true;
}

bool gaudi::TpcQueue::separateSizeSectionEnabled(pNode n) const
{
    if (!m_graphHasDynamicity) return false;
    if (!GCFG_ENABLE_BIG_TENSOR_PP_PRUNE.value()) return false;
    return true;
}

std::vector<DescSection> gaudi::TpcQueue::getPredicatedSections(pNode n, const gaudi::TpcDesc& desc) const
{
    if(!predicateSectionEnabled(n)) return {};

    return {DescSection {&desc.m_desc.tid_base_dim_0,
                         reinterpret_cast<const char*>(&desc.m_desc.kernel_config) -
                             reinterpret_cast<const char*>(&desc.m_desc.tid_base_dim_0),
                         reinterpret_cast<const char*>(&desc.m_desc.tid_base_dim_0) -
                             reinterpret_cast<const char*>(&desc)}};
}

void gaudi::TpcQueue::setDescriptorSignaling(gaudi::TpcDesc& desc, const std::shared_ptr<SyncObject>& sync)
{
    if (sync != nullptr)
    {
        desc.m_so.addr.v                 = gaudi::getSyncObjectAddress(sync->id);
        desc.m_so.message.so_write_value = (int16_t)sync->value;
        desc.m_so.message.so_operation   = sync->operation;
        bool sendSyncEvents              = GCFG_TPC_SYNC_TRACE_EN_MASK.value() & GetEngineIndex();
        if (sendSyncEvents)
        {
            setSendSyncEvents(desc.m_so.message._raw);
        }
    }
}

//---------------------------------------------------------
//-------------------- DmaDescQueue -----------------------
//---------------------------------------------------------

gaudi::DmaDescQueue::DmaDescQueue(unsigned logicalQueue, unsigned engineId, unsigned engineIndex, bool sendSyncEvents) :
    ::DmaDescQueue<gaudi::DmaDesc>(QueueCommandFactory::instance(),
                                   (unsigned)getQueueID(DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL, engineId)),
    m_logicalQueue(logicalQueue),
    m_lastNodeOpType(DMA_OP_TYPE::DMA_OP_COPY)
{
    m_engineId       = engineId;
    m_engineIndex    = engineIndex;
    m_maxStreams     = CompilationHalReader::getHalReader()->getNumEngineStreams();
    m_sendSyncEvents = sendSyncEvents;
}

DescriptorShadow::AllRegistersProperties gaudi::DmaDescQueue::registersPropertiesForDesc(pNode n, const DescriptorWrapper<DmaDesc>& descWrapper)
{
    const DmaDesc desc = descWrapper.getDescriptor();
    std::vector<DescriptorShadow::RegisterProperties> mask(
        sizeof(desc) / sizeof(uint32_t),
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Banned));

    // addresses
    auto* base = reinterpret_cast<const uint32_t*>(&desc);
    DescriptorShadow::setRegisterPropertyOnSegment(
                mask,
                Segment {reinterpret_cast<const uint32_t*>(&desc.src_base_lo) - base, reinterpret_cast<const uint32_t*>(&desc._pad36) - base},
                DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data));

    // source size and stride
    DescriptorShadow::setRegisterPropertyOnSegment(
            mask,
            Segment {reinterpret_cast<const uint32_t*>(&desc.src_tsize_1) - base, reinterpret_cast<const uint32_t*>(&desc._pad80) - base},
            DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data));

    // destination size and stride
    DescriptorShadow::setRegisterPropertyOnSegment(
            mask,
            Segment {reinterpret_cast<const uint32_t*>(&desc.dst_tsize_1) - base, reinterpret_cast<const uint32_t*>(&desc.commit) - base},
            DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data));

    // Signaling data
    DescriptorShadow::setRegisterPropertyOnSegment(
        mask,
        Segment {reinterpret_cast<const uint32_t*>(&desc.wr_comp_wdata) - base,
                  reinterpret_cast<const uint32_t*>(&desc.wr_comp_awuser_31_11) - base},
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data));

    // Transpose
    mask[reinterpret_cast<const uint32_t*>(&desc.te_numrows) - base] =
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data);

    // reduction
    mask[reinterpret_cast<const uint32_t*>(&desc.wr_awuser_31_11) - base] =
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data);

    if (canUseLinDmaPacket(n))
    {
        // Those registers are set by the packet itself
        // addresses
        DescriptorShadow::setRegisterPropertyOnSegment(
            mask,
            Segment {reinterpret_cast<const uint32_t*>(&desc.src_base_lo) - base,
                     reinterpret_cast<const uint32_t*>(&desc._pad36) - base},
            DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Ignore));

        // source size and stride
        DescriptorShadow::setRegisterPropertyOnSegment(
            mask,
            Segment {reinterpret_cast<const uint32_t*>(&desc.src_tsize_1) - base,
                     reinterpret_cast<const uint32_t*>(&desc._pad80) - base},
            DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Ignore));

        // destination size and stride
        DescriptorShadow::setRegisterPropertyOnSegment(
            mask,
            Segment {reinterpret_cast<const uint32_t*>(&desc.dst_tsize_1) - base,
                     reinterpret_cast<const uint32_t*>(&desc.commit) - base},
            DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Ignore));
    }

    // Using data/ignore masking from descriptor generator to data registers.
    if (descWrapper.getMask().has_value())
    {
        for (size_t i = 0; i < descWrapper.getMask().value().size(); i++)
        {
            if (!descWrapper.getMask().value()[i])
            {
                mask[i].ignore = true;
            }
        }
    }
    return std::make_shared<std::vector<DescriptorShadow::RegisterProperties>>(mask);
}

QueueCommandPtr gaudi::DmaDescQueue::getExeCmd(pNode n, const DescriptorWrapper<DmaDesc>& descWrap, bool enableSignal)
{
    HB_ASSERT_PTR(std::dynamic_pointer_cast<DMANode>(n));

    enableSignal = true;  // Gaudi must signal on every command

    const DmaDesc& desc = descWrap.getDescriptor();

    dma_core::reg_commit commit;
    std::shared_ptr<DMANode> dmaNode = std::dynamic_pointer_cast<DMANode>(n);

    commit._raw = 0;
    commit.ctx_id = (unsigned)((dmaNode->getContextId() & 0xFF));  // upper 8 bits are passed via dma dest address 8 MSB
    uint8_t ctxIdHi = (uint8_t)(((dmaNode->getContextId() & 0xFF00) >> 8));
    if (desc.src_tsize_0.val == 0 && desc.dst_tsize_0.val == 0)
    {
        // Empty job
        commit.wr_comp_en = 1;
        commit.lin = 1;
        return std::make_unique<ExecuteDmaDesc>(commit._raw,
                                                GetDeviceType(),
                                                GetEngineID(),
                                                false,
                                                DEFAULT_PREDICATE,
                                                ctxIdHi);
    }

    bool                      isLinear = dmaNode->isLinearDma();

    if (canUseLinDmaPacket(n))
    {
        updateLinDMALoadedDescSections(desc);

        return getLinDmaCmd(n, desc, enableSignal);
    }

    if (enableSignal)
    {
        commit.wr_comp_en = 1;  // enables the sync manager message
    }

    // Memset is implemented with strided DMA copy because of HW bug https://jira.habana-labs.com/browse/SIV-23
    // Zero desciptors must be linear (simulator supports zero jobs only for linear )
    if (isLinear || (desc.dst_tsize_0.val == 0 && desc.src_tsize_0.val == 0))
    {
        commit.lin = 1;

        // Memset are implemented with Memcpy as a WA for HW bug.
        // This code is in order to be able to use DMA memset
        if (dmaNode->getOpType() == DMA_OP_TYPE::DMA_OP_MEMSET)
        {
            commit.mem_set = 1;
        }
    }

    commit.transpose = dmaNode->isTranspose();
    if (dmaNode->isTranspose())
    {
        HB_ASSERT(dmaNode->getInput(0)->getElementSizeInBytes() == 2 || dmaNode->getInput(0)->getElementSizeInBytes() ==4, "Unsupported dma transpose for element size {}", dmaNode->getInput(0)->getElementSizeInBytes());
        commit.dtype = (dmaNode->getInput(0)->getElementSizeInBytes() == 4);
    }

    bool setEngBarrier = (m_lastNodeOpType != dmaNode->getOpType());
    if (GCFG_PROTECT_UNSAFE_DMA_TRANSPOSE.value())
    {
        setEngBarrier |= dmaNode->isTranspose(); //Work-around for issue in TE getting stuck
    }
    return std::make_unique<ExecuteDmaDesc>(commit._raw,
                                            GetDeviceType(),
                                            GetEngineID(),
                                            setEngBarrier,
                                            DEFAULT_PREDICATE,
                                            ctxIdHi);
}

void gaudi::DmaDescQueue::setDescriptorSignaling(gaudi::DmaDesc& desc, const std::shared_ptr<SyncObject>& sync)
{
    if (sync != nullptr)
    {
        ptrToInt soAddress;
        soAddress.u64 = getSyncObjectAddress(sync->id);

        // wr_comp_wdata is the data that is sent to the Sync Manager upon completion.
        desc.wr_comp_wdata.val  = 0;
        desc.wr_comp_wdata.val |= (0x00007FFF & sync->value);              // lower 15 bits are data
        desc.wr_comp_wdata.val |= (0xC0000000 & (sync->operation << 29));  // operation and trace go to the 2 MSB

        desc.wr_comp_addr_lo.val = soAddress.u32[0];
        desc.wr_comp_addr_hi.val = soAddress.u32[1];

        bool sendSyncEvents = GCFG_DMA_SYNC_TRACE_EN_MASK.value() & GetEngineIndex();
        if (sendSyncEvents)
        {
            setSendSyncEvents(desc.wr_comp_wdata._raw);
        }
    }
}

QueueCommandPtr gaudi::DmaDescQueue::getLinDmaCmd(pNode n, const DmaDesc& desc, bool enableSignal)
{
    auto dmaNode = std::dynamic_pointer_cast<DMANode>(n);
    HB_ASSERT_PTR(dmaNode);
    static const uint32_t SRC_DWORD_OFFSET_FROM_HEADER = 0;
    static const uint32_t DST_DWORD_OFFSET_FROM_HEADER = 2;

    ptrToInt src;
    ptrToInt dst;
    uint64_t size          = desc.dst_tsize_0.val;
    bool     isMemset      = n->isMemset();
    bool     setEngBarrier = (m_lastNodeOpType != dmaNode->getOpType());
    bool     srcInDram     = !isMemset && n->getInput(0)->tensorAllocatedInDram(); //beware: in memset there is no input
    bool     dstInDram     = n->getOutput(0)->tensorAllocatedInDram();

    src.u32[0] = desc.src_base_lo._raw;
    src.u32[1] = desc.src_base_hi._raw;
    dst.u32[0] = desc.dst_base_lo._raw;
    dst.u32[1] = desc.dst_base_hi._raw;

    QueueCommandPtr cmd = std::make_unique<DmaDeviceInternal>(src.u64,
                                                              srcInDram,
                                                              dst.u64,
                                                              dstInDram,
                                                              size,
                                                              setEngBarrier,
                                                              isMemset,
                                                              enableSignal,
                                                              n->getContextId());

    BasicFieldsContainerInfo cmdAddressFieldsInfo;
    uint64_t memID;
    uint64_t targetAddress;

    if (srcInDram)
    {
        memID         = getMemoryIDFromVirtualAddress(src.u64);
        targetAddress = src.u64;
        cmdAddressFieldsInfo.addAddressEngineFieldInfo(n,
                                                       getMemorySectionNameForMemoryID(memID),
                                                       memID,
                                                       targetAddress,
                                                       SRC_DWORD_OFFSET_FROM_HEADER,
                                                       FIELD_MEMORY_TYPE_DRAM);
    }

    if (dstInDram)
    {
        memID         = getMemoryIDFromVirtualAddress(dst.u64);
        targetAddress = dst.u64;
        cmdAddressFieldsInfo.addAddressEngineFieldInfo(n,
                                                       getMemorySectionNameForMemoryID(memID),
                                                       memID,
                                                       targetAddress,
                                                       DST_DWORD_OFFSET_FROM_HEADER,
                                                       FIELD_MEMORY_TYPE_DRAM);
    }

    cmd->SetContainerInfo(cmdAddressFieldsInfo);
    return cmd;
}

void gaudi::DmaDescQueue::updateQueueStateAfterPush(pNode n)
{
    std::shared_ptr<DMANode> dmaNode = std::dynamic_pointer_cast<DMANode>(n);
    HB_ASSERT_PTR(dmaNode);
    m_lastNodeOpType = dmaNode->getOpType();
}

bool gaudi::DmaDescQueue::allowNoDescUpdates(pNode n)
{
    return true; // due to memory reuse, we may have consecutive transactions with the same addresses
}

bool gaudi::DmaDescQueue::canUseLinDmaPacket(pNode n)
{
    HB_ASSERT_PTR(std::dynamic_pointer_cast<DMANode>(n));

    // In case we are doing linear DMA and the LinDMA optimization is enabled by GCFG,
    // we may use LinDMA packet instead of descriptor.
    bool isLinear = std::dynamic_pointer_cast<DMANode>(n)->isLinearDma();
    bool optimize = GCFG_LIN_DMA_OPTIMIZATION_ENABLED.value();
    return isLinear && optimize;
}

void gaudi::DmaDescQueue::updateLinDMALoadedDescSections(const DmaDesc& desc)
{
    /**
     * Linear DMA doesn't use the DescriptorShadow, but it still uses some registeres we need to update.
     * Hence we update these registers in DescriptorShadow
     */
    uint32_t  regOffsetInSection;

    // sizes
    regOffsetInSection = offsetof(block_dma_core, dst_tsize_0) / sizeof(uint32_t);
    getDescShadow().updateLoadedReg(regOffsetInSection, desc.dst_tsize_0.val);

    // address
    regOffsetInSection = offsetof(block_dma_core, src_base_lo) / sizeof(uint32_t);
    getDescShadow().updateLoadedReg(regOffsetInSection, desc.src_base_lo.val);

    regOffsetInSection = (offsetof(block_dma_core, src_base_hi)) / sizeof(uint32_t);
    getDescShadow().updateLoadedReg(regOffsetInSection, desc.src_base_hi.val);

    regOffsetInSection = (offsetof(block_dma_core, dst_base_lo)) / sizeof(uint32_t);
    getDescShadow().updateLoadedReg(regOffsetInSection, desc.dst_base_lo.val);

    regOffsetInSection = (offsetof(block_dma_core, dst_base_hi)) / sizeof(uint32_t);
    getDescShadow().updateLoadedReg(regOffsetInSection, desc.dst_base_hi.val);
}
