#include "recipe_hal.h"

// eager includes (relative to src/eager/lib/)
#include "chip_specification/gaudi2/recipe/command_packets_wrappers.h"
#include "chip_specification/gaudi2/recipe/recipe_hal_defs.h"
#include "chip_specification/gaudi2/recipe/template_structs.h"
#include "recipe_gen/mini_command_packets.h"
#include "synapse_common_types.h"
#include "utils/memory_utils.h"
#include <optional>

// synapse-internal includes (relative to src/)
#ifndef NDEBUG
#include "graph_compiler/data_type_utils.h"
#endif

// synapse-internal gaudi2-specific includes (relative to src/)
#include "hal_reader/gaudi2/hal.h"
#include "platform/gaudi2/graph_compiler/block_data.h"
#include "platform/gaudi2/graph_compiler/gaudi2_types.h"
#include "platform/gaudi2/graph_compiler/recipe_generator.h"

// relative to <qman_fw>/engines-arc/include/
#include "gaudi2_arc_eng_packets.h"

// relative to <specs>/
#include "gaudi2/gaudi2_packets.h"

using namespace gaudi2;

namespace eager_mode::gaudi2_spec_info
{
static_assert(hal::numMmeEngines <= RecipeHalBase::maxMmeEnginesNr);
static_assert(hal::numInternalDmaEngines <= RecipeHalBase::maxDmaEnginesNr);
static_assert(hal::baseRegistersCacheSize <= RecipeHalBase::maxBaseRegistersCacheSize);

RecipeHal::RecipeHal()
: RecipeHalBase({sizeof(eng_arc_cmd_list_size_t),
                 sizeof(eng_arc_cmd_nop_t),
                 sizeof(eng_arc_cmd_wd_fence_and_exec_t),
                 sizeof(eng_arc_cmd_sched_dma_t),
                 sizeof(eng_arc_cmd_static_desc_v2_t)})
{
}

unsigned RecipeHal::getConstantNodeRecipeMemoryRequirement() const
{
    // execution blob = 296, dynamic blob = 80, patching blob = 24
    constexpr unsigned CONSTANT_BLOBS_SIZE = 400;
    // constant kernel size is identical regardless of type
    constexpr unsigned CONSTANT_KERNEL_SIZE = 2208;
    return CONSTANT_BLOBS_SIZE + CONSTANT_KERNEL_SIZE;
}

static inline unsigned getCastKernelSizeFromInt8(synDataType to)
{
    switch (to)
    {
        case syn_type_uint8:
            return 2880;
        case syn_type_int16:
        case syn_type_uint16:
            return 2400;
        case syn_type_int32:
        case syn_type_uint32:
            return 6880;
        case syn_type_fp16:
            return 2624;
        case syn_type_bf16:
            return 2528;
        case syn_type_float:
            return 6400;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSizeFromUint8(synDataType to)
{
    switch (to)
    {
        case syn_type_int8:
            return 3328;
        case syn_type_int16:
            return 2752;
        case syn_type_uint16:
            return 2400;
        case syn_type_int32:
        case syn_type_uint32:
            return 6880;
        case syn_type_fp16:
            return 2624;
        case syn_type_bf16:
            return 2528;
        case syn_type_float:
            return 6400;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSizeFromInt16(synDataType to)
{
    switch (to)
    {
        case syn_type_int8:
            return 3584;
        case syn_type_uint8:
            return 2656;
        case syn_type_uint16:
            return 2336;
        case syn_type_int32:
        case syn_type_uint32:
            return 6176;
        case syn_type_fp16:
        case syn_type_bf16:
            return 5856;
        case syn_type_float:
            return 6176;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSizeFromUint16(synDataType to)
{
    switch (to)
    {
        case syn_type_int8:
        case syn_type_uint8:
            return 2272;
        case syn_type_int16:
            return 2336;
        case syn_type_int32:
        case syn_type_uint32:
            return 6176;
        case syn_type_fp16:
        case syn_type_bf16:
            return 5856;
        case syn_type_float:
            return 6656;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSizeFromInt32(synDataType to)
{
    switch (to)
    {
        case syn_type_int8:
            return 8192;
        case syn_type_uint8:
            return 6464;
        case syn_type_int16:
        case syn_type_uint16:
            return 6112;
        case syn_type_uint32:
            return 2336;
        case syn_type_int64:
            return 3712;
        case syn_type_fp16:
        case syn_type_bf16:
            return 5920;
        case syn_type_float:
            return 5888;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSizeFromUint32(synDataType to)
{
    switch (to)
    {
        case syn_type_int8:
            return 6464;
        case syn_type_uint8:
            return 8192;
        case syn_type_int16:
        case syn_type_uint16:
            return 6112;
        case syn_type_int32:
            return 2336;
        case syn_type_uint64:
            return 3232;
        case syn_type_fp16:
        case syn_type_bf16:
            return 2368;
        case syn_type_float:
            return 5888;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSizeFromInt64(synDataType to)
{
    switch (to)
    {
        case syn_type_int32:
            return 3456;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSizeFromUint64(synDataType to)
{
    switch (to)
    {
        case syn_type_uint32:
            return 3456;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSizeFromFloat(synDataType to)
{
    switch (to)
    {
        case syn_type_int8:
            return 6496;
        case syn_type_uint8:
            return 6752;
        case syn_type_int16:
        case syn_type_uint16:
            return 2528;
        case syn_type_int32:
        case syn_type_uint32:
            return 5888;
        case syn_type_fp16:
        case syn_type_bf16:
            return 3040;
        case syn_type_fp8_152:
            return 6432;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSizeFromBfloat16(synDataType to)
{
    switch (to)
    {
        case syn_type_int8:
        case syn_type_uint8:
            return 6144;
        case syn_type_int16:
        case syn_type_uint16:
            return 5856;
        case syn_type_int32:
        case syn_type_uint32:
            return 6112;
        case syn_type_fp16:
            return 2304;
        case syn_type_float:
        case syn_type_fp8_152:
            return 6112;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSizeFromFloat16(synDataType to)
{
    switch (to)
    {
        case syn_type_int8:
        case syn_type_uint8:
            return 6240;
        case syn_type_int16:
        case syn_type_uint16:
            return 5856;
        case syn_type_int32:
        case syn_type_uint32:
            return 6112;
        case syn_type_bf16:
            return 2304;
        case syn_type_float:
            return 6112;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSizeFromFloat8(synDataType to)
{
    switch (to)
    {
        case syn_type_bf16:
            return 2400;
        case syn_type_float:
            return 6272;
        default:
            return 0;
    }
}

static inline unsigned getCastKernelSize(synDataType from, synDataType to)
{
    switch (from)
    {
        case syn_type_int8:
            return getCastKernelSizeFromInt8(to);
        case syn_type_uint8:
            return getCastKernelSizeFromUint8(to);
        case syn_type_int16:
            return getCastKernelSizeFromInt16(to);
        case syn_type_uint16:
            return getCastKernelSizeFromUint16(to);
        case syn_type_int32:
            return getCastKernelSizeFromInt32(to);
        case syn_type_uint32:
            return getCastKernelSizeFromUint32(to);
        case syn_type_int64:
            return getCastKernelSizeFromInt64(to);
        case syn_type_uint64:
            return getCastKernelSizeFromUint64(to);
        case syn_type_fp16:
            return getCastKernelSizeFromFloat16(to);
        case syn_type_bf16:
            return getCastKernelSizeFromBfloat16(to);
        case syn_type_float:
            return getCastKernelSizeFromFloat(to);
        case syn_type_fp8_152:
            return getCastKernelSizeFromFloat8(to);
        default:
            return 0;
    }
}

unsigned RecipeHal::getCastNodeRecipeMemoryRequirement(synDataType from, synDataType to) const
{
    // execution blob = 392, dynamic blob = 80, patching blob = 32
    constexpr unsigned CONSTANT_BLOBS_SIZE = 504;
    unsigned           kernelSize          = getCastKernelSize(from, to);
    EAGER_ASSERT(kernelSize > 0,
                 "non optimized code path for cast_{}_to_{} removal",
                 getDtypeSuffixFromSynDataType(from),
                 getDtypeSuffixFromSynDataType(to));
    return CONSTANT_BLOBS_SIZE + kernelSize;
}

PacketId RecipeHal::getQmanCommandOpcode(const Byte* data) const
{
    // command opcode has a fixed position for all packets, so NOP is good enough
    auto opcode = readAs<packet_nop>(data).opcode;
    switch (opcode)
    {
        case PACKET_WREG_32:
            return PacketId::WREG_32;
        case PACKET_WREG_BULK:
            return PacketId::WREG_BULK;
        case PACKET_FENCE:
            return PacketId::FENCE;
        case PACKET_NOP:
            return PacketId::NOP;
        case PACKET_CB_LIST:
            return PacketId::CB_LIST;
        case PACKET_WREG_64_LONG:
            return PacketId::WREG_64_LONG;
        default:
            EAGER_ASSERT(false, "getBlobOpcode called with unsupported packet type");
            return PacketId::MAX_ID;
    }
}

QmanCommandSizeType RecipeHal::getQmanCommandSize(PacketId opCode) const
{
    switch (opCode)
    {
        case PacketId::WREG_32:
            return sizeof(packet_wreg32);
        case PacketId::WREG_BULK:
            return sizeof(packet_wreg_bulk);
        case PacketId::FENCE:
            return sizeof(packet_fence);
        case PacketId::NOP:
            return sizeof(packet_nop);
        case PacketId::CB_LIST:
            return sizeof(packet_cb_list);
        case PacketId::WREG_64_LONG:
            return sizeof(packet_wreg64_long);
        default:
            EAGER_ASSERT(false, "getCommandSize called with unsupported packet type");
            return -1;
    }
}

EngineArcCommandId RecipeHal::getEcbCommandOpcode(unsigned opcodeVal) const
{
    switch (opcodeVal)
    {
        case ECB_CMD_LIST_SIZE:
            return EngineArcCommandId::LIST_SIZE;
        case ECB_CMD_NOP:
            return EngineArcCommandId::NOP;
        case ECB_CMD_WD_FENCE_AND_EXE:
            return EngineArcCommandId::WD_FENCE_AND_EXE;
        case ECB_CMD_SCHED_DMA:
            return EngineArcCommandId::SCHED_DMA;
        case ECB_CMD_STATIC_DESC_V2:
            return EngineArcCommandId::STATIC_DESC_V2;
        default:
            EAGER_ASSERT(false, "getArcCommandOpcode called with unsupported ecb command type");
            return EngineArcCommandId::COUNT;
    }
}

EngineArcCommandId RecipeHal::getEcbCommandOpcode(const Byte* data) const
{
    // command opcode has a fixed position for all packets, so NOP is good enough
    auto opcode = readAs<eng_arc_cmd_generic_t>(data).cmd_type;
    return getEcbCommandOpcode(opcode);
}

unsigned RecipeHal::getBaseRegistersCacheSize() const
{
    return hal::baseRegistersCacheSize;
}

unsigned RecipeHal::getRegForBaseAddress(unsigned regIndex) const
{
    return gaudi2::getRegForBaseAddress(regIndex);
}

StructSizeType RecipeHal::getFirstRegPosInDesc(EngineType engineType, SecondaryEngineType secondaryEngineTypes) const
{
    return getRegForLoadDesc(engineType2HabanaDeviceType(engineType), 0);
}

StructSizeType RecipeHal::getDescSize(EngineType engineType) const
{
    switch (engineType)
    {
        case EngineType::MME:
            return sizeof(MmeDesc);
        case EngineType::TPC:
            return sizeof(TpcDesc);
        case EngineType::DMA:
            return sizeof(DmaDesc);
        default:
            EAGER_ASSERT(false, "getDescSize called with unsupported device");
    };
    return 0;
}

StructSizeType RecipeHal::getWorkDistributionContextSize(EngineType engineType) const
{
    switch (engineType)
    {
        case EngineType::MME:
            return sizeof(mme_wd_ctxt_t);
        case EngineType::TPC:
            return sizeof(tpc_wd_ctxt_t);
        case EngineType::DMA:
            return sizeof(edma_wd_ctxt_t);
        default:
            EAGER_ASSERT(false, "getWorkDistributionContextSize called with unsupported device");
    };
    return 0;
}

unsigned RecipeHal::getMaxEngines(EngineType engineType) const
{
    switch (engineType)
    {
        case EngineType::TPC:
            return hal::numTpcEngines;
        case EngineType::MME:
            return hal::numMmeEngines;
        case EngineType::DMA:
            return hal::numInternalDmaEngines;
        default:
            EAGER_ASSERT(false, "getMaxEngines called with unsupported device");
    }
    return -1;
}

TensorsNrType RecipeHal::getMaxTpcTensorsNr() const
{
    // One patching point is reserved for the tpc program
    static_assert(maxTpcTensorsNr == hal::baseRegistersCacheSize - 1);
    return maxTpcTensorsNr;
}

ChipType RecipeHal::getChipType() const
{
    return ChipType::GAUDI2;
}

unsigned RecipeHal::getCacheLineSizeInBytes() const
{
    return hal::clSize;
}

unsigned RecipeHal::getVersionMinor() const
{
    return Gaudi2RecipeGenerator::getGaudi2VersionMinor();
}

AsicRegType RecipeHal::getQmanBlockStart() const
{
    return QMAN_BLOCK_BASE;
}

AsicRegType RecipeHal::getQmanBlockEnd() const
{
    return QMAN_BLOCK_BASE | BLOCK_OFFSET_MASK;
}

AsicRange RecipeHal::getMmeDescriptorBlackListRange(SecondaryEngineType secondaryEngineTypes) const
{
    constexpr AsicRegType rangeBase      = GET_ADDR_OF_MME_BLOCK_FIELD(arch_non_tensor_end);
    constexpr AsicRegType inclusiveFirst = rangeBase + offsetof(block_mme_non_tensor_descriptor, pcu);
    constexpr AsicRegType inclusiveLast  = rangeBase + offsetof(block_mme_non_tensor_descriptor, wkl_id);
    return {inclusiveFirst, inclusiveLast};
}

AsicRegType RecipeHal::getTpcDescriptorBlackListReg() const
{
    return GET_ADDR_OF_TPC_BLOCK_FIELD(tpc_cmd);
}

std::optional<AsicRange> RecipeHal::getTpcDescriptorWhiteListRange() const
{
    return std::nullopt;
}

StructSizeType RecipeHal::getMmeTensorAOffsetInDescriptor() const
{
    return offsetof(gaudi2::MmeDesc, baseAddrA);
}

StructSizeType RecipeHal::getMmeTensorBOffsetInDescriptor() const
{
    return offsetof(gaudi2::MmeDesc, baseAddrB);
}

StructSizeType RecipeHal::getMmeTensorCOffsetInDescriptor() const
{
    return offsetof(gaudi2::MmeDesc, baseAddrCOut0);
}

StructSizeType RecipeHal::getDmaTensorInputOffsetInDescriptor() const
{
    return offsetof(gaudi2::DmaDesc, ctx) + offsetof(gaudi2::block_dma_core_ctx, src_base_lo);
}

StructSizeType RecipeHal::getDmaTensorOutputOffsetInDescriptor() const
{
    return offsetof(gaudi2::DmaDesc, ctx) + offsetof(gaudi2::block_dma_core_ctx, dst_base_lo);
}

StructSizeType RecipeHal::getTpcFirstTensorOffsetInDescriptor() const
{
    return offsetof(gaudi2::TpcDesc, m_tensors);
}

StructSizeType RecipeHal::getTpcTensorsOffsetEndInDescriptor() const
{
    return offsetof(gaudi2::TpcDesc, m_tensors) + sizeof(gaudi2::TpcDesc::m_tensors);
}

StructSizeType RecipeHal::getTpcTensorSizeInDescriptor() const
{
    return sizeof(gaudi2::TpcDesc::m_tensors[0]);
}

mini_ecb_packets::Nop RecipeHal::getNopCommand(const Byte* data) const
{
    auto cmd = readAs<eng_arc_cmd_nop_t>(data);
    EAGER_ASSERT(cmd.cmd_type == ECB_CMD_NOP, "getNopCommand wrong command type");
    mini_ecb_packets::Nop ret = {};
    ret.padding               = cmd.padding;
    ret.switchCq              = cmd.switch_cq;
    ret.yield                 = cmd.yield;
    ret.dmaCompletion         = cmd.dma_completion;
    return ret;
}

mini_ecb_packets::StaticDescV2 RecipeHal::getStaticDescCommand(const Byte* data) const
{
    auto cmd = readAs<eng_arc_cmd_static_desc_v2_t>(data);
    EAGER_ASSERT(cmd.cmd_type == ECB_CMD_STATIC_DESC_V2, "getStaticDescCommand wrong command type");
    mini_ecb_packets::StaticDescV2 ret = {};
    ret.addrOffset                     = cmd.addr_offset;
    ret.size                           = cmd.size;
    ret.cpuIndex   = (cmd.cpu_index == CPU_ID_ALL) ? mini_ecb_packets::StaticDescV2::CPU_ID_ALL : cmd.cpu_index;
    ret.addrIndex  = cmd.addr_index;
    ret.yield      = cmd.yield;
    return ret;
}

mini_ecb_packets::ListSize RecipeHal::getListSizeCommand(const Byte* data) const
{
    auto cmd = readAs<eng_arc_cmd_list_size_t>(data);
    EAGER_ASSERT(cmd.cmd_type == ECB_CMD_LIST_SIZE, "getStaticDescCommand wrong command type");
    mini_ecb_packets::ListSize ret = {};
    ret.size                       = cmd.list_size;
    ret.topologyStart              = cmd.topology_start;
    ret.yield                      = cmd.yield;
    return ret;
}

mini_ecb_packets::SchedDma RecipeHal::getSchedDmaCommand(const Byte* data) const
{
    auto cmd = readAs<eng_arc_cmd_sched_dma_t>(data);
    EAGER_ASSERT(cmd.cmd_type == ECB_CMD_SCHED_DMA, "getStaticDescCommand wrong command type");
    mini_ecb_packets::SchedDma ret = {};
    ret.addrOffset                 = cmd.addr_offset;
    ret.size                       = cmd.size;
    ret.gcCtxtOffset               = cmd.gc_ctxt_offset;
    ret.addrIndex                  = cmd.addr_index;
    ret.yield                      = cmd.yield;
    return ret;
}

void RecipeHal::writeNopCommand(Byte* data, const mini_ecb_packets::Nop& cmd) const
{
    eng_arc_cmd_nop_t nopPacket = {};
    nopPacket.cmd_type          = ECB_CMD_NOP;
    nopPacket.yield             = cmd.yield;
    nopPacket.dma_completion    = cmd.dmaCompletion;
    nopPacket.switch_cq         = cmd.switchCq;
    nopPacket.padding           = cmd.padding;
    std::memcpy(data, &nopPacket, sizeof(nopPacket));
}

void RecipeHal::writeStaticDescCommand(Byte* data, const mini_ecb_packets::StaticDescV2& cmd) const
{
    eng_arc_cmd_static_desc_v2_t descPacket = {};
    descPacket.cmd_type                     = ECB_CMD_STATIC_DESC_V2;
    descPacket.yield                        = cmd.yield;
    descPacket.cpu_index   = (cmd.cpuIndex == mini_ecb_packets::StaticDescV2::CPU_ID_ALL) ? CPU_ID_ALL : cmd.cpuIndex;
    descPacket.size        = cmd.size;
    descPacket.addr_index  = cmd.addrIndex;
    descPacket.addr_offset = cmd.addrOffset;
    std::memcpy(data, &descPacket, sizeof(descPacket));
}

void RecipeHal::writeListSizeCommand(Byte* data, const mini_ecb_packets::ListSize& cmd) const
{
    eng_arc_cmd_list_size_t listSizePacket = {};
    listSizePacket.cmd_type                = ECB_CMD_LIST_SIZE;
    listSizePacket.yield                   = cmd.yield;
    listSizePacket.topology_start          = cmd.topologyStart;
    listSizePacket.list_size               = cmd.size;
    std::memcpy(data, &listSizePacket, sizeof(listSizePacket));
}

void RecipeHal::writeSchedDmaCommand(Byte* data, const mini_ecb_packets::SchedDma& cmd) const
{
    eng_arc_cmd_sched_dma_t schedDmaPacket = {};
    schedDmaPacket.cmd_type                = ECB_CMD_SCHED_DMA;
    schedDmaPacket.yield                   = cmd.yield;
    schedDmaPacket.addr_index              = cmd.addrIndex;
    schedDmaPacket.size                    = cmd.size;
    schedDmaPacket.gc_ctxt_offset          = cmd.gcCtxtOffset;
    schedDmaPacket.addr_offset             = cmd.addrOffset;
    std::memcpy(data, &schedDmaPacket, sizeof(schedDmaPacket));
}

void RecipeHal::writeFenceCommand(Byte* data, const mini_ecb_packets::Fence& cmd) const
{
    eng_arc_cmd_wd_fence_and_exec_t fencePacket = {};
    fencePacket.cmd_type                        = ECB_CMD_WD_FENCE_AND_EXE;
    fencePacket.yield                           = cmd.yield;
    fencePacket.dma_completion                  = cmd.dmaCompletion;
    fencePacket.wd_ctxt_id                      = cmd.wdCtxtId;
    std::memcpy(data, &fencePacket, sizeof(fencePacket));
}

mini_qman_packets::Wreg32 RecipeHal::getWreg32Command(const Byte* data) const
{
    auto cmd = readAs<packet_wreg32>(data);
    EAGER_ASSERT(cmd.opcode == PACKET_WREG_32, "getWreg32Command wrong command type");
    mini_qman_packets::Wreg32 ret = {};
    ret.regOffset                 = cmd.reg_offset;
    return ret;
}

mini_qman_packets::WregBulk RecipeHal::getWregBulkCommand(const Byte* data) const
{
    auto cmd = readAs<packet_wreg_bulk>(data);
    EAGER_ASSERT(cmd.opcode == PACKET_WREG_BULK, "getWregBulkCommand wrong command type");
    mini_qman_packets::WregBulk ret = {};
    ret.size64                      = cmd.size64;
    ret.regOffset                   = cmd.reg_offset;
    return ret;
}

mini_qman_packets::Wreg64Long RecipeHal::getWreg64LongCommand(const Byte* data) const
{
    auto cmd = readAs<packet_wreg64_long>(data);
    EAGER_ASSERT(cmd.opcode == PACKET_WREG_64_LONG, "getWreg64LongCommand wrong command type");
    mini_qman_packets::Wreg64Long ret = {};
    ret.dwEnable                      = cmd.dw_enable;
    ret.rel                           = cmd.rel;
    ret.dregOffset                    = cmd.dreg_offset;
    return ret;
}

void RecipeHal::printEcbCommand(const Byte* data) const
{
    switch (getEcbCommandOpcode(data))
    {
        case EngineArcCommandId::LIST_SIZE:
        {
            auto cmd = readAs<eng_arc_cmd_list_size_t>(data);
            LOG_DEBUG(GC_ARC,
                      "        ListSizeEngArcCommand cmd_type={}, yield={}, topology_start={}, list_size={}",
                      cmd.cmd_type,
                      cmd.yield,
                      cmd.topology_start,
                      cmd.list_size);
        }
        break;

        case EngineArcCommandId::NOP:
        {
            auto cmd = readAs<eng_arc_cmd_nop_t>(data);
            LOG_DEBUG(GC_ARC,
                      "        NopEngArcCommand cmd_type={}, yield={}, padding={}, switch_cq={}",
                      cmd.cmd_type,
                      cmd.yield,
                      cmd.padding,
                      cmd.switch_cq);
        }
        break;

        case EngineArcCommandId::WD_FENCE_AND_EXE:
        {
            auto cmd = readAs<eng_arc_cmd_wd_fence_and_exec_t>(data);
            LOG_DEBUG(GC_ARC,
                      "        DynamicWorkDistEngArcCommand cmd_type={}, yield={}, numDmaCompletion={}, wdCtxSlot={}",
                      cmd.cmd_type,
                      cmd.yield,
                      cmd.dma_completion,
                      cmd.wd_ctxt_id);
        }
        break;

        case EngineArcCommandId::SCHED_DMA:
        {
            auto cmd = readAs<eng_arc_cmd_sched_dma_t>(data);
            LOG_DEBUG(GC_ARC,
                      "        ScheduleDmaEngArcCommand cmd_type={}, yield={}, dataSize={}, srcAddrBaseId={}, "
                      "srcOffset={}, "
                      "dstOffset={}",
                      cmd.cmd_type,
                      cmd.yield,
                      cmd.size,
                      cmd.addr_index,
                      cmd.addr_offset,
                      cmd.gc_ctxt_offset);
        }
        break;

        case EngineArcCommandId::STATIC_DESC_V2:
        {
            auto cmd = readAs<eng_arc_cmd_static_desc_v2_t>(data);
            LOG_DEBUG(GC_ARC,
                      "        StaticCpDmaEngArcCommand cmd_type={}, yield={}, dataSize={}, srcAddrBaseId={}, "
                      "srcOffset={}, engId={}",
                      cmd.cmd_type,
                      cmd.yield,
                      cmd.size,
                      cmd.addr_index,
                      cmd.addr_offset,
                      cmd.cpu_index);
        }
        break;

        default:
            EAGER_ASSERT(false, "Unsupported ECB command");
            break;
    }
}

EcbCommandSizeType RecipeHal::getEcbCommandAlignmentConstraint(EcbType ecbType) const
{
    if (ecbType == EcbType::STATIC) return STATIC_COMPUTE_ECB_LIST_BUFF_SIZE;
    return DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;
}

unsigned RecipeHal::getMaxSupportedWorkDistributionContextCount() const
{
    return WD_CTXT_COUNT;
}

unsigned RecipeHal::getEcbNopCommandSize() const
{
    return ecb_packets_wrappers::nop::getCmdSize();
}

bool RecipeHal::isFenceCmdExistsInDynamicEcbCmdBuf() const
{
    return DynamicEcbCmdBuf::isFenceCmdExist();
}

// Given the commands layout is not expected to change we can simply rely on the current
// positions and sizes instead of re-calculating it at compilation\runtime.
// unlike the previous more general code for OffsetsOfCommands in blob_to_desc_structs.cpp

BlobSizeType RecipeHal::getWreg32RegPos(BlobSizeType cmdOffset) const
{
    // value field offset within packet_wreg32 is zero.
    return cmdOffset;
}

BlobSizeType RecipeHal::getWrBulkRegPos(BlobSizeType cmdOffset, AsicRegType regId) const
{
    return cmdOffset + sizeof(packet_wreg_bulk) + static_cast<BlobSizeType>(regId) * sizeOfAsicRegVal;
}

BlobSizeType RecipeHal::getWreg64LongOffsetPos(BlobSizeType cmdOffset) const
{
    return cmdOffset + offsetof(packet_wreg64_long, offset);
}

UnalignedPosType RecipeHal::getWreg64LongBasePos(BlobSizeType cmdOffset) const
{
    constexpr BitPosInBlobType    baseOffsetInBits = 37;
    constexpr BitPosInBlobType    baseBitCount     = 4;
    static const UnalignedPosType WREG64_LONG_BASE_POS(baseOffsetInBits, (1 << baseBitCount) - 1);
    return UnalignedPosType(WREG64_LONG_BASE_POS, cmdOffset);
}

}  // namespace eager_mode::gaudi2_spec_info