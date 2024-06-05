#pragma once

// eager includes (relative to src/eager/lib/)
#include "chip_specification/gaudi2/recipe/command_packets_wrappers.h"
#include "chip_specification/gaudi2/recipe/recipe_hal_defs.h"
#include "recipe_gen/recipe_defs.h"
#include "recipe_gen/template_structs_utils.h"
#include "utils/numeric_utils.h"

// relative to <synapse>/
#include "recipe.h"

// This file contains reusable structure for template creation of various engines.
// Note that none of the classes or structures in this file should define virtual methods or destructors.
// Keep this in mind when using these structures, and avoid dynamically allocating any of them.

namespace eager_mode::gaudi2_spec_info
{
// define aliases for register write sequences
template<AsicRegType FirstRegOffset, BlobSizeType RegBulkSize>
using WrRegBulk =
    WrRegBulk<qman_packets_wrappers::wreg32, qman_packets_wrappers::wreg_bulk, FirstRegOffset, RegBulkSize>;
template<AsicRegType FirstRegOffset, BlobSizeType RegBulkSize>
using WrReg32x2 =
    WrReg32x2<qman_packets_wrappers::wreg32, qman_packets_wrappers::wreg_bulk, FirstRegOffset, RegBulkSize>;
template<AsicRegType FirstRegOffset, BlobSizeType RegBulkSize>
using WrRegBulk_WrReg32 =
    WrRegBulk_WrReg32<qman_packets_wrappers::wreg32, qman_packets_wrappers::wreg_bulk, FirstRegOffset, RegBulkSize>;
template<AsicRegType FirstRegOffset, BlobSizeType RegBulkSize>
using WrReg32_WrRegBulk =
    WrReg32_WrRegBulk<qman_packets_wrappers::wreg32, qman_packets_wrappers::wreg_bulk, FirstRegOffset, RegBulkSize>;
template<AsicRegType FirstRegOffset, BlobSizeType RegBulkSize>
using WrReg32_WrRegBulk_WrReg32 = WrReg32_WrRegBulk_WrReg32<qman_packets_wrappers::wreg32,
                                                            qman_packets_wrappers::wreg_bulk,
                                                            FirstRegOffset,
                                                            RegBulkSize>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct PatchingBlobsBuffer
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
struct PatchingBlobsBuffer final
{
    static constexpr BlobSizeType sizeOfTensorsAddrDesc = tensorsNr * sizeOfAddressVal;
    // It's always one WREG_BULK command
    using WrBlkCmd = WrRegBulk<qman_regs::cacheBaseRegBase, sizeOfTensorsAddrDesc>;

    WrBlkCmd wreg_bulk;

    void init()
    {
        static_assert(sizeOfAddressVal == sizeof(uint64_t));
        wreg_bulk.init(false);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct StaticEcbCmdBuf
///////////////////////////////////////////////////////////////////////////////////////////////////

template<BlobsNrType staticBlobsPerEngineENr>
struct StaticEcbCmdBuf final
{
    ecb_packets_wrappers::list_size      list_size;
    ecb_packets_wrappers::nop            nopForListSizeAlignment;
    ecb_packets_wrappers::static_desc_v2 static_desc_v2[staticBlobsPerEngineENr];
    ecb_packets_wrappers::nop            nopForSwitching;
    ecb_packets_wrappers::nop            nopForPadding;
    // Padding..
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct DynamicEcbCmdBuf
///////////////////////////////////////////////////////////////////////////////////////////////////

struct DynamicEcbCmdBuf final
{
    ecb_packets_wrappers::list_size         list_size;
    ecb_packets_wrappers::nop               nopForSwitching;
    ecb_packets_wrappers::sched_dma         sched_dma;
    ecb_packets_wrappers::wd_fence_and_exec wd_fence_and_exec;
    ecb_packets_wrappers::nop               nopForFenceAlignment;
    ecb_packets_wrappers::nop               nopForPadding;
    // Padding..

    static inline bool isFenceCmdExist() { return true; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct PaddedEcbCmdBuf
///////////////////////////////////////////////////////////////////////////////////////////////////

template<class EcbCmdBuf, EcbCommandSizeType ecbChunkSize>
struct PaddedEcbCmdBuf final
{
    EcbCmdBuf ecbCmdBuf;
    Byte      padding[calcPaddingSize<size_t>(sizeof(EcbCmdBuf), ecbChunkSize)];
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct ArcJobs
///////////////////////////////////////////////////////////////////////////////////////////////////

template<BlobsNrType descNr, BlobsNrType staticBlobsPerEngineENr>
struct ArcJobs final
{
    arc_job_t arc_jobs;
    using StaticEcbCmdBufType =
        PaddedEcbCmdBuf<StaticEcbCmdBuf<staticBlobsPerEngineENr>, STATIC_COMPUTE_ECB_LIST_BUFF_SIZE>;
    using DynamicEcbCmdBufType = PaddedEcbCmdBuf<DynamicEcbCmdBuf, DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE>;
    StaticEcbCmdBufType  staticEcbCmdBuf[descNr];
    DynamicEcbCmdBufType dynamicEcbCmdBuf;

    // Note that because we did not made reinterpret case on the original data buffer: objBase != this
    void initArcJob(Recipe::EngineType engineType, uint8_t* objBase)
    {
        using ArcJobsType                   = ArcJobs<descNr, staticBlobsPerEngineENr>;
        arc_jobs.logical_engine_id          = engineType;
        arc_jobs.static_ecb.cmds            = objBase + offsetof(ArcJobsType, staticEcbCmdBuf);
        arc_jobs.static_ecb.cmds_size       = sizeof(staticEcbCmdBuf);
        arc_jobs.static_ecb.cmds_eng_offset = (descNr == 1) ? 0 : sizeof(staticEcbCmdBuf[0]);
        EAGER_ASSERT(arc_jobs.static_ecb.cmds_eng_offset % STATIC_COMPUTE_ECB_LIST_BUFF_SIZE == 0,
                     "Invalid static ECB command buffer structure");
        arc_jobs.dynamic_ecb.cmds      = objBase + offsetof(ArcJobsType, dynamicEcbCmdBuf);
        arc_jobs.dynamic_ecb.cmds_size = sizeof(dynamicEcbCmdBuf);
    }

    void initDynamicEcb(BlobSizeType sizeOfDynamicBuf)
    {
        auto& dynamicBuf = dynamicEcbCmdBuf.ecbCmdBuf;
        dynamicBuf.list_size.init(sizeof(dynamicEcbCmdBuf));
        // Switching NOP
        dynamicBuf.nopForSwitching.init(/*paddingSize*/ 0, /*switchBit*/ true);
        // SCHED_DMA
        static constexpr ecb_packets_wrappers::sched_dma::AddrOffsetType addrOffset =
            0;  // In recipe templates there is only one dynamic blob
        dynamicBuf.sched_dma.init(sizeOfDynamicBuf, addrOffset);
        // WD_FENCE_AND_EXE
        dynamicBuf.wd_fence_and_exec.init();
        // alignment NOP for previous command
        dynamicBuf.nopForFenceAlignment.init();
        // Padding NOP
        static constexpr size_t paddingSizeInBytes = sizeof(dynamicEcbCmdBuf.padding);
        static_assert(paddingSizeInBytes % sizeof(uint32_t) == 0);
        static constexpr ecb_packets_wrappers::nop::PaddingSizeType paddingSize = paddingSizeInBytes / sizeof(uint32_t);
        dynamicBuf.nopForPadding.init(paddingSize);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct BaseRegLatencyWaBlob
///////////////////////////////////////////////////////////////////////////////////////////////////

// Static blob for the H6-3262 workaround
struct BaseRegLatencyWaBlob final
{
    qman_packets_wrappers::wreg32 wreg32;
    qman_packets_wrappers::fence  fence;

    void init()
    {
        using namespace gaudi2;
        static constexpr AsicRegType regOffset =
            QMAN_BLOCK_BASE + offsetof(block_qman, cp_fence2_rdata) + sizeof(struct qman::reg_cp_fence2_rdata) * 4;
        wreg32.init(regOffset, false, 1);
        fence.init();
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct InvalidateTpcCaches
///////////////////////////////////////////////////////////////////////////////////////////////////

struct InvalidateTpcCaches
{
    qman_packets_wrappers::wreg32 wreg32;

    void init(AsicRegType regOffset, AsicRegValType val) { wreg32.init(regOffset, false, val, true); }
};

}  // namespace eager_mode::gaudi2_spec_info