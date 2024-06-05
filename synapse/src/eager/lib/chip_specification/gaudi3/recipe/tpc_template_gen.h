#pragma once

// eager includes (relative to src/eager/lib/)
#include "chip_specification/gaudi3/recipe/recipe_hal_defs.h"
#include "chip_specification/gaudi3/recipe/template_structs.h"
#include "recipe_gen/recipe_defs.h"
#include "recipe_gen/recipe_templates_defs.h"
#include "recipe_gen/template_structs.h"
#include "utils/general_utils.h"
#include "utils/memory_utils.h"

// synapse-internal includes (relative to src/)
#include "include/recipe_version.h"

// synapse-internal gaudi3-specific includes (relative to src/)
#include "hal_reader/gaudi3/hal.h"
#include "platform/gaudi3/graph_compiler/queue_command.h"
#include "platform/gaudi3/graph_compiler/recipe_generator.h"

// relative to <synapse>/
#include "recipe.h"

namespace eager_mode::gaudi3_spec_info
{
using namespace gaudi3;

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct TpcTensorsDesc
///////////////////////////////////////////////////////////////////////////////////////////////////

// We decided to write the tensors block as single write bulk operation although this might include
// QM_TENSOR_x_BASE_ADDR_LOW/HIGH of each tensor (but 1st).
// This assumption obligate us to set these registers afterwards.
template<AsicRegType    tensorDescAddr,
         AsicRegType    subFirstTensorOffset,
         StructSizeType tensorDescSize,
         TensorsNrType  tensorsNr,
         bool           requiresWreg32 = tensorsNr % 2>
struct TpcTensorsDesc
{
    static constexpr BlobSizeType calcRegBulkSize()
    {
        static_assert(tensorDescSize > subFirstTensorOffset);
        const StructSizeType regsSizeOf1stTensor = tensorDescSize - subFirstTensorOffset;
        static_assert(tensorsNr != 0);
        const StructSizeType regsSizeOfRest = tensorDescSize * (tensorsNr - 1);
        return regsSizeOf1stTensor + regsSizeOfRest;
    }

    static constexpr BlobSizeType regBulkSize  = calcRegBulkSize();
    static constexpr AsicRegType  firstRegAddr = tensorDescAddr + subFirstTensorOffset;

    WrRegBulk<firstRegAddr, regBulkSize> wrRegBulk;

    void init() { wrRegBulk.init(/*switchBit*/ false); }
};

template<AsicRegType    tensorDescAddr,
         AsicRegType    subFirstTensorOffset,
         StructSizeType tensorDescSize,
         TensorsNrType  tensorsNr>
struct TpcTensorsDesc<tensorDescAddr, subFirstTensorOffset, tensorDescSize, tensorsNr, true>
{
    static constexpr BlobSizeType calcRegBulkSize()
    {
        static_assert(tensorDescSize > subFirstTensorOffset);
        const StructSizeType regsSizeOf1stTensor = tensorDescSize - subFirstTensorOffset;
        static_assert(tensorsNr != 0);
        const StructSizeType regsSizeOfRest = tensorDescSize * (tensorsNr - 1);
        return regsSizeOf1stTensor + regsSizeOfRest;
    }

    static constexpr BlobSizeType regBulkSize  = calcRegBulkSize();
    static constexpr AsicRegType  firstRegAddr = tensorDescAddr + subFirstTensorOffset;

    WrRegBulk_WrReg32<firstRegAddr, regBulkSize> wrRegBulk;

    void init() { wrRegBulk.init(/*switchBit*/ false); }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct TpcEngineDescBlob
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
struct TpcEngineDescBlob final
{
    using WriteRegBulkCommands = WrReg32_WrRegBulk<tpc_regs::inclusiveFirst, tpc_regs::descSize>;

    qman_packets_wrappers::wreg64_long wreg64_long[tensorsNr + /*kernel*/ 1];
    WriteRegBulkCommands       writeRegBulkCommands;
    qman_packets_wrappers::wreg32      cacheRegs[2];

    void init()
    {
        // Tensors
        for (TensorsNrType i = 0; i < tensorsNr; ++i)
        {
            const AsicRegType tensorBaseAddr = tpc_regs::tensorBaseAddr + i * tpc_regs::tensorDescSize;
            wreg64_long[i].init(tensorBaseAddr / sizeOfAsicRegVal);
        }
        // Kernel
        wreg64_long[tensorsNr].init(tpc_regs::kernelBaseAddr / sizeOfAsicRegVal);
        // Non-tensor descriptor
        writeRegBulkCommands.init(/*switchBit*/ false);
        cacheRegs[0].init(tpc_regs::ICACHE_AXI_CFG);
        cacheRegs[1].init(tpc_regs::DCACHE_AXI_CFG, /*switchBit*/ true);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct ExecBlob
///////////////////////////////////////////////////////////////////////////////////////////////////

// Execution blobs are different between engines, it's better to define them in engine's specific file rather than
// template_structs.h
template<TensorsNrType tensorsNr>
struct TpcMainExecBlob final
{
    using TpcTensorsDescType =
        TpcTensorsDesc<tpc_regs::tensorDescAddr, tpc_regs::subFirstTensorOffset, tpc_regs::tensorDescSize, tensorsNr>;

    InvalidateTpcCaches          invalidateTpcCaches;
    TpcTensorsDescType           tensorDesc;
    qman_packets_wrappers::wreg32 synObjMsg;
    TpcEngineDescBlob<tensorsNr> engineDescBlob;

    void init()
    {
        // First invalidate TPC caches
        static const uint32_t tpcCmdVal = InvalidateTPCCaches::calcTpcCmdVal();
        invalidateTpcCaches.init(tpc_regs::tpcCmdRegAddr, tpcCmdVal);
        // Tensors descriptor
        tensorDesc.init();
        // QM_SYNC_OBJECT_MESSAGE
        synObjMsg.init(tpc_regs::synObjMsgRegAddr, false);
        engineDescBlob.init();
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct ExecBlobForTpc
///////////////////////////////////////////////////////////////////////////////////////////////////

// All execution blobs
template<TensorsNrType tensorsNr>
struct ExecBlobForTpc final
{
    TpcMainExecBlob<tensorsNr> mainExecBlob;
    BaseRegLatencyWaBlob       baseRegLatencyWaBlob;  // Must be last to allow const blob reuse

    void init()
    {
        static_assert(offsetof(ExecBlobForTpc, baseRegLatencyWaBlob) + sizeof(baseRegLatencyWaBlob) == sizeof(*this),
                      "Const execution blob must be last to allow const blob reuse");
        // Main exec blob
        mainExecBlob.init();
        // Blob of cache base reg latency WA
        baseRegLatencyWaBlob.init();
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct TpcDataBuffer
///////////////////////////////////////////////////////////////////////////////////////////////////

// To begin, it's important to determine which chip structs are participating.
// It's recommended to list the structs in the order in which they appear in the data buffer.
// By default, all buffers are ordered in the same way as the recipe_t struct.
template<TensorsNrType tensorsNr>
struct TpcDataBuffer final
{
    // External access to template parameters
    static constexpr TensorsNrType getTensorsNr() { return tensorsNr; }

    // Enum to define order of TPC template's blobs
    // Note that current TPC instantiation assumes patching blob to be at index 0 and dynamic blob at index 1
    enum BlobsOrder
    {
        PATCHING,
        DYNAMIC,
        MAIN_EXEC_BLOB,
        // More can be added before this line..

        // Must be last:
        CONST_EXEC_BLOB,
        INVALID,
        BLOBS_NR = INVALID  // Specifies total number of blobs in TPC template
    };

    // Number of TPC descriptors (eventually reflected in number of static ECB chunks)
    static constexpr BlobsNrType descNr = 1;
    // Number of static blobs (that belong to execution or patching blob buffers)
    static constexpr BlobsNrType staticBlobsPerEngineENr = BlobsOrder::BLOBS_NR - (/*one dynamic blob*/ 1);

    // Compact aliases
    using TpcExecBlob     = ExecBlobForTpc<tensorsNr>;
    using TpcPatchingBlob = PatchingBlobsBuffer<tensorsNr + /*kernel*/ 1>;
    using TpcDynamicBlob  = tpc_wd_ctxt_t;
    using TpcArcJobs      = ArcJobs<descNr, staticBlobsPerEngineENr>;
    using TpcNodeExeList  = NodeExeList<tensorsNr>;
    using TpcPatchPoints  = PatchPoints<tensorsNr + /*kernel*/ 1>;

    // All fields, their names must comply with recipe_t's names
    TpcExecBlob           execution_blobs_buffer;
    TpcPatchingBlob       patching_blobs_buffer;
    TpcDynamicBlob        dynamic_blobs_buffer;
    blob_t                blobs[BlobsOrder::BLOBS_NR];
    TpcArcJobs            arc_jobs;
    persist_tensor_info_t tensors[tensorsNr];
    // program_data_blobs_buffer: Not required for TPC
    // program_data_blobs: Not required for TPC
    TpcPatchPoints   patch_points;
    TpcNodeExeList   node_exe_list;
    WorkspaceSizes   workspace_sizes;
    RecipeConfParams recipe_conf_params;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// class TpcTemplateCreator
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
class TpcTemplateCreator final : public TemplateOfEngineCreatorBase<tensorsNr>
{
public:
    virtual void create(recipe_t& recipe, Byte* dataBuffers, BlobSizeType dataBufSize) const override;
    static constexpr StructSizeType calcTpcDataBufSize();

    virtual std::optional<EcbCommandSizeType> getNopSizeForDynamicEcbMisalignment() const override
    {
        return std::nullopt;  // Misalignment is not relevant for TPC as it has one activation
    }

private:
    static void initRecipeStruct(recipe_t& recipe, TpcDataBuffer<tensorsNr>& dataBuf);
    static void initDataBuffers(TpcDataBuffer<tensorsNr>& dataBuf);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// class TpcTemplateCreator Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
constexpr StructSizeType TpcTemplateCreator<tensorsNr>::calcTpcDataBufSize()
{
    EAGER_ASSERT(tensorsNr <= maxTpcTensorsNr, "Unsupported number of TPC tensors for template recipe");
    return sizeof(TpcDataBuffer<tensorsNr>);
}

template<TensorsNrType tensorsNr>
void TpcTemplateCreator<tensorsNr>::create(recipe_t& recipe, Byte* dataBuffers, BlobSizeType dataBufSize) const
{
    EAGER_ASSERT(calcTpcDataBufSize() == dataBufSize, "Invalid data buffer size");
    // We did memset before to avoid uninitialized possible paddings
    auto& dataBuf = doExactPlacement<TpcDataBuffer<tensorsNr>>(dataBuffers);
    EAGER_ASSERT(sizeof(dataBuf) == dataBufSize, "Invalid data buffer size");
    initDataBuffers(dataBuf);
    initRecipeStruct(recipe, dataBuf);
}

template<TensorsNrType tensorsNr>
void TpcTemplateCreator<tensorsNr>::initRecipeStruct(recipe_t& recipe, TpcDataBuffer<tensorsNr>& dataBuf)
{
    static const auto recipeMinorVersion = Gaudi3RecipeGenerator::getGaudi3VersionMinor();
    recipe.version_major                 = RECIPE_VERSION_MAJOR;
    recipe.version_minor                 = recipeMinorVersion;

#define ASSIGN_PTR(FIELD)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        static_assert(alignof(decltype(*recipe.FIELD)) == alignof(decltype(dataBuf.FIELD)));                           \
        recipe.FIELD = reinterpret_cast<POINTER_TYPE(recipe_t, FIELD)*>(&dataBuf.FIELD);                               \
    } while (false)

    recipe.execution_blobs_buffer_size = sizeof(dataBuf.execution_blobs_buffer);
    ASSIGN_PTR(execution_blobs_buffer);
    recipe.patching_blobs_buffer_size = sizeof(dataBuf.patching_blobs_buffer);
    ASSIGN_PTR(patching_blobs_buffer);
    recipe.dynamic_blobs_buffer_size = sizeof(dataBuf.dynamic_blobs_buffer);
    ASSIGN_PTR(dynamic_blobs_buffer);
    recipe.blobs_nr = sizeof(dataBuf.blobs) / sizeof(blob_t);
    ASSIGN_PTR(blobs);
    recipe.arc_jobs_nr = 1;
    ASSIGN_PTR(arc_jobs);
    recipe.persist_tensors_nr = sizeof(dataBuf.tensors) / sizeof(persist_tensor_info_t);
    ASSIGN_PTR(tensors);
    recipe.patch_points_nr = sizeof(dataBuf.patch_points) / sizeof(patch_point_t);
    ASSIGN_PTR(patch_points);
    recipe.node_nr = 1;
    ASSIGN_PTR(node_exe_list);
    recipe.workspace_nr =
        sizeof(dataBuf.workspace_sizes.workspace_sizes) / sizeof(POINTER_TYPE(recipe_t, workspace_sizes));
    ASSIGN_PTR(workspace_sizes);
    recipe.recipe_conf_nr = sizeof(dataBuf.recipe_conf_params.recipe_conf_params) / sizeof(gc_conf_t);
    ASSIGN_PTR(recipe_conf_params);
#undef ASSIGN_PTR
}

template<TensorsNrType tensorsNr>
void TpcTemplateCreator<tensorsNr>::initDataBuffers(TpcDataBuffer<tensorsNr>& dataBuf)
{
    // Patching blob buffer
    {
        dataBuf.patching_blobs_buffer.init();
    }
    // Dynamic blob buffer
    {
        // Nothing to do, it should be overwritten anyway
    }  // Blobs
    {
        // blob of cache base reg latency WA
        blob_t& constExeBlob       = dataBuf.blobs[TpcDataBuffer<tensorsNr>::BlobsOrder::CONST_EXEC_BLOB];
        constExeBlob.blob_type_all = blob_t::EBlobType::EXE;
        constExeBlob.size          = sizeof(BaseRegLatencyWaBlob);
        constExeBlob.data          = &dataBuf.execution_blobs_buffer.baseRegLatencyWaBlob;
        // Patching blob
        blob_t& patchingBlob       = dataBuf.blobs[TpcDataBuffer<tensorsNr>::BlobsOrder::PATCHING];
        patchingBlob.blob_type_all = blob_t::EBlobType::PATCHING;
        patchingBlob.size          = sizeof(typename TpcDataBuffer<tensorsNr>::TpcPatchingBlob);
        patchingBlob.data          = &dataBuf.patching_blobs_buffer;
        // Dynamic blob
        blob_t& dynamicBlob       = dataBuf.blobs[TpcDataBuffer<tensorsNr>::BlobsOrder::DYNAMIC];
        dynamicBlob.blob_type_all = blob_t::EBlobType::DYNAMIC;
        dynamicBlob.size          = sizeof(typename TpcDataBuffer<tensorsNr>::TpcDynamicBlob);
        dynamicBlob.data          = &dataBuf.dynamic_blobs_buffer;
        // Main execution blob of TPC
        blob_t& tpcDescBlob       = dataBuf.blobs[TpcDataBuffer<tensorsNr>::BlobsOrder::MAIN_EXEC_BLOB];
        tpcDescBlob.blob_type_all = blob_t::EBlobType::EXE;
        tpcDescBlob.size          = sizeof(dataBuf.execution_blobs_buffer.mainExecBlob);
        tpcDescBlob.data          = &dataBuf.execution_blobs_buffer.mainExecBlob;
    }
    // ARC job
    {
        auto* objBase = reinterpret_cast<uint8_t*>(&dataBuf.arc_jobs);
        dataBuf.arc_jobs.initArcJob(Recipe::EngineType::TPC, objBase);
    }
    // Static ECB
    {
        static constexpr BlobsNrType chunkId = TpcDataBuffer<tensorsNr>::descNr - 1;
        static_assert(chunkId == 0);

        auto& staticBuf = dataBuf.arc_jobs.staticEcbCmdBuf[chunkId].ecbCmdBuf;
        // List size
        {
            staticBuf.list_size.init(sizeof(dataBuf.arc_jobs.staticEcbCmdBuf[chunkId]));
        }
        // Alignment NOP for previous command
        {
            staticBuf.nopForListSizeAlignment.init(/*paddingSize*/ 0, /*switchBit*/ false);
        }
        // Init static blobs commands
        static_assert(TpcDataBuffer<tensorsNr>::staticBlobsPerEngineENr == 3);
        using AddrOffsetType = ecb_packets_wrappers::static_desc_v2::AddrOffsetType;
        // Patching blob
        {
            static constexpr BlobSizeType   blobSize   = sizeof(typename TpcDataBuffer<tensorsNr>::TpcPatchingBlob);
            static constexpr AddrOffsetType addrOffset = 0;  // One patching blob only
            staticBuf.static_desc_v2[0].init(CPU_ID_ALL,
                                             /*yieldEn*/ false,
                                             blobSize,
                                             EngArcBufferAddrBase::PATCHING_ADDR_BASE,
                                             addrOffset);
        }
        // Cache base reg latency WA
        {
            static constexpr BlobSizeType blobSize =
                sizeof(TpcDataBuffer<tensorsNr>::TpcExecBlob::baseRegLatencyWaBlob);
            static constexpr AddrOffsetType addrOffset =
                offsetof(typename TpcDataBuffer<tensorsNr>::TpcExecBlob, baseRegLatencyWaBlob);
            staticBuf.static_desc_v2[1].init(CPU_ID_ALL,
                                             /*yieldEn*/ false,
                                             blobSize,
                                             EngArcBufferAddrBase::EXECUTE_ADDR_BASE,
                                             addrOffset);
        }
        // Descriptor of current engine
        {
            static constexpr BlobSizeType   blobSize = sizeof(TpcDataBuffer<tensorsNr>::TpcExecBlob::mainExecBlob);
            static constexpr AddrOffsetType addrOffsetBase =
                offsetof(typename TpcDataBuffer<tensorsNr>::TpcExecBlob, mainExecBlob);
            static constexpr AddrOffsetType addrOffset = addrOffsetBase + chunkId * blobSize;
            staticBuf.static_desc_v2[2].init(CPU_ID_ALL,
                                             /*yieldEn*/ true,
                                             blobSize,
                                             EngArcBufferAddrBase::EXECUTE_ADDR_BASE,
                                             addrOffset);
        }
        // Switching NOP
        {
            staticBuf.nopForSwitching.init(/*paddingSize*/ 0, /*switchBit*/ true);
        }
        // Padding NOP
        {
            static constexpr size_t paddingSizeInBytes = sizeof(dataBuf.arc_jobs.staticEcbCmdBuf[chunkId].padding);
            static_assert(paddingSizeInBytes % nopPaddingUnits == 0);
            static constexpr ecb_packets_wrappers::nop::PaddingSizeType paddingSize =
                paddingSizeInBytes / nopPaddingUnits;
            staticBuf.nopForPadding.init(paddingSize);
        }
    }
    // Dynamic ECB
    {
        dataBuf.arc_jobs.initDynamicEcb(sizeof(typename TpcDataBuffer<tensorsNr>::TpcDynamicBlob));
    }
    // Persistent tensors: initial value 0 is good enough, as those structure will be overwritten at instantiation
    // Patch points
    {
        dataBuf.patch_points.init(TpcDataBuffer<tensorsNr>::BlobsOrder::PATCHING);
    }
    // Node exe list
    {
        node_program_t& nodeExeList  = dataBuf.node_exe_list.node_exe_list;
        nodeExeList.program_blobs_nr = &dataBuf.node_exe_list.program_blobs_nr[0];
        nodeExeList.patch_points_nr  = TpcDataBuffer<tensorsNr>::TpcPatchPoints::getPatchPointsNr();
    }
    // Workspaces: initial values 0 are good enough, as those will be overwritten at instantiation
    // Recipe conf params
    {
        dataBuf.recipe_conf_params.init(synDeviceType::synDeviceGaudi3,
                                        halFullChipSpecificInfo.tpcEnginesMask,
                                        halFullChipSpecificInfo.numMmeEngines,
                                        hal::internalDmaEnginesMask);
    }
    // Execution blobs
    {
        dataBuf.execution_blobs_buffer.init();
    }
}

}  // namespace eager_mode::gaudi3_spec_info