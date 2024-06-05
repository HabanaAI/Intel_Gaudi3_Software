#pragma once

// eager includes (relative to src/eager/lib/)
#include "chip_specification/gaudi2/recipe/recipe_hal_defs.h"
#include "chip_specification/gaudi2/recipe/template_structs.h"
#include "recipe_gen/recipe_defs.h"
#include "recipe_gen/recipe_templates_defs.h"
#include "recipe_gen/template_structs.h"
#include "utils/general_utils.h"
#include "utils/memory_utils.h"

// relative to <synapse>/
#include "recipe.h"

// synapse-internal includes (relative to src/)
#include "hal_reader/gaudi2/hal.h"
#include "include/recipe_version.h"

// synapse-internal gaudi2-specific includes (relative to src/)
#include "platform/gaudi2/graph_compiler/command_queue.h"
#include "platform/gaudi2/graph_compiler/recipe_generator.h"

// std includes

namespace eager_mode::gaudi2_spec_info
{
using namespace gaudi2;

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct ExecBlob
///////////////////////////////////////////////////////////////////////////////////////////////////

// DMA template is different than other engines in that it represents a single NULL descriptor (NOP).
// With that being said, multiple DMA engines has no representation in this template except static ECB.
template<TensorsNrType tensorsNr>
struct DmaMainExecBlob final
{
    static constexpr BlobSizeType regBulkSize = dma_regs::exclusiveLast - dma_regs::inclusiveFirst;
    using WriteRegBulkCommands                = WrRegBulk<dma_regs::inclusiveFirst, regBulkSize>;

    qman_packets_wrappers::wreg64_long wreg64_long[tensorsNr];
    qman_packets_wrappers::wreg32      wreg32HbWrReduction;
    qman_packets_wrappers::wreg32      wreg32DstTSize0;
    WriteRegBulkCommands       writeRegBulkCommands;

    void init()
    {
        static_assert((tensorsNr == /*memset*/ 1) || (tensorsNr == /*memcpy*/ 2));
        // Tensors
        if constexpr (tensorsNr == /*memset*/ 1)
        {
            wreg64_long[0].init(dma_regs::dstTensorAddr / sizeOfAsicRegVal);
        }
        else if constexpr (tensorsNr == /*memcpy*/ 2)
        {
            wreg64_long[0].init(dma_regs::srcTensorAddr / sizeOfAsicRegVal);
            wreg64_long[1].init(dma_regs::dstTensorAddr / sizeOfAsicRegVal);
        }
        // Desc regs
        wreg32HbWrReduction.init(dma_regs::hbWrReduction);
        wreg32DstTSize0.init(dma_regs::dstTSize0);
        writeRegBulkCommands.init(/*switchBit*/ true);
        // Init the NULL descriptor AKA NOP
        static const AsicRegType     wrCompWdata        = gaudi2::DmaDescQueue::calcDescriptorSignaling();
        static constexpr AsicRegType compWdataRegOffset = dma_regs::compWdataRegAddr - dma_regs::inclusiveFirst;
        std::memcpy(writeRegBulkCommands.regBulkBuf + compWdataRegOffset, &wrCompWdata, sizeOfAsicRegVal);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct ExecBlobForDma
///////////////////////////////////////////////////////////////////////////////////////////////////

// All execution blobs
template<TensorsNrType tensorsNr>
struct ExecBlobForDma final
{
    DmaMainExecBlob<tensorsNr> mainExecBlob;
    BaseRegLatencyWaBlob       baseRegLatencyWaBlob;  // Must be last to allow const blob reuse

    void init()
    {
        static_assert(offsetof(ExecBlobForDma, baseRegLatencyWaBlob) + sizeof(baseRegLatencyWaBlob) == sizeof(*this),
                      "Const execution blob must be last to allow const blob reuse");
        // Blob of cache base reg latency WA
        baseRegLatencyWaBlob.init();
        // Main exec blob
        mainExecBlob.init();
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct DmaDataBuffer
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
struct DmaDataBuffer final
{
    // External access to template parameters
    static constexpr TensorsNrType getTensorsNr() { return tensorsNr; }

    // Enum to define order of DMA template's blobs
    // Note that current DMA instantiation assumes patching blob to be at index 0 and dynamic blob at index 1
    enum BlobsOrder
    {
        PATCHING,
        DYNAMIC,
        MAIN_EXEC_BLOB,
        // More can be added before this line..

        // Must be last:
        CONST_EXEC_BLOB,
        INVALID,
        BLOBS_NR = INVALID  // Specifies total number of blobs in DMA template
    };

    // Number of DMA engines
    static constexpr BlobsNrType descNr = hal::numInternalDmaEngines;
    // Number of static blobs (that belong to execution or patching blob buffers)
    static constexpr BlobsNrType staticBlobsPerEngineENr = BlobsOrder::BLOBS_NR - (/*one dynamic blob*/ 1);

    // Compact aliases
    using DmaExecBlob     = ExecBlobForDma<tensorsNr>;
    using DmaPatchingBlob = PatchingBlobsBuffer<tensorsNr>;
    using DmaDynamicBlob  = edma_wd_ctxt_t;
    using DmaArcJobs      = ArcJobs<descNr, staticBlobsPerEngineENr>;
    using DmaNodeExeList  = NodeExeList<tensorsNr>;
    using DmaPatchPoints  = PatchPoints<tensorsNr>;

    // All fields, their names must comply with recipe_t's names
    DmaExecBlob           execution_blobs_buffer;
    DmaPatchingBlob       patching_blobs_buffer;
    DmaDynamicBlob        dynamic_blobs_buffer;
    blob_t                blobs[BlobsOrder::BLOBS_NR];
    DmaArcJobs            arc_jobs;
    persist_tensor_info_t tensors[tensorsNr];
    // program_data_blobs_buffer: Not required for DMA
    // program_data_blobs: Not required for DMA
    DmaPatchPoints   patch_points;
    DmaNodeExeList   node_exe_list;
    WorkspaceSizes   workspace_sizes;
    RecipeConfParams recipe_conf_params;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// class DmaTemplateCreator
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
class DmaTemplateCreator final : public TemplateOfEngineCreatorBase<tensorsNr>
{
public:
    virtual void create(recipe_t& recipe, Byte* dataBuffers, BlobSizeType dataBufSize) const override;
    static constexpr StructSizeType calcDmaDataBufSize();

    virtual std::optional<EcbCommandSizeType> getNopSizeForDynamicEcbMisalignment() const override
    {
        return ecb_packets_wrappers::nop::getCmdSize();
    }

private:
    static void initRecipeStruct(recipe_t& recipe, DmaDataBuffer<tensorsNr>& dataBuf);
    static void initDataBuffers(DmaDataBuffer<tensorsNr>& dataBuf);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// class DmaTemplateCreator Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
constexpr StructSizeType DmaTemplateCreator<tensorsNr>::calcDmaDataBufSize()
{
    EAGER_ASSERT(tensorsNr <= RecipeHalBase::maxDmaTensorsNr, "Unsupported number of DMA tensors for template recipe");
    return sizeof(DmaDataBuffer<tensorsNr>);
}

template<TensorsNrType tensorsNr>
void DmaTemplateCreator<tensorsNr>::create(recipe_t& recipe, Byte* dataBuffers, BlobSizeType dataBufSize) const
{
    EAGER_ASSERT(calcDmaDataBufSize() == dataBufSize, "Invalid data buffer size");
    // We did memset before to avoid uninitialized possible paddings
    auto& dataBuf = doExactPlacement<DmaDataBuffer<tensorsNr>>(dataBuffers);
    EAGER_ASSERT(sizeof(dataBuf) == dataBufSize, "Invalid data buffer size");
    initDataBuffers(dataBuf);
    initRecipeStruct(recipe, dataBuf);
}

template<TensorsNrType tensorsNr>
void DmaTemplateCreator<tensorsNr>::initRecipeStruct(recipe_t& recipe, DmaDataBuffer<tensorsNr>& dataBuf)
{
    static const auto recipeMinorVersion = Gaudi2RecipeGenerator::getGaudi2VersionMinor();
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
void DmaTemplateCreator<tensorsNr>::initDataBuffers(DmaDataBuffer<tensorsNr>& dataBuf)
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
        // Const blob is defacto blob of cache base reg latency WA
        blob_t& constExeBlob       = dataBuf.blobs[DmaDataBuffer<tensorsNr>::BlobsOrder::CONST_EXEC_BLOB];
        constExeBlob.blob_type_all = blob_t::EBlobType::EXE;
        constExeBlob.size          = sizeof(BaseRegLatencyWaBlob);
        constExeBlob.data          = &dataBuf.execution_blobs_buffer.baseRegLatencyWaBlob;
        // Patching blob
        blob_t& patchingBlob       = dataBuf.blobs[DmaDataBuffer<tensorsNr>::BlobsOrder::PATCHING];
        patchingBlob.blob_type_all = blob_t::EBlobType::PATCHING;
        patchingBlob.size          = sizeof(typename DmaDataBuffer<tensorsNr>::DmaPatchingBlob);
        patchingBlob.data          = &dataBuf.patching_blobs_buffer;
        // Dynamic blob
        blob_t& dynamicBlob       = dataBuf.blobs[DmaDataBuffer<tensorsNr>::BlobsOrder::DYNAMIC];
        dynamicBlob.blob_type_all = blob_t::EBlobType::DYNAMIC;
        dynamicBlob.size          = sizeof(typename DmaDataBuffer<tensorsNr>::DmaDynamicBlob);
        dynamicBlob.data          = &dataBuf.dynamic_blobs_buffer;
        // Main execution blob of DMA
        blob_t& dmaDescBlob       = dataBuf.blobs[DmaDataBuffer<tensorsNr>::BlobsOrder::MAIN_EXEC_BLOB];
        dmaDescBlob.blob_type_all = blob_t::EBlobType::EXE;
        dmaDescBlob.size          = sizeof(dataBuf.execution_blobs_buffer.mainExecBlob);
        dmaDescBlob.data          = &dataBuf.execution_blobs_buffer.mainExecBlob;
    }
    // ARC job
    {
        auto* objBase = reinterpret_cast<uint8_t*>(&dataBuf.arc_jobs);
        dataBuf.arc_jobs.initArcJob(Recipe::EngineType::DMA, objBase);
    }
    // Static ECB
    for (BlobsNrType i = 0; i < DmaDataBuffer<tensorsNr>::descNr; ++i)
    {
        auto& staticBuf = dataBuf.arc_jobs.staticEcbCmdBuf[i].ecbCmdBuf;
        // List size
        {
            staticBuf.list_size.init(sizeof(dataBuf.arc_jobs.staticEcbCmdBuf[i]));
        }
        // Alignment NOP for previous command
        {
            staticBuf.nopForListSizeAlignment.init(/*paddingSize*/ 0, /*switchBit*/ false);
        }
        // Init static blobs commands
        static_assert(DmaDataBuffer<tensorsNr>::staticBlobsPerEngineENr == 3);
        using AddrOffsetType = ecb_packets_wrappers::static_desc_v2::AddrOffsetType;
        // Patching blob
        {
            static constexpr BlobSizeType   blobSize   = sizeof(typename DmaDataBuffer<tensorsNr>::DmaPatchingBlob);
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
                sizeof(DmaDataBuffer<tensorsNr>::DmaExecBlob::baseRegLatencyWaBlob);
            static constexpr AddrOffsetType addrOffset =
                offsetof(typename DmaDataBuffer<tensorsNr>::DmaExecBlob, baseRegLatencyWaBlob);
            staticBuf.static_desc_v2[1].init(CPU_ID_ALL,
                                             /*yieldEn*/ false,
                                             blobSize,
                                             EngArcBufferAddrBase::EXECUTE_ADDR_BASE,
                                             addrOffset);
        }
        // Descriptor of current engine
        {
            static constexpr BlobSizeType   blobSize = sizeof(DmaDataBuffer<tensorsNr>::DmaExecBlob::mainExecBlob);
            static constexpr AddrOffsetType addrOffsetBase =
                offsetof(typename DmaDataBuffer<tensorsNr>::DmaExecBlob, mainExecBlob);
            staticBuf.static_desc_v2[2].init(CPU_ID_ALL,
                                             /*yieldEn*/ true,
                                             blobSize,
                                             EngArcBufferAddrBase::EXECUTE_ADDR_BASE,
                                             addrOffsetBase);
        }
        // Switching NOP
        {
            staticBuf.nopForSwitching.init(/*paddingSize*/ 0, /*switchBit*/ true);
        }
        // Padding NOP
        {
            static constexpr size_t paddingSizeInBytes = sizeof(dataBuf.arc_jobs.staticEcbCmdBuf[i].padding);
            static_assert(paddingSizeInBytes % nopPaddingUnits == 0);
            static constexpr ecb_packets_wrappers::nop::PaddingSizeType paddingSize =
                paddingSizeInBytes / nopPaddingUnits;
            staticBuf.nopForPadding.init(paddingSize);
        }
    }
    // Dynamic ECB
    {
        dataBuf.arc_jobs.initDynamicEcb(sizeof(typename DmaDataBuffer<tensorsNr>::DmaDynamicBlob));
    }
    // Persistent tensors: initial value 0 is good enough, as those structure will be overwritten at instantiation
    // Patch points
    {
        dataBuf.patch_points.init(DmaDataBuffer<tensorsNr>::BlobsOrder::PATCHING);
    }
    // Node exe list
    {
        node_program_t& nodeExeList  = dataBuf.node_exe_list.node_exe_list;
        nodeExeList.program_blobs_nr = &dataBuf.node_exe_list.program_blobs_nr[0];
        nodeExeList.patch_points_nr  = DmaDataBuffer<tensorsNr>::DmaPatchPoints::getPatchPointsNr();
    }
    // Workspaces: initial values 0 are good enough, as those will be overwritten at instantiation
    // Recipe conf params
    {
        dataBuf.recipe_conf_params.init(synDeviceType::synDeviceGaudi2,
                                        hal::tpcEnginesMask,
                                        hal::numMmeEngines,
                                        hal::internalDmaEnginesMask);
    }
    // Execution blobs
    {
        dataBuf.execution_blobs_buffer.init();
    }
}

}  // namespace eager_mode::gaudi2_spec_info