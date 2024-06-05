#pragma once

// eager includes (relative to src/eager/lib/)
#include "chip_specification/gaudi2/recipe/recipe_hal_defs.h"
#include "chip_specification/gaudi2/recipe/template_structs.h"
#include "recipe_gen/recipe_defs.h"
#include "recipe_gen/recipe_hal_base.h"
#include "recipe_gen/recipe_templates_defs.h"
#include "recipe_gen/template_structs.h"
#include "utils/general_utils.h"
#include "utils/memory_utils.h"

// relative to <synapse>/
#include "recipe.h"

// synapse-internal includes (relative to src/)
#include "include/recipe_version.h"

// synapse-internal gaudi2-specific includes (relative to src/)
#include "platform/gaudi2/graph_compiler/command_queue.h"
#include "platform/gaudi2/graph_compiler/recipe_generator.h"

// std includes
#include <cstddef>

namespace eager_mode::gaudi2_spec_info
{
using namespace gaudi2;

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct MmeEngineDescBlob
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
struct MmeEngineDescBlob final
{
    qman_packets_wrappers::wreg64_long                              wreg64_long[tensorsNr];
    WrRegBulk_WrReg32<mme_regs::inclusiveFirst, mme_regs::descSize> writeRegBulkCommands;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct ExecBlobForMme
///////////////////////////////////////////////////////////////////////////////////////////////////

// Execution blobs are different between engines, it's better to define them it engin's specific file rather than
// template_structs.h
template<TensorsNrType tensorsNr, BlobsNrType descNr>
struct ExecBlobForMme final
{
    MmeEngineDescBlob<tensorsNr> engineDescBlob[descNr];
    BaseRegLatencyWaBlob         baseRegLatencyWaBlob;  // Must be last to allow const blob reuse

    void init(TensorAddressType firstTensorAddr)
    {
        static_assert(offsetof(ExecBlobForMme, baseRegLatencyWaBlob) + sizeof(baseRegLatencyWaBlob) == sizeof(*this),
                      "Const execution blob must be last to allow const blob reuse");
        // Create QMAN commands for each engine
        for (BlobsNrType d = 0; d < descNr; ++d)
        {
            auto& engBlob = engineDescBlob[d];
            for (TensorsNrType i = 0; i < tensorsNr; ++i)
            {
                engBlob.wreg64_long[i].init(firstTensorAddr / sizeOfAsicRegVal +
                                            static_cast<TensorAddressType>(i) * asicRegsPerEntry);
            }
            engBlob.writeRegBulkCommands.init(/*switchBit*/ true);
        }
        // blob of cache base reg latency WA
        baseRegLatencyWaBlob.init();
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct MmeDataBuffer
///////////////////////////////////////////////////////////////////////////////////////////////////

// To begin, it's important to determine which chip structs are participating.
// It's recommended to list the structs in the order in which they appear in the data buffer.
// By default, all buffers are ordered in the same way as the recipe_t struct.
template<TensorsNrType tensorsNr>
struct MmeDataBuffer final
{
    // External access to template parameters
    static constexpr TensorsNrType getTensorsNr() { return tensorsNr; }

    // Enum to define order of MME template's blobs
    // Note that current MME instantiation assumes patching blob to be at index 0 and dynamic blob at index 1
    enum BlobsOrder
    {
        PATCHING,
        DYNAMIC,
        SOUTH_ENGINE,
        NORTH_ENGINE,
        // More can be added before this line..

        // Must be last:
        CONST_EXEC_BLOB,
        INVALID,
        BLOBS_NR = INVALID  // Specifies total number of blobs in MME template
    };

    // Number of MME engines
    static constexpr BlobsNrType descNr = hal::numMmeEngines;
    // Number of static blobs (that belong to execution or patching blob buffers)
    static constexpr BlobsNrType staticBlobsPerEngineENr =
        BlobsOrder::BLOBS_NR - (/*one dynamic blob*/ 1) - (/*per engine*/ descNr - 1);

    // Compact aliases
    using MmeExecBlob     = ExecBlobForMme<tensorsNr, descNr>;
    using MmePatchingBlob = PatchingBlobsBuffer<tensorsNr>;
    using MmeDynamicBlob  = mme_wd_ctxt_t;
    using MmeArcJobs      = ArcJobs<descNr, staticBlobsPerEngineENr>;
    using MmePatchPoints  = PatchPoints<tensorsNr>;
    using MmeNodeExeList  = NodeExeList<tensorsNr>;

    // All fields, their names must comply with recipe_t's names
    MmeExecBlob           execution_blobs_buffer;
    MmePatchingBlob       patching_blobs_buffer;
    MmeDynamicBlob        dynamic_blobs_buffer;
    blob_t                blobs[BlobsOrder::BLOBS_NR];
    MmeArcJobs            arc_jobs;
    persist_tensor_info_t tensors[tensorsNr];
    // program_data_blobs_buffer: Not required for MME
    // program_data_blobs: Not required for MME
    PatchPoints<tensorsNr> patch_points;
    MmeNodeExeList         node_exe_list;
    WorkspaceSizes         workspace_sizes;
    RecipeConfParams       recipe_conf_params;
};

using TemplateMmeDataBuf = MmeDataBuffer<RecipeHalBase::maxMmeTensorsNr>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// class MmeTemplateCreator
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
class MmeTemplateCreator final : public TemplateOfEngineCreatorBase<tensorsNr>
{
public:
    virtual void create(recipe_t& recipe, Byte* dataBuffers, BlobSizeType dataBufSize) const override;
    static constexpr StructSizeType calcMmeDataBufSize();

    virtual std::optional<EcbCommandSizeType> getNopSizeForDynamicEcbMisalignment() const override
    {
        return ecb_packets_wrappers::nop::getCmdSize();
    }

private:
    static void initRecipeStruct(recipe_t& recipe, TemplateMmeDataBuf& dataBuf);
    static void initDataBuffers(TemplateMmeDataBuf& dataBuf);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// class MmeTemplateCreator Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
constexpr StructSizeType MmeTemplateCreator<tensorsNr>::calcMmeDataBufSize()
{
    static_assert(tensorsNr == RecipeHalBase::maxMmeTensorsNr, "Unsupported number of MME tensors for template recipe");
    return sizeof(MmeDataBuffer<tensorsNr>);
}

template<TensorsNrType tensorsNr>
void MmeTemplateCreator<tensorsNr>::create(recipe_t& recipe, Byte* dataBuffers, BlobSizeType dataBufSize) const
{
    EAGER_ASSERT(calcMmeDataBufSize() == dataBufSize, "Invalid data buffer size");
    // We did memset before to avoid uninitialized possible paddings
    auto& dataBuf = doExactPlacement<TemplateMmeDataBuf>(dataBuffers);
    EAGER_ASSERT(sizeof(dataBuf) == dataBufSize, "Invalid data buffer size");
    initDataBuffers(dataBuf);
    initRecipeStruct(recipe, dataBuf);
}

template<TensorsNrType tensorsNr>
void MmeTemplateCreator<tensorsNr>::initRecipeStruct(recipe_t& recipe, TemplateMmeDataBuf& dataBuf)
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
void MmeTemplateCreator<tensorsNr>::initDataBuffers(TemplateMmeDataBuf& dataBuf)
{
    // Patching blob buffer
    {
        dataBuf.patching_blobs_buffer.init();
    }
    // Dynamic blob buffer
    {
        TemplateMmeDataBuf::MmeDynamicBlob& dynamicBlobsBuf = dataBuf.dynamic_blobs_buffer;
        static const auto                   commitRegVal    = gaudi2::MmeQueue::getCommitRegVal();
        dynamicBlobsBuf.mme_commit_reg                      = commitRegVal;
        dynamicBlobsBuf.switch_bit                          = 1;
    }
    // Blobs
    {
        using BlobsOrder  = TemplateMmeDataBuf::BlobsOrder;
        using MmeExecBlob = TemplateMmeDataBuf::MmeExecBlob;
        // Const blob is defacto blob of cache base reg latency WA
        blob_t& constExeBlob       = dataBuf.blobs[BlobsOrder::CONST_EXEC_BLOB];
        constExeBlob.blob_type_all = blob_t::EBlobType::EXE;
        constExeBlob.size          = sizeof(BaseRegLatencyWaBlob);
        constExeBlob.data          = &dataBuf.execution_blobs_buffer.baseRegLatencyWaBlob;
        // Patching blob
        blob_t& patchingBlob       = dataBuf.blobs[BlobsOrder::PATCHING];
        patchingBlob.blob_type_all = blob_t::EBlobType::PATCHING;
        patchingBlob.size          = sizeof(TemplateMmeDataBuf::MmePatchingBlob);
        patchingBlob.data          = &dataBuf.patching_blobs_buffer;
        // Dynamic blob
        blob_t& dynamicBlob       = dataBuf.blobs[BlobsOrder::DYNAMIC];
        dynamicBlob.blob_type_all = blob_t::EBlobType::DYNAMIC;
        dynamicBlob.size          = sizeof(TemplateMmeDataBuf::MmeDynamicBlob);
        dynamicBlob.data          = &dataBuf.dynamic_blobs_buffer;
        // Descriptor of south engine
        blob_t& southMmeBlob       = dataBuf.blobs[BlobsOrder::SOUTH_ENGINE];
        southMmeBlob.blob_type_all = blob_t::EBlobType::EXE;
        southMmeBlob.size          = sizeof(MmeExecBlob::engineDescBlob[0]);
        southMmeBlob.data          = &dataBuf.execution_blobs_buffer.engineDescBlob[0];
        // Descriptor of north engine
        blob_t& northMmeBlob       = dataBuf.blobs[BlobsOrder::NORTH_ENGINE];
        northMmeBlob.blob_type_all = blob_t::EBlobType::EXE;
        northMmeBlob.size          = sizeof(MmeExecBlob::engineDescBlob[1]);
        northMmeBlob.data          = &dataBuf.execution_blobs_buffer.engineDescBlob[1];
    }
    // ARC job
    {
        auto* objBase = reinterpret_cast<uint8_t*>(&dataBuf.arc_jobs);
        dataBuf.arc_jobs.initArcJob(Recipe::EngineType::MME, objBase);
    }
    // Static ECB
    for (BlobsNrType i = 0; i < TemplateMmeDataBuf::descNr; ++i)
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
        static_assert(TemplateMmeDataBuf::staticBlobsPerEngineENr == 3);
        using AddrOffsetType = ecb_packets_wrappers::static_desc_v2::AddrOffsetType;
        // Patching blob
        {
            static constexpr BlobSizeType   blobSize   = sizeof(TemplateMmeDataBuf::MmePatchingBlob);
            static constexpr AddrOffsetType addrOffset = 0;  // One patching blob only
            staticBuf.static_desc_v2[0].init(i,
                                             /*yieldEn*/ false,
                                             blobSize,
                                             EngArcBufferAddrBase::PATCHING_ADDR_BASE,
                                             addrOffset);
        }
        // Cache base reg latency WA
        {
            static constexpr BlobSizeType   blobSize = sizeof(TemplateMmeDataBuf::MmeExecBlob::baseRegLatencyWaBlob);
            static constexpr AddrOffsetType addrOffset =
                offsetof(TemplateMmeDataBuf::MmeExecBlob, baseRegLatencyWaBlob);
            staticBuf.static_desc_v2[1].init(i,
                                             /*yieldEn*/ false,
                                             blobSize,
                                             EngArcBufferAddrBase::EXECUTE_ADDR_BASE,
                                             addrOffset);
        }
        // Descriptor of current engine
        {
            static constexpr BlobSizeType   blobSize = sizeof(TemplateMmeDataBuf::MmeExecBlob::engineDescBlob[i]);
            static constexpr AddrOffsetType addrOffsetBase =
                offsetof(TemplateMmeDataBuf::MmeExecBlob, engineDescBlob[0]);
            const AddrOffsetType addrOffset = addrOffsetBase + i * blobSize;
            staticBuf.static_desc_v2[2].init(i,
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
            static constexpr size_t paddingSizeInBytes = sizeof(dataBuf.arc_jobs.staticEcbCmdBuf[i].padding);
            static_assert(paddingSizeInBytes % nopPaddingUnits == 0);
            static constexpr ecb_packets_wrappers::nop::PaddingSizeType paddingSize =
                paddingSizeInBytes / nopPaddingUnits;
            staticBuf.nopForPadding.init(paddingSize);
        }
    }
    // Dynamic ECB
    {
        dataBuf.arc_jobs.initDynamicEcb(sizeof(TemplateMmeDataBuf::MmeDynamicBlob));
    }
    // Persistent tensors: initial value 0 is good enough, as those structure will be overwritten at instantiation
    // Patch points
    {
        dataBuf.patch_points.init(TemplateMmeDataBuf::BlobsOrder::PATCHING);
    }
    // Node exe list
    {
        node_program_t& nodeExeList  = dataBuf.node_exe_list.node_exe_list;
        nodeExeList.program_blobs_nr = &dataBuf.node_exe_list.program_blobs_nr[0];
        nodeExeList.patch_points_nr  = TemplateMmeDataBuf::MmePatchPoints::getPatchPointsNr();
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
        dataBuf.execution_blobs_buffer.init(mme_regs::firstTensorAddr);
    }
}

}  // namespace eager_mode::gaudi2_spec_info