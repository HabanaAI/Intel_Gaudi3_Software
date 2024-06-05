#pragma once

// eager includes (relative to src/eager/lib/)
#include "chip_specification/gaudi3/recipe/recipe_hal_defs.h"
#include "chip_specification/gaudi3/recipe/template_structs.h"
#include "recipe_gen/recipe_defs.h"
#include "recipe_gen/recipe_hal_base.h"
#include "recipe_gen/recipe_templates_defs.h"
#include "recipe_gen/template_structs.h"
#include "utils/general_utils.h"
#include "utils/memory_utils.h"

// synapse-internal includes (relative to src/)
#include "recipe_version.h"
#include "hal_reader/gaudi3/hal.h"

// synapse-internal gaudi3-specific includes (relative to src/)
#include "platform/gaudi3/graph_compiler/command_queue.h"
#include "platform/gaudi3/graph_compiler/recipe_generator.h"

// relative to <synapse>/
#include "recipe.h"

// std includes
#include <cstddef>

namespace eager_mode::gaudi3_spec_info
{
using namespace gaudi3;

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct MmeEngineDescBlob
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
struct MmeEngineDescBlob final
{
    static constexpr bool        isTranspose    = isMmeTranspose(tensorsNr);
    static constexpr AsicRegType inclusiveFirst = mme_regs::getInclusiveFirst(isTranspose);
    static constexpr AsicRegType descSize       = mme_regs::getDescSize(isTranspose);

    qman_packets_wrappers::wreg64_long          wreg64_long[tensorsNr];
    WrRegBulk_WrReg32<inclusiveFirst, descSize> writeRegBulkCommands;
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

    // Number of MME engines
    static constexpr BlobsNrType descNr = halFullChipSpecificInfo.numMmeEngines;

    // Enum to define order of MME template's blobs
    // Note that current MME instantiation assumes patching blob to be at index 0 and dynamic blob at index 1
    enum BlobsOrder
    {
        PATCHING,
        DYNAMIC,
        FIRST_ENGINE,
        LAST_ENGINE = FIRST_ENGINE + descNr - 1,
        // More can be added before this line..

        // Must be last:
        CONST_EXEC_BLOB,
        INVALID,
        BLOBS_NR = INVALID  // Specifies total number of blobs in MME template
    };

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
    static void initRecipeStruct(recipe_t& recipe, MmeDataBuffer<tensorsNr>& dataBuf);
    static void initDataBuffers(MmeDataBuffer<tensorsNr>& dataBuf);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// class MmeTemplateCreator Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

template<TensorsNrType tensorsNr>
constexpr StructSizeType MmeTemplateCreator<tensorsNr>::calcMmeDataBufSize()
{
    static_assert((tensorsNr == RecipeHalBase::maxMmeTensorsNr) || (tensorsNr == maxMmeTransposeTensorsNr),
                  "Unsupported number of MME tensors for template recipe");
    return sizeof(MmeDataBuffer<tensorsNr>);
}

template<TensorsNrType tensorsNr>
void MmeTemplateCreator<tensorsNr>::create(recipe_t& recipe, Byte* dataBuffers, BlobSizeType dataBufSize) const
{
    EAGER_ASSERT(calcMmeDataBufSize() == dataBufSize, "Invalid data buffer size");
    // We did memset before to avoid uninitialized possible paddings
    auto& dataBuf = doExactPlacement<MmeDataBuffer<tensorsNr>>(dataBuffers);
    EAGER_ASSERT(sizeof(dataBuf) == dataBufSize, "Invalid data buffer size");
    initDataBuffers(dataBuf);
    initRecipeStruct(recipe, dataBuf);
}

template<TensorsNrType tensorsNr>
void MmeTemplateCreator<tensorsNr>::initRecipeStruct(recipe_t& recipe, MmeDataBuffer<tensorsNr>& dataBuf)
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
void MmeTemplateCreator<tensorsNr>::initDataBuffers(MmeDataBuffer<tensorsNr>& dataBuf)
{
    // Patching blob buffer
    {
        dataBuf.patching_blobs_buffer.init();
    }
    // Dynamic blob buffer
    {
        typename MmeDataBuffer<tensorsNr>::MmeDynamicBlob& dynamicBlobsBuf = dataBuf.dynamic_blobs_buffer;
        // node type apart from transpose is unimportant
        constexpr bool isTranspose     = isMmeTranspose(tensorsNr);
        dynamicBlobsBuf.mme_commit_reg = gaudi3::MmeQueue::getCommitRegVal(isTranspose);
        dynamicBlobsBuf.mme_op_type    = gaudi3::MmeQueue::getOpType(isTranspose);
        dynamicBlobsBuf.switch_bit     = 1;
    }
    // Blobs
    {
        using BlobsOrder  = typename MmeDataBuffer<tensorsNr>::BlobsOrder;
        using MmeExecBlob = typename MmeDataBuffer<tensorsNr>::MmeExecBlob;
        // blob of cache base reg latency WA
        blob_t& constExeBlob       = dataBuf.blobs[BlobsOrder::CONST_EXEC_BLOB];
        constExeBlob.blob_type_all = blob_t::EBlobType::EXE;
        constExeBlob.size          = sizeof(BaseRegLatencyWaBlob);
        constExeBlob.data          = &dataBuf.execution_blobs_buffer.baseRegLatencyWaBlob;
        // Patching blob
        blob_t& patchingBlob       = dataBuf.blobs[BlobsOrder::PATCHING];
        patchingBlob.blob_type_all = blob_t::EBlobType::PATCHING;
        patchingBlob.size          = sizeof(typename MmeDataBuffer<tensorsNr>::MmePatchingBlob);
        patchingBlob.data          = &dataBuf.patching_blobs_buffer;
        // Dynamic blob
        blob_t& dynamicBlob       = dataBuf.blobs[BlobsOrder::DYNAMIC];
        dynamicBlob.blob_type_all = blob_t::EBlobType::DYNAMIC;
        dynamicBlob.size          = sizeof(typename MmeDataBuffer<tensorsNr>::MmeDynamicBlob);
        dynamicBlob.data          = &dataBuf.dynamic_blobs_buffer;
        // Engines descriptors
        constexpr BlobSizeType engineBlobSize = sizeof(MmeExecBlob::engineDescBlob[0]);
        uint8_t* firstEngineBlobOffset = reinterpret_cast<uint8_t*>(&dataBuf.execution_blobs_buffer.engineDescBlob[0]);
        for (unsigned engineIdx = 0; engineIdx < MmeDataBuffer<tensorsNr>::descNr; ++engineIdx)
        {
            blob_t& engineMmeBlob       = dataBuf.blobs[BlobsOrder::FIRST_ENGINE + engineIdx];
            engineMmeBlob.blob_type_all = blob_t::EBlobType::EXE;
            engineMmeBlob.size          = engineBlobSize;
            engineMmeBlob.data          = firstEngineBlobOffset + static_cast<size_t>(engineBlobSize) * engineIdx;
        }
    }
    // ARC job
    {
        auto* objBase = reinterpret_cast<uint8_t*>(&dataBuf.arc_jobs);
        dataBuf.arc_jobs.initArcJob(Recipe::EngineType::MME, objBase);
    }
    // Static ECB
    for (BlobsNrType i = 0; i < MmeDataBuffer<tensorsNr>::descNr; ++i)
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
        static_assert(MmeDataBuffer<tensorsNr>::staticBlobsPerEngineENr == 3);
        using AddrOffsetType = ecb_packets_wrappers::static_desc_v2::AddrOffsetType;
        // Patching blob
        {
            static constexpr BlobSizeType   blobSize   = sizeof(typename MmeDataBuffer<tensorsNr>::MmePatchingBlob);
            static constexpr AddrOffsetType addrOffset = 0;  // One patching blob only
            staticBuf.static_desc_v2[0].init(i,
                                             /*yieldEn*/ false,
                                             blobSize,
                                             EngArcBufferAddrBase::PATCHING_ADDR_BASE,
                                             addrOffset);
        }
        // Cache base reg latency WA
        {
            static constexpr BlobSizeType blobSize =
                sizeof(MmeDataBuffer<tensorsNr>::MmeExecBlob::baseRegLatencyWaBlob);
            static constexpr AddrOffsetType addrOffset =
                offsetof(typename MmeDataBuffer<tensorsNr>::MmeExecBlob, baseRegLatencyWaBlob);
            staticBuf.static_desc_v2[1].init(i,
                                             /*yieldEn*/ false,
                                             blobSize,
                                             EngArcBufferAddrBase::EXECUTE_ADDR_BASE,
                                             addrOffset);
        }
        // Descriptor of current engine
        {
            static constexpr BlobSizeType   blobSize = sizeof(MmeDataBuffer<tensorsNr>::MmeExecBlob::engineDescBlob[i]);
            static constexpr AddrOffsetType addrOffsetBase =
                offsetof(typename MmeDataBuffer<tensorsNr>::MmeExecBlob, engineDescBlob[0]);
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
        dataBuf.arc_jobs.initDynamicEcb(sizeof(typename MmeDataBuffer<tensorsNr>::MmeDynamicBlob));
    }
    // Persistent tensors: initial value 0 is good enough, as those structure will be overwritten at instantiation
    // Patch points
    {
        dataBuf.patch_points.init(MmeDataBuffer<tensorsNr>::BlobsOrder::PATCHING);
    }
    // Node exe list
    {
        node_program_t& nodeExeList  = dataBuf.node_exe_list.node_exe_list;
        nodeExeList.program_blobs_nr = &dataBuf.node_exe_list.program_blobs_nr[0];
        nodeExeList.patch_points_nr  = MmeDataBuffer<tensorsNr>::MmePatchPoints::getPatchPointsNr();
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
        constexpr bool        isTranspose     = isMmeTranspose(tensorsNr);
        constexpr AsicRegType firstTensorAddr = mme_regs::getFirstTensorAddr(isTranspose);
        dataBuf.execution_blobs_buffer.init(firstTensorAddr);
    }
}

}  // namespace eager_mode::gaudi3_spec_info