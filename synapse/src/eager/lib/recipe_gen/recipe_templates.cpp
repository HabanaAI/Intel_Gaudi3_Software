#include "recipe_templates.h"

// eager includes (relative to src/eager/lib/)
#include "chip_info.h"
#include "chip_specification/gaudi2/recipe/templates_creator.h"
#include "chip_specification/gaudi3/recipe/templates_creator.h"
#include "recipe_gen/recipe_defs.h"
#include "recipe_gen/recipe_hal_base.h"
#include "recipe_gen/recipe_instantiation_mme.h"
#include "recipe_gen/recipe_instantiation.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_nodes/node_factory.h"
#include "graph_compiler/kernel_db.h"
#include "tpc_kernel_lib_interface.h"
namespace eager_mode
{
// Some aliases
using Blob2MmeDescMapCreator = Blob2DescMapCreator<EngineType::MME>;
using Blob2TpcDescMapCreator = Blob2DescMapCreator<EngineType::TPC>;
using Blob2DmaDescMapCreator = Blob2DescMapCreator<EngineType::DMA>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// RecipeTemplates Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

RecipeTemplates& RecipeTemplates::getInstance()
{
    static RecipeTemplates instance;
    return instance;
}

void RecipeTemplates::createAllTemplates()
{
    if (GCFG_ENABLE_EAGER_NOP_IN_RECIPE.value() && GCFG_ENABLE_EAGER_ARCH_OPTIMIZATIONS.value())
    {
        initNOPNodes();
    }

    if (m_isTemplatesCreated) return;
    m_isTemplatesCreated = true;

    // Create all templates for all chips
    gaudi2_spec_info::TemplatesCreator().create(m_templates[static_cast<unsigned>(ChipType::GAUDI2)]);
    gaudi3_spec_info::TemplatesCreator().create(m_templates[static_cast<unsigned>(ChipType::GAUDI3)]);

    // Fill desc/struct maps
    for (unsigned chipType = 0; chipType < static_cast<unsigned>(ChipType::CHIPS_NR); ++chipType)
    {
        const RecipeHalBase& recipeHal = ChipInfo::getRecipeHal(static_cast<ChipType>(chipType));
        TemplatesOfChip& tc = m_templates[chipType];
        for (unsigned engine = 0; engine < static_cast<unsigned>(EngineType::ENGINES_NR); ++engine)
        {
            TemplatesOfEngine& templatesOfEngine = tc[engine];
            for (unsigned tensors = 0; tensors < templatesOfEngine.size(); ++tensors)
            {
                auto& t = templatesOfEngine[tensors];
                if (t == nullptr) continue;
                switch (static_cast<EngineType>(engine))
                {
                    case EngineType::MME:
                        Blob2MmeDescMapCreator(t->blob2DescMaps, tensors + 1, recipeHal, t->secondaryEngineType)
                            .fillMaps(t->recipe);
                        MmeArcJobInstantiationInfoManager::getInstance(recipeHal.getChipType())
                            .onetimeInit(t->recipe.arc_jobs[0].static_ecb, recipeHal);
                        break;
                    case EngineType::TPC:
                        Blob2TpcDescMapCreator(t->blob2DescMaps, tensors + 1, recipeHal).fillMaps(t->recipe);
                        break;
                    case EngineType::DMA:
                        Blob2DmaDescMapCreator(t->blob2DescMaps, tensors + 1, recipeHal).fillMaps(t->recipe);
                        break;
                    default:
                        EAGER_ASSERT_0;
                        break;
                }
                t->ecbsNetSize.initNetSize(t->recipe, recipeHal);
                initSpecialBlobInfo(*t);
#ifndef NDEBUG
                RecipeInstantiation::checkTemplateAssumptions(*t, recipeHal);
#endif  //#ifndef NDEBUG
            }
        }
    }
}

const TemplateOfEngine& RecipeTemplates::getTemplate(ChipType chip, EngineType engine, TensorsNrType tensorsNr) const
{
    EAGER_ASSERT(tensorsNr != 0, "Invalid number of tensors");
    const TemplatesOfEngine& templates = m_templates[static_cast<unsigned>(chip)][static_cast<unsigned>(engine)];
    EAGER_ASSERT(templates.empty() == false, "Templates were not created for the given engine");
    EAGER_ASSERT(tensorsNr <= templates.size(), "Template does not support given number of tensors");
    const TemplateOfEngine* res = templates[tensorsNr - 1].get();
    EAGER_ASSERT(res != nullptr, "Template is not available for the given engine and numer of tensors");
    return *res;
}

const KernelInfo& RecipeTemplates::getNOPKernelInfo(ChipType chip) const
{
    return m_nopKernels[static_cast<unsigned>(chip)];
}

// Find the indices of patching and dynamic blobs
void RecipeTemplates::initSpecialBlobInfo(TemplateOfEngine& templateOfEngine)
{
    for (BlobsNrType i = 0; i < templateOfEngine.recipe.blobs_nr; ++i)
    {
        if (templateOfEngine.recipe.blobs[i].blob_type_all == blob_t::EBlobType::PATCHING)
        {
            EAGER_ASSERT(templateOfEngine.patchingBlobIndex == -1, "Multiple patching blobs detected in the template");
            templateOfEngine.patchingBlobIndex = i;
        }
        else if (templateOfEngine.recipe.blobs[i].blob_type_all == blob_t::EBlobType::DYNAMIC)
        {
            EAGER_ASSERT(templateOfEngine.dynamicBlobIndex == -1, "Multiple dynamic blobs detected in the template");
            templateOfEngine.dynamicBlobIndex = i;
        }
    }
    EAGER_ASSERT((templateOfEngine.patchingBlobIndex != -1) && (templateOfEngine.dynamicBlobIndex != -1),
                 "Invalid template");

    // Initialize const execution blob offset in static ECB
    if (templateOfEngine.constExeBlobIndex != -1)
    {
        EAGER_ASSERT(templateOfEngine.constExeBlobIndex < templateOfEngine.recipe.blobs_nr,
                     "Invalid const execution blob index");
        auto* constBlobPtr = static_cast<Byte*>(templateOfEngine.recipe.blobs[templateOfEngine.constExeBlobIndex].data);
        auto* execBufPtr   = reinterpret_cast<Byte*>(templateOfEngine.recipe.execution_blobs_buffer);
        EAGER_ASSERT(constBlobPtr >= execBufPtr, "Invalid template allocation");
        templateOfEngine.constExeBlobOffset = static_cast<BlobSizeType>(constBlobPtr - execBufPtr);
    }
}

void RecipeTemplates::initNOPNodes()
{
    m_nopKernels.fill({});
    // Not using TPC nop if not initialized - Case of tests which not initialize kernelDB
    if (!KernelDB::instance().initialized())
    {
        EAGER_REPORT_ERROR("Failed to load kernelDB");
        return;
    }

    const TSize inputSizes[] = {1};
    TensorPtr   inTensor     = std::make_shared<Tensor>(1, inputSizes, syn_type_float);

    initAndExtractKernelInfo(tpc_lib_api::DEVICE_ID_GAUDI2, inTensor, 1 /*kernelUniqueId*/);
    initAndExtractKernelInfo(tpc_lib_api::DEVICE_ID_GAUDI3, inTensor, 2 /*kernelUniqueId*/);
}

void RecipeTemplates::initAndExtractKernelInfo(tpc_lib_api::DeviceId device,
                                               const TensorPtr&      inTensor,
                                               uint32_t              kernelUniqueId)
{
    NodePtr nopNode = NodeFactory::createGenericTPCNode({inTensor}, {}, nullptr, TPCNode::NOP_KERNEL_NAME);
    if (nopNode != nullptr)
    {
        TPCNodePtr nopTPCNode = std::static_pointer_cast<TPCNode>(nopNode);
        if (nopTPCNode->init(device, nullptr, kernelUniqueId) == tpc_lib_api::GLUE_SUCCESS)
        {
            m_nopKernels[kernelUniqueId - 1] = nopTPCNode->getKernelInfo();
        }
    }
    if (m_nopKernels[kernelUniqueId - 1].kernelBinary == nullptr)
    {
        EAGER_REPORT_ERROR("Failed to init NOP kernel for gaudi{}", kernelUniqueId + 1);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// MME Template Creator
///////////////////////////////////////////////////////////////////////////////////////////////////
template<>
void Blob2MmeDescMapCreator::initEngineBlackList()
{
    // Tail of MME descriptor
    m_asicRegsBlackList.addBlackRange(m_recipeHal.getMmeDescriptorBlackListRange(m_secondaryEngineType));
}

template<>
PatchPointNrType Blob2MmeDescMapCreator::calcTensorId(StructSizeType structPos)
{
    // MME compute activation has two inputs (A, B) and one output (cOut0).
    // MME transpose activation has a single input (A) and one output (cOut0).
    if (structPos == m_recipeHal.getMmeTensorAOffsetInDescriptor())
    {
        return 0;  // Tensor A will become base-reg[0]
    }
    if (structPos == m_recipeHal.getMmeTensorBOffsetInDescriptor())
    {
        return 1;  // Tensor B will become base-reg[1] (not applicable to transpose)
    }
    if (structPos == m_recipeHal.getMmeTensorCOffsetInDescriptor())
    {
        return m_tensorsNr - 1;  // Tensor cOut0 will become base-reg[2] (or base-reg[1] for transpose)
    }
    EAGER_ASSERT(false, "Invalid MME tensor field position in descriptor");
    return -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TPC Template Creator
///////////////////////////////////////////////////////////////////////////////////////////////////

template<>
void Blob2TpcDescMapCreator::initEngineBlackList()
{
    m_asicRegsBlackList.addReg(m_recipeHal.getTpcDescriptorBlackListReg());

    auto whiteListRange = m_recipeHal.getTpcDescriptorWhiteListRange();
    if (whiteListRange.has_value())
    {
        m_asicRegsBlackList.addWhiteRange(*whiteListRange);
    }
}

template<>
PatchPointNrType Blob2TpcDescMapCreator::calcTensorId(StructSizeType structPos)
{
    const std::size_t offsetBegin = m_recipeHal.getTpcFirstTensorOffsetInDescriptor();
    const std::size_t offsetEnd   = m_recipeHal.getTpcTensorsOffsetEndInDescriptor();
    const std::size_t step        = m_recipeHal.getTpcTensorSizeInDescriptor();

    if (offsetBegin <= structPos && structPos < offsetEnd)
    {
        // Note that we want an int division to round down
        return (structPos - offsetBegin) / step;
    }

    // Kernel address is expected to get here, -1 is a valid value
    return -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// DMA Template Creator
///////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
template <> void Blob2DmaDescMapCreator::initEngineBlackList()
{
    // Yet, there is no need for a black list for DMA as would include all fields
    //      from 'axuser' to 'ctx' + offsetof(block_dma_core_ctx, pwrlp)
    // while excluding hb_wr_reduction to be atr white list. It's possible to be implemented as following.
    // In practice the blob constrains WReg32 command to hb_wr_reduction and WRegBulk to block_dma_core_ctx
    //
    // Add header to black list:
    //    constexpr AsicRegType inclusiveBlackFirst = GET_ADDR_OF_DMA_BLOCK_FIELD(axuser);
    //    constexpr AsicRegType inclusiveBlackLast = GET_ADDR_OF_DMA_BLOCK_FIELD(ctx) + offsetof(block_dma_core_ctx, pwrlp);
    //    m_asicRegsBlackList.addBlackRange({inclusiveBlackFirst, inclusiveBlackLast});
    // Exclude hb_wr_reduction field:
    //    constexpr AsicRegType inclusiveWhite = GET_ADDR_OF_DMA_BLOCK_FIELD(axuser) + offsetof(block_axuser, hb_wr_reduction);
    //    m_asicRegsBlackList.addWhiteRange({inclusiveWhite, inclusiveWhite});
}
#endif

template<>
PatchPointNrType Blob2DmaDescMapCreator::calcTensorId(StructSizeType structPos)
{
    // Handling memset operator which has one tensor
    if (m_tensorsNr == 1) return 0;

    // DMA has one input and one output.
    // Array of tensors field offsets to. Indices are the base regs
    const std::array<unsigned, RecipeHalBase::maxDmaTensorsNr> tensorOffsets = {
        m_recipeHal.getDmaTensorInputOffsetInDescriptor(),    // Src tensor will become base-reg[0]
        m_recipeHal.getDmaTensorOutputOffsetInDescriptor()};  // Dst tensor will become base-reg[1]

    PatchPointNrType tensorId = -1;
    for (PatchPointNrType i = 0; i < tensorOffsets.size(); ++i)
    {
        if (structPos == tensorOffsets[i])
        {
            tensorId = i;
            break;
        }
    }
    EAGER_ASSERT(tensorId != -1, "Invalid DMA tensor field position in descriptor");
    return tensorId;
}

}  // namespace eager_mode
