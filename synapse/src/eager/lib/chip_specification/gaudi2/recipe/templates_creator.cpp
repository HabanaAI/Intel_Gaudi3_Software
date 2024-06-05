#include "templates_creator.h"

// eager includes (relative to src/eager/lib/)
#include "chip_specification/gaudi2/recipe/dma_template_gen.h"
#include "chip_specification/gaudi2/recipe/mme_template_gen.h"
#include "chip_specification/gaudi2/recipe/recipe_hal.h"
#include "chip_specification/gaudi2/recipe/tpc_template_gen.h"
#include "recipe_gen/recipe_defs.h"

namespace eager_mode::gaudi2_spec_info
{
void TemplatesCreator::create(TemplatesOfChip& templatesOfChip)
{
    createMmeTemplates(templatesOfChip);
    createTpcTemplates(templatesOfChip);
    createDmaTemplates(templatesOfChip);
}

void TemplatesCreator::createMmeTemplates(TemplatesOfChip& templatesOfChip)
{
    TemplatesOfEngine&             templates       = templatesOfChip[static_cast<EngineIdType>(EngineType::MME)];
    static constexpr TensorsNrType maxMmeTensorsNr = RecipeHalBase::maxMmeTensorsNr;
    templates.resize(maxMmeTensorsNr);
    static_assert(maxMmeTensorsNr == 3);
    using Creator                         = MmeTemplateCreator<maxMmeTensorsNr>;
    using Builder                         = TemplateOfEngineBuilder<maxMmeTensorsNr, Creator::calcMmeDataBufSize()>;
    TemplateOfEnginePtr& templateOfEngine = templates[maxMmeTensorsNr - 1];
    templateOfEngine                      = std::make_unique<Builder>(Creator());
    templateOfEngine->descNr              = TemplateMmeDataBuf::descNr;
    if constexpr (RecipeHal::supportConstExeBlob)
    {
        templateOfEngine->constExeBlobIndex = TemplateMmeDataBuf::BlobsOrder::CONST_EXEC_BLOB;
    }
}

void TemplatesCreator::createTpcTemplates(TemplatesOfChip& templatesOfChip)
{
    TemplatesOfEngine&      templates    = templatesOfChip[static_cast<EngineIdType>(EngineType::TPC)];
    constexpr TensorsNrType maxTensorsNr = maxTpcTensorsNr;
    templates.resize(maxTensorsNr);

#define CREATE_TPC_TEMPLATE(tensorsNr)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        static_assert(tensorsNr <= maxTensorsNr);                                                                      \
        using Creator                         = TpcTemplateCreator<tensorsNr>;                                         \
        using Builder                         = TemplateOfEngineBuilder<tensorsNr, Creator::calcTpcDataBufSize()>;     \
        TemplateOfEnginePtr& templateOfEngine = templates[tensorsNr - 1];                                              \
        templateOfEngine                      = std::make_unique<Builder>(Creator());                                  \
        static_assert(TpcDataBuffer<tensorsNr>::descNr == 1);                                                          \
        templateOfEngine->descNr = TpcDataBuffer<tensorsNr>::descNr;                                                   \
        if constexpr (RecipeHal::supportConstExeBlob)                                                                  \
        {                                                                                                              \
            templateOfEngine->constExeBlobIndex = TpcDataBuffer<tensorsNr>::BlobsOrder::CONST_EXEC_BLOB;               \
        }                                                                                                              \
    } while (0)
    EAGER_ASSERT(maxTensorsNr == 15, "createTpcTemplates wrong tensorsNr");
    CREATE_TPC_TEMPLATE(1);
    CREATE_TPC_TEMPLATE(2);
    CREATE_TPC_TEMPLATE(3);
    CREATE_TPC_TEMPLATE(4);
    CREATE_TPC_TEMPLATE(5);
    CREATE_TPC_TEMPLATE(6);
    CREATE_TPC_TEMPLATE(7);
    CREATE_TPC_TEMPLATE(8);
    CREATE_TPC_TEMPLATE(9);
    CREATE_TPC_TEMPLATE(10);
    CREATE_TPC_TEMPLATE(11);
    CREATE_TPC_TEMPLATE(12);
    CREATE_TPC_TEMPLATE(13);
    CREATE_TPC_TEMPLATE(14);
    CREATE_TPC_TEMPLATE(15);
#undef CREATE_TPC_TEMPLATE
}

void TemplatesCreator::createDmaTemplates(TemplatesOfChip& templatesOfChip)
{
    TemplatesOfEngine& templates = templatesOfChip[static_cast<EngineIdType>(EngineType::DMA)];
    templates.resize(RecipeHalBase::maxDmaTensorsNr);

#define CREATE_DMA_TEMPLATE(tensorsNr)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        static_assert(tensorsNr <= RecipeHalBase::maxDmaTensorsNr);                                                    \
        using Creator                         = DmaTemplateCreator<tensorsNr>;                                         \
        using Builder                         = TemplateOfEngineBuilder<tensorsNr, Creator::calcDmaDataBufSize()>;     \
        TemplateOfEnginePtr& templateOfEngine = templates[tensorsNr - 1];                                              \
        templateOfEngine                      = std::make_unique<Builder>(Creator());                                  \
        templateOfEngine->descNr              = 1; /* Only single NOP descriptor is included in DMA template */        \
        if constexpr (RecipeHal::supportConstExeBlob)                                                                  \
        {                                                                                                              \
            templateOfEngine->constExeBlobIndex = DmaDataBuffer<tensorsNr>::BlobsOrder::CONST_EXEC_BLOB;               \
        }                                                                                                              \
    } while (0)
    static_assert(RecipeHalBase::maxDmaTensorsNr == 2);
    CREATE_DMA_TEMPLATE(1);
    CREATE_DMA_TEMPLATE(2);
#undef CREATE_DMA_TEMPLATE
}

}  // namespace eager_mode::gaudi2_spec_info