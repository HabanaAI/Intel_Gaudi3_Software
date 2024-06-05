#include "templates_creator.h"

// eager includes (relative to src/eager/lib/)
#include "chip_specification/gaudi3/recipe/mme_template_gen.h"
#include "chip_specification/gaudi3/recipe/recipe_hal.h"
#include "chip_specification/gaudi3/recipe/tpc_template_gen.h"
#include "recipe_gen/recipe_defs.h"

namespace eager_mode::gaudi3_spec_info
{
void TemplatesCreator::create(TemplatesOfChip& templatesOfChip)
{
    createMmeTemplates(templatesOfChip);
    createTpcTemplates(templatesOfChip);
}

void TemplatesCreator::createMmeTemplates(TemplatesOfChip& templatesOfChip)
{
    TemplatesOfEngine&             templates       = templatesOfChip[static_cast<EngineIdType>(EngineType::MME)];
    static constexpr TensorsNrType maxMmeTensorsNr = RecipeHalBase::maxMmeTensorsNr;
    templates.resize(maxMmeTensorsNr);
    static_assert(maxMmeTensorsNr == 3);

#define CREATE_MME_TEMPLATE(tensorsNr, secondaryMmeEngineType)                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        static_assert(tensorsNr <= maxMmeTensorsNr);                                                                   \
        using Creator                         = MmeTemplateCreator<tensorsNr>;                                         \
        using Builder                         = TemplateOfEngineBuilder<tensorsNr, Creator::calcMmeDataBufSize()>;     \
        TemplateOfEnginePtr& templateOfEngine = templates[tensorsNr - 1];                                              \
        templateOfEngine                      = std::make_unique<Builder>(Creator());                                  \
        templateOfEngine->descNr              = halFullChipSpecificInfo.numMmeEngines;                                 \
        templateOfEngine->secondaryEngineType = secondaryMmeEngineType;                                                \
        if constexpr (RecipeHal::supportConstExeBlob)                                                                  \
        {                                                                                                              \
            templateOfEngine->constExeBlobIndex = MmeDataBuffer<tensorsNr>::BlobsOrder::CONST_EXEC_BLOB;               \
        }                                                                                                              \
    } while (0)

    CREATE_MME_TEMPLATE(2, SecondaryEngineType::TRANSPOSE);
    CREATE_MME_TEMPLATE(3, SecondaryEngineType::NONE);
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
    CREATE_TPC_TEMPLATE(16);
#undef CREATE_TPC_TEMPLATE
}

}  // namespace eager_mode::gaudi3_spec_info