#include "recipe_arc_job_utils.h"

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_defs.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "infra/log_manager.h"

namespace eager_mode
{
// Info on commands in ECB chunk/block
struct EcbCommandsInfo
{
    unsigned           num  = 0;  // Number of commands
    EcbCommandSizeType size = 0;  // Size of commands without padding

    EcbCommandsInfo(const uint8_t*       cmds,
                    EcbCommandSizeType   cmdsSize,
                    bool                 ignorePadding,
                    const RecipeHalBase& recipeHal);
};

// Calc size of commands and their number w/ or w/o padding
EcbCommandsInfo::EcbCommandsInfo(const uint8_t*       cmds,
                                 EcbCommandSizeType   cmdsSize,
                                 bool                 ignorePadding,
                                 const RecipeHalBase& recipeHal)
{
    for (EcbCommandSizeType pos = 0; pos < cmdsSize;)
    {
        unsigned           padding     = 0;
        const Byte*        curCmd      = reinterpret_cast<const Byte*>(cmds + pos);
        EngineArcCommandId commandType = recipeHal.getEcbCommandOpcode(curCmd);
        EcbCommandSizeType commandSize = recipeHal.getEcbCommandSize(commandType);

        switch (commandType)
        {
            case EngineArcCommandId::LIST_SIZE:
            {
                EAGER_ASSERT(recipeHal.getListSizeCommand(curCmd).size != 0, "Invalid ECB");
            }
            break;

            case EngineArcCommandId::NOP:
            {
                padding = recipeHal.getNopCommand(curCmd).padding * nopPaddingUnits;
            }
            break;

            default:
                // validity check for unsupported command types is found in recipe hal
                break;
        }
        pos += commandSize + padding;
        size += commandSize + (ignorePadding ? 0 : padding);
        ++num;
    }
}

// Calculate ECB command sizes without padding and without head and tail
void EcbCmdsListInfo::initNetSize(const recipe_t& recipe, const RecipeHalBase& recipeHal)
{
    EAGER_ASSERT(recipe.arc_jobs_nr == 1, "Unsupported recipe template");
    const arc_job_t& arcJob = recipe.arc_jobs[0];

    // Static Ecb
    {
        // Get all sizes that are not CP DMA in bytes
        staticSz = EcbCommandsInfo(arcJob.static_ecb.cmds, arcJob.static_ecb.cmds_size, true, recipeHal).size;
        EAGER_ASSERT(staticSz != 0, "Invalid static ECB size");
        staticChunksNr = calcChunksNr(arcJob.static_ecb);
        EAGER_ASSERT(staticChunksNr != 0, "Invalid static ECB chunk size");
        staticSz /= staticChunksNr;
        const EcbCommandSizeType nonCpDmaSizes =
            recipeHal.getTemplateHeadSize(EcbType::STATIC) + recipeHal.getTailSize() + recipeHal.getSwitchingSize();
        EAGER_ASSERT(staticSz > nonCpDmaSizes, "Invalid static ECB");
        staticSz -= nonCpDmaSizes;
    }

    // Dynamic ECB
    {
        dynamicSz = EcbCommandsInfo(arcJob.dynamic_ecb.cmds, arcJob.dynamic_ecb.cmds_size, true, recipeHal).size;
        EAGER_ASSERT(dynamicSz != 0, "Invalid dynamic ECB size");
        EAGER_ASSERT(calcChunksNr(arcJob.dynamic_ecb) == 1, "Invalid dynamic ECB");
        const EcbCommandSizeType nonCpDmaSizes =
            recipeHal.getTemplateHeadSize(EcbType::DYNAMIC) + recipeHal.getTailSize() + recipeHal.getSwitchingSize();
        EAGER_ASSERT(dynamicSz > nonCpDmaSizes, "Invalid dynamic chunk ECB");
        dynamicSz -= nonCpDmaSizes;
    }
}

void printEcbArr(const uint8_t*       cmds,
                 EcbCommandSizeType   size,
                 uint32_t             chunksNr,
                 EcbCommandSizeType   sizeOfChunk,
                 const char*          name,
                 const RecipeHalBase& recipeHal)
{
    EAGER_ASSERT_PTR(cmds);
    EAGER_ASSERT(size != 0, "Invalid ECB size");
    EAGER_ASSERT(sizeOfChunk == (size / chunksNr), "Invalid ECB chunk size");
    const EcbCommandsInfo info(cmds, size, false, recipeHal);

    // Print summary
    LOG_DEBUG(GC_ARC, "    Num commands in {} ECB = {}, Size = {}", name, info.num, info.size);
    if (chunksNr == 1)
    {
        LOG_DEBUG(GC_ARC, "    Num engine-specific {} ECBs = 1", name);
    }
    else
    {
        LOG_DEBUG(GC_ARC, "    Num engine-specific {} ECBs = {}, Size of one = {}", name, chunksNr, sizeOfChunk);
    }
    EAGER_ASSERT(info.size == size, "Invalid commands size");

    // Print commands
    for (EcbCommandSizeType pos = 0; pos < size;)
    {
        const Byte*        curCmd      = reinterpret_cast<const Byte*>(cmds + pos);
        EngineArcCommandId commandType = recipeHal.getEcbCommandOpcode(curCmd);
        EcbCommandSizeType commandSize = recipeHal.getEcbCommandSize(commandType);
        pos += commandSize;
        recipeHal.printEcbCommand(curCmd);
        if (commandType == EngineArcCommandId::LIST_SIZE)
        {
            bool isValidCmd = recipeHal.getListSizeCommand(curCmd).size != 0;
            if (!isValidCmd) break;
        }
        else if (commandType == EngineArcCommandId::NOP)
        {
            pos += nopPaddingUnits * recipeHal.getNopCommand(curCmd).padding;
        }
    }
}

void printEcb(const ecb_t& ecb, const char* name, const RecipeHalBase& recipeHal)
{
    const uint32_t           chunksNr    = EcbCmdsListInfo::calcChunksNr(ecb);
    const EcbCommandSizeType sizeOfChunk = (chunksNr == 1) ? ecb.cmds_size : ecb.cmds_eng_offset;
    printEcbArr(ecb.cmds, ecb.cmds_size, chunksNr, sizeOfChunk, name, recipeHal);
}

void printAllEcbs(const arc_job_t& arcJob, const RecipeHalBase& recipeHal)
{
    Recipe::EngineType logicalId = arcJob.logical_engine_id;
    std::string_view   logicalIdStr;
    switch (logicalId)
    {
        case Recipe::EngineType::TPC:
            logicalIdStr = "TPC";
            break;
        case Recipe::EngineType::MME:
            logicalIdStr = "MME";
            break;
        case Recipe::EngineType::DMA:
            logicalIdStr = "DMA";
            break;
        case Recipe::EngineType::ROT:
            logicalIdStr = "ROT";
            break;
        case Recipe::EngineType::CME:
            logicalIdStr = "CME";
            break;
        default:
            EAGER_ASSERT(0, "Unrecognized engine");
            break;
    };
    LOG_DEBUG(GC_ARC, "Arc job for {}:", logicalIdStr);

    printEcb(arcJob.static_ecb, "static", recipeHal);
    printEcb(arcJob.dynamic_ecb, "dynamic", recipeHal);
}

void printAllJobs(const arc_job_t* arcJobs, ArcJobsNrType jobsNr, const RecipeHalBase& recipeHal)
{
    LOG_DEBUG(GC_ARC, "Number of Arc jobs = {}", jobsNr);
    for (ArcJobsNrType i = 0; i < jobsNr; ++i)
    {
        printAllEcbs(arcJobs[i], recipeHal);
    }
}

}  // namespace eager_mode