#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_defs.h"
#include "recipe_gen/recipe_hal_base.h"

// synapse api (relative to include/)
#include "internal/recipe.h"

namespace eager_mode
{
// Sizes of ECB commands without padding and without head and tail
struct EcbCmdsListInfo
{
    EcbCommandSizeType staticSz       = 0;  // Size in bytes of a single static ECB chunk
    EcbCommandSizeType dynamicSz      = 0;  // Size in bytes of dynamic ECB block
    uint32_t           staticChunksNr = 0;  // Number of chunks in the static ECB

    void initNetSize(const recipe_t& recipe, const RecipeHalBase& recipeHal);

    static uint32_t calcChunksNr(const ecb_t& ecb)
    {
        return (ecb.cmds_eng_offset == 0) ? 1 : (ecb.cmds_size / ecb.cmds_eng_offset);
    }
};

// Representation of position in global blob buffer, used track multiple writes cross different engines
class PositionInBlob
{
public:
    PositionInBlob& operator=(BitPosInBlobType pos)
    {
        m_posChanged = (m_pos != pos);
        m_pos        = pos;
        return *this;
    }
    operator BitPosInBlobType() const { return m_pos; }
    bool isPosChanged() const { return m_posChanged; }

private:
    BitPosInBlobType m_pos        = 0;      // Position of writer in global blob buffer
    bool             m_posChanged = false;  // Global position has changed or not
};

// Various interfaces top log ECB info
void printEcbArr(const uint8_t*       cmds,
                 EcbCommandSizeType   size,
                 uint32_t             chunksNr,
                 EcbCommandSizeType   sizeOfChunk,
                 const char*          name,
                 const RecipeHalBase& recipeHal);
void printEcb(const ecb_t& ecb, const char* name, const RecipeHalBase& recipeHal);
void printAllEcbs(const arc_job_t& arcJob, const RecipeHalBase& recipeHal);
void printAllJobs(const arc_job_t* arcJobs, ArcJobsNrType jobsNr, const RecipeHalBase& recipeHal);

}  // namespace eager_mode