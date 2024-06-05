#pragma once

#include "recipe.h"
#include "types.h"

#include <stdint.h>
#include <unordered_map>
#include <set>
#include <limits>
#include <vector>

namespace patching
{
using SectionToSectionType = SmallVector<uint64_t, 8>;
using SectionTypesToPatch  = std::set<uint64_t>;

static const uint64_t INVALID_SECTION_TYPE_NUM = std::numeric_limits<uint64_t>::max();
static const uint64_t PP_TYPE_ID_ALL           = std::numeric_limits<uint64_t>::max();

static const uint64_t DEFAULT_SECTION_TYPE_ID = 0;
static const uint64_t PD_SECTION_TYPE_ID      = 1;

static const uint32_t INVALID_NODE_INDEX = std::numeric_limits<uint32_t>::max();
static const uint32_t LAST_VALID_NODE_INDEX =
    INVALID_NODE_INDEX - 2;  // -2 and not -1 because executable nodes start at index 1
static const uint32_t STAGED_SUBMISSION_PATCH_POINT_ADDITION = 1;
}  // namespace patching

// Should probably be moved into the recipe_t
struct data_chunk_patch_point_t
{
    // offset (in Bytes) to the patch-point location inside the data-chunk
    uint64_t offset_in_data_chunk;

    struct
    {
        uint64_t effective_address;
        uint64_t section_idx;
    } memory_patch_point;

    uint64_t data_chunk_index;
    uint32_t node_exe_index;  // holds node execution index, index 0 is reserved for general patch points, executable
                              // nodes start at index 1.
    // patching type
    patch_point_t::EPatchPointType type;
};

#pragma pack(push, 4)
struct DataChunkPatchPointsInfo
{
    DataChunkPatchPointsInfo() : m_dataChunkPatchPoints(nullptr), m_ppsPerDataChunkDbSize(0), m_singleChunkSize(0) {};

    ~DataChunkPatchPointsInfo() { delete[] m_dataChunkPatchPoints; };

    // all data chunk patch point + 1 dummy to mark the last patch point (with invalid node index)
    data_chunk_patch_point_t* m_dataChunkPatchPoints;

    uint64_t m_ppsPerDataChunkDbSize;

    uint32_t m_singleChunkSize;
};
#pragma pack(pop)

struct data_chunk_sm_patch_point_t
{
    EFieldType patch_point_type;  // The type of the field.

    union
    {
        struct
        {
            uint32_t data_chunk_index;
            // offset (in Bytes) to the patach-point location inside the data-chunk
            uint32_t offset_in_data_chunk;
        };

        struct
        {
            // Index of the patch point of low to be patched  (invalid = -1)
            uint32_t patch_point_idx_low;
            // Index of the patch point of high to be patched
            uint32_t patch_point_idx_high;
        };
    };

    // The size to be patched for the specific ROI descriptor. size in DW. used also for validation of the amount.
    uint32_t patch_size_dw;
    // Index of the ROI in the node rois. equal to the activation (descriptor). may be used by runtime to bypass
    // all blobs related to same activation index
    uint64_t roi_idx;

    // (pointer to origin DB in the recipe_t) will be used instead of the function pointer with the same params
    sm_function_id_t* p_smf_id;

    // (pointer to origin DB in the recipe_t) user params pointer to the user params used for the
    // specific node/descriptor.
    uint64_t* p_pp_metdata;

    bool is_unskippable;  // If true, cannot bypass this pp even if SMF decides so
};

#pragma pack(push, 4)
struct DataChunkSmPatchPointsInfo
{
    DataChunkSmPatchPointsInfo() : m_dataChunkSmPatchPoints(nullptr), m_singleChunkSize(0) {};

    ~DataChunkSmPatchPointsInfo() { delete[] m_dataChunkSmPatchPoints; };

    void init(uint64_t amountOfSmPatchPoints)
    {
        m_dataChunkSmPatchPoints = new data_chunk_sm_patch_point_t[amountOfSmPatchPoints];
    }

    data_chunk_sm_patch_point_t* m_dataChunkSmPatchPoints;
    uint32_t                     m_singleChunkSize;
};
#pragma pack(pop)