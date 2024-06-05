#pragma once
#include <synapse_common_types.h>
#include <cstdint>
#include "synapse_common_types.h"
#include "tpc_kernel_lib_interface.h"
#include "habana_device_types.h"


namespace Recipe
{

enum EngineType : uint8_t
{
    TPC,
    MME,
    DMA,
    ROT,
    CME,
    // more can be added here

    // must be last:
    INVALID,
    ENGINES_NR = INVALID  // specifies total number of engine types in this enum
};

}  // namespace Recipe

struct blob_t
{
    enum EBlobType : uint8_t
    {
        PATCHING = 1,  // turns on first bit of blob_type
        EXE      = 2,  // turns on second bit of blob_type
        DYNAMIC  = 4   // turns on third bit of blob_type
    };

    void*    data;  // pointer to the data of the blob
    uint32_t size;  // size of the data in bytes
    union
    {
        struct
        {
            uint8_t requires_patching : 1;
            uint8_t static_exe : 1;
            uint8_t dynamic_exe : 1;
            uint8_t reserved : 5;
        } blob_type;
        EBlobType blob_type_all;
    };
};

struct program_t
{
    uint64_t *blob_indices;     // list of indices of the blob within the recipe that make up the complete program
    uint32_t program_length;    // denotes the number of dynamic CP_DMAs in the program - the size of blob_indices
};

struct job_t
{
    uint32_t engine_id;
    uint32_t program_idx;
};

struct persist_tensor_info_t
{
    const char* name;
    const char* layout;
    uint64_t    offset_in_section;   // offsets in bytes
    uint64_t    size;                // size in bytes
    double      zp;
    double      scale;
    TSize       dimensionsSize[HABANA_DIM_MAX];

    // TEMPORARY - Tensor's Multi-view
    // This database is only relevant for the PTs with multi-views
    uint64_t    multi_views_indices_nr; // Should be 0 for the permute-tensors
    uint32_t*   multi_views_indices;    // Should be nullptr for the permute-tensors

    uint32_t    elementType;         // synDataType
    uint32_t    dimensions;
    uint32_t    batchSize;
    uint32_t    extTensorExeOrder;
    uint16_t    section_idx;
    uint8_t     permutation[HABANA_DIM_MAX];
    uint8_t     section_type;
    uint8_t     tensorType;

    bool        isInput;             // whether the tensor is a graph input or output
    bool        isExternal;
};

struct const_section_t
{
    char*    data;           // section data
    uint64_t size;           // size in bytes
    uint16_t section_idx;    // section index
};

struct program_data_blob_t
{
    uint64_t   size;                  // size in bytes
    const char *data;                 // pointer to the data
    uint64_t   offset_in_section;     // offsets in bytes
    uint16_t   section_idx;           // will be 1 (for workspace 1)
};

struct node_program_t
{
    uint32_t* program_blobs_nr; // total number of blobs for each program up until (and including) the current node
    uint32_t  patch_points_nr;  // the total number of patch-points up until (and including) the current node
};

#define RECIPE_DEBUG_INFO_WORKING_ENGINES_SUPPORT 1 // temp define to work around cross-promotion
typedef HabanaDeviceType EDeviceType;
struct node_symbol_info_t
{
    EDeviceType device_type;          // the HW device type that the node is going to run on
    uint16_t    context_id;           // node context id (casted to uint16_t)
    uint32_t    full_context_id;      // full node context id
    uint32_t    num_descriptors;      // number of jobs into which node is divided
    uint32_t    kernel_blob_index;    // index of binary kernel data in kernelBlob array
    const char* node_name;            // name of the node as it appears in the application (null terminated)
    const char* operation;            // name of the operation (null terminated)
    const char* data_type;            // data type of the operation
    uint16_t    num_rois;             // number of logical rois in this node
    uint8_t*    num_working_engines;  // number of working engines in each logical roi
};

struct debug_info_t
{
    uint32_t             version_major;
    uint32_t             version_minor;
    uint16_t             recipe_id;
    uint32_t             num_nodes;          // number of nodes in the graph
    node_symbol_info_t*  nodes;              // array of nodes
    uint32_t             printf_addr_nr;     // number of printf addresses
    uint64_t*            printf_addr;        // array of printf addresses
    uint64_t             printf_section_idx; // for easy patching of the printf addresses
};

//--------------- Support for sync scheme debugging starts here ---------------
struct node_sync_info_arc_t  // For ARC-based platforms
{
    uint32_t           node_exe_index;
    uint16_t           pipe_level;
    uint16_t           emitted_signal;
    Recipe::EngineType engine_type;

    // HLDBG can infer the monitors from registers readout so the following fields are probably not needed
    //   uint16_t mme_dep;
    //   uint16_t tpc_dep;
    //   uint16_t dma_dep;
    //   uint16_t rot_dep;
    //   uint32_t reset_id;
};
struct node_sync_info_legacy_t  // For legacy platforms
{
    uint32_t           node_exe_index;
    uint16_t           pipe_level;
    uint16_t           emitted_signal;
    uint16_t           sob_id;  // the first physical engine sob ID, the rest can be inferred
    uint16_t           num_engines;
    Recipe::EngineType engine_type;

    // HLDBG can infer the monitors from registers readout so the following fields are probably not needed
    //   struct Monitor
    //   {
    //       uint16_t sob_id;
    //       uint16_t sob_value;
    //       uint8_t  mask;
    //   };
    //   uint8_t  mons_nr;
    //   Monitor* mons;
};
struct debug_sync_scheme_t
{
    uint64_t node_sync_info_nr;
    union
    {
        node_sync_info_arc_t*    node_sync_info_arc;     // used by ARC-based platforms
        node_sync_info_legacy_t* node_sync_info_legacy;  // used by legacy platforms
    };

    bool    sync_scheme_legacy_mode;
};
//--------------- Support for sync scheme debugging ends here ---------------

// a single patch point changes a single point
// if it is required to change multiple locations in a given blob, we will see multiple unrelated patch points
struct patch_point_t
{
    enum EPatchPointType : uint8_t
    {
        SIMPLE_DW_LOW_MEM_PATCH_POINT = 0,
        SIMPLE_DW_HIGH_MEM_PATCH_POINT,
        SIMPLE_DDW_MEM_PATCH_POINT,
        SOB_PATCH_POINT,

        PP_TYPE_LAST = SOB_PATCH_POINT
    };

    union
    {
        struct
        {
            uint64_t effective_address;
            uint16_t section_idx;
        } memory_patch_point;
        struct
        {
            uint32_t tensor_db_index;
        } sob_patch_point;
    };

    uint32_t blob_idx;            // the blob to patch
    uint32_t dw_offset_in_blob;   // offset (in 32bit word units) to the word that is being patched within the blob

    uint32_t        node_exe_index;     // holds node execution index, index 0 is reserved for general patch points, executable nodes start at index 1.
    EPatchPointType type;               // patching type
};

struct section_group_t
{
    uint32_t *patch_points_index_list;  // holds list of indices to the main patch points array
    uint32_t patch_points_nr;           // holds the number of patch point for the section
    uint8_t  section_group;             // holds the section group
};

struct section_blobs_t
{
    uint32_t  *blob_indices;           // list of patching blob indices of section id
    uint32_t  blobs_nr;                // number of patching blob indices for the section id
    uint16_t  section_idx;             // holds the section id
};

struct gc_conf_t
{
    // Note: DO NOT change the order of the enum. New enum values should be added at the bottom,
    // increasing LAST_COMPILE_PARAM appropriately
    enum recipeCompileParams : uint8_t
    {
        DEVICE_TYPE            = 0,
        TIME_STAMP             = 1,  // seconds since the Epoch
        TPC_ENGINE_MASK        = 2,
        MME_NUM_OF_ENGINES     = 3,
        DMA_ENGINE_MASK        = 4,
        RCP_NOT_USED           = 5,  // not used
        ROTATOR_NUM_OF_ENGINES = 6,  // greco only
        LAST_COMPILE_PARAM     = 7   // should be last
        // more can be added
    };

    uint64_t            conf_value;
    recipeCompileParams conf_id;
};

//--------------- Support for ARC architecture starts here ---------------
struct ecb_t               // Engine Command Buffer
{
    uint8_t* cmds;             // consecutive buffer holding engine-arc commands for the entire workload
    uint32_t cmds_size;        // size in bytes of the cmds buffer, multiple of ECB_CHUNK_SIZE
    uint32_t cmds_eng_offset;  // offset in bytes to engine-specific commands (if any), multiple of ECB_CHUNK_SIZE
};

struct arc_job_t  // use this structure to construct DIPATCH_COMPUTE_ECBLIST_CMD
{
    ecb_t              static_ecb;         // the static ecb which uses static blobs only (patching and execution)
    ecb_t              dynamic_ecb;        // the dynamic ecb which uses dynamic blobs only
    uint32_t           engines_filter;     // bitmask allowing to exclude physical engines from this work scheduling
    Recipe::EngineType logical_engine_id;  // the target logical engine for this work scheduling
};
//--------------- Support for ARC architecture ends here ---------------

struct recipe_t
{
    uint32_t              version_major;
    uint32_t              version_minor;

    uint32_t              nameSize;
    char*                 name;

    uint64_t              execution_blobs_buffer_size;   // holds the execution blobs buffer size
    uint64_t              *execution_blobs_buffer;       // holds all execution blobs (program code)

    uint64_t              patching_blobs_buffer_size;    // holds the patching blobs buffer size
    uint64_t              *patching_blobs_buffer;        // holds all patching blobs

    uint64_t              dynamic_blobs_buffer_size;     // holds the dynamic blobs buffer size
    uint32_t              *dynamic_blobs_buffer;         // holds all dynamic blobs

    uint64_t              blobs_nr;                      // The number of unique blobs in the recipe
    blob_t                *blobs;                        // points to execution_blobs_buffer & patching_blobs_buffer,

    uint32_t              programs_nr;      // The number of programs on the recipe.
    program_t             *programs;        // The recipe's programs.

    uint32_t              activate_jobs_nr; // The number of jobs in the recipe for the activate stage. A job is a program executed by a HW engine.
    job_t                 *activate_jobs;   // The actual jobs for the activate stage.
                                            // For example, if all 8 TPCs are executing the same program, there will be 1 program and 8 jobs

    uint32_t              execute_jobs_nr;  // The number of jobs in the recipe for the execute stage.
    job_t                 *execute_jobs;    // The actual jobs for the execute stage.

    uint32_t              arc_jobs_nr;      // The number of ARC jobs in the recipe (naively 4 for tpc, mme, dma, rot)
    arc_job_t             *arc_jobs;        // The actual ARC jobs

    uint32_t              persist_tensors_nr; // The number of persist tensors in the persist tensors list that require patching
    persist_tensor_info_t *tensors;           // Pointer to the persist tensor compilation information list

    uint32_t              h2di_tensors_nr;  // Number of H2D intermediate tensors

    // TEMPORARY - Tensor's Multi-view
    // For supporting views over a single persistent-tensor, which actually holds numerous tensors
    uint64_t              permute_tensors_views_nr; // The number of permutations infos
    persist_tensor_info_t *permute_tensors_views;   // Pointer to the permute tensors' compilation information-list

    uint32_t              const_sections_nr;    // The number of const sections
    const_section_t       *const_sections;      // List of const sections

    uint64_t              program_data_blobs_size;        // holds program data blobs total size
    char                  *program_data_blobs_buffer;     // holds all program data blobs (program_data_blobs)
    uint32_t              program_data_blobs_nr;
    program_data_blob_t   *program_data_blobs;            // points to program_data_blobs_buffer

    uint32_t              patch_points_nr;   // The total number of patch points
    uint32_t              activate_patch_points_nr; // The number of patch points in activate program
    patch_point_t         *patch_points;     // Pointer to the patch points array

    uint32_t              sections_nr;       // only used by synapse to verify that patching address of all sections is resolved

    uint32_t              section_groups_nr;            // The number of section groups
    section_group_t       *section_groups_patch_points; // patch points list per section group container

    section_group_t       sobj_section_group_patch_points;

    uint32_t              section_ids_nr;         // The number of section ids
    section_blobs_t       *section_blobs_indices; // blob indices list per section id

    uint32_t              node_nr;        // The number of nodes
    node_program_t*       node_exe_list;  // list of nodes ordered by execution

    uint32_t              workspace_nr;      // number of workspaces
    uint64_t              *workspace_sizes;  // workspace size in bytes for each workspace. When reporting the workspace size to the user
                                             // it shall be the sum of all the sizes. workspace size must be a multiple of 128 bytes.

    debug_info_t          debug_profiler_info;     // profiler-related info (when enabled by compilation flag)
    debug_sync_scheme_t   debug_sync_scheme_info;  // sync scheme info for debugging purposes

    uint32_t              recipe_conf_nr;      // holds the number of habana global conf params
    gc_conf_t*            recipe_conf_params;  // holds global conf param values

    uint64_t              nop_kernel_offset;   // holds nop kernel offset
    uint64_t              nop_kernel_section;  // holds nop kernel section
    bool                  valid_nop_kernel;    // do we have valid nop kernel

    uint16_t              max_used_mcid_discard;// number of used mcids for discard (gaudi3+)
    uint16_t              max_used_mcid_degrade;// number of used mcids for degrade (gaudi3+)
};

////////////////////////////////////////////////////////////////////////////////////////////
// Dynamic shapes recipe starts here

struct sm_function_id_t
{
    union
    {
        struct
        {
            uint64_t sm_funcid : 56;
            uint64_t sm_tableid : 8;
        };
        uint64_t sm_func_index;
    };
};

struct shape_tensor_info_t
{
    const char* name;  // used to resolve synLaunch shape tensor name to id.
};

// tensor shape information.
typedef tpc_lib_api::TensorShapeInfo tensor_shape_infer_info_t;

// tensor shape information.
struct tensor_info_t
{
    enum ETensorType : uint8_t
    {
        PERSISTENT_TENSOR = 0,
        SHAPE_TENSOR      = 1,
        INTERNAL_TENSOR   = 2,
    };

    // Bit flags. When adding new flags, use (1<<n) values.
    enum ETensorFlags : uint8_t
    {
        NO_FLAGS         = 0,
        HAS_HOST_ADDRESS = (1 << 0),
    };

    tensor_shape_infer_info_t infer_info;
    const char*               tensor_info_name;
    uint64_t                  section_offset;
    TSize                     max_dims[MAX_DIMENSIONS_NUM];  // holds for each tensor the bucket max size
    TSize                     min_dims[MAX_DIMENSIONS_NUM];  // holds for each tensor the bucket min size
    TStride                   strides[MAX_DIMENSIONS_NUM];   // holds for each tensor the strides required
    uint32_t                  data_type;                     // holds the data type of the tensor

    uint32_t                  tensor_db_index;               // holds the index in the recipe_t persistent tensor/shape tensor database,
                                                             // uint max if its internal
    synTensorType             user_tensor_type;              // holds the type of the tensor as specified by the user
    ETensorType               tensor_type;                   // holds the type of the tensor persistent/shape/internal data tensor
    ETensorFlags              tensor_flags;
    uint8_t                   permutation[MAX_DIMENSIONS_NUM];  // holds permutation for each tensor
};

// single ROI mapping from the tensor
struct tensor_roi_t
{
    TSize roi_offset_dims[MAX_DIMENSIONS_NUM];  // offset of the specific ROI offset from the tensor base used for
                                                // the activation
    TSize roi_size_dims[MAX_DIMENSIONS_NUM];    // size of the ROI for the activation
};

// holds the relvant information for each descriptor. the ROI its mapping and the list of the patch points for it.
struct roi_info_t
{
    uint32_t      roi_in_tensor_nr;
    uint32_t      roi_out_tensor_nr;
    tensor_roi_t* roi_in_tensors;
    tensor_roi_t* roi_out_tensors;
    TSize         index_space_offset[MAX_DIMENSIONS_NUM];
    TSize         index_space_size[MAX_DIMENSIONS_NUM];
};

enum EFieldType : uint8_t
{
    FIELD_ADDRESS_PART_LOW = 1,
    FIELD_ADDRESS_PART_HIGH,
    FIELD_ADDRESS_PART_FULL,
    FIELD_DYNAMIC_EXECUTE_WITH_SIGNAL,
    FIELD_DYNAMIC_EXECUTE_NO_SIGNAL,
    FIELD_DYNAMIC_EXECUTE_MME,
    FIELD_DYNAMIC_DMA_SRC,
    FIELD_DYNAMIC_DMA_DST,
    FIELD_DYNAMIC_DST_BULK_SIZE_STRIDE,
    FIELD_DYNAMIC_DST_LAST_STRIDE,
    FIELD_DYNAMIC_SRC_BULK_SIZE_STRIDE,
    FIELD_DYNAMIC_SRC_LAST_STRIDE,
    FIELD_DYNAMIC_MME_VALID_ELEMENTS,
    FIELD_DYNAMIC_TPC_SIZE,
    FIELD_DYNAMIC_TPC_STRIDE,
    FIELD_DYNAMIC_TPC_TID,
    FIELD_DYNAMIC_ADDRESS,
    FIELD_DYNAMIC_TEST,
    FIELD_SYNC_OBJECT_ADDRESS,
    FIELD_DYNAMIC_DMA_COMMIT,
    FIELD_DYNAMIC_MME_COMMIT,
    FIELD_DYNAMIC_MME_SYNC,
    FIELD_DYNAMIC_OFFSET,
    FIELD_DYNAMIC_MME_PADDING,
    FIELD_CM_MCID,
    FIELD_DYNAMIC_TPC_TID_GAUDI2,
    FIELD_DYNAMIC_TPC_TID_GAUDI3,
    _LAST
};
// descriptor patch points that contain the Shape manipulation method
struct sm_patch_point_t
{
    uint64_t* metadata;       // a buffer aligned to uint64_t, will contain a comy of one of the metadata structs

    union
    {
        struct
        {
            uint32_t blob_idx;           // index of the blob for direct access of the blob
            uint32_t dw_offset_in_blob;  // offset within the blob for the specific patch point
        };
        struct
        {
            uint32_t patch_point_idx_low;   // index of the patch point of low to be patched  (invalid = -1)
            uint32_t patch_point_idx_high;  // index of the patch point of high to be patched
        };
    };
    uint32_t patch_size_dw;  // the size to be patched for the specific ROI descriptor. size in DW. used also for
                             // validation of the amount.
    uint32_t roi_idx;        // index of the ROI in the node rois. equal to the activation (descriptor). may be used by
                             // runtime to bypass all blobs related to same activation index

    sm_function_id_t smf_id;  // will be used instead of the function pointer with the same params
    uint16_t  metadata_size;  // size of the buffer, in uint64_t units

    EFieldType patch_point_type;  // The type of the field.
    bool       is_unskippable;    // If true, cannot bypass this pp even if SMF decides so
};

using sif_permutation_t = tpc_lib_api::NodeTensorPermutation;

static const size_t MAX_SHAPE_PLANE_NODE_NAME_LEN = 90;

struct shape_plane_basic_node_t
{
    enum EShapePlanceTensorDb : uint8_t
    {
        GRAPH_TENSOR_DB = 0,
        NODE_TENSOR_DB  = 1
    };

    uint32_t             *input_tensors;      // If corresponding nput_tensors_db element is GRAPH_TENSOR_DB, then
                                              // these are indices in shape_plane_graph_t::sp_tensors
                                              // If set to NODE_TENSOR_DB, they are indices in
                                              // shape_plane_node_t::node_db_tensors.
    uint32_t             *output_tensors;     // ditto

    EShapePlanceTensorDb *input_tensors_db;   // Array indicating per input tensor to which db it belongs
    EShapePlanceTensorDb *output_tensors_db;  // Array indicating per output tensor to which db it belongs

    uint8_t              *sif_params;         // params to be used by the sif function
    sif_permutation_t    *input_permutations;
    sm_function_id_t     sif_id;              // will be used instead of the function pointer with the same params
    uint32_t             sif_params_nr;       // the size of the sif params.
    uint64_t             sif_version;         // The version of the sif function (Its library version)

    uint16_t             input_tensors_nr;
    uint16_t             output_tensors_nr;

    uint16_t             input_permutations_nr;

    // Debug Section
    char node_name[MAX_SHAPE_PLANE_NODE_NAME_LEN];
};

struct shape_plane_node_t
{
    static const unsigned SCRATCH_PAD_SIZE = 12;

    struct shape_plane_basic_node_t *basic_nodes;

    sm_patch_point_t                *node_patch_points;         // include the list of patch points for the node.
                                                                //including tensor level shared patch points
                                                                // and per descriptor patch points including execute/signaling and dims
                                                                // per descriptor patch point order is first execute and then dims.
    uint32_t                        *input_tensors;             // input tensor list for the node in the same order as for the node Kernels execution (tensor_info_t)
    uint32_t                        *output_tensors;            // tensor_info_t

    tensor_info_t                   *node_db_tensors;           // If the node is fused, these are tensors that were elininated by fusion

    uint32_t                        *output_src_tensors;        //output tensors indexes that need to be copied as src
    uint32_t                        *output_dst_tensors;        //output tensors indexes that need to be updated by the copy (dst).

    roi_info_t                      *activation_rois;           //holds the list of rois for all activations (descriptors) of the specific node. referenced by the patch points
    uint64_t                        nodeData[SCRATCH_PAD_SIZE]; //Node scratch pad. used by the smf to pass state between functions within the node scope.
    sm_function_id_t                sif_id;                     // will be used instead of the function pointer with the same params

    uint32_t                        node_patch_points_nr;
    uint32_t                        activation_rois_nr;

    uint16_t                        basic_nodes_nr;
    uint16_t                        input_tensors_nr;
    uint16_t                        output_tensors_nr;
    uint16_t                        node_db_tensors_nr;            // They are still needed for purposes of SIF
    uint16_t                        node_match_output_tensors_nr;  // number of output tensors that need to be duplicated

    // Debug Section
    char node_name[MAX_SHAPE_PLANE_NODE_NAME_LEN];
};

struct shape_plane_graph_t
{
    uint32_t version_major;
    uint32_t version_minor;

    uint32_t            sp_node_nr;  // The number of nodes in the shape graph
    shape_plane_node_t* sp_nodes;    // nodes in execution order.

    uint64_t       sp_tensors_nr;  // The number of tensors in the shape graph
    tensor_info_t* sp_tensors;     // The array of tensors in the shape graph

    uint64_t             shape_tensors_list_nr;  // The number of tensors in the shape graph
    shape_tensor_info_t* shape_tensors;          // The array of shape tensors in the shape graph
};
