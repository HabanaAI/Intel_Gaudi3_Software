#include "recipe_compare.hpp"

#include "defs.h"
#include "infra/fasthash.h"
#include "log_manager.h"
#include "recipe.h"

namespace RecipeCompare
{
template<typename T>
struct EqualTo
{
    constexpr bool operator()(const T& lhs, const T& rhs) const { return operator==(lhs, rhs); }
};

template<typename T, typename F = EqualTo<T>>
bool isArrEqualElementwise(const T* const arr1, const T* const arr2, uint64_t num_elements, F comparator = F())
{
    for (int i = 0; i < num_elements; i++)
    {
        if (!comparator(arr1[i], arr2[i]))
        {
            LOG_ERR(SYN_COMPARE, "comparison failed for collection item number {}", i);
            return false;
        }
    }
    return true;
}

template<typename T>
bool isArrEqualBySize(const T* const arr1, const T* const arr2, int size)
{
    if (size > 0)
    {
        if (memcmp(arr1, arr2, size) == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    return true;
}

template<typename T>
inline bool isArrEqualByNumElements(const T* const arr1, const T* const arr2, uint64_t num_elements)
{
    return isArrEqualBySize(arr1, arr2, num_elements * sizeof(T));
}

bool compare(const recipe_t& lhs, const recipe_t& rhs, bool compareNames)
{
    auto tensor_comp = [&compareNames](const persist_tensor_info_t& a, const persist_tensor_info_t& b) {
        return compare(a, b, compareNames);
    };

    // This comparator should run after checking number of programs in both recipes is equal
    auto node_program_comp = [&lhs](const node_program_t& a, const node_program_t& b) {
        return compare(a, b, lhs.programs_nr);
    };

    // set of configurations not supported for comparison
    const std::unordered_set<uint32_t> unsupported_configs = {gc_conf_t::TIME_STAMP};
    auto                               conf_comp = [&unsupported_configs](const gc_conf_t& a, const gc_conf_t& b) {
        return compare(a, b, unsupported_configs);
    };

    if (lhs.version_major != rhs.version_major)
    {
        LOG_ERR(SYN_COMPARE, "version major differs between recipe compilations");
        return false;
    }
    else if (lhs.version_minor != rhs.version_minor)
    {
        LOG_ERR(SYN_COMPARE, "version minor differs between recipe compilations");
        return false;
    }
    else if (lhs.execution_blobs_buffer_size != rhs.execution_blobs_buffer_size)
    {
        LOG_ERR(SYN_COMPARE, "execution blobs buffer size differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualBySize(lhs.execution_blobs_buffer, rhs.execution_blobs_buffer, lhs.execution_blobs_buffer_size))
    {
        LOG_ERR(SYN_COMPARE, "execution blobs buffer differs between recipe compilations");
        return false;
    }
    else if (lhs.patching_blobs_buffer_size != rhs.patching_blobs_buffer_size)
    {
        LOG_ERR(SYN_COMPARE, "patching blobs buffer size differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualBySize(lhs.patching_blobs_buffer, rhs.patching_blobs_buffer, lhs.patching_blobs_buffer_size))
    {
        LOG_ERR(SYN_COMPARE, "patching blobs buffer differs between recipe compilations");
        return false;
    }
    else if (lhs.dynamic_blobs_buffer_size != rhs.dynamic_blobs_buffer_size)
    {
        LOG_ERR(SYN_COMPARE, "dynamic blobs buffer size differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualBySize(lhs.dynamic_blobs_buffer, rhs.dynamic_blobs_buffer, lhs.dynamic_blobs_buffer_size))
    {
        LOG_ERR(SYN_COMPARE, "dynamic blobs buffer differs between recipe compilations");
        return false;
    }
    else if (lhs.blobs_nr != rhs.blobs_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of blobs differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.blobs, rhs.blobs, lhs.blobs_nr))
    {
        LOG_ERR(SYN_COMPARE, "blobs differ between recipe compilations");
        return false;
    }
    else if (lhs.programs_nr != rhs.programs_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of programs differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.programs, rhs.programs, lhs.programs_nr))
    {
        LOG_ERR(SYN_COMPARE, "programs differ between recipe compilations");
        return false;
    }
    else if (lhs.activate_jobs_nr != rhs.activate_jobs_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of activate jobs differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.activate_jobs, rhs.activate_jobs, lhs.activate_jobs_nr))
    {
        LOG_ERR(SYN_COMPARE, "activate jobs differ between recipe compilations");
        return false;
    }
    else if (lhs.execute_jobs_nr != rhs.execute_jobs_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of execute jobs differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.execute_jobs, rhs.execute_jobs, lhs.execute_jobs_nr))
    {
        LOG_ERR(SYN_COMPARE, "execute jobs differ between recipe compilations");
        return false;
    }
    else if (lhs.arc_jobs_nr != rhs.arc_jobs_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of arc jobs differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.arc_jobs, rhs.arc_jobs, lhs.arc_jobs_nr))
    {
        LOG_ERR(SYN_COMPARE, "arc jobs differ between recipe compilations");
        return false;
    }
    else if (lhs.persist_tensors_nr != rhs.persist_tensors_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of persistent tensors differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.tensors, rhs.tensors, lhs.persist_tensors_nr, tensor_comp))
    {
        LOG_ERR(SYN_COMPARE, "persistent tensors differ between recipe compilations");
        return false;
    }
    else if (lhs.permute_tensors_views_nr != rhs.permute_tensors_views_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of permute tensor views differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.permute_tensors_views,
                                    rhs.permute_tensors_views,
                                    lhs.permute_tensors_views_nr,
                                    tensor_comp))
    {
        LOG_ERR(SYN_COMPARE, "permute tensor views differ between recipe compilations");
        return false;
    }
    else if (lhs.const_sections_nr != rhs.const_sections_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of const sections differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.const_sections, rhs.const_sections, lhs.const_sections_nr))
    {
        LOG_ERR(SYN_COMPARE, "const sections differ between recipe compilations");
        return false;
    }
    else if (lhs.program_data_blobs_size != rhs.program_data_blobs_size)
    {
        LOG_ERR(SYN_COMPARE, "number of program data blobs size differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualBySize(lhs.program_data_blobs_buffer,
                               rhs.program_data_blobs_buffer,
                               lhs.program_data_blobs_size))
    {
        LOG_ERR(SYN_COMPARE, "program data blobs buffer differs between recipe compilations");
        return false;
    }
    else if (lhs.program_data_blobs_nr != rhs.program_data_blobs_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of program data blobs differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.program_data_blobs, rhs.program_data_blobs, lhs.program_data_blobs_nr))
    {
        LOG_ERR(SYN_COMPARE, "program data blobs differ between recipe compilations");
        return false;
    }
    else if (lhs.patch_points_nr != rhs.patch_points_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of patch points differs between recipe compilations");
        return false;
    }
    else if (lhs.activate_patch_points_nr != rhs.activate_patch_points_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of activate patch points differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.patch_points, rhs.patch_points, lhs.patch_points_nr))
    {
        LOG_ERR(SYN_COMPARE, "patch points differ between recipe compilations");
        return false;
    }
    else if (lhs.sections_nr != rhs.sections_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of sections differs between recipe compilations");
        return false;
    }
    else if (lhs.section_groups_nr != rhs.section_groups_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of section groups differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.section_groups_patch_points,
                                    rhs.section_groups_patch_points,
                                    lhs.section_groups_nr))
    {
        LOG_ERR(SYN_COMPARE, "section groups patch points differ between recipe compilations");
        return false;
    }
    else if (lhs.section_ids_nr != rhs.section_ids_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of section ids differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.section_blobs_indices, rhs.section_blobs_indices, lhs.section_ids_nr))
    {
        LOG_ERR(SYN_COMPARE, "section blobs indices differ between recipe compilations");
        return false;
    }
    else if (lhs.node_nr != rhs.node_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of nodes differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.node_exe_list, rhs.node_exe_list, lhs.node_nr, node_program_comp))
    {
        LOG_ERR(SYN_COMPARE, "node executaions differ between recipe compilations");
        return false;
    }
    else if (lhs.workspace_nr != rhs.workspace_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of workspaces differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualByNumElements(lhs.workspace_sizes, rhs.workspace_sizes, lhs.workspace_nr))
    {
        LOG_ERR(SYN_COMPARE, "workspaces sizes differ between recipe compilations");
        return false;
    }
    else if (lhs.recipe_conf_nr != rhs.recipe_conf_nr)
    {
        LOG_ERR(SYN_COMPARE, "number of recipe configurations differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualElementwise(lhs.recipe_conf_params, rhs.recipe_conf_params, lhs.recipe_conf_nr, conf_comp))
    {
        LOG_ERR(SYN_COMPARE, "recipe configurations differ between recipe compilations");
        return false;
    }

    LOG_INFO(SYN_COMPARE, "recipe comparison finished successfully");
    return true;
}

bool operator==(const blob_t& lhs, const blob_t& rhs)
{
    // this struct member holds the hash of the data in the blob
    if (fasthash(lhs.data, lhs.size) != fasthash(rhs.data, rhs.size))
    {
        LOG_ERR(SYN_COMPARE, "blob data differs between recipe compilations");
        return false;
    }

    return true;
}

bool operator==(const program_t& lhs, const program_t& rhs)
{
    /*
     *  IMPORTANT NOTE!
     *  As part of the effort to keep recipe compilation determinstic, the following assert intends to prevent making
     *  changes in the program_t struct without respectively updating it's comparator. If you've added new fields to the
     *  program_t struct, please consider whether or not the contents of those fields are expected to remain equal
     *  between different compilations of the same graph. If they should - please kindly update the comparator so that
     *  it takes into account your changes. If not - do not change the comparator. After that - please update structs
     *  size in the assert, so build can pass successfully. Thanks for paying attention, an updated comparator is
     *  important in order to keep track of deterministic recipe compilation.
     */
    size_t constexpr STRUCT_SIZE = 16;
    static_assert(sizeof(program_t) == STRUCT_SIZE, "'Equal To' Operator has to be updated after struct is updated");

    if (lhs.program_length != rhs.program_length)
    {
        LOG_ERR(SYN_COMPARE, "program length differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualBySize(lhs.blob_indices, rhs.blob_indices, lhs.program_length))
    {
        LOG_ERR(SYN_COMPARE, "program blob indices differ between recipe compilations");
        return false;
    }

    return true;
}

bool operator==(const job_t& lhs, const job_t& rhs)
{
    /*
     *  IMPORTANT NOTE!
     *  As part of the effort to keep recipe compilation determinstic, the following assert intends to prevent making
     *  changes in the job_t struct without respectively updating it's comparator. If you've added new fields to the
     *  job_t struct, please consider whether or not the contents of those fields are expected to remain equal between
     *  different compilations of the same graph. If they should - please kindly update the comparator so that it takes
     *  into account your changes. If not - do not change the comparator. After that - please update structs size in
     *  the assert, so build can pass successfully. Thanks for paying attention, an updated comparator is important in
     *  order to keep track of deterministic recipe compilation.
     */
    size_t constexpr STRUCT_SIZE = 8;
    static_assert(sizeof(job_t) == STRUCT_SIZE, "'Equal To' Operator has to be updated after struct is updated");

    if (lhs.engine_id != rhs.engine_id)
    {
        LOG_ERR(SYN_COMPARE, "job engine id differs between recipe compilations");
        return false;
    }
    else if (lhs.program_idx != rhs.program_idx)
    {
        LOG_ERR(SYN_COMPARE, "job program index differs between recipe compilations");
        return false;
    }

    return true;
}

bool operator==(const ecb_t& lhs, const ecb_t& rhs)
{
    /*
     *  IMPORTANT NOTE!
     *  As part of the effort to keep recipe compilation determinstic, the following assert intends to prevent making
     *  changes in the ecb_t struct without respectively updating it's comparator. If you've added new fields to the
     *  ecb_t struct, please consider whether or not the contents of those fields are expected to remain equal between
     *  different compilations of the same graph. If they should - please kindly update the comparator so that it takes
     *  into account your changes. If not - do not change the comparator. After that - please update structs size in
     *  the assert, so build can pass successfully. Thanks for paying attention, an updated comparator is important in
     *  order to keep track of deterministic recipe compilation.
     */
    size_t constexpr STRUCT_SIZE = 16;
    static_assert(sizeof(ecb_t) == STRUCT_SIZE, "'Equal To' Operator has to be updated after struct is updated");

    if (lhs.cmds_size != rhs.cmds_size)
    {
        LOG_ERR(SYN_COMPARE, "ecb commands size differ between recipe compilations");
        return false;
    }
    else if (lhs.cmds_eng_offset != rhs.cmds_eng_offset)
    {
        LOG_ERR(SYN_COMPARE, "ecb commands offset differ between recipe compilations");
        return false;
    }
    else if (!isArrEqualBySize(lhs.cmds, rhs.cmds, lhs.cmds_size))
    {
        LOG_ERR(SYN_COMPARE, "ecb commands differ between recipe compilations");
        return false;
    }

    return true;
}

bool operator==(const arc_job_t& lhs, const arc_job_t& rhs)
{
    /*
     *  IMPORTANT NOTE!
     *  As part of the effort to keep recipe compilation determinstic, the following assert intends to prevent making
     *  changes in the arc_job_t struct without respectively updating it's comparator. If you've added new fields to the
     *  arc_job_t struct, please consider whether or not the contents of those fields are expected to remain equal
     *  between different compilations of the same graph. If they should - please kindly update the comparator so that
     *  it takes into account your changes. If not - do not change the comparator. After that - please update structs
     *  size in the assert, so build can pass successfully. Thanks for paying attention, an updated comparator is
     *  important in order to keep track of deterministic recipe compilation.
     */
    size_t constexpr STRUCT_SIZE = 40;
    static_assert(sizeof(arc_job_t) == STRUCT_SIZE, "'Equal To' Operator has to be updated after struct is updated");

    if (lhs.static_ecb != rhs.static_ecb)
    {
        LOG_ERR(SYN_COMPARE, "archive job static ecb differs between recipe compilations");
        return false;
    }
    else if (lhs.dynamic_ecb != rhs.dynamic_ecb)
    {
        LOG_ERR(SYN_COMPARE, "archive job dynamic ecb differs between recipe compilations");
        return false;
    }
    else if (lhs.engines_filter != rhs.engines_filter)
    {
        LOG_ERR(SYN_COMPARE, "archive job engines filter differs between recipe compilations");
        return false;
    }
    else if (lhs.logical_engine_id != rhs.logical_engine_id)
    {
        LOG_ERR(SYN_COMPARE, "archive job logical engine id differs between recipe compilations");
        return false;
    }

    return true;
}

bool compare(const persist_tensor_info_t& lhs, const persist_tensor_info_t& rhs, bool compareNames)
{
    if (lhs.offset_in_section != rhs.offset_in_section)
    {
        LOG_ERR(SYN_COMPARE, "tensor offset in section differs between recipe compilations");
        return false;
    }
    else if (lhs.size != rhs.size)
    {
        LOG_ERR(SYN_COMPARE, "tensor size differs between recipe compilations");
        return false;
    }
    else if (lhs.zp != rhs.zp)
    {
        LOG_ERR(SYN_COMPARE, "tensor zp differs between recipe compilations");
        return false;
    }
    else if (lhs.scale != rhs.scale)
    {
        LOG_ERR(SYN_COMPARE, "tensor scale differs between recipe compilations");
        return false;
    }
    else if (lhs.section_idx != rhs.section_idx)
    {
        LOG_ERR(SYN_COMPARE, "tensor section index differs between recipe compilations");
        return false;
    }
    else if (lhs.elementType != rhs.elementType)
    {
        LOG_ERR(SYN_COMPARE, "tensor element type differs between recipe compilations");
        return false;
    }
    else if (lhs.dimensions != rhs.dimensions)
    {
        LOG_ERR(SYN_COMPARE, "tensor dimensions differ between recipe compilations");
        return false;
    }
    else if (!std::equal(std::begin(lhs.dimensionsSize), std::end(lhs.dimensionsSize), std::begin(rhs.dimensionsSize)))
    {
        LOG_ERR(SYN_COMPARE, "tensor dimensions differ between recipe compilations");
        return false;
    }
    else if (lhs.batchSize != rhs.batchSize)
    {
        LOG_ERR(SYN_COMPARE, "tensor batch size differs between recipe compilations");
        return false;
    }
    else if (lhs.tensorType != rhs.tensorType)
    {
        LOG_ERR(SYN_COMPARE, "tensor type differs between recipe compilations");
        return false;
    }
    else if (lhs.extTensorExeOrder != rhs.extTensorExeOrder)
    {
        LOG_ERR(SYN_COMPARE, "external tensor execution order differs between recipe compilations");
        return false;
    }
    else if (lhs.isExternal != rhs.isExternal)
    {
        LOG_ERR(SYN_COMPARE, "whether tensor is external or not differs between recipe compilations");
        return false;
    }
    else if (!std::equal(std::begin(lhs.permutation), std::end(lhs.permutation), std::begin(rhs.permutation)))
    {
        LOG_ERR(SYN_COMPARE, "tensor permutation differs between recipe compilations");
        return false;
    }
    else if (lhs.section_type != rhs.section_type)
    {
        LOG_ERR(SYN_COMPARE, "tensor section type differs between recipe compilations");
        return false;
    }
    else if (lhs.isInput != rhs.isInput)
    {
        LOG_ERR(SYN_COMPARE, "whether the tensor is an input differs between recipe compilations");
        return false;
    }
    else if (lhs.layout == nullptr || rhs.layout == nullptr)
    {
        if (!(lhs.layout == nullptr && rhs.layout == nullptr))
        {
            LOG_ERR(SYN_COMPARE, "tensor layout differs between recipe compilations");
            return false;
        }
    }
    else if (strcmp(lhs.layout, rhs.layout) != 0)
    {
        LOG_ERR(SYN_COMPARE, "tensor layout differs between recipe compilations");
        return false;
    }
    else if (compareNames)
    {
        if (strcmp(lhs.name, rhs.name) != 0)
        {
            LOG_ERR(SYN_COMPARE, "tensor name differs between recipe compilations");
            return false;
        }
    }

    return true;
}

bool operator==(const program_data_blob_t& lhs, const program_data_blob_t& rhs)
{
    /*
     *  IMPORTANT NOTE!
     *  As part of the effort to keep recipe compilation determinstic, the following assert intends to prevent making
     *  changes in the program_data_blob_t struct without respectively updating it's comparator. If you've added new
     *  fields to the program_data_blob_t struct, please consider whether or not the contents of those fields are
     *  expected to remain equal between different compilations of the same graph. If they should - please kindly update
     *  the comparator so that it takes into account your changes. If not - do not change the comparator. After that -
     *  please update structs size in the assert, so build can pass successfully. Thanks for paying attention, an
     *  updated comparator is important in order to keep track of deterministic recipe compilation.
     */
    size_t constexpr STRUCT_SIZE = 32;
    static_assert(sizeof(program_data_blob_t) == STRUCT_SIZE,
                  "'Equal To' Operator has to be updated after struct is updated");

    if (lhs.offset_in_section != rhs.offset_in_section)
    {
        LOG_ERR(SYN_COMPARE, "program data blob offset in section differs between recipe compilations");
        return false;
    }
    else if (lhs.section_idx != rhs.section_idx)
    {
        LOG_ERR(SYN_COMPARE, "program data blob section index differs between recipe compilations");
        return false;
    }
    else if (lhs.size != rhs.size)
    {
        LOG_ERR(SYN_COMPARE, "program data blob size differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualBySize(lhs.data, rhs.data, lhs.size))
    {
        LOG_ERR(SYN_COMPARE, "program data blob differs between recipe compilations");
        return false;
    }

    return true;
}

bool operator==(const patch_point_t& lhs, const patch_point_t& rhs)
{
    /*
     *  IMPORTANT NOTE!
     *  As part of the effort to keep recipe compilation determinstic, the following assert intends to prevent making
     *  changes in the patch_point_t struct without respectively updating it's comparator. If you've added new fields to
     *  the patch_point_t struct, please consider whether or not the contents of those fields are expected to remain
     *  equal between different compilations of the same graph. If they should - please kindly update the comparator so
     *  that it takes into account your changes. If not - do not change the comparator. After that - please update
     *  structs size in the assert, so build can pass successfully. Thanks for paying attention, an updated comparator
     *  is important in order to keep track of deterministic recipe compilation.
     */
    size_t constexpr STRUCT_SIZE = 32;
    static_assert(sizeof(patch_point_t) == STRUCT_SIZE,
                  "'Equal To' Operator has to be updated after struct is updated");

    if (lhs.blob_idx != rhs.blob_idx)
    {
        LOG_ERR(SYN_COMPARE, "patch point blob index differs between recipe compilations");
        return false;
    }
    else if (lhs.dw_offset_in_blob != rhs.dw_offset_in_blob)
    {
        LOG_ERR(SYN_COMPARE, "patch point dw offset in blob differs between recipe compilations");
        return false;
    }
    else if (lhs.node_exe_index != rhs.node_exe_index)
    {
        LOG_ERR(SYN_COMPARE, "patch point node execution index differs between recipe compilations");
        return false;
    }
    else if (lhs.type != rhs.type)
    {
        LOG_ERR(SYN_COMPARE, "patch point type differs between recipe compilations");
        return false;
    }
    else if (lhs.type == patch_point_t::SOB_PATCH_POINT)
    {
        if (lhs.sob_patch_point.tensor_db_index != rhs.sob_patch_point.tensor_db_index)
        {
            LOG_ERR(SYN_COMPARE, "sob patch point tensor db index differs between recipe compilations");
            return false;
        }
    }
    else
    {
        if (lhs.memory_patch_point.effective_address != rhs.memory_patch_point.effective_address)
        {
            LOG_ERR(SYN_COMPARE, "memory patch point effective address differs between recipe compilations");
            return false;
        }
        else if (lhs.memory_patch_point.section_idx != rhs.memory_patch_point.section_idx)
        {
            LOG_ERR(SYN_COMPARE, "memory patch point section index differs between recipe compilations");
            return false;
        }
    }
    return true;
}

bool operator==(const section_group_t& lhs, const section_group_t& rhs)
{
    if (lhs.section_group != rhs.section_group)
    {
        LOG_ERR(SYN_COMPARE, "section group differs between recipe compilations");
        return false;
    }
    else if (lhs.patch_points_nr != rhs.patch_points_nr)
    {
        LOG_ERR(SYN_COMPARE, "section group number of patch points differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualByNumElements(lhs.patch_points_index_list, rhs.patch_points_index_list, lhs.patch_points_nr))
    {
        LOG_ERR(SYN_COMPARE, "section group patch points index list differs between recipe compilations");
        return false;
    }

    return true;
}

bool operator==(const section_blobs_t& lhs, const section_blobs_t& rhs)
{
    if (lhs.blobs_nr != rhs.blobs_nr)
    {
        LOG_ERR(SYN_COMPARE, "section blob number of blobs differs between recipe compilations");
        return false;
    }
    else if (lhs.section_idx != rhs.section_idx)
    {
        LOG_ERR(SYN_COMPARE, "section blob section index differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualByNumElements(lhs.blob_indices, rhs.blob_indices, lhs.blobs_nr))
    {
        LOG_ERR(SYN_COMPARE, "section blob indices differ between recipe compilations");
        return false;
    }

    return true;
}

bool operator==(const const_section_t& lhs, const const_section_t& rhs)
{
    if (lhs.size != rhs.size)
    {
        LOG_ERR(SYN_COMPARE, "const section size in bytes differs between recipe compilations");
        return false;
    }
    else if (lhs.section_idx != rhs.section_idx)
    {
        LOG_ERR(SYN_COMPARE, "const section index differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualByNumElements(lhs.data, rhs.data, lhs.size))
    {
        LOG_ERR(SYN_COMPARE, "const section data differs between recipe compilations");
        return false;
    }

    return true;
}

bool compare(const node_program_t& lhs, const node_program_t& rhs, int program_nr)
{
    if (lhs.patch_points_nr != rhs.patch_points_nr)
    {
        LOG_ERR(SYN_COMPARE, "node program patch points number differs between recipe compilations");
        return false;
    }
    else if (!isArrEqualByNumElements(lhs.program_blobs_nr, rhs.program_blobs_nr, program_nr))
    {
        LOG_ERR(SYN_COMPARE, "node program number of program blobs differs between recipe compilations");
        return false;
    }

    return true;
}

bool compare(const gc_conf_t& lhs, const gc_conf_t& rhs, const std::unordered_set<uint32_t>& unsupported_configs)
{
    if (lhs.conf_id != rhs.conf_id)
    {
        LOG_ERR(SYN_COMPARE, "comparing configurations with different ids is not supported");
        return false;
    }

    bool skipValidation = unsupported_configs.find(lhs.conf_id) != unsupported_configs.end();
    if (skipValidation)
    {
        LOG_DEBUG(SYN_COMPARE, "skipping configuration of id {}, not supported for comparison", lhs.conf_id);
        return true;
    }
    else if (lhs.conf_value != rhs.conf_value)
    {
        LOG_ERR(SYN_COMPARE,
                "graph compiler supported for comparison configuration value differs between recipe compilations");
        return false;
    }
    else
    {
        return true;
    }
}

bool operator==(const persist_tensor_info_t& lhs, const persist_tensor_info_t& rhs)
{
    return compare(lhs, rhs, false);
}

bool operator==(const gc_conf_t& lhs, const gc_conf_t& rhs)
{
    return compare(lhs, rhs, {});
}

bool operator==(const recipe_t& lhs, const recipe_t& rhs)
{
    return compare(lhs, rhs, false);
}

bool operator!=(const blob_t& lhs, const blob_t& rhs)
{
    return !(lhs == rhs);
}

bool operator!=(const program_t& lhs, const program_t& rhs)
{
    return !(lhs == rhs);
}

bool operator!=(const job_t& lhs, const job_t& rhs)
{
    return !(lhs == rhs);
}

bool operator!=(const program_data_blob_t& lhs, const program_data_blob_t& rhs)
{
    return !(lhs == rhs);
}

bool operator!=(const patch_point_t& lhs, const patch_point_t& rhs)
{
    return !(lhs == rhs);
}

inline bool operator!=(const section_group_t& lhs, const section_group_t& rhs)
{
    return !(lhs == rhs);
}

inline bool operator!=(const section_blobs_t& lhs, const section_blobs_t& rhs)
{
    return !(lhs == rhs);
}

inline bool operator!=(const ecb_t& lhs, const ecb_t& rhs)
{
    return !(lhs == rhs);
}

inline bool operator!=(const arc_job_t& lhs, const arc_job_t& rhs)
{
    return !(lhs == rhs);
}

bool operator!=(const persist_tensor_info_t& lhs, const persist_tensor_info_t& rhs)
{
    return !(lhs == rhs);
}

bool operator!=(const gc_conf_t& lhs, const gc_conf_t& rhs)
{
    return !(lhs == rhs);
}

bool operator!=(const const_section_t& lhs, const const_section_t& rhs)
{
    return !(lhs == rhs);
}

bool operator!=(const recipe_t& lhs, const recipe_t& rhs)
{
    return !(lhs == rhs);
}
}  // namespace RecipeCompare
