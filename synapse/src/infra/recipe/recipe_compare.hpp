#pragma once

#include "recipe.h"
#include <unordered_set>

namespace RecipeCompare
{
bool operator==(const blob_t& lhs, const blob_t& rhs);
bool operator!=(const blob_t& lhs, const blob_t& rhs);

bool operator==(const program_t& lhs, const program_t& rhs);
bool operator!=(const program_t& lhs, const program_t& rhs);

bool operator==(const job_t& lhs, const job_t& rhs);
bool operator!=(const job_t& lhs, const job_t& rhs);

bool operator==(const program_data_blob_t& lhs, const program_data_blob_t& rhs);
bool operator!=(const program_data_blob_t& lhs, const program_data_blob_t& rhs);

bool operator==(const const_section_t& lhs, const const_section_t& rhs);
bool operator!=(const const_section_t& lhs, const const_section_t& rhs);

bool operator==(const patch_point_t& lhs, const patch_point_t& rhs);
bool operator!=(const patch_point_t& lhs, const patch_point_t& rhs);

bool operator==(const section_group_t& lhs, const section_group_t& rhs);
bool operator!=(const section_group_t& lhs, const section_group_t& rhs);

bool operator==(const section_blobs_t& lhs, const section_blobs_t& rhs);
bool operator!=(const section_blobs_t& lhs, const section_blobs_t& rhs);

bool operator==(const ecb_t& lhs, const ecb_t& rhs);
bool operator!=(const ecb_t& lhs, const ecb_t& rhs);

bool operator==(const arc_job_t& lhs, const arc_job_t& rhs);
bool operator!=(const arc_job_t& lhs, const arc_job_t& rhs);

bool compare(const node_program_t& lhs, const node_program_t& rhs, int programs_nr);

bool compare(const persist_tensor_info_t& lhs, const persist_tensor_info_t& rhs, bool compareNames);
bool operator==(const persist_tensor_info_t& lhs, const persist_tensor_info_t& rhs);
bool operator!=(const persist_tensor_info_t& lhs, const persist_tensor_info_t& rhs);

bool compare(const gc_conf_t& lhs, const gc_conf_t& rhs, const std::unordered_set<uint32_t>& unsupported_configs);
bool operator==(const gc_conf_t& lhs, const gc_conf_t& rhs);
bool operator!=(const gc_conf_t& lhs, const gc_conf_t& rhs);

bool compare(const recipe_t& lhs, const recipe_t& rhs, bool compareNames);
bool operator==(const recipe_t& lhs, const recipe_t& rhs);
bool operator!=(const recipe_t& lhs, const recipe_t& rhs);
}  // namespace RecipeCompare