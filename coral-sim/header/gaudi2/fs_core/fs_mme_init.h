#pragma once
#include <vector>
#include "coral_infra.h"
#include "derater.h"
#include "mme_half.h"
#include "psoc.h"

void set_end_timestamp(unsigned int mme_id, double timestamp);
void create_fs_mme(LBWHub* lbw, Specs* specs, std::vector<Gaudi2_MME_Half*>, Psoc* psoc, Derater* derater);
void bind_fs_mme_clk(coral::Clock& clk);
void fs_mme_send_so(unsigned mme_index, uint64_t num_sos);
void fs_mme_set_mme_debug_memaccess(unsigned mme_idx, void* mem_access);
void fs_mme_set_mme_dump_dir(unsigned mme_idx, const std::string& dump_path, const std::string& dump_unit);
void fs_mme_reset();
