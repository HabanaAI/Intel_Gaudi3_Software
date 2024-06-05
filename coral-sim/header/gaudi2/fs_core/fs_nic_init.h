#pragma once
#include "coral_infra.h"
#include "lbw_hub.h"
#include "psoc.h"
#include "specs_base.h"

void create_fs_nic(LBWHub* lbw, Specs* specs, Psoc* m_psoc);
void bind_fs_nic_clk(coral::Clock& clk);
void nic_reset(uint32_t mask = UINT32_MAX);
