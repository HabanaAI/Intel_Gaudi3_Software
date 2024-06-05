#pragma once
#include <vector>
#include "coral_infra.h"
#include "lbw_hub.h"
#include "specs.h"
#define DLL_PUBLIC __attribute__((visibility("default")))
DLL_PUBLIC void create_fs_decoder(LBWHub* lbw, Specs* specs);
DLL_PUBLIC void bind_fs_decoder_clk(coral::Clock& clk);
DLL_PUBLIC void fs_decoder_reset();