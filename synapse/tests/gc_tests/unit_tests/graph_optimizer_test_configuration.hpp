#pragma once

#include <hl_gcfg/hlgcfg_item.hpp>

hl_gcfg::GcfgItemBool GCFG_RUN_GC_MLIR_TESTS("RUN_GC_MLIR_TESTS",
                                      "Run graph optimizer tests related to synapse_mlir flow",
                                      false,
                                      hl_gcfg::MakePublic);
