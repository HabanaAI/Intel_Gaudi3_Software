#pragma once

#include <synapse_common_types.h>
#include <utils/small_uint_set.h>

class TPCNode;

using DimSet = SmallUintSet<HABANA_DIM_MAX>;

// Returns a set of INDEX SPACE dimensicns that cannot be sliced
DimSet getUnsliceableIndexSpaceDims (const TPCNode& node);
