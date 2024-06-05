#pragma once

#include "habana_graph.h"

bool isDenseAfterPermute(const TensorPtr& input, const gc::Permutation& perm);