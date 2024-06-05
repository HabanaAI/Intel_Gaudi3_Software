#pragma once

//
// This file contains generic definitions and utilities regarding reduction operations
//

#include "defs.h"

namespace gc::reduction
{
// Returns whether the datatype can stably be used to accumulate values for reduction operation
bool datatypeValidForAccumulation(const synDataType datatype);
}  // namespace gc::reduction