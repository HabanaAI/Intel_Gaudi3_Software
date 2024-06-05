#pragma once

#include "habana_graph.h"

namespace MemoryReuseHandler
{
void handleMemoryReuse(HabanaGraph& graph);

bool isStridedOverlap(const TensorPtr& t1,
                      const TensorPtr  t2);  // in header for using in unit-tests

bool isStridedOverlap(const std::vector<uint64_t>& sizes1,
                      const std::vector<uint64_t>& sizes2,
                      const std::vector<uint64_t>& strides1,
                      const std::vector<uint64_t>& strides2,
                      uint64_t                     offset1,
                      uint64_t                     offset2);

bool isDenseOverlap(const TensorPtr& t1, const TensorPtr& t2);

uint64_t getRealTensorOffset(const TensorPtr& t);

bool isExactOverlap(const TensorPtr& t1, const TensorPtr& t2);

bool sameMemorySection(const TensorPtr& t1, const TensorPtr& t2);

// return true if n1 and n2 share the same storage for output, and n1 writes to a lower offset
bool hasLowerWritingOffset(const NodePtr& n1, const NodePtr& n2);

// return true if n1 and n2 share the same storage for input, and n1 reads from a lower offset
bool hasLowerReadingOffset(const NodePtr& n1, const NodePtr& n2);
}  // namespace MemoryReuseHandler
