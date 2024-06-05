#ifndef _TRANSPOSEPERMUTATION_H_
#define _TRANSPOSEPERMUTATION_H_

#include <vector>
#include <memory>
#include "llvm/small_vector.h"

#include "synapse_types.h"

// We utilize SYN_MAX_TENSOR_DIM in here instead of HABANA_DIM_MAX since SYN_MAX_TENSOR_DIM is sufficient to
// avoid memory allocations for most use cases, and TransposePermutationDim is a 4 byte enum exposed to the
// Synapse API, so we do not wish to increase transpose nodes having TransposePermutationArray as member too
// much.
using TransposePermutationArray = llvm_vecsmall::SmallVector<TransposePermutationDim, SYN_MAX_TENSOR_DIM>;

// should be sufficient to avoid memory allocations for all current use cases
using TransposePermutationArrayVec = llvm_vecsmall::SmallVector<TransposePermutationArray, 3>;

struct SifTransposeMetadata;
using pSifTransposeMetadata = std::shared_ptr<SifTransposeMetadata>;

#endif  // _TRANSPOSEPERMUTATION_H_
