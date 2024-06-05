#pragma once

// relative to <3rd-parties>/
#include "llvm/small_vector.h"

// std includes
#include <cstdint>

namespace eager_mode
{
// Type representing number of nodes per graph
using NodesNrType = uint32_t;
// Type representing number of tensors per graph
using TensorsNrType = uint32_t;

// Default maximum number of nodes that will be served by eager, exceeding this limit will cause fall back to graph mode
inline constexpr NodesNrType defaultMaxNodesPerGraph = 16;
// Default tensors per node
inline constexpr TensorsNrType defaultInputsPerNode  = 6;
inline constexpr TensorsNrType defaultOutputsPerNode = 3;
inline constexpr TensorsNrType defaultTensorsPerNode = defaultInputsPerNode + defaultOutputsPerNode;
// Default tensors per graph
inline constexpr TensorsNrType defaultInputsPerGraph  = defaultMaxNodesPerGraph * defaultTensorsPerNode;
inline constexpr TensorsNrType defaultOutputsPerGraph = defaultMaxNodesPerGraph * defaultTensorsPerNode;
inline constexpr TensorsNrType defaultTensorsPerGraph = defaultMaxNodesPerGraph * defaultTensorsPerNode;

// Template of vector its size match default maximum number of nodes in graph
template<class Type>
using VecNodes = llvm_vecsmall::SmallVector<Type, defaultMaxNodesPerGraph>;
// Template of vector its size match default number of tensors in graph
template<class Type>
using VecTensors = llvm_vecsmall::SmallVector<Type, defaultTensorsPerGraph>;

}  // namespace eager_mode
