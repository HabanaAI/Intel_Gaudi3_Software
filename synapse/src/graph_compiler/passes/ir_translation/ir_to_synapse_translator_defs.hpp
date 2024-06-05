#pragma once
#include "gc_protocol.hpp"
#include "types.h"

namespace ir_translation_defs
{
// AbstractIRToSynapseTranslator defs

static constexpr size_t SMALL_MAP_NEW_TENSOR_COUNT = 16;
using NewTensorMap = SmallMap<std::map<uint64_t, TensorPtr>, SMALL_MAP_NEW_TENSOR_COUNT>;

static constexpr size_t SMALL_MAP_SECTION_TYPE_COUNT = 8;
using SectionTypeMap =
    SmallMap<std::map<uint64_t, gc_protocol::ProtocolTensorSectionType_t>, SMALL_MAP_SECTION_TYPE_COUNT>;

static constexpr size_t SMALL_VECTOR_BLOCKING_NODES_COUNT = 1;
using IdsVector                                           = SmallVector<uint64_t, SMALL_VECTOR_BLOCKING_NODES_COUNT>;

static constexpr size_t SMALL_MAP_NEW_NODES_COUNT = 16;
using IrNodeToGCNodeIdxMap                        = SmallMap<std::map<uint64_t, size_t>, SMALL_MAP_NEW_NODES_COUNT>;

// no need for small map at the moment
using IrNodeToGCBlockingNodesIdMap = std::unordered_map<uint64_t, ir_translation_defs::IdsVector>;

// IRToSynapseTranslatorBase defs

static constexpr size_t SMALL_MAP_ORIG_TENSOR_COUNT = 8;
using OrigTensorMap = SmallMap<std::map<uint64_t, const TensorPtr&>, SMALL_MAP_ORIG_TENSOR_COUNT>;

static constexpr size_t SMALL_SET_ORIG_SECTION_COUNT = 8;
using OrigSectionSet                                 = SmallSet<std::set<uint64_t>, SMALL_SET_ORIG_SECTION_COUNT>;

// no need for small map at the moment
using SectionIdMap = std::unordered_map<uint64_t, uint64_t>;
}  // namespace ir_translation_defs