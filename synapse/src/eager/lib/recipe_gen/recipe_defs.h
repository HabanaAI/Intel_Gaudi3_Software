#pragma once

// eager includes (relative to src/eager/lib/)
#include "utils/general_utils.h"

// synapse api (relative to include/)
#include "internal/recipe.h"

// std includes
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// Basic types and info of pointers and registers
///////////////////////////////////////////////////////////////////////////////////////////////////

// Macro to extract type of pointer variable at recipe_t
#define RECIPE_PINTER_TYPE(VAR) POINTER_TYPE(recipe_t, VAR)

using Byte                      = std::byte;
using EngineIdType              = decltype(job_t::engine_id);           // Id type of computation engine
using JobsNrType                = decltype(recipe_t::execute_jobs_nr);  // Type of jobs number
using ProgramsNrType            = decltype(recipe_t::programs_nr);      // Type of programs number
using ProgramLengthType         = decltype(program_t::program_length);  // Type of programs length
using BlobsNrType               = decltype(recipe_t::blobs_nr);         // Type of blobs number
using BlobSizeType              = decltype(blob_t::size);               // Position in blob
using BitPosInBlobType          = uint64_t;      // Position of a bit in blob. Hold a value in bits
using StructSizeType            = BlobSizeType;  // Position in software struct (descriptor or cache base register)
using AsicRegType               = BlobSizeType;  // Id of architectural register (as it defined in SOC Online)
using DataNrType                = BlobSizeType;  // Number of ASIC registers to be mapped
using QmanCommandSizeType       = uint32_t;      // Size type of Qman command
using CommandGranularityRawType = uint64_t;      // Equivelent to raw value of Qman command multiples
using TensorAddressType         = uint64_t;      // Low and high parts of tensor address
using AsicRegValType            = uint32_t;      // Type of data that match ASIC register length
using CacheBaseRegIdType        = uint16_t;      // Smallest type that should store ids of all cache base registers
using PatchPointNrType          = decltype(recipe_t::patch_points_nr);  // Size type of patch point array
using SectionIdxType            = decltype(patch_point_t::memory_patch_point.section_idx);
using NodesNrType               = decltype(recipe_t::node_nr);
using TensorsNrType             = decltype(recipe_t::persist_tensors_nr);
using ArcJobsNrType             = decltype(recipe_t::arc_jobs_nr);
using EcbCommandSizeType        = decltype(ecb_t::cmds_size);
using ProgramDataBlobsNrType    = decltype(recipe_t::program_data_blobs_nr);
using WorkspaceSizesType        = RECIPE_PINTER_TYPE(workspace_sizes);
using RecipeIdType              = decltype(recipe_t::debug_profiler_info.recipe_id);
using ProfilerNumEnginesType    = RECIPE_PINTER_TYPE(debug_profiler_info.nodes->num_working_engines);

static_assert(std::is_same<AsicRegValType, QmanCommandSizeType>(), "Unsupported HW registers and commands layout");

constexpr unsigned sizeOfAsicRegVal = sizeof(AsicRegValType);  // Length of ASIC registers in bytes
constexpr unsigned sizeOfAddressVal = sizeof(TensorAddressType);  // Length of address in bytes

// Number of ASIC registers that fits into one entry of "WREG_BULK" commands
constexpr DataNrType asicRegsPerEntry = sizeOfAddressVal / sizeOfAsicRegVal;
static_assert((sizeOfAddressVal % sizeOfAsicRegVal) == 0, "Invalid register layout");

// Total data buffers in recipe: patching, executable and dynamic
constexpr unsigned dataBufNr = 3;

// This value will be assigned to recipe_t::workspace_nr. It represents the following:
// MEMORY_ID_RESERVED_FOR_WORKSPACE
// MEMORY_ID_RESERVED_FOR_PROGRAM_DATA
// MEMORY_ID_RESERVED_FOR_PROGRAM
constexpr unsigned workspaceSizesNr = 3;

// This value will is used for recipe_t::recipe_conf_nr for a basiline empty recipe to pass validation
// With the following conf values being set:
// - gc_conf_t::DEVICE_TYPE
// - gc_conf_t::TPC_ENGINE_MASK
constexpr uint32_t confParamsNr = 2;

///////////////////////////////////////////////////////////////////////////////////////////////////
// ASIC regs operations
///////////////////////////////////////////////////////////////////////////////////////////////////

// ECB packet NOP does padding in DWORDS units (see eng_arc_cmd_nop_t::padding documentation)
constexpr EcbCommandSizeType nopPaddingUnits = sizeof(uint32_t);

enum class EcbType : uint8_t
{
    STATIC,
    DYNAMIC
};

}  // namespace eager_mode