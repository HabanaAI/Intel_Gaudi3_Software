#pragma once
#include <limits>
#include <map>
#include <string_view>
#include <string>
#include <unordered_map>
#include <vector>

#include "recipe.h"
#include "tpc_kernel_lib_interface.h"
#include "statistics.hpp"
#include "types.h"

#include "define_synapse_common.hpp"
#include "smf_callbacks.hpp"

//-----------------------------------------------------------------------------
// Dynamic Shape related types
//-----------------------------------------------------------------------------

static constexpr uint32_t INVALID_SHAPE_FUNC_ID = std::numeric_limits<uint32_t>::max();
// #define SFR_STATS // Enable this to collect detailed statistics about sif and smf run time

static constexpr uint32_t GC_SIF_VERSION = 0;

// using the higher 4bits:
enum ShapeFuncOrigin : uint8_t
{
    LIB_ID_RESERVED_FOR_GC_SIF = 0,
    LIB_ID_RESERVED_FOR_GC_SMF = 1 << 4,
    LIB_ID_FIRST_GLUE_CODE_SIF = 2 << 4,
    LIB_ID_TPC_FUSER_SIF       = 3 << 4,
    LIB_ID_COMPLEX_GUID_SIF    = 4 << 4
};

struct ShapeManipulationParams
{
    uint64_t        inputTensorsNr;
    tensor_info_t** inputTensors;
    uint64_t        outputTensorsNr;
    tensor_info_t** outputTensors;
    roi_info_t*     activationRoi;
    void*           metadata;
    uint32_t        inPatchValuesNr;
    uint32_t        nodeIdx;
};

struct ShapeManipulationOutputs
{
    uint64_t* nodeData;
    uint32_t* outputPatchValues;
    uint32_t  outPatchValuesNr;
    uint32_t  outputShouldBypass;
};

// Shape Manipulation Function
using non_ptr_smf_t = void(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);

using smf_t = non_ptr_smf_t*;

using sif_t = tpc_lib_api::pfnGetShapeInference;

enum ShapeFuncID : uint64_t
{
    // SMFs
    SMF_GAUDI_FIRST = 0,
    SMF_DYNAMIC_EXE = SMF_GAUDI_FIRST,
    SMF_DMA_SIZE,
    SMF_MME,
    SMF_TPC_SIZE,
    SMF_TPC_STRIDE,
    SMF_TPC_INDEX_SPACE,
    SMF_TPC_SLICE_STRIDE,
    SMF_TPC_SLICE_OFFSET,
    SMF_TPC_VIEW_STRIDE,
    SMF_TPC_VIEW_OFFSET,
    SMF_MANY_STRIDES,
    SMF_LAST_STRIDE,
    SMF_DENSE_ADDRESS_UNUSED,  // unified with dynamic offset SMF
    SMF_DMA_BASEADDR,
    SMF_DMA_VIEW_STRIDE,
    SMF_DMA_VIEW_OFFSET,
    SMF_DMA_SLICE_STRIDE,
    SMF_DMA_SLICE_OFFSET,
    SMF_MME_PADDING,
    SMF_GAUDI_LAST = 0xFFF,

    SMF_GAUDI2_FIRST    = SMF_GAUDI_LAST + 1,
    SMF_GAUDI2_MME_SIZE = SMF_GAUDI2_FIRST,
    SMF_GAUDI2_MME_NULL_DESC,
    SMF_GAUDI2_MME_SYNC,
    SMF_TPC_INDEX_SPACE_GAUDI2,
    SMF_GAUDI2_LAST = 0x1FFF,

    SMF_GAUDI3_FIRST = SMF_GAUDI2_LAST + 1,
    SMF_TPC_INDEX_SPACE_GAUDI3,
    SMF_GAUDI3_MME_SIZE,
    SMF_GAUDI3_LAST  = 0x2FFF,

    // add more devices here

    SMF_COMMON_FIRST       = 0x100000,
    SMF_PATCH_ON_ZERO_SIZE = SMF_COMMON_FIRST,
    SMF_PATCH_ON_ZERO_SIZE_FIRST_INPUT,
    SMF_DYNAMIC_OFFSET,

    SMF_MAX_ID = 0x7FFFFF,

    // SIFs
    SIF_DMA_MEMCPY = 0x800000,
    SIF_DMA_MEMSET,
    SIF_DMA_UNUSED001,  // was DMA_TRANSPOSE
    SIF_CONCATENATE,
    SIF_FLATTEN,
    SIF_SPLIT,
    SIF_EXPAND_DIMS,
    SIF_MERGE_SHAPES,
    SIF_SLICE,
    SIF_SLICE_AXIS,
    SIF_SLICE_BACKWARD,
    SIF_SLICE_INSERT,
    SIF_RESHAPE,
    SIF_STATIC_RESHAPE,
    SIF_BROADCAST,
    SIF_IDENTITY,
    SIF_REDUCTION,
    SIF_STRIDED_VIEW,
    SIF_STRIDED_INSERT,
    SIF_TENSOR_VIEW,
    SIF_TRANSPOSE,
    SIF_CONVOLUTION,
    SIF_CONV_DEDW,
    SIF_CONV_DEDX,
    SIF_GEMM,
    SIF_GEMM_DEDW,
    SIF_GEMM_DEDX,
    SIF_GEMM_FC,
    SIF_BATCH_GEMM,
    SIF_BATCH_GEMM_DEDW,
    SIF_BATCH_GEMM_DEDX,
    SIF_DMA_PHYS_CONCAT_SPLIT,
    SIF_CONTAINER_PHYS_CONCAT,
    SIF_SQUEEZE,
    SIF_FROBENIUS_NORM,
    SIF_MOMENTS,
    SIF_WAIT,
    SIF_DEBUG,
    SIF_ROTATE,
    SIF_TF_BATCH_NORM,
    SIF_NO_SUPPORT,  // Used to mark the node does not support dynamic shapes.
    SIF_DYNAMIC_SPLIT,
    SIF_PHYSICAL_SPLIT,
    SIF_EINSUM,
    SIF_DYNAMIC_RESHAPE,
    SIF_SPLIT_FUSED,
    SIF_EINSUM_EXPAND,
    SIF_REINTERPRET_CAST,
    SIF_INFER_MAX_SHAPE,
    SIF_TILE_SHAPE,
    SIF_H2D_BATCHSIZE,
    SIF_H2D_DYN_STRIDE_DMA_EXPAND,
    SIF_H2D_DYN_STRIDE_DMA_REINTERPRET,
    SIF_H2D_DYN_SLICE_DMA,
    SIF_SHAPE_TO_H2D_STRIDED,
    SIF_SHAPE_TO_H2D_SLICE,
    SIF_H2D_TRANSPOSE_SLICE,
    SHAPE_FUNC_MAX_ID = 0xFFFFFFFFFFFFFF  // we have 56 bits for this field
};

struct StaticSifEntry
{
    ShapeFuncID      id;
    sif_t            pFunc;
    std::string_view name;
};
struct StaticSmfEntry
{
    ShapeFuncID      id;
    smf_t            pFunc;
    std::string_view name;
};

// By default init for all device types (passing synDeviceTypeInvalid to signify that)
void initShapeFuncRegistry(synDeviceType deviceType = synDeviceTypeInvalid);

// A repository to hold Shape Inference and Shape Manipulation function pointers (SIF & SMF)
class ShapeFuncRegistry
{
public:
    static ShapeFuncRegistry& instance();

    void init(synDeviceType deviceType);
    void destroy();

    void registerSIF(ShapeFuncID        id,
                     sif_t              pFunc,
                     const std::string& name,
                     uint64_t           version,
                     ShapeFuncOrigin    originator = LIB_ID_RESERVED_FOR_GC_SIF);
    void registerSIF(sm_function_id_t id, sif_t pFunc, const std::string& name, uint64_t version);
    void registerSIF(sm_function_id_t id, sif_t pFunc, uint64_t version, const tpc_lib_api::GuidInfo& guid);

    smf_t getSMF(ShapeFuncID id);
    sif_t getSIF(ShapeFuncID id, ShapeFuncOrigin originator = LIB_ID_RESERVED_FOR_GC_SIF);

    smf_t getSMF(sm_function_id_t id);
    sif_t getSIF(sm_function_id_t id);
    std::pair<sif_t, tpc_lib_api::GuidInfo*>  getSIFandGuidInfo(sm_function_id_t id);

    const char* getSmfName(sm_function_id_t id);
    const char* getSifName(sm_function_id_t id);

    uint64_t getSifVersion(sm_function_id_t id);

#ifdef SFR_STATS
    void initStats();

    void sifStatCollect(sif_t pFunc, uint64_t sum);
    void smfStatCollect(smf_t pFunc, uint64_t sum);
#endif

private:
    struct SifInfo
    {
        sif_t                 func;
        uint64_t              version;
        std::string           name;
        tpc_lib_api::GuidInfo guid;
    };

    struct SmfInfo
    {
        smf_t       func;
        std::string name;
    };

    ShapeFuncRegistry();

    void registerSMF(ShapeFuncID id, smf_t pFunc, const std::string& name);
    void registerSMFTable(const StaticSmfEntry* smfTable, unsigned tableSize);
    void registerSIFTable(const StaticSifEntry* sifTable, unsigned tableSize);

    std::unordered_map<decltype(sm_function_id_t::sm_func_index), SmfInfo> m_smfBank;  // SMF ID->SmfInfo
    std::unordered_map<decltype(sm_function_id_t::sm_func_index), SifInfo> m_sifBank;  // SIF ID->SifInfo

#ifdef SFR_STATS
    using IdxAndName = std::pair<decltype(sm_function_id_t::sm_func_index), std::string>;
    template<class func, typename info>
    static void buildStatIdx(std::unordered_map<uint32_t, info>& bank,
                             std::map<func, IdxAndName>&         map,
                             std::vector<StatEnumMsg<int>>&      stat,
                             const std::string&                  msg);
    void        buildSifStatIdx();
    void        buildSmfStatIdx();
    void        dumpSifMap();
    void        dumpSmfMap();

    std::map<sif_t, IdxAndName>   m_sifMap;
    std::vector<StatEnumMsg<int>> m_sifStatIdx;
    StatisticsVec*                m_pSifStats;

    std::map<smf_t, IdxAndName>   m_smfMap;
    std::vector<StatEnumMsg<int>> m_smfStatIdx;
    StatisticsVec*                m_pSmfStats;
#endif

    bool m_initDone = false;

public:
    const decltype(m_sifBank) getAllSifTestingOnly() { return m_sifBank; };
    const decltype(m_smfBank) getAllSmfTestingOnly() { return m_smfBank; };
    void registerSMFTestingOnly(ShapeFuncID id, smf_t pFunc, std::string name) { registerSMF(id, pFunc, name); }
};

class SmfCallbacks
{
public:
    static void                   set(smf_callbacks_t* smfCallbacks) { m_smfCallbacks = smfCallbacks; }
    static const smf_callbacks_t* get() { return m_smfCallbacks; }

private:
    static smf_callbacks_t* m_smfCallbacks;
};
