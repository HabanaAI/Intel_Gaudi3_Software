#ifndef _NODE_ANNOTATION_H_
#define _NODE_ANNOTATION_H_

#include <memory>

#include "include/mme_common/mme_common_enum.h"
#include "infra/settable.h"
#include "layout.h"
#include <list>
#include <optional>
#include <vector>
#include "synapse_common_types.h"
#include "tensor_shape.h"
#include "sync/sync_types.h"
#include "cache_types.h"
#include "dma_cost_model.h"
#include "tile.h"

class Node;
namespace MmeCommon
{
struct PerfAttr;
}
//This is a place to write the auxiliary data related to a node
//The optimizer uses this to inform the code generator about its decisions
typedef enum
{
    PACKING_X = 0,
    PACKING_Y
} PackingStrategy;

/*
 * Be careful when using default values when using this object from within the SyncOrMonitor union,
 * you probably want to explicitly init them first.
 */
struct SyncObject
{
    unsigned int id = 0;
    int16_t value = 0;
    int operation = 0;
    int barrier = 0;
};

/*
 * Be careful when using default values when using this object from within the SyncOrMonitor union,
 * you probably want to explicitly init them first.
 */
struct MonObject
{
    unsigned int id     = 0;
    unsigned int syncId = 0; // if mask is set, sync id is group id

    // The armValue is the sync value to wait for. When used only as setupMonitor, this value is ignored.
    int16_t   armValue  = 0;
    MonitorOp operation = MONITOR_SO_OP_EQ;
    // If set wait for fence
    Settable<unsigned int> fenceTargetVal;
    Settable<uint8_t>      mask;
    bool                   predicateTheSetup = false;
    WaitID                 fenceId           = WaitID::ID_0;  // Relevant when fenceTargetVal is set

    // When used as setupMonitor, the setupValue is the value set/inc due to this monitor.
    int16_t setupValue = 1;
    // The sync id to signal (For fences, set FENCE_MONITOR_ID)
    unsigned int signalSyncId = 0;
    // Should we increment the value or set it
    bool shouldInc = false;
};

struct SyncOrMonitor
{
    SyncOrMonitor() {}
    union
    {
        SyncObject sync;
        MonObject monitor;
    };
    enum Type
    {
        NONE,
        SYNC_OBJ,
        MONITOR_OBJ
    };
    Type type = NONE;
};


struct PipelineSyncScheme
{
    std::list<MonObject> monitors;
    std::list<SyncObject> cpSyncs;
    std::shared_ptr<SyncObject> sync;
    uint32_t                    syncTotalValue;
    uint32_t                    numSignalsForDbg;
};

struct PatchableMonitor
{
    MonObject monObject;
    uint32_t  tensorId;
};

struct SyncInteraction
{
    std::pair<unsigned, unsigned>   preExecSyncObjIdxAndValue;
    std::list<SyncOrMonitor>        preSyncsAndMon;
    std::vector<PipelineSyncScheme> pipelineSyncs;
    std::list<SyncOrMonitor>        postSyncsAndMons;
    std::vector<PatchableMonitor>   patchableMonitors;
};

struct ArcSyncInteraction
{
    ArcSyncInteraction() {}
    ArcSyncInteraction(DependencyMap a, Settable<unsigned> b, Settable<unsigned> c)
    : dependencies(a), emittedSigVal(b), breakpoint(c)
    {
    }
    ArcSyncInteraction(DependencyMap a, unsigned b, Settable<unsigned> c)
    : dependencies(a), emittedSigVal(Settable<unsigned>(b)), breakpoint(c)
    {
    }

    DependencyMap      dependencies;  // pipeline, control-edges, etc.
    Settable<unsigned> emittedSigVal;
    Settable<unsigned> breakpoint;
    unsigned           sobResetTotalNumEngs = 0;  // trigger SOB reset after activation with this amount of engines
    unsigned           sobResetId           = 0;
};

struct MemorySpaceInfo
{
    struct PrefetchInfo
    {
        bool     prefetch = false;
        uint32_t epoch = 0;
    };

    PrefetchInfo prefetchInfo;

    // Nodes blocking current memory space
    std::list<std::shared_ptr<Node>> barriers;
};

struct BaseRegsCacheEntry
{
    unsigned indexInCache;
    uint64_t sectionID;
};

struct BundleInfo
{
    BundleInfo(unsigned _bundleIndex, BundleType _bundleType, unsigned _operationIndex = 0)
    : bundleIndex(_bundleIndex)
    , bundleType(_bundleType)
    , operationIndex(_operationIndex)
    {}
    BundleInfo()                = default;
    unsigned     bundleIndex    = 0;  // Sliced nodes in gaudi are part of a bundle
    BundleType   bundleType     = BundleType::UNDEFINED;
    unsigned     operationIndex = 0;  // execution schedule is pre-set by slicing brain per bundle
    BundleEngine bundleEngine   = BundleEngine::ENGINE_UNDEFINED;

    std::optional<unsigned> perforationGroup;  // Represents bundle nodes which are perforated together
    std::optional<unsigned> threadIndex = std::nullopt;
};

struct FlashAttentionInfo
{
    FlashAttentionInfo(unsigned _origNodeId, std::optional<unsigned> _chainInfo = {})
    : origNodeId(_origNodeId), chainInfo(_chainInfo)
    {
    }
    FlashAttentionInfo() = default;

    unsigned           origNodeId = 0;
    std::optional<unsigned> chainInfo {};
};

// Temporary hack to try splitting to cores and using on-dcore caching in Gaudi3.
// These hints help split the node work into dcores homogeneously throughout a bundle, so that on-dcore cache
// can be used effectively.
struct LitePerforationHints
{
    unsigned indexSpaceDim;
    TSize    granularity;
    bool     isPerforated = false;  // was this node actually perforated
};

struct MmeMetaData
{
    MmeMetaData()
    {
        // Note that in eager-mode, GCFG_DEDW_UNROLL is replaced with GCFG_ENABLE_EAGER_BATCH_CONCURRENCY,
        // during node_displacement.
        mmeStrategy.batchConcurrencyEn =
            GCFG_DEDW_UNROLL.value() ? MmeCommon::BoolWithUndef::TurnedOn : MmeCommon::BoolWithUndef::TurnedOff;
    }
    MmeCommon::EMmeOperand sbReuseOnOperand = MmeCommon::EMmeOperand::e_mme_op_o;  // operand being reused.
    mutable MmeCommon::MmeStrategy mmeStrategy;
    bool takeStrategyFromAnnotation = false;
    std::string mmeStrategyDebugString;

    bool signalEarlyOnReusedOperand = false;
    MmeCommon::EMmeRateLimiter rateLimitReusedOperand  = MmeCommon::RL_NONE;

    std::shared_ptr<MmeCommon::PerfAttr> mmePerfAttr        = nullptr;
    unsigned packing[MME_MAX_CONV_DIMS] = {1, 1, 1, 1}; /* packing for each dimension.
                                                  1 for no packing. */
    bool groupPacking = false;  // was group packing applied to this node

};

struct UtilizationParams
{
    unsigned totalNumWorkingEngines = 0;
    float    engineUtilization      = 0;
    float    totalUtilization       = 0;
};
using UtilizationParamsVec = SmallVector<UtilizationParams, MAX_NUM_DCORES>;

struct TpcMetaData
{
    //Utilization Params per logicalROI per Dcore.
    SmallVector<UtilizationParamsVec, 1> utilizationPerLogicalRoi;
};

struct NodeAnnotation
{
    NodeAnnotation(size_t numInputs) : inputPermutations(numInputs) {}

    unsigned             pipelineDepth = 1;
    DimVector            tpcSplitDims;
    MemorySpaceInfo      memorySpaceInfo;
    // The permutations performed on the node's input tensors (in the node's perspective) by the data layout adjustment passes
    PermutationVector            inputPermutations;
    Settable<BundleInfo>         bundleInfo;
    std::optional<FlashAttentionInfo> flashAttentionInfo;
    // TODO SW-120522: move to OperationSlice once we have the support for all node types.
    NodePtr   origBigNode;     // If the node is sliced, this field indicates the original node pre-slicing.
    std::optional<gc::access_pattern::NodeTile>  sliceROI;       // If the node is sliced, this field indicates the sliced node ISR.
    unsigned  rangeIndex = 0;  // Graphs are split into collection of nodes called ranges
    unsigned  sliceIndex = 0;  // Ranges are split into horizontal slices. Todo: add link to presentation
    SizeArray baseOffset = {}; // Used when slicing graphs, for TPC nodes, offset in indices from the large index space (5d)
    std::vector<SyncInteraction>                 syncScheme;     // Sync scheme for each engine
    std::vector<std::vector<Settable<unsigned>>> prevSyncId;     // Prev sync id for every pipe level in every queue
    std::vector<std::vector<unsigned>>           prevSyncVal;    // Prev sync value for every pipe level in every queue
    std::vector<ArcSyncInteraction>              arcSyncScheme;  // ARC sync scheme for every pipe level
    std::vector<BaseRegsCacheEntry> baseRegsCacheUpdate;
    Settable<unsigned> sfgSyncObjValue;
    unsigned waitCycles = 0;
    bool insertedNode = false;  // for nodes inserted by the GC (didn't come from user)
    bool splitToLogicalROIs    = true;   // Can specify to ROI splitter whether to leave this node as a single ROI
    bool isExtracted           = false;  // Whether this node is extracted from multi node
    bool canSkipSplitToLogical = true;   // whether we can decide to not split.
    bool originatedFromCguid = false;  // hint for internal features to minimize compilation time

    std::optional<LitePerforationHints> perforation;
    std::optional<unsigned>             perforationDim;

    // mme related fields
    MmeMetaData mmeMetaData;
    // tpc related fields
    TpcMetaData tpcMetaData;
    // expecting 4 DCORE rois from brain
    DcoreRoisVec m_dcoreROIs;

    DmaCost dmaCost;

    // info relevant to all DCOREs
    std::vector<CacheMetaData> inputsCacheMetaData;
    std::vector<CacheMetaData> outputsCacheMetaData;

    char     originalComplexGuid[tpc_lib_api::MAX_NODE_NAME] = {0};  // null terminated c-string
    uint64_t originalComplexGuidId                           = ~uint64_t(0);

    NodeList fusedNodes;

    void updateSplitToLogicalROIs(bool value)
    {
        if (!canSkipSplitToLogical && !value)
        {
            return;
        }
        splitToLogicalROIs = value;
    }

    bool isPerforated() const { return (perforation.has_value() && perforation->isPerforated) || !m_dcoreROIs.empty(); }
};

#endif
