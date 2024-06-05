#pragma once

#include "descriptor_shadow.h"
#include "descriptor_wrapper.h"
#include "habana_device_types.h"
#include "infra/defs.h"
#include "node_annotation.h"
#include "params_file_manager.h"
#include "segment.h"
#include "sync/sync_conventions.h"
#include "types.h"
#include "utils.h"

#include <memory>
#include <tuple>
#include <vector>

unsigned getMaxTpcEnginesCount();
unsigned getMaxDmaEnginesCount();
unsigned getMaxMmeEnginesCount();
unsigned getMaxRotatorEnginesCount();
unsigned getMcidBaseRegsFirstIndex();

struct DescSection
{
    DescSection(const void* addr, unsigned size, unsigned offset)
    : addr(addr), size(size), offset(offset), isCommitter(false)
    {
    }

    DescSection(const void* addr, unsigned size, unsigned offset, bool isCommitter)
    : addr(addr), size(size), offset(offset), isCommitter(isCommitter)
    {
    }

    template<typename Desc>
    explicit DescSection(const Desc& desc) : DescSection(&desc, sizeof(desc), 0)
    {}

    const void* addr;           // address of consecutive descriptor section
    unsigned    size;           // size of consecutive descriptor section
    unsigned    offset;         // offset of the section within the overall descriptor structur
    bool        isCommitter;    // make the last command of this section a blob committer

    uint32_t  offsetInRegs() const { return offset / sizeof(uint32_t); }
    uint32_t  sizeInRegs() const   { return size / sizeof(uint32_t); }

    const uint32_t* regArray() const { return static_cast<const uint32_t*>(addr); }
};

namespace gc_recipe
{
    class CommandBuffer;
    class generic_packets_container;
}

class QueueCommand;

using QueueCommandPtr = std::unique_ptr<QueueCommand>;

class BasicFieldsContainerInfo;
class HabanaGraph;
class QueueCommandFactory;
struct NodeROI;

//---------------------------------------------------------
//--------------------- CommandQueue ----------------------
//------------------ Base for all queues ------------------
//---------------------------------------------------------
class CommandQueue

{
public:
    virtual ~CommandQueue() = default;

    void              Clear();
    virtual unsigned  GetBinarySize(bool isSetup, bool isLoad = false) const;
    unsigned          Size(bool isSetup, bool isLoad = false) const;
    bool              Empty() const;
    virtual void      PushBack(QueueCommandPtr cmd, bool isSetup = true, bool isLoad = false);
    void              SerializePB(gc_recipe::generic_packets_container* pktCon, bool isSetup, bool isLoad = false) const;
    void              SetParams(ParamsManager* params);
    virtual void      Print() const;

    virtual unsigned            GetQueueID() const;      // returns final hardware queue ID
    virtual unsigned            GetEngineID() const;     // returns engine ID for multi-engines like TPC in goya and TPC, MME, DMA in gaudi
    virtual unsigned            GetEngineIndex() const;  // returns engine index within a dispatcher engine array
    virtual unsigned            GetEngineStream() const; // returns queue stream, for goya it's always 0, for gaudi it's 0..3
    virtual unsigned            GetLogicalQueue() const;
    virtual HabanaDeviceType    GetDeviceType() const;
    virtual unsigned            GetCpDmaCmdSize() const;
    virtual QueueCommand&       operator[] (int index);
    virtual const std::string   getName() const;
    virtual bool                requiresAlignment() const { return false; }
    virtual bool                requiresSort() const { return false; }

    virtual void setMonIdToSetupMonitors(std::map<unsigned, MonObject>& setupMonitors);
    virtual void setMonIdAsSetup(unsigned monitorId);

    // Add node sync scheme to queue
    // numOfParallelEngines - for case as TPC multiple engine
    virtual void addPreSyncScheme(std::shared_ptr<Node> node, bool isSetup = true);
    virtual void addPostExeSyncs(std::shared_ptr<Node>  node, bool isSetup = true);

    virtual const QueueCommandFactory& getCommandFactory() const;

    // These have default empty implementation in this class allowing unified interface
    virtual void AddNode(const pNode& n, HabanaGraph* g, bool isSetup);
    virtual void finalizeSetupPhase();
    virtual void finalizeQueue(bool isSetup);
    virtual void finalizeInitQueue(bool isSetup);

    // This can be added to any queue and thus added in base class
    void AddSuspend(unsigned cyclesToWait);

    std::vector<std::pair<QueueCommandPtr, bool>>& getInitialQueueCommands();

    bool isQueueActive();
    void setQueueAsActive();

    enum class MonitorCommandParts
    {
        SetupArmFence,
        SetupArm,
        Fence
    };

    void addMonitor(const MonObject&    monitor,
                    bool                isSetup,
                    MonitorCommandParts parts = MonitorCommandParts::SetupArmFence,
                    Settable<unsigned>  engID = {});

    void getMonitorCmds(const MonObject&              monitor,
                        std::vector<QueueCommandPtr>& cmds,
                        MonitorCommandParts           parts = MonitorCommandParts::SetupArmFence,
                        Settable<unsigned>            engID = {});

    void addSignal(const SyncObject& sync, bool isSetup);
    void getSignalCmds(const SyncObject& sync, std::vector<QueueCommandPtr>& cmds);
    void setArmMonBeforeDesc(bool value);
    void loadBaseRegsCacheUpdate(const pNode& node);
    unsigned getFirstFenceMonitorId();

    virtual const std::vector<QueueCommandPtr>& getCommands(bool isSetup, bool isLoad = false) const;

protected:
    // only child of goya or gaudi type can be instantiated
    CommandQueue(const QueueCommandFactory&  cmdFactory,
                 unsigned                    queId,
                 HabanaDeviceType            devType);

    QueueCommandPtr makeWriteBulkCacheEntries(unsigned cacheIndex, const std::vector<uint64_t>& sectionIDs, const pNode& node) const;

    void pushAdditionalDynamicCmds(const pNode& node, unsigned pipeLevel, bool isLastPipelineLevel);
    virtual void pushAdditionalDynamicCmds4sfg(const pNode& node, unsigned pipeLevel, bool isLastPipelineLevel);
    virtual void pushAdditionalDynamicCmds4sobReset(const pNode& node, unsigned pipeLevel);
    virtual void pushAdditionalDynamicCmds4mcidRollover(const pNode& node, unsigned pipeLevel);

    const QueueCommandFactory&    m_commandFactory;
    unsigned                      m_queueId;      // final hardware queue ID
    HabanaDeviceType              m_deviceType;
    unsigned                      m_engineId;     // engine ID for multi-engines like TPC in goya and TPC, MME, DMA in gaudi
    unsigned                      m_engineIndex;  // engine index within a dispatcher engine array
    unsigned                      m_stream;       // queue stream, for goya1 it's always 0, for gaudi/goya2 it's 0..3
    unsigned                      m_maxStreams;   // maximum number of streams; for goya1 it's always 1, for gaudi/goya2 it's 4
    unsigned                      m_packetSizeActivate;
    unsigned                      m_packetSizeLoad;
    unsigned                      m_packetSizeExe;
    std::vector<QueueCommandPtr>  m_queueActivate;
    std::vector<QueueCommandPtr>  m_queueLoad;
    std::vector<QueueCommandPtr>  m_queueExe;
    ParamsManager*                m_params;
    std::set<unsigned>            m_usedSetupMonitors;
    std::map<unsigned, MonObject> m_monIdToSetupMonitors;
    bool                          m_activeQueue;       // Indicate if queue is active, when changed - commands are added
    bool                          m_sendSyncEvents;    // indicating whether to send sync events for profiling sync manager
    bool                          m_armMonBeforeDesc;  // indicate if to separate the monitor from its fence and arm it before the descriptor
    bool                          m_qmanMutexRequired; // indicate if we need to put mutex between the QMAN and its engine (goya2 HW bug WA)

    // The boolean indicates if the command should be executed in the Activate or Execute phase.
    // True means that the command should be pushed to the Activate part.
    std::vector<std::pair<QueueCommandPtr, bool>>  m_initialQueueCommands;
};

//---------------------------------------------------------
//--------------------- DmaCommandQueue -------------------
//----------------- Base for all DMA queues ---------------
//---------------------------------------------------------

class DmaCommandQueue : public CommandQueue
{
public:
    virtual ~DmaCommandQueue();

    virtual void AddNode(const pNode& n, HabanaGraph* g, bool isSetup) override;

protected:
    // only child of goya or gaudi type can be instantiated
    DmaCommandQueue(const QueueCommandFactory&  cmdFactory,
                    unsigned                    queId,
                    HabanaDeviceType            devType);

    virtual void pushQueueCommand(const pNode& n, const NodeROI& roi, bool isSetup, bool wrComplete) = 0;
};

//---------------------------------------------------------
//------------------- DmaHostToDevQueue -------------------
//---------------------------------------------------------

class DmaHostToDevQueue : public DmaCommandQueue
{
public:
    virtual ~DmaHostToDevQueue();

protected:
    // only child of goya or gaudi type can be instantiated
    DmaHostToDevQueue(const QueueCommandFactory&  cmdFactory,
                      unsigned                    queId,
                      const SyncConventions&      syncConventions);
    virtual void pushQueueCommand(const pNode& n, const NodeROI& roi, bool isSetup, bool wrComplete) override;

    const SyncConventions& m_syncConventions;
};

//---------------------------------------------------------
//------------------- DmaDevToHostQueue -------------------
//---------------------------------------------------------

class DmaDevToHostQueue : public DmaCommandQueue
{
public:
    virtual ~DmaDevToHostQueue();
    void startSetupPhase();

protected:
    // only child of goya or gaudi type can be instantiated
    DmaDevToHostQueue(const QueueCommandFactory& cmdFactory, unsigned queId);
    virtual void pushQueueCommand(const pNode& n, const NodeROI& roi, bool isSetup, bool wrComplete) override;
};

//---------------------------------------------------------
//------------------ DmaDramToSramQueue -------------------
//---------------------------------------------------------

class DmaDramToSramQueue : public DmaCommandQueue
{
public:
    virtual ~DmaDramToSramQueue();

protected:
    // only child of goya or gaudi type can be instantiated
    DmaDramToSramQueue(const QueueCommandFactory&  cmdFactory,
                       unsigned                    queId,
                       HabanaDeviceType            type,
                       SyncConventions&            syncConventions);
    virtual void pushQueueCommand(const pNode& n, const NodeROI& roi, bool isSetup, bool wrComplete) override;

    SyncConventions& m_syncConventions;
};

//---------------------------------------------------------
//------------------ DmaSramToDramQueue -------------------
//---------------------------------------------------------

class DmaSramToDramQueue : public DmaCommandQueue
{
public:
    virtual ~DmaSramToDramQueue();

protected:
    // only child of goya or gaudi type can be instantiated
    DmaSramToDramQueue(const QueueCommandFactory& cmdFactory, unsigned queId);
    virtual void pushQueueCommand(const pNode& n, const NodeROI& roi, bool isSetup, bool wrComplete) override;
};

//---------------------------------------------------------
//--------------------- DescCommandQueue-------------------
//------------ Base for all descriptor-based queues -------
//---------------------------------------------------------

template <class DescType>
bool canSignal(const DescType& desc)
{
    return true;
}

template <class DescType>
void getDescriptorsWrappers(const pNode& n, HabanaGraph* g, std::vector< DescriptorWrapper<DescType> >& ret)
{
    HB_ASSERT(false, "Unknown descriptor");
}

template <typename DescType>
class DescCommandQueue : public CommandQueue
{
public:
    virtual ~DescCommandQueue();

    virtual bool addLoadDesc(const NodePtr&            n,
                             DescSection               descSection,
                             BasicFieldsContainerInfo* pBasicFieldsContainerInfo = nullptr,
                             uint32_t                  predicate                 = 0,
                             DescriptorShadow*         relevantDescriptorShadow  = nullptr);

    virtual bool loadDescWithPredicates(pNode                                               n,
                                        DescSection                                         descSection,
                                        std::vector<Settable<DescriptorWrapper<DescType>>>* pPipeDescs);

    virtual std::vector<DescSection> getPredicatedSections(pNode n, const DescType& desc) const;
    virtual std::vector<DescSection> getUnpredicatedSections(pNode n, const DescType& desc) const;

    virtual void AddPartialNode(
        pNode                                               n,
        DescriptorWrapper<DescType>&                        desc,        // current descriptor to push to queue
        std::vector<Settable<DescriptorWrapper<DescType>>>* pPipeDescs,  // all descriptors of current pipeline stage
        unsigned                                            pipeStage,
        bool                                                isSetup,
        const std::vector<uint64_t>&                        baseRegsCache,  // list of cache resident section IDs
        bool                                                isLastPipelineLevel,
        std::vector<QueueCommandPtr>*                       preSyncCmds             = nullptr,
        std::vector<QueueCommandPtr>*                       postSyncCmds            = nullptr,
        bool                                                isFirstInEnginePerLevel = true,
        bool                                                isLastInEnginePerLevel  = true);

    virtual void setDescriptorSignaling(DescType& desc, const std::shared_ptr<SyncObject>& sync) {}

    virtual void finalizeQueue(bool isSetup) override;

    virtual void finalizeInitQueue(bool isSetup) override;

protected:
    virtual QueueCommandPtr  getExeCmd(pNode n, const DescriptorWrapper<DescType>& descWrap, bool enableSignal = true);
    std::vector<uint32_t>&   getInvalidRegsIndices() { return m_nullDescRegs; }
    virtual void             createNullDescRegsList() {}
    virtual void             updateQueueStateAfterPush(pNode n);
    virtual bool             allowNoDescUpdates(pNode n);
    void                     delimitBlobOnCmd(QueueCommand* cmd, unsigned nodeExeIdx);

    void                     pushWriteReg64Commands(const NodePtr& n,
                                                    unsigned       cacheIndex,
                                                    uint32_t       fieldOffsetLow,
                                                    uint32_t       fieldOffsetHigh,
                                                    ptrToInt       descVal);

    virtual void             optimizePatchpoints(const NodePtr&               n,
                                                 const DescType&              desc,
                                                 BasicFieldsContainerInfo*    bfci,
                                                 const std::vector<uint64_t>& baseRegsCache = {});

    virtual void             optimizeAddressPatchpoints(const NodePtr&               n,
                                                        const DescType&              desc,
                                                        BasicFieldsContainerInfo*    bfci,
                                                        const std::vector<uint64_t>& baseRegsCache = {});

    virtual void             optimizeMcidPatchpoints(const NodePtr&               n,
                                                     const DescType&              desc,
                                                     BasicFieldsContainerInfo*    bfci,
                                                     const std::vector<uint64_t>& baseRegsCache = {});

    void                     optimizeDsdPatchpoints(const NodePtr& n, BasicFieldsContainerInfo* bfci);

    // returns: low_position, high_position, low_patchpoint (if relevant)
    std::tuple<uint32_t, uint32_t, AddressFieldInfoSharedPtr> getLowAndHighPositions(AddressFieldInfoSharedPtr patchPoint);

    virtual DescriptorShadow::AllRegistersProperties registersPropertiesForDesc(pNode n, const DescriptorWrapper<DescType>& desc);

    // only child of goya or gaudi type can be instantiated
    DescCommandQueue(const QueueCommandFactory& cmdFactory, unsigned queId, HabanaDeviceType devType);

    virtual unsigned getMaxEngineCount() = 0;

    virtual void forceStaticConfig() {}

    virtual DescriptorShadow& getDescShadow(const NodePtr& n = nullptr) { return m_descriptorShadow; }

    virtual void validateAllAddressPatchpointsDropped(const NodePtr& n) const;

    bool m_allPatchpointsDropped;

    // Predicate Support: vector of loaded predicated descriptors per engine
    std::vector<DescriptorShadow> m_descriptorShadowWithPred;

    // Holds indices of all registers affected by Null desc so later we can invalidate them in shadowDescriptor
    std::vector<uint32_t> m_nullDescRegs;

private:

    QueueCommandPtr createSkipSignal(SyncObject& syncObj, NodeROI* roi);
    QueueCommandPtr createExecuteCommand(const pNode&                       n,
                                         const DescriptorWrapper<DescType>& descWrap,
                                         std::shared_ptr<SyncObject>        syncPtr,
                                         NodeROI*                           roi,
                                         bool                               isLastDescForRoi,
                                         bool                               isLastInPipeline);

    // Only Gaudi1 should be signaling from the queue
    // Gaudi2/3 use different mechanisms
    // This function sgould be overriden for Gaudi1 queues
    virtual bool isSignalingFromQman() { return false; }

    DescriptorShadow m_descriptorShadow;
};

//---------------------------------------------------------
//------------------------ MmeQueue -----------------------
//---------------------------------------------------------

template <typename DescType>
class MmeQueue : public DescCommandQueue<DescType>
{
public:
    virtual ~MmeQueue();
    virtual void AddNode(const pNode& n, HabanaGraph* g, bool isSetup) override;

protected:
    // only child of goya or gaudi type can be instantiated
    MmeQueue(const QueueCommandFactory& cmdFactory, unsigned queId);
    unsigned getMaxEngineCount() override;
};

//---------------------------------------------------------
//------------------------ TpcQueue -----------------------
//---------------------------------------------------------

template <typename DescType>
class TpcQueue : public DescCommandQueue<DescType>
{
public:
    virtual ~TpcQueue();

protected:
    // only child of goya or gaudi type can be instantiated
    TpcQueue(const QueueCommandFactory& cmdFactory, unsigned queId);
    unsigned getMaxEngineCount() override;
};

//---------------------------------------------------------
//------------------------ RotatorQueue -----------------------
//---------------------------------------------------------

template <typename DescType>
class RotatorQueue : public DescCommandQueue<DescType>
{
public:
    virtual ~RotatorQueue();

protected:
    // only child of goya or gaudi type can be instantiated
    RotatorQueue(const QueueCommandFactory& cmdFactory, unsigned queId);
    unsigned getMaxEngineCount() override;
};

//---------------------------------------------------------
//---------------------- DmaDescQueue ---------------------
//---------------------------------------------------------

template <typename DescType>
class DmaDescQueue : public DescCommandQueue<DescType>
{
public:
    virtual ~DmaDescQueue();

protected:
    // only child of goya or gaudi type can be instantiated
    DmaDescQueue(const QueueCommandFactory& cmdFactory, unsigned queId);
    unsigned getMaxEngineCount() override;
};


//---------------------------------------------------------
// Templeate implementation inlined
//---------------------------------------------------------
#include "desc_command_queue.inl"

//--------------------------------------------------------------------------------
// Two general purpose down-caster to get a specific queue out of the base pointer
//--------------------------------------------------------------------------------
template <typename T>
T* downcaster(const CommandQueuePtr& q)
{
    T* ptr = dynamic_cast<T*>(q.get());
    HB_ASSERT_PTR(ptr);
    return ptr;
}

template <typename T>
const T* downcaster(const ConstCommandQueuePtr& q)
{
    const T* ptr = dynamic_cast<const T*>(q.get());
    HB_ASSERT_PTR(ptr);
    return ptr;
}
