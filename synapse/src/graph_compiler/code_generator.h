#pragma once

#include "graph.h"
#include "mcid_generator.h"
#include "node_utility.h"
#include "program_data_blob.h"
#include "section_id_generator.h"
#include "types.h"
#include "settable.h"
#include "habana_global_conf.h"
#include "memory_management/memory_allocator.h"
#include "graph_compiler/sync/sync_object_manager.h"
#include "graph_compiler/sync/monitor_setup_manager.h"
#include "queue_dispatcher.h"
#include "eng_arc_command.h"
#include "mcid_converter.h"

class CodeGenerator;
namespace CodeGeneration
{
// holds NOP kernel info
struct NOPKernel
{
    Settable<deviceAddrOffset> nopKernelOffset;
    uint64_t                   nopKernelSection = 0;
    uint64_t                   nopKernelSize    = 0;
};

using DeviceAddrOffsetPair = std::pair<deviceAddrOffset, deviceAddrOffset>;
using CodeGeneratorPtr     = std::unique_ptr<CodeGenerator>;
}  // namespace CodeGeneration

using namespace CodeGeneration;
// A class for managing the code generation of the graph compilation process.
class CodeGenerator
{
public:
    CodeGenerator(HabanaGraph* graph) : m_kernelsAddrMap(), m_kernelsPrintf(), m_MemSectionIDGenerator(), m_graph(graph)
    {
    }
    CodeGenerator(const CodeGenerator& other, HabanaGraph* graph);

    virtual ~CodeGenerator() { clear(); }

    virtual bool init();
    void generate(const HabanaGraph* graph);

    unsigned         getKernelsBinarySize() const;
    deviceAddrOffset getKernelAddress(kernelID kid) const;
    deviceAddrOffset getKernelAddress(kernelID kid, bool& wasFound) const;
    const DeviceAddrOffsetPair getKernelsLowerAndUpperBounds() const;

    std::map<kernelID, std::shared_ptr<ProgramDataBlob>> getKernelMap() const { return m_kernelsAddrMap; }
    void setKernelMap(std::map<kernelID, std::shared_ptr<ProgramDataBlob>> kernelsAddrMap)
    {
        m_kernelsAddrMap = kernelsAddrMap;
    }
    void configNOPKernel(deviceAddrOffset addrOffset, uint64_t section, unsigned int kernelSize);

    std::vector<deviceAddrOffset> getKernelsPrintfs() const { return m_kernelsPrintf; }
    void addKernelPrintf(deviceAddrOffset printfAddr) { m_kernelsPrintf.push_back(printfAddr); }
    bool usingPrintf() { return m_usingPrintf; }
    void setUsingPrintf() { m_usingPrintf = true; }

    unsigned getMaxDMAChunks() const { return GCFG_MAX_NUM_DMA_CHUNKS.value(); }

    uint64_t getNumberOfMemorySections(SectionIDGenerator::AllocationManagementType allocType) const;
    uint64_t getNextMemorySectionID(SectionIDGenerator::AllocationManagementType allocType);

    LogicalMcid getNextMCID(MCIDGenerator::MCIDType mcidType);

    std::list<NodeROI>& getPhysicalRois(const NodePtr& node) { return m_physicalRois[node]; }

    // temporary dup in habana graph and codeGen
    std::list<NodeROI>*               getNodeROIs(const NodePtr n) const;
    const std::shared_ptr<HalReader>& getHALReader() const;
    static NodePtr                    getNodeSharedPtr(const Node& node);

    CodeGenerator& operator=(const CodeGenerator& other);
    virtual CodeGeneratorPtr clone(HabanaGraph* graph, bool cloneAllocators = false) const = 0;
    void           clear();
    template<typename T>
    void registerProgramDataBlobForDownload(T hostAddr, deviceAddrOffset deviceAddr, uint64_t size);
    template<typename T>
    void registerTPCProgramDataBlobForDownload(T hostAddr, deviceAddrOffset deviceAddr, uint64_t size, kernelID kid);
    ProgramDataBlobSet&       getProgramDataBlobs() { return m_programDataBlobs; }
    const ProgramDataBlobSet& getProgramDataBlobs() const { return m_programDataBlobs; }

    NOPKernel&       getNOPKernel() { return m_NOPKernel; }
    const NOPKernel& getNOPKernel() const { return m_NOPKernel; }

    void     initSram(uint64_t sramSize, uint64_t sramBaseAddr);
    void     initDram(uint64_t dramSize, uint64_t dramBaseAddr);
    uint64_t getDramBaseAddr() const { return m_dramBaseAddr; }
    uint64_t getDramSize() const { return m_dramSize; }
    void     setDramBaseAddr(uint64_t val) { m_dramBaseAddr = val; }
    void     setDramSize(uint64_t val) { m_dramSize = val; }
    uint64_t getSramBaseAddr() const { return m_synapseSramBaseAddr; }
    uint64_t getSramSize() const { return m_sramSize; }
    void     setSramBaseAddr(uint64_t val) { m_synapseSramBaseAddr = val; }
    void     setSramSize(uint64_t val) { m_sramSize = val; }

    virtual MemoryAllocator& getWorkspaceAllocator() = 0;
    virtual std::shared_ptr<MemoryAllocator> getWorkspaceAllocatorPtr() { return nullptr; };
    virtual MemoryAllocator& getAllocatorForProgramData() = 0;
    virtual MemoryAllocator& getSramAllocator() = 0;
    virtual const MemoryAllocator& getWorkspaceAllocator() const = 0;
    virtual const MemoryAllocator& getAllocatorForProgramData() const = 0;
    virtual const MemoryAllocator& getSramAllocator() const = 0;
    virtual void generateRecipes(const HabanaGraph& graph) {};  // temporary public, until GenerateQueues and
                                                                // addAllDescriptors will move to codeGen
    virtual void addAllDescriptors() {};  // temporary public
    virtual recipe_t*            serializeDataPlane(RecipeAllocator* recipeAlloc) const { return nullptr; };
    virtual shape_plane_graph_t* serializeShapePlane(RecipeAllocator* recipeAlloc) const { return nullptr; };
    virtual std::shared_ptr<SyncObjectManager> getSyncObjectManager() const { return m_syncObjectManager; }
    virtual std::shared_ptr<MonitorSetupManager> getMonitorSetupManager()   { return m_monitorSetupManager; }

    virtual void setupQueuesMonitors();
    std::map<HabanaDeviceType, std::vector<std::list<SyncOrMonitor>>>& getInitialSyncInstructionsByQueueId();
    virtual unsigned                           getQueueID(HabanaDeviceType type, unsigned id) = 0;
    const std::map<uint32_t, CommandQueuePtr>& getCommandQueueById() const { return m_commandQueueById; }
    std::map<uint32_t, CommandQueuePtr>&       getCommandQueueByIdForTesting() { return m_commandQueueById; }
    const QueueCommandFactory&                 getCommandFactory() const;
    void                                       addCommandQueue(const CommandQueuePtr& queue);
    void                                       registerDispatcher(QueueDispatcher& dispatcher);
    void                    setDmaDispatchers(const QueueDispatcherMap& map) { m_dmaDispatchers = map; }
    void                    setDmaDispatchers(QueueDispatcherMap&& map) { m_dmaDispatchers = std::move(map); }
    virtual void            fillQueuesWithDmaNode(NodePtr node) = 0;
    virtual void            initDMAQueues() {};
    virtual void            addExecuteDMANode(NodePtr n, uint32_t* inputDmaInd, uint32_t* outputDmaInd);
    virtual void            downstreamSetupNode(NodePtr n);
    virtual CommandQueuePtr getActivateDramSramQueue();
    virtual void            downloadProgramDataBlobs();

    virtual std::map<uint32_t, std::list<SyncOrMonitor>>& getFinalSyncInstructions(bool bIsActivate = false);
    virtual void                                          addInitialSyncs(bool bIsActivate = false);
    virtual void                                          addFinalSyncs(bool bIsActivate = false);
    virtual void                                          markLastCommandInQueuesForCommit(bool bLastCommand = false);
    virtual void                                          finalizeQueues(bool isSetup);
    virtual void                                          finalizeInitQueues(bool isSetup);
    virtual void                                          finalizeFillQueues();
    virtual void                                          fillQueues();
    virtual void                                          fillSetupNodes();
    virtual void                                          printQueues() const;
    virtual void                                          generateCmeCommands(const NodePtr& n) {}
    virtual void                                          initQueues();  // temporary public
    virtual bool hasSfgDeviceInitData() const { return m_deviceSfgInitValue.size() > 0; }
    std::unordered_map<HabanaDeviceType, unsigned>& getDeviceSfgInitValue() { return m_deviceSfgInitValue; };
    const NodeUtility&                              getNodeUtility() const { return m_nodeUtility; }
    NodeUtility&                                    getNodeUtility() { return m_nodeUtility; }

    // temporary until all queues handling will transfer to codeGen
    QueueDispatcherPtr&       getMmeDispatcher() { return m_mmeDispatcher; }
    QueueDispatcherPtr&       getTpcDispatcher() { return m_tpcDispatcher; }
    const QueueDispatcherMap& getDmaDispatchers() { return m_dmaDispatchers; }
    CommandQueuePtr&          getCompletionQueue() { return m_completionQueue; }
    QueueDispatcherPtr&       getRotatorDispatcher() { return m_rotatorDispatcher; }
    CommandQueuePtr&          getDownstreamQueue() { return m_downstreamQueue; }
    // dummy value temp for goya2
    QueueDispatcherPtr& getDmaDispatcher() { return m_dmaDispatchers.begin()->second; }

    void cacheBlobBuffer(const std::shared_ptr<char>& buffer) { m_cachedBuffers.emplace_back(buffer); };

    const McidConverter&           getMcidConverter() const { return m_mcidConverter; }
    const std::list<EngArcCmdPtr>& getCmeCommands() const   { return m_cmeCommands; }
    McidConverter&                 getMcidConverter()       { return m_mcidConverter; }
    std::list<EngArcCmdPtr>&       getCmeCommands()         { return m_cmeCommands; }

protected:
    virtual void initAllocators() = 0;
    virtual void generateQueues() {};     // not implemented
    virtual void runPassManager() {};     // not implemented
    virtual void addAllPasses() {};       // not implemented

    virtual void                              initTPCQueues();
    std::vector<std::vector<QueueCommandPtr>> getInitTPCQueuesCmds();
    void                                      prefetchTPCKernels(std::vector<QueueCommandPtr>& cmdsToQueue);

    NodeUtility                                          m_nodeUtility;
    std::map<kernelID, std::shared_ptr<ProgramDataBlob>> m_kernelsAddrMap;
    std::vector<deviceAddrOffset>                        m_kernelsPrintf;
    bool                                                 m_allocatorsWereInit = false;
    bool                                                 m_usingPrintf = false;
    NOPKernel                                            m_NOPKernel;
    SectionIDGenerator                                   m_MemSectionIDGenerator;
    MCIDGenerator                                        m_mcidGenerator;
    HabanaGraph*                                         m_graph;
    std::map<NodePtr, std::list<NodeROI>>                m_physicalRois;
    uint64_t                                             m_dramBaseAddr        = 0;
    uint64_t                                             m_dramSize            = 0;
    uint64_t                                             m_sramSize            = 0;
    uint64_t                                             m_synapseSramBaseAddr = 0;
    ProgramDataBlobSet                                   m_programDataBlobs;
    std::shared_ptr<SyncObjectManager>                   m_syncObjectManager   = nullptr;
    std::shared_ptr<MonitorSetupManager>                 m_monitorSetupManager = nullptr;
    QueueDispatcherPtr                                   m_mmeDispatcher;
    QueueDispatcherPtr                                   m_tpcDispatcher;
    QueueDispatcherMap                                   m_dmaDispatchers; // maps from DMA parallel level to its dispatcher
    QueueDispatcherPtr                                   m_rotatorDispatcher;
    std::map<uint32_t, CommandQueuePtr>                  m_commandQueueById;
    CommandQueuePtr                                      m_completionQueue;
    CommandQueuePtr                                      m_downstreamQueue;
    CommandQueuePtr                                      m_upstreamQueue;
    std::vector<std::pair<TensorPtr, uint32_t>>          m_inputDmaIndices;
    std::vector<std::pair<TensorPtr, uint32_t>>          m_outputDmaIndices;

    // Inserted to the queue after queue is filled with all nodes instructions
    std::map<uint32_t, std::list<SyncOrMonitor>> m_finalSyncInstructionsByQueueId;
    std::map<uint32_t, std::list<SyncOrMonitor>> m_finalSyncInstructionsByQueueIdForActivate;

    // Sync instructions by queue id,
    // Inserted to the queue after queue initialization
    std::map<HabanaDeviceType, std::vector<std::list<SyncOrMonitor>>> m_initialSyncInstructionsByQueue;

    std::unordered_map<HabanaDeviceType, unsigned> m_deviceSfgInitValue; // SFG: maps each device with its syncObj initVal

    std::vector<std::shared_ptr<char>>                   m_cachedBuffers;
    McidConverter                                        m_mcidConverter;
    std::list<EngArcCmdPtr>                              m_cmeCommands;
};

//--------------------------------------------------------------------------------
// Template function implementation
//--------------------------------------------------------------------------------
template<typename T>
void CodeGenerator::registerProgramDataBlobForDownload(T hostAddr, deviceAddrOffset deviceAddr, uint64_t size)
{
    m_programDataBlobs.insert(std::make_shared<ProgramDataBlob>(deviceAddr, hostAddr, size));
}

template<typename T>
void CodeGenerator::registerTPCProgramDataBlobForDownload(T                hostAddr,
                                                          deviceAddrOffset deviceAddr,
                                                          uint64_t         size,
                                                          kernelID         kid)
{
    const auto programDataBlobSharedPtr = std::make_shared<TPCProgramDataBlob>(deviceAddr, hostAddr, size, kid);
    m_programDataBlobs.insert(programDataBlobSharedPtr);
    m_kernelsAddrMap[kid] = programDataBlobSharedPtr;
}

//--------------------------------------------------------------------------------
// Two general purpose down-caster to get a specific CodeGenerator out of the base pointer
//--------------------------------------------------------------------------------
template<typename T>
T* downcaster(CodeGenerator* g)
{
    T* ptr = dynamic_cast<T*>(g);
    HB_ASSERT_PTR(ptr);
    return ptr;
}
template<typename T>
const T* downcaster(const CodeGenerator* g)
{
    const T* ptr = dynamic_cast<const T*>(g);
    HB_ASSERT_PTR(ptr);
    return ptr;
}
