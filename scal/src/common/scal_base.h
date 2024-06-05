#pragma once
#include <bitset>
#include <deque>
#include <string>
#include <vector>
#include <map>
#include <stdint.h>
#include <chrono>
#include <mutex>
#include "hlthunk.h"
#include "scal.h"
#include "logger.h"
#include "scal_qman_program.h"
#include "cfg_parsing_helper.h"
#include "dev_specific_info.hpp"
#include "infra/utils.h"
#include "scal_sfg_configuration_helpers.h"
#include <unordered_set>


#define EVEN_PORT_MAX_DB_FIFO 11
#define ODD_PORT_MAX_DB_FIFO  23

using namespace std::chrono;

const unsigned scal_illegal_index =  std::numeric_limits<unsigned>::max();
template <typename T>
constexpr bool isPowerOf2Unsigned(T v)
{
    static_assert(std::is_unsigned<T>::value, "type T must be unsigned");
    return v && !(v & (v - 1));
}

// return -1 for error
// by using const alignment, we can statically assert that it is a power of 2
template <unsigned alignment, typename T>
inline T alignSizeUpPowerOf2Unsigned(T size)
{
    static_assert(isPowerOf2Unsigned(alignment), "alignment must be power of 2");

    T res = (size + (alignment - 1)) & ~(alignment-1);
    if(res < size)
    {
        LOG_ERR(SCAL,"{}: failed to align {} to {}. result exceeds max size_t value ({})", __FUNCTION__, size, alignment, res);
        assert(0);
        return -1;
    }
    return res;
}


enum CoreType
{
    TPC = 0,
    MME,
    EDMA,
    PDMA,
    ROT,
    NIC,
    SCHEDULER,
    NUM_OF_CORE_TYPES
};

bool getSfgSobIdx(CoreType type,
                  uint32_t &sfgBaseSobId,
                  int (&sfgBaseSobIdPerEngine)[unsigned(EngineTypes::items_count)],
                  const int (&sfgBaseSobIdPerEngineIncrement)[unsigned(EngineTypes::items_count)]);

struct PdmaChannelInfo;
class Scal
{
public:
    struct Core;
    struct Scheduler;
    struct Stream;
    struct StreamSet;
    struct Cluster;
    struct SyncManager;

    struct ConfigVersion
    {
        unsigned major;
        unsigned minor;
        unsigned revision;
        scaljson_DEFINE_TYPE_INTRUSIVE(ConfigVersion, major, minor, revision);
    };

    class Allocator
    {
    public:
        static constexpr uint64_t c_bad_alloc = -1;
        static constexpr uint64_t c_cl_size   = 1;

        virtual ~Allocator() = default;
        virtual void setSize(uint64_t size) = 0;
        virtual uint64_t alloc(uint64_t size, uint64_t alignment = c_cl_size) = 0;
        virtual void free(uint64_t offset) = 0;
        virtual void getInfo(uint64_t& totalSize, uint64_t& freeSize) = 0;
    };

    struct MonitorsPool
    {
        std::string name;
        unsigned baseIdx = -1;
        unsigned globalBaseIdx = -1;
        unsigned size = -1;
        unsigned dcoreIndex = -1;
        unsigned smIndex = -1;
        unsigned nextAvailableIdx = -1;
        uint64_t smBaseAddr = -1;
        Scal *scal = nullptr;
    };

    struct SyncObjectsPool
    {
        std::string name;
        unsigned baseIdx = -1;
        unsigned size = -1;
        unsigned dcoreIndex = -1;
        unsigned smIndex = -1;
        unsigned nextAvailableIdx = -1;
        uint64_t smBaseAddr = -1;
        Scal *scal = nullptr;
    };

    struct SlaveSchedulerInCQ
    {
        const Scheduler* scheduler      = nullptr;
        unsigned         idxInScheduler = -1;
    };


    struct CompQTdr
    {
        bool               enabled    = false;
        unsigned           sos        = -1;
        SyncObjectsPool*   sosPool    = nullptr;
        unsigned           monitor    = -1;
        unsigned           monSmIdx   = -1;
        MonitorsPool*      monPool    = nullptr;
        unsigned           cqIdx      = -1;
        unsigned           globalCqIndex = -1;
        volatile uint64_t *enginesCtr = nullptr;
        uint64_t           prevCtr    = 0;

        static const unsigned NUM_MON = 3;

        bool                      armed = false;
        time_point<steady_clock>  armTime;
        uint64_t                  expectedCqCtr = 0;

        uint64_t debugLastCounter       = 0; // Completed
        uint64_t debugLastEnginesCtr    = 0; // Handled by S-ARC
        uint64_t debugLastExpectedCqCtr = 0; // Pushed to S-ARC

        struct StatusTdr
        {
            bool timeout;
            bool hasChanged;
        };
        StatusTdr tdr(const std::string& cgName, uint64_t ctr, uint64_t timeoutUs, bool timeoutDisabled, void (*logFunc)(int,const char *));
        void debugCheckStatus(const std::string& cgName, uint64_t ctr);
    };

    struct CompletionGroupSFGInfo
    {
        bool sfgEnabled                  = false;
        MonitorsPool *sfgMonitorsPool    = nullptr;
        SyncObjectsPool *sfgSosPool      = nullptr;
        MonitorsPool *sfgCqMonitorsPool  = nullptr;
        uint32_t sobsOffsetToNextStream  = 0;
        uint32_t baseSfgSob[unsigned(EngineTypes::items_count)] = {0};
    };
    struct CompletionGroup;
    struct HostFenceCounter
    {
        std::string      name;
        MonitorsPool    *monitorsPool    = nullptr;
        SyncObjectsPool *sosPool         = nullptr;
        SyncManager     *syncManager     = nullptr;
        bool             isrEnable       = false;
        CompletionGroup *completionGroup = nullptr;
        unsigned         isrIdx          = 0;
        unsigned         monBase         = 0;
        unsigned         soIdx           = 0;
        mutable uint64_t requestCounter  = 0; // sum of all the credits that user waited on this fence
        Scal            *scal            = nullptr;
        bool             isStub          = false;
    };

    struct CompletionGroupInterface
    {

        CompletionGroupInterface() = delete;
        CompletionGroupInterface(Scal *scal, bool isDirectMode)
        : scal(scal), isCgDirectMode(isDirectMode)
        {};

        CompletionGroupInterface(bool               isDirectMode,
                                 const std::string& name,
                                 Scal*              pScal,
                                 unsigned           globalCqIndex,
                                 unsigned           isrIndex,
                                 unsigned           longSoSmIndex,
                                 unsigned           longSoIndex)
        : name(name), scal(pScal), isCgDirectMode(isDirectMode), globalCqIndex(globalCqIndex),
          isrIdx(isrIndex), longSoSmIndex(longSoSmIndex), longSoIndex(longSoIndex)
        {};

        virtual ~CompletionGroupInterface() {};

        virtual bool getInfo(scal_completion_group_info_t& info) const = 0;
        virtual bool getInfo(scal_completion_group_infoV2_t& info) const = 0;
        virtual std::string getLogInfo(uint64_t target) const = 0;

        void getTimestampInfo(int&      fd,
                              unsigned& isrIndex,
                              uint64_t& cqCountersHandle,
                              unsigned& cqIndex) const;

        inline bool isStub() const { return scal->scalStub || isCgStub; };

        std::string        name;
        Scal*              scal           = nullptr;
        bool               isCgStub       = false;
        bool               isCgDirectMode = false;
        // the index of the cq index regard all HDcores and not inside a specific HDcore
        unsigned           globalCqIndex  = 0;
        unsigned           isrIdx         = 0;
        unsigned           longSoSmIndex  = 0;
        unsigned           longSoIndex    = 0;
        volatile uint64_t* pCounter       = nullptr;
        CompQTdr           compQTdr       = {.enabled = false};
    };

    struct CompletionGroup : public CompletionGroupInterface
    {
        CompletionGroup(Scal *scal)
        : CompletionGroupInterface(scal, false)
        {};

        virtual bool getInfo(scal_completion_group_info_t& info) const override;
        virtual bool getInfo(scal_completion_group_infoV2_t& info) const override;
        virtual std::string getLogInfo(uint64_t target) const override;

        const Scheduler*                scheduler = nullptr;
        std::vector<SlaveSchedulerInCQ> slaveSchedulers;
        unsigned                        idxInScheduler = 0;
        unsigned                        monBase = 0;
        unsigned                        monNum = 0;
        unsigned                        qmanID = -1;
        bool                            force_order  = true;
        SyncManager                     *syncManager = nullptr;
        unsigned                        sosBase = 0;
        unsigned                        sosNum = 0;
        // cq index inside a specific HDcore
        unsigned                        cqIdx = 0;
        unsigned                        actualNumberOfMonitors = 0;
        unsigned                        creditManagementSobIndex = 0;
        unsigned                        creditManagementMonIndex = 0;
        MonitorsPool*                   monitorsPool = nullptr;
        SyncObjectsPool*                sosPool = nullptr;
        SyncObjectsPool*                longSosPool = nullptr;
        CompletionGroupSFGInfo          sfgInfo;
        std::string                     fenceCounterName;
        unsigned                        nextSyncObjectMonitorId = -1;
    };

    struct DirectModeCompletionGroup : public CompletionGroupInterface
    {
        DirectModeCompletionGroup(const std::string& name,
                                  Scal*              pScal,
                                  unsigned           cqIndex,
                                  uint64_t           isrRegister,
                                  unsigned           isrIndex,
                                  unsigned           longSoSmIndex,
                                  unsigned           longSoIndex);

        virtual bool getInfo(scal_completion_group_info_t& info) const override;
        virtual bool getInfo(scal_completion_group_infoV2_t& info) const override;
        virtual std::string getLogInfo(uint64_t target) const override;

        uint64_t m_isrRegister;
    };

    class BgWork
    {
    public:
        BgWork(uint64_t timeoutUs, uint64_t timeoutDisabled);
        void addCompletionGroup(CompletionGroupInterface* pCompletionGroup);
        int tdr(void (*logFunc)(int, const char*), char *errMsg, int errMsgSize);
        void setTimeouts(uint64_t timeoutUs, uint64_t timeoutDisabled);
        int debugCheckStatus();

    private:
        time_point<steady_clock>              m_lastChange;
        std::deque<CompletionGroupInterface*> m_cgs;

        uint64_t m_timeoutUsNoProgress;
        bool     m_timeoutDisabled;
    };

    struct SyncManager
    {
        uint64_t baseAddr = 0; // 0 if the dcore is disabled
        unsigned dcoreIndex = -1;
        unsigned smIndex = -1;
        std::string qman; // empty if the dcore is disabled
        unsigned qmanID = -1;  // -1 if the qman is disabled
        bool map2userSpace = false;
        volatile uint32_t *objsHostAddress = nullptr;
        volatile uint32_t *glblHostAddress = nullptr;
        std::vector<MonitorsPool*> monitorPools;  // empty if the dcore is disabled
        std::vector<SyncObjectsPool*> soPools;    // empty if the dcore is disabled
        std::vector<CompletionGroup*> completionGroups;  // empty if the dcore is disabled
        unsigned activeCQsNr = 0;
    };

    struct SyncObjectsSetGroup
    {
        std::string name;
        Scheduler * scheduler = nullptr;
        SyncObjectsPool *sosPool = nullptr;
        MonitorsPool *resetMonitorsPool = nullptr;
        MonitorsPool *gcMonitorsPool = nullptr;
        MonitorsPool *topologyDebuggerMonitorsPool = nullptr;
        MonitorsPool *computeBack2BackMonitorsPool = nullptr;
        unsigned setSize;
        unsigned numSets;
    };

    struct Pool
    {
        enum Type
        {
            HOST = 0,
            HBM = 1,
            HBM_EXTERNAL = 2 // not allocated by SCAL, but reserved for external usage
        };

        Scal *scal = nullptr;
        std::string name;
        uint32_t coreBase = 0;       // 0 - if the memory is not addressable by the cores
        uint64_t deviceBase = 0;
        uint64_t deviceBaseAllocatedAddress = 0;
        union
        {
            uint64_t deviceHandle;   // for HBM memory
            void *hostBase;          // for HOST memory
        };
        Type type;
        uint64_t size = 0;
        unsigned addressExtensionIdx = 0; // 0 - if the memory is not addressable by the cores
        unsigned globalIdx = 0;
        Allocator *allocator = nullptr;
        bool       fromFullHbmPool = false;
        bool       fillRegion = false;

        friend void from_json(const scaljson::json& json, Pool& pool)
        {
            pool.globalIdx = 0;
            pool.deviceHandle = 0;
            pool.deviceBase = 0;
            pool.coreBase = 0;
            pool.allocator = nullptr;

            json.at("name").get_to(pool.name);
            json.at("size").get_to(pool.size);
            pool.size *= 1024 * 1024;
            json.at("type").get_to(pool.type);
            json.at("region_base").get_to(pool.addressExtensionIdx);
            if (json.find("fill_region") != json.end())
            {
                json.at("fill_region").get_to(pool.fillRegion);
            }
        }
        scaljson_JSON_SERIALIZE_ENUM_INTRUSIVE( Pool::Type, {
            {Pool::HOST, "HOST"},
            {Pool::HBM, "HBM"},
            {Pool::HBM_EXTERNAL, "HBM_EXTERNAL"},
        });
    };

    struct PdmaChannel
    {
        PdmaChannelInfo const *pdmaChannelInfo;
        unsigned engineGroup = -1;
        unsigned priority = -1;
    };

    struct Core
    {
        virtual ~Core() = default;

        // return nullptr if cannot cast
        template <class T>
        T * getAs()
        {
            return dynamic_cast<T*>(this);
        }
        template <class T>
        const T * getAs() const
        {
            return dynamic_cast<const T*>(this);
        }

        virtual unsigned getHdCoreIndex() const { return 0; }; // Only for G3

        Scal*                           scal = nullptr;
        std::string                     name;
        std::string                     qman;             // the name of a qman that initializes the core
        unsigned                        qmanID       = 0; // arcCore
        unsigned                        cpuId        = 0; // uniqie id
        unsigned                        indexInGroup = 0;
        bool                            isScheduler  = false;
        std::map<std::string, Cluster*> clusters;  // earc has only 1 entry and schedulers have more
    };

    static scal_core_handle_t toCoreHandle(const Core * core)
    {
        return (scal_core_handle_t)core;
    }

    struct ArcCore : virtual Core
    {
        std::string         arcName;
        std::string         imageName;
        uint64_t            dccmDevAddress             = 0;
        uint64_t            dccmMessageQueueDevAddress = 0;
        std::vector<Pool *> pools;
        Pool*               configPool                 = nullptr;
        void*               dccmHostAddress            = nullptr; // for engine ARCs only the first entry is valid.
        unsigned            numEnginesInGroup          = -1;      // filled for all cores - used in network EDMA only - TBD remove once every EARC can belong to a single cluster
    };

    struct Scheduler : virtual ArcCore
    {
        std::vector<Stream *>              streams;                         // scheduler only - empty for engine arcs.
        std::vector<CompletionGroup *>     completionGroups;                // scheduler only - empty for engine arcs.
        void*                              acpHostAddress        = nullptr; // scheduler only - null for engine arcs.
        void*                              afHostAddress         = nullptr; // scheduler only - null for engine arcs.
        bool                               arcFarm               = false;   // used only for scheduler; n/a for earcs
        uint64_t                           dupEngLocalDevAddress = 0;       // used only for scheduler; n/a for earcs
        std::vector<SyncObjectsSetGroup *> m_sosSetGroups;                  // TODO: check scheduler only ?
    };

    struct NicCore : virtual Core
    {
    };

    struct portToDbFifo
    {
        unsigned dbFifo;
    };

    struct EngineWithImage
    {
        std::string engine;
        std::string image;
        friend void from_json(const scaljson::json& scaljson, EngineWithImage& val)
        {
            if(scaljson.is_string())
            {
                scaljson.get_to(val.engine);
            }
            else if (scaljson.is_object())
            {
                scaljson.at("name").get_to(val.engine);
                scaljson.at("binary_name").get_to(val.image);
            }
            else
            {
                THROW_INVALID_CONFIG(scaljson, "error parsing engine. it must be string or an object with 'name' and 'binary_name' fields");
            }
        }

    };

    struct Cluster
    {
        struct DupConfig
        {
            unsigned dupTrigger = 0;
        };

        struct Queue
        {
            unsigned size = 0;
            Scheduler  *scheduler = nullptr;
            // keep so start index so that the cluster engines use the same so as their scheduler
            uint32_t sobjBaseIndex = 0;
            uint64_t sobjBaseAddr = 0;
            uint32_t monitorBaseIndex = 0;
            unsigned index = 0; // in case of nic this is the db_fifo
            unsigned group_index = 0;
            std::vector<DupConfig> dupConfigs;
            unsigned dupTrigger = 0;
            std::vector<unsigned> secondaryDupTriggers;
            unsigned group = 0;
            unsigned bit_mask_offset = 0;
            unsigned dup_trans_data_q_index = 0;
            unsigned dup_grp_eng_addr_offset = 0;
            unsigned mode = -1; // nics only
            std::map<Core*, portToDbFifo> dbFifos;
            unsigned dbFifo;
        };

        std::string name;
        std::vector<Core*> engines;
        std::map<unsigned, Queue> queues; // keep the queues sorted by index to simplify configuration
        CoreType type = NUM_OF_CORE_TYPES;
        bool localDup = false;
        bool isCompute = false;
        std::vector<unsigned> enginesPerDCore;
        std::vector<unsigned> enginesPerHDCore;
        std::map<unsigned /*hdcore*/, unsigned> edup_sync_scheme_lbw_addr; // Supported for clusters with localDup enabled
        std::map<unsigned /*hdcore*/, unsigned> edup_b2b_lbw_addr;         // Supported for clusters with localDup enabled
        unsigned numOfCentralSignals = 0;                                  // Supported for clusters with localDup enabled
        unsigned dbFifoSize = 0;      // nic cluster only
        unsigned dbFifoThreshold = 0; // nic cluster only
    };

    struct Buffer
    {
        Pool *pool = nullptr;
        uint64_t base = 0;
        uint64_t size = 0;
    };

    struct StreamInterface
    {
        virtual int submit(const unsigned pi, const unsigned submission_alignment) = 0;
        virtual int setBuffer(Buffer *buffer) = 0;
        virtual int setPriority(unsigned priority) = 0;
        virtual int getInfo(scal_stream_info_t& info) const = 0;
        virtual int getCcbBufferAlignment(unsigned& ccbBufferAlignment) const = 0;
    };

    struct Stream : public StreamInterface
    {
        virtual int submit(const unsigned pi, const unsigned submission_alignment) override;
        virtual int setBuffer(Buffer* buffer) override;
        virtual int setPriority(unsigned priority) override;
        virtual int getInfo(scal_stream_info_t& info) const override;
        virtual int getCcbBufferAlignment(unsigned& ccbBufferAlignment) const override;

        struct ClusterQueueIdxPair
        {
            Cluster *cluster = nullptr;
            unsigned queueIdx = 0;
        };

        std::string name;
        Scheduler * scheduler = nullptr;
        volatile uint32_t *pi = nullptr;
        uint16_t localPiValue = 0;
        unsigned id = 0;
        Buffer * coreBuffer = nullptr;
        unsigned dccmBufferSize = 0;
        unsigned priority = 0;
        std::vector<ClusterQueueIdxPair> queues;
        // Alignment of the Submission-Queue, which is required due to the Auto-Fetcher
        unsigned ccbBufferAlignment = 1;
        bool isStub = false;
    };

    struct StreamSet
    {
        std::string name;
        bool        isDirectMode  = false; // dma cluster only
        unsigned    streamsAmount = 0;
    };

    class PDMAStream
    {
    public:
        virtual ~PDMAStream() = default;
        virtual int setBuffer(const uint64_t deviceAddr, const unsigned size) = 0;
        virtual int setPi(const unsigned pi) = 0;
    protected:
        uint32_t m_buffSize = 0;
    };

    struct ComputeCompletionQueuesSos
    {
        uint32_t base_index = 0;
        uint32_t size       = 0;
    };

    static int create(const int fd, const std::string & configFileName, scal_arc_fw_config_handle_t fwCfg, Scal **scal);
    static void destroy(Scal * scal);

    int setup();

    int getFD() const {return m_fd;}
    virtual const char* getDefaultJsonName() const { return ":/default.json"; }
    const Pool *getPoolByName(const std::string &poolName) const;
    const Pool *getPoolByID(const unsigned poolID) const;
    Core*       getCoreByName(const std::string& coreName) const;
    template <class TCore>
    TCore*      getCoreByName(const std::string& coreName) const
    {
        Core * core = getCoreByName(coreName);
        return core ? core->getAs<TCore>() : nullptr;
    }
    template <class TCore>
    TCore * getCore(unsigned coreIdx)
    {
        return (coreIdx < m_cores.size() && m_cores[coreIdx]) ? m_cores[coreIdx]->getAs<TCore>() : nullptr;
    }
    template <class TCore>
    const TCore * getCore(unsigned coreIdx) const
    {
        return (coreIdx < m_cores.size() && m_cores[coreIdx]) ? m_cores[coreIdx]->getAs<TCore>() : nullptr;
    }
    const Core *getCoreByID(const unsigned coreID) const;
    const StreamInterface *getStreamByName(const std::string & streamName) const ;
    const Stream *getStreamByID(const Scheduler * scheduler, const unsigned streamID) const;
    const Scal::StreamSet *getStreamSetByName(const std::string &streamSetName) const;
    const CompletionGroupInterface *getCompletionGroupByName(const std::string &completionGroupName) const;
    const Cluster* getClusterByName(const std::string  &clusterName) const;
    const SyncObjectsPool *getSoPool(const std::string &poolName) const;
    const SyncManager *getSyncManager(const unsigned index) const;
    const MonitorsPool *getMonitorPool(const std::string &poolName) const;
    const HostFenceCounter *getHostFenceCounter(const std::string &counterName) const;
    unsigned getFWCombinedVersion(bool scheduler = true) const;
    uint64_t getTimeout() const { return m_timeoutMicroSec; }
    bool     getTimeoutDisabled() const { return m_timeoutDisabled; }
    int      bgWork(void (*logFunc)(int, const char*), char *errMsg, int errMsgSize);
    int      debugBackgroundWork();
    int      runCores(const uint32_t* coreIds, const uint32_t* coreQmanIds, uint32_t numOfCores);

    virtual uint32_t getSRAMSize() const = 0;
    virtual int setStreamPriority(Stream *stream, unsigned priority) const = 0;
    virtual int setStreamBuffer(Stream *stream, Buffer *buffer) const = 0;
    virtual int streamSubmit(Stream *stream, const unsigned pi, const unsigned submission_alignment) = 0;

    static int streamSetBuffer(StreamInterface *stream, Buffer *buffer);
    static int streamSetPriority(StreamInterface *stream, unsigned priority);

    static int hostFenceCounterWait(const HostFenceCounter *hostFenceCounter, const uint64_t num_credits, uint64_t timeout);
    static int hostFenceCounterEnableIsr(HostFenceCounter *hostFenceCounter, bool enableIsr);
    static int getHostFenceCounterInfo(const HostFenceCounter * hostFenceCounter, scal_host_fence_counter_info_t *info);
    int completionGroupWait(const CompletionGroupInterface *completionGroup, const uint64_t target, uint64_t timeout, bool alwaysWaitForInterrupt);
    static int completionGroupRegisterTimestamp(const CompletionGroupInterface *completionGroup, const uint64_t target, const uint64_t timestampsHandle, const uint32_t timestampsOffset);
    static bool scalStub;
    static const scaljson::json getAsicJson(const scaljson::json &json, const char* asicType);
    static void getConfigVersion(const scaljson::json &json, ConfigVersion &version);
    virtual int getNumberOfSignalsPerMme() const = 0;
    Buffer * createAllocatedBuffer();
    void deleteAllocatedBuffer(const Buffer * buffer);
    void deleteAllAllocatedBuffers();

    std::unique_ptr<DevSpecificInfo> m_devSpecificInfo;
    static void printConfigInfo(const std::string &configFileName, const std::string &content);

    virtual unsigned getSchedulerModeCQsCount() const = 0;

    // The Direct-Mode's PDMA uses a local CQ instead of the Scheduler's ones
    uint64_t getDirectModeCqsAmount() const { return m_directModePdmaChannelsAmount; };
    inline uint64_t getCqsSize() const { return getDirectModeCqsAmount() + m_completionGroups.size(); };

    virtual int allocAndSetupPortDBFifo(ibv_context* ibv_ctxt) { LOG_ERR(SCAL, "allocAndSetupPortDBFifo unsupported"); return SCAL_FAILURE; };
    virtual int allocAndSetupPortDBFifoV2(const scal_ibverbs_init_params* ibvInitParams,
                                          struct hlibdv_usr_fifo       ** createdFifoBuffers,
                                          uint32_t                      * createdFifoBuffersCount)
    {
        LOG_ERR(SCAL, "allocAndSetupPortDBFifoV2 unsupported");
        return SCAL_FAILURE;
    }
    virtual int getDbFifoParams_tmp(struct hlibdv_usr_fifo_attr_tmp* nicUserDbFifoParams, unsigned* nicUserDbFifoParamsCount) { LOG_ERR(SCAL, "getDbFifoParams unsupported"); return SCAL_FAILURE; };

    virtual unsigned getUsedSmBaseAddrs(const scal_sm_base_addr_tuple_t ** smBaseAddrDb) = 0;

    int setTimeouts(const scal_timeouts_t * timeouts);
    int getTimeouts(scal_timeouts_t * timeouts);
    int disableTimeouts(bool disableTimeouts_);
public:
    static constexpr unsigned c_sync_object_group_size = 8;
    static constexpr unsigned c_sos_for_completion_group_credit_management = 2;

protected:
    enum MemoryExtensionRange
    {
        ICCM            = 0x0,   // not configurable
        SRAM            = 0x1,   // SRAM_BASE
        CFG             = 0x2,   // not configurable
        GP0             = 0x3,   // GP[0]
        HBM0            = 0x4,   // HBM[0] +Offset
        HBM1            = 0x5,   // HBM[1] +Offset
        HBM2            = 0x6,   // HBM[2] +Offset
        HBM3            = 0x7,   // HBM[3] +Offset
        DCCM            = 0x8,   // not configurable
        PCI             = 0x9,   // PCI
        GP1             = 0xa,   // GP[1];
        GP2             = 0xb,   // GP[2];
        GP3             = 0xc,   // GP[3];
        GP4             = 0xd,   // GP[4];
        GP5             = 0xe,   // GP[5];
        LBU             = 0xf,   // not configurable

        RANGES_NR
    };

    static constexpr unsigned c_cl_size = 128;
    static constexpr unsigned c_host_page_size = 4096;
    static constexpr uint32_t c_arc_lbw_access_msb = 0xf0000000;
    static constexpr unsigned c_image_hbm_size = 256 * 1024;
    static constexpr unsigned c_engine_image_dccm_size = 32 * 1024;
    static constexpr unsigned c_arc_farm_dccm_size = 64 * 1024;
    static constexpr unsigned c_cq_ctrs_in_dcore = 64;
    static constexpr unsigned c_max_so_range_per_monitor = 2048;
    static constexpr unsigned c_host_fence_ctr_mon_nr = 3;


    static constexpr unsigned c_core_memory_extension_range_size = 256 * 1024 * 1024; // 256 MBs
    static constexpr unsigned c_host_pool_size_alignment = 2 * 1024 * 1024; // 2MB
    static constexpr unsigned c_scheduler_image_dccm_size = 64 * 1024;
    static constexpr unsigned c_acp_block_size = 4 * 1024;
    static constexpr unsigned c_aux_block_size = 4 * 1024;
    static constexpr uint64_t c_minimal_pool_size = 512*1024*1024;  //512MB
    static constexpr unsigned c_core_counter_max_value = (1<< 16);
    static constexpr unsigned c_core_counter_max_ccb_size = (1<< 24);
    static constexpr unsigned c_num_max_user_streams = 32;
    static constexpr unsigned c_wait_cycles = 32;
    static constexpr unsigned c_max_sos_per_sync_manager = 8192;
    static constexpr unsigned c_completion_queue_monitors_set_size = 3;
    static constexpr unsigned c_so_group_size = 8; // in terms of regular so's
    static constexpr unsigned c_monitors_for_completion_group_credit_management = 6;
    static constexpr unsigned c_sos_for_distributed_completion_group_credit_management = 2;
    static constexpr unsigned c_monitors_for_distributed_completion_group_credit_management = 6;
    static constexpr unsigned c_cq_size_log2 = 3;
    static constexpr unsigned c_max_cq_cntrs = 536;
    static constexpr unsigned c_cq_size      = 1 << c_cq_size_log2;

    static constexpr char c_config_key_version[] = "version";
    static constexpr char c_bin_path_env_var_name[] = "HABANA_SCAL_BIN_PATH";
    static constexpr char c_engines_fw_build_path_env_var_name[] = "ENGINES_FW_RELEASE_BUILD";
    static constexpr char c_config_key_backward_compatibleness[] = "backward_compatibleness";
    static constexpr char c_config_key_memory[] = "memory";
        static constexpr char c_config_key_memory_control_cores_memory_pools[] = "control_cores_memory_pools";
        static constexpr char c_config_key_memory_memory_groups[] = "memory_groups";
        static constexpr char c_config_key_memory_binary_pool[] = "binary_pool";
        static constexpr char c_config_key_memory_global_pool[] = "global_pool";
    static constexpr char c_config_key_memory_memory_groups_name[] = "name";
            static constexpr char c_config_key_memory_memory_groups_pools[] = "pools";
            static constexpr char c_config_key_memory_memory_groups_config_pool[] = "configuration_pool";
    static constexpr char c_config_key_cores[] = "cores";
        static constexpr char c_config_key_cores_schedulers[] = "schedulers";
            static constexpr char c_config_key_cores_schedulers_name[] = "name";
            static constexpr char c_config_key_cores_schedulers_binary_name[] = "binary_name";
            static constexpr char c_config_key_cores_schedulers_memory_group[] = "memory_group";
            static constexpr char c_config_key_cores_schedulers_core[] = "core";
            static constexpr char c_config_key_cores_schedulers_qman[] = "qman";
        static constexpr char c_config_key_cores_engine_clusters[] = "engine_clusters";
            static constexpr char c_config_key_cores_engine_clusters_name[] = "name";
            static constexpr char c_config_key_cores_engine_clusters_qman[] = "qman";
            static constexpr char c_config_key_cores_engine_clusters_binary_name[] = "binary_name";
            static constexpr char c_config_key_cores_engine_clusters_memory_group[] = "memory_group";
            static constexpr char c_config_key_cores_engine_clusters_engines[] = "engines";
            static constexpr char c_config_key_cores_engine_clusters_is_compute[] = "is_compute";
            static constexpr char c_config_key_cores_engine_clusters_queues[] = "queues";
                static constexpr char c_config_key_cores_engine_clusters_queues_index[] = "index";
                static constexpr char c_config_key_cores_engine_clusters_queues_scheduler[] = "scheduler";
                    static constexpr char c_config_key_cores_engine_clusters_queues_scheduler_name[] = "name";
                    static constexpr char c_config_key_cores_engine_clusters_queues_scheduler_dup_trigger[] = "dup_trigger";
                    static constexpr char c_config_key_cores_engine_clusters_queues_scheduler_group[] = "group";
    static constexpr char c_config_key_streams_set[] = "streams_sets";
        static constexpr char c_config_key_streams_set_name_prefix[] = "name_prefix";
        static constexpr char c_config_key_streams_set_scheduler[] = "scheduler";
        static constexpr char c_config_key_streams_set_base_idx[] = "base_idx";
        static constexpr char c_config_key_streams_set_streams_nr[] = "streams_nr";
        static constexpr char c_config_key_streams_set_dccm_buffer_size[] = "dccm_buffer_size";
        static constexpr char c_config_key_streams_set_is_stub[] = "is_stub";
    static constexpr char c_config_key_sync[] = "sync";
        static constexpr char c_config_key_sync_managers[] = "sync_managers";
            static constexpr char c_config_key_sync_managers_sos_pools[] = "sos_pools";
                static constexpr char c_config_key_sync_managers_sos_pools_name[] = "name";
                static constexpr char c_config_key_sync_managers_sos_pools_base_index[] = "base_index";
                static constexpr char c_config_key_sync_managers_sos_pools_size[] = "size";
                static constexpr char c_config_key_sync_managers_sos_pools_align[] = "align";
            static constexpr char c_config_key_sync_managers_monitors_pools[] = "monitors_pools";
                static constexpr char c_config_key_sync_managers_monitors_pools_name[] = "name";
                static constexpr char c_config_key_sync_managers_monitors_pools_base_index[] = "base_index";
                static constexpr char c_config_key_sync_managers_monitors_pools_size[] = "size";
                static constexpr char c_config_key_sync_managers_monitors_pools_align[] = "align";
            static constexpr char c_config_key_sync_managers_map_to_userspace[] = "map_to_userspace";
            static constexpr char c_config_key_sync_managers_qman[] = "qman";
            static constexpr char c_config_key_sync_managers_completion_queues[] = "completion_queues";
                static constexpr char c_config_key_sync_managers_completion_queues_name_prefix[] = "name_prefix";
                static constexpr char c_config_key_sync_managers_completion_queues_schedulers[] = "schedulers";
                static constexpr char c_config_key_sync_managers_completion_queues_number_of_instances[] = "number_of_instances";
                static constexpr char c_config_key_sync_managers_completion_queues_sos_pool[] = "sos_pool";
                static constexpr char c_config_key_sync_managers_completion_queues_sos_depth[] = "sos_depth";
                static constexpr char c_config_key_sync_managers_completion_queues_monitors_pool[] = "monitors_pool";
                static constexpr char c_config_key_sync_managers_completion_queues_monitors_depth[] = "monitors_depth";
                static constexpr char c_config_key_sync_managers_completion_queues_force_order[] = "force_order";
                static constexpr char c_config_key_sync_managers_completion_queues_is_compute_completion_queue[] = "is_compute_completion_queue";
                static constexpr char c_config_key_sync_managers_completion_queues_is_tdr[] = "tdr";
                static constexpr char c_config_key_sync_managers_completion_queues_enable_isr[] = "enable_isr";
                static constexpr char c_config_key_sync_managers_completion_queues_is_stub[] = "is_stub";
                static constexpr char c_config_key_sync_managers_completion_queues_long_sos[] = "long_sos";
            static constexpr char c_config_key_sync_managers_host_fence_counters[] = "host_fence_counters";
                static constexpr char c_config_key_sync_managers_host_fence_counters_name_prefix[] = "name_prefix";
                static constexpr char c_config_key_sync_managers_host_fence_counters_number_of_instances[] = "number_of_instances";
                static constexpr char c_config_key_sync_managers_host_fence_counters_sos_pool[] = "sos_pool";
                static constexpr char c_config_key_sync_managers_host_fence_counters_monitors_pool[] = "monitors_pool";
                static constexpr char c_config_key_sync_managers_host_fence_counters_enable_isr[] = "enable_isr";
                static constexpr char c_config_key_sync_managers_host_fence_counters_is_stub[] = "is_stub";
        static constexpr char c_config_key_direct_mode_pdma_channels_long_so_pool[] = "direct_mode_pdma_channels_long_sos";
        static constexpr char c_config_key_completion_queues_long_so_pool[] = "completion_queues_long_so_pool";
        static constexpr char c_config_key_sos_sets[] = "sos_sets";
            static constexpr char c_config_key_sos_sets_name[] = "name";
            static constexpr char c_config_key_sos_sets_scheduler[] = "scheduler";
            static constexpr char c_config_key_sos_sets_set_size[] = "set_size";
            static constexpr char c_config_key_sos_sets_num_sets[] = "num_sets";
            static constexpr char c_config_key_sos_sets_sos_pool[] = "sos_pool";
            static constexpr char c_config_key_sos_sets_reset_monitors_pool[] = "reset_monitors_pool";
            static constexpr char c_config_key_sos_sets_gc_monitors_pool[] = "gc_monitors_pool";
            static constexpr char c_config_key_sos_sets_cme_monitors_pool[] = "cme_monitors_pool";
            static constexpr char c_config_key_sos_sets_compute_back2back_monitors[] = "compute_back2back_monitors";
            static constexpr char c_config_key_sos_sets_topology_debugger_monitors_pool[] = "topology_debugger_monitors_pool";
        static constexpr char c_config_key_completion_group_credits[] = "completion_group_credits";
            static constexpr char c_config_key_completion_group_credits_sos_pool[] = "sos_pool";
            static constexpr char c_config_key_completion_group_credits_monitors_pool[] = "monitors_pool";
        static constexpr char c_config_key_distributed_completion_group_credits[] = "distributed_completion_group_credits";
            static constexpr char c_config_key_distributed_completion_group_credits_sos_pool[] = "sos_pool";
            static constexpr char c_config_key_distributed_completion_group_credits_monitors_pool[] = "monitors_pool";

    virtual bool submitQmanWkld(const Qman::Workload &wkld) = 0;
    void initCountersArray(uint64_t countersArray[], unsigned long arraySize) const;
    void updateCountersArray(uint64_t countersArray[], unsigned long arraySize, bool &inProgress) const;
    void logCompletionGroupsCtrs() const;

    class DeviceDmaBuffer
    {
    public:
        DeviceDmaBuffer();
        DeviceDmaBuffer(Pool * pool, const uint64_t size, const uint64_t alignment = c_cl_size);
        DeviceDmaBuffer(const uint64_t deviceAddr, Scal * scal, const uint64_t size, const uint64_t alignment = c_cl_size);
        virtual ~DeviceDmaBuffer();

        // delete copy constructor and override move constructor to prevent call to destructor and releasing allocated memory
        DeviceDmaBuffer(const DeviceDmaBuffer &other) = delete;
        DeviceDmaBuffer& operator=(const DeviceDmaBuffer&) = delete;
        DeviceDmaBuffer(DeviceDmaBuffer &&other) = delete; // move constructor

        void *getHostAddress();
        uint32_t getCoreAddress();
        uint64_t getDeviceAddress();
        bool commit(Qman::Workload *wkld = nullptr);
        bool init(Pool * pool, const uint64_t size, const uint64_t alignment = c_cl_size);
        bool init(const uint64_t deviceAddr, Scal * scal, const uint64_t size, const uint64_t alignment = c_cl_size);
        uint64_t getSize() const { return m_size;}

    protected:

        void *m_hostAddr;
        uint64_t m_base;
        Pool *m_pool;
        uint64_t m_size;
        uint64_t m_alignment;
        bool m_initialized;
        Scal *m_scal;
        uint64_t m_mappedHostAddr;
    };

    struct MemoryGroup
    {
        std::vector<Pool *> pools;
        Pool * configPool = nullptr;
    };

    typedef std::map<std::string, MemoryGroup> MemoryGroups;

    // scal_cfg functions
    virtual int parseConfigFile(const std::string & configFileName) = 0;
    void parseBinPath(const scaljson::json &json, const ConfigVersion &version);
    virtual void parseBackwardCompatibleness(const scaljson::json &json) = 0;
    virtual void parseMemorySettings(const scaljson::json &json, const ConfigVersion &version, MemoryGroups &groups) = 0;
    void parseMemoryGroups(const scaljson::json &memoryGroupJson, const ConfigVersion &version, MemoryGroups &groups);
    void parseCores(const scaljson::json &json, const ConfigVersion &version, const MemoryGroups &groups);
    virtual void parseSchedulerCores(const scaljson::json &schedulersJson, const ConfigVersion &version, const MemoryGroups &groups) = 0;
    virtual void parseClusterQueues(const scaljson::json& queuesJson, const ConfigVersion& version, Cluster* cluster) = 0;
    virtual void parseEngineCores(const scaljson::json &enginesJson, const ConfigVersion &version, const MemoryGroups &groups) = 0;
    void parseStreams(const scaljson::json &json, const ConfigVersion &version);
    void parseSyncInfo(const scaljson::json &json, const ConfigVersion &version);
    virtual void parseSyncManagers(const scaljson::json &syncManagersJson, const ConfigVersion &version) = 0;
    virtual void parseSosPools(const scaljson::json &sosPoolsJson, const ConfigVersion &version, const unsigned smID) = 0;
    virtual void parseMonitorPools(const scaljson::json &monitorPoolJson, const ConfigVersion &version, const unsigned smID) = 0;
    virtual void parseSyncManagersCompletionQueues(const scaljson::json &syncObjectJson, const ConfigVersion &version) = 0;
    virtual void parseTdrCompletionQueues(const scaljson::json & completionQueueJsonItem, unsigned numberOfInstances, CompletionGroupInterface& cq) = 0;
    // todo - nazulay - comment: parseCompletionQueues has different parameters
    virtual void parseHostFenceCounters(const scaljson::json &syncObjectJson, const ConfigVersion &version) = 0;
    virtual void parseSoSets(const scaljson::json &soSetJson, const ConfigVersion &version) = 0;
    void parseCompletionGroupCredits(const scaljson::json &fwCreditsJson, const ConfigVersion &version);
    void parseDistributedCompletionGroupCredits(const scaljson::json &fwCreditsJson, const ConfigVersion &version);
    void parseSfgPool(const scaljson::json &syncObjectJson, SyncObjectsPool** sfgSosPool, MonitorsPool** sfgMonitorsPool, MonitorsPool** sfgCqMonitorsPool);
    void parseSfgCompletionQueueConfig(const scaljson::json &completionQueueJsonItem, CompletionGroup &cq, SyncObjectsPool* sfgSosPool, MonitorsPool* sfgMonitorsPool, MonitorsPool* sfgCqMonitorsPool);

    typedef std::map<uint64_t, uint32_t> RegToVal;
    static bool addRegisterToMap(RegToVal &regToVal, uint64_t key, uint32_t value, bool replace = false);
    void * mapLBWBlock(const uint64_t lbwAddress, const uint32_t size);
    int unmapLBWBlocks();
    int releaseMemoryPools();
    int fillRegions(unsigned core_memory_extension_range_size);
    int releaseCompletionQueues();
    int allocateCompletionQueues();
    int initTimeout();
    void allocateCqIndex(unsigned& cqIndex, unsigned& globalCqIndex, const scaljson::json& json, const unsigned smID, unsigned dcoreIndex, uint32_t firstAvailableCq);

    virtual void configureMonitor(Qman::Program& prog, unsigned monIdx, uint64_t smBase, uint32_t configValue, uint64_t payloadAddress, uint32_t payloadData) = 0;
    template <class TsfgMonitorsHierarchyMetaData>
    void configSfgSyncHierarchy(Qman::Program & prog, const CompletionGroup *cg, uint32_t maxNbMessagesPerMonitor);
    template <template<class> class TsfgMonitorsHierarchyMetaData, class TSobTypes>
    void configSfgMetaData(const CompletionGroup *cg, TsfgMonitorsHierarchyMetaData<TSobTypes> &sfgMD, uint32_t maxNbMessagesPerMonitor);
    template <template<class> class TsfgMonitorsHierarchyMetaData, class TSobTypes>
    void configSfgFirstLayer(Qman::Program & prog, TsfgMonitorsHierarchyMetaData<TSobTypes> &sfgMD);
    template <template<class> class TsfgMonitorsHierarchyMetaData, class TSobTypes>
    void configSfgLongSoAndCqLayer(Qman::Program & prog, TsfgMonitorsHierarchyMetaData<TSobTypes> &sfgMD);
    template <template<class> class TsfgMonitorsHierarchyMetaData, class TSobTypes>
    void configSfgRearmLayer(Qman::Program & prog, TsfgMonitorsHierarchyMetaData<TSobTypes> &sfgMD);
    template <template<class> class TsfgMonitorsHierarchyMetaData, class TSobTypes>
    void configSfgChainInRearmLayer(Qman::Program & prog, TsfgMonitorsHierarchyMetaData<TSobTypes> &sfgMD, uint32_t prevMonIdx);
    const int                                                m_fd;
    const struct hlthunk_hw_ip_info                          m_hw_ip;
    struct hl_info_module_params                             m_hl_info = {};
    std::vector<Core*>                                       m_cores;
    std::vector<SyncManager>                                 m_syncManagers;
    std::map<std::string, Pool>                              m_pools;
    std::mutex                                               m_allocatedBuffersMtx;
    std::unordered_set<const Buffer*>                        m_allocatedBuffers; // for deletion in scal_destroy in case user did not delete them
    std::map<std::string, CompletionGroup>                   m_completionGroups;
    std::map<std::string, Stream>                            m_streams;
    std::map<std::string, StreamSet>                         m_streamSets;
    std::map<std::string, Cluster>                           m_clusters;
    std::vector<Cluster*>                                    m_computeClusters;
    std::vector<Cluster*>                                    m_nicClusters;
    std::map<std::string, SyncObjectsPool>                   m_soPools;
    std::map<std::string, MonitorsPool>                      m_monitorPools;
    std::map<std::string, HostFenceCounter>                  m_hostFenceCounters;
    std::map<std::string, std::shared_ptr<SyncObjectsSetGroup>> m_soSetsGroups;
    std::map<std::string, std::vector<CompletionGroup*>>     m_schedulersCqsMap; // map of: scheduler -> CQs
    std::map<std::string, std::vector<SyncObjectsSetGroup*>> m_schedulersSosMap; // map of: scheduler -> so sets
    SyncObjectsPool*                                         m_completionGroupCreditsSosPool;
    MonitorsPool*                                            m_completionGroupCreditsMonitorsPool;
    SyncObjectsPool*                                         m_distributedCompletionGroupCreditsSosPool;
    MonitorsPool*                                            m_distributedCompletionGroupCreditsMonitorsPool;
    MonitorsPool*                                            m_cmeMonitorsPool                   = nullptr;
    MonitorsPool*                                            m_cmeEnginesMonitorsPool            = nullptr;
    std::vector<std::string>                                 m_fwImageSearchPath;
    Pool*                                                    m_binaryPool                        = nullptr;
    Pool*                                                    m_globalPool                        = nullptr;
    Pool                                                     m_fullHbmPool{};
    std::vector<uint64_t>                                    m_extraMapping;
    uint64_t                                                 m_coresBinaryDeviceAddress          = 0ULL;
    uint64_t                                                 m_completionQueuesHandle            = 0ULL;
    volatile uint64_t*                                       m_completionQueueCounters           = nullptr;
    uint64_t                                                 m_completionQueueCountersDeviceAddr = 0ULL;
    uint64_t                                                 m_completionQueueCountersSize       = 0ULL;
    uint32_t                                                 m_fw_sched_major_version            = 0;
    uint32_t                                                 m_fw_sched_minor_version            = 0;
    uint32_t                                                 m_fw_eng_major_version              = 0;
    uint32_t                                                 m_fw_eng_minor_version              = 0;
    std::vector<std::pair<void *, uint32_t>>                 m_mappedLBWBlocks;
    uint64_t                                                 m_timeoutMicroSec                   = 0;
    uint64_t                                                 m_timeoutUsNoProgress               = 0;
    bool                                                     m_timeoutDisabled                   = false;
    SyncObjectsPool*                                         m_computeCompletionQueuesSos        = nullptr;
    unsigned                                                 m_nextIsr                           = 0;
    std::unique_ptr<BgWork>                                  m_bgWork;
    unsigned                                                 m_schedulerNr                       = 0;
    bool                                                     m_isInternalJson                    = false;

    std::map<std::string, CompletionGroupInterface*> m_directModeCompletionGroups;
    std::map<std::string, StreamInterface*>          m_directModePdmaChannelStreams;
    uint32_t                                         m_directModePdmaChannelsAmount = 0;

    std::deque<CompletionGroupInterface*> m_cgs;

    Scal(const int fd,
         const struct hlthunk_hw_ip_info & hw_ip,
         const unsigned coresNr,
         const unsigned dcoresNr,
         const unsigned enginesTypesNr,
         std::unique_ptr<DevSpecificInfo> devApi);

    virtual ~Scal();
    virtual int init(const std::string & configFileName) = 0;
    int openConfigFileAndParseJson(const std::string & configFileName, scaljson::json &json);
    virtual void addFencePacket(Qman::Program& program, unsigned id, uint8_t targetVal, unsigned decVal) = 0;
    virtual void enableHostFenceCounterIsr(CompletionGroup * cg, bool enableIsr) = 0;
    void handleSlaveCqs(CompletionGroup* pCQ, const std::string& masterSchedulerName);
};
