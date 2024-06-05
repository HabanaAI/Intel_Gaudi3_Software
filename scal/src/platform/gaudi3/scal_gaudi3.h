#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <bitset>
#include <future>

#include "common/scal_base.h"
#include "common/json.hpp"
#include "gaudi3/asic_reg/gaudi3_blocks.h"
#include "gaudi3_arc_common_packets.h"
#include "gaudi3_arc_host_packets.h"
#include "common/scal_qman_program.h"

namespace gaudi3{
    struct block_pdma_ch_a;
}
class Scal_Gaudi3 : public Scal
{
    struct G3Core : virtual Core
    {
        virtual unsigned getHdCoreIndex() const override { return hdCore; };

        unsigned dCore  = -1; // Gaudi3 only
        unsigned hdCore = -1; // Gaudi3 only
        unsigned indexInGroupInDCore  = -1; // Gaudi3 only
        unsigned indexInGroupInHdCore = -1; // Gaudi3 only
    };

    struct G3ArcCore : virtual ArcCore, virtual G3Core
    {
    };

    struct G3Scheduler : virtual Scheduler, virtual G3ArcCore
    {
        std::vector<PdmaChannel> pdmaChannels;
    };

    struct G3NicCore : virtual NicCore, virtual G3ArcCore
    {
        unsigned        ports[2];
        std::bitset<2>  portsMask;
    };

    struct G3CmeCore : virtual G3Scheduler
    {
    };

public:
    Scal_Gaudi3(
            const int fd,
            const struct hlthunk_hw_ip_info & hw_ip,
            scal_arc_fw_config_handle_t fwCfg);

    virtual ~Scal_Gaudi3();

    virtual const char* getDefaultJsonName() const override { return "gaudi3/default.json"; }
    uint32_t getSRAMSize() const override;
    int setStreamPriority(Stream *stream, unsigned priority) const override;
    int setStreamBuffer(Stream *stream, Buffer *buffer) const override;
    int getNumberOfSignalsPerMme() const override;
    int streamSubmit(Stream *stream, const unsigned pi, const unsigned submission_alignment) override;
    int allocAndSetupPortDBFifo(ibv_context* ibv_ctxt) override;
    int allocAndSetupPortDBFifoV2(const scal_ibverbs_init_params* ibvInitParams,
                                  struct hlibdv_usr_fifo       ** createdFifoBuffers,
                                  uint32_t                      * createdFifoBuffersCount) override;

    int getDbFifoParams_tmp(struct hlibdv_usr_fifo_attr_tmp* nicUserDbFifoParams, unsigned* nicUserDbFifoParamsCount) override;
    unsigned getUsedSmBaseAddrs(const scal_sm_base_addr_tuple_t ** smBaseAddrDb) override;

    static constexpr unsigned c_sync_managers_per_hdcores = 2;
    static constexpr unsigned c_cores_nr = CPU_ID_MAX;
    static constexpr unsigned c_scheduler_nr = CPU_ID_SCHED_MAX;
    static constexpr unsigned c_af_block_size = 4 * 1024;
    static constexpr unsigned c_pdma_ch_block_size = 4 * 1024;
    static constexpr unsigned c_dcores_nr = 4;
    static constexpr unsigned c_hdcores_nr = 8;
    static constexpr unsigned c_cq_ctrs_in_hdcore = 64;
    static constexpr unsigned c_hbm_bin_buffer_size = c_cores_nr * c_image_hbm_size;
    static constexpr uint64_t c_local_address = 0x0300007ffc500000ull;
    static constexpr unsigned c_ccb_buffer_alignment = 128;
    static constexpr unsigned c_dccm_to_qm_offset = (unsigned)mmHD0_TPC0_QM_BASE - (unsigned)mmHD0_TPC0_QM_DCCM_BASE;
    static constexpr unsigned c_nic_dccm_to_qm_offset = (unsigned)mmD0_NIC0_QM_BASE - (unsigned)mmD0_NIC0_QM_DCCM_BASE;
    static constexpr unsigned c_nic_dccm_to_qpc_offset = (unsigned)mmD0_NIC0_QPC_BASE - (unsigned)mmD0_NIC0_QM_DCCM_BASE;
    static constexpr unsigned c_dccm_to_dup_offset = (unsigned)mmHD0_ARC_FARM_ARC0_DUP_ENG_BASE - (unsigned)mmHD0_ARC_FARM_ARC0_DCCM0_BASE;
    static constexpr unsigned c_dccm_to_core_offset = (unsigned)mmHD0_TPC0_CFG_BASE - (unsigned)mmHD0_TPC0_QM_DCCM_BASE;
    static constexpr unsigned c_message_short_base_index = 0;
    static constexpr unsigned c_sync_managers_nr = c_sync_managers_per_hdcores * c_hdcores_nr;
    static constexpr unsigned c_max_monitors_per_sync_manager = 1024;
    static constexpr unsigned c_user_isr_nr = 256;
    static constexpr unsigned c_max_push_regs_per_dup = 512;
    static constexpr unsigned c_max_queues_per_group_in_arc = 8;
    static constexpr unsigned c_maximum_messages_per_monitor = 16;
    static constexpr unsigned c_ports_count_per_nic = 2;
    static constexpr unsigned c_nics_count = 12;
    static constexpr unsigned c_ports_count = c_ports_count_per_nic * c_nics_count;
    static constexpr unsigned c_soset_local_monitors_nr = 10;
    static constexpr unsigned c_soset_local_sobs_nr = NUM_SOS_PER_LOCAL_SO_SET_IN_HDCORE;
    static constexpr unsigned c_min_priority = 0;
    static constexpr unsigned c_max_priority = 7;
    static constexpr unsigned c_pdma_channels_nr = 24;
    static constexpr unsigned c_nic_db_fifo_size = 256;
    static constexpr unsigned c_nic_db_fifo_bp_treshold = 128;

    static_assert(Scal_Gaudi3::c_hdcores_nr * c_cq_ctrs_in_dcore + Scal_Gaudi3::c_pdma_channels_nr <= c_max_cq_cntrs);

protected:

    static constexpr char c_config_key_backward_compatibleness_auto_fetcher[] = "auto_fetcher";
    static constexpr char c_config_key_memory_pdma_pool[] = "pdma_pool";
    static constexpr char c_config_key_pdma_channel[] = "pdma_channel_config";
    static constexpr char c_config_key_pdma_channel_syncman_index[] = "syncman_idx";
    static constexpr char c_config_key_pdma_channel_monitor_index[] = "monitor_idx";
    static constexpr char c_config_key_pdma_channel_completion_queue_index[] = "cq_idx";
    static constexpr char c_config_key_pdma_channel_pdma_channel[] = "pdma_channel";
    static constexpr char c_config_key_pdma_clusters[] = "pdma_clusters";
    static constexpr char c_config_key_pdma_clusters_name[] = "name";
    static constexpr char c_config_key_pdma_clusters_is_direct_mode[] = "is_direct_mode";
    static constexpr char c_config_key_pdma_clusters_is_isr_enabled[] = "is_isr_enabled";
    static constexpr char c_config_key_pdma_clusters_is_tdr[] = "tdr";
    static constexpr char c_config_key_pdma_clusters_priority[] = "cluster_priority";
    static constexpr char c_config_key_pdma_clusters_streams[] = "streams";
    static constexpr char c_config_key_pdma_clusters_stream_engines[] = "engines";
    static constexpr char c_config_key_pdma_clusters_engines[] = "engines";
    static constexpr char c_config_key_pdma_clusters_channels[] = "channels";
    static constexpr char c_config_key_pdma_clusters_channels_index[] = "index";
    static constexpr char c_config_key_pdma_clusters_channels_priority[] = "priority";
    static constexpr char c_config_key_pdma_clusters_channels_scheduler[] = "scheduler";
    static constexpr char c_config_key_pdma_clusters_channels_scheduler_name[] = "name";
    static constexpr char c_config_key_pdma_clusters_channels_scheduler_group[] = "group";
    static constexpr char c_config_key_cores_engine_clusters_is_local_dup[] = "is_local_dup";
    static constexpr char c_config_key_cores_engine_clusters_queues_scheduler_dup_trigger_dup[] = "dup";
    static constexpr char c_config_key_cores_nic_clusters[] = "nic_clusters";
    static constexpr char c_config_key_cores_nic_clusters_name[] = "name";
    static constexpr char c_config_key_cores_nic_clusters_ports[] = "ports"; // {name, mask}
    static constexpr char c_config_key_cores_nic_clusters_queues[] = "queues";
    static constexpr char c_config_key_sync_managers_index[] = "index";
    static constexpr char c_config_key_sos_sets_local_sos_pool[] = "local_sos_pool";
    static constexpr char c_config_key_sos_sets_local_mon_pool[] = "local_mon_pool";
    static constexpr char c_config_key_sos_sets_local_barrier_sos_pool[] = "local_barrier_sos_pool";
    static constexpr char c_config_key_sos_sets_local_barrier_mons_pool[] = "local_barrier_mons_pool";

    union sync_object_update
    {
        struct
        {
            uint32_t sync_value :16;
            uint32_t reserved1  :8;
            uint32_t long_mode  :1;
            uint32_t reserved2  :5;
            uint32_t te         :1;
            uint32_t mode       :1;
        } so_update;
        uint32_t raw;
    };

private:
    std::vector<uint64_t>                  m_spdmaMsgBaseAddrDb;
    std::vector<scal_sm_base_addr_tuple_t> m_smBaseAddrDb;

    class PDMAStream_Gaudi3 : public PDMAStream
    {
    public:
        const unsigned c_max_buffer_size = 1024 * 1024 * 1024; // 1G

        PDMAStream_Gaudi3();
        PDMAStream_Gaudi3(const PDMAStream_Gaudi3&) = delete;
        ~PDMAStream_Gaudi3();
        int setBuffer(const uint64_t deviceAddr, const unsigned size);
        int setPi(const unsigned pi);
        unsigned getPi() const { return m_localPi; }
        int configurePdmaPtrsMode();
        void init(const unsigned qid, const int fd);

    protected:
        gaudi3::block_pdma_ch_a *m_chBlock;
        void handlePiOverflowAsync(uint32_t hwPiRemainder);
        uint32_t handlePiOverflow(uint32_t hwPi);

    private:
        uint32_t m_allocatedSize = 0;
        unsigned m_localPi       = 0;
        unsigned m_hwPi          = 0;
        unsigned m_hwPiOffset    = 0;
        unsigned m_qid           = 0;
        bool m_asyncThreadActive = false;
        bool m_wraparound        = false;
        std::mutex m_wraparoundMutex;
        std::future<void> m_asyncThreadFuture;
    };

    class DirectModePDMAStream : public PDMAStream_Gaudi3, public StreamInterface
    {
    public:
        DirectModePDMAStream(std::string&   streamName,
                             unsigned       qmanId,
                             unsigned       fenceCounterIndex,
                             uint64_t       fenceCounterAddress,
                             unsigned       priority,
                             unsigned       id);

        DirectModePDMAStream(const DirectModePDMAStream&) = delete;
        ~DirectModePDMAStream() {};

        int initChannelCQ(uint64_t countersMmuAddress,
                          uint64_t isrRegister,
                          unsigned isrIndex);

        virtual int submit(const unsigned pi, const unsigned submission_alignment) override;

        virtual int setBuffer(Buffer *buffer) override;
        virtual int setPriority(unsigned priority) override { return SCAL_SUCCESS; }; // MR change to FAILURE but adjust a test that fails because of it
        virtual int getInfo(scal_stream_info_t& info) const override;
        virtual int getCcbBufferAlignment(unsigned& ccbBufferAlignment) const override { return SCAL_FAILURE; };

        const std::string& getName() { return m_name; };
        const unsigned     getPriority() {return m_priority; };

    private:
        std::string m_name;
        Buffer*     m_coreBuffer = nullptr;

        unsigned m_qmanId = 0;
        unsigned m_priority = 0;
        unsigned m_id       = 0;
        unsigned m_fenceCounterIndex = 0;
        uint64_t m_fenceCounterAddr = 0;

        bool     m_isCqInitialized = false;
    };

    class ConfigPdmaChannel
    {
    public:
        static constexpr unsigned c_command_buffer_size = 1024 * 1024; // 1 MB
        static constexpr unsigned c_fence_ctr_idx = 0;

        ConfigPdmaChannel(int fd) : m_fd(fd) {}
        ConfigPdmaChannel(const ConfigPdmaChannel & other) = delete;
        ~ConfigPdmaChannel() {deinit();}

        int init();
        void deinit();
        bool submit(const Qman::Workload &wkld);

        int configurePdmaPtrsMode();

        bool submitPdmaConfiguration(const Qman::Program& configurationProgram);

        unsigned                    m_smIdx                 = -1;
        unsigned                    m_monIdx                = -1;
        unsigned                    m_cqIdx                 = -1;
        unsigned                    m_isrIdx                = -1;
        unsigned                    m_qid                   = -1;
        PDMAStream_Gaudi3*          m_stream                = nullptr;

    private:
        void _configInitialization(Qman::Program& pdmaProg);

        const int                   m_fd;

        void *                      m_buff_ptr              = nullptr;
        uint64_t                    m_buff_dev_addr         = 0;
        unsigned                    m_pi                    = 0;
        volatile uint64_t *         m_ctr_ptr               = nullptr;
        uint64_t                    m_ctr_dev_addr          = 0;
        uint64_t                    m_ch_base               = 0;
        std::set<unsigned>          m_used_qids;
    };

    class DirectModePdmaChannel
    {
    public:
        DirectModePdmaChannel(std::string&       streamName,
                              std::string&       pdmaEngineName,
                              const std::string& completionGroupName,
                              Scal*              pScal,
                              unsigned           qid,
                              uint64_t           isrRegister,
                              unsigned           isrIndex,
                              unsigned           longSoSmIndex,
                              unsigned           longSoIndex,
                              uint64_t           fenceCounterAddress,
                              unsigned           cqIndex,
                              unsigned           priority,
                              unsigned           id);
        ~DirectModePdmaChannel() {};

        int initChannelCQ();

        DirectModePDMAStream*      getStream()          { return &m_stream; };
        DirectModeCompletionGroup* getCompletionGroup() { return &m_completionGroup; };

        const std::string&         getPdmaEngineName() { return m_pdmaEngineName; };

        void setCounterHost(volatile uint64_t* counterHost) { m_counterHost = counterHost;
                                                              m_completionGroup.pCounter = counterHost;};
        void setCounterMmuAddr(uint64_t counterMmuAddress)  { m_counterMmuAddress = counterMmuAddress; };

    private:
        static constexpr unsigned c_fence_ctr_idx = 0;

        DirectModePDMAStream      m_stream;
        DirectModeCompletionGroup m_completionGroup;

        volatile uint64_t* m_counterHost       = nullptr;
        uint64_t           m_counterMmuAddress = 0;

        std::string        m_pdmaEngineName;

        bool m_isInitialized = false;
    };

    struct FWImage
    {
        unsigned image_dccm_size = 0;
        unsigned image_hbm_size = 0;
        uint8_t dccm[c_scheduler_image_dccm_size] {};
        uint8_t hbm[c_image_hbm_size] {};
    };

    struct SyncObjectsSetGroupGaudi3 : public SyncObjectsSetGroup
    {
        struct SoSetResourcesPerHDCore
        {
            SyncObjectsPool *localSosPool = nullptr;
            MonitorsPool *localMonitorsPool = nullptr;
            SyncObjectsPool *localBarrierSosPool = nullptr;
            MonitorsPool *localBarrierMonitorsPool = nullptr;
        };
        std::array<SoSetResourcesPerHDCore, c_hdcores_nr> localSoSetResources;
    };

    typedef std::map<std::string, FWImage> ImageMap;

    struct Nic
    {
        std::string name;
        std::bitset<c_ports_count_per_nic> portsInNicMask;

        friend void from_json(const scaljson::json& json, Nic& nic)
        {
            json.at("name").get_to(nic.name);
            unsigned mask = 0b11;
            if (json.find("mask") != json.end())
            {
                json.at("mask").get_to(mask);
            }
            nic.portsInNicMask = mask;
        }
    };

    static uint64_t getCoreAuxOffset(const ArcCore * core);
    static uint64_t getCoreQmOffset();
    static inline bool coreType2VirtualSobIndexes(const CoreType coreType, std::vector<int8_t> &virtualSobIndexes);
    static int getSmInfo(int fd, unsigned smId, struct hlthunk_sync_manager_info* info);
    bool submitQmanWkld(const Qman::Workload &wkld) override;

    int init(const std::string & configFileName) override;
        int parseConfigFile(const std::string & configFileName) override;
            void parsePdmaChannel(const scaljson::json &json, const ConfigVersion &version);
            void parseBackwardCompatibleness(const scaljson::json &json) override;
            void parseMemorySettings(const scaljson::json &json, const ConfigVersion &version, MemoryGroups &groups) override;
                Pool * loadPoolByName(const scaljson::json & memElement, const std::string & pool_key, bool failIfNotExists, Pool * defaultValue);
                void parseSchedulerCores(const scaljson::json &schedulersJson, const ConfigVersion &version, const MemoryGroups &groups) override;
                void parseEngineCores(const scaljson::json &enginesJson, const ConfigVersion &version, const MemoryGroups &groups) override;
                void parseClusterQueues(const scaljson::json &queuesJson, const ConfigVersion &version, Cluster* cluster) override;
            void parseNics(const scaljson::json &json, const ConfigVersion &version);
            void parseUserPdmaChannels(const scaljson::json &json, const ConfigVersion &version);
                int createSingleDirectModePdmaChannel(PdmaChannelInfo const* pdmaChannelInfo, unsigned streamIndex, const std::string streamSetName,
                                                      bool isIsrEnabled, unsigned isrIdx, unsigned priority, unsigned longSoSmIndex,
                                                      unsigned longSoIndex, unsigned& cqIndex,
                                                      bool isTdrMode, const scaljson::json &json);
                void parseSyncManagers(const scaljson::json &syncManagersJson, const ConfigVersion &version) override;
                    void parseSosPools(const scaljson::json &sosPoolsJson, const ConfigVersion &version, const unsigned smID) override;
                    void parseMonitorPools(const scaljson::json &monitorPoolJson, const ConfigVersion &version, const unsigned smID) override;
                void parseSyncManagersCompletionQueues(const scaljson::json &syncManagerJson, const ConfigVersion &version) override;
                    void parseCompletionQueues(const scaljson::json &completionQueueJson, const ConfigVersion &version, const unsigned smID,
                                               SyncObjectsPool* longSoPool,
                                               SyncObjectsPool* sfgSosPool, MonitorsPool* sfgMonitorsPool, MonitorsPool* sfgCqMonitorsPool);
                        void parseTdrCompletionQueues(const scaljson::json & completionQueueJsonItem, unsigned numberOfInstances,
                                                      CompletionGroupInterface& cq) override;
                void parseHostFenceCounters(const scaljson::json &syncObjectJson, const ConfigVersion &version) override;
                void parseSoSets(const scaljson::json &soSetJson, const ConfigVersion &version) override;
            void validatePdmaChannels(const scaljson::json &json, const ConfigVersion &version);
        int initMemoryPools();
            int allocateHBMPoolMem(uint64_t size, uint64_t* handle, uint64_t* addr, uint64_t hint, bool shared = false, bool contiguous = false);
            int allocateHostPoolMem(uint64_t size, void** hostAddr, uint64_t* deviceAddr, uint64_t hintOffset);
        int mapLBWBlocks();
        int openConfigChannel();
            int isLocal(const ArcCore* core, bool & localMode);
        int initDirectModePdmaChannels();
            int initDirectModeSinglePdmaChannel(DirectModePdmaChannel& directModePdmaChannel, RegToVal& regToVal, Qman::Program& prog);
        int checkCanary();
        int configureSMs();
            int configureCQs();
                int configureTdrCq(Qman::Program & prog, CompletionGroupInterface &cg);
                static void configureCQ(Qman::Program &prog, const uint64_t smBase, const unsigned cqIdx, const uint64_t ctrAddr, const unsigned isrIdx);
            void configureMonitor(Qman::Program& prog, unsigned monIdx, uint64_t smBase, uint32_t configValue, uint64_t payloadAddress, uint32_t payloadData) override;
            static void configureOneMonitor(Qman::Program& prog, unsigned monIdx, uint64_t smBase, uint32_t configValue, uint64_t payloadAddress, uint32_t payloadData);
            void AddIncNextSyncObjectMonitor(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned soIdx);
            void AddSlaveXDccmQueueMonitor(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned slaveIndex);
            unsigned AddCompletionGroupSupportForHCL(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned soIdx);
            int configureMonitors();
                void configureTdrMon(Qman::Program & prog, const CompletionGroupInterface *cg);
                void configSfgSyncHierarchy(Qman::Program & prog, const CompletionGroup *cg);
                void configFenceMonitorCounterSMs(Qman::Program & prog, const CompletionGroup *cg);
            int configureLocalMonChain();
        int loadFWImage();
            int loadFWImagesFromFiles(ImageMap &images);
            int LoadFWHbm(const ImageMap &images);
            int LoadFWDccm(const ImageMap &images);
                static void genFwDccmLoadProgram(Qman::Program &prog, const uint8_t * dccmFWBuff, const unsigned dccmSize);
        int configSchedulers();
            struct AddMaskLogParams
            {
                std::string schedulerName;
                std::string clusterName;
                unsigned    groupIndex;
                unsigned    edupIndex;
            };
            int add32BitMask(RegToVal & regToVal, uint64_t edupBaseEng, uint32_t fenceEdupTrigger, uint64_t mask, AddMaskLogParams const & logParams);
            int allocSchedulerConfigs(const unsigned coreIdx, DeviceDmaBuffer &buff);
            int fillSchedulerConfigs(const unsigned coreIdx, DeviceDmaBuffer &buff, RegToVal &regToVal);
            int createCoreConfigQmanProgram(const unsigned coreIdx,DeviceDmaBuffer &buff, Qman::Program & program, RegToVal *regToVal = nullptr);
            int handleSarcCreditsToEdup(Qman::Program & program);
        int configEngines();
            int allocEngineConfigs(const G3ArcCore * arcCore, DeviceDmaBuffer &buff);
            int fillEngineConfigs(const G3ArcCore* arcCore, DeviceDmaBuffer &buff, int sfgBaseSobId = -1, bool sfgConfEnabled = false);
            int fillCmeEngineConfigs(const G3CmeCore * core, DeviceDmaBuffer &buff);
        int closeConfigChannel();
        int configurePdmaPtrsMode();
        int activateEngines();
        int configureStreams();

    int getCreditManagmentBaseIndices(unsigned& cmSobjBaseIndex,
                                      uint64_t& cmSobjBaseAddr,
                                      unsigned& cmMonBaseIndex,
                                      bool      isCompletionGroupCM) const;

    int configureCreditManagementMonitors(Qman::Program& prog,
                                           unsigned       sobjIndex,
                                           unsigned       monitorBaseIndex,
                                           unsigned       numOfSlaves,
                                           uint64_t       counterAddress,
                                           uint32_t       counterValue);

    uint32_t getDistributedCompletionGroupCreditManagmentCounterValue(uint32_t completionGroupIndex);

    uint32_t getCompletionGroupCreditManagmentCounterValue(uint32_t engineGroupType);

    static unsigned getSobjSmId(unsigned sobjIndex);
    static unsigned getMonitorSmId(unsigned monitorIndex);

    unsigned getSchedulerModeCQsCount() const override { return c_hdcores_nr * c_cq_ctrs_in_dcore; }

    virtual void addFencePacket(Qman::Program& program, unsigned id, uint8_t targetVal, unsigned decVal) override;
    virtual void enableHostFenceCounterIsr(CompletionGroup * cg, bool enableIsr) override;
    struct arc_fw_synapse_config_t                           m_arc_fw_synapse_config;

    ConfigPdmaChannel                                          m_configChannel;
    std::map<uint64_t, std::unique_ptr<DirectModePdmaChannel>> m_directModePdmaChannels;

    unsigned                                                 m_runningIsrIdx                     = 0;
    Pool*                                                    m_pdmaPool                          = nullptr;
    bool                                                     m_use_auto_fetcher                  = true;
    std::vector<G3NicCore*>                                  m_nicCores;
    std::set<unsigned>                                       m_hdcores;
};
