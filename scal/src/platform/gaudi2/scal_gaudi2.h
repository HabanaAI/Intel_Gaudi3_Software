#pragma once
#include <string>
#include <vector>
#include <map>

#include "scal_base.h"
#include "json.hpp"
#include "gaudi2/asic_reg/gaudi2_blocks.h"
#include "gaudi2_arc_host_packets.h"

class Scal_Gaudi2 : public Scal
{
public:
    enum DupTrigger
    {
        dupTriggerTPC       = 0,
        dupTriggerMME       = 1,
        dupTriggerEDMA      = 2,
        dupTriggerPDMA      = 3,
        dupTriggerROT       = 4,
        dupTriggerRSRVD     = 5,
        dupTriggerNicPri0_I = 6,
        dupTriggerNicPri1_I = 7,
        dupTriggerNicPri2_I = 8,
        dupTriggerNicPri3_I = 9,
        dupTriggerNicPri0_E = 10,
        dupTriggerNicPri1_E = 11,
        dupTriggerNicPri2_E = 12,
        dupTriggerNicPri3_E = 13
    };

    Scal_Gaudi2(
            const int fd,
            const struct hlthunk_hw_ip_info & hw_ip,
            scal_arc_fw_config_handle_t fwCfg);

    virtual ~Scal_Gaudi2();

    virtual const char* getDefaultJsonName() const override { return "gaudi2/default.json"; }
    uint32_t getSRAMSize() const override;
    int setStreamPriority(Stream *stream, unsigned priority) const override;
    int setStreamBuffer(Stream *stream, Buffer *buffer) const override;
    int getNumberOfSignalsPerMme() const override;
    int streamSubmit(Stream *stream, const unsigned pi, const unsigned submission_alignment) override;
    unsigned getSchedulerModeCQsCount() const override { return c_sync_managers_nr * c_cq_ctrs_in_dcore; }
    unsigned getUsedSmBaseAddrs(const scal_sm_base_addr_tuple_t ** smBaseAddrDb) override { return 0; }

protected:
    static constexpr unsigned c_cores_nr = CPU_ID_MAX;
    static constexpr unsigned c_scheduler_nr = CPU_ID_SCHED_MAX;
    static constexpr unsigned c_arc_farm_half_dccm_size = 32 * 1024;
    static constexpr unsigned c_hbm_bin_buffer_size = c_cores_nr * c_image_hbm_size;
    static constexpr uint64_t c_local_address = 0x1000007FFCA00000ull;
    static constexpr uint64_t c_local_range_size = 0x10000;  //64 Kb
    static constexpr uint64_t c_local_address_mask = c_local_range_size - 1;
    static constexpr unsigned c_dccm_to_qm_offset = mmDCORE1_TPC0_QM_BASE & c_local_address_mask;
    static constexpr unsigned c_cps_nr = 5;
    static constexpr unsigned c_message_short_base_index =3;
    static constexpr unsigned c_sync_managers_nr = 4;
    static constexpr unsigned c_max_monitors_per_sync_manager = 2048;
    static constexpr uint32_t c_arc_acc_engs_virtual_addr = 0x4A00000;
    static constexpr uint32_t c_local_dup_offset = 0x9000;
    static constexpr unsigned c_maximum_messages_per_monitor = 4;
    static constexpr unsigned c_ports_count_per_nic = 1;
    static constexpr unsigned c_nics_count = 24;
    static constexpr unsigned c_ports_count = c_ports_count_per_nic * c_nics_count;
    static constexpr unsigned c_dcores_nr = 4;

    static_assert(Scal_Gaudi2::c_dcores_nr * c_cq_ctrs_in_dcore <= c_max_cq_cntrs);

    static constexpr char c_config_key_memory_tpc_barrier_in_sram[] = "tpc_barrier_in_sram";
    static constexpr char c_config_key_cores_engine_clusters_queues_scheduler_bit_mask_offset[] = "bit_mask_offset";
    static constexpr char c_config_key_sync_managers_dcore[] = "dcore";

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

    struct FWImage
    {
        unsigned dccmChunksNr = 0;
        unsigned image_dccm_size = 0;
        unsigned image_hbm_size = 0;
        uint8_t dccm[2][c_engine_image_dccm_size] {};
        uint8_t hbm[c_image_hbm_size] {};
    };

    typedef std::map<std::string, FWImage> ImageMap;

    static uint64_t getCoreAuxOffset(const ArcCore * core);
    static inline bool coreType2VirtualSobIndex(const CoreType coreType, unsigned& virtualSobIndex);

    bool submitQmanWkld(const Qman::Workload &wkld) override;

    int init(const std::string & configFileName) override;
        int parseConfigFile(const std::string & configFileName) override;
            void parseBackwardCompatibleness(const scaljson::json &json) override;
            void parseMemorySettings(const scaljson::json &json, const ConfigVersion &version, MemoryGroups &groups) override;
                void parseSchedulerCores(const scaljson::json &schedulersJson, const ConfigVersion &version, const MemoryGroups &groups) override;
                void parseEngineCores(const scaljson::json &enginesJson, const ConfigVersion &version, const MemoryGroups &groups) override;
                void parseClusterQueues(const scaljson::json &queuesJson, const ConfigVersion &version, Cluster* cluster) override;
                void parseSyncManagers(const scaljson::json &syncManagersJson, const ConfigVersion &version) override;
                    void parseSosPools(const scaljson::json &sosPoolsJson, const ConfigVersion &version, const unsigned smID) override;
                    void parseMonitorPools(const scaljson::json &monitorPoolJson, const ConfigVersion &version, const unsigned smID) override;
                void parseSyncManagersCompletionQueues(const scaljson::json &syncManagerJson, const ConfigVersion &version) override;

                    void parseCompletionQueues(const scaljson::json &completionQueueJson, const ConfigVersion &version, const unsigned smID,
                                                   SyncObjectsPool* longSoPool,
                                                   SyncObjectsPool* sfgSosPool = nullptr, MonitorsPool* sfgMonitorsPool = nullptr,
                                                   MonitorsPool* sfgCqMonitorsPool = nullptr);

                        void parseTdrCompletionQueues(const scaljson::json & completionQueueJsonItem, unsigned numberOfInstances,
                                                      CompletionGroupInterface& cq) override;
                void parseHostFenceCounters(const scaljson::json &syncObjectJson, const ConfigVersion &version) override;
                void parseSoSets(const scaljson::json &soSetJson, const ConfigVersion &version) override;

        int initMemoryPools();
            int allocateHBMPoolMem(uint64_t size, uint64_t* handle, uint64_t* addr, uint64_t hint, bool shared = false, bool contiguous = false);
            int allocateHostPoolMem(uint64_t size, void** hostAddr, uint64_t* deviceAddr, uint64_t hintOffset);
        int warmupHbmTlb();
        int mapLBWBlocks();
        int resetQMANs();
            int isLocal(const ArcCore* arcCore, bool & localMode);
        int checkCanary();
        int configureSMs();
            int configureCQs();
                int configureTdrCq(Qman::Program & prog, CompletionGroup &cg);
            void configureMonitor(Qman::Program& prog, unsigned monIdx, uint64_t smBase, uint32_t configValue, uint64_t payloadAddress, uint32_t payloadData) override;
            static void configureOneMonitor(Qman::Program& prog, unsigned monIdx, uint64_t smBase, uint32_t configValue, uint64_t payloadAddress, uint32_t payloadData);
            void AddIncNextSyncObjectMonitor(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned soIdx, unsigned numWrites);
            void AddSlaveXDccmQueueMonitor(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned slaveIndex, unsigned numWrites);
            void AddTriggerDemiMasterMonitor(Qman::Program& prog, uint64_t smBase, unsigned monIdx, unsigned soIdx, unsigned monIdxToTrigger);
            unsigned AddCompletionGroupSupportForHCL(Qman::Program& prog, CompletionGroup* cq, uint64_t smBase, unsigned monIdx, unsigned soIdx);
            int configureMonitors();
                void configureTdrMon(Qman::Program & prog, const CompletionGroup *cg);
                void configSfgSyncHierarchy(Qman::Program & prog, const CompletionGroup *cg);
                void configFenceMonitorCounterSMs(Qman::Program & prog, const CompletionGroup *cg);
        int loadFWImage();
            int loadFWImagesFromFiles(ImageMap &images);
            int LoadFWHbm(const ImageMap &images);
            int LoadFWDccm(const ImageMap &images);
                static void genLowerCpFWLoad(Qman::Program &prog, const uint8_t * dccmFWBuff);
        int loadTPCNopKernel();
        int configSchedulers();
            int allocSchedulerConfigs(const unsigned coreIdx, DeviceDmaBuffer &buff);
            int fillSchedulerConfigs(const unsigned coreIdx, DeviceDmaBuffer &buff, RegToVal &regToVal);
            int createCoreConfigQmanProgram(const unsigned coreIdx,DeviceDmaBuffer &buff, const bool upperCP, Qman::Program & program, RegToVal *regToVal = nullptr);
        int configEngines();
            int allocEngineConfigs(const ArcCore * arcCore, DeviceDmaBuffer &buff);
            int fillEngineConfigs(const ArcCore* arcCore, DeviceDmaBuffer &buff, int sfgBaseSobId = -1, bool sfgConfEnabled = false);
        int activateEngines();
        int configureStreams();

    int haltCores();

    int getCreditManagmentBaseIndices(unsigned& cmSobjBaseIndex,
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

    static unsigned getSobjDcoreId(unsigned sobjIndex);
    static unsigned getMonitorDcoreId(unsigned monitorIndex);

    virtual void addFencePacket(Qman::Program& program, unsigned id, uint8_t targetVal, unsigned decVal) override;
    virtual void enableHostFenceCounterIsr(CompletionGroup * cg, bool enableIsr) override;

    bool                                                     m_tpcBarrierInSram                  = false;
    uint64_t                                                 m_tpcNopKernelDeviceAddr            = 0ULL;
    DeviceDmaBuffer*                                         m_dmaBuff4tpcNopKernel              = nullptr;
    struct arc_fw_synapse_config_t                           m_arc_fw_synapse_config_t;
};
