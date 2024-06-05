#pragma once
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <condition_variable>
#include <list>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "gaudi2/asic_reg_structs/mstr_if_regs.h"
#include "gaudi2/mme.h"

#include "cbb.h"
#include "cluster_config.h"
#include "derater.h"
#include "event.hpp"
#include "fs_common.h"
#include "fs_coral_regspace.h"
#include "fs_mme_init.h"
#include "lbw_hub.h"
#include "specs.h"

#include "fs_mme_accap.h"
#include "fs_mme_accap_brain.h"
#include "fs_mme_agu.h"
#include "fs_mme_comp_msg.h"
#include "fs_mme_condition_variable.h"
#include "fs_mme_debug.h"
#include "fs_mme_dec.h"
#include "fs_mme_eu_brain.h"
#include "fs_mme_eu_fp.h"
#include "fs_mme_eus.h"
#include "fs_mme_md.h"
#include "fs_mme_queue.h"
#include "fs_mme_sb.h"
#include "fs_mme_sbte.h"
#include "fs_mme_te.h"
#include "fs_mme_utils.h"
#include "fs_mme_wbc.h"

#include "mme_half.h"

#define NUM_PORTS_PER_FS_MME 4 // 4 ports per MME, 4 such MMEs.So there are 4 fs MMEs each with 4 hbw ports
#define EVENT_BITMASK_SIZE 2 // Perf trace event mask has 2 bits
#define START_EVENT_BIT 0 // Bit 0 in the mask represents start event indication
#define END_EVENT_BIT 1 // Bit 1 in the mask represents end event indication

class FS_Mme : public Cbb_Base
{
    enum MmeRegisterRanges
    {
        CTRL  = 0xCB000,
        ACC   = 0xF8000,
        SBTE0 = 0xD0000,
        SBTE1 = 0xD8000,
        SBTE2 = 0xDE000,
        SBTE3 = 0xE8000
    };

   public:
    static const unsigned c_mme_num             = ClusterCfg::c_mme_num;
    static const unsigned c_mme_desc_nr         = 4;
    static const unsigned c_mme_brains_nr       = 9;
    static const unsigned c_mme_sbte_nr         = 5;
    static const unsigned c_comp_if_queue_depth = 128; // TODO: what's the correct value
    static const unsigned c_wbc_num             = 2;

    FS_Mme(coral_module_name name, Gaudi2_MME_Half*, Derater*);
    virtual ~FS_Mme() override;

    void               setInstanceName(const std::string& str);
    const std::string& getInstanceName() const { return m_name; }
    unsigned           getInstanceID() const { return m_id; }
    void               setDumpDir(const std::string& dumpPath) { m_dumpPath = dumpPath; }
    void               setDebugMemAccess(MmeCommon::MemAccess* memAccess);
    const std::string  getDumpDir() const { return m_dumpPath; }

    Gaudi2::Mme::LbwWrMaster* m_lbwWrMstr = nullptr;

    bool canWrite(uint32_t offset);
    void check_register_for_protection(Addr_t addr, Lbw_Protection_e prot);
    void
             tcl_update_register_from_qman(Addr_t offset, uint32_t& value, Lbw_Protection_e prot, bool write) override final;
    void     wait_idle();
    uint64_t getRegsBlock() const;

    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Dec2Eus>* getRemoteSB(unsigned idx)
    {
        return idx ? m_sbRemoteOut1_Q : m_sbRemoteOut0_Q;
    }

    void setConnectivity(unsigned id, FS_Mme* remote_mme);

    void launch(bool launch_after_reset);

    void registerBlock(LBWHub*  lbw,
                       uint64_t baseAddressCtrlLo,
                       uint64_t baseAddressCtrlHi,
                       uint64_t baseAddressCtrlMstrIf,
                       uint64_t baseAddressAcc,
                       uint64_t baseAddressWb0MstrIf,
                       uint64_t baseAddressWb1MstrIf,
                       uint64_t baseAddressSbte0,
                       uint64_t baseAddressSbte1,
                       uint64_t baseAddressSbte2,
                       uint64_t baseAddressSbte3,
                       uint64_t baseAddressSbte4,
                       uint64_t baseAddressSbte0mstrIf,
                       uint64_t baseAddressSbte1mstrIf,
                       uint64_t baseAddressSbte2mstrIf,
                       uint64_t baseAddressSbte3mstrIf,
                       uint64_t baseAddressSbte4mstrIf);

    void hbwWriteAccess(std::string& name, uint64_t address, uint64_t size, uint32_t usr, const uint8_t* data);

    void hbwReadAccess(std::string&                        name,
                       const Gaudi2::Mme::EMmeInputOperand operand,
                       uint64_t                            address,
                       uint64_t                            size,
                       uint32_t                            usr,
                       uint8_t*                            data);

    void genPCLoops(const Gaudi2::Mme::Desc* desc, unsigned mask, std::list<unsigned>* loops);
    bool waitForDescriptor(uint8_t id, Gaudi2::Mme::EMmeBrainIdx brainIdx);
    void clearStatusBit(uint8_t id, Gaudi2::Mme::EMmeBrainIdx brainIdx);
    void getDescriptor(uint8_t id, Gaudi2::Mme::Desc** desc);
    void tcl_update_register(Addr_t addr_offset, uint32_t& value, Lbw_Protection_e prot, bool write) override final;
    void connect_to_lbw(LBWHub* lbw, Specs* specs, uint32_t itr) override;

    unsigned getBrainCmdCtr(const unsigned descIdx, const Gaudi2::Mme::EMmeBrainIdx brainIdx)
    {
        return m_brainCmdCtr[brainIdx][descIdx % c_mme_desc_nr];
    }
    uint32_t    getAxUser(const Gaudi2::Mme::EMmeBrainIdx brainIdx, const uint32_t core_user = 0);
    inline bool isMaster() const { return m_isMaster; }
    inline bool isCoreSeparationMode() const { return m_coreSeparationMode; }
    inline bool is2MastersMode() const { return false; }
    void        lbwWrite32(uint64_t addr, uint32_t data)
    {
        const std::lock_guard<std::mutex> lock(m_lbwLock);
        m_lbwData.emplace_back(std::make_pair(addr, data));
        m_lbwSize++;
    }
    void cycle() final;
    void send_so(uint64_t num_sos, bool drain_q = false, double end_time = 0);
    void post_end_event(unsigned);
    void do_register_reset();
    void do_soft_reset();
    void delete_queues(bool delete_sbte_and_accap);
    void create_queues(bool no_sbte_and_accp);
    void halt_mme();

    // ------------------- Transactionless ------------------- //
    void set_start_timestamp(double timestamp) override;
    void set_port_id(uint64_t port_id);

    int32_t eu_cycles         = 0;
    int32_t tensor_vol[7]     = {0}; // 5 SBs + 2 WBs
    bool    tensor_in_sram[7] = {false}; // 5 SBs + 2 WBs
    double  start_timestamp;
    double  end_timestamp;
    // ------------------------------------------------------ //

    // Coral perf model
    Gaudi2_MME_Half*         m_perf_mme;
    std::mutex               m_perfQLock;
    bool                     m_computeStart{false};
    bool                     m_null_command{false};
    bool                     m_halted{false};
    Gaudi2_MME_Half::PerfEvt m_perf_evt_in;
    Gaudi2_MME_Half::PerfEvt m_perf_evt_out;
    std::string              m_perf_evt_operation;

    Gaudi2::Mme::FS_SbTe* m_sbte[c_mme_sbte_nr];
    Gaudi2::Mme::AccAp*   m_accAp;
    Gaudi2::Mme::EUBrain* m_euBrain;
    Gaudi2::Mme::EUS*     m_eus;
    const coral_knob::Knob<bool> knob_fp32_ieee;

   private:
    bool wreg32Private(uint32_t offset, uint32_t value);
    bool setLfsrSeedPrivate(uint32_t value);
    bool getLfsrSeedPrivate();
    bool setLfsrPolynomPrivate(uint32_t value);
    void execute(uint32_t value);
    void set_busy();
    void set_idle();

    bool is_start_event_enabled(Gaudi2::Mme::MmePerfEvt desc);
    bool is_end_event_enabled(Gaudi2::Mme::MmePerfEvt desc);

    // ------------------- Transactionless ------------------- //
    struct TxnlessCmd
    {
        Gaudi2::Mme::Desc desc{};
        uint64_t          job_id            = -1;
        uint64_t          event_id          = -1;
        double            start_time        = 0;
        double            end_time          = 0;
        int32_t           derates_requested = 0;
        int32_t           derates_received  = 0;
        bool              is_null           = false;

        bool is_derate_done() { return (derates_received == derates_requested); }

        template <typename OStream>
        friend OStream& operator<<(OStream& ost, const TxnlessCmd cmd)
        {
            ost << " job = " << cmd.job_id << (cmd.is_null ? "(null)" : "") << ", context = " << cmd.event_id
                << ", derates_done = " << cmd.derates_received << "/" << cmd.derates_requested
                << ", start_time = " << cmd.start_time << " us, end_time = " << cmd.end_time << "us ";
            return ost;
        }

        friend std::ostream& operator<<(std::ostream& ost, const TxnlessCmd cmd)
        {
            ost << " job = " << cmd.job_id << (cmd.is_null ? "(null)" : "") << ", context = " << cmd.event_id
                << ", derates_done = " << cmd.derates_received << "/" << cmd.derates_requested
                << ", start_time = " << cmd.start_time << " us, end_time = " << cmd.end_time << "us ";
            return ost;
        }
    };

    void trigger_txnless(uint32_t value);
    void execute_txnless();
    void send_desc_to_derater(TxnlessCmd& cmd);
    void txnless_derate_done(Derater::TransferDesc xfer);
    void send_so_txnless(TxnlessCmd& cmd);
    void finish_txnless(TxnlessCmd& cmd);
    void post_txnless_begin(TxnlessCmd& cmd);
    void post_txnless_end(TxnlessCmd& cmd);

    bool                   m_cmd_in_progress = false;
    uint64_t               m_trigger_count   = 0;
    Derater*               m_derater         = nullptr;
    uint64_t               m_derater_id      = -1;
    std::vector<uint64_t>  m_ports;
    std::deque<TxnlessCmd> m_txnless_cmd_q;

    coral_infra::Event<EVENT_NAME(mme_txnless_active)> m_mme_txnless_active_evt{"MME transaction-less mode active",
                                                                                (coral::Module*)this,
                                                                                "MME"};
    // ------------------------------------------------------- //

    unsigned          m_id;
    FS_Mme*           m_remote;
    std::string       m_name;
    coral_module_name c_name;
    bool              m_terminate;
    uint32_t          event_id = 0;

    // debug infra
    std::string m_dumpPath;

    Coral_RegSpace                 m_ctrlLoRegSpace;
    Coral_RegSpace                 m_ctrlHiRegSpace;
    Coral_RegSpace                 m_ctrlMstrIfRegSpace;
    Coral_RegSpace                 m_accRegSpace;
    Coral_RegSpace                 m_wb0MstrIfRegSpace;
    Coral_RegSpace                 m_wb1MstrIfRegSpace;
    Coral_RegSpace                 m_sbte0RegSpace;
    Coral_RegSpace                 m_sbte1RegSpace;
    Coral_RegSpace                 m_sbte2RegSpace;
    Coral_RegSpace                 m_sbte3RegSpace;
    Coral_RegSpace                 m_sbte4RegSpace;
    Coral_RegSpace                 m_sbte0mstrIfRegSpace;
    Coral_RegSpace                 m_sbte1mstrIfRegSpace;
    Coral_RegSpace                 m_sbte2mstrIfRegSpace;
    Coral_RegSpace                 m_sbte3mstrIfRegSpace;
    Coral_RegSpace                 m_sbte4mstrIfRegSpace;
    unsigned                       m_descCtr;
    unsigned                       m_cmdCtr;
    std::mutex                     m_statusMutex;
    Gaudi2::Mme::ConditionVariable m_execCond[c_mme_brains_nr];
    unsigned                       m_brainCmdCtr[c_mme_brains_nr][c_mme_desc_nr];

    bool m_isMaster;
    bool m_coreSeparationMode;

    // SBTEs
    // Gaudi2::Mme::FS_SbTe*                         m_sbte[c_mme_sbte_nr];
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_sbte2eus_Q[c_mme_sbte_nr];

    // remote SB (out)
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_sbRemoteOut0_Q;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_sbRemoteOut1_Q;

    // WBC-Cout channel
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Agu2Sb>*     m_agu2sbCout0_Q;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Agu2Sb>*     m_agu2sbCout1_Q;
    Gaudi2::Mme::AGU*                               m_aguCout0;
    Gaudi2::Mme::AGU*                               m_aguCout1;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Ap2Wbc>*     m_ap2wbc0_Q;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Ap2Wbc>*     m_ap2wbc1_Q;
    Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* m_wbc0ToCompMsg0_Q;
    Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* m_wbc1ToCompMsg0_Q;
    Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* m_wbc0ToCompMsg1_Q;
    Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* m_wbc1ToCompMsg1_Q;
    Gaudi2::Mme::Queue<bool>*                       m_remoteCompMsg0_Q;
    Gaudi2::Mme::Queue<bool>*                       m_remoteCompMsg1_Q;
    Gaudi2::Mme::WBC*                               m_wbc0;
    Gaudi2::Mme::WBC*                               m_wbc1;
    Gaudi2::Mme::CompMsg*                           m_compMsg0;
    Gaudi2::Mme::CompMsg*                           m_compMsg1;

    // ACC + Activation Pipe
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_AccApBrain>* m_accApBrain_Q;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Eus2Acc>*    m_eus2acc_Q;
    // Gaudi2::Mme::AccAp*                             m_accAp;
    Gaudi2::Mme::AccApBrain*                        m_accApBrain;

    // EUS + EU
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_EusBrain>* m_eusBrain_Q;
    Gaudi2::Mme::EU_fp*                           m_euFp;
    std::mutex                                    m_lbwLock;
    std::atomic<std::uint64_t>                    m_lbwSize{0};
    std::list<std::pair<uint64_t, uint32_t>>      m_lbwData;
    std::unordered_map<unsigned, unsigned>        m_endCmdCtrs;
    std::mutex                                    m_endCmdCtrsLock;

    coral_infra::Event<EVENT_NAME(mme_func_active)> m_mme_func_active_evt{"MME func active",
                                                                          (coral::Module*)this,
                                                                          "MMEFUNC"};

#ifdef MME_DEADLOCK_DEBUG
    // Debug instance
    Gaudi2::Mme::Debug* m_debug;
#endif
};
