#pragma once
#include <stdint.h>

#include <list>
#include <set>
#include <unordered_map>

#include "gaudi2/asic_reg_structs/nic_mac_ch_regs.h"
#include "gaudi2/asic_reg_structs/nic_mac_rs_fec_regs.h"
#include "gaudi2/asic_reg_structs/nic_qpc_regs.h"
#include "gaudi2/asic_reg_structs/nic_rxb_core_regs.h"
#include "gaudi2/asic_reg_structs/nic_rxe_axuser_regs.h"
#include "gaudi2/asic_reg_structs/nic_rxe_regs.h"
#include "gaudi2/asic_reg_structs/nic_tmr_regs.h"
#include "gaudi2/asic_reg_structs/nic_txe_regs.h"
#include "gaudi2/asic_reg_structs/nic_txs_regs.h"
#include "gaudi2/asic_reg_structs/nic_umr_regs.h"
#include "gaudi2/asic_reg_structs/prt_mac_core_regs.h"
#include "gaudi2_prm.h"

#include "cbb.h"
#include "cluster_config.h"
#include "fs_common.h"
#include "fs_coral_regspace.h"
#include "fs_pcap.h"
#include "nic_vswitch.h"

enum execution_type_enum
{
    EXECUTION_TYPE_PACKET,
    EXECUTION_TYPE_WQE,
    EXECUTION_TYPE_RDV,
    EXECUTION_TYPE_OTHER,
};

enum qpce_rdv_peer_rx_op
{
    QPCE_RDV_PRX_OP_CI_UPDATE,
    QPCE_RDV_PRX_OP_PI_UPDATE,
};

enum
{
    ERROR_SYNDROME_REQ_RETRY_EXCEEDED = 1,
    ERROR_SYNDROME_QPC_NOT_VALID      = 0x41,
};

enum encapsulation_type_enum
{
    ENCAPSULATION_TYPE_NONE,
    ENCAPSULATION_TYPE_IPV4,
    ENCAPSULATION_TYPE_UDP,
};

struct FS_packetPointer
{
    struct Vswitch::fs_eth*  eth;
    struct Vswitch::fs_ipv4* ip;

    struct Vswitch::fs_udp* udpEncap;
    uint8_t*       encap;

    struct Vswitch::fs_udp*      udp;
    struct Vswitch::fs_bth*      bth;
    struct Vswitch::fs_reth*     reth;
    uint8_t*            sob;
    uint8_t*            tsd;
    uint8_t*            payload;
    uint16_t            payloadSize;
    struct Vswitch::fs_aeth*     aeth;
    struct Vswitch::fs_rndv_cih* cih;
};

class FS_Nic : public Cbb_Base
{
   public:
    static constexpr unsigned c_max_message_size            = UINT32_MAX;
    static constexpr unsigned c_max_packet_payload_size     = Vswitch::c_max_packet_payload_len;
    static constexpr unsigned c_num_nic_macros              = ClusterCfg::c_num_nic_macros;
    static constexpr unsigned c_num_logical_nics            = ClusterCfg::c_num_logical_nics;
    static constexpr unsigned c_max_num_mac_per_logical_nic = ClusterCfg::c_max_num_mac_per_logical_nic;
    static constexpr unsigned c_logical_nic_num_ports       = ClusterCfg::c_logical_nic_num_ports;
    static constexpr unsigned c_logical_nic_num_rings       = ClusterCfg::c_logical_nic_num_rings;
    static constexpr unsigned c_num_umr_pages               = ClusterCfg::c_num_umr_pages;
    static constexpr unsigned c_max_num_cq                  = 32;
    static constexpr unsigned c_min_ethernet_packet_size    = 64;
    static constexpr unsigned c_cl_size                     = 128;
    static constexpr unsigned c_rndv_data_size              = 0x10;
    static constexpr unsigned c_max_raw_unsupported_size    = 1536;
    static constexpr unsigned c_min_raw_unsupported_size    = 1500;

    FS_Nic(coral_module_name mod_name);
    virtual ~FS_Nic() override = default;

    class FS_LogicalNic : public Cbb_Base
    {
       public:
        FS_LogicalNic(coral_module_name mod_name);
        virtual ~FS_LogicalNic() override = default;

        void               setInstanceName(const std::string& str);
        const std::string& getInstanceName() const { return m_name; }

        bool     canWrite(uint32_t offset);
        void     wreg64(uint32_t offset, uint64_t value, unsigned dwe, unsigned prot);
        void     wait_idle();
        uint64_t getRegsBlock() const;
        void     connect_to_lbw(LBWHub* lbw, Specs* specs, uint32_t itr) override;
        void     tcl_update_register(Addr_t addr_offset, uint32_t& value, Lbw_Protection_e prot, bool write) override;
        virtual bool
                 tcl_update_register_fast(Addr_t addr_offset, uint32_t& value, Lbw_Protection_e prot, bool write) override;
        void     reset() override;
        bool     is_reg_acc_thrd_safe() override { return true; }

        enum interrupt_type
        {
            e_interrupt_type_tx_ring0      = 0,
            e_interrupt_type_tx_ring1      = 1,
            e_interrupt_type_tx_ring2      = 2,
            e_interrupt_type_tx_ring3      = 3,
            e_interrupt_type_rx_ring0      = 4,
            e_interrupt_type_rx_ring1      = 5,
            e_interrupt_type_rx_ring2      = 6,
            e_interrupt_type_rx_ring3      = 7,
            e_interrupt_type_rdma_qp_error = 8,
            e_interrupt_event_queue        = 9,
        };

        enum cq_interrupt_type
        {
            e_cq_interrupt_new_cqe = 0,
            e_cq_interrupt_overrun = 1,
        };

        enum tx_job_origin
        {
            e_tx_job_origin_qman,
            e_tx_job_origin_privilege,
            e_tx_job_origin_secured_db,
            e_tx_job_origin_unsecured_db,
            e_tx_job_origin_wtd,
            e_tx_job_origin_qpc_update,
            e_tx_job_origin_process_wqe,
        };

        struct TxReqestorJob
        {
            tx_job_origin origin;
            uint32_t qpn;
            uint32_t pi;
            int      trustLevel;
            bool     internalDoorbell;
        };

        struct TxResponderJob
        {
            uint32_t qpn;
        };

        struct CqTimerJob
        {
            time_t   startTime;
            time_t   timeout;
            uint32_t cqn;
        };

        class CqTimerQueue
        {
           public:
            CqTimerQueue() {}
            ~CqTimerQueue() {}

            void setCqTimer(FS_LogicalNic* logicalNic, uint16_t cqn, uint32_t timeout);
            void clearCqTimer(FS_LogicalNic* logicalNic, uint16_t cqn);
            bool waitCqTimeout(FS_LogicalNic* logicalNic, uint16_t& cqn);
            bool isListEmpty() { return m_list.empty(); }

           private:
            std::list<CqTimerJob>   m_list;
        };

        void setId(unsigned id) { m_nic_id = id; }
        void setConnectivity(FS_Nic* nic, Vswitch* vswitch, unsigned vswitchHandle);

        void registerBlock(LBWHub*         lbwHub,
                           uint64_t        qpcBaseAddr,
                           uint64_t        txeBaseAddr,
                           uint64_t        txsBaseAddr,
                           uint64_t        rxeBaseAddr,
                           uint64_t        rxeAxuserBaseAddr,
                           const uint64_t* umrBaseAddrArr,
                           const uint64_t* macChBaseAddrArr);

        Coral_RegSpace m_qpcRegSpace;
        Coral_RegSpace m_txeRegSpace;
        Coral_RegSpace m_txsRegSpace;
        Coral_RegSpace m_rxeRegSpace;
        Coral_RegSpace m_rxeAxuserRegSpace;
        Coral_RegSpace m_umrRegSpace[c_num_umr_pages];
        Coral_RegSpace m_macChRegSpace[c_max_num_mac_per_logical_nic];

        static uint64_t getPortMac(FS_LogicalNic* logicalNic, int port);
        Lbw_Protection_e getLbwProt(uint32_t prot);
        static void     generateInterrupt(FS_LogicalNic* logicalNic, interrupt_type interruptType);

        static void enqueueErrFifo(FS_LogicalNic* logicalNic, const Gaudi2PrmErrFifo_t* errFifo);
        static void enqueueCqe(FS_LogicalNic*      logicalNic,
                               uint32_t            qpn,
                               uint16_t            cqn,
                               const Gaudi2PrmCqe* cqe,
                               bool                cqOverrunEqEn = true);
        static void enqueueEqe(FS_LogicalNic* logicalNic, uint16_t eqn, const Gaudi2Eqe* eqe);

        bool     isLoopbackEnabled(unsigned port);
        unsigned getNicId() const { return m_nic_id; }

        static bool qpcUpdateRequestorDoorbell(FS_LogicalNic*                 logicalNic,
                                               Gaudi2PrmRequestorQPContext_t* reqQpc,
                                               uint32_t                       qpn,
                                               uint32_t                       doorbellProducerIndex,
                                               bool                           internalDoorbell);
        static bool qpcUpdateRequestorTx(FS_LogicalNic*                  logicalNic,
                                         Gaudi2PrmRequestorQPContext_t*  reqQpc,
                                         uint32_t                        qpn,
                                         uint32_t                        packetPsn,
                                         bool                            executionDone,
                                         bool                            ackreq,
                                         enum execution_type_enum        executionType,
                                         bool                            messageDone,
                                         const Gaudi2MultiStrideState_t* txMutiStrideState);
        static void qpcUpdateRequestorRx(FS_LogicalNic*                 logicalNic,
                                         Gaudi2PrmRequestorQPContext_t* reqQpc,
                                         uint32_t                       qpn,
                                         bool                           rollback,
                                         bool                           messageDone,
                                         bool                           nop = false,
                                         bool                           rdv = false);
        static void qpcUpdateRequestorRxPeerQpUpdate(FS_LogicalNic*           logicalNic,
                                                     uint32_t                 qpn,
                                                     enum qpce_rdv_peer_rx_op rxOp,
                                                     uint32_t                 index);
        static void qpcUpdateResponderRx(FS_LogicalNic*                 logicalNic,
                                         Gaudi2PrmResponderQPContext_t* resQpc,
                                         uint32_t                       qpn,
                                         int                            status,
                                         int                            packetAckReq,
                                         uint32_t                       rawWqSize);
        static void qpcUpdateResponderTx(FS_LogicalNic*                 logicalNic,
                                         Gaudi2PrmResponderQPContext_t* resQpc,
                                         uint32_t                       qpn,
                                         uint32_t                       packetPsn,
                                         int                            status);

        static void readTxDescriptor(FS_LogicalNic*                       logicalNic,
                                     const Gaudi2PrmRequestorQPContext_t* reqQpc,
                                     uint32_t                             qpn,
                                     uint32_t                             descIdx,
                                     uint8_t*                             desc,
                                     uint32_t                             descSize);
        static void readRxDescriptor(FS_LogicalNic*                       logicalNic,
                                     const Gaudi2PrmRequestorQPContext_t* reqQpc,
                                     uint32_t                             qpn,
                                     uint32_t                             descIdx,
                                     uint8_t*                             desc,
                                     uint32_t                             descSize);

        static void requestorWriteTx(FS_LogicalNic* logicalNic, const TxReqestorJob* job);
        static void requestorWriteRxAck(FS_LogicalNic*                 logicalNic,
                                        uint32_t                       qpn,
                                        Gaudi2PrmRequestorQPContext_t* reqQpc,
                                        const FS_packetPointer*        packetPointer);
        static void requestorWriteRxNack(FS_LogicalNic*                 logicalNic,
                                         uint32_t                       qpn,
                                         Gaudi2PrmRequestorQPContext_t* reqQpc,
                                         const FS_packetPointer*        packetPointer);
        static void responderWriteRx(FS_LogicalNic* logicalNic, uint32_t qpn, const FS_packetPointer* packetPointer);
        static void responderWriteTx(FS_LogicalNic* logicalNic, const TxResponderJob* job);

        static void handleRoceRxAck(FS_LogicalNic* logicalNic, uint32_t qpn, const FS_packetPointer* packetPointer);
        static void handleRoceRxPacket(FS_LogicalNic* logicalNic,
                                       const uint8_t* packetBuf,
                                       size_t         packetSize,
                                       uint32_t       decapsulationType,
                                       uint8_t        decapsulationSize);
        static void handleEthernetRxPacket(FS_LogicalNic* logicalNic,
                                           int            port,
                                           int            ring,
                                           const uint8_t* packetBuf,
                                           size_t         packetSize,
                                           uint64_t       packetMac);

        static void dumpReqQpc(FS_LogicalNic* logicalNic, const Gaudi2PrmRequestorQPContext_t* reqQpc);
        static void dumpResQpc(FS_LogicalNic* logicalNic, const Gaudi2PrmResponderQPContext_t* resQpc);

        static void readReqQpc(FS_LogicalNic* logicalNic, uint32_t qpn, Gaudi2PrmRequestorQPContext_t* reqQpc);
        static void writeReqQpc(FS_LogicalNic* logicalNic, uint32_t qpn, const Gaudi2PrmRequestorQPContext_t* reqQpc);
        static void readResQpc(FS_LogicalNic* logicalNic, uint32_t qpn, Gaudi2PrmResponderQPContext_t* resQpc);
        static void writeResQpc(FS_LogicalNic* logicalNic, uint32_t qpn, const Gaudi2PrmResponderQPContext_t* resQpc);
        static bool
                    packetChecksRaw(FS_LogicalNic* logicalNic, size_t packetSize, uint64_t packetMac, uint8_t logRawEntrySize);
        static bool qpcChecksRes(FS_LogicalNic*                       logicalNic,
                                 const Gaudi2PrmResponderQPContext_t* resQpc,
                                 const FS_packetPointer*              packetPointer,
                                 uint32_t                             qpn);
        static bool qpcChecksReq(FS_LogicalNic*                       logicalNic,
                                 const Gaudi2PrmRequestorQPContext_t* reqQpc,
                                 const FS_packetPointer*              packetPointer);
        static void
        wqeChecksRx(FS_LogicalNic* logicalNic, const Gaudi2PrmRequestorQPContext_t* reqQpc, const uint8_t* desc);
        static void
        wqeChecksTx(FS_LogicalNic* logicalNic, const Gaudi2PrmRequestorQPContext_t* reqQpc, const uint8_t* desc);

        static FS_Nic* getNic(const FS_LogicalNic* logicalNic) { return logicalNic->m_nic; }

       private:
        std::string                                                 m_name;
        std::unordered_map<uint32_t, Gaudi2PrmRequestorQPContext_t> m_reqQPCache;
        std::unordered_map<uint32_t, Gaudi2PrmResponderQPContext_t> m_resQPCache;
        unsigned                                                    m_nic_id;
        FS_Nic*                                                     m_nic;
        // 2 for each umr and 2 for secure/priv
        std::array<uint32_t, 2 * c_num_umr_pages + 2>               m_dbFifoCi;
        bool                                                        m_acceptPackets = false;

        Vswitch*    m_vswitch;
        int         m_vswitchHandle;

        CqTimerQueue m_cqTimerQueue;

        bool     m_cqArm[c_max_num_cq];
        uint32_t m_cqArmIndex[c_max_num_cq];

        std::array<std::set<uint32_t>, 4> m_prio_full_qps; // QPs that have reached threshold

        void        updateDbFifoCi(unsigned idx);
        static bool threadCqTimerMain(void* ctx);

        static uint32_t rreg32PrivateQpc(void* ctx, uint32_t offset, unsigned axuser);
        static bool     wreg32PrivateQpc(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool     wreg32PrivateTxe(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool     wreg32PrivateTxs(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool     wreg32PrivateRxe(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmrCommon(unsigned idx, void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr0(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr1(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr2(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr3(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr4(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr5(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr6(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr7(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr8(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr9(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr10(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr11(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr12(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr13(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        static bool wreg32PrivateUmr14(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
        friend class FS_Nic;
    };

    void               setInstanceName(const std::string& str);
    const std::string& getInstanceName() const { return m_name; }

    FS_Nic::FS_LogicalNic* getLogicalNic(unsigned idx) { return m_logical_nic_ptrs[idx]; }

    bool canWrite(uint32_t offset) { return true; }

    void wreg64(uint32_t offset, uint64_t value, unsigned dwe, unsigned prot) {}

    void wait_idle() {}

    uint64_t getRegsBlock() const { return 0; }

    void setId(unsigned id) { m_nic_id = id; }

    void setConnectivity(Vswitch* vswitch);

    void registerBlock(LBWHub*         lbwHub,
                       uint64_t        prtMacCoreAddr,
                       uint64_t        macRsFecAddr,
                       uint64_t        rxbBaseAddr,
                       uint64_t        tmrBaseAddr,
                       uint64_t        phyBaseAddr,
                       uint64_t        mstrIfBaseAddr,
                       const uint64_t* qpcBaseAddrArr,
                       const uint64_t* txeBaseAddrArr,
                       const uint64_t* txsBaseAddrArr,
                       const uint64_t* rxeBaseAddrArr,
                       const uint64_t* rxeAxuserBaseAddrArr,
                       const uint64_t* umrBaseAddrArr,
                       const uint64_t* macChBaseAddrArr);
    void connect_to_lbw(LBWHub* lbw, Specs* specs, uint32_t itr) override;
    void tcl_update_register(Addr_t addr_offset, uint32_t& value, Lbw_Protection_e prot, bool write) override;
    void cycle() override;
    void reset() override;
    void startCycle();

    struct TxJob
    {
        bool     isReq;
        unsigned nicIdx;
        union
        {
            struct FS_LogicalNic::TxReqestorJob  req;
            struct FS_LogicalNic::TxResponderJob res;
        };
    };

    struct CqWaitEntry
    {
        FS_LogicalNic* logicalNic;
        uint32_t       qpn;
        uint16_t       cqn;
        Gaudi2PrmCqe   cqe;
    };

    Coral_RegSpace m_rxbRegSpace;
    Coral_RegSpace m_mstrIfRegSpace;

   private:
    std::string m_name;
    uint8_t     m_txPacketBuf[Vswitch::c_max_packet_len];
    uint8_t     m_rxPacketBuf[Vswitch::c_max_packet_len];
    unsigned    m_nic_id;

    std::list<FS_LogicalNic>    m_logical_nics;
    std::vector<FS_LogicalNic*> m_logical_nic_ptrs;


    std::deque<TxJob>                   m_txWorkQueue;
    std::deque<TxJob>                   m_txAckWorkQueue;
    std::deque<Vswitch::DistrubutedPacket> m_vswitchQueue;
    std::deque<CqWaitEntry>             m_cqWaitQueue;

    Coral_RegSpace m_prtMacCoreRegSpace;
    Coral_RegSpace m_macRsFecRegSpace;
    Coral_RegSpace m_tmrRegSpace;
    Coral_RegSpace m_phyRegSpace;

    Vswitch*    m_vswitch;
    int         m_vswitchHandle;

    FS_Pcap m_pcap;

    static bool threadTxMain(void* ctx);
    static bool threadRxMain(void* ctx);

    static bool                     wreg32PrivateRxb(void* ctx, uint32_t offset, uint32_t value, unsigned axuser);
    static enum execution_type_enum processTxDescriptor(FS_Nic*                        nic,
                                                        unsigned                       nicIdx,
                                                        uint32_t                       qpn,
                                                        Gaudi2PrmRequestorQPContext_t* reqQpc,
                                                        const uint8_t*                 desc,
                                                        bool                           ackreq,
                                                        bool                           messageDone,
                                                        bool                           firstPacketInMessage,
                                                        Vswitch::Packet*               packet,
                                                        Gaudi2MultiStrideState_t*      txMultiStrideState);
    static void                     writeRxPacketData(FS_Nic*                              nic,
                                                      unsigned                             nicIdx,
                                                      const Gaudi2PrmResponderQPContext_t* resQpc,
                                                      uint32_t                             qpn,
                                                      uint16_t                             cqn,
                                                      const FS_packetPointer*              packetPointer);

    static void     readPayloadData(FS_Nic*                        nic,
                                    unsigned                       nicIdx,
                                    uint32_t                       qpn,
                                    Gaudi2PrmRequestorQPContext_t* reqQpc,
                                    const uint8_t*                 desc,
                                    uint32_t                       descSize,
                                    uint8_t*                       payload,
                                    uint16_t&                      payloadSize,
                                    Gaudi2MultiStrideState_t*      packetMultiStrideState,
                                    Gaudi2MultiStrideState_t*      txMultiStrideState);
    static uint16_t buildReqPacket(FS_Nic*                        nic,
                                   unsigned                       nicIdx,
                                   uint32_t                       qpn,
                                   Gaudi2PrmRequestorQPContext_t* reqQpc,
                                   const uint8_t*                 desc,
                                   bool                           ackreq,
                                   bool                           messageDone,
                                   bool                           se,
                                   FS_packetPointer*              packetPointer,
                                   Gaudi2MultiStrideState_t*      txMultiStrideState);
    static uint16_t buildResPacket(FS_Nic*                        nic,
                                   unsigned                       nicIdx,
                                   Gaudi2PrmResponderQPContext_t* resQpc,
                                   uint32_t                       qpn,
                                   uint32_t                       packetPsn,
                                   FS_packetPointer*              packetPointer);
    void            push2TxWq(TxJob& job, FS_LogicalNic* lnic);
    void            push2TxAckWq(TxJob& job);
};
