#ifndef GAUDI2_PRM_H
#define GAUDI2_PRM_H

#include <stdint.h>

#pragma pack(push, 1)

#define GAUDI2_PRM_WQE_CT_SO_UPDATE 0x1
#define GAUDI2_PRM_WQE_CT_CQ_UPDATE 0x2

enum gaudi2PrmOpcodeEnum
{
    GAUDI2_PRM_OPCODE_NOP                           = 0x0,
    GAUDI2_PRM_OPCODE_SEND                          = 0x1,
    GAUDI2_PRM_OPCODE_LINEAR_WRITE                  = 0x2,
    GAUDI2_PRM_OPCODE_LOCAL_MULTI_STRIDE_WRITE      = 0x3,
    GAUDI2_PRM_OPCODE_MULTI_STRIDE_WRITE_SINGLE_STD = 0x4,
    GAUDI2_PRM_OPCODE_RENDEZVOUS_WRITE              = 0x5,
    GAUDI2_PRM_OPCODE_RENDEZVOUS_READ               = 0x6,
    GAUDI2_PRM_OPCODE_QOS_UPDATE                    = 0x7,
    GAUDI2_PRM_OPCODE_MULTI_STRIDE_DUAL             = 0x8,
    GAUDI2_PRM_OPCODE_ATOMIC_FETCH_AND_ADD_WRITE    = 0x9,
};

enum gaudi2PrmBthOpcodeEnum
{
    GAUDI2_PRM_BTH_OPCODE_SEND                      = 0x4,
    GAUDI2_PRM_BTH_OPCODE_LINEAR_WRITE              = 0xA,
    GAUDI2_PRM_BTH_OPCODE_ACK                       = 0x11,
    GAUDI2_PRM_BTH_OPCODE_GAUDI1_MULTI_STRIDE_WRITE = 0x19,
    GAUDI2_PRM_BTH_OPCODE_GAUDI2_MULTI_STRIDE_WRITE = 0x1A,
    GAUDI2_PRM_BTH_OPCODE_WRITE_RDV_RECEIVE         = 0x1B,
    GAUDI2_PRM_BTH_OPCODE_WRITE_RDV_READ            = 0x1C,
};

enum
{
    GAUDI2_PRM_NAK_SYNDROME_PSN_SEQUENCE_ERROR        = 0x0,
    GAUDI2_PRM_NAK_SYNDROME_INVALID_REQUEST           = 0x1,
    GAUDI2_PRM_NAK_SYNDROME_REMOTE_ACCESS_ERROR       = 0x2,
    GAUDI2_PRM_NAK_SYNDROME_REMOTE_OPERATIONAL_ERROR  = 0x3,
    GAUDI2_PRM_NAK_SYNDROME_REMOTE_INVALID_RD_REQUEST = 0x4,
};

enum
{
    DISABLE = 0,
};

enum connection_state_enum
{
    CONNECTION_STATE_OPEN   = 0,
    CONNECTION_STATE_CLOSED = 1,
    CONNECTION_STATE_RESYNC = 2,
    CONNECTION_STATE_ERROR  = 3,
};

enum
{
    TRANSPORT_SERVICE_RC  = 0,
    TRANSPORT_SERVICE_RAW = 1,
};

enum status_enum
{
    STATUS_ACK,
    STATUS_DUPLICATE_ACK,
    STATUS_SEQ_NACK,
    STATUS_ERROR,
};

enum Gaudi2PrmTrustLevelEnum
{
    GAUDI2_PRM_TRUST_LEVEL_UNSECURED = 0x0,
    GAUDI2_PRM_TRUST_LEVEL_SECURED   = 0x1,
    GAUDI2_PRM_TRUST_LEVEL_PRIVILEGE = 0x2,
};

enum Gaudi2PrmEventType
{
    GAUDI2_PRM_EVENT_TYPE_COMPLETION         = 0x0,
    GAUDI2_PRM_EVENT_TYPE_COMPLETION_ERROR   = 0x1,
    GAUDI2_PRM_EVENT_TYPE_QP_ERROR           = 0x2,
    GAUDI2_PRM_EVENT_TYPE_LINK_STATUS_CHANGE = 0x3,
    GAUDI2_PRM_EVENT_TYPE_RAW_TX_COMPLETION  = 0x4,
};

enum wq_type_enum
{
    WQ_TYPE_RESERVED   = 0x0,
    WQ_TYPE_WRITE      = 0x1,
    WQ_TYPE_READ       = 0x2,
    WQ_TYPE_RENDEZVOUS = 0x3,
};

struct Gaudi2Doorbell_t
{
    uint32_t work_queue_pi : 22;
    uint32_t : 10;

    uint32_t qpn : 24;
    uint32_t port : 8;
};
static_assert((sizeof(Gaudi2Doorbell_t) == sizeof(uint32_t) * 2), "Gaudi2Doorbell_t size is not as expected");

struct Gaudi2MultiStrideState_t
{
    uint32_t offset_in_stride;

    uint16_t stride1_index;
    uint16_t stride2_index;

    uint16_t stride3_index;
    uint16_t stride4_index;
};
static_assert((sizeof(Gaudi2MultiStrideState_t) == 12), "Gaudi2MultiStrideState_t size is not as expected");

struct Gaudi2PrmRequestorQPContext_t
{
    uint32_t destination_qp : 24;
    uint32_t : 8;

    Gaudi2MultiStrideState_t multi_stride_state;

    uint32_t remote_key;

    uint32_t destination_ip;

    uint32_t destination_mac_31_0;

    uint32_t destination_mac_47_32 : 16;
    uint32_t sequence_error_retry_count : 8;
    uint32_t timeout_retry_counter : 8;

    uint32_t next_to_send_psn : 24;
    uint32_t : 8;

    uint32_t base_currently_sending_psn : 24;
    uint32_t sq_number : 8;

    uint32_t oldest_unacked_psn : 24;
    uint32_t : 8;

    uint32_t base_currently_completing_psn : 24;
    uint32_t timer_granularity : 7;
    uint32_t wq_back_pressure : 1;

    uint32_t congestion_marked_ack : 24;
    uint32_t remote_wq_log_size : 5;
    uint32_t encapsulation_type : 3;

    uint32_t congestion_non_marked_ack : 24;
    uint32_t cq_number : 5;
    uint32_t rtt_state : 2;
    uint32_t encapsulation_enable : 1;

    uint32_t congestion_window : 24;
    uint32_t : 8;

    uint32_t rtt_timestamp : 25;
    uint32_t : 7;

    uint32_t rtt_marked_psn : 24;
    uint32_t : 8;

    uint32_t burst_size : 22;
    uint32_t asid : 10;

    uint32_t last_index : 22;
    uint32_t : 10;

    uint32_t execution_index : 22;
    uint32_t : 10;

    uint32_t consumer_index : 22;
    uint32_t : 10;

    uint32_t local_producer_index : 22;
    uint32_t : 10;

    uint32_t remote_producer_index : 22;
    uint32_t : 10;

    uint32_t remote_consumer_index : 22;
    uint32_t : 10;

    uint32_t oldest_unacked_remote_producer_index : 22;
    uint32_t : 10;

    uint32_t psn_since_ackreq : 8;
    uint32_t ackreq_freq : 8;
    uint32_t : 16;

    uint32_t : 32;

    uint32_t : 32;

    uint32_t : 32;

    uint32_t : 32;

    uint32_t : 32;

    uint32_t : 11;
    uint32_t data_mmu_bypass : 1;
    uint32_t gaudi1 : 1;
    uint32_t port : 2;
    uint32_t wq_type : 2;
    uint32_t swq_granularity : 1;
    uint32_t transport_service : 1;
    uint32_t priority : 2;
    uint32_t congestion_enable : 2;
    uint32_t mtu : 2;
    uint32_t wq_base_address : 2;
    uint32_t trust_level : 2;
    uint32_t in_work : 1;
    uint32_t error : 1;
    uint32_t valid : 1;
};
static_assert((sizeof(Gaudi2PrmRequestorQPContext_t) == sizeof(uint32_t) * 32),
              "Gaudi2PrmRequestorQPContext_t size is not as expected");

struct Gaudi2PrmResponderQPContext_t
{
    uint32_t destination_qp : 24;
    uint32_t port : 2;
    uint32_t priority : 2;
    uint32_t connection_state : 2;
    uint32_t nack_syndrome : 2;

    uint32_t local_key;

    uint32_t destination_ip;

    uint32_t destination_mac_31_0;

    uint32_t destination_mac_47_32 : 16;
    uint32_t ecn_count : 5;
    uint32_t transport_service : 1;
    uint32_t asid : 10;

    uint32_t peer_qp : 24;
    uint32_t sq_number : 8;

    uint32_t expected_psn : 24;
    uint32_t trust_level : 2;
    uint32_t gaudi1 : 1;
    uint32_t data_mmu_bypass : 1;
    uint32_t encapsulation_type : 3;
    uint32_t encapsulation_enable : 1;

    uint32_t cyclic_index : 24;
    uint32_t cq_number : 5;
    uint32_t peer_wq_granularity : 1;
    uint32_t in_work : 1;
    uint32_t valid : 1;
};
static_assert((sizeof(Gaudi2PrmResponderQPContext_t) == sizeof(uint32_t) * 8),
              "Gaudi2PrmResponderQPContext_t size is not as expected");

struct Gaudi2PrmRequesterCqe_t
{
    uint32_t qpn : 24;
    uint32_t r : 1; // should be 1
    uint32_t : 6;
    uint32_t v : 1;

    uint32_t wqe_index;

    uint32_t tag;

    uint32_t : 32;
};
static_assert((sizeof(Gaudi2PrmRequesterCqe_t) == sizeof(uint32_t) * 4),
              "Gaudi2PrmRequesterCqe_t size is not as expected");

struct Gaudi2PrmResponderCqe_t
{
    uint32_t qpn : 24;
    uint32_t r : 1; // should be 0
    uint32_t : 6;
    uint32_t v : 1;

    uint32_t wqe_index;

    uint32_t tag_immediate_data;

    uint32_t size;
};
static_assert((sizeof(Gaudi2PrmResponderCqe_t) == sizeof(uint32_t) * 4),
              "Gaudi2PrmResponderCqe_t size is not as expected");

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

union Gaudi2PrmErrFifo_t
{
    struct
    {
        uint32_t qpn : 24;
        uint32_t syndrome : 7;
        uint32_t req : 1;
    };
    uint32_t _raw;
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
static_assert((sizeof(Gaudi2PrmErrFifo_t) == sizeof(uint32_t) * 1), "Gaudi2PrmErrFifo_t size is not as expected");

union Gaudi2PrmCqe
{
    Gaudi2PrmRequesterCqe_t req;
    Gaudi2PrmResponderCqe_t res;
    uint32_t                _raw[4];
};
static_assert((sizeof(Gaudi2PrmCqe) == sizeof(uint32_t) * 4), "Gaudi2PrmCqe size is not as expected");

struct Gaudi2PrmSwqe32Pointer
{
    uint32_t opcode : 5;
    uint32_t td : 1;
    uint32_t te : 1;
    uint32_t : 1;
    uint32_t wqe_index : 8;
    uint32_t reduction_opcode : 13;
    uint32_t se : 1;
    uint32_t i : 1; // must be 0
    uint32_t a : 1;

    uint32_t size;

    uint32_t local_address_31_0;

    uint32_t local_address_63_32;

    uint32_t remote_address_31_0;

    uint32_t remote_address_63_32;

    uint32_t tag_immediate_data;

    uint32_t remote_completion_address : 27;
    uint32_t so_data : 2;
    uint32_t sc : 1;
    uint32_t ct : 2;
};
static_assert((sizeof(Gaudi2PrmSwqe32Pointer) == sizeof(uint32_t) * 8),
              "Gaudi2PrmSwqe32Pointer size is not as expected");

struct Gaudi2PrmSwqe32Inline
{
    uint32_t opcode : 5;
    uint32_t : 11;
    uint32_t reduction_opcode : 13;
    uint32_t : 1;
    uint32_t i : 1; // must be 1
    uint32_t a : 1;

    uint32_t size;

    uint32_t remote_address_31_0;

    uint32_t remote_address_63_32;

    uint32_t data[4];
};
static_assert((sizeof(Gaudi2PrmSwqe32Inline) == sizeof(uint32_t) * 8), "Gaudi2PrmSwqe32Inline size is not as expected");

struct Gaudi2PrmSwqe64Pointer
{
    uint32_t opcode : 5;
    uint32_t td : 1;
    uint32_t te : 1;
    uint32_t : 1;
    uint32_t wqe_index : 8;
    uint32_t reduction_opcode : 13;
    uint32_t se : 1;
    uint32_t i : 1; // must be 0
    uint32_t a : 1;

    uint32_t size;

    uint32_t local_address_31_0;

    uint32_t local_address_63_32;

    uint32_t remote_address_31_0;

    uint32_t remote_address_63_32;

    uint32_t tag_immediate_data;

    uint32_t remote_completion_address : 27;
    uint32_t so_data : 2;
    uint32_t sc : 1;
    uint32_t ct : 2;

    uint32_t stride1;

    uint32_t stride2;

    union
    {
        uint32_t stride3;
        uint32_t remote_stride1;
    };

    union
    {
        uint32_t stride4;
        uint32_t remote_stride2;
    };

    uint32_t number_of_strides1 : 16;
    uint32_t number_of_strides2 : 16;

    uint32_t number_of_strides3 : 16;
    uint32_t number_of_strides4 : 16;

    uint32_t stride_size;

    uint32_t : 32;
};
static_assert((sizeof(Gaudi2PrmSwqe64Pointer) == sizeof(uint32_t) * 16),
              "Gaudi2PrmSwqe64Pointer size is not as expected");

struct Gaudi2PrmSwqe64Inline
{
    uint32_t opcode : 5;
    uint32_t : 11;
    uint32_t reduction_opcode : 13;
    uint32_t : 1;
    uint32_t i : 1; // must be 1
    uint32_t a : 1;

    uint32_t size;

    uint32_t remote_address_31_0;

    uint32_t remote_address_63_32;

    uint32_t data[12];
};
static_assert((sizeof(Gaudi2PrmSwqe64Inline) == sizeof(uint32_t) * 16),
              "Gaudi2PrmSwqe64Inline size is not as expected");

struct Gaudi2PrmRwqe
{
    uint32_t opcode : 5;
    uint32_t : 3;
    uint32_t wqe_index : 8;
    uint32_t : 15;
    uint32_t sc : 1;

    uint32_t sync_object_address : 27;
    uint32_t so_data : 3;
    uint32_t ct : 2;

    uint32_t message_size;

    uint32_t tag_immediate_data;
};
static_assert((sizeof(Gaudi2PrmRwqe) == sizeof(uint32_t) * 4), "Gaudi2PrmRwqe size is not as expected");

struct Gaudi2PrmWtdLinearStaticRegisters
{
    struct rx
    {
        uint32_t opcode : 5;
        uint32_t : 3;
        uint32_t wqe_index : 8;
        uint32_t : 12;
        uint32_t pt : 2;
        uint32_t : 1;
        uint32_t sc : 1;

        uint32_t sync_object_address : 27;
        uint32_t so_data : 3;
        uint32_t ct : 2;

        uint32_t tag;
    } rx;

    struct tx
    {
        uint32_t opcode : 5;
        uint32_t td : 1;
        uint32_t te : 1;
        uint32_t : 1;
        uint32_t wqe_index : 8;
        uint32_t reduction_opcode : 13;
        uint32_t se : 1;
        uint32_t i : 1;
        uint32_t a : 1;

        uint32_t local_address_31_0;

        uint32_t local_address_63_32;

        uint32_t remote_address_31_0;

        uint32_t remote_address_63_32;

        uint32_t tag_immediate_data;

        uint32_t remote_completion_address : 27;
        uint32_t so_data : 2;
        uint32_t sc : 1;
        uint32_t ct : 2;
    } tx;
};
static_assert((sizeof(Gaudi2PrmWtdLinearStaticRegisters) == sizeof(uint32_t) * 10),
              "Gaudi2PrmWtdLinearStaticRegisters size is not as expected");

struct Gaudi2PrmWtdLinearDynamicRegisters
{
    uint32_t rx_size;

    uint32_t tx_size;

    uint32_t local_offset_31_0;

    uint32_t local_offset_63_32;

    uint32_t remote_offset_31_0;

    uint32_t remote_offset_63_32;
};
static_assert((sizeof(Gaudi2PrmWtdLinearDynamicRegisters) == sizeof(uint32_t) * 6),
              "Gaudi2PrmWtdLinearDynamicRegisters size is not as expected");

struct Gaudi2PrmWtdMultiStrideStaticRegisters
{
    struct rx
    {
        uint32_t opcode : 5;
        uint32_t : 3;
        uint32_t wqe_index : 8;
        uint32_t : 12;
        uint32_t pt : 2;
        uint32_t : 1;
        uint32_t sc : 1;

        uint32_t sync_object_address : 27;
        uint32_t so_data : 3;
        uint32_t ct : 2;

        uint32_t tag;
    } rx;

    struct tx
    {
        uint32_t opcode : 5;
        uint32_t td : 1;
        uint32_t te : 1;
        uint32_t : 1;
        uint32_t wqe_index : 8;
        uint32_t reduction_opcode : 13;
        uint32_t se : 1;
        uint32_t i : 1;
        uint32_t a : 1;

        uint32_t local_address_31_0;

        uint32_t local_address_63_32;

        uint32_t remote_address_31_0;

        uint32_t remote_address_63_32;

        uint32_t tag_immediate_data;

        uint32_t remote_completion_address : 27;
        uint32_t so_data : 2;
        uint32_t sc : 1;
        uint32_t ct : 2;

        uint32_t stride1;

        uint32_t stride2;

        uint32_t stride3;

        uint32_t stride4;

        uint32_t number_of_strides1 : 16;
        uint32_t number_of_strides2 : 16;

        uint32_t number_of_strides3 : 16;
        uint32_t number_of_strides4 : 16;

        uint32_t stride_size;

        uint32_t : 32;
    } tx;
};
static_assert((sizeof(Gaudi2PrmWtdMultiStrideStaticRegisters) == sizeof(uint32_t) * 18),
              "Gaudi2PrmWtdMultiStrideStaticRegisters size is not as expected");

struct Gaudi1TSD
{
    uint32_t number_of_strides0 : 16;
    uint32_t size : 15;
    uint32_t st : 1;

    uint32_t stride0;

    uint32_t stride1;

    uint32_t stride2;

    uint32_t stride3;

    uint32_t stride4;

    uint32_t number_of_strides2 : 16;
    uint32_t number_of_strides1 : 16;

    uint32_t number_of_strides4 : 16;
    uint32_t number_of_strides3 : 16;
};
static_assert((sizeof(Gaudi1TSD) == sizeof(uint32_t) * 8), "Gaudi1TSD size is not as expected");

struct Gaudi1SOB
{
    uint32_t completion_address : 26;
    uint32_t : 5;
    uint32_t v : 1;

    uint32_t completion_data;
};
static_assert((sizeof(Gaudi1SOB) == sizeof(uint32_t) * 2), "Gaudi1SOB size is not as expected");

struct Gaudi2TSD
{
    uint32_t stride1;

    uint32_t stride2;

    uint32_t stride3;

    uint32_t stride4;

    uint32_t number_of_strides2 : 16;
    uint32_t number_of_strides1 : 16;

    uint32_t number_of_strides4 : 16;
    uint32_t number_of_strides3 : 16;

    uint32_t stride_size;

    uint32_t strides2_index : 16;
    uint32_t strides1_index : 16;

    uint32_t strides4_index : 16;
    uint32_t strides3_index : 16;

    uint32_t offset_in_stride;
};
static_assert((sizeof(Gaudi2TSD) == sizeof(uint32_t) * 10), "Gaudi2TSD size is not as expected");

struct Gaudi2SOB
{
    uint32_t object_address : 27;
    uint32_t so_data : 2;
    uint32_t sc : 1;
    uint32_t ct : 2;

    uint32_t completion_data;
};
static_assert((sizeof(Gaudi2SOB) == sizeof(uint32_t) * 2), "Gaudi2SOB size is not as expected");

union Gaudi2Eqe
{
    struct eqe
    {
        uint32_t event_type : 4;
        uint32_t : 27;
        uint32_t v : 1;

        union event_data
        {
            struct completion_event
            {
                uint32_t cq_number : 16;
                uint32_t : 16;

                uint32_t producer_index;

                uint32_t : 32;
            } completion_event;

            struct qp_error_event
            {
                uint32_t qpn : 24;
                uint32_t : 7;
                uint32_t r : 1;

                uint32_t error_syndrome;

                uint32_t : 32;
            } qp_error_event;

            struct port_link_status_changed
            {
                uint32_t timestamp;

                uint32_t current_link_status : 4;
                uint32_t : 28;

                uint32_t : 32;
            } port_link_status_changed;

            struct raw_tx_completion
            {
                uint32_t qpn : 24;
                uint32_t : 8;

                uint32_t execution_index;

                uint32_t : 32;
            } raw_tx_completion;
        } event_data;
    } eqe;
    uint32_t _raw[4];
};
static_assert((sizeof(Gaudi2Eqe) == sizeof(uint32_t) * 4), "Gaudi2Eqe size is not as expected");

#pragma pack(pop)

#endif
