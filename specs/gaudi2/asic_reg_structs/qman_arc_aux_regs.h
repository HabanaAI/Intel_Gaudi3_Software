/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI2_QMAN_ARC_AUX_H_
#define ASIC_REG_STRUCTS_GAUDI2_QMAN_ARC_AUX_H_

#include <stdint.h>
#include "gaudi2_types.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi2 {
namespace qman_arc_aux {
#else
#	ifndef static_assert
#		if defined( __STDC__ ) && defined( __STDC_VERSION__ ) && __STDC_VERSION__ >= 201112L
#			define static_assert(...) _Static_assert(__VA_ARGS__)
#		else
#			define static_assert(...)
#		endif
#	endif
#endif

/*
 RUN_HALT_REQ 
 b'ARC: RUN/HALT Request'
*/
typedef struct reg_run_halt_req {
	union {
		struct {
			uint32_t run_req : 1,
				halt_req : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_run_halt_req;
static_assert((sizeof(struct reg_run_halt_req) == 4), "reg_run_halt_req size is not 32-bit");
/*
 RUN_HALT_ACK 
 b'ARC: RUN/HALT ACK'
*/
typedef struct reg_run_halt_ack {
	union {
		struct {
			uint32_t run_ack : 1,
				_reserved4 : 3,
				halt_ack : 1,
				_reserved8 : 3,
				sys_halt_r : 1,
				_reserved12 : 3,
				sys_tf_halt_r : 1,
				_reserved16 : 3,
				sys_sleep_r : 1,
				sys_sleep_mode_r : 3,
				watchdog_reset : 1,
				_reserved21 : 11;
		};
		uint32_t _raw;
	};
} reg_run_halt_ack;
static_assert((sizeof(struct reg_run_halt_ack) == 4), "reg_run_halt_ack size is not 32-bit");
/*
 RST_VEC_ADDR 
 b'ARC: Reset Vector Address'
*/
typedef struct reg_rst_vec_addr {
	union {
		struct {
			uint32_t val : 22,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_rst_vec_addr;
static_assert((sizeof(struct reg_rst_vec_addr) == 4), "reg_rst_vec_addr size is not 32-bit");
/*
 DBG_MODE 
 b'ARC: Debug Authentication Mode'
*/
typedef struct reg_dbg_mode {
	union {
		struct {
			uint32_t dbg_prot_sel : 1,
				_reserved4 : 3,
				dbgen : 1,
				_reserved8 : 3,
				niden : 1,
				_reserved12 : 3,
				cashe_rst_disable : 1,
				_reserved16 : 3,
				ddcm_dmi_priority : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_dbg_mode;
static_assert((sizeof(struct reg_dbg_mode) == 4), "reg_dbg_mode size is not 32-bit");
/*
 CLUSTER_NUM 
 b'ARC: Cluster Number (FARM-0..7)'
*/
typedef struct reg_cluster_num {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_cluster_num;
static_assert((sizeof(struct reg_cluster_num) == 4), "reg_cluster_num size is not 32-bit");
/*
 ARC_NUM 
 b'ARC: ARC Number (ARC enumeration)'
*/
typedef struct reg_arc_num {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_arc_num;
static_assert((sizeof(struct reg_arc_num) == 4), "reg_arc_num size is not 32-bit");
/*
 WAKE_UP_EVENT 
 b'ARC: Wake Up Event (ARC sleep wake-up event)'
*/
typedef struct reg_wake_up_event {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_wake_up_event;
static_assert((sizeof(struct reg_wake_up_event) == 4), "reg_wake_up_event size is not 32-bit");
/*
 DCCM_SYS_ADDR_BASE 
 b'ARC: DCCM ARC Base-Address'
*/
typedef struct reg_dccm_sys_addr_base {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dccm_sys_addr_base;
static_assert((sizeof(struct reg_dccm_sys_addr_base) == 4), "reg_dccm_sys_addr_base size is not 32-bit");
/*
 CTI_AP_STS 
 b'ARC: CTI (FA): Hit indicators for each AP'
*/
typedef struct reg_cti_ap_sts {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_cti_ap_sts;
static_assert((sizeof(struct reg_cti_ap_sts) == 4), "reg_cti_ap_sts size is not 32-bit");
/*
 CTI_CFG_MUX_SEL 
 b'ARC: CTI (FA): Run/Halt via CTI/REG Control'
*/
typedef struct reg_cti_cfg_mux_sel {
	union {
		struct {
			uint32_t run_halt : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cti_cfg_mux_sel;
static_assert((sizeof(struct reg_cti_cfg_mux_sel) == 4), "reg_cti_cfg_mux_sel size is not 32-bit");
/*
 ARC_RST 
 b'ARC: Processor RESET set (Asynchronous RST)'
*/
typedef struct reg_arc_rst {
	union {
		struct {
			uint32_t core : 1,
				_reserved4 : 3,
				presetdbgn : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_arc_rst;
static_assert((sizeof(struct reg_arc_rst) == 4), "reg_arc_rst size is not 32-bit");
/*
 ARC_RST_REQ 
 b'ARC: Processor RESET request due to TIMER-WD'
*/
typedef struct reg_arc_rst_req {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_arc_rst_req;
static_assert((sizeof(struct reg_arc_rst_req) == 4), "reg_arc_rst_req size is not 32-bit");
/*
 SRAM_LSB_ADDR 
 b'ARC:ADDR-EXT: SRAM LSB (31..26)'
*/
typedef struct reg_sram_lsb_addr {
	union {
		struct {
			uint32_t val : 6,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_sram_lsb_addr;
static_assert((sizeof(struct reg_sram_lsb_addr) == 4), "reg_sram_lsb_addr size is not 32-bit");
/*
 SRAM_MSB_ADDR 
 b'ARC:ADDR-EXT: SRAM MSB (63..32)'
*/
typedef struct reg_sram_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sram_msb_addr;
static_assert((sizeof(struct reg_sram_msb_addr) == 4), "reg_sram_msb_addr size is not 32-bit");
/*
 PCIE_LSB_ADDR 
 b'ARC:ADDR-EXT: PCIE LSB (31..28)'
*/
typedef struct reg_pcie_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_pcie_lsb_addr;
static_assert((sizeof(struct reg_pcie_lsb_addr) == 4), "reg_pcie_lsb_addr size is not 32-bit");
/*
 PCIE_MSB_ADDR 
 b'ARC:ADDR-EXT: PCIE MSB (63..32)'
*/
typedef struct reg_pcie_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_pcie_msb_addr;
static_assert((sizeof(struct reg_pcie_msb_addr) == 4), "reg_pcie_msb_addr size is not 32-bit");
/*
 CFG_LSB_ADDR 
 b'ARC:ADDR-EXT: LBW-CFG LSB (31..28)'
*/
typedef struct reg_cfg_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_cfg_lsb_addr;
static_assert((sizeof(struct reg_cfg_lsb_addr) == 4), "reg_cfg_lsb_addr size is not 32-bit");
/*
 CFG_MSB_ADDR 
 b'ARC:ADDR-EXT: LBW-CFG MSB (63..32)'
*/
typedef struct reg_cfg_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cfg_msb_addr;
static_assert((sizeof(struct reg_cfg_msb_addr) == 4), "reg_cfg_msb_addr size is not 32-bit");
/*
 HBM0_LSB_ADDR 
 b'ARC:ADDR-EXT: HBM_0 LSB (31..28)'
*/
typedef struct reg_hbm0_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_hbm0_lsb_addr;
static_assert((sizeof(struct reg_hbm0_lsb_addr) == 4), "reg_hbm0_lsb_addr size is not 32-bit");
/*
 HBM0_MSB_ADDR 
 b'ARC:ADDR-EXT: HBM_0 MSB (63..32)'
*/
typedef struct reg_hbm0_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_hbm0_msb_addr;
static_assert((sizeof(struct reg_hbm0_msb_addr) == 4), "reg_hbm0_msb_addr size is not 32-bit");
/*
 HBM1_LSB_ADDR 
 b'ARC:ADDR-EXT: HBM_1 LSB (31..28)'
*/
typedef struct reg_hbm1_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_hbm1_lsb_addr;
static_assert((sizeof(struct reg_hbm1_lsb_addr) == 4), "reg_hbm1_lsb_addr size is not 32-bit");
/*
 HBM1_MSB_ADDR 
 b'ARC:ADDR-EXT: HBM_1 MSB (63..32)'
*/
typedef struct reg_hbm1_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_hbm1_msb_addr;
static_assert((sizeof(struct reg_hbm1_msb_addr) == 4), "reg_hbm1_msb_addr size is not 32-bit");
/*
 HBM2_LSB_ADDR 
 b'ARC:ADDR-EXT: HBM_2 LSB (31..28)'
*/
typedef struct reg_hbm2_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_hbm2_lsb_addr;
static_assert((sizeof(struct reg_hbm2_lsb_addr) == 4), "reg_hbm2_lsb_addr size is not 32-bit");
/*
 HBM2_MSB_ADDR 
 b'ARC:ADDR-EXT: HBM_2 MSB (63..32)'
*/
typedef struct reg_hbm2_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_hbm2_msb_addr;
static_assert((sizeof(struct reg_hbm2_msb_addr) == 4), "reg_hbm2_msb_addr size is not 32-bit");
/*
 HBM3_LSB_ADDR 
 b'ARC:ADDR-EXT: HBM_3 LSB (31..28)'
*/
typedef struct reg_hbm3_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_hbm3_lsb_addr;
static_assert((sizeof(struct reg_hbm3_lsb_addr) == 4), "reg_hbm3_lsb_addr size is not 32-bit");
/*
 HBM3_MSB_ADDR 
 b'ARC:ADDR-EXT: HBM_3 MSB (63..32)'
*/
typedef struct reg_hbm3_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_hbm3_msb_addr;
static_assert((sizeof(struct reg_hbm3_msb_addr) == 4), "reg_hbm3_msb_addr size is not 32-bit");
/*
 HBM0_OFFSET 
 b'ARC:ADDR-EXT: HBM_0 offset  (27..0)'
*/
typedef struct reg_hbm0_offset {
	union {
		struct {
			uint32_t val : 28,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_hbm0_offset;
static_assert((sizeof(struct reg_hbm0_offset) == 4), "reg_hbm0_offset size is not 32-bit");
/*
 HBM1_OFFSET 
 b'ARC:ADDR-EXT: HBM_1 offset  (27..0)'
*/
typedef struct reg_hbm1_offset {
	union {
		struct {
			uint32_t val : 28,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_hbm1_offset;
static_assert((sizeof(struct reg_hbm1_offset) == 4), "reg_hbm1_offset size is not 32-bit");
/*
 HBM2_OFFSET 
 b'ARC:ADDR-EXT: HBM_2 offset  (27..0)'
*/
typedef struct reg_hbm2_offset {
	union {
		struct {
			uint32_t val : 28,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_hbm2_offset;
static_assert((sizeof(struct reg_hbm2_offset) == 4), "reg_hbm2_offset size is not 32-bit");
/*
 HBM3_OFFSET 
 b'ARC:ADDR-EXT: HBM_3 offset  (27..0)'
*/
typedef struct reg_hbm3_offset {
	union {
		struct {
			uint32_t val : 28,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_hbm3_offset;
static_assert((sizeof(struct reg_hbm3_offset) == 4), "reg_hbm3_offset size is not 32-bit");
/*
 GENERAL_PURPOSE_LSB_ADDR 
 b'ARC:ADDR-EXT: 7xGP (31..28)'
*/
typedef struct reg_general_purpose_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_general_purpose_lsb_addr;
static_assert((sizeof(struct reg_general_purpose_lsb_addr) == 4), "reg_general_purpose_lsb_addr size is not 32-bit");
/*
 GENERAL_PURPOSE_MSB_ADDR 
 b'ARC:ADDR-EXT: 7xGP (63..32)'
*/
typedef struct reg_general_purpose_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_general_purpose_msb_addr;
static_assert((sizeof(struct reg_general_purpose_msb_addr) == 4), "reg_general_purpose_msb_addr size is not 32-bit");
/*
 ARC_CBU_AWCACHE_OVR 
 b'ARC: AXI: CBU AWCACHE (Override)'
*/
typedef struct reg_arc_cbu_awcache_ovr {
	union {
		struct {
			uint32_t axi_write : 4,
				axi_write_en : 4,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_arc_cbu_awcache_ovr;
static_assert((sizeof(struct reg_arc_cbu_awcache_ovr) == 4), "reg_arc_cbu_awcache_ovr size is not 32-bit");
/*
 ARC_LBU_AWCACHE_OVR 
 b'ARC: AXI: LBU AWCACHE (Override)'
*/
typedef struct reg_arc_lbu_awcache_ovr {
	union {
		struct {
			uint32_t axi_write : 4,
				axi_write_en : 4,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_arc_lbu_awcache_ovr;
static_assert((sizeof(struct reg_arc_lbu_awcache_ovr) == 4), "reg_arc_lbu_awcache_ovr size is not 32-bit");
/*
 CONTEXT_ID 
 b'ARC: Atomic: 8xCounters (Set/Inc/Dec)'
*/
typedef struct reg_context_id {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_context_id;
static_assert((sizeof(struct reg_context_id) == 4), "reg_context_id size is not 32-bit");
/*
 CID_OFFSET 
 b'ARC: Atomic: 8xCounters offset'
*/
typedef struct reg_cid_offset {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_cid_offset;
static_assert((sizeof(struct reg_cid_offset) == 4), "reg_cid_offset size is not 32-bit");
/*
 SW_INTR 
 b'ARC: INTR: External IP Interrupt set (32xsources)'
*/
typedef struct reg_sw_intr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sw_intr;
static_assert((sizeof(struct reg_sw_intr) == 4), "reg_sw_intr size is not 32-bit");
/*
 IRQ_INTR_MASK 
 b'ARC: INTR: Interrupt mask'
*/
typedef struct reg_irq_intr_mask {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_irq_intr_mask;
static_assert((sizeof(struct reg_irq_intr_mask) == 4), "reg_irq_intr_mask size is not 32-bit");
/*
 ARC_SEI_INTR_STS 
 b'ARC:SEI: System Interrupt status'
*/
typedef struct reg_arc_sei_intr_sts {
	union {
		struct {
			uint32_t val : 14,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_arc_sei_intr_sts;
static_assert((sizeof(struct reg_arc_sei_intr_sts) == 4), "reg_arc_sei_intr_sts size is not 32-bit");
/*
 ARC_SEI_INTR_CLR 
 b'ARC:SEI: System Interrupt clear'
*/
typedef struct reg_arc_sei_intr_clr {
	union {
		struct {
			uint32_t val : 14,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_arc_sei_intr_clr;
static_assert((sizeof(struct reg_arc_sei_intr_clr) == 4), "reg_arc_sei_intr_clr size is not 32-bit");
/*
 ARC_SEI_INTR_MASK 
 b'ARC:SEI: System Interrupt mask'
*/
typedef struct reg_arc_sei_intr_mask {
	union {
		struct {
			uint32_t val : 14,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_arc_sei_intr_mask;
static_assert((sizeof(struct reg_arc_sei_intr_mask) == 4), "reg_arc_sei_intr_mask size is not 32-bit");
/*
 ARC_EXCPTN_CAUSE 
 b'ARC: Exception: Cause'
*/
typedef struct reg_arc_excptn_cause {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_excptn_cause;
static_assert((sizeof(struct reg_arc_excptn_cause) == 4), "reg_arc_excptn_cause size is not 32-bit");
/*
 SEI_INTR_HALT_EN 
 b'ARC-HALT: Enable ARC Halt/ARM INTR @ SEI INTR'
*/
typedef struct reg_sei_intr_halt_en {
	union {
		struct {
			uint32_t intr_en : 1,
				halt_en : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_sei_intr_halt_en;
static_assert((sizeof(struct reg_sei_intr_halt_en) == 4), "reg_sei_intr_halt_en size is not 32-bit");
/*
 ARC_SEI_INTR_HALT_MASK 
 b'ARC-HALT: Which SEI interrupts cause ARC HALT'
*/
typedef struct reg_arc_sei_intr_halt_mask {
	union {
		struct {
			uint32_t val : 14,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_arc_sei_intr_halt_mask;
static_assert((sizeof(struct reg_arc_sei_intr_halt_mask) == 4), "reg_arc_sei_intr_halt_mask size is not 32-bit");
/*
 QMAN_SEI_INTR_HALT_MASK 
 b'ARC-HALT: Which SEI interrupts cause QMAN HALT'
*/
typedef struct reg_qman_sei_intr_halt_mask {
	union {
		struct {
			uint32_t val : 14,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_qman_sei_intr_halt_mask;
static_assert((sizeof(struct reg_qman_sei_intr_halt_mask) == 4), "reg_qman_sei_intr_halt_mask size is not 32-bit");
/*
 ARC_REI_INTR_STS 
 b'ARC-REI: ARC RAM Error Interrupt status'
*/
typedef struct reg_arc_rei_intr_sts {
	union {
		struct {
			uint32_t serr : 1,
				derr : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_arc_rei_intr_sts;
static_assert((sizeof(struct reg_arc_rei_intr_sts) == 4), "reg_arc_rei_intr_sts size is not 32-bit");
/*
 ARC_REI_INTR_CLR 
 b'ARC-REI: ARC RAM Error Interrupt clear'
*/
typedef struct reg_arc_rei_intr_clr {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_arc_rei_intr_clr;
static_assert((sizeof(struct reg_arc_rei_intr_clr) == 4), "reg_arc_rei_intr_clr size is not 32-bit");
/*
 ARC_REI_INTR_MASK 
 b'ARC-REI: ARC RAM Error Interrupt mask'
*/
typedef struct reg_arc_rei_intr_mask {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_arc_rei_intr_mask;
static_assert((sizeof(struct reg_arc_rei_intr_mask) == 4), "reg_arc_rei_intr_mask size is not 32-bit");
/*
 DCCM_ECC_ERR_ADDR 
 b'ARC-REI: DCCM ECC ERR Address'
*/
typedef struct reg_dccm_ecc_err_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dccm_ecc_err_addr;
static_assert((sizeof(struct reg_dccm_ecc_err_addr) == 4), "reg_dccm_ecc_err_addr size is not 32-bit");
/*
 DCCM_ECC_SYNDROME 
 b'ARC-REI: DCCM ECC ERR Sync'
*/
typedef struct reg_dccm_ecc_syndrome {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dccm_ecc_syndrome;
static_assert((sizeof(struct reg_dccm_ecc_syndrome) == 4), "reg_dccm_ecc_syndrome size is not 32-bit");
/*
 I_CACHE_ECC_ERR_ADDR 
 b'ARC-ECC: Instruction-Cache ECC ERR Address'
*/
typedef struct reg_i_cache_ecc_err_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_i_cache_ecc_err_addr;
static_assert((sizeof(struct reg_i_cache_ecc_err_addr) == 4), "reg_i_cache_ecc_err_addr size is not 32-bit");
/*
 I_CACHE_ECC_SYNDROME 
 b'ARC-ECC: Instruction-Cache ECC ERR Syndrome'
*/
typedef struct reg_i_cache_ecc_syndrome {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_i_cache_ecc_syndrome;
static_assert((sizeof(struct reg_i_cache_ecc_syndrome) == 4), "reg_i_cache_ecc_syndrome size is not 32-bit");
/*
 D_CACHE_ECC_ERR_ADDR 
 b'ARC-ECC: Data-Cache ECC ERR Address'
*/
typedef struct reg_d_cache_ecc_err_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_d_cache_ecc_err_addr;
static_assert((sizeof(struct reg_d_cache_ecc_err_addr) == 4), "reg_d_cache_ecc_err_addr size is not 32-bit");
/*
 D_CACHE_ECC_SYNDROME 
 b'ARC-ECC: Data-Cache ECC ERR Syndrom'
*/
typedef struct reg_d_cache_ecc_syndrome {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_d_cache_ecc_syndrome;
static_assert((sizeof(struct reg_d_cache_ecc_syndrome) == 4), "reg_d_cache_ecc_syndrome size is not 32-bit");
/*
 LBW_TRMINATE_AWADDR_ERR 
 b'ARC-LBW-T: WR Address that caused LBW termination'
*/
typedef struct reg_lbw_trminate_awaddr_err {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_lbw_trminate_awaddr_err;
static_assert((sizeof(struct reg_lbw_trminate_awaddr_err) == 4), "reg_lbw_trminate_awaddr_err size is not 32-bit");
/*
 LBW_TRMINATE_ARADDR_ERR 
 b'ARC-LBW-T: RD Address that caused LBW termination'
*/
typedef struct reg_lbw_trminate_araddr_err {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_lbw_trminate_araddr_err;
static_assert((sizeof(struct reg_lbw_trminate_araddr_err) == 4), "reg_lbw_trminate_araddr_err size is not 32-bit");
/*
 CFG_LBW_TERMINATE_BRESP 
 b'ARC-LBW-T: LBW termination BRESP value (WR)'
*/
typedef struct reg_cfg_lbw_terminate_bresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cfg_lbw_terminate_bresp;
static_assert((sizeof(struct reg_cfg_lbw_terminate_bresp) == 4), "reg_cfg_lbw_terminate_bresp size is not 32-bit");
/*
 CFG_LBW_TERMINATE_RRESP 
 b'ARC-LBW-T: LBW termination RRESP value (RD)'
*/
typedef struct reg_cfg_lbw_terminate_rresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cfg_lbw_terminate_rresp;
static_assert((sizeof(struct reg_cfg_lbw_terminate_rresp) == 4), "reg_cfg_lbw_terminate_rresp size is not 32-bit");
/*
 CFG_LBW_TERMINATE_AXLEN 
 b'ARC-LBW-T:AXLEN threshold to enable termination'
*/
typedef struct reg_cfg_lbw_terminate_axlen {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_cfg_lbw_terminate_axlen;
static_assert((sizeof(struct reg_cfg_lbw_terminate_axlen) == 4), "reg_cfg_lbw_terminate_axlen size is not 32-bit");
/*
 CFG_LBW_TERMINATE_AXSIZE 
 b"ARC-LBW-T:AXSIZE value which don't cause terminate"
*/
typedef struct reg_cfg_lbw_terminate_axsize {
	union {
		struct {
			uint32_t val : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_cfg_lbw_terminate_axsize;
static_assert((sizeof(struct reg_cfg_lbw_terminate_axsize) == 4), "reg_cfg_lbw_terminate_axsize size is not 32-bit");
/*
 SCRATCHPAD 
 b'ARC-GP: 8x Scratch-Pad (GP) registers'
*/
typedef struct reg_scratchpad {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_scratchpad;
static_assert((sizeof(struct reg_scratchpad) == 4), "reg_scratchpad size is not 32-bit");
/*
 TOTAL_CBU_WR_CNT 
 b'ARC-CTR: Total ARC-CBU WR counter'
*/
typedef struct reg_total_cbu_wr_cnt {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_total_cbu_wr_cnt;
static_assert((sizeof(struct reg_total_cbu_wr_cnt) == 4), "reg_total_cbu_wr_cnt size is not 32-bit");
/*
 INFLIGHT_CBU_WR_CNT 
 b'ARC-CTR: Inflight ARC-LBU WR counter'
*/
typedef struct reg_inflight_cbu_wr_cnt {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_inflight_cbu_wr_cnt;
static_assert((sizeof(struct reg_inflight_cbu_wr_cnt) == 4), "reg_inflight_cbu_wr_cnt size is not 32-bit");
/*
 TOTAL_CBU_RD_CNT 
 b'ARC-CTR: Total ARC-CBU RD counter'
*/
typedef struct reg_total_cbu_rd_cnt {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_total_cbu_rd_cnt;
static_assert((sizeof(struct reg_total_cbu_rd_cnt) == 4), "reg_total_cbu_rd_cnt size is not 32-bit");
/*
 INFLIGHT_CBU_RD_CNT 
 b'ARC-CTR: Inflight ARC-CBU RD counter'
*/
typedef struct reg_inflight_cbu_rd_cnt {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_inflight_cbu_rd_cnt;
static_assert((sizeof(struct reg_inflight_cbu_rd_cnt) == 4), "reg_inflight_cbu_rd_cnt size is not 32-bit");
/*
 TOTAL_LBU_WR_CNT 
 b'ARC-CTR: Total ARC-LBU WR counter'
*/
typedef struct reg_total_lbu_wr_cnt {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_total_lbu_wr_cnt;
static_assert((sizeof(struct reg_total_lbu_wr_cnt) == 4), "reg_total_lbu_wr_cnt size is not 32-bit");
/*
 INFLIGHT_LBU_WR_CNT 
 b'ARC-CTR: Inflight ARC-LBU WR counter'
*/
typedef struct reg_inflight_lbu_wr_cnt {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_inflight_lbu_wr_cnt;
static_assert((sizeof(struct reg_inflight_lbu_wr_cnt) == 4), "reg_inflight_lbu_wr_cnt size is not 32-bit");
/*
 TOTAL_LBU_RD_CNT 
 b'ARC-CTR: Total ARC-LBU RD counter'
*/
typedef struct reg_total_lbu_rd_cnt {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_total_lbu_rd_cnt;
static_assert((sizeof(struct reg_total_lbu_rd_cnt) == 4), "reg_total_lbu_rd_cnt size is not 32-bit");
/*
 INFLIGHT_LBU_RD_CNT 
 b'ARC-CTR: Inflight ARC-LBU RD counter'
*/
typedef struct reg_inflight_lbu_rd_cnt {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_inflight_lbu_rd_cnt;
static_assert((sizeof(struct reg_inflight_lbu_rd_cnt) == 4), "reg_inflight_lbu_rd_cnt size is not 32-bit");
/*
 CBU_ARUSER_OVR 
 b'ARC-CBU-OVR: ARUSER Overide (LSB)'
*/
typedef struct reg_cbu_aruser_ovr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_aruser_ovr;
static_assert((sizeof(struct reg_cbu_aruser_ovr) == 4), "reg_cbu_aruser_ovr size is not 32-bit");
/*
 CBU_ARUSER_OVR_EN 
 b'ARC-CBU-OVR: ARUSER Overide (LSB) Enable'
*/
typedef struct reg_cbu_aruser_ovr_en {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_aruser_ovr_en;
static_assert((sizeof(struct reg_cbu_aruser_ovr_en) == 4), "reg_cbu_aruser_ovr_en size is not 32-bit");
/*
 CBU_AWUSER_OVR 
 b'ARC-CBU-OVR: AWUSER Overide (LSB)'
*/
typedef struct reg_cbu_awuser_ovr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_awuser_ovr;
static_assert((sizeof(struct reg_cbu_awuser_ovr) == 4), "reg_cbu_awuser_ovr size is not 32-bit");
/*
 CBU_AWUSER_OVR_EN 
 b'ARC-CBU-OVR: AWUSER Overide (LSB) Enable'
*/
typedef struct reg_cbu_awuser_ovr_en {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_awuser_ovr_en;
static_assert((sizeof(struct reg_cbu_awuser_ovr_en) == 4), "reg_cbu_awuser_ovr_en size is not 32-bit");
/*
 CBU_ARUSER_MSB_OVR 
 b'ARC-CBU-OVR: ARUSER Overide (MSB)'
*/
typedef struct reg_cbu_aruser_msb_ovr {
	union {
		struct {
			uint32_t val : 10,
				_reserved10 : 22;
		};
		uint32_t _raw;
	};
} reg_cbu_aruser_msb_ovr;
static_assert((sizeof(struct reg_cbu_aruser_msb_ovr) == 4), "reg_cbu_aruser_msb_ovr size is not 32-bit");
/*
 CBU_ARUSER_MSB_OVR_EN 
 b'ARC-CBU-OVR: ARUSER Overide (MSB) Enable'
*/
typedef struct reg_cbu_aruser_msb_ovr_en {
	union {
		struct {
			uint32_t val : 10,
				_reserved10 : 22;
		};
		uint32_t _raw;
	};
} reg_cbu_aruser_msb_ovr_en;
static_assert((sizeof(struct reg_cbu_aruser_msb_ovr_en) == 4), "reg_cbu_aruser_msb_ovr_en size is not 32-bit");
/*
 CBU_AWUSER_MSB_OVR 
 b'ARC-CBU-OVR: AWUSER Overide (MSB)'
*/
typedef struct reg_cbu_awuser_msb_ovr {
	union {
		struct {
			uint32_t val : 10,
				_reserved10 : 22;
		};
		uint32_t _raw;
	};
} reg_cbu_awuser_msb_ovr;
static_assert((sizeof(struct reg_cbu_awuser_msb_ovr) == 4), "reg_cbu_awuser_msb_ovr size is not 32-bit");
/*
 CBU_AWUSER_MSB_OVR_EN 
 b'ARC-CBU-OVR: AWUSER Overide (MSB) Enable'
*/
typedef struct reg_cbu_awuser_msb_ovr_en {
	union {
		struct {
			uint32_t val : 10,
				_reserved10 : 22;
		};
		uint32_t _raw;
	};
} reg_cbu_awuser_msb_ovr_en;
static_assert((sizeof(struct reg_cbu_awuser_msb_ovr_en) == 4), "reg_cbu_awuser_msb_ovr_en size is not 32-bit");
/*
 CBU_AXCACHE_OVR 
 b'ARC-CBU-OVR: AxCACHE Overide'
*/
typedef struct reg_cbu_axcache_ovr {
	union {
		struct {
			uint32_t cbu_read : 4,
				cbu_write : 4,
				cbu_rd_en : 4,
				cbu_wr_en : 4,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_cbu_axcache_ovr;
static_assert((sizeof(struct reg_cbu_axcache_ovr) == 4), "reg_cbu_axcache_ovr size is not 32-bit");
/*
 CBU_LOCK_OVR 
 b'ARC-CBU-OVR: Lock Overide'
*/
typedef struct reg_cbu_lock_ovr {
	union {
		struct {
			uint32_t cbu_read : 2,
				_reserved4 : 2,
				cbu_write : 2,
				_reserved8 : 2,
				cbu_rd_en : 2,
				_reserved12 : 2,
				cbu_wr_en : 2,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_cbu_lock_ovr;
static_assert((sizeof(struct reg_cbu_lock_ovr) == 4), "reg_cbu_lock_ovr size is not 32-bit");
/*
 CBU_PROT_OVR 
 b'ARC-CBU-OVR: Protection override'
*/
typedef struct reg_cbu_prot_ovr {
	union {
		struct {
			uint32_t cbu_read : 3,
				_reserved4 : 1,
				cbu_write : 3,
				_reserved8 : 1,
				cbu_rd_en : 3,
				_reserved12 : 1,
				cbu_wr_en : 3,
				_reserved15 : 17;
		};
		uint32_t _raw;
	};
} reg_cbu_prot_ovr;
static_assert((sizeof(struct reg_cbu_prot_ovr) == 4), "reg_cbu_prot_ovr size is not 32-bit");
/*
 CBU_MAX_OUTSTANDING 
 b'ARC-CBU: Set maximum outstanding requests'
*/
typedef struct reg_cbu_max_outstanding {
	union {
		struct {
			uint32_t cbu_read : 8,
				cbu_write : 8,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_cbu_max_outstanding;
static_assert((sizeof(struct reg_cbu_max_outstanding) == 4), "reg_cbu_max_outstanding size is not 32-bit");
/*
 CBU_EARLY_BRESP_EN 
 b'ARC-CBU-BRESP: Early BRESP Configuration'
*/
typedef struct reg_cbu_early_bresp_en {
	union {
		struct {
			uint32_t cbu_val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cbu_early_bresp_en;
static_assert((sizeof(struct reg_cbu_early_bresp_en) == 4), "reg_cbu_early_bresp_en size is not 32-bit");
/*
 CBU_FORCE_RSP_OK 
 b'ARC-CBU-BRESP: FORCE BRESP OK'
*/
typedef struct reg_cbu_force_rsp_ok {
	union {
		struct {
			uint32_t cbu_val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cbu_force_rsp_ok;
static_assert((sizeof(struct reg_cbu_force_rsp_ok) == 4), "reg_cbu_force_rsp_ok size is not 32-bit");
/*
 CBU_NO_WR_INFLIGHT 
 b'ARC-CBU-SPLIT: No WR Inflight'
*/
typedef struct reg_cbu_no_wr_inflight {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cbu_no_wr_inflight;
static_assert((sizeof(struct reg_cbu_no_wr_inflight) == 4), "reg_cbu_no_wr_inflight size is not 32-bit");
/*
 CBU_SEI_INTR_ID 
 b'ARC-CBU-SPLIT: SEI Interrupt ID'
*/
typedef struct reg_cbu_sei_intr_id {
	union {
		struct {
			uint32_t val : 7,
				_reserved7 : 25;
		};
		uint32_t _raw;
	};
} reg_cbu_sei_intr_id;
static_assert((sizeof(struct reg_cbu_sei_intr_id) == 4), "reg_cbu_sei_intr_id size is not 32-bit");
/*
 LBU_ARUSER_OVR 
 b'ARC-LBU-OVR: ARUSER Overide (LSB)'
*/
typedef struct reg_lbu_aruser_ovr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_lbu_aruser_ovr;
static_assert((sizeof(struct reg_lbu_aruser_ovr) == 4), "reg_lbu_aruser_ovr size is not 32-bit");
/*
 LBU_ARUSER_OVR_EN 
 b'ARC-LBU-OVR: ARUSER Overide (LSB) Enable'
*/
typedef struct reg_lbu_aruser_ovr_en {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_lbu_aruser_ovr_en;
static_assert((sizeof(struct reg_lbu_aruser_ovr_en) == 4), "reg_lbu_aruser_ovr_en size is not 32-bit");
/*
 LBU_AWUSER_OVR 
 b'ARC-LBU-OVR: AWUSER Overide (LSB)'
*/
typedef struct reg_lbu_awuser_ovr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_lbu_awuser_ovr;
static_assert((sizeof(struct reg_lbu_awuser_ovr) == 4), "reg_lbu_awuser_ovr size is not 32-bit");
/*
 LBU_AWUSER_OVR_EN 
 b'ARC-LBU-OVR: AWUSER Overide (LSB) Enable'
*/
typedef struct reg_lbu_awuser_ovr_en {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_lbu_awuser_ovr_en;
static_assert((sizeof(struct reg_lbu_awuser_ovr_en) == 4), "reg_lbu_awuser_ovr_en size is not 32-bit");
/*
 LBU_AXCACHE_OVR 
 b'ARC-LBU-OVR: AxCACHE Overide'
*/
typedef struct reg_lbu_axcache_ovr {
	union {
		struct {
			uint32_t lbu_read : 4,
				lbu_write : 4,
				lbu_rd_en : 4,
				lbu_wr_en : 4,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_lbu_axcache_ovr;
static_assert((sizeof(struct reg_lbu_axcache_ovr) == 4), "reg_lbu_axcache_ovr size is not 32-bit");
/*
 LBU_LOCK_OVR 
 b'ARC-LBU-OVR: Lock Overide'
*/
typedef struct reg_lbu_lock_ovr {
	union {
		struct {
			uint32_t lbu_read : 2,
				_reserved4 : 2,
				lbu_write : 2,
				_reserved8 : 2,
				lbu_rd_en : 2,
				_reserved12 : 2,
				lbu_wr_en : 2,
				_reserved14 : 18;
		};
		uint32_t _raw;
	};
} reg_lbu_lock_ovr;
static_assert((sizeof(struct reg_lbu_lock_ovr) == 4), "reg_lbu_lock_ovr size is not 32-bit");
/*
 LBU_PROT_OVR 
 b'ARC-LBU-OVR: Protection override'
*/
typedef struct reg_lbu_prot_ovr {
	union {
		struct {
			uint32_t lbu_read : 3,
				_reserved4 : 1,
				lbu_write : 3,
				_reserved8 : 1,
				lbu_rd_en : 3,
				_reserved12 : 1,
				lbu_wr_en : 3,
				_reserved15 : 17;
		};
		uint32_t _raw;
	};
} reg_lbu_prot_ovr;
static_assert((sizeof(struct reg_lbu_prot_ovr) == 4), "reg_lbu_prot_ovr size is not 32-bit");
/*
 LBU_MAX_OUTSTANDING 
 b'ARC-LBU: Set maximum outstanding requests'
*/
typedef struct reg_lbu_max_outstanding {
	union {
		struct {
			uint32_t lbu_read : 8,
				lbu_write : 8,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_lbu_max_outstanding;
static_assert((sizeof(struct reg_lbu_max_outstanding) == 4), "reg_lbu_max_outstanding size is not 32-bit");
/*
 LBU_EARLY_BRESP_EN 
 b'ARC-LBU-BRESP: Early BRESP Configuration'
*/
typedef struct reg_lbu_early_bresp_en {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_lbu_early_bresp_en;
static_assert((sizeof(struct reg_lbu_early_bresp_en) == 4), "reg_lbu_early_bresp_en size is not 32-bit");
/*
 LBU_FORCE_RSP_OK 
 b'ARC-LBU-BRESP: FORCE BRESP OK'
*/
typedef struct reg_lbu_force_rsp_ok {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_lbu_force_rsp_ok;
static_assert((sizeof(struct reg_lbu_force_rsp_ok) == 4), "reg_lbu_force_rsp_ok size is not 32-bit");
/*
 LBU_NO_WR_INFLIGHT 
 b'ARC-LBU-SPLIT: No WR Inflight'
*/
typedef struct reg_lbu_no_wr_inflight {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_lbu_no_wr_inflight;
static_assert((sizeof(struct reg_lbu_no_wr_inflight) == 4), "reg_lbu_no_wr_inflight size is not 32-bit");
/*
 LBU_SEI_INTR_ID 
 b'ARC-LBU-SPLIT: SEI Interrupt ID'
*/
typedef struct reg_lbu_sei_intr_id {
	union {
		struct {
			uint32_t val : 10,
				_reserved10 : 22;
		};
		uint32_t _raw;
	};
} reg_lbu_sei_intr_id;
static_assert((sizeof(struct reg_lbu_sei_intr_id) == 4), "reg_lbu_sei_intr_id size is not 32-bit");
/*
 DCCM_QUEUE_BASE_ADDR 
 b'ARC-DCCM-QUEUE: Base Address'
*/
typedef struct reg_dccm_queue_base_addr {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_base_addr;
static_assert((sizeof(struct reg_dccm_queue_base_addr) == 4), "reg_dccm_queue_base_addr size is not 32-bit");
/*
 DCCM_QUEUE_SIZE 
 b'ARC-DCCM-QUEUE: Size'
*/
typedef struct reg_dccm_queue_size {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_size;
static_assert((sizeof(struct reg_dccm_queue_size) == 4), "reg_dccm_queue_size size is not 32-bit");
/*
 DCCM_QUEUE_PI 
 b'ARC-DCCM-QUEUE: PI'
*/
typedef struct reg_dccm_queue_pi {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_pi;
static_assert((sizeof(struct reg_dccm_queue_pi) == 4), "reg_dccm_queue_pi size is not 32-bit");
/*
 DCCM_QUEUE_CI 
 b'ARC-DCCM-QUEUE: CI'
*/
typedef struct reg_dccm_queue_ci {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_ci;
static_assert((sizeof(struct reg_dccm_queue_ci) == 4), "reg_dccm_queue_ci size is not 32-bit");
/*
 DCCM_QUEUE_PUSH_REG 
 b'ARC-DCCM-QUEUE: PUSH REG'
*/
typedef struct reg_dccm_queue_push_reg {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_push_reg;
static_assert((sizeof(struct reg_dccm_queue_push_reg) == 4), "reg_dccm_queue_push_reg size is not 32-bit");
/*
 DCCM_QUEUE_MAX_OCCUPANCY 
 b'ARC-DCCM-QUEUES (x8): Maximum occupancy'
*/
typedef struct reg_dccm_queue_max_occupancy {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_max_occupancy;
static_assert((sizeof(struct reg_dccm_queue_max_occupancy) == 4), "reg_dccm_queue_max_occupancy size is not 32-bit");
/*
 DCCM_QUEUE_VALID_ENTRIES 
 b'ARC-DCCM-QUEUES (x8): Valid Entries'
*/
typedef struct reg_dccm_queue_valid_entries {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_valid_entries;
static_assert((sizeof(struct reg_dccm_queue_valid_entries) == 4), "reg_dccm_queue_valid_entries size is not 32-bit");
/*
 GENERAL_Q_VLD_ENTRY_MASK 
 b'ARC-DCCM-QUEUES (x8): Valid Entries mask'
*/
typedef struct reg_general_q_vld_entry_mask {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_general_q_vld_entry_mask;
static_assert((sizeof(struct reg_general_q_vld_entry_mask) == 4), "reg_general_q_vld_entry_mask size is not 32-bit");
/*
 NIC_Q_VLD_ENTRY_MASK 
 b'NIC: NIC QUEUE Valid Entries Mask'
*/
typedef struct reg_nic_q_vld_entry_mask {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_nic_q_vld_entry_mask;
static_assert((sizeof(struct reg_nic_q_vld_entry_mask) == 4), "reg_nic_q_vld_entry_mask size is not 32-bit");
/*
 DCCM_QUEUE_DROP_EN 
 b'ARC-DCCM-QUEUES (x8): Drop request / Back-pressure'
*/
typedef struct reg_dccm_queue_drop_en {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_drop_en;
static_assert((sizeof(struct reg_dccm_queue_drop_en) == 4), "reg_dccm_queue_drop_en size is not 32-bit");
/*
 DCCM_QUEUE_WARN_MSG 
 b'ARC-DCCM-QUEUES (x8): Warn MSG(Q close to full)'
*/
typedef struct reg_dccm_queue_warn_msg {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_warn_msg;
static_assert((sizeof(struct reg_dccm_queue_warn_msg) == 4), "reg_dccm_queue_warn_msg size is not 32-bit");
/*
 DCCM_QUEUE_ALERT_MSG 
 b'ARC-DCCM-QUEUES (x8): Alert MSG(Q close to full)'
*/
typedef struct reg_dccm_queue_alert_msg {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_alert_msg;
static_assert((sizeof(struct reg_dccm_queue_alert_msg) == 4), "reg_dccm_queue_alert_msg size is not 32-bit");
/*
 DCCM_GEN_AXI_AWPROT 
 b'ARC-DCCM-QUEUES (x8): AXI AWPROT Value'
*/
typedef struct reg_dccm_gen_axi_awprot {
	union {
		struct {
			uint32_t val : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_dccm_gen_axi_awprot;
static_assert((sizeof(struct reg_dccm_gen_axi_awprot) == 4), "reg_dccm_gen_axi_awprot size is not 32-bit");
/*
 DCCM_GEN_AXI_AWUSER 
 b'ARC-DCCM-QUEUES (x8): AXI AWUSER Value'
*/
typedef struct reg_dccm_gen_axi_awuser {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_dccm_gen_axi_awuser;
static_assert((sizeof(struct reg_dccm_gen_axi_awuser) == 4), "reg_dccm_gen_axi_awuser size is not 32-bit");
/*
 DCCM_GEN_AXI_AWBURST 
 b'ARC-DCCM-QUEUES (x8): AXI AWBURST Value'
*/
typedef struct reg_dccm_gen_axi_awburst {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_dccm_gen_axi_awburst;
static_assert((sizeof(struct reg_dccm_gen_axi_awburst) == 4), "reg_dccm_gen_axi_awburst size is not 32-bit");
/*
 DCCM_GEN_AXI_AWLOCK 
 b'ARC-DCCM-QUEUES (x8): AXI AWLOCK Value'
*/
typedef struct reg_dccm_gen_axi_awlock {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_dccm_gen_axi_awlock;
static_assert((sizeof(struct reg_dccm_gen_axi_awlock) == 4), "reg_dccm_gen_axi_awlock size is not 32-bit");
/*
 DCCM_GEN_AXI_AWCACHE 
 b'ARC-DCCM-QUEUES (x8): AXI AWCACHE Value'
*/
typedef struct reg_dccm_gen_axi_awcache {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_dccm_gen_axi_awcache;
static_assert((sizeof(struct reg_dccm_gen_axi_awcache) == 4), "reg_dccm_gen_axi_awcache size is not 32-bit");
/*
 DCCM_WRR_ARB_WEIGHT 
 b'ARC-DMI: CPU DCCM Vs DMI prioritization'
*/
typedef struct reg_dccm_wrr_arb_weight {
	union {
		struct {
			uint32_t lbw_slv_axi : 4,
				gen_axi : 4,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_dccm_wrr_arb_weight;
static_assert((sizeof(struct reg_dccm_wrr_arb_weight) == 4), "reg_dccm_wrr_arb_weight size is not 32-bit");
/*
 DCCM_Q_PUSH_FIFO_FULL_CFG 
 b'ARC-DCCM-QUEUES (x8): FIFO Full Threshold'
*/
typedef struct reg_dccm_q_push_fifo_full_cfg {
	union {
		struct {
			uint32_t val : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_dccm_q_push_fifo_full_cfg;
static_assert((sizeof(struct reg_dccm_q_push_fifo_full_cfg) == 4), "reg_dccm_q_push_fifo_full_cfg size is not 32-bit");
/*
 DCCM_Q_PUSH_FIFO_CNT 
 b'ARC-DCCM-QUEUES (x8): FIFO queue occupancy'
*/
typedef struct reg_dccm_q_push_fifo_cnt {
	union {
		struct {
			uint32_t val : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_dccm_q_push_fifo_cnt;
static_assert((sizeof(struct reg_dccm_q_push_fifo_cnt) == 4), "reg_dccm_q_push_fifo_cnt size is not 32-bit");
/*
 QMAN_CQ_IFIFO_SHADOW_CI 
 b'ARC-QMAN: CQ In FIFO SHADOW CI'
*/
typedef struct reg_qman_cq_ififo_shadow_ci {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_qman_cq_ififo_shadow_ci;
static_assert((sizeof(struct reg_qman_cq_ififo_shadow_ci) == 4), "reg_qman_cq_ififo_shadow_ci size is not 32-bit");
/*
 QMAN_ARC_CQ_IFIFO_SHADOW_CI 
 b'ARC-QMAN: ARC-CQ In FIFO SHADOW CI'
*/
typedef struct reg_qman_arc_cq_ififo_shadow_ci {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_qman_arc_cq_ififo_shadow_ci;
static_assert((sizeof(struct reg_qman_arc_cq_ififo_shadow_ci) == 4), "reg_qman_arc_cq_ififo_shadow_ci size is not 32-bit");
/*
 QMAN_CQ_SHADOW_CI 
 b'ARC-QMAN: CQ SHADOW CI'
*/
typedef struct reg_qman_cq_shadow_ci {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_qman_cq_shadow_ci;
static_assert((sizeof(struct reg_qman_cq_shadow_ci) == 4), "reg_qman_cq_shadow_ci size is not 32-bit");
/*
 QMAN_ARC_CQ_SHADOW_CI 
 b'ARC-QMAN: ARC-CQ SHADOW CI'
*/
typedef struct reg_qman_arc_cq_shadow_ci {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_qman_arc_cq_shadow_ci;
static_assert((sizeof(struct reg_qman_arc_cq_shadow_ci) == 4), "reg_qman_arc_cq_shadow_ci size is not 32-bit");
/*
 AUX2APB_PROT 
 b'ARC-APP: AUX2APB protection Value (PPROT)'
*/
typedef struct reg_aux2apb_prot {
	union {
		struct {
			uint32_t val : 3,
				_reserved3 : 29;
		};
		uint32_t _raw;
	};
} reg_aux2apb_prot;
static_assert((sizeof(struct reg_aux2apb_prot) == 4), "reg_aux2apb_prot size is not 32-bit");
/*
 LBW_FORK_WIN_EN 
 b'QMAN-LBW-FORK: Enable'
*/
typedef struct reg_lbw_fork_win_en {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lbw_fork_win_en;
static_assert((sizeof(struct reg_lbw_fork_win_en) == 4), "reg_lbw_fork_win_en size is not 32-bit");
/*
 QMAN_LBW_FORK_BASE_ADDR0 
 b'QMAN-LBW-FORK: ARC-AUX Base address'
*/
typedef struct reg_qman_lbw_fork_base_addr0 {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_base_addr0;
static_assert((sizeof(struct reg_qman_lbw_fork_base_addr0) == 4), "reg_qman_lbw_fork_base_addr0 size is not 32-bit");
/*
 QMAN_LBW_FORK_ADDR_MASK0 
 b'QMAN-LBW-FORK: ARC-AUX (Mask bits)'
*/
typedef struct reg_qman_lbw_fork_addr_mask0 {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_addr_mask0;
static_assert((sizeof(struct reg_qman_lbw_fork_addr_mask0) == 4), "reg_qman_lbw_fork_addr_mask0 size is not 32-bit");
/*
 QMAN_LBW_FORK_BASE_ADDR1 
 b'QMAN-LBW-FORK: DCCM Address'
*/
typedef struct reg_qman_lbw_fork_base_addr1 {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_base_addr1;
static_assert((sizeof(struct reg_qman_lbw_fork_base_addr1) == 4), "reg_qman_lbw_fork_base_addr1 size is not 32-bit");
/*
 QMAN_LBW_FORK_ADDR_MASK1 
 b'QMAN-LBW-FORK: DCCM Address mask'
*/
typedef struct reg_qman_lbw_fork_addr_mask1 {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_addr_mask1;
static_assert((sizeof(struct reg_qman_lbw_fork_addr_mask1) == 4), "reg_qman_lbw_fork_addr_mask1 size is not 32-bit");
/*
 FARM_LBW_FORK_BASE_ADDR0 
 b'FARM LBW FORK: ARC-AUX Base Address'
*/
typedef struct reg_farm_lbw_fork_base_addr0 {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_base_addr0;
static_assert((sizeof(struct reg_farm_lbw_fork_base_addr0) == 4), "reg_farm_lbw_fork_base_addr0 size is not 32-bit");
/*
 FARM_LBW_FORK_ADDR_MASK0 
 b'FARM-LBW-FORK: ARC-AUX (Mask bits)'
*/
typedef struct reg_farm_lbw_fork_addr_mask0 {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_addr_mask0;
static_assert((sizeof(struct reg_farm_lbw_fork_addr_mask0) == 4), "reg_farm_lbw_fork_addr_mask0 size is not 32-bit");
/*
 FARM_LBW_FORK_BASE_ADDR1 
 b'FARM-LBW-FORK: DCCM Address'
*/
typedef struct reg_farm_lbw_fork_base_addr1 {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_base_addr1;
static_assert((sizeof(struct reg_farm_lbw_fork_base_addr1) == 4), "reg_farm_lbw_fork_base_addr1 size is not 32-bit");
/*
 FARM_LBW_FORK_ADDR_MASK1 
 b'FARM-LBW-FORK: DCCM Address mask'
*/
typedef struct reg_farm_lbw_fork_addr_mask1 {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_addr_mask1;
static_assert((sizeof(struct reg_farm_lbw_fork_addr_mask1) == 4), "reg_farm_lbw_fork_addr_mask1 size is not 32-bit");
/*
 LBW_APB_FORK_MAX_ADDR0 
 b'LBW_APB_FORK: Above ADDR0 route to ERR'
*/
typedef struct reg_lbw_apb_fork_max_addr0 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_lbw_apb_fork_max_addr0;
static_assert((sizeof(struct reg_lbw_apb_fork_max_addr0) == 4), "reg_lbw_apb_fork_max_addr0 size is not 32-bit");
/*
 LBW_APB_FORK_MAX_ADDR1 
 b'LBW_APB_FORK: Above ADDR1 route to ARC-AUX'
*/
typedef struct reg_lbw_apb_fork_max_addr1 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_lbw_apb_fork_max_addr1;
static_assert((sizeof(struct reg_lbw_apb_fork_max_addr1) == 4), "reg_lbw_apb_fork_max_addr1 size is not 32-bit");
/*
 ARC_ACC_ENGS_LBW_FORK_MASK 
 b'ARC ACC Engs LBW FORK Mask'
*/
typedef struct reg_arc_acc_engs_lbw_fork_mask {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_arc_acc_engs_lbw_fork_mask;
static_assert((sizeof(struct reg_arc_acc_engs_lbw_fork_mask) == 4), "reg_arc_acc_engs_lbw_fork_mask size is not 32-bit");
/*
 ARC_DUP_ENG_LBW_FORK_ADDR 
 b'ARC DUP FORK: ARC DUP LBW FORK Address'
*/
typedef struct reg_arc_dup_eng_lbw_fork_addr {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_arc_dup_eng_lbw_fork_addr;
static_assert((sizeof(struct reg_arc_dup_eng_lbw_fork_addr) == 4), "reg_arc_dup_eng_lbw_fork_addr size is not 32-bit");
/*
 ARC_ACP_ENG_LBW_FORK_ADDR 
 b'ARC ACP FORK: ARC ACP LBW Fork Address'
*/
typedef struct reg_arc_acp_eng_lbw_fork_addr {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_arc_acp_eng_lbw_fork_addr;
static_assert((sizeof(struct reg_arc_acp_eng_lbw_fork_addr) == 4), "reg_arc_acp_eng_lbw_fork_addr size is not 32-bit");
/*
 ARC_ACC_ENGS_VIRTUAL_ADDR 
 b'Virtual addr for internal engines'
*/
typedef struct reg_arc_acc_engs_virtual_addr {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_arc_acc_engs_virtual_addr;
static_assert((sizeof(struct reg_arc_acc_engs_virtual_addr) == 4), "reg_arc_acc_engs_virtual_addr size is not 32-bit");
/*
 CBU_FORK_WIN_EN 
 b'CBU FORK: Window Enable'
*/
typedef struct reg_cbu_fork_win_en {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_win_en;
static_assert((sizeof(struct reg_cbu_fork_win_en) == 4), "reg_cbu_fork_win_en size is not 32-bit");
/*
 CBU_FORK_BASE_ADDR0_LSB 
 b'CBU FORK: CBU RD FORK Base Address0 LSB'
*/
typedef struct reg_cbu_fork_base_addr0_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_base_addr0_lsb;
static_assert((sizeof(struct reg_cbu_fork_base_addr0_lsb) == 4), "reg_cbu_fork_base_addr0_lsb size is not 32-bit");
/*
 CBU_FORK_BASE_ADDR0_MSB 
 b'CBU FORK: CBU RD FORK Base Address0 MSB'
*/
typedef struct reg_cbu_fork_base_addr0_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_base_addr0_msb;
static_assert((sizeof(struct reg_cbu_fork_base_addr0_msb) == 4), "reg_cbu_fork_base_addr0_msb size is not 32-bit");
/*
 CBU_FORK_ADDR_MASK0_LSB 
 b'CBU FORK: CBU RD FORK Mask Address0 LSB'
*/
typedef struct reg_cbu_fork_addr_mask0_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_addr_mask0_lsb;
static_assert((sizeof(struct reg_cbu_fork_addr_mask0_lsb) == 4), "reg_cbu_fork_addr_mask0_lsb size is not 32-bit");
/*
 CBU_FORK_ADDR_MASK0_MSB 
 b'CBU FORK: CBU RD FORK Mask Address0 MSB'
*/
typedef struct reg_cbu_fork_addr_mask0_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_addr_mask0_msb;
static_assert((sizeof(struct reg_cbu_fork_addr_mask0_msb) == 4), "reg_cbu_fork_addr_mask0_msb size is not 32-bit");
/*
 CBU_FORK_BASE_ADDR1_LSB 
 b'CBU FORK: CBU RD FORK Base Address1 LSB'
*/
typedef struct reg_cbu_fork_base_addr1_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_base_addr1_lsb;
static_assert((sizeof(struct reg_cbu_fork_base_addr1_lsb) == 4), "reg_cbu_fork_base_addr1_lsb size is not 32-bit");
/*
 CBU_FORK_BASE_ADDR1_MSB 
 b'CBU FORK: CBU RD FORK Base Address1 MSB'
*/
typedef struct reg_cbu_fork_base_addr1_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_base_addr1_msb;
static_assert((sizeof(struct reg_cbu_fork_base_addr1_msb) == 4), "reg_cbu_fork_base_addr1_msb size is not 32-bit");
/*
 CBU_FORK_ADDR_MASK1_LSB 
 b'CBU FORK: CBU RD FORK Mask Address1 LSB'
*/
typedef struct reg_cbu_fork_addr_mask1_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_addr_mask1_lsb;
static_assert((sizeof(struct reg_cbu_fork_addr_mask1_lsb) == 4), "reg_cbu_fork_addr_mask1_lsb size is not 32-bit");
/*
 CBU_FORK_ADDR_MASK1_MSB 
 b'CBU FORK: CBU RD FORK Mask Address1 MSB'
*/
typedef struct reg_cbu_fork_addr_mask1_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_addr_mask1_msb;
static_assert((sizeof(struct reg_cbu_fork_addr_mask1_msb) == 4), "reg_cbu_fork_addr_mask1_msb size is not 32-bit");
/*
 CBU_FORK_BASE_ADDR2_LSB 
 b'CBU FORK: CBU RD FORK Base Address2 LSB'
*/
typedef struct reg_cbu_fork_base_addr2_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_base_addr2_lsb;
static_assert((sizeof(struct reg_cbu_fork_base_addr2_lsb) == 4), "reg_cbu_fork_base_addr2_lsb size is not 32-bit");
/*
 CBU_FORK_BASE_ADDR2_MSB 
 b'CBU FORK: CBU RD FORK Base Address2 MSB'
*/
typedef struct reg_cbu_fork_base_addr2_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_base_addr2_msb;
static_assert((sizeof(struct reg_cbu_fork_base_addr2_msb) == 4), "reg_cbu_fork_base_addr2_msb size is not 32-bit");
/*
 CBU_FORK_ADDR_MASK2_LSB 
 b'CBU FORK: CBU RD FORK Mask Address2 LSB'
*/
typedef struct reg_cbu_fork_addr_mask2_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_addr_mask2_lsb;
static_assert((sizeof(struct reg_cbu_fork_addr_mask2_lsb) == 4), "reg_cbu_fork_addr_mask2_lsb size is not 32-bit");
/*
 CBU_FORK_ADDR_MASK2_MSB 
 b'CBU FORK: CBU RD FORK Mask Address2 MSB'
*/
typedef struct reg_cbu_fork_addr_mask2_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_addr_mask2_msb;
static_assert((sizeof(struct reg_cbu_fork_addr_mask2_msb) == 4), "reg_cbu_fork_addr_mask2_msb size is not 32-bit");
/*
 CBU_FORK_BASE_ADDR3_LSB 
 b'CBU FORK: CBU RD FORK Base Address3 LSB'
*/
typedef struct reg_cbu_fork_base_addr3_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_base_addr3_lsb;
static_assert((sizeof(struct reg_cbu_fork_base_addr3_lsb) == 4), "reg_cbu_fork_base_addr3_lsb size is not 32-bit");
/*
 CBU_FORK_BASE_ADDR3_MSB 
 b'CBU FORK: CBU RD FORK Base Address3 MSB'
*/
typedef struct reg_cbu_fork_base_addr3_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_base_addr3_msb;
static_assert((sizeof(struct reg_cbu_fork_base_addr3_msb) == 4), "reg_cbu_fork_base_addr3_msb size is not 32-bit");
/*
 CBU_FORK_ADDR_MASK3_LSB 
 b'CBU FORK: CBU RD FORK Mask Address3 LSB'
*/
typedef struct reg_cbu_fork_addr_mask3_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_addr_mask3_lsb;
static_assert((sizeof(struct reg_cbu_fork_addr_mask3_lsb) == 4), "reg_cbu_fork_addr_mask3_lsb size is not 32-bit");
/*
 CBU_FORK_ADDR_MASK3_MSB 
 b'CBU FORK: CBU RD FORK Mask Address3 MSB'
*/
typedef struct reg_cbu_fork_addr_mask3_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_addr_mask3_msb;
static_assert((sizeof(struct reg_cbu_fork_addr_mask3_msb) == 4), "reg_cbu_fork_addr_mask3_msb size is not 32-bit");
/*
 CBU_TRMINATE_ARADDR_LSB 
 b'CBU TRMINATE ARADDR ERR LSB'
*/
typedef struct reg_cbu_trminate_araddr_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_trminate_araddr_lsb;
static_assert((sizeof(struct reg_cbu_trminate_araddr_lsb) == 4), "reg_cbu_trminate_araddr_lsb size is not 32-bit");
/*
 CBU_TRMINATE_ARADDR_MSB 
 b'CBU TRMINATE ARADDR ERR MSB'
*/
typedef struct reg_cbu_trminate_araddr_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_trminate_araddr_msb;
static_assert((sizeof(struct reg_cbu_trminate_araddr_msb) == 4), "reg_cbu_trminate_araddr_msb size is not 32-bit");
/*
 CFG_CBU_TERMINATE_BRESP 
 b'CFG CBU TERMINATE BRESP VAL'
*/
typedef struct reg_cfg_cbu_terminate_bresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cfg_cbu_terminate_bresp;
static_assert((sizeof(struct reg_cfg_cbu_terminate_bresp) == 4), "reg_cfg_cbu_terminate_bresp size is not 32-bit");
/*
 CFG_CBU_TERMINATE_RRESP 
 b'CFG CBU TERMINATE RRESP VAL'
*/
typedef struct reg_cfg_cbu_terminate_rresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cfg_cbu_terminate_rresp;
static_assert((sizeof(struct reg_cfg_cbu_terminate_rresp) == 4), "reg_cfg_cbu_terminate_rresp size is not 32-bit");
/*
 ARC_REGION_CFG 
 b'ARC-REGION (x16) : MMU_BP/ASID/Protection bits'
*/
typedef struct reg_arc_region_cfg {
	union {
		struct {
			uint32_t asid : 10,
				_reserved12 : 2,
				mmu_bp : 1,
				_reserved16 : 3,
				prot_val : 3,
				_reserved20 : 1,
				prot_val_en : 3,
				_reserved23 : 9;
		};
		uint32_t _raw;
	};
} reg_arc_region_cfg;
static_assert((sizeof(struct reg_arc_region_cfg) == 4), "reg_arc_region_cfg size is not 32-bit");
/*
 DCCM_TRMINATE_AWADDR_ERR 
 b'DCCM-ERR: Address caused DCCM termination (WR)'
*/
typedef struct reg_dccm_trminate_awaddr_err {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_dccm_trminate_awaddr_err;
static_assert((sizeof(struct reg_dccm_trminate_awaddr_err) == 4), "reg_dccm_trminate_awaddr_err size is not 32-bit");
/*
 DCCM_TRMINATE_ARADDR_ERR 
 b'DCCM-ERR: Address caused DCCM termination (RD)'
*/
typedef struct reg_dccm_trminate_araddr_err {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_dccm_trminate_araddr_err;
static_assert((sizeof(struct reg_dccm_trminate_araddr_err) == 4), "reg_dccm_trminate_araddr_err size is not 32-bit");
/*
 CFG_DCCM_TERMINATE_BRESP 
 b'DCCM-ERR: DCCM BRESP error (WR)'
*/
typedef struct reg_cfg_dccm_terminate_bresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cfg_dccm_terminate_bresp;
static_assert((sizeof(struct reg_cfg_dccm_terminate_bresp) == 4), "reg_cfg_dccm_terminate_bresp size is not 32-bit");
/*
 CFG_DCCM_TERMINATE_RRESP 
 b'DCCM-ERR: DCCM RRESP error (WR)'
*/
typedef struct reg_cfg_dccm_terminate_rresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cfg_dccm_terminate_rresp;
static_assert((sizeof(struct reg_cfg_dccm_terminate_rresp) == 4), "reg_cfg_dccm_terminate_rresp size is not 32-bit");
/*
 CFG_DCCM_TERMINATE_EN 
 b'DCCM: DCCM termination enable'
*/
typedef struct reg_cfg_dccm_terminate_en {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cfg_dccm_terminate_en;
static_assert((sizeof(struct reg_cfg_dccm_terminate_en) == 4), "reg_cfg_dccm_terminate_en size is not 32-bit");
/*
 CFG_DCCM_SECURE_REGION 
 b'DCCM: Secure region limit (0 MEAN all DCCM @ USER)'
*/
typedef struct reg_cfg_dccm_secure_region {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_cfg_dccm_secure_region;
static_assert((sizeof(struct reg_cfg_dccm_secure_region) == 4), "reg_cfg_dccm_secure_region size is not 32-bit");
/*
 ARC_AXI_ORDERING_WR_IF_CNT 
 b'AXI-ORDER: Writes'
*/
typedef struct reg_arc_axi_ordering_wr_if_cnt {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_axi_ordering_wr_if_cnt;
static_assert((sizeof(struct reg_arc_axi_ordering_wr_if_cnt) == 4), "reg_arc_axi_ordering_wr_if_cnt size is not 32-bit");
/*
 ARC_AXI_ORDERING_CTL 
 b'AXI-ORDER: Enable  & delay between CTR=0 to RD'
*/
typedef struct reg_arc_axi_ordering_ctl {
	union {
		struct {
			uint32_t enable_bp : 1,
				rd_delay_cc : 5,
				_reserved6 : 26;
		};
		uint32_t _raw;
	};
} reg_arc_axi_ordering_ctl;
static_assert((sizeof(struct reg_arc_axi_ordering_ctl) == 4), "reg_arc_axi_ordering_ctl size is not 32-bit");
/*
 ARC_AXI_ORDERING_ADDR_MSK 
 b'AXI-ORDER: Selects relevant portion of the addr'
*/
typedef struct reg_arc_axi_ordering_addr_msk {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_arc_axi_ordering_addr_msk;
static_assert((sizeof(struct reg_arc_axi_ordering_addr_msk) == 4), "reg_arc_axi_ordering_addr_msk size is not 32-bit");
/*
 ARC_AXI_ORDERING_ADDR 
 b'AXI-ORDER: Addr to activate the axi reordering'
*/
typedef struct reg_arc_axi_ordering_addr {
	union {
		struct {
			uint32_t val : 27,
				_reserved27 : 5;
		};
		uint32_t _raw;
	};
} reg_arc_axi_ordering_addr;
static_assert((sizeof(struct reg_arc_axi_ordering_addr) == 4), "reg_arc_axi_ordering_addr size is not 32-bit");
/*
 ARC_ACC_ENGS_BUSER 
 b'ARC-AXI:Buser data to reply for axi trans from arc'
*/
typedef struct reg_arc_acc_engs_buser {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_arc_acc_engs_buser;
static_assert((sizeof(struct reg_arc_acc_engs_buser) == 4), "reg_arc_acc_engs_buser size is not 32-bit");
/*
 MME_ARC_UPPER_DCCM_EN 
 b'DCCM-ADDR: Select Upper/Lower DCCM @ 32K window'
*/
typedef struct reg_mme_arc_upper_dccm_en {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_mme_arc_upper_dccm_en;
static_assert((sizeof(struct reg_mme_arc_upper_dccm_en) == 4), "reg_mme_arc_upper_dccm_en size is not 32-bit");

#ifdef __cplusplus
} /* qman_arc_aux namespace */
#endif

/*
 QMAN_ARC_AUX block
*/

#ifdef __cplusplus

struct block_qman_arc_aux {
	uint32_t _pad0[64];
	struct qman_arc_aux::reg_run_halt_req run_halt_req;
	struct qman_arc_aux::reg_run_halt_ack run_halt_ack;
	struct qman_arc_aux::reg_rst_vec_addr rst_vec_addr;
	struct qman_arc_aux::reg_dbg_mode dbg_mode;
	struct qman_arc_aux::reg_cluster_num cluster_num;
	struct qman_arc_aux::reg_arc_num arc_num;
	struct qman_arc_aux::reg_wake_up_event wake_up_event;
	struct qman_arc_aux::reg_dccm_sys_addr_base dccm_sys_addr_base;
	struct qman_arc_aux::reg_cti_ap_sts cti_ap_sts;
	struct qman_arc_aux::reg_cti_cfg_mux_sel cti_cfg_mux_sel;
	struct qman_arc_aux::reg_arc_rst arc_rst;
	struct qman_arc_aux::reg_arc_rst_req arc_rst_req;
	struct qman_arc_aux::reg_sram_lsb_addr sram_lsb_addr;
	struct qman_arc_aux::reg_sram_msb_addr sram_msb_addr;
	struct qman_arc_aux::reg_pcie_lsb_addr pcie_lsb_addr;
	struct qman_arc_aux::reg_pcie_msb_addr pcie_msb_addr;
	struct qman_arc_aux::reg_cfg_lsb_addr cfg_lsb_addr;
	struct qman_arc_aux::reg_cfg_msb_addr cfg_msb_addr;
	uint32_t _pad328[2];
	struct qman_arc_aux::reg_hbm0_lsb_addr hbm0_lsb_addr;
	struct qman_arc_aux::reg_hbm0_msb_addr hbm0_msb_addr;
	struct qman_arc_aux::reg_hbm1_lsb_addr hbm1_lsb_addr;
	struct qman_arc_aux::reg_hbm1_msb_addr hbm1_msb_addr;
	struct qman_arc_aux::reg_hbm2_lsb_addr hbm2_lsb_addr;
	struct qman_arc_aux::reg_hbm2_msb_addr hbm2_msb_addr;
	struct qman_arc_aux::reg_hbm3_lsb_addr hbm3_lsb_addr;
	struct qman_arc_aux::reg_hbm3_msb_addr hbm3_msb_addr;
	struct qman_arc_aux::reg_hbm0_offset hbm0_offset;
	struct qman_arc_aux::reg_hbm1_offset hbm1_offset;
	struct qman_arc_aux::reg_hbm2_offset hbm2_offset;
	struct qman_arc_aux::reg_hbm3_offset hbm3_offset;
	struct qman_arc_aux::reg_general_purpose_lsb_addr general_purpose_lsb_addr[7];
	struct qman_arc_aux::reg_general_purpose_msb_addr general_purpose_msb_addr[7];
	struct qman_arc_aux::reg_arc_cbu_awcache_ovr arc_cbu_awcache_ovr;
	struct qman_arc_aux::reg_arc_lbu_awcache_ovr arc_lbu_awcache_ovr;
	struct qman_arc_aux::reg_context_id context_id[8];
	struct qman_arc_aux::reg_cid_offset cid_offset[8];
	struct qman_arc_aux::reg_sw_intr sw_intr[16];
	uint32_t _pad576[16];
	struct qman_arc_aux::reg_irq_intr_mask irq_intr_mask[2];
	uint32_t _pad648[2];
	struct qman_arc_aux::reg_arc_sei_intr_sts arc_sei_intr_sts;
	struct qman_arc_aux::reg_arc_sei_intr_clr arc_sei_intr_clr;
	struct qman_arc_aux::reg_arc_sei_intr_mask arc_sei_intr_mask;
	struct qman_arc_aux::reg_arc_excptn_cause arc_excptn_cause;
	struct qman_arc_aux::reg_sei_intr_halt_en sei_intr_halt_en;
	struct qman_arc_aux::reg_arc_sei_intr_halt_mask arc_sei_intr_halt_mask;
	struct qman_arc_aux::reg_qman_sei_intr_halt_mask qman_sei_intr_halt_mask;
	uint32_t _pad684[1];
	struct qman_arc_aux::reg_arc_rei_intr_sts arc_rei_intr_sts;
	struct qman_arc_aux::reg_arc_rei_intr_clr arc_rei_intr_clr;
	struct qman_arc_aux::reg_arc_rei_intr_mask arc_rei_intr_mask;
	struct qman_arc_aux::reg_dccm_ecc_err_addr dccm_ecc_err_addr;
	struct qman_arc_aux::reg_dccm_ecc_syndrome dccm_ecc_syndrome;
	struct qman_arc_aux::reg_i_cache_ecc_err_addr i_cache_ecc_err_addr;
	struct qman_arc_aux::reg_i_cache_ecc_syndrome i_cache_ecc_syndrome;
	struct qman_arc_aux::reg_d_cache_ecc_err_addr d_cache_ecc_err_addr;
	struct qman_arc_aux::reg_d_cache_ecc_syndrome d_cache_ecc_syndrome;
	uint32_t _pad724[3];
	struct qman_arc_aux::reg_lbw_trminate_awaddr_err lbw_trminate_awaddr_err;
	struct qman_arc_aux::reg_lbw_trminate_araddr_err lbw_trminate_araddr_err;
	struct qman_arc_aux::reg_cfg_lbw_terminate_bresp cfg_lbw_terminate_bresp;
	struct qman_arc_aux::reg_cfg_lbw_terminate_rresp cfg_lbw_terminate_rresp;
	struct qman_arc_aux::reg_cfg_lbw_terminate_axlen cfg_lbw_terminate_axlen;
	struct qman_arc_aux::reg_cfg_lbw_terminate_axsize cfg_lbw_terminate_axsize;
	uint32_t _pad760[2];
	struct qman_arc_aux::reg_scratchpad scratchpad[8];
	struct qman_arc_aux::reg_total_cbu_wr_cnt total_cbu_wr_cnt;
	struct qman_arc_aux::reg_inflight_cbu_wr_cnt inflight_cbu_wr_cnt;
	struct qman_arc_aux::reg_total_cbu_rd_cnt total_cbu_rd_cnt;
	struct qman_arc_aux::reg_inflight_cbu_rd_cnt inflight_cbu_rd_cnt;
	struct qman_arc_aux::reg_total_lbu_wr_cnt total_lbu_wr_cnt;
	struct qman_arc_aux::reg_inflight_lbu_wr_cnt inflight_lbu_wr_cnt;
	struct qman_arc_aux::reg_total_lbu_rd_cnt total_lbu_rd_cnt;
	struct qman_arc_aux::reg_inflight_lbu_rd_cnt inflight_lbu_rd_cnt;
	uint32_t _pad832[4];
	struct qman_arc_aux::reg_cbu_aruser_ovr cbu_aruser_ovr;
	struct qman_arc_aux::reg_cbu_aruser_ovr_en cbu_aruser_ovr_en;
	struct qman_arc_aux::reg_cbu_awuser_ovr cbu_awuser_ovr;
	struct qman_arc_aux::reg_cbu_awuser_ovr_en cbu_awuser_ovr_en;
	struct qman_arc_aux::reg_cbu_aruser_msb_ovr cbu_aruser_msb_ovr;
	struct qman_arc_aux::reg_cbu_aruser_msb_ovr_en cbu_aruser_msb_ovr_en;
	struct qman_arc_aux::reg_cbu_awuser_msb_ovr cbu_awuser_msb_ovr;
	struct qman_arc_aux::reg_cbu_awuser_msb_ovr_en cbu_awuser_msb_ovr_en;
	struct qman_arc_aux::reg_cbu_axcache_ovr cbu_axcache_ovr;
	struct qman_arc_aux::reg_cbu_lock_ovr cbu_lock_ovr;
	struct qman_arc_aux::reg_cbu_prot_ovr cbu_prot_ovr;
	struct qman_arc_aux::reg_cbu_max_outstanding cbu_max_outstanding;
	struct qman_arc_aux::reg_cbu_early_bresp_en cbu_early_bresp_en;
	struct qman_arc_aux::reg_cbu_force_rsp_ok cbu_force_rsp_ok;
	uint32_t _pad904[1];
	struct qman_arc_aux::reg_cbu_no_wr_inflight cbu_no_wr_inflight;
	struct qman_arc_aux::reg_cbu_sei_intr_id cbu_sei_intr_id;
	uint32_t _pad916[27];
	struct qman_arc_aux::reg_lbu_aruser_ovr lbu_aruser_ovr;
	struct qman_arc_aux::reg_lbu_aruser_ovr_en lbu_aruser_ovr_en;
	struct qman_arc_aux::reg_lbu_awuser_ovr lbu_awuser_ovr;
	struct qman_arc_aux::reg_lbu_awuser_ovr_en lbu_awuser_ovr_en;
	uint32_t _pad1040[4];
	struct qman_arc_aux::reg_lbu_axcache_ovr lbu_axcache_ovr;
	struct qman_arc_aux::reg_lbu_lock_ovr lbu_lock_ovr;
	struct qman_arc_aux::reg_lbu_prot_ovr lbu_prot_ovr;
	struct qman_arc_aux::reg_lbu_max_outstanding lbu_max_outstanding;
	struct qman_arc_aux::reg_lbu_early_bresp_en lbu_early_bresp_en;
	struct qman_arc_aux::reg_lbu_force_rsp_ok lbu_force_rsp_ok;
	uint32_t _pad1080[1];
	struct qman_arc_aux::reg_lbu_no_wr_inflight lbu_no_wr_inflight;
	struct qman_arc_aux::reg_lbu_sei_intr_id lbu_sei_intr_id;
	uint32_t _pad1092[47];
	struct qman_arc_aux::reg_dccm_queue_base_addr dccm_queue_base_addr[8];
	struct qman_arc_aux::reg_dccm_queue_size dccm_queue_size[8];
	struct qman_arc_aux::reg_dccm_queue_pi dccm_queue_pi[8];
	struct qman_arc_aux::reg_dccm_queue_ci dccm_queue_ci[8];
	struct qman_arc_aux::reg_dccm_queue_push_reg dccm_queue_push_reg[8];
	struct qman_arc_aux::reg_dccm_queue_max_occupancy dccm_queue_max_occupancy[8];
	struct qman_arc_aux::reg_dccm_queue_valid_entries dccm_queue_valid_entries[8];
	struct qman_arc_aux::reg_general_q_vld_entry_mask general_q_vld_entry_mask;
	struct qman_arc_aux::reg_nic_q_vld_entry_mask nic_q_vld_entry_mask;
	uint32_t _pad1512[14];
	struct qman_arc_aux::reg_dccm_queue_drop_en dccm_queue_drop_en;
	struct qman_arc_aux::reg_dccm_queue_warn_msg dccm_queue_warn_msg;
	struct qman_arc_aux::reg_dccm_queue_alert_msg dccm_queue_alert_msg;
	uint32_t _pad1580[1];
	struct qman_arc_aux::reg_dccm_gen_axi_awprot dccm_gen_axi_awprot;
	struct qman_arc_aux::reg_dccm_gen_axi_awuser dccm_gen_axi_awuser;
	struct qman_arc_aux::reg_dccm_gen_axi_awburst dccm_gen_axi_awburst;
	struct qman_arc_aux::reg_dccm_gen_axi_awlock dccm_gen_axi_awlock;
	struct qman_arc_aux::reg_dccm_gen_axi_awcache dccm_gen_axi_awcache;
	struct qman_arc_aux::reg_dccm_wrr_arb_weight dccm_wrr_arb_weight;
	struct qman_arc_aux::reg_dccm_q_push_fifo_full_cfg dccm_q_push_fifo_full_cfg;
	struct qman_arc_aux::reg_dccm_q_push_fifo_cnt dccm_q_push_fifo_cnt;
	struct qman_arc_aux::reg_qman_cq_ififo_shadow_ci qman_cq_ififo_shadow_ci;
	struct qman_arc_aux::reg_qman_arc_cq_ififo_shadow_ci qman_arc_cq_ififo_shadow_ci;
	struct qman_arc_aux::reg_qman_cq_shadow_ci qman_cq_shadow_ci;
	struct qman_arc_aux::reg_qman_arc_cq_shadow_ci qman_arc_cq_shadow_ci;
	uint32_t _pad1632[40];
	struct qman_arc_aux::reg_aux2apb_prot aux2apb_prot;
	struct qman_arc_aux::reg_lbw_fork_win_en lbw_fork_win_en;
	struct qman_arc_aux::reg_qman_lbw_fork_base_addr0 qman_lbw_fork_base_addr0;
	struct qman_arc_aux::reg_qman_lbw_fork_addr_mask0 qman_lbw_fork_addr_mask0;
	struct qman_arc_aux::reg_qman_lbw_fork_base_addr1 qman_lbw_fork_base_addr1;
	struct qman_arc_aux::reg_qman_lbw_fork_addr_mask1 qman_lbw_fork_addr_mask1;
	struct qman_arc_aux::reg_farm_lbw_fork_base_addr0 farm_lbw_fork_base_addr0;
	struct qman_arc_aux::reg_farm_lbw_fork_addr_mask0 farm_lbw_fork_addr_mask0;
	struct qman_arc_aux::reg_farm_lbw_fork_base_addr1 farm_lbw_fork_base_addr1;
	struct qman_arc_aux::reg_farm_lbw_fork_addr_mask1 farm_lbw_fork_addr_mask1;
	struct qman_arc_aux::reg_lbw_apb_fork_max_addr0 lbw_apb_fork_max_addr0;
	struct qman_arc_aux::reg_lbw_apb_fork_max_addr1 lbw_apb_fork_max_addr1;
	struct qman_arc_aux::reg_arc_acc_engs_lbw_fork_mask arc_acc_engs_lbw_fork_mask;
	struct qman_arc_aux::reg_arc_dup_eng_lbw_fork_addr arc_dup_eng_lbw_fork_addr;
	struct qman_arc_aux::reg_arc_acp_eng_lbw_fork_addr arc_acp_eng_lbw_fork_addr;
	struct qman_arc_aux::reg_arc_acc_engs_virtual_addr arc_acc_engs_virtual_addr;
	struct qman_arc_aux::reg_cbu_fork_win_en cbu_fork_win_en;
	uint32_t _pad1860[3];
	struct qman_arc_aux::reg_cbu_fork_base_addr0_lsb cbu_fork_base_addr0_lsb;
	struct qman_arc_aux::reg_cbu_fork_base_addr0_msb cbu_fork_base_addr0_msb;
	struct qman_arc_aux::reg_cbu_fork_addr_mask0_lsb cbu_fork_addr_mask0_lsb;
	struct qman_arc_aux::reg_cbu_fork_addr_mask0_msb cbu_fork_addr_mask0_msb;
	struct qman_arc_aux::reg_cbu_fork_base_addr1_lsb cbu_fork_base_addr1_lsb;
	struct qman_arc_aux::reg_cbu_fork_base_addr1_msb cbu_fork_base_addr1_msb;
	struct qman_arc_aux::reg_cbu_fork_addr_mask1_lsb cbu_fork_addr_mask1_lsb;
	struct qman_arc_aux::reg_cbu_fork_addr_mask1_msb cbu_fork_addr_mask1_msb;
	struct qman_arc_aux::reg_cbu_fork_base_addr2_lsb cbu_fork_base_addr2_lsb;
	struct qman_arc_aux::reg_cbu_fork_base_addr2_msb cbu_fork_base_addr2_msb;
	struct qman_arc_aux::reg_cbu_fork_addr_mask2_lsb cbu_fork_addr_mask2_lsb;
	struct qman_arc_aux::reg_cbu_fork_addr_mask2_msb cbu_fork_addr_mask2_msb;
	struct qman_arc_aux::reg_cbu_fork_base_addr3_lsb cbu_fork_base_addr3_lsb;
	struct qman_arc_aux::reg_cbu_fork_base_addr3_msb cbu_fork_base_addr3_msb;
	struct qman_arc_aux::reg_cbu_fork_addr_mask3_lsb cbu_fork_addr_mask3_lsb;
	struct qman_arc_aux::reg_cbu_fork_addr_mask3_msb cbu_fork_addr_mask3_msb;
	struct qman_arc_aux::reg_cbu_trminate_araddr_lsb cbu_trminate_araddr_lsb;
	struct qman_arc_aux::reg_cbu_trminate_araddr_msb cbu_trminate_araddr_msb;
	struct qman_arc_aux::reg_cfg_cbu_terminate_bresp cfg_cbu_terminate_bresp;
	struct qman_arc_aux::reg_cfg_cbu_terminate_rresp cfg_cbu_terminate_rresp;
	uint32_t _pad1952[24];
	struct qman_arc_aux::reg_arc_region_cfg arc_region_cfg[16];
	struct qman_arc_aux::reg_dccm_trminate_awaddr_err dccm_trminate_awaddr_err;
	struct qman_arc_aux::reg_dccm_trminate_araddr_err dccm_trminate_araddr_err;
	struct qman_arc_aux::reg_cfg_dccm_terminate_bresp cfg_dccm_terminate_bresp;
	struct qman_arc_aux::reg_cfg_dccm_terminate_rresp cfg_dccm_terminate_rresp;
	struct qman_arc_aux::reg_cfg_dccm_terminate_en cfg_dccm_terminate_en;
	struct qman_arc_aux::reg_cfg_dccm_secure_region cfg_dccm_secure_region;
	uint32_t _pad2136[42];
	struct qman_arc_aux::reg_arc_axi_ordering_wr_if_cnt arc_axi_ordering_wr_if_cnt;
	struct qman_arc_aux::reg_arc_axi_ordering_ctl arc_axi_ordering_ctl;
	struct qman_arc_aux::reg_arc_axi_ordering_addr_msk arc_axi_ordering_addr_msk;
	struct qman_arc_aux::reg_arc_axi_ordering_addr arc_axi_ordering_addr;
	struct qman_arc_aux::reg_arc_acc_engs_buser arc_acc_engs_buser;
	uint32_t _pad2324[3];
	struct qman_arc_aux::reg_mme_arc_upper_dccm_en mme_arc_upper_dccm_en;
	uint32_t _pad2340[343];
	struct block_special_regs special;
};
#else

typedef struct block_qman_arc_aux {
	uint32_t _pad0[64];
	reg_run_halt_req run_halt_req;
	reg_run_halt_ack run_halt_ack;
	reg_rst_vec_addr rst_vec_addr;
	reg_dbg_mode dbg_mode;
	reg_cluster_num cluster_num;
	reg_arc_num arc_num;
	reg_wake_up_event wake_up_event;
	reg_dccm_sys_addr_base dccm_sys_addr_base;
	reg_cti_ap_sts cti_ap_sts;
	reg_cti_cfg_mux_sel cti_cfg_mux_sel;
	reg_arc_rst arc_rst;
	reg_arc_rst_req arc_rst_req;
	reg_sram_lsb_addr sram_lsb_addr;
	reg_sram_msb_addr sram_msb_addr;
	reg_pcie_lsb_addr pcie_lsb_addr;
	reg_pcie_msb_addr pcie_msb_addr;
	reg_cfg_lsb_addr cfg_lsb_addr;
	reg_cfg_msb_addr cfg_msb_addr;
	uint32_t _pad328[2];
	reg_hbm0_lsb_addr hbm0_lsb_addr;
	reg_hbm0_msb_addr hbm0_msb_addr;
	reg_hbm1_lsb_addr hbm1_lsb_addr;
	reg_hbm1_msb_addr hbm1_msb_addr;
	reg_hbm2_lsb_addr hbm2_lsb_addr;
	reg_hbm2_msb_addr hbm2_msb_addr;
	reg_hbm3_lsb_addr hbm3_lsb_addr;
	reg_hbm3_msb_addr hbm3_msb_addr;
	reg_hbm0_offset hbm0_offset;
	reg_hbm1_offset hbm1_offset;
	reg_hbm2_offset hbm2_offset;
	reg_hbm3_offset hbm3_offset;
	reg_general_purpose_lsb_addr general_purpose_lsb_addr[7];
	reg_general_purpose_msb_addr general_purpose_msb_addr[7];
	reg_arc_cbu_awcache_ovr arc_cbu_awcache_ovr;
	reg_arc_lbu_awcache_ovr arc_lbu_awcache_ovr;
	reg_context_id context_id[8];
	reg_cid_offset cid_offset[8];
	reg_sw_intr sw_intr[16];
	uint32_t _pad576[16];
	reg_irq_intr_mask irq_intr_mask[2];
	uint32_t _pad648[2];
	reg_arc_sei_intr_sts arc_sei_intr_sts;
	reg_arc_sei_intr_clr arc_sei_intr_clr;
	reg_arc_sei_intr_mask arc_sei_intr_mask;
	reg_arc_excptn_cause arc_excptn_cause;
	reg_sei_intr_halt_en sei_intr_halt_en;
	reg_arc_sei_intr_halt_mask arc_sei_intr_halt_mask;
	reg_qman_sei_intr_halt_mask qman_sei_intr_halt_mask;
	uint32_t _pad684[1];
	reg_arc_rei_intr_sts arc_rei_intr_sts;
	reg_arc_rei_intr_clr arc_rei_intr_clr;
	reg_arc_rei_intr_mask arc_rei_intr_mask;
	reg_dccm_ecc_err_addr dccm_ecc_err_addr;
	reg_dccm_ecc_syndrome dccm_ecc_syndrome;
	reg_i_cache_ecc_err_addr i_cache_ecc_err_addr;
	reg_i_cache_ecc_syndrome i_cache_ecc_syndrome;
	reg_d_cache_ecc_err_addr d_cache_ecc_err_addr;
	reg_d_cache_ecc_syndrome d_cache_ecc_syndrome;
	uint32_t _pad724[3];
	reg_lbw_trminate_awaddr_err lbw_trminate_awaddr_err;
	reg_lbw_trminate_araddr_err lbw_trminate_araddr_err;
	reg_cfg_lbw_terminate_bresp cfg_lbw_terminate_bresp;
	reg_cfg_lbw_terminate_rresp cfg_lbw_terminate_rresp;
	reg_cfg_lbw_terminate_axlen cfg_lbw_terminate_axlen;
	reg_cfg_lbw_terminate_axsize cfg_lbw_terminate_axsize;
	uint32_t _pad760[2];
	reg_scratchpad scratchpad[8];
	reg_total_cbu_wr_cnt total_cbu_wr_cnt;
	reg_inflight_cbu_wr_cnt inflight_cbu_wr_cnt;
	reg_total_cbu_rd_cnt total_cbu_rd_cnt;
	reg_inflight_cbu_rd_cnt inflight_cbu_rd_cnt;
	reg_total_lbu_wr_cnt total_lbu_wr_cnt;
	reg_inflight_lbu_wr_cnt inflight_lbu_wr_cnt;
	reg_total_lbu_rd_cnt total_lbu_rd_cnt;
	reg_inflight_lbu_rd_cnt inflight_lbu_rd_cnt;
	uint32_t _pad832[4];
	reg_cbu_aruser_ovr cbu_aruser_ovr;
	reg_cbu_aruser_ovr_en cbu_aruser_ovr_en;
	reg_cbu_awuser_ovr cbu_awuser_ovr;
	reg_cbu_awuser_ovr_en cbu_awuser_ovr_en;
	reg_cbu_aruser_msb_ovr cbu_aruser_msb_ovr;
	reg_cbu_aruser_msb_ovr_en cbu_aruser_msb_ovr_en;
	reg_cbu_awuser_msb_ovr cbu_awuser_msb_ovr;
	reg_cbu_awuser_msb_ovr_en cbu_awuser_msb_ovr_en;
	reg_cbu_axcache_ovr cbu_axcache_ovr;
	reg_cbu_lock_ovr cbu_lock_ovr;
	reg_cbu_prot_ovr cbu_prot_ovr;
	reg_cbu_max_outstanding cbu_max_outstanding;
	reg_cbu_early_bresp_en cbu_early_bresp_en;
	reg_cbu_force_rsp_ok cbu_force_rsp_ok;
	uint32_t _pad904[1];
	reg_cbu_no_wr_inflight cbu_no_wr_inflight;
	reg_cbu_sei_intr_id cbu_sei_intr_id;
	uint32_t _pad916[27];
	reg_lbu_aruser_ovr lbu_aruser_ovr;
	reg_lbu_aruser_ovr_en lbu_aruser_ovr_en;
	reg_lbu_awuser_ovr lbu_awuser_ovr;
	reg_lbu_awuser_ovr_en lbu_awuser_ovr_en;
	uint32_t _pad1040[4];
	reg_lbu_axcache_ovr lbu_axcache_ovr;
	reg_lbu_lock_ovr lbu_lock_ovr;
	reg_lbu_prot_ovr lbu_prot_ovr;
	reg_lbu_max_outstanding lbu_max_outstanding;
	reg_lbu_early_bresp_en lbu_early_bresp_en;
	reg_lbu_force_rsp_ok lbu_force_rsp_ok;
	uint32_t _pad1080[1];
	reg_lbu_no_wr_inflight lbu_no_wr_inflight;
	reg_lbu_sei_intr_id lbu_sei_intr_id;
	uint32_t _pad1092[47];
	reg_dccm_queue_base_addr dccm_queue_base_addr[8];
	reg_dccm_queue_size dccm_queue_size[8];
	reg_dccm_queue_pi dccm_queue_pi[8];
	reg_dccm_queue_ci dccm_queue_ci[8];
	reg_dccm_queue_push_reg dccm_queue_push_reg[8];
	reg_dccm_queue_max_occupancy dccm_queue_max_occupancy[8];
	reg_dccm_queue_valid_entries dccm_queue_valid_entries[8];
	reg_general_q_vld_entry_mask general_q_vld_entry_mask;
	reg_nic_q_vld_entry_mask nic_q_vld_entry_mask;
	uint32_t _pad1512[14];
	reg_dccm_queue_drop_en dccm_queue_drop_en;
	reg_dccm_queue_warn_msg dccm_queue_warn_msg;
	reg_dccm_queue_alert_msg dccm_queue_alert_msg;
	uint32_t _pad1580[1];
	reg_dccm_gen_axi_awprot dccm_gen_axi_awprot;
	reg_dccm_gen_axi_awuser dccm_gen_axi_awuser;
	reg_dccm_gen_axi_awburst dccm_gen_axi_awburst;
	reg_dccm_gen_axi_awlock dccm_gen_axi_awlock;
	reg_dccm_gen_axi_awcache dccm_gen_axi_awcache;
	reg_dccm_wrr_arb_weight dccm_wrr_arb_weight;
	reg_dccm_q_push_fifo_full_cfg dccm_q_push_fifo_full_cfg;
	reg_dccm_q_push_fifo_cnt dccm_q_push_fifo_cnt;
	reg_qman_cq_ififo_shadow_ci qman_cq_ififo_shadow_ci;
	reg_qman_arc_cq_ififo_shadow_ci qman_arc_cq_ififo_shadow_ci;
	reg_qman_cq_shadow_ci qman_cq_shadow_ci;
	reg_qman_arc_cq_shadow_ci qman_arc_cq_shadow_ci;
	uint32_t _pad1632[40];
	reg_aux2apb_prot aux2apb_prot;
	reg_lbw_fork_win_en lbw_fork_win_en;
	reg_qman_lbw_fork_base_addr0 qman_lbw_fork_base_addr0;
	reg_qman_lbw_fork_addr_mask0 qman_lbw_fork_addr_mask0;
	reg_qman_lbw_fork_base_addr1 qman_lbw_fork_base_addr1;
	reg_qman_lbw_fork_addr_mask1 qman_lbw_fork_addr_mask1;
	reg_farm_lbw_fork_base_addr0 farm_lbw_fork_base_addr0;
	reg_farm_lbw_fork_addr_mask0 farm_lbw_fork_addr_mask0;
	reg_farm_lbw_fork_base_addr1 farm_lbw_fork_base_addr1;
	reg_farm_lbw_fork_addr_mask1 farm_lbw_fork_addr_mask1;
	reg_lbw_apb_fork_max_addr0 lbw_apb_fork_max_addr0;
	reg_lbw_apb_fork_max_addr1 lbw_apb_fork_max_addr1;
	reg_arc_acc_engs_lbw_fork_mask arc_acc_engs_lbw_fork_mask;
	reg_arc_dup_eng_lbw_fork_addr arc_dup_eng_lbw_fork_addr;
	reg_arc_acp_eng_lbw_fork_addr arc_acp_eng_lbw_fork_addr;
	reg_arc_acc_engs_virtual_addr arc_acc_engs_virtual_addr;
	reg_cbu_fork_win_en cbu_fork_win_en;
	uint32_t _pad1860[3];
	reg_cbu_fork_base_addr0_lsb cbu_fork_base_addr0_lsb;
	reg_cbu_fork_base_addr0_msb cbu_fork_base_addr0_msb;
	reg_cbu_fork_addr_mask0_lsb cbu_fork_addr_mask0_lsb;
	reg_cbu_fork_addr_mask0_msb cbu_fork_addr_mask0_msb;
	reg_cbu_fork_base_addr1_lsb cbu_fork_base_addr1_lsb;
	reg_cbu_fork_base_addr1_msb cbu_fork_base_addr1_msb;
	reg_cbu_fork_addr_mask1_lsb cbu_fork_addr_mask1_lsb;
	reg_cbu_fork_addr_mask1_msb cbu_fork_addr_mask1_msb;
	reg_cbu_fork_base_addr2_lsb cbu_fork_base_addr2_lsb;
	reg_cbu_fork_base_addr2_msb cbu_fork_base_addr2_msb;
	reg_cbu_fork_addr_mask2_lsb cbu_fork_addr_mask2_lsb;
	reg_cbu_fork_addr_mask2_msb cbu_fork_addr_mask2_msb;
	reg_cbu_fork_base_addr3_lsb cbu_fork_base_addr3_lsb;
	reg_cbu_fork_base_addr3_msb cbu_fork_base_addr3_msb;
	reg_cbu_fork_addr_mask3_lsb cbu_fork_addr_mask3_lsb;
	reg_cbu_fork_addr_mask3_msb cbu_fork_addr_mask3_msb;
	reg_cbu_trminate_araddr_lsb cbu_trminate_araddr_lsb;
	reg_cbu_trminate_araddr_msb cbu_trminate_araddr_msb;
	reg_cfg_cbu_terminate_bresp cfg_cbu_terminate_bresp;
	reg_cfg_cbu_terminate_rresp cfg_cbu_terminate_rresp;
	uint32_t _pad1952[24];
	reg_arc_region_cfg arc_region_cfg[16];
	reg_dccm_trminate_awaddr_err dccm_trminate_awaddr_err;
	reg_dccm_trminate_araddr_err dccm_trminate_araddr_err;
	reg_cfg_dccm_terminate_bresp cfg_dccm_terminate_bresp;
	reg_cfg_dccm_terminate_rresp cfg_dccm_terminate_rresp;
	reg_cfg_dccm_terminate_en cfg_dccm_terminate_en;
	reg_cfg_dccm_secure_region cfg_dccm_secure_region;
	uint32_t _pad2136[42];
	reg_arc_axi_ordering_wr_if_cnt arc_axi_ordering_wr_if_cnt;
	reg_arc_axi_ordering_ctl arc_axi_ordering_ctl;
	reg_arc_axi_ordering_addr_msk arc_axi_ordering_addr_msk;
	reg_arc_axi_ordering_addr arc_axi_ordering_addr;
	reg_arc_acc_engs_buser arc_acc_engs_buser;
	uint32_t _pad2324[3];
	reg_mme_arc_upper_dccm_en mme_arc_upper_dccm_en;
	uint32_t _pad2340[343];
	block_special_regs special;
} block_qman_arc_aux;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_qman_arc_aux_defaults[] =
{
	// offset	// value
	{ 0x108 , 0x9ff00             , 1 }, // rst_vec_addr
	{ 0x10c , 0x10111             , 1 }, // dbg_mode
	{ 0x11c , 0x80000000          , 1 }, // dccm_sys_addr_base
	{ 0x128 , 0x1                 , 1 }, // arc_rst
	{ 0x130 , 0x3f                , 1 }, // sram_lsb_addr
	{ 0x134 , 0x1000ffff          , 1 }, // sram_msb_addr
	{ 0x140 , 0xf                 , 1 }, // cfg_lsb_addr
	{ 0x144 , 0x1000007f          , 1 }, // cfg_msb_addr
	{ 0x154 , 0x10010000          , 1 }, // hbm0_msb_addr
	{ 0x15c , 0x10010000          , 1 }, // hbm1_msb_addr
	{ 0x164 , 0x10010000          , 1 }, // hbm2_msb_addr
	{ 0x16c , 0x10010000          , 1 }, // hbm3_msb_addr
	{ 0x2a0 , 0x1                 , 1 }, // sei_intr_halt_en
	{ 0x2a4 , 0x3fff              , 1 }, // arc_sei_intr_halt_mask
	{ 0x2a8 , 0x3fff              , 1 }, // qman_sei_intr_halt_mask
	{ 0x2e8 , 0x3                 , 1 }, // cfg_lbw_terminate_bresp
	{ 0x2ec , 0x3                 , 1 }, // cfg_lbw_terminate_rresp
	{ 0x2f4 , 0x2                 , 1 }, // cfg_lbw_terminate_axsize
	{ 0x350 , 0x40000c00          , 1 }, // cbu_aruser_ovr
	{ 0x358 , 0x40000c00          , 1 }, // cbu_awuser_ovr
	{ 0x378 , 0x7711              , 1 }, // cbu_prot_ovr
	{ 0x37c , 0x2007              , 1 }, // cbu_max_outstanding
	{ 0x400 , 0x40000c00          , 1 }, // lbu_aruser_ovr
	{ 0x408 , 0x40000c00          , 1 }, // lbu_awuser_ovr
	{ 0x428 , 0x7711              , 1 }, // lbu_prot_ovr
	{ 0x42c , 0xff04              , 1 }, // lbu_max_outstanding
	{ 0x624 , 0xabcd              , 1 }, // dccm_queue_warn_msg
	{ 0x628 , 0xefba              , 1 }, // dccm_queue_alert_msg
	{ 0x644 , 0xf0                , 1 }, // dccm_wrr_arb_weight
	{ 0x648 , 0xc                 , 1 }, // dccm_q_push_fifo_full_cfg
	{ 0x700 , 0x2                 , 1 }, // aux2apb_prot
	{ 0x704 , 0x3                 , 1 }, // lbw_fork_win_en
	{ 0x70c , 0x8000              , 1 }, // qman_lbw_fork_addr_mask0
	{ 0x710 , 0x8000              , 1 }, // qman_lbw_fork_base_addr1
	{ 0x714 , 0x8000              , 1 }, // qman_lbw_fork_addr_mask1
	{ 0x718 , 0x90000             , 1 }, // farm_lbw_fork_base_addr0
	{ 0x71c , 0x90000             , 1 }, // farm_lbw_fork_addr_mask0
	{ 0x720 , 0x88000             , 1 }, // farm_lbw_fork_base_addr1
	{ 0x724 , 0x98000             , 1 }, // farm_lbw_fork_addr_mask1
	{ 0x728 , 0x7fff              , 1 }, // lbw_apb_fork_max_addr0
	{ 0x72c , 0x8fff              , 1 }, // lbw_apb_fork_max_addr1
	{ 0x730 , 0x7fff000           , 1 }, // arc_acc_engs_lbw_fork_mask
	{ 0x734 , 0x9000              , 1 }, // arc_dup_eng_lbw_fork_addr
	{ 0x738 , 0xf000              , 1 }, // arc_acp_eng_lbw_fork_addr
	{ 0x73c , 0x4a00000           , 1 }, // arc_acc_engs_virtual_addr
	{ 0x740 , 0x3                 , 1 }, // cbu_fork_win_en
	{ 0x750 , 0xfc000000          , 1 }, // cbu_fork_base_addr0_lsb
	{ 0x754 , 0x1000ffff          , 1 }, // cbu_fork_base_addr0_msb
	{ 0x758 , 0xff000000          , 1 }, // cbu_fork_addr_mask0_lsb
	{ 0x75c , 0xffffffff          , 1 }, // cbu_fork_addr_mask0_msb
	{ 0x760 , 0xf0000000          , 1 }, // cbu_fork_base_addr1_lsb
	{ 0x764 , 0x1000007f          , 1 }, // cbu_fork_base_addr1_msb
	{ 0x768 , 0xf0000000          , 1 }, // cbu_fork_addr_mask1_lsb
	{ 0x76c , 0xffffffff          , 1 }, // cbu_fork_addr_mask1_msb
	{ 0x778 , 0xffffffff          , 1 }, // cbu_fork_addr_mask2_lsb
	{ 0x77c , 0xffffffff          , 1 }, // cbu_fork_addr_mask2_msb
	{ 0x788 , 0xffffffff          , 1 }, // cbu_fork_addr_mask3_lsb
	{ 0x78c , 0xffffffff          , 1 }, // cbu_fork_addr_mask3_msb
	{ 0x798 , 0x3                 , 1 }, // cfg_cbu_terminate_bresp
	{ 0x79c , 0x3                 , 1 }, // cfg_cbu_terminate_rresp
	{ 0x848 , 0x3                 , 1 }, // cfg_dccm_terminate_bresp
	{ 0x84c , 0x3                 , 1 }, // cfg_dccm_terminate_rresp
	{ 0x850 , 0x1                 , 1 }, // cfg_dccm_terminate_en
	{ 0x904 , 0x1                 , 1 }, // arc_axi_ordering_ctl
	{ 0x908 , 0x7fc0000           , 1 }, // arc_axi_ordering_addr_msk
	{ 0x90c , 0x4a00000           , 1 }, // arc_axi_ordering_addr
	{ 0x910 , 0x1                 , 1 }, // arc_acc_engs_buser
	{ 0xe80 , 0xffffffff          , 32 }, // glbl_priv
	{ 0xf24 , 0xffff              , 1 }, // mem_ecc_err_addr
	{ 0xf44 , 0xffffffff          , 1 }, // glbl_err_addr
	{ 0xf80 , 0xffffffff          , 32 }, // glbl_sec
};
#endif

#ifdef __cplusplus
} /* gaudi2 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI2_QMAN_ARC_AUX_H_ */
