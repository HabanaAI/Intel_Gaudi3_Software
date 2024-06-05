/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_QMAN_ARC_AUX_H_
#define ASIC_REG_STRUCTS_GAUDI3_QMAN_ARC_AUX_H_

#include <stdint.h>
#include "gaudi3_types.h"
#include "special_regs_regs.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
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
 RUN_REQ 
 b'ARC RUN Request'
*/
typedef struct reg_run_req {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_run_req;
static_assert((sizeof(struct reg_run_req) == 4), "reg_run_req size is not 32-bit");
/*
 HALT_REQ 
 b'ARC HALT Request'
*/
typedef struct reg_halt_req {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_halt_req;
static_assert((sizeof(struct reg_halt_req) == 4), "reg_halt_req size is not 32-bit");
/*
 RUN_HALT_ACK 
 b'ARC RUN/HALT ACK'
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
 b'ARC Reset Vector Address'
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
 b'ARC Debug Authentication Mode'
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
 b'ARC Cluster Number'
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
 b'ARC Core Number'
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
 b'ARC Wake Up Event'
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
 b'DCCM SYS Address Base'
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
 b'Hit indicators for each AP'
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
 b'MUX to select CTI/CFG Control'
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
 b'Processor asynchronous reset'
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
 b'ARC Reset Request'
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
 b'ARC Extension Address bits 31 downto 28'
*/
typedef struct reg_sram_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_sram_lsb_addr;
static_assert((sizeof(struct reg_sram_lsb_addr) == 4), "reg_sram_lsb_addr size is not 32-bit");
/*
 SRAM_MSB_ADDR 
 b'ARC Extension Address bits 63 downto 32'
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
 FW_MEM_LSB_ADDR 
 b'ARC Extension Address bits 31 downto 28'
*/
typedef struct reg_fw_mem_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_fw_mem_lsb_addr;
static_assert((sizeof(struct reg_fw_mem_lsb_addr) == 4), "reg_fw_mem_lsb_addr size is not 32-bit");
/*
 FW_MEM_MSB_ADDR 
 b'ARC Extension Address bits 63 downto 32'
*/
typedef struct reg_fw_mem_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_fw_mem_msb_addr;
static_assert((sizeof(struct reg_fw_mem_msb_addr) == 4), "reg_fw_mem_msb_addr size is not 32-bit");
/*
 VIR_MEM0_LSB_ADDR 
 b'ARC Extension Address bits 31 downto 28'
*/
typedef struct reg_vir_mem0_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_vir_mem0_lsb_addr;
static_assert((sizeof(struct reg_vir_mem0_lsb_addr) == 4), "reg_vir_mem0_lsb_addr size is not 32-bit");
/*
 VIR_MEM0_MSB_ADDR 
 b'ARC Extension Address bits 63 downto 32'
*/
typedef struct reg_vir_mem0_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_vir_mem0_msb_addr;
static_assert((sizeof(struct reg_vir_mem0_msb_addr) == 4), "reg_vir_mem0_msb_addr size is not 32-bit");
/*
 VIR_MEM1_LSB_ADDR 
 b'ARC Extension Address bits 31 downto 28'
*/
typedef struct reg_vir_mem1_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_vir_mem1_lsb_addr;
static_assert((sizeof(struct reg_vir_mem1_lsb_addr) == 4), "reg_vir_mem1_lsb_addr size is not 32-bit");
/*
 VIR_MEM1_MSB_ADDR 
 b'ARC Extension Address bits 63 downto 32'
*/
typedef struct reg_vir_mem1_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_vir_mem1_msb_addr;
static_assert((sizeof(struct reg_vir_mem1_msb_addr) == 4), "reg_vir_mem1_msb_addr size is not 32-bit");
/*
 VIR_MEM2_LSB_ADDR 
 b'ARC Extension Address bits 31 downto 28'
*/
typedef struct reg_vir_mem2_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_vir_mem2_lsb_addr;
static_assert((sizeof(struct reg_vir_mem2_lsb_addr) == 4), "reg_vir_mem2_lsb_addr size is not 32-bit");
/*
 VIR_MEM2_MSB_ADDR 
 b'ARC Extension Address bits 63 downto 32'
*/
typedef struct reg_vir_mem2_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_vir_mem2_msb_addr;
static_assert((sizeof(struct reg_vir_mem2_msb_addr) == 4), "reg_vir_mem2_msb_addr size is not 32-bit");
/*
 VIR_MEM3_LSB_ADDR 
 b'ARC Extension Address bits 31 downto 28'
*/
typedef struct reg_vir_mem3_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_vir_mem3_lsb_addr;
static_assert((sizeof(struct reg_vir_mem3_lsb_addr) == 4), "reg_vir_mem3_lsb_addr size is not 32-bit");
/*
 VIR_MEM3_MSB_ADDR 
 b'ARC Extension Address bits 63 downto 32'
*/
typedef struct reg_vir_mem3_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_vir_mem3_msb_addr;
static_assert((sizeof(struct reg_vir_mem3_msb_addr) == 4), "reg_vir_mem3_msb_addr size is not 32-bit");
/*
 VIR_MEM0_OFFSET 
 b'ARC HBM Offset address (bits 27 downto 0)'
*/
typedef struct reg_vir_mem0_offset {
	union {
		struct {
			uint32_t val : 28,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_vir_mem0_offset;
static_assert((sizeof(struct reg_vir_mem0_offset) == 4), "reg_vir_mem0_offset size is not 32-bit");
/*
 VIR_MEM1_OFFSET 
 b'ARC HBM Offset address (bits 27 downto 0)'
*/
typedef struct reg_vir_mem1_offset {
	union {
		struct {
			uint32_t val : 28,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_vir_mem1_offset;
static_assert((sizeof(struct reg_vir_mem1_offset) == 4), "reg_vir_mem1_offset size is not 32-bit");
/*
 VIR_MEM2_OFFSET 
 b'ARC HBM Offset address (bits 27 downto 0)'
*/
typedef struct reg_vir_mem2_offset {
	union {
		struct {
			uint32_t val : 28,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_vir_mem2_offset;
static_assert((sizeof(struct reg_vir_mem2_offset) == 4), "reg_vir_mem2_offset size is not 32-bit");
/*
 VIR_MEM3_OFFSET 
 b'ARC HBM Offset address (bits 27 downto 0)'
*/
typedef struct reg_vir_mem3_offset {
	union {
		struct {
			uint32_t val : 28,
				_reserved28 : 4;
		};
		uint32_t _raw;
	};
} reg_vir_mem3_offset;
static_assert((sizeof(struct reg_vir_mem3_offset) == 4), "reg_vir_mem3_offset size is not 32-bit");
/*
 PCIE_LOWER_LSB_ADDR 
 b'ARC Extension Address bits 31 downto 28'
*/
typedef struct reg_pcie_lower_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_pcie_lower_lsb_addr;
static_assert((sizeof(struct reg_pcie_lower_lsb_addr) == 4), "reg_pcie_lower_lsb_addr size is not 32-bit");
/*
 PCIE_LOWER_MSB_ADDR 
 b'ARC Extension Address bits 63 downto 32'
*/
typedef struct reg_pcie_lower_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_pcie_lower_msb_addr;
static_assert((sizeof(struct reg_pcie_lower_msb_addr) == 4), "reg_pcie_lower_msb_addr size is not 32-bit");
/*
 PCIE_UPPER_LSB_ADDR 
 b'ARC Extension Address bits 31 downto 28'
*/
typedef struct reg_pcie_upper_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_pcie_upper_lsb_addr;
static_assert((sizeof(struct reg_pcie_upper_lsb_addr) == 4), "reg_pcie_upper_lsb_addr size is not 32-bit");
/*
 PCIE_UPPER_MSB_ADDR 
 b'ARC Extension Address bits 63 downto 31'
*/
typedef struct reg_pcie_upper_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_pcie_upper_msb_addr;
static_assert((sizeof(struct reg_pcie_upper_msb_addr) == 4), "reg_pcie_upper_msb_addr size is not 32-bit");
/*
 D2D_HBW_LSB_ADDR 
 b'ARC Extension Address bits 31 downto 28'
*/
typedef struct reg_d2d_hbw_lsb_addr {
	union {
		struct {
			uint32_t val : 4,
				_reserved4 : 28;
		};
		uint32_t _raw;
	};
} reg_d2d_hbw_lsb_addr;
static_assert((sizeof(struct reg_d2d_hbw_lsb_addr) == 4), "reg_d2d_hbw_lsb_addr size is not 32-bit");
/*
 D2D_HBW_MSB_ADDR 
 b'ARC Extension Address bits 63 downto 31'
*/
typedef struct reg_d2d_hbw_msb_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_d2d_hbw_msb_addr;
static_assert((sizeof(struct reg_d2d_hbw_msb_addr) == 4), "reg_d2d_hbw_msb_addr size is not 32-bit");
/*
 GENERAL_PURPOSE_LSB_ADDR 
 b'ARC Extension Address bits 31 downto 28'
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
 b'ARC Extension Address bits 63 downto 32'
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
 b'ARC CBU AWCACHE Overide'
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
 b'ARC LBU AWCACHE Overide'
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
 b'Context ID'
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
 b'Context ID offset to increment/decrement: signed'
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
 b'ARC SW Interrupt Register'
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
 b'ARC IRQ Interrupt MASK'
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
 b'ARC SEI Interrupt Status'
*/
typedef struct reg_arc_sei_intr_sts {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_sei_intr_sts;
static_assert((sizeof(struct reg_arc_sei_intr_sts) == 4), "reg_arc_sei_intr_sts size is not 32-bit");
/*
 ARC_SEI_INTR_CLR 
 b'ARC SEI Interrupt Clear'
*/
typedef struct reg_arc_sei_intr_clr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_sei_intr_clr;
static_assert((sizeof(struct reg_arc_sei_intr_clr) == 4), "reg_arc_sei_intr_clr size is not 32-bit");
/*
 ARC_SEI_INTR_MASK 
 b'ARC SEI Interrupt Mask'
*/
typedef struct reg_arc_sei_intr_mask {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_sei_intr_mask;
static_assert((sizeof(struct reg_arc_sei_intr_mask) == 4), "reg_arc_sei_intr_mask size is not 32-bit");
/*
 ARC_EXCPTN_CAUSE 
 b'ARC Exception Cause'
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
 b'Enable ARC Halt/ARM INTR in case SEI INTR'
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
 b'HALT ARC on SEI INTR MASK Register'
*/
typedef struct reg_arc_sei_intr_halt_mask {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_arc_sei_intr_halt_mask;
static_assert((sizeof(struct reg_arc_sei_intr_halt_mask) == 4), "reg_arc_sei_intr_halt_mask size is not 32-bit");
/*
 QMAN_SEI_INTR_HALT_MASK 
 b'HALT QMAN on SEI INTR MASK Register'
*/
typedef struct reg_qman_sei_intr_halt_mask {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_qman_sei_intr_halt_mask;
static_assert((sizeof(struct reg_qman_sei_intr_halt_mask) == 4), "reg_qman_sei_intr_halt_mask size is not 32-bit");
/*
 ARC_REI_INTR_STS 
 b'ARC REI Interrupt Status'
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
 b'ARC REI Interrupt Clear'
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
 b'ARC REI Interrupt Mask'
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
 b'DCCM ECC ERR ADDR'
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
 b'DCCM ECC SYNDROME'
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
 b'Instruction Cache ECC ERR ADDR'
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
 b'Instruction Cache ECC SYNDROME'
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
 b'Data Cache ECC ERR ADDR'
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
 b'Data Cache ECC SYNDROME'
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
 b'LBW TRMINATE AWADDR ERR'
*/
typedef struct reg_lbw_trminate_awaddr_err {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_lbw_trminate_awaddr_err;
static_assert((sizeof(struct reg_lbw_trminate_awaddr_err) == 4), "reg_lbw_trminate_awaddr_err size is not 32-bit");
/*
 LBW_TRMINATE_ARADDR_ERR 
 b'LBW TRMINATE ARADDR ERR'
*/
typedef struct reg_lbw_trminate_araddr_err {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_lbw_trminate_araddr_err;
static_assert((sizeof(struct reg_lbw_trminate_araddr_err) == 4), "reg_lbw_trminate_araddr_err size is not 32-bit");
/*
 CFG_LBW_TERMINATE_BRESP 
 b'CFG LBW TERMINATE BRESP VAL'
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
 b'CFG LBW TERMINATE RRESP VAL'
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
 b'LBW TERMINATE AXLEN VAL'
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
 b'LBW TERMINATE AXSIZE VAL'
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
 b'General Purpuse Register'
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
 b'Total ARC CBU WR Request Counter'
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
 b'ARC Inflight CBU WR Request Counter'
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
 b'Total ARC CBU RD Request Counter'
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
 b'ARC CBU Inflight RD Request Counter'
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
 b'Total ARC LBU WR Request Counter'
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
 b'ARC Inflight LBU WR Request Counter'
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
 b'Total ARC LBU RD Request Counter'
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
 b'ARC Inflight LBU RD Request Counter'
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
 CBU_ARUSER_LSB_OVR 
 b'ARUSER LSB Overide'
*/
typedef struct reg_cbu_aruser_lsb_ovr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_aruser_lsb_ovr;
static_assert((sizeof(struct reg_cbu_aruser_lsb_ovr) == 4), "reg_cbu_aruser_lsb_ovr size is not 32-bit");
/*
 CBU_ARUSER_LSB_OVR_EN 
 b'ARUSER LSB Overide Enable'
*/
typedef struct reg_cbu_aruser_lsb_ovr_en {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_aruser_lsb_ovr_en;
static_assert((sizeof(struct reg_cbu_aruser_lsb_ovr_en) == 4), "reg_cbu_aruser_lsb_ovr_en size is not 32-bit");
/*
 CBU_ARUSER_MSB_OVR 
 b'ARUSER MSB Overide'
*/
typedef struct reg_cbu_aruser_msb_ovr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_aruser_msb_ovr;
static_assert((sizeof(struct reg_cbu_aruser_msb_ovr) == 4), "reg_cbu_aruser_msb_ovr size is not 32-bit");
/*
 CBU_ARUSER_MSB_OVR_EN 
 b'ARUSER MSB Overide Enable'
*/
typedef struct reg_cbu_aruser_msb_ovr_en {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_aruser_msb_ovr_en;
static_assert((sizeof(struct reg_cbu_aruser_msb_ovr_en) == 4), "reg_cbu_aruser_msb_ovr_en size is not 32-bit");
/*
 CBU_AWUSER_LSB_OVR 
 b'AWUSER LSB Overide'
*/
typedef struct reg_cbu_awuser_lsb_ovr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_awuser_lsb_ovr;
static_assert((sizeof(struct reg_cbu_awuser_lsb_ovr) == 4), "reg_cbu_awuser_lsb_ovr size is not 32-bit");
/*
 CBU_AWUSER_LSB_OVR_EN 
 b'AWUSER LSB Overide Enable'
*/
typedef struct reg_cbu_awuser_lsb_ovr_en {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_awuser_lsb_ovr_en;
static_assert((sizeof(struct reg_cbu_awuser_lsb_ovr_en) == 4), "reg_cbu_awuser_lsb_ovr_en size is not 32-bit");
/*
 CBU_AWUSER_MSB_OVR 
 b'AWUSER MSB Overide'
*/
typedef struct reg_cbu_awuser_msb_ovr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_awuser_msb_ovr;
static_assert((sizeof(struct reg_cbu_awuser_msb_ovr) == 4), "reg_cbu_awuser_msb_ovr size is not 32-bit");
/*
 CBU_AWUSER_MSB_OVR_EN 
 b'AWUSER MSB Overide Enable'
*/
typedef struct reg_cbu_awuser_msb_ovr_en {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_awuser_msb_ovr_en;
static_assert((sizeof(struct reg_cbu_awuser_msb_ovr_en) == 4), "reg_cbu_awuser_msb_ovr_en size is not 32-bit");
/*
 CBU_AXCACHE_OVR 
 b'CACHE Overide'
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
 b'LOCK Override'
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
 CBU_RD_PROT_0_OVR 
 b'CBU AXI SPLIT ARPROT[0] Override'
*/
typedef struct reg_cbu_rd_prot_0_ovr {
	union {
		struct {
			uint32_t cbu_read : 1,
				_reserved4 : 3,
				cbu_read_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_cbu_rd_prot_0_ovr;
static_assert((sizeof(struct reg_cbu_rd_prot_0_ovr) == 4), "reg_cbu_rd_prot_0_ovr size is not 32-bit");
/*
 CBU_RD_PROT_1_OVR 
 b'CBU AXI SPLIT ARPROT[1] Override'
*/
typedef struct reg_cbu_rd_prot_1_ovr {
	union {
		struct {
			uint32_t cbu_read : 1,
				_reserved4 : 3,
				cbu_read_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_cbu_rd_prot_1_ovr;
static_assert((sizeof(struct reg_cbu_rd_prot_1_ovr) == 4), "reg_cbu_rd_prot_1_ovr size is not 32-bit");
/*
 CBU_RD_PROT_2_OVR 
 b'CBU AXI SPLIT ARPROT[2] Override'
*/
typedef struct reg_cbu_rd_prot_2_ovr {
	union {
		struct {
			uint32_t cbu_read : 1,
				_reserved4 : 3,
				cbu_read_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_cbu_rd_prot_2_ovr;
static_assert((sizeof(struct reg_cbu_rd_prot_2_ovr) == 4), "reg_cbu_rd_prot_2_ovr size is not 32-bit");
/*
 CBU_WR_PROT_0_OVR 
 b'CBU AXI SPLIT AWPROT[0] Override'
*/
typedef struct reg_cbu_wr_prot_0_ovr {
	union {
		struct {
			uint32_t cbu_write : 1,
				_reserved4 : 3,
				cbu_write_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_cbu_wr_prot_0_ovr;
static_assert((sizeof(struct reg_cbu_wr_prot_0_ovr) == 4), "reg_cbu_wr_prot_0_ovr size is not 32-bit");
/*
 CBU_WR_PROT_1_OVR 
 b'CBU AXI SPLIT AWPROT[1] Override'
*/
typedef struct reg_cbu_wr_prot_1_ovr {
	union {
		struct {
			uint32_t cbu_write : 1,
				_reserved4 : 3,
				cbu_write_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_cbu_wr_prot_1_ovr;
static_assert((sizeof(struct reg_cbu_wr_prot_1_ovr) == 4), "reg_cbu_wr_prot_1_ovr size is not 32-bit");
/*
 CBU_WR_PROT_2_OVR 
 b'CBU AXI SPLIT AWPROT[2] Override'
*/
typedef struct reg_cbu_wr_prot_2_ovr {
	union {
		struct {
			uint32_t cbu_write : 1,
				_reserved4 : 3,
				cbu_write_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_cbu_wr_prot_2_ovr;
static_assert((sizeof(struct reg_cbu_wr_prot_2_ovr) == 4), "reg_cbu_wr_prot_2_ovr size is not 32-bit");
/*
 CBU_MAX_OUTSTANDING 
 b'MAX RD WR OUTSTANDING REQ'
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
 b'Early BRESP Configuration'
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
 b'FORCE Response OK'
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
 b'CBU AXI SPLIT No WR Inflight'
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
 b'CBU AXI SPLIT SEI Interrupt ID'
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
 b'ARUSER Overide'
*/
typedef struct reg_lbu_aruser_ovr {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_lbu_aruser_ovr;
static_assert((sizeof(struct reg_lbu_aruser_ovr) == 4), "reg_lbu_aruser_ovr size is not 32-bit");
/*
 LBU_ARUSER_OVR_EN 
 b'ARUSER Overide Enable'
*/
typedef struct reg_lbu_aruser_ovr_en {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_lbu_aruser_ovr_en;
static_assert((sizeof(struct reg_lbu_aruser_ovr_en) == 4), "reg_lbu_aruser_ovr_en size is not 32-bit");
/*
 LBU_AWUSER_OVR 
 b'AWUSER Overide'
*/
typedef struct reg_lbu_awuser_ovr {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_lbu_awuser_ovr;
static_assert((sizeof(struct reg_lbu_awuser_ovr) == 4), "reg_lbu_awuser_ovr size is not 32-bit");
/*
 LBU_AWUSER_OVR_EN 
 b'AWUSER Overide Enable'
*/
typedef struct reg_lbu_awuser_ovr_en {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_lbu_awuser_ovr_en;
static_assert((sizeof(struct reg_lbu_awuser_ovr_en) == 4), "reg_lbu_awuser_ovr_en size is not 32-bit");
/*
 LBU_AXCACHE_OVR 
 b'CACHE Overide'
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
 b'LOCK Override'
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
 LBU_RD_PROT_0_OVR 
 b'LBU AXI SPLIT ARPROT[0] Override'
*/
typedef struct reg_lbu_rd_prot_0_ovr {
	union {
		struct {
			uint32_t lbu_read : 1,
				_reserved4 : 3,
				lbu_read_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_lbu_rd_prot_0_ovr;
static_assert((sizeof(struct reg_lbu_rd_prot_0_ovr) == 4), "reg_lbu_rd_prot_0_ovr size is not 32-bit");
/*
 LBU_RD_PROT_1_OVR 
 b'LBU AXI SPLIT ARPROT[1] Override'
*/
typedef struct reg_lbu_rd_prot_1_ovr {
	union {
		struct {
			uint32_t lbu_read : 1,
				_reserved4 : 3,
				lbu_read_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_lbu_rd_prot_1_ovr;
static_assert((sizeof(struct reg_lbu_rd_prot_1_ovr) == 4), "reg_lbu_rd_prot_1_ovr size is not 32-bit");
/*
 LBU_RD_PROT_2_OVR 
 b'LBU AXI SPLIT ARPROT[2] Override'
*/
typedef struct reg_lbu_rd_prot_2_ovr {
	union {
		struct {
			uint32_t lbu_read : 1,
				_reserved4 : 3,
				lbu_read_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_lbu_rd_prot_2_ovr;
static_assert((sizeof(struct reg_lbu_rd_prot_2_ovr) == 4), "reg_lbu_rd_prot_2_ovr size is not 32-bit");
/*
 LBU_WR_PROT_0_OVR 
 b'LBU AXI SPLIT AWPROT[0] Override'
*/
typedef struct reg_lbu_wr_prot_0_ovr {
	union {
		struct {
			uint32_t lbu_write : 1,
				_reserved4 : 3,
				lbu_write_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_lbu_wr_prot_0_ovr;
static_assert((sizeof(struct reg_lbu_wr_prot_0_ovr) == 4), "reg_lbu_wr_prot_0_ovr size is not 32-bit");
/*
 LBU_WR_PROT_1_OVR 
 b'LBU AXI SPLIT AWPROT[1] Override'
*/
typedef struct reg_lbu_wr_prot_1_ovr {
	union {
		struct {
			uint32_t lbu_write : 1,
				_reserved4 : 3,
				lbu_write_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_lbu_wr_prot_1_ovr;
static_assert((sizeof(struct reg_lbu_wr_prot_1_ovr) == 4), "reg_lbu_wr_prot_1_ovr size is not 32-bit");
/*
 LBU_WR_PROT_2_OVR 
 b'LBU AXI SPLIT AWPROT[2] Override'
*/
typedef struct reg_lbu_wr_prot_2_ovr {
	union {
		struct {
			uint32_t lbu_write : 1,
				_reserved4 : 3,
				lbu_write_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_lbu_wr_prot_2_ovr;
static_assert((sizeof(struct reg_lbu_wr_prot_2_ovr) == 4), "reg_lbu_wr_prot_2_ovr size is not 32-bit");
/*
 LBU_MAX_OUTSTANDING 
 b'MAX RD WR OUTSTANDING REQ'
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
 b'Early BRESP Configuration'
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
 b'FORCE Response OK'
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
 b'LBU AXI SPLIT No WR Inflight'
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
 b'LBU AXI SPLIT SEI Interrupt ID'
*/
typedef struct reg_lbu_sei_intr_id {
	union {
		struct {
			uint32_t val : 9,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_lbu_sei_intr_id;
static_assert((sizeof(struct reg_lbu_sei_intr_id) == 4), "reg_lbu_sei_intr_id size is not 32-bit");
/*
 DCCM_QUEUE_BASE_ADDR 
 b'DCCM QUEUE BASE ADDR'
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
 b'DCCM QUEUE SIZE'
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
 b'DCCM QUEUE PI'
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
 b'DCCM QUEUE CI'
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
 b'DCCM QUEUE PUSH REG'
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
 b'DCCM QUEUE MAX OCCUPANCY'
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
 b'DCCM QUEUE Valid Entries'
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
 DCCM_QUEUE_EARLY_PI 
 b'DCCM QUEUE EARLY PI'
*/
typedef struct reg_dccm_queue_early_pi {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_early_pi;
static_assert((sizeof(struct reg_dccm_queue_early_pi) == 4), "reg_dccm_queue_early_pi size is not 32-bit");
/*
 DCCM_QUEUE_TOTAL_ENTRIES 
 b'DCCM QUEUE Valid Entries in DCCM + inflight PUSH requests'
*/
typedef struct reg_dccm_queue_total_entries {
	union {
		struct {
			uint32_t val : 24,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_total_entries;
static_assert((sizeof(struct reg_dccm_queue_total_entries) == 4), "reg_dccm_queue_total_entries size is not 32-bit");
/*
 GENERAL_Q_VLD_ENTRY_MASK 
 b'GENERAL QEUEU Valid Entries Mask'
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
 b'NIC QEUEU Valid Entries Mask'
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
 DCCM_QUEUE_NOT_EMPTY 
 b'DCCM QUEUE NOT EMPTY Indication'
*/
typedef struct reg_dccm_queue_not_empty {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_not_empty;
static_assert((sizeof(struct reg_dccm_queue_not_empty) == 4), "reg_dccm_queue_not_empty size is not 32-bit");
/*
 DCCM_QUEUE_DROP_INTR_STS 
 b'DCCM QUEUE DROP SEI Interrupt Status'
*/
typedef struct reg_dccm_queue_drop_intr_sts {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_drop_intr_sts;
static_assert((sizeof(struct reg_dccm_queue_drop_intr_sts) == 4), "reg_dccm_queue_drop_intr_sts size is not 32-bit");
/*
 DCCM_QUEUE_DROP_INTR_CLR 
 b'DCCM QUEUE DROP Interrupt Clear'
*/
typedef struct reg_dccm_queue_drop_intr_clr {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_drop_intr_clr;
static_assert((sizeof(struct reg_dccm_queue_drop_intr_clr) == 4), "reg_dccm_queue_drop_intr_clr size is not 32-bit");
/*
 DCCM_QUEUE_DROP_INTR_MASK 
 b'DCCM QUEUE DROP Interrupt Mask'
*/
typedef struct reg_dccm_queue_drop_intr_mask {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_dccm_queue_drop_intr_mask;
static_assert((sizeof(struct reg_dccm_queue_drop_intr_mask) == 4), "reg_dccm_queue_drop_intr_mask size is not 32-bit");
/*
 DCCM_QUEUE_DROP_EN 
 b'DCCM QUEUE DROP EN'
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
 b'DCCM_QUEUE_WARN_MSG'
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
 b'DCCM_QUEUE_ALERT_MSG'
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
 b'DCCM GEN AXI AWPROT Value'
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
 b'DCCM GEN AXI AWUSER Value'
*/
typedef struct reg_dccm_gen_axi_awuser {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_dccm_gen_axi_awuser;
static_assert((sizeof(struct reg_dccm_gen_axi_awuser) == 4), "reg_dccm_gen_axi_awuser size is not 32-bit");
/*
 DCCM_GEN_AXI_AWBURST 
 b'DCCM GEN AXI AWBURST Value'
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
 b'DCCM GEN AXI AWLOCK Value'
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
 b'DCCM GEN AXI AWCACHE Value'
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
 b'DCCM WRR Arbiter Weights'
*/
typedef struct reg_dccm_wrr_arb_weight {
	union {
		struct {
			uint32_t dccm_lbw_axi_slv : 4,
				dccm_push_q : 4,
				dccm_af : 4,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_dccm_wrr_arb_weight;
static_assert((sizeof(struct reg_dccm_wrr_arb_weight) == 4), "reg_dccm_wrr_arb_weight size is not 32-bit");
/*
 DCCM_Q_PUSH_FIFO_FULL_CFG 
 b'DCCM QUEUE PUSH FIFO Full Threshold'
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
 b'DCCM QUEUE PUSH FIFO Counter'
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
 b'QMAN CQ In FIFO SHADOW CI'
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
 b'QMAN ARC CQ In FIFO SHADOW CI'
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
 b'QMAN CQ SHADOW CI'
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
 b'QMAN ARC CQ SHADOW CI'
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
 AUX2APB_PPROT_0 
 b'AUX2APB PPROT[0] Value'
*/
typedef struct reg_aux2apb_pprot_0 {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_aux2apb_pprot_0;
static_assert((sizeof(struct reg_aux2apb_pprot_0) == 4), "reg_aux2apb_pprot_0 size is not 32-bit");
/*
 AUX2APB_PPROT_1 
 b'AUX2APB PPROT[1] Value'
*/
typedef struct reg_aux2apb_pprot_1 {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_aux2apb_pprot_1;
static_assert((sizeof(struct reg_aux2apb_pprot_1) == 4), "reg_aux2apb_pprot_1 size is not 32-bit");
/*
 AUX2APB_PPROT_2 
 b'AUX2APB PPROT[2] Value'
*/
typedef struct reg_aux2apb_pprot_2 {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_aux2apb_pprot_2;
static_assert((sizeof(struct reg_aux2apb_pprot_2) == 4), "reg_aux2apb_pprot_2 size is not 32-bit");
/*
 LBW_FORK_ARADDR_INTR_INFO 
 b'LBW SLV FORK ARADDR INTR INFO'
*/
typedef struct reg_lbw_fork_araddr_intr_info {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_lbw_fork_araddr_intr_info;
static_assert((sizeof(struct reg_lbw_fork_araddr_intr_info) == 4), "reg_lbw_fork_araddr_intr_info size is not 32-bit");
/*
 LBW_FORK_AWADDR_INTR_INFO 
 b'LBW SLV FORK AWADDR INTR INFO'
*/
typedef struct reg_lbw_fork_awaddr_intr_info {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_lbw_fork_awaddr_intr_info;
static_assert((sizeof(struct reg_lbw_fork_awaddr_intr_info) == 4), "reg_lbw_fork_awaddr_intr_info size is not 32-bit");
/*
 LBW_FORK_WIN_EN 
 b'LBW FORK Windows Enable'
*/
typedef struct reg_lbw_fork_win_en {
	union {
		struct {
			uint32_t val : 5,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_lbw_fork_win_en;
static_assert((sizeof(struct reg_lbw_fork_win_en) == 4), "reg_lbw_fork_win_en size is not 32-bit");
/*
 QMAN_LBW_FORK_MIN_ADDR0 
 b'QMAN LBW FORK MIN Address0'
*/
typedef struct reg_qman_lbw_fork_min_addr0 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_min_addr0;
static_assert((sizeof(struct reg_qman_lbw_fork_min_addr0) == 4), "reg_qman_lbw_fork_min_addr0 size is not 32-bit");
/*
 QMAN_LBW_FORK_MAX_ADDR0 
 b'QMAN LBW FORK MAX Address0'
*/
typedef struct reg_qman_lbw_fork_max_addr0 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_max_addr0;
static_assert((sizeof(struct reg_qman_lbw_fork_max_addr0) == 4), "reg_qman_lbw_fork_max_addr0 size is not 32-bit");
/*
 QMAN_LBW_FORK_MASK_ADDR0 
 b'QMAN LBW FORK MASK Address0'
*/
typedef struct reg_qman_lbw_fork_mask_addr0 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_mask_addr0;
static_assert((sizeof(struct reg_qman_lbw_fork_mask_addr0) == 4), "reg_qman_lbw_fork_mask_addr0 size is not 32-bit");
/*
 QMAN_LBW_FORK_MIN_ADDR1 
 b'QMAN LBW FORK MIN Address1'
*/
typedef struct reg_qman_lbw_fork_min_addr1 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_min_addr1;
static_assert((sizeof(struct reg_qman_lbw_fork_min_addr1) == 4), "reg_qman_lbw_fork_min_addr1 size is not 32-bit");
/*
 QMAN_LBW_FORK_MAX_ADDR1 
 b'QMAN LBW FORK MAX Address1'
*/
typedef struct reg_qman_lbw_fork_max_addr1 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_max_addr1;
static_assert((sizeof(struct reg_qman_lbw_fork_max_addr1) == 4), "reg_qman_lbw_fork_max_addr1 size is not 32-bit");
/*
 QMAN_LBW_FORK_MASK_ADDR1 
 b'QMAN LBW FORK MASK Address1'
*/
typedef struct reg_qman_lbw_fork_mask_addr1 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_mask_addr1;
static_assert((sizeof(struct reg_qman_lbw_fork_mask_addr1) == 4), "reg_qman_lbw_fork_mask_addr1 size is not 32-bit");
/*
 QMAN_LBW_FORK_MIN_ADDR2 
 b'QMAN LBW FORK MIN Address2'
*/
typedef struct reg_qman_lbw_fork_min_addr2 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_min_addr2;
static_assert((sizeof(struct reg_qman_lbw_fork_min_addr2) == 4), "reg_qman_lbw_fork_min_addr2 size is not 32-bit");
/*
 QMAN_LBW_FORK_MAX_ADDR2 
 b'QMAN LBW FORK MAX Address2'
*/
typedef struct reg_qman_lbw_fork_max_addr2 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_max_addr2;
static_assert((sizeof(struct reg_qman_lbw_fork_max_addr2) == 4), "reg_qman_lbw_fork_max_addr2 size is not 32-bit");
/*
 QMAN_LBW_FORK_MASK_ADDR2 
 b'QMAN LBW FORK MASK Address2'
*/
typedef struct reg_qman_lbw_fork_mask_addr2 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_mask_addr2;
static_assert((sizeof(struct reg_qman_lbw_fork_mask_addr2) == 4), "reg_qman_lbw_fork_mask_addr2 size is not 32-bit");
/*
 QMAN_LBW_FORK_MIN_ADDR3 
 b'QMAN LBW FORK MIN Address3'
*/
typedef struct reg_qman_lbw_fork_min_addr3 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_min_addr3;
static_assert((sizeof(struct reg_qman_lbw_fork_min_addr3) == 4), "reg_qman_lbw_fork_min_addr3 size is not 32-bit");
/*
 QMAN_LBW_FORK_MAX_ADDR3 
 b'QMAN LBW FORK MAX Address3'
*/
typedef struct reg_qman_lbw_fork_max_addr3 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_max_addr3;
static_assert((sizeof(struct reg_qman_lbw_fork_max_addr3) == 4), "reg_qman_lbw_fork_max_addr3 size is not 32-bit");
/*
 QMAN_LBW_FORK_MASK_ADDR3 
 b'QMAN LBW FORK MASK Address3'
*/
typedef struct reg_qman_lbw_fork_mask_addr3 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_mask_addr3;
static_assert((sizeof(struct reg_qman_lbw_fork_mask_addr3) == 4), "reg_qman_lbw_fork_mask_addr3 size is not 32-bit");
/*
 QMAN_LBW_FORK_MIN_ADDR4 
 b'QMAN LBW FORK MIN Address4'
*/
typedef struct reg_qman_lbw_fork_min_addr4 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_min_addr4;
static_assert((sizeof(struct reg_qman_lbw_fork_min_addr4) == 4), "reg_qman_lbw_fork_min_addr4 size is not 32-bit");
/*
 QMAN_LBW_FORK_MAX_ADDR4 
 b'QMAN LBW FORK MAX Address4'
*/
typedef struct reg_qman_lbw_fork_max_addr4 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_max_addr4;
static_assert((sizeof(struct reg_qman_lbw_fork_max_addr4) == 4), "reg_qman_lbw_fork_max_addr4 size is not 32-bit");
/*
 QMAN_LBW_FORK_MASK_ADDR4 
 b'QMAN LBW FORK MASK Address4'
*/
typedef struct reg_qman_lbw_fork_mask_addr4 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_qman_lbw_fork_mask_addr4;
static_assert((sizeof(struct reg_qman_lbw_fork_mask_addr4) == 4), "reg_qman_lbw_fork_mask_addr4 size is not 32-bit");
/*
 FARM_LBW_FORK_MIN_ADDR0 
 b'FARM LBW FORK MIN Address0'
*/
typedef struct reg_farm_lbw_fork_min_addr0 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_min_addr0;
static_assert((sizeof(struct reg_farm_lbw_fork_min_addr0) == 4), "reg_farm_lbw_fork_min_addr0 size is not 32-bit");
/*
 FARM_LBW_FORK_MAX_ADDR0 
 b'FARM LBW FORK MAX Address0'
*/
typedef struct reg_farm_lbw_fork_max_addr0 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_max_addr0;
static_assert((sizeof(struct reg_farm_lbw_fork_max_addr0) == 4), "reg_farm_lbw_fork_max_addr0 size is not 32-bit");
/*
 FARM_LBW_FORK_MASK_ADDR0 
 b'FARM LBW FORK MASK Address0'
*/
typedef struct reg_farm_lbw_fork_mask_addr0 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_mask_addr0;
static_assert((sizeof(struct reg_farm_lbw_fork_mask_addr0) == 4), "reg_farm_lbw_fork_mask_addr0 size is not 32-bit");
/*
 FARM_LBW_FORK_MIN_ADDR1 
 b'FARM LBW FORK MIN Address1'
*/
typedef struct reg_farm_lbw_fork_min_addr1 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_min_addr1;
static_assert((sizeof(struct reg_farm_lbw_fork_min_addr1) == 4), "reg_farm_lbw_fork_min_addr1 size is not 32-bit");
/*
 FARM_LBW_FORK_MAX_ADDR1 
 b'FARM LBW FORK MAX Address1'
*/
typedef struct reg_farm_lbw_fork_max_addr1 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_max_addr1;
static_assert((sizeof(struct reg_farm_lbw_fork_max_addr1) == 4), "reg_farm_lbw_fork_max_addr1 size is not 32-bit");
/*
 FARM_LBW_FORK_MASK_ADDR1 
 b'FARM LBW FORK MASK Address1'
*/
typedef struct reg_farm_lbw_fork_mask_addr1 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_mask_addr1;
static_assert((sizeof(struct reg_farm_lbw_fork_mask_addr1) == 4), "reg_farm_lbw_fork_mask_addr1 size is not 32-bit");
/*
 FARM_LBW_FORK_MIN_ADDR2 
 b'FARM LBW FORK MIN Address2'
*/
typedef struct reg_farm_lbw_fork_min_addr2 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_min_addr2;
static_assert((sizeof(struct reg_farm_lbw_fork_min_addr2) == 4), "reg_farm_lbw_fork_min_addr2 size is not 32-bit");
/*
 FARM_LBW_FORK_MAX_ADDR2 
 b'FARM LBW FORK MAX Address2'
*/
typedef struct reg_farm_lbw_fork_max_addr2 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_max_addr2;
static_assert((sizeof(struct reg_farm_lbw_fork_max_addr2) == 4), "reg_farm_lbw_fork_max_addr2 size is not 32-bit");
/*
 FARM_LBW_FORK_MASK_ADDR2 
 b'FARM LBW FORK MASK Address2'
*/
typedef struct reg_farm_lbw_fork_mask_addr2 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_mask_addr2;
static_assert((sizeof(struct reg_farm_lbw_fork_mask_addr2) == 4), "reg_farm_lbw_fork_mask_addr2 size is not 32-bit");
/*
 FARM_LBW_FORK_MIN_ADDR3 
 b'FARM LBW FORK MIN Address3'
*/
typedef struct reg_farm_lbw_fork_min_addr3 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_min_addr3;
static_assert((sizeof(struct reg_farm_lbw_fork_min_addr3) == 4), "reg_farm_lbw_fork_min_addr3 size is not 32-bit");
/*
 FARM_LBW_FORK_MAX_ADDR3 
 b'FARM LBW FORK MAX Address3'
*/
typedef struct reg_farm_lbw_fork_max_addr3 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_max_addr3;
static_assert((sizeof(struct reg_farm_lbw_fork_max_addr3) == 4), "reg_farm_lbw_fork_max_addr3 size is not 32-bit");
/*
 FARM_LBW_FORK_MASK_ADDR3 
 b'FARM LBW FORK MASK Address3'
*/
typedef struct reg_farm_lbw_fork_mask_addr3 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_mask_addr3;
static_assert((sizeof(struct reg_farm_lbw_fork_mask_addr3) == 4), "reg_farm_lbw_fork_mask_addr3 size is not 32-bit");
/*
 FARM_LBW_FORK_MIN_ADDR4 
 b'FARM LBW FORK MIN Address4'
*/
typedef struct reg_farm_lbw_fork_min_addr4 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_min_addr4;
static_assert((sizeof(struct reg_farm_lbw_fork_min_addr4) == 4), "reg_farm_lbw_fork_min_addr4 size is not 32-bit");
/*
 FARM_LBW_FORK_MAX_ADDR4 
 b'FARM LBW FORK MAX Address4'
*/
typedef struct reg_farm_lbw_fork_max_addr4 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_max_addr4;
static_assert((sizeof(struct reg_farm_lbw_fork_max_addr4) == 4), "reg_farm_lbw_fork_max_addr4 size is not 32-bit");
/*
 FARM_LBW_FORK_MASK_ADDR4 
 b'FARM LBW FORK MASK Address4'
*/
typedef struct reg_farm_lbw_fork_mask_addr4 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_farm_lbw_fork_mask_addr4;
static_assert((sizeof(struct reg_farm_lbw_fork_mask_addr4) == 4), "reg_farm_lbw_fork_mask_addr4 size is not 32-bit");
/*
 LBW_FORK_TERMINATE_BRESP 
 b'LBW FORK TERMINATE BRESP VAL'
*/
typedef struct reg_lbw_fork_terminate_bresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lbw_fork_terminate_bresp;
static_assert((sizeof(struct reg_lbw_fork_terminate_bresp) == 4), "reg_lbw_fork_terminate_bresp size is not 32-bit");
/*
 LBW_FORK_TERMINATE_RRESP 
 b'LBW FORK TERMINATE RRESP VAL'
*/
typedef struct reg_lbw_fork_terminate_rresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lbw_fork_terminate_rresp;
static_assert((sizeof(struct reg_lbw_fork_terminate_rresp) == 4), "reg_lbw_fork_terminate_rresp size is not 32-bit");
/*
 LBW_DUP_PUSH_TERMINATOR_BRESP 
 b'LBW FORK TERMINATE BRESP VAL'
*/
typedef struct reg_lbw_dup_push_terminator_bresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lbw_dup_push_terminator_bresp;
static_assert((sizeof(struct reg_lbw_dup_push_terminator_bresp) == 4), "reg_lbw_dup_push_terminator_bresp size is not 32-bit");
/*
 LBW_DUP_PUSH_TERMINATOR_RRESP 
 b'LBW FORK TERMINATE RRESP VAL'
*/
typedef struct reg_lbw_dup_push_terminator_rresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lbw_dup_push_terminator_rresp;
static_assert((sizeof(struct reg_lbw_dup_push_terminator_rresp) == 4), "reg_lbw_dup_push_terminator_rresp size is not 32-bit");
/*
 LBW_DUP_CFG_TERMINATOR_BRESP 
 b'LBW FORK TERMINATE BRESP VAL'
*/
typedef struct reg_lbw_dup_cfg_terminator_bresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lbw_dup_cfg_terminator_bresp;
static_assert((sizeof(struct reg_lbw_dup_cfg_terminator_bresp) == 4), "reg_lbw_dup_cfg_terminator_bresp size is not 32-bit");
/*
 LBW_DUP_CFG_TERMINATOR_RRESP 
 b'LBW FORK TERMINATE RRESP VAL'
*/
typedef struct reg_lbw_dup_cfg_terminator_rresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lbw_dup_cfg_terminator_rresp;
static_assert((sizeof(struct reg_lbw_dup_cfg_terminator_rresp) == 4), "reg_lbw_dup_cfg_terminator_rresp size is not 32-bit");
/*
 LBW_AF_CFG_TERMINATOR_BRESP 
 b'LBW FORK TERMINATE BRESP VAL'
*/
typedef struct reg_lbw_af_cfg_terminator_bresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lbw_af_cfg_terminator_bresp;
static_assert((sizeof(struct reg_lbw_af_cfg_terminator_bresp) == 4), "reg_lbw_af_cfg_terminator_bresp size is not 32-bit");
/*
 LBW_AF_CFG_TERMINATOR_RRESP 
 b'LBW FORK TERMINATE RRESP VAL'
*/
typedef struct reg_lbw_af_cfg_terminator_rresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_lbw_af_cfg_terminator_rresp;
static_assert((sizeof(struct reg_lbw_af_cfg_terminator_rresp) == 4), "reg_lbw_af_cfg_terminator_rresp size is not 32-bit");
/*
 LBW_AXI2APB_CFG 
 b'LBW AXI2APB CFG'
*/
typedef struct reg_lbw_axi2apb_cfg {
	union {
		struct {
			uint32_t mask_err_rsp : 1,
				_reserved4 : 3,
				blk_partial_strb : 1,
				_reserved8 : 3,
				spl_err_rsp : 1,
				_reserved9 : 23;
		};
		uint32_t _raw;
	};
} reg_lbw_axi2apb_cfg;
static_assert((sizeof(struct reg_lbw_axi2apb_cfg) == 4), "reg_lbw_axi2apb_cfg size is not 32-bit");
/*
 LBW_APB_FORK_MAX_ADDR0 
 b'ARC APB LBW FORK Max Address 0'
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
 b'ARC APB LBW FORK Max Address 1'
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
 CBU_FORK_WIN_EN 
 b'CBU FORK Window Enable'
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
 CBU_FORK_MIN_ADDR0_LSB 
 b'CBU RD FORK MIN Address0 LSB'
*/
typedef struct reg_cbu_fork_min_addr0_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_min_addr0_lsb;
static_assert((sizeof(struct reg_cbu_fork_min_addr0_lsb) == 4), "reg_cbu_fork_min_addr0_lsb size is not 32-bit");
/*
 CBU_FORK_MIN_ADDR0_MSB 
 b'CBU RD FORK MIN Address0 MSB'
*/
typedef struct reg_cbu_fork_min_addr0_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_min_addr0_msb;
static_assert((sizeof(struct reg_cbu_fork_min_addr0_msb) == 4), "reg_cbu_fork_min_addr0_msb size is not 32-bit");
/*
 CBU_FORK_MAX_ADDR0_LSB 
 b'CBU RD FORK MAX Address0 LSB'
*/
typedef struct reg_cbu_fork_max_addr0_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_max_addr0_lsb;
static_assert((sizeof(struct reg_cbu_fork_max_addr0_lsb) == 4), "reg_cbu_fork_max_addr0_lsb size is not 32-bit");
/*
 CBU_FORK_MAX_ADDR0_MSB 
 b'CBU RD FORK MAX Address0 MSB'
*/
typedef struct reg_cbu_fork_max_addr0_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_max_addr0_msb;
static_assert((sizeof(struct reg_cbu_fork_max_addr0_msb) == 4), "reg_cbu_fork_max_addr0_msb size is not 32-bit");
/*
 CBU_FORK_MASK_ADDR0_LSB 
 b'CBU RD FORK Mask Address0 LSB'
*/
typedef struct reg_cbu_fork_mask_addr0_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_mask_addr0_lsb;
static_assert((sizeof(struct reg_cbu_fork_mask_addr0_lsb) == 4), "reg_cbu_fork_mask_addr0_lsb size is not 32-bit");
/*
 CBU_FORK_MASK_ADDR0_MSB 
 b'CBU RD FORK Mask Address0 MSB'
*/
typedef struct reg_cbu_fork_mask_addr0_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_mask_addr0_msb;
static_assert((sizeof(struct reg_cbu_fork_mask_addr0_msb) == 4), "reg_cbu_fork_mask_addr0_msb size is not 32-bit");
/*
 CBU_FORK_MIN_ADDR1_LSB 
 b'CBU RD FORK MIN Address1 LSB'
*/
typedef struct reg_cbu_fork_min_addr1_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_min_addr1_lsb;
static_assert((sizeof(struct reg_cbu_fork_min_addr1_lsb) == 4), "reg_cbu_fork_min_addr1_lsb size is not 32-bit");
/*
 CBU_FORK_MIN_ADDR1_MSB 
 b'CBU RD FORK MIN Address1 MSB'
*/
typedef struct reg_cbu_fork_min_addr1_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_min_addr1_msb;
static_assert((sizeof(struct reg_cbu_fork_min_addr1_msb) == 4), "reg_cbu_fork_min_addr1_msb size is not 32-bit");
/*
 CBU_FORK_MAX_ADDR1_LSB 
 b'CBU RD FORK MAX Address1 LSB'
*/
typedef struct reg_cbu_fork_max_addr1_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_max_addr1_lsb;
static_assert((sizeof(struct reg_cbu_fork_max_addr1_lsb) == 4), "reg_cbu_fork_max_addr1_lsb size is not 32-bit");
/*
 CBU_FORK_MAX_ADDR1_MSB 
 b'CBU RD FORK MAX Address1 MSB'
*/
typedef struct reg_cbu_fork_max_addr1_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_max_addr1_msb;
static_assert((sizeof(struct reg_cbu_fork_max_addr1_msb) == 4), "reg_cbu_fork_max_addr1_msb size is not 32-bit");
/*
 CBU_FORK_MASK_ADDR1_LSB 
 b'CBU RD FORK Mask Address1 LSB'
*/
typedef struct reg_cbu_fork_mask_addr1_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_mask_addr1_lsb;
static_assert((sizeof(struct reg_cbu_fork_mask_addr1_lsb) == 4), "reg_cbu_fork_mask_addr1_lsb size is not 32-bit");
/*
 CBU_FORK_MASK_ADDR1_MSB 
 b'CBU RD FORK Mask Address1 MSB'
*/
typedef struct reg_cbu_fork_mask_addr1_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_mask_addr1_msb;
static_assert((sizeof(struct reg_cbu_fork_mask_addr1_msb) == 4), "reg_cbu_fork_mask_addr1_msb size is not 32-bit");
/*
 CBU_FORK_MIN_ADDR2_LSB 
 b'CBU RD FORK MIN Address2 LSB'
*/
typedef struct reg_cbu_fork_min_addr2_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_min_addr2_lsb;
static_assert((sizeof(struct reg_cbu_fork_min_addr2_lsb) == 4), "reg_cbu_fork_min_addr2_lsb size is not 32-bit");
/*
 CBU_FORK_MIN_ADDR2_MSB 
 b'CBU RD FORK MIN Address2 MSB'
*/
typedef struct reg_cbu_fork_min_addr2_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_min_addr2_msb;
static_assert((sizeof(struct reg_cbu_fork_min_addr2_msb) == 4), "reg_cbu_fork_min_addr2_msb size is not 32-bit");
/*
 CBU_FORK_MAX_ADDR2_LSB 
 b'CBU RD FORK MAX Address2 LSB'
*/
typedef struct reg_cbu_fork_max_addr2_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_max_addr2_lsb;
static_assert((sizeof(struct reg_cbu_fork_max_addr2_lsb) == 4), "reg_cbu_fork_max_addr2_lsb size is not 32-bit");
/*
 CBU_FORK_MAX_ADDR2_MSB 
 b'CBU RD FORK MAX Address2 MSB'
*/
typedef struct reg_cbu_fork_max_addr2_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_max_addr2_msb;
static_assert((sizeof(struct reg_cbu_fork_max_addr2_msb) == 4), "reg_cbu_fork_max_addr2_msb size is not 32-bit");
/*
 CBU_FORK_MASK_ADDR2_LSB 
 b'CBU RD FORK Mask Address2 LSB'
*/
typedef struct reg_cbu_fork_mask_addr2_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_mask_addr2_lsb;
static_assert((sizeof(struct reg_cbu_fork_mask_addr2_lsb) == 4), "reg_cbu_fork_mask_addr2_lsb size is not 32-bit");
/*
 CBU_FORK_MASK_ADDR2_MSB 
 b'CBU RD FORK Mask Address2 MSB'
*/
typedef struct reg_cbu_fork_mask_addr2_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_mask_addr2_msb;
static_assert((sizeof(struct reg_cbu_fork_mask_addr2_msb) == 4), "reg_cbu_fork_mask_addr2_msb size is not 32-bit");
/*
 CBU_FORK_MIN_ADDR3_LSB 
 b'CBU RD FORK MIN Address3 LSB'
*/
typedef struct reg_cbu_fork_min_addr3_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_min_addr3_lsb;
static_assert((sizeof(struct reg_cbu_fork_min_addr3_lsb) == 4), "reg_cbu_fork_min_addr3_lsb size is not 32-bit");
/*
 CBU_FORK_MIN_ADDR3_MSB 
 b'CBU RD FORK MIN Address3 MSB'
*/
typedef struct reg_cbu_fork_min_addr3_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_min_addr3_msb;
static_assert((sizeof(struct reg_cbu_fork_min_addr3_msb) == 4), "reg_cbu_fork_min_addr3_msb size is not 32-bit");
/*
 CBU_FORK_MAX_ADDR3_LSB 
 b'CBU RD FORK MAX Address3 LSB'
*/
typedef struct reg_cbu_fork_max_addr3_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_max_addr3_lsb;
static_assert((sizeof(struct reg_cbu_fork_max_addr3_lsb) == 4), "reg_cbu_fork_max_addr3_lsb size is not 32-bit");
/*
 CBU_FORK_MAX_ADDR3_MSB 
 b'CBU RD FORK MAX Address3 MSB'
*/
typedef struct reg_cbu_fork_max_addr3_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_max_addr3_msb;
static_assert((sizeof(struct reg_cbu_fork_max_addr3_msb) == 4), "reg_cbu_fork_max_addr3_msb size is not 32-bit");
/*
 CBU_FORK_MASK_ADDR3_LSB 
 b'CBU RD FORK Mask Address3 LSB'
*/
typedef struct reg_cbu_fork_mask_addr3_lsb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_mask_addr3_lsb;
static_assert((sizeof(struct reg_cbu_fork_mask_addr3_lsb) == 4), "reg_cbu_fork_mask_addr3_lsb size is not 32-bit");
/*
 CBU_FORK_MASK_ADDR3_MSB 
 b'CBU RD FORK Mask Address3 MSB'
*/
typedef struct reg_cbu_fork_mask_addr3_msb {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_mask_addr3_msb;
static_assert((sizeof(struct reg_cbu_fork_mask_addr3_msb) == 4), "reg_cbu_fork_mask_addr3_msb size is not 32-bit");
/*
 CBU_FORK_TERMINATE_BRESP 
 b'CBU RD FORK TERMINATE BRESP Value'
*/
typedef struct reg_cbu_fork_terminate_bresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_terminate_bresp;
static_assert((sizeof(struct reg_cbu_fork_terminate_bresp) == 4), "reg_cbu_fork_terminate_bresp size is not 32-bit");
/*
 CBU_FORK_TERMINATE_RRESP 
 b'CBU RD FORK TERMINATE RRESP Value'
*/
typedef struct reg_cbu_fork_terminate_rresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_terminate_rresp;
static_assert((sizeof(struct reg_cbu_fork_terminate_rresp) == 4), "reg_cbu_fork_terminate_rresp size is not 32-bit");
/*
 CBU_FORK_ARADDR_INTR_INFO 
 b'CBU RD FORK ARADDR INTR INFO'
*/
typedef struct reg_cbu_fork_araddr_intr_info {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_araddr_intr_info;
static_assert((sizeof(struct reg_cbu_fork_araddr_intr_info) == 4), "reg_cbu_fork_araddr_intr_info size is not 32-bit");
/*
 CBU_FORK_AWADDR_INTR_INFO 
 b'CBU RD FORK AWADDR INTR INFO'
*/
typedef struct reg_cbu_fork_awaddr_intr_info {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cbu_fork_awaddr_intr_info;
static_assert((sizeof(struct reg_cbu_fork_awaddr_intr_info) == 4), "reg_cbu_fork_awaddr_intr_info size is not 32-bit");
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
 CFG_CBU_TERMINATOR_BRESP 
 b'CFG CBU TERMINATOR BRESP VAL'
*/
typedef struct reg_cfg_cbu_terminator_bresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cfg_cbu_terminator_bresp;
static_assert((sizeof(struct reg_cfg_cbu_terminator_bresp) == 4), "reg_cfg_cbu_terminator_bresp size is not 32-bit");
/*
 CFG_CBU_TERMINATOR_RRESP 
 b'CFG CBU TERMINATOR RRESP VAL'
*/
typedef struct reg_cfg_cbu_terminator_rresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_cfg_cbu_terminator_rresp;
static_assert((sizeof(struct reg_cfg_cbu_terminator_rresp) == 4), "reg_cfg_cbu_terminator_rresp size is not 32-bit");
/*
 ARC_REGION_CFG 
 b'ARC Region Config (ASID, MMU_BP)'
*/
typedef struct reg_arc_region_cfg {
	union {
		struct {
			uint32_t asid : 10,
				_reserved31 : 21,
				mmu_bp : 1;
		};
		uint32_t _raw;
	};
} reg_arc_region_cfg;
static_assert((sizeof(struct reg_arc_region_cfg) == 4), "reg_arc_region_cfg size is not 32-bit");
/*
 ARC_CBU_ARPROT_0_CFG 
 b'ARC CBU ARPROT[0] CFG'
*/
typedef struct reg_arc_cbu_arprot_0_cfg {
	union {
		struct {
			uint32_t prot_val : 1,
				_reserved4 : 3,
				prot_val_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_arc_cbu_arprot_0_cfg;
static_assert((sizeof(struct reg_arc_cbu_arprot_0_cfg) == 4), "reg_arc_cbu_arprot_0_cfg size is not 32-bit");
/*
 ARC_CBU_ARPROT_1_CFG 
 b'ARC CBU ARPROT[1] CFG'
*/
typedef struct reg_arc_cbu_arprot_1_cfg {
	union {
		struct {
			uint32_t prot_val : 1,
				_reserved4 : 3,
				prot_val_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_arc_cbu_arprot_1_cfg;
static_assert((sizeof(struct reg_arc_cbu_arprot_1_cfg) == 4), "reg_arc_cbu_arprot_1_cfg size is not 32-bit");
/*
 ARC_CBU_ARPROT_2_CFG 
 b'ARC CBU ARPROT[2] CFG'
*/
typedef struct reg_arc_cbu_arprot_2_cfg {
	union {
		struct {
			uint32_t prot_val : 1,
				_reserved4 : 3,
				prot_val_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_arc_cbu_arprot_2_cfg;
static_assert((sizeof(struct reg_arc_cbu_arprot_2_cfg) == 4), "reg_arc_cbu_arprot_2_cfg size is not 32-bit");
/*
 ARC_CBU_AWPROT_0_CFG 
 b'ARC CBU AWPROT[0] CFG'
*/
typedef struct reg_arc_cbu_awprot_0_cfg {
	union {
		struct {
			uint32_t prot_val : 1,
				_reserved4 : 3,
				prot_val_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_arc_cbu_awprot_0_cfg;
static_assert((sizeof(struct reg_arc_cbu_awprot_0_cfg) == 4), "reg_arc_cbu_awprot_0_cfg size is not 32-bit");
/*
 ARC_CBU_AWPROT_1_CFG 
 b'ARC CBU AWPROT[1] CFG'
*/
typedef struct reg_arc_cbu_awprot_1_cfg {
	union {
		struct {
			uint32_t prot_val : 1,
				_reserved4 : 3,
				prot_val_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_arc_cbu_awprot_1_cfg;
static_assert((sizeof(struct reg_arc_cbu_awprot_1_cfg) == 4), "reg_arc_cbu_awprot_1_cfg size is not 32-bit");
/*
 ARC_CBU_AWPROT_2_CFG 
 b'ARC CBU AWPROT[2] CFG'
*/
typedef struct reg_arc_cbu_awprot_2_cfg {
	union {
		struct {
			uint32_t prot_val : 1,
				_reserved4 : 3,
				prot_val_en : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_arc_cbu_awprot_2_cfg;
static_assert((sizeof(struct reg_arc_cbu_awprot_2_cfg) == 4), "reg_arc_cbu_awprot_2_cfg size is not 32-bit");
/*
 DCCM_TRMINATE_AWADDR_ERR 
 b'DCCM TRMINATE AWADDR ERR'
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
 b'DCCM TRMINATE ARADDR ERR'
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
 b'CFG LBW TERMINATE BRESP VAL'
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
 b'CFG LBW TERMINATE RRESP VAL'
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
 b'DCCM TERMINATE ENABLE'
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
 CFG_DBG_APB_TERMINATE_EN 
 b'ARC DBG APB TERMINATE ENABLE'
*/
typedef struct reg_cfg_dbg_apb_terminate_en {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cfg_dbg_apb_terminate_en;
static_assert((sizeof(struct reg_cfg_dbg_apb_terminate_en) == 4), "reg_cfg_dbg_apb_terminate_en size is not 32-bit");
/*
 CFG_DBG_APB_TERMINATE_RESP 
 b'ARC DBG APB TERMINATE RESP VAL'
*/
typedef struct reg_cfg_dbg_apb_terminate_resp {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cfg_dbg_apb_terminate_resp;
static_assert((sizeof(struct reg_cfg_dbg_apb_terminate_resp) == 4), "reg_cfg_dbg_apb_terminate_resp size is not 32-bit");
/*
 ARC_AXI_ORDERING_WR_IF_CNT 
 b'AXI ordering In flight wr trans counter'
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
 b'Enable axi ordering, set wr to rd delay'
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
 b'Selects relevant portion of the addr'
*/
typedef struct reg_arc_axi_ordering_addr_msk {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_arc_axi_ordering_addr_msk;
static_assert((sizeof(struct reg_arc_axi_ordering_addr_msk) == 4), "reg_arc_axi_ordering_addr_msk size is not 32-bit");
/*
 ARC_AXI_ORDERING_ADDR 
 b'Addr to activate the axi reordering'
*/
typedef struct reg_arc_axi_ordering_addr {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_arc_axi_ordering_addr;
static_assert((sizeof(struct reg_arc_axi_ordering_addr) == 4), "reg_arc_axi_ordering_addr size is not 32-bit");
/*
 ARC_ACC_ENGS_BUSER 
 b'Buser data to reply for axi trans from arc'
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
 ACC_ENGS_FORK_VIRTUAL_WIN_EN 
 b'ARC ACC Engs VIRTUAL FORK Window Enable'
*/
typedef struct reg_acc_engs_fork_virtual_win_en {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_acc_engs_fork_virtual_win_en;
static_assert((sizeof(struct reg_acc_engs_fork_virtual_win_en) == 4), "reg_acc_engs_fork_virtual_win_en size is not 32-bit");
/*
 ACC_ENGS_FORK_VIRTUAL_ADDR_MASK 
 b'ARC ACC Engs VIRTUAL FORK ADDR MASK'
*/
typedef struct reg_acc_engs_fork_virtual_addr_mask {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_fork_virtual_addr_mask;
static_assert((sizeof(struct reg_acc_engs_fork_virtual_addr_mask) == 4), "reg_acc_engs_fork_virtual_addr_mask size is not 32-bit");
/*
 ACC_ENGS_FORK_VIRTUAL_ADDR_MIN 
 b'ARC ACC Engs VIRTUAL FORK ADDR MIN'
*/
typedef struct reg_acc_engs_fork_virtual_addr_min {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_fork_virtual_addr_min;
static_assert((sizeof(struct reg_acc_engs_fork_virtual_addr_min) == 4), "reg_acc_engs_fork_virtual_addr_min size is not 32-bit");
/*
 ACC_ENGS_FORK_VIRTUAL_ADDR_MAX 
 b'ARC ACC Engs VIRTUAL FORK ADDR MAX'
*/
typedef struct reg_acc_engs_fork_virtual_addr_max {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_fork_virtual_addr_max;
static_assert((sizeof(struct reg_acc_engs_fork_virtual_addr_max) == 4), "reg_acc_engs_fork_virtual_addr_max size is not 32-bit");
/*
 ACC_ENGS_FORK_VIRTUAL_TERM_BRESP 
 b'ARC ACC Engs VIRTUAL FORK Terminator BRESP value'
*/
typedef struct reg_acc_engs_fork_virtual_term_bresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_acc_engs_fork_virtual_term_bresp;
static_assert((sizeof(struct reg_acc_engs_fork_virtual_term_bresp) == 4), "reg_acc_engs_fork_virtual_term_bresp size is not 32-bit");
/*
 ACC_ENGS_FORK_VIRTUAL_TERM_RRESP 
 b'ARC ACC Engs VIRTUAL FORK Terminator RRESP value'
*/
typedef struct reg_acc_engs_fork_virtual_term_rresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_acc_engs_fork_virtual_term_rresp;
static_assert((sizeof(struct reg_acc_engs_fork_virtual_term_rresp) == 4), "reg_acc_engs_fork_virtual_term_rresp size is not 32-bit");
/*
 ACC_ENGS_FORK_VIRTUAL_ARADDR_INTR_INFO 
 b'ACC ENGS FORK VIRTUAL ARADDR INTR INFO'
*/
typedef struct reg_acc_engs_fork_virtual_araddr_intr_info {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_fork_virtual_araddr_intr_info;
static_assert((sizeof(struct reg_acc_engs_fork_virtual_araddr_intr_info) == 4), "reg_acc_engs_fork_virtual_araddr_intr_info size is not 32-bit");
/*
 ACC_ENGS_FORK_VIRTUAL_AWADDR_INTR_INFO 
 b'ACC ENGS FORK VIRTUAL AWADDR INTR INFO'
*/
typedef struct reg_acc_engs_fork_virtual_awaddr_intr_info {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_fork_virtual_awaddr_intr_info;
static_assert((sizeof(struct reg_acc_engs_fork_virtual_awaddr_intr_info) == 4), "reg_acc_engs_fork_virtual_awaddr_intr_info size is not 32-bit");
/*
 ACC_ENGS_ARC_FORK_WIN_EN 
 b'ACC Engs ARC FORK Window Enable'
*/
typedef struct reg_acc_engs_arc_fork_win_en {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_acc_engs_arc_fork_win_en;
static_assert((sizeof(struct reg_acc_engs_arc_fork_win_en) == 4), "reg_acc_engs_arc_fork_win_en size is not 32-bit");
/*
 ACC_ENGS_ARC_FORK_MIN_ADDR0 
 b'ACC Engs ARC FORK MIN ADDR0'
*/
typedef struct reg_acc_engs_arc_fork_min_addr0 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_arc_fork_min_addr0;
static_assert((sizeof(struct reg_acc_engs_arc_fork_min_addr0) == 4), "reg_acc_engs_arc_fork_min_addr0 size is not 32-bit");
/*
 ACC_ENGS_ARC_FORK_MAX_ADDR0 
 b'ARC ACC Engs FORK MAX ADDR0'
*/
typedef struct reg_acc_engs_arc_fork_max_addr0 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_arc_fork_max_addr0;
static_assert((sizeof(struct reg_acc_engs_arc_fork_max_addr0) == 4), "reg_acc_engs_arc_fork_max_addr0 size is not 32-bit");
/*
 ACC_ENGS_ARC_FORK_MASK_ADDR0 
 b'ACC Engs ARC FORK MASK ADDR0'
*/
typedef struct reg_acc_engs_arc_fork_mask_addr0 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_arc_fork_mask_addr0;
static_assert((sizeof(struct reg_acc_engs_arc_fork_mask_addr0) == 4), "reg_acc_engs_arc_fork_mask_addr0 size is not 32-bit");
/*
 ACC_ENGS_ARC_FORK_MIN_ADDR1 
 b'ACC Engs ARC FORK MIN ADDR1'
*/
typedef struct reg_acc_engs_arc_fork_min_addr1 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_arc_fork_min_addr1;
static_assert((sizeof(struct reg_acc_engs_arc_fork_min_addr1) == 4), "reg_acc_engs_arc_fork_min_addr1 size is not 32-bit");
/*
 ACC_ENGS_ARC_FORK_MAX_ADDR1 
 b'ARC ACC Engs FORK MAX ADDR1'
*/
typedef struct reg_acc_engs_arc_fork_max_addr1 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_arc_fork_max_addr1;
static_assert((sizeof(struct reg_acc_engs_arc_fork_max_addr1) == 4), "reg_acc_engs_arc_fork_max_addr1 size is not 32-bit");
/*
 ACC_ENGS_ARC_FORK_MASK_ADDR1 
 b'ACC Engs ARC FORK MASK ADDR1'
*/
typedef struct reg_acc_engs_arc_fork_mask_addr1 {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_arc_fork_mask_addr1;
static_assert((sizeof(struct reg_acc_engs_arc_fork_mask_addr1) == 4), "reg_acc_engs_arc_fork_mask_addr1 size is not 32-bit");
/*
 ACC_ENGS_ARC_FORK_TERM_BRESP 
 b'ACC Engs ARC FORK Terminator BRESP value'
*/
typedef struct reg_acc_engs_arc_fork_term_bresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_acc_engs_arc_fork_term_bresp;
static_assert((sizeof(struct reg_acc_engs_arc_fork_term_bresp) == 4), "reg_acc_engs_arc_fork_term_bresp size is not 32-bit");
/*
 ACC_ENGS_ARC_FORK_TERM_RRESP 
 b'ACC Engs ARC FORK Terminator RRESP value'
*/
typedef struct reg_acc_engs_arc_fork_term_rresp {
	union {
		struct {
			uint32_t val : 2,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_acc_engs_arc_fork_term_rresp;
static_assert((sizeof(struct reg_acc_engs_arc_fork_term_rresp) == 4), "reg_acc_engs_arc_fork_term_rresp size is not 32-bit");
/*
 ACC_ENGS_ARC_FORK_ARADDR_INTR_INFO 
 b'ACC ENGS ARC FORK ARADDR INTR INFO'
*/
typedef struct reg_acc_engs_arc_fork_araddr_intr_info {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_arc_fork_araddr_intr_info;
static_assert((sizeof(struct reg_acc_engs_arc_fork_araddr_intr_info) == 4), "reg_acc_engs_arc_fork_araddr_intr_info size is not 32-bit");
/*
 ACC_ENGS_ARC_FORK_AWADDR_INTR_INFO 
 b'ACC ENGS ARC FORK AWADDR INTR INFO'
*/
typedef struct reg_acc_engs_arc_fork_awaddr_intr_info {
	union {
		struct {
			uint32_t val : 29,
				_reserved29 : 3;
		};
		uint32_t _raw;
	};
} reg_acc_engs_arc_fork_awaddr_intr_info;
static_assert((sizeof(struct reg_acc_engs_arc_fork_awaddr_intr_info) == 4), "reg_acc_engs_arc_fork_awaddr_intr_info size is not 32-bit");

#ifdef __cplusplus
} /* qman_arc_aux namespace */
#endif

/*
 QMAN_ARC_AUX block
*/

#ifdef __cplusplus

struct block_qman_arc_aux {
	uint32_t _pad0[64];
	struct qman_arc_aux::reg_run_req run_req;
	struct qman_arc_aux::reg_halt_req halt_req;
	struct qman_arc_aux::reg_run_halt_ack run_halt_ack;
	uint32_t _pad268[1];
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
	struct qman_arc_aux::reg_fw_mem_lsb_addr fw_mem_lsb_addr;
	struct qman_arc_aux::reg_fw_mem_msb_addr fw_mem_msb_addr;
	struct qman_arc_aux::reg_vir_mem0_lsb_addr vir_mem0_lsb_addr;
	struct qman_arc_aux::reg_vir_mem0_msb_addr vir_mem0_msb_addr;
	struct qman_arc_aux::reg_vir_mem1_lsb_addr vir_mem1_lsb_addr;
	struct qman_arc_aux::reg_vir_mem1_msb_addr vir_mem1_msb_addr;
	struct qman_arc_aux::reg_vir_mem2_lsb_addr vir_mem2_lsb_addr;
	struct qman_arc_aux::reg_vir_mem2_msb_addr vir_mem2_msb_addr;
	struct qman_arc_aux::reg_vir_mem3_lsb_addr vir_mem3_lsb_addr;
	struct qman_arc_aux::reg_vir_mem3_msb_addr vir_mem3_msb_addr;
	struct qman_arc_aux::reg_vir_mem0_offset vir_mem0_offset;
	struct qman_arc_aux::reg_vir_mem1_offset vir_mem1_offset;
	struct qman_arc_aux::reg_vir_mem2_offset vir_mem2_offset;
	struct qman_arc_aux::reg_vir_mem3_offset vir_mem3_offset;
	struct qman_arc_aux::reg_pcie_lower_lsb_addr pcie_lower_lsb_addr;
	struct qman_arc_aux::reg_pcie_lower_msb_addr pcie_lower_msb_addr;
	struct qman_arc_aux::reg_pcie_upper_lsb_addr pcie_upper_lsb_addr;
	struct qman_arc_aux::reg_pcie_upper_msb_addr pcie_upper_msb_addr;
	struct qman_arc_aux::reg_d2d_hbw_lsb_addr d2d_hbw_lsb_addr;
	struct qman_arc_aux::reg_d2d_hbw_msb_addr d2d_hbw_msb_addr;
	struct qman_arc_aux::reg_general_purpose_lsb_addr general_purpose_lsb_addr[5];
	struct qman_arc_aux::reg_general_purpose_msb_addr general_purpose_msb_addr[5];
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
	struct qman_arc_aux::reg_cbu_aruser_lsb_ovr cbu_aruser_lsb_ovr;
	struct qman_arc_aux::reg_cbu_aruser_lsb_ovr_en cbu_aruser_lsb_ovr_en;
	struct qman_arc_aux::reg_cbu_aruser_msb_ovr cbu_aruser_msb_ovr;
	struct qman_arc_aux::reg_cbu_aruser_msb_ovr_en cbu_aruser_msb_ovr_en;
	uint32_t _pad848[2];
	struct qman_arc_aux::reg_cbu_awuser_lsb_ovr cbu_awuser_lsb_ovr;
	struct qman_arc_aux::reg_cbu_awuser_lsb_ovr_en cbu_awuser_lsb_ovr_en;
	struct qman_arc_aux::reg_cbu_awuser_msb_ovr cbu_awuser_msb_ovr;
	struct qman_arc_aux::reg_cbu_awuser_msb_ovr_en cbu_awuser_msb_ovr_en;
	uint32_t _pad872[2];
	struct qman_arc_aux::reg_cbu_axcache_ovr cbu_axcache_ovr;
	struct qman_arc_aux::reg_cbu_lock_ovr cbu_lock_ovr;
	struct qman_arc_aux::reg_cbu_rd_prot_0_ovr cbu_rd_prot_0_ovr;
	struct qman_arc_aux::reg_cbu_rd_prot_1_ovr cbu_rd_prot_1_ovr;
	struct qman_arc_aux::reg_cbu_rd_prot_2_ovr cbu_rd_prot_2_ovr;
	struct qman_arc_aux::reg_cbu_wr_prot_0_ovr cbu_wr_prot_0_ovr;
	struct qman_arc_aux::reg_cbu_wr_prot_1_ovr cbu_wr_prot_1_ovr;
	struct qman_arc_aux::reg_cbu_wr_prot_2_ovr cbu_wr_prot_2_ovr;
	uint32_t _pad912[7];
	struct qman_arc_aux::reg_cbu_max_outstanding cbu_max_outstanding;
	struct qman_arc_aux::reg_cbu_early_bresp_en cbu_early_bresp_en;
	struct qman_arc_aux::reg_cbu_force_rsp_ok cbu_force_rsp_ok;
	uint32_t _pad952[1];
	struct qman_arc_aux::reg_cbu_no_wr_inflight cbu_no_wr_inflight;
	struct qman_arc_aux::reg_cbu_sei_intr_id cbu_sei_intr_id;
	uint32_t _pad964[15];
	struct qman_arc_aux::reg_lbu_aruser_ovr lbu_aruser_ovr;
	struct qman_arc_aux::reg_lbu_aruser_ovr_en lbu_aruser_ovr_en;
	struct qman_arc_aux::reg_lbu_awuser_ovr lbu_awuser_ovr;
	struct qman_arc_aux::reg_lbu_awuser_ovr_en lbu_awuser_ovr_en;
	uint32_t _pad1040[4];
	struct qman_arc_aux::reg_lbu_axcache_ovr lbu_axcache_ovr;
	struct qman_arc_aux::reg_lbu_lock_ovr lbu_lock_ovr;
	struct qman_arc_aux::reg_lbu_rd_prot_0_ovr lbu_rd_prot_0_ovr;
	struct qman_arc_aux::reg_lbu_rd_prot_1_ovr lbu_rd_prot_1_ovr;
	struct qman_arc_aux::reg_lbu_rd_prot_2_ovr lbu_rd_prot_2_ovr;
	struct qman_arc_aux::reg_lbu_wr_prot_0_ovr lbu_wr_prot_0_ovr;
	struct qman_arc_aux::reg_lbu_wr_prot_1_ovr lbu_wr_prot_1_ovr;
	struct qman_arc_aux::reg_lbu_wr_prot_2_ovr lbu_wr_prot_2_ovr;
	uint32_t _pad1088[7];
	struct qman_arc_aux::reg_lbu_max_outstanding lbu_max_outstanding;
	struct qman_arc_aux::reg_lbu_early_bresp_en lbu_early_bresp_en;
	struct qman_arc_aux::reg_lbu_force_rsp_ok lbu_force_rsp_ok;
	uint32_t _pad1128[1];
	struct qman_arc_aux::reg_lbu_no_wr_inflight lbu_no_wr_inflight;
	struct qman_arc_aux::reg_lbu_sei_intr_id lbu_sei_intr_id;
	uint32_t _pad1140[3];
	struct qman_arc_aux::reg_dccm_queue_base_addr dccm_queue_base_addr[8];
	struct qman_arc_aux::reg_dccm_queue_size dccm_queue_size[8];
	struct qman_arc_aux::reg_dccm_queue_pi dccm_queue_pi[8];
	struct qman_arc_aux::reg_dccm_queue_ci dccm_queue_ci[8];
	struct qman_arc_aux::reg_dccm_queue_push_reg dccm_queue_push_reg[8];
	struct qman_arc_aux::reg_dccm_queue_max_occupancy dccm_queue_max_occupancy[8];
	struct qman_arc_aux::reg_dccm_queue_valid_entries dccm_queue_valid_entries[8];
	struct qman_arc_aux::reg_dccm_queue_early_pi dccm_queue_early_pi[8];
	struct qman_arc_aux::reg_dccm_queue_total_entries dccm_queue_total_entries[8];
	uint32_t _pad1440[16];
	struct qman_arc_aux::reg_general_q_vld_entry_mask general_q_vld_entry_mask;
	struct qman_arc_aux::reg_nic_q_vld_entry_mask nic_q_vld_entry_mask;
	struct qman_arc_aux::reg_dccm_queue_not_empty dccm_queue_not_empty;
	uint32_t _pad1516[1];
	struct qman_arc_aux::reg_dccm_queue_drop_intr_sts dccm_queue_drop_intr_sts;
	struct qman_arc_aux::reg_dccm_queue_drop_intr_clr dccm_queue_drop_intr_clr;
	struct qman_arc_aux::reg_dccm_queue_drop_intr_mask dccm_queue_drop_intr_mask;
	uint32_t _pad1532[9];
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
	struct qman_arc_aux::reg_aux2apb_pprot_0 aux2apb_pprot_0;
	struct qman_arc_aux::reg_aux2apb_pprot_1 aux2apb_pprot_1;
	struct qman_arc_aux::reg_aux2apb_pprot_2 aux2apb_pprot_2;
	struct qman_arc_aux::reg_lbw_fork_araddr_intr_info lbw_fork_araddr_intr_info;
	struct qman_arc_aux::reg_lbw_fork_awaddr_intr_info lbw_fork_awaddr_intr_info;
	struct qman_arc_aux::reg_lbw_fork_win_en lbw_fork_win_en;
	struct qman_arc_aux::reg_qman_lbw_fork_min_addr0 qman_lbw_fork_min_addr0;
	struct qman_arc_aux::reg_qman_lbw_fork_max_addr0 qman_lbw_fork_max_addr0;
	struct qman_arc_aux::reg_qman_lbw_fork_mask_addr0 qman_lbw_fork_mask_addr0;
	struct qman_arc_aux::reg_qman_lbw_fork_min_addr1 qman_lbw_fork_min_addr1;
	struct qman_arc_aux::reg_qman_lbw_fork_max_addr1 qman_lbw_fork_max_addr1;
	struct qman_arc_aux::reg_qman_lbw_fork_mask_addr1 qman_lbw_fork_mask_addr1;
	struct qman_arc_aux::reg_qman_lbw_fork_min_addr2 qman_lbw_fork_min_addr2;
	struct qman_arc_aux::reg_qman_lbw_fork_max_addr2 qman_lbw_fork_max_addr2;
	struct qman_arc_aux::reg_qman_lbw_fork_mask_addr2 qman_lbw_fork_mask_addr2;
	struct qman_arc_aux::reg_qman_lbw_fork_min_addr3 qman_lbw_fork_min_addr3;
	struct qman_arc_aux::reg_qman_lbw_fork_max_addr3 qman_lbw_fork_max_addr3;
	struct qman_arc_aux::reg_qman_lbw_fork_mask_addr3 qman_lbw_fork_mask_addr3;
	struct qman_arc_aux::reg_qman_lbw_fork_min_addr4 qman_lbw_fork_min_addr4;
	struct qman_arc_aux::reg_qman_lbw_fork_max_addr4 qman_lbw_fork_max_addr4;
	struct qman_arc_aux::reg_qman_lbw_fork_mask_addr4 qman_lbw_fork_mask_addr4;
	uint32_t _pad1716[15];
	struct qman_arc_aux::reg_farm_lbw_fork_min_addr0 farm_lbw_fork_min_addr0;
	struct qman_arc_aux::reg_farm_lbw_fork_max_addr0 farm_lbw_fork_max_addr0;
	struct qman_arc_aux::reg_farm_lbw_fork_mask_addr0 farm_lbw_fork_mask_addr0;
	uint32_t _pad1788[1];
	struct qman_arc_aux::reg_farm_lbw_fork_min_addr1 farm_lbw_fork_min_addr1;
	struct qman_arc_aux::reg_farm_lbw_fork_max_addr1 farm_lbw_fork_max_addr1;
	struct qman_arc_aux::reg_farm_lbw_fork_mask_addr1 farm_lbw_fork_mask_addr1;
	struct qman_arc_aux::reg_farm_lbw_fork_min_addr2 farm_lbw_fork_min_addr2;
	struct qman_arc_aux::reg_farm_lbw_fork_max_addr2 farm_lbw_fork_max_addr2;
	struct qman_arc_aux::reg_farm_lbw_fork_mask_addr2 farm_lbw_fork_mask_addr2;
	struct qman_arc_aux::reg_farm_lbw_fork_min_addr3 farm_lbw_fork_min_addr3;
	struct qman_arc_aux::reg_farm_lbw_fork_max_addr3 farm_lbw_fork_max_addr3;
	struct qman_arc_aux::reg_farm_lbw_fork_mask_addr3 farm_lbw_fork_mask_addr3;
	struct qman_arc_aux::reg_farm_lbw_fork_min_addr4 farm_lbw_fork_min_addr4;
	struct qman_arc_aux::reg_farm_lbw_fork_max_addr4 farm_lbw_fork_max_addr4;
	struct qman_arc_aux::reg_farm_lbw_fork_mask_addr4 farm_lbw_fork_mask_addr4;
	struct qman_arc_aux::reg_lbw_fork_terminate_bresp lbw_fork_terminate_bresp;
	struct qman_arc_aux::reg_lbw_fork_terminate_rresp lbw_fork_terminate_rresp;
	struct qman_arc_aux::reg_lbw_dup_push_terminator_bresp lbw_dup_push_terminator_bresp;
	struct qman_arc_aux::reg_lbw_dup_push_terminator_rresp lbw_dup_push_terminator_rresp;
	struct qman_arc_aux::reg_lbw_dup_cfg_terminator_bresp lbw_dup_cfg_terminator_bresp;
	struct qman_arc_aux::reg_lbw_dup_cfg_terminator_rresp lbw_dup_cfg_terminator_rresp;
	struct qman_arc_aux::reg_lbw_af_cfg_terminator_bresp lbw_af_cfg_terminator_bresp;
	struct qman_arc_aux::reg_lbw_af_cfg_terminator_rresp lbw_af_cfg_terminator_rresp;
	struct qman_arc_aux::reg_lbw_axi2apb_cfg lbw_axi2apb_cfg;
	struct qman_arc_aux::reg_lbw_apb_fork_max_addr0 lbw_apb_fork_max_addr0;
	struct qman_arc_aux::reg_lbw_apb_fork_max_addr1 lbw_apb_fork_max_addr1;
	struct qman_arc_aux::reg_cbu_fork_win_en cbu_fork_win_en;
	struct qman_arc_aux::reg_cbu_fork_min_addr0_lsb cbu_fork_min_addr0_lsb;
	struct qman_arc_aux::reg_cbu_fork_min_addr0_msb cbu_fork_min_addr0_msb;
	struct qman_arc_aux::reg_cbu_fork_max_addr0_lsb cbu_fork_max_addr0_lsb;
	struct qman_arc_aux::reg_cbu_fork_max_addr0_msb cbu_fork_max_addr0_msb;
	struct qman_arc_aux::reg_cbu_fork_mask_addr0_lsb cbu_fork_mask_addr0_lsb;
	struct qman_arc_aux::reg_cbu_fork_mask_addr0_msb cbu_fork_mask_addr0_msb;
	struct qman_arc_aux::reg_cbu_fork_min_addr1_lsb cbu_fork_min_addr1_lsb;
	struct qman_arc_aux::reg_cbu_fork_min_addr1_msb cbu_fork_min_addr1_msb;
	struct qman_arc_aux::reg_cbu_fork_max_addr1_lsb cbu_fork_max_addr1_lsb;
	struct qman_arc_aux::reg_cbu_fork_max_addr1_msb cbu_fork_max_addr1_msb;
	struct qman_arc_aux::reg_cbu_fork_mask_addr1_lsb cbu_fork_mask_addr1_lsb;
	struct qman_arc_aux::reg_cbu_fork_mask_addr1_msb cbu_fork_mask_addr1_msb;
	struct qman_arc_aux::reg_cbu_fork_min_addr2_lsb cbu_fork_min_addr2_lsb;
	struct qman_arc_aux::reg_cbu_fork_min_addr2_msb cbu_fork_min_addr2_msb;
	struct qman_arc_aux::reg_cbu_fork_max_addr2_lsb cbu_fork_max_addr2_lsb;
	struct qman_arc_aux::reg_cbu_fork_max_addr2_msb cbu_fork_max_addr2_msb;
	struct qman_arc_aux::reg_cbu_fork_mask_addr2_lsb cbu_fork_mask_addr2_lsb;
	struct qman_arc_aux::reg_cbu_fork_mask_addr2_msb cbu_fork_mask_addr2_msb;
	struct qman_arc_aux::reg_cbu_fork_min_addr3_lsb cbu_fork_min_addr3_lsb;
	struct qman_arc_aux::reg_cbu_fork_min_addr3_msb cbu_fork_min_addr3_msb;
	struct qman_arc_aux::reg_cbu_fork_max_addr3_lsb cbu_fork_max_addr3_lsb;
	struct qman_arc_aux::reg_cbu_fork_max_addr3_msb cbu_fork_max_addr3_msb;
	struct qman_arc_aux::reg_cbu_fork_mask_addr3_lsb cbu_fork_mask_addr3_lsb;
	struct qman_arc_aux::reg_cbu_fork_mask_addr3_msb cbu_fork_mask_addr3_msb;
	struct qman_arc_aux::reg_cbu_fork_terminate_bresp cbu_fork_terminate_bresp;
	struct qman_arc_aux::reg_cbu_fork_terminate_rresp cbu_fork_terminate_rresp;
	struct qman_arc_aux::reg_cbu_fork_araddr_intr_info cbu_fork_araddr_intr_info[2];
	struct qman_arc_aux::reg_cbu_fork_awaddr_intr_info cbu_fork_awaddr_intr_info[2];
	struct qman_arc_aux::reg_cbu_trminate_araddr_lsb cbu_trminate_araddr_lsb;
	struct qman_arc_aux::reg_cbu_trminate_araddr_msb cbu_trminate_araddr_msb;
	struct qman_arc_aux::reg_cfg_cbu_terminator_bresp cfg_cbu_terminator_bresp;
	struct qman_arc_aux::reg_cfg_cbu_terminator_rresp cfg_cbu_terminator_rresp;
	uint32_t _pad2024[6];
	struct qman_arc_aux::reg_arc_region_cfg arc_region_cfg;
	struct qman_arc_aux::reg_arc_cbu_arprot_0_cfg arc_cbu_arprot_0_cfg;
	struct qman_arc_aux::reg_arc_cbu_arprot_1_cfg arc_cbu_arprot_1_cfg;
	struct qman_arc_aux::reg_arc_cbu_arprot_2_cfg arc_cbu_arprot_2_cfg;
	struct qman_arc_aux::reg_arc_cbu_awprot_0_cfg arc_cbu_awprot_0_cfg;
	struct qman_arc_aux::reg_arc_cbu_awprot_1_cfg arc_cbu_awprot_1_cfg;
	struct qman_arc_aux::reg_arc_cbu_awprot_2_cfg arc_cbu_awprot_2_cfg;
	uint32_t _pad2076[9];
	struct qman_arc_aux::reg_dccm_trminate_awaddr_err dccm_trminate_awaddr_err;
	struct qman_arc_aux::reg_dccm_trminate_araddr_err dccm_trminate_araddr_err;
	struct qman_arc_aux::reg_cfg_dccm_terminate_bresp cfg_dccm_terminate_bresp;
	struct qman_arc_aux::reg_cfg_dccm_terminate_rresp cfg_dccm_terminate_rresp;
	struct qman_arc_aux::reg_cfg_dccm_terminate_en cfg_dccm_terminate_en;
	uint32_t _pad2132[3];
	struct qman_arc_aux::reg_cfg_dbg_apb_terminate_en cfg_dbg_apb_terminate_en;
	struct qman_arc_aux::reg_cfg_dbg_apb_terminate_resp cfg_dbg_apb_terminate_resp;
	uint32_t _pad2152[38];
	struct qman_arc_aux::reg_arc_axi_ordering_wr_if_cnt arc_axi_ordering_wr_if_cnt;
	struct qman_arc_aux::reg_arc_axi_ordering_ctl arc_axi_ordering_ctl;
	struct qman_arc_aux::reg_arc_axi_ordering_addr_msk arc_axi_ordering_addr_msk;
	struct qman_arc_aux::reg_arc_axi_ordering_addr arc_axi_ordering_addr;
	struct qman_arc_aux::reg_arc_acc_engs_buser arc_acc_engs_buser;
	uint32_t _pad2324[2];
	struct qman_arc_aux::reg_acc_engs_fork_virtual_win_en acc_engs_fork_virtual_win_en;
	struct qman_arc_aux::reg_acc_engs_fork_virtual_addr_mask acc_engs_fork_virtual_addr_mask;
	struct qman_arc_aux::reg_acc_engs_fork_virtual_addr_min acc_engs_fork_virtual_addr_min;
	uint32_t _pad2344[1];
	struct qman_arc_aux::reg_acc_engs_fork_virtual_addr_max acc_engs_fork_virtual_addr_max;
	struct qman_arc_aux::reg_acc_engs_fork_virtual_term_bresp acc_engs_fork_virtual_term_bresp;
	struct qman_arc_aux::reg_acc_engs_fork_virtual_term_rresp acc_engs_fork_virtual_term_rresp;
	struct qman_arc_aux::reg_acc_engs_fork_virtual_araddr_intr_info acc_engs_fork_virtual_araddr_intr_info;
	struct qman_arc_aux::reg_acc_engs_fork_virtual_awaddr_intr_info acc_engs_fork_virtual_awaddr_intr_info;
	struct qman_arc_aux::reg_acc_engs_arc_fork_win_en acc_engs_arc_fork_win_en;
	struct qman_arc_aux::reg_acc_engs_arc_fork_min_addr0 acc_engs_arc_fork_min_addr0;
	struct qman_arc_aux::reg_acc_engs_arc_fork_max_addr0 acc_engs_arc_fork_max_addr0;
	struct qman_arc_aux::reg_acc_engs_arc_fork_mask_addr0 acc_engs_arc_fork_mask_addr0;
	struct qman_arc_aux::reg_acc_engs_arc_fork_min_addr1 acc_engs_arc_fork_min_addr1;
	uint32_t _pad2388[1];
	struct qman_arc_aux::reg_acc_engs_arc_fork_max_addr1 acc_engs_arc_fork_max_addr1;
	struct qman_arc_aux::reg_acc_engs_arc_fork_mask_addr1 acc_engs_arc_fork_mask_addr1;
	struct qman_arc_aux::reg_acc_engs_arc_fork_term_bresp acc_engs_arc_fork_term_bresp;
	struct qman_arc_aux::reg_acc_engs_arc_fork_term_rresp acc_engs_arc_fork_term_rresp;
	struct qman_arc_aux::reg_acc_engs_arc_fork_araddr_intr_info acc_engs_arc_fork_araddr_intr_info;
	struct qman_arc_aux::reg_acc_engs_arc_fork_awaddr_intr_info acc_engs_arc_fork_awaddr_intr_info;
	uint32_t _pad2416[324];
	struct block_special_regs special;
};
#else

typedef struct block_qman_arc_aux {
	uint32_t _pad0[64];
	reg_run_req run_req;
	reg_halt_req halt_req;
	reg_run_halt_ack run_halt_ack;
	uint32_t _pad268[1];
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
	reg_fw_mem_lsb_addr fw_mem_lsb_addr;
	reg_fw_mem_msb_addr fw_mem_msb_addr;
	reg_vir_mem0_lsb_addr vir_mem0_lsb_addr;
	reg_vir_mem0_msb_addr vir_mem0_msb_addr;
	reg_vir_mem1_lsb_addr vir_mem1_lsb_addr;
	reg_vir_mem1_msb_addr vir_mem1_msb_addr;
	reg_vir_mem2_lsb_addr vir_mem2_lsb_addr;
	reg_vir_mem2_msb_addr vir_mem2_msb_addr;
	reg_vir_mem3_lsb_addr vir_mem3_lsb_addr;
	reg_vir_mem3_msb_addr vir_mem3_msb_addr;
	reg_vir_mem0_offset vir_mem0_offset;
	reg_vir_mem1_offset vir_mem1_offset;
	reg_vir_mem2_offset vir_mem2_offset;
	reg_vir_mem3_offset vir_mem3_offset;
	reg_pcie_lower_lsb_addr pcie_lower_lsb_addr;
	reg_pcie_lower_msb_addr pcie_lower_msb_addr;
	reg_pcie_upper_lsb_addr pcie_upper_lsb_addr;
	reg_pcie_upper_msb_addr pcie_upper_msb_addr;
	reg_d2d_hbw_lsb_addr d2d_hbw_lsb_addr;
	reg_d2d_hbw_msb_addr d2d_hbw_msb_addr;
	reg_general_purpose_lsb_addr general_purpose_lsb_addr[5];
	reg_general_purpose_msb_addr general_purpose_msb_addr[5];
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
	reg_cbu_aruser_lsb_ovr cbu_aruser_lsb_ovr;
	reg_cbu_aruser_lsb_ovr_en cbu_aruser_lsb_ovr_en;
	reg_cbu_aruser_msb_ovr cbu_aruser_msb_ovr;
	reg_cbu_aruser_msb_ovr_en cbu_aruser_msb_ovr_en;
	uint32_t _pad848[2];
	reg_cbu_awuser_lsb_ovr cbu_awuser_lsb_ovr;
	reg_cbu_awuser_lsb_ovr_en cbu_awuser_lsb_ovr_en;
	reg_cbu_awuser_msb_ovr cbu_awuser_msb_ovr;
	reg_cbu_awuser_msb_ovr_en cbu_awuser_msb_ovr_en;
	uint32_t _pad872[2];
	reg_cbu_axcache_ovr cbu_axcache_ovr;
	reg_cbu_lock_ovr cbu_lock_ovr;
	reg_cbu_rd_prot_0_ovr cbu_rd_prot_0_ovr;
	reg_cbu_rd_prot_1_ovr cbu_rd_prot_1_ovr;
	reg_cbu_rd_prot_2_ovr cbu_rd_prot_2_ovr;
	reg_cbu_wr_prot_0_ovr cbu_wr_prot_0_ovr;
	reg_cbu_wr_prot_1_ovr cbu_wr_prot_1_ovr;
	reg_cbu_wr_prot_2_ovr cbu_wr_prot_2_ovr;
	uint32_t _pad912[7];
	reg_cbu_max_outstanding cbu_max_outstanding;
	reg_cbu_early_bresp_en cbu_early_bresp_en;
	reg_cbu_force_rsp_ok cbu_force_rsp_ok;
	uint32_t _pad952[1];
	reg_cbu_no_wr_inflight cbu_no_wr_inflight;
	reg_cbu_sei_intr_id cbu_sei_intr_id;
	uint32_t _pad964[15];
	reg_lbu_aruser_ovr lbu_aruser_ovr;
	reg_lbu_aruser_ovr_en lbu_aruser_ovr_en;
	reg_lbu_awuser_ovr lbu_awuser_ovr;
	reg_lbu_awuser_ovr_en lbu_awuser_ovr_en;
	uint32_t _pad1040[4];
	reg_lbu_axcache_ovr lbu_axcache_ovr;
	reg_lbu_lock_ovr lbu_lock_ovr;
	reg_lbu_rd_prot_0_ovr lbu_rd_prot_0_ovr;
	reg_lbu_rd_prot_1_ovr lbu_rd_prot_1_ovr;
	reg_lbu_rd_prot_2_ovr lbu_rd_prot_2_ovr;
	reg_lbu_wr_prot_0_ovr lbu_wr_prot_0_ovr;
	reg_lbu_wr_prot_1_ovr lbu_wr_prot_1_ovr;
	reg_lbu_wr_prot_2_ovr lbu_wr_prot_2_ovr;
	uint32_t _pad1088[7];
	reg_lbu_max_outstanding lbu_max_outstanding;
	reg_lbu_early_bresp_en lbu_early_bresp_en;
	reg_lbu_force_rsp_ok lbu_force_rsp_ok;
	uint32_t _pad1128[1];
	reg_lbu_no_wr_inflight lbu_no_wr_inflight;
	reg_lbu_sei_intr_id lbu_sei_intr_id;
	uint32_t _pad1140[3];
	reg_dccm_queue_base_addr dccm_queue_base_addr[8];
	reg_dccm_queue_size dccm_queue_size[8];
	reg_dccm_queue_pi dccm_queue_pi[8];
	reg_dccm_queue_ci dccm_queue_ci[8];
	reg_dccm_queue_push_reg dccm_queue_push_reg[8];
	reg_dccm_queue_max_occupancy dccm_queue_max_occupancy[8];
	reg_dccm_queue_valid_entries dccm_queue_valid_entries[8];
	reg_dccm_queue_early_pi dccm_queue_early_pi[8];
	reg_dccm_queue_total_entries dccm_queue_total_entries[8];
	uint32_t _pad1440[16];
	reg_general_q_vld_entry_mask general_q_vld_entry_mask;
	reg_nic_q_vld_entry_mask nic_q_vld_entry_mask;
	reg_dccm_queue_not_empty dccm_queue_not_empty;
	uint32_t _pad1516[1];
	reg_dccm_queue_drop_intr_sts dccm_queue_drop_intr_sts;
	reg_dccm_queue_drop_intr_clr dccm_queue_drop_intr_clr;
	reg_dccm_queue_drop_intr_mask dccm_queue_drop_intr_mask;
	uint32_t _pad1532[9];
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
	reg_aux2apb_pprot_0 aux2apb_pprot_0;
	reg_aux2apb_pprot_1 aux2apb_pprot_1;
	reg_aux2apb_pprot_2 aux2apb_pprot_2;
	reg_lbw_fork_araddr_intr_info lbw_fork_araddr_intr_info;
	reg_lbw_fork_awaddr_intr_info lbw_fork_awaddr_intr_info;
	reg_lbw_fork_win_en lbw_fork_win_en;
	reg_qman_lbw_fork_min_addr0 qman_lbw_fork_min_addr0;
	reg_qman_lbw_fork_max_addr0 qman_lbw_fork_max_addr0;
	reg_qman_lbw_fork_mask_addr0 qman_lbw_fork_mask_addr0;
	reg_qman_lbw_fork_min_addr1 qman_lbw_fork_min_addr1;
	reg_qman_lbw_fork_max_addr1 qman_lbw_fork_max_addr1;
	reg_qman_lbw_fork_mask_addr1 qman_lbw_fork_mask_addr1;
	reg_qman_lbw_fork_min_addr2 qman_lbw_fork_min_addr2;
	reg_qman_lbw_fork_max_addr2 qman_lbw_fork_max_addr2;
	reg_qman_lbw_fork_mask_addr2 qman_lbw_fork_mask_addr2;
	reg_qman_lbw_fork_min_addr3 qman_lbw_fork_min_addr3;
	reg_qman_lbw_fork_max_addr3 qman_lbw_fork_max_addr3;
	reg_qman_lbw_fork_mask_addr3 qman_lbw_fork_mask_addr3;
	reg_qman_lbw_fork_min_addr4 qman_lbw_fork_min_addr4;
	reg_qman_lbw_fork_max_addr4 qman_lbw_fork_max_addr4;
	reg_qman_lbw_fork_mask_addr4 qman_lbw_fork_mask_addr4;
	uint32_t _pad1716[15];
	reg_farm_lbw_fork_min_addr0 farm_lbw_fork_min_addr0;
	reg_farm_lbw_fork_max_addr0 farm_lbw_fork_max_addr0;
	reg_farm_lbw_fork_mask_addr0 farm_lbw_fork_mask_addr0;
	uint32_t _pad1788[1];
	reg_farm_lbw_fork_min_addr1 farm_lbw_fork_min_addr1;
	reg_farm_lbw_fork_max_addr1 farm_lbw_fork_max_addr1;
	reg_farm_lbw_fork_mask_addr1 farm_lbw_fork_mask_addr1;
	reg_farm_lbw_fork_min_addr2 farm_lbw_fork_min_addr2;
	reg_farm_lbw_fork_max_addr2 farm_lbw_fork_max_addr2;
	reg_farm_lbw_fork_mask_addr2 farm_lbw_fork_mask_addr2;
	reg_farm_lbw_fork_min_addr3 farm_lbw_fork_min_addr3;
	reg_farm_lbw_fork_max_addr3 farm_lbw_fork_max_addr3;
	reg_farm_lbw_fork_mask_addr3 farm_lbw_fork_mask_addr3;
	reg_farm_lbw_fork_min_addr4 farm_lbw_fork_min_addr4;
	reg_farm_lbw_fork_max_addr4 farm_lbw_fork_max_addr4;
	reg_farm_lbw_fork_mask_addr4 farm_lbw_fork_mask_addr4;
	reg_lbw_fork_terminate_bresp lbw_fork_terminate_bresp;
	reg_lbw_fork_terminate_rresp lbw_fork_terminate_rresp;
	reg_lbw_dup_push_terminator_bresp lbw_dup_push_terminator_bresp;
	reg_lbw_dup_push_terminator_rresp lbw_dup_push_terminator_rresp;
	reg_lbw_dup_cfg_terminator_bresp lbw_dup_cfg_terminator_bresp;
	reg_lbw_dup_cfg_terminator_rresp lbw_dup_cfg_terminator_rresp;
	reg_lbw_af_cfg_terminator_bresp lbw_af_cfg_terminator_bresp;
	reg_lbw_af_cfg_terminator_rresp lbw_af_cfg_terminator_rresp;
	reg_lbw_axi2apb_cfg lbw_axi2apb_cfg;
	reg_lbw_apb_fork_max_addr0 lbw_apb_fork_max_addr0;
	reg_lbw_apb_fork_max_addr1 lbw_apb_fork_max_addr1;
	reg_cbu_fork_win_en cbu_fork_win_en;
	reg_cbu_fork_min_addr0_lsb cbu_fork_min_addr0_lsb;
	reg_cbu_fork_min_addr0_msb cbu_fork_min_addr0_msb;
	reg_cbu_fork_max_addr0_lsb cbu_fork_max_addr0_lsb;
	reg_cbu_fork_max_addr0_msb cbu_fork_max_addr0_msb;
	reg_cbu_fork_mask_addr0_lsb cbu_fork_mask_addr0_lsb;
	reg_cbu_fork_mask_addr0_msb cbu_fork_mask_addr0_msb;
	reg_cbu_fork_min_addr1_lsb cbu_fork_min_addr1_lsb;
	reg_cbu_fork_min_addr1_msb cbu_fork_min_addr1_msb;
	reg_cbu_fork_max_addr1_lsb cbu_fork_max_addr1_lsb;
	reg_cbu_fork_max_addr1_msb cbu_fork_max_addr1_msb;
	reg_cbu_fork_mask_addr1_lsb cbu_fork_mask_addr1_lsb;
	reg_cbu_fork_mask_addr1_msb cbu_fork_mask_addr1_msb;
	reg_cbu_fork_min_addr2_lsb cbu_fork_min_addr2_lsb;
	reg_cbu_fork_min_addr2_msb cbu_fork_min_addr2_msb;
	reg_cbu_fork_max_addr2_lsb cbu_fork_max_addr2_lsb;
	reg_cbu_fork_max_addr2_msb cbu_fork_max_addr2_msb;
	reg_cbu_fork_mask_addr2_lsb cbu_fork_mask_addr2_lsb;
	reg_cbu_fork_mask_addr2_msb cbu_fork_mask_addr2_msb;
	reg_cbu_fork_min_addr3_lsb cbu_fork_min_addr3_lsb;
	reg_cbu_fork_min_addr3_msb cbu_fork_min_addr3_msb;
	reg_cbu_fork_max_addr3_lsb cbu_fork_max_addr3_lsb;
	reg_cbu_fork_max_addr3_msb cbu_fork_max_addr3_msb;
	reg_cbu_fork_mask_addr3_lsb cbu_fork_mask_addr3_lsb;
	reg_cbu_fork_mask_addr3_msb cbu_fork_mask_addr3_msb;
	reg_cbu_fork_terminate_bresp cbu_fork_terminate_bresp;
	reg_cbu_fork_terminate_rresp cbu_fork_terminate_rresp;
	reg_cbu_fork_araddr_intr_info cbu_fork_araddr_intr_info[2];
	reg_cbu_fork_awaddr_intr_info cbu_fork_awaddr_intr_info[2];
	reg_cbu_trminate_araddr_lsb cbu_trminate_araddr_lsb;
	reg_cbu_trminate_araddr_msb cbu_trminate_araddr_msb;
	reg_cfg_cbu_terminator_bresp cfg_cbu_terminator_bresp;
	reg_cfg_cbu_terminator_rresp cfg_cbu_terminator_rresp;
	uint32_t _pad2024[6];
	reg_arc_region_cfg arc_region_cfg;
	reg_arc_cbu_arprot_0_cfg arc_cbu_arprot_0_cfg;
	reg_arc_cbu_arprot_1_cfg arc_cbu_arprot_1_cfg;
	reg_arc_cbu_arprot_2_cfg arc_cbu_arprot_2_cfg;
	reg_arc_cbu_awprot_0_cfg arc_cbu_awprot_0_cfg;
	reg_arc_cbu_awprot_1_cfg arc_cbu_awprot_1_cfg;
	reg_arc_cbu_awprot_2_cfg arc_cbu_awprot_2_cfg;
	uint32_t _pad2076[9];
	reg_dccm_trminate_awaddr_err dccm_trminate_awaddr_err;
	reg_dccm_trminate_araddr_err dccm_trminate_araddr_err;
	reg_cfg_dccm_terminate_bresp cfg_dccm_terminate_bresp;
	reg_cfg_dccm_terminate_rresp cfg_dccm_terminate_rresp;
	reg_cfg_dccm_terminate_en cfg_dccm_terminate_en;
	uint32_t _pad2132[3];
	reg_cfg_dbg_apb_terminate_en cfg_dbg_apb_terminate_en;
	reg_cfg_dbg_apb_terminate_resp cfg_dbg_apb_terminate_resp;
	uint32_t _pad2152[38];
	reg_arc_axi_ordering_wr_if_cnt arc_axi_ordering_wr_if_cnt;
	reg_arc_axi_ordering_ctl arc_axi_ordering_ctl;
	reg_arc_axi_ordering_addr_msk arc_axi_ordering_addr_msk;
	reg_arc_axi_ordering_addr arc_axi_ordering_addr;
	reg_arc_acc_engs_buser arc_acc_engs_buser;
	uint32_t _pad2324[2];
	reg_acc_engs_fork_virtual_win_en acc_engs_fork_virtual_win_en;
	reg_acc_engs_fork_virtual_addr_mask acc_engs_fork_virtual_addr_mask;
	reg_acc_engs_fork_virtual_addr_min acc_engs_fork_virtual_addr_min;
	uint32_t _pad2344[1];
	reg_acc_engs_fork_virtual_addr_max acc_engs_fork_virtual_addr_max;
	reg_acc_engs_fork_virtual_term_bresp acc_engs_fork_virtual_term_bresp;
	reg_acc_engs_fork_virtual_term_rresp acc_engs_fork_virtual_term_rresp;
	reg_acc_engs_fork_virtual_araddr_intr_info acc_engs_fork_virtual_araddr_intr_info;
	reg_acc_engs_fork_virtual_awaddr_intr_info acc_engs_fork_virtual_awaddr_intr_info;
	reg_acc_engs_arc_fork_win_en acc_engs_arc_fork_win_en;
	reg_acc_engs_arc_fork_min_addr0 acc_engs_arc_fork_min_addr0;
	reg_acc_engs_arc_fork_max_addr0 acc_engs_arc_fork_max_addr0;
	reg_acc_engs_arc_fork_mask_addr0 acc_engs_arc_fork_mask_addr0;
	reg_acc_engs_arc_fork_min_addr1 acc_engs_arc_fork_min_addr1;
	uint32_t _pad2388[1];
	reg_acc_engs_arc_fork_max_addr1 acc_engs_arc_fork_max_addr1;
	reg_acc_engs_arc_fork_mask_addr1 acc_engs_arc_fork_mask_addr1;
	reg_acc_engs_arc_fork_term_bresp acc_engs_arc_fork_term_bresp;
	reg_acc_engs_arc_fork_term_rresp acc_engs_arc_fork_term_rresp;
	reg_acc_engs_arc_fork_araddr_intr_info acc_engs_arc_fork_araddr_intr_info;
	reg_acc_engs_arc_fork_awaddr_intr_info acc_engs_arc_fork_awaddr_intr_info;
	uint32_t _pad2416[324];
	block_special_regs special;
} block_qman_arc_aux;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_qman_arc_aux_defaults[] =
{
	// offset	// value
	{ 0x110 , 0x100000            , 1 }, // rst_vec_addr
	{ 0x114 , 0x10111             , 1 }, // dbg_mode
	{ 0x124 , 0x80000000          , 1 }, // dccm_sys_addr_base
	{ 0x130 , 0x1                 , 1 }, // arc_rst
	{ 0x138 , 0xf                 , 1 }, // sram_lsb_addr
	{ 0x13c , 0x200ffff           , 1 }, // sram_msb_addr
	{ 0x140 , 0xd                 , 1 }, // fw_mem_lsb_addr
	{ 0x144 , 0x300007f           , 1 }, // fw_mem_msb_addr
	{ 0x14c , 0x2010000           , 1 }, // vir_mem0_msb_addr
	{ 0x150 , 0x1                 , 1 }, // vir_mem1_lsb_addr
	{ 0x154 , 0x2010000           , 1 }, // vir_mem1_msb_addr
	{ 0x158 , 0x2                 , 1 }, // vir_mem2_lsb_addr
	{ 0x15c , 0x2010000           , 1 }, // vir_mem2_msb_addr
	{ 0x160 , 0x3                 , 1 }, // vir_mem3_lsb_addr
	{ 0x164 , 0x2010000           , 1 }, // vir_mem3_msb_addr
	{ 0x184 , 0xff000000          , 1 }, // pcie_upper_msb_addr
	{ 0x188 , 0xc                 , 1 }, // d2d_hbw_lsb_addr
	{ 0x18c , 0x300007f           , 1 }, // d2d_hbw_msb_addr
	{ 0x2a0 , 0x1                 , 1 }, // sei_intr_halt_en
	{ 0x2a4 , 0xffffffff          , 1 }, // arc_sei_intr_halt_mask
	{ 0x2a8 , 0xffffffff          , 1 }, // qman_sei_intr_halt_mask
	{ 0x2e8 , 0x3                 , 1 }, // cfg_lbw_terminate_bresp
	{ 0x2ec , 0x3                 , 1 }, // cfg_lbw_terminate_rresp
	{ 0x2f4 , 0x2                 , 1 }, // cfg_lbw_terminate_axsize
	{ 0x340 , 0x40000c00          , 1 }, // cbu_aruser_lsb_ovr
	{ 0x358 , 0x40000c00          , 1 }, // cbu_awuser_lsb_ovr
	{ 0x378 , 0x10                , 1 }, // cbu_rd_prot_0_ovr
	{ 0x37c , 0x11                , 1 }, // cbu_rd_prot_1_ovr
	{ 0x384 , 0x10                , 1 }, // cbu_wr_prot_0_ovr
	{ 0x388 , 0x11                , 1 }, // cbu_wr_prot_1_ovr
	{ 0x3ac , 0x2018              , 1 }, // cbu_max_outstanding
	{ 0x3b0 , 0x1                 , 1 }, // cbu_early_bresp_en
	{ 0x400 , 0xc00               , 1 }, // lbu_aruser_ovr
	{ 0x408 , 0xc00               , 1 }, // lbu_awuser_ovr
	{ 0x428 , 0x10                , 1 }, // lbu_rd_prot_0_ovr
	{ 0x42c , 0x11                , 1 }, // lbu_rd_prot_1_ovr
	{ 0x434 , 0x10                , 1 }, // lbu_wr_prot_0_ovr
	{ 0x438 , 0x11                , 1 }, // lbu_wr_prot_1_ovr
	{ 0x45c , 0x8005              , 1 }, // lbu_max_outstanding
	{ 0x460 , 0x1                 , 1 }, // lbu_early_bresp_en
	{ 0x620 , 0x1                 , 1 }, // dccm_queue_drop_en
	{ 0x624 , 0xabcd              , 1 }, // dccm_queue_warn_msg
	{ 0x628 , 0xefba              , 1 }, // dccm_queue_alert_msg
	{ 0x644 , 0x1e1               , 1 }, // dccm_wrr_arb_weight
	{ 0x648 , 0xc                 , 1 }, // dccm_q_push_fifo_full_cfg
	{ 0x664 , 0x1                 , 1 }, // aux2apb_pprot_1
	{ 0x674 , 0x1f                , 1 }, // lbw_fork_win_en
	{ 0x67c , 0x7fff              , 1 }, // qman_lbw_fork_max_addr0
	{ 0x680 , 0xf000              , 1 }, // qman_lbw_fork_mask_addr0
	{ 0x684 , 0x8000              , 1 }, // qman_lbw_fork_min_addr1
	{ 0x688 , 0x8fff              , 1 }, // qman_lbw_fork_max_addr1
	{ 0x68c , 0xf000              , 1 }, // qman_lbw_fork_mask_addr1
	{ 0x690 , 0x1fffffff          , 1 }, // qman_lbw_fork_min_addr2
	{ 0x698 , 0x1fffffff          , 1 }, // qman_lbw_fork_mask_addr2
	{ 0x69c , 0x1fffffff          , 1 }, // qman_lbw_fork_min_addr3
	{ 0x6a4 , 0x1fffffff          , 1 }, // qman_lbw_fork_mask_addr3
	{ 0x6a8 , 0x1fffffff          , 1 }, // qman_lbw_fork_min_addr4
	{ 0x6b0 , 0x1fffffff          , 1 }, // qman_lbw_fork_mask_addr4
	{ 0x6f0 , 0x10000             , 1 }, // farm_lbw_fork_min_addr0
	{ 0x6f4 , 0x1ffff             , 1 }, // farm_lbw_fork_max_addr0
	{ 0x6f8 , 0x1f000             , 1 }, // farm_lbw_fork_mask_addr0
	{ 0x700 , 0x8000              , 1 }, // farm_lbw_fork_min_addr1
	{ 0x704 , 0x8fff              , 1 }, // farm_lbw_fork_max_addr1
	{ 0x708 , 0x1f000             , 1 }, // farm_lbw_fork_mask_addr1
	{ 0x710 , 0x3fff              , 1 }, // farm_lbw_fork_max_addr2
	{ 0x714 , 0x1f000             , 1 }, // farm_lbw_fork_mask_addr2
	{ 0x718 , 0x4000              , 1 }, // farm_lbw_fork_min_addr3
	{ 0x71c , 0x5fff              , 1 }, // farm_lbw_fork_max_addr3
	{ 0x720 , 0x1f000             , 1 }, // farm_lbw_fork_mask_addr3
	{ 0x724 , 0xd000              , 1 }, // farm_lbw_fork_min_addr4
	{ 0x728 , 0xffff              , 1 }, // farm_lbw_fork_max_addr4
	{ 0x72c , 0x1f000             , 1 }, // farm_lbw_fork_mask_addr4
	{ 0x730 , 0x3                 , 1 }, // lbw_fork_terminate_bresp
	{ 0x734 , 0x3                 , 1 }, // lbw_fork_terminate_rresp
	{ 0x738 , 0x3                 , 1 }, // lbw_dup_push_terminator_bresp
	{ 0x73c , 0x3                 , 1 }, // lbw_dup_push_terminator_rresp
	{ 0x740 , 0x3                 , 1 }, // lbw_dup_cfg_terminator_bresp
	{ 0x744 , 0x3                 , 1 }, // lbw_dup_cfg_terminator_rresp
	{ 0x748 , 0x3                 , 1 }, // lbw_af_cfg_terminator_bresp
	{ 0x74c , 0x3                 , 1 }, // lbw_af_cfg_terminator_rresp
	{ 0x750 , 0x110               , 1 }, // lbw_axi2apb_cfg
	{ 0x754 , 0x7fff              , 1 }, // lbw_apb_fork_max_addr0
	{ 0x758 , 0x8fff              , 1 }, // lbw_apb_fork_max_addr1
	{ 0x75c , 0x1                 , 1 }, // cbu_fork_win_en
	{ 0x760 , 0xe0000000          , 1 }, // cbu_fork_min_addr0_lsb
	{ 0x764 , 0x300007f           , 1 }, // cbu_fork_min_addr0_msb
	{ 0x768 , 0xffffffff          , 1 }, // cbu_fork_max_addr0_lsb
	{ 0x76c , 0x300007f           , 1 }, // cbu_fork_max_addr0_msb
	{ 0x770 , 0xf0000000          , 1 }, // cbu_fork_mask_addr0_lsb
	{ 0x774 , 0xffffffff          , 1 }, // cbu_fork_mask_addr0_msb
	{ 0x778 , 0xffffffff          , 1 }, // cbu_fork_min_addr1_lsb
	{ 0x77c , 0xffffffff          , 1 }, // cbu_fork_min_addr1_msb
	{ 0x788 , 0xffffffff          , 1 }, // cbu_fork_mask_addr1_lsb
	{ 0x78c , 0xffffffff          , 1 }, // cbu_fork_mask_addr1_msb
	{ 0x790 , 0xffffffff          , 1 }, // cbu_fork_min_addr2_lsb
	{ 0x794 , 0xffffffff          , 1 }, // cbu_fork_min_addr2_msb
	{ 0x7a0 , 0xffffffff          , 1 }, // cbu_fork_mask_addr2_lsb
	{ 0x7a4 , 0xffffffff          , 1 }, // cbu_fork_mask_addr2_msb
	{ 0x7a8 , 0xffffffff          , 1 }, // cbu_fork_min_addr3_lsb
	{ 0x7ac , 0xffffffff          , 1 }, // cbu_fork_min_addr3_msb
	{ 0x7b8 , 0xffffffff          , 1 }, // cbu_fork_mask_addr3_lsb
	{ 0x7bc , 0xffffffff          , 1 }, // cbu_fork_mask_addr3_msb
	{ 0x7c0 , 0x3                 , 1 }, // cbu_fork_terminate_bresp
	{ 0x7c4 , 0x3                 , 1 }, // cbu_fork_terminate_rresp
	{ 0x7e0 , 0x3                 , 1 }, // cfg_cbu_terminator_bresp
	{ 0x7e4 , 0x3                 , 1 }, // cfg_cbu_terminator_rresp
	{ 0x804 , 0x10                , 1 }, // arc_cbu_arprot_0_cfg
	{ 0x808 , 0x11                , 1 }, // arc_cbu_arprot_1_cfg
	{ 0x810 , 0x10                , 1 }, // arc_cbu_awprot_0_cfg
	{ 0x814 , 0x11                , 1 }, // arc_cbu_awprot_1_cfg
	{ 0x848 , 0x3                 , 1 }, // cfg_dccm_terminate_bresp
	{ 0x84c , 0x3                 , 1 }, // cfg_dccm_terminate_rresp
	{ 0x850 , 0x1                 , 1 }, // cfg_dccm_terminate_en
	{ 0x860 , 0x1                 , 1 }, // cfg_dbg_apb_terminate_en
	{ 0x864 , 0x1                 , 1 }, // cfg_dbg_apb_terminate_resp
	{ 0x904 , 0x1                 , 1 }, // arc_axi_ordering_ctl
	{ 0x908 , 0x1ffc0000          , 1 }, // arc_axi_ordering_addr_msk
	{ 0x90c , 0x1c500000          , 1 }, // arc_axi_ordering_addr
	{ 0x910 , 0x1                 , 1 }, // arc_acc_engs_buser
	{ 0x91c , 0x1                 , 1 }, // acc_engs_fork_virtual_win_en
	{ 0x920 , 0x1ffff000          , 1 }, // acc_engs_fork_virtual_addr_mask
	{ 0x924 , 0x1c500000          , 1 }, // acc_engs_fork_virtual_addr_min
	{ 0x92c , 0x1c50ffff          , 1 }, // acc_engs_fork_virtual_addr_max
	{ 0x930 , 0x3                 , 1 }, // acc_engs_fork_virtual_term_bresp
	{ 0x934 , 0x3                 , 1 }, // acc_engs_fork_virtual_term_rresp
	{ 0x940 , 0x3                 , 1 }, // acc_engs_arc_fork_win_en
	{ 0x948 , 0x5fff              , 1 }, // acc_engs_arc_fork_max_addr0
	{ 0x94c , 0xf000              , 1 }, // acc_engs_arc_fork_mask_addr0
	{ 0x950 , 0xd000              , 1 }, // acc_engs_arc_fork_min_addr1
	{ 0x958 , 0xffff              , 1 }, // acc_engs_arc_fork_max_addr1
	{ 0x95c , 0xf000              , 1 }, // acc_engs_arc_fork_mask_addr1
	{ 0x960 , 0x3                 , 1 }, // acc_engs_arc_fork_term_bresp
	{ 0x964 , 0x3                 , 1 }, // acc_engs_arc_fork_term_rresp
	{ 0xe80 , 0xffffffff          , 32 }, // glbl_priv
	{ 0xf24 , 0xffff              , 1 }, // mem_ecc_err_addr
	{ 0xf44 , 0xffffffff          , 1 }, // glbl_err_addr
	{ 0xf80 , 0xffffffff          , 32 }, // glbl_sec
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_QMAN_ARC_AUX_H_ */
