/* SPDX-License-Identifier: MIT
 *
 * Copyright 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 */

#ifndef GAUDI3_PQM_PACKETS_H
#define GAUDI3_PQM_PACKETS_H

#include <linux/types.h>

#define PQM_PACKET_HEADER_PACKET_ID_SHIFT		56
#define PQM_PACKET_HEADER_PACKET_ID_MASK		0x1F00000000000000ull

enum pqm_packet_id {
	PQM_PACKET_WREG_32 = 0x1,
	PQM_PACKET_WREG_BULK = 0x2,
	PQM_PACKET_MSG_LONG = 0x3,
	PQM_PACKET_MSG_SHORT = 0x4,
	PQM_PACKET_FENCE = 0x8,
	PQM_PACKET_LIN_PDMA = 0x9,
	PQM_PACKET_NOP = 0xA,
	PQM_PACKET_STOP = 0xB,
	PQM_PACKET_WAIT = 0xD,
	PQM_PACKET_WREG_64_SHORT = 0x12,
	PQM_PACKET_WREG_64_LONG = 0x13,
	PQM_PACKET_CH_WREG_32 = 0x14,
	PQM_PACKET_CH_WREG_BULK = 0x15,
	PQM_PACKET_CH_WREG_64_SHORT = 0x16,
	PQM_PACKET_CH_WREG_64_LONG = 0x17,
	MAX_PQM_PACKET_ID = (PQM_PACKET_HEADER_PACKET_ID_MASK >>
				PQM_PACKET_HEADER_PACKET_ID_SHIFT) + 1
};

struct pqm_packet_wreg32 {
	__u32 value;
	union {
		struct {
			__u32 pred :5;
			__u32:1;
			__u32 reg_offset :18;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32:2;
		};
		__u32 ctl;
	};
};

struct pqm_packet_wreg_bulk {
	__u32 size64 :16;
	__u32:16;
	union {
		struct {
			__u32 pred :5;
			__u32:1;
			__u32 reg_offset :18;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32:2;
		};
		__u32 ctl;
	};
	__u64 values[0];
};

struct pqm_packet_nop {
	__u32 msg_barrier :1;
	__u32:31;
	union {
		struct {
			__u32:24;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32:2;
		};
		__u32 ctl;
	};
};

struct pqm_packet_stop {
	__u32 reserved;
	union {
		struct {
			__u32:24;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32:2;
		};
		__u32 ctl;
	};
};

struct pqm_packet_msg_long {
	__u32 value;
	union {
		struct {
			__u32 pred :5;
			__u32:15;
			__u32 op :2; /* 0: write <value>. 1: write timestamp. */
			__u32:2;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32:2;
		};
		__u32 ctl;
	};
	__u64 addr;
};

struct pqm_packet_msg_short {
	union {
		struct {
			__u32 sync_group_id :8;
			__u32 mask :8;
			__u32 mode :1;
			__u32 sync_value :15;
		} mon_arm_register;
		struct {
			__u32 long_sob :1;
			__u32:3;
			__u32 cq_en :1;
			__u32:3;
			__u32 lbw_en :1;
			__u32:6;
			__u32 sm_data_config :1;
			__u32 msb_sid :4;
			__u32 auto_zero :1;
			__u32:3;
			__u32 wr_num :4;
			__u32:3;
			__u32 long_high_group :1;
		} mon_config_register;
		struct {
			__u32 sync_value :15;
			__u32:9;
			__u32 long_mode :1;
			__u32:3;
			__u32 zero_sob_cntr :1;
			__u32:1;
			__u32 te :1;
			__u32 mode :1;
		} so_upd;
		__u32 value;
	};
	union {
		struct {
			__u32 msg_addr_offset :16;
			__u32 base :3;
			__u32 dw :1;
			__u32 op :2;
			__u32:2;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32:2;
		};
		__u32 ctl;
	};
};

struct pqm_packet_fence {
	union {
		struct {
			__u32 dec_val :4;
			__u32 ch_id :5;
			__u32:7;
			__u32 target_val :14;
			__u32 id :2;
		};
		__u32 cfg;
	};
	union {
		struct {
			__u32 pred :5;
			__u32:19;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32:2;
		};
		__u32 ctl;
	};
};

struct pqm_packet_lin_pdma {
	__u32 tsize :32;
	union {
		struct {
			__u32 wrcomp :2;
			__u32 endian_swap :2;
			__u32 memset :1;
			__u32 fence_en :1;
			__u32 use_wr_comp_addr_hi_from_reg :1;
			__u32 consecutive_comp1_2_addr_l :1;
			__u32 inc_context_id :1;
			__u32 add_offset_0 :1;
			__u32 src_dest_msb_from_reg :1;
			__u32 en_desc_commit :1;
			__u32 advcnt_hbw_complq :1;
			__u32 direction :1;
			__u32:10;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32:2;
		};
		__u32 ctl;
	};
	__u32 src_addr_lo :32;
	__u32 dst_addr_lo :32;
	__u64 values[0]; /* data starts here */
};

struct pqm_packet_wait {
	__u32 num_cycles_to_wait :24;
	__u32 inc_val :4;
	__u32 id :2;
	__u32:2;
	union {
		struct {
			__u32:24;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32:2;
		};
		__u32 ctl;
	};
};

struct pqm_packet_wreg64_short {
	__u32 offset;
	union {
		struct {
			__u32 pred :5;
			__u32 base :5;
			__u32 dreg_offset_lsb :14;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32 dreg_offset_msb :2;
		};
		__u32 ctl;
	};
};

struct pqm_packet_wreg64_long {
	__u32 dw_enable :2;
	__u32 rel :1;
	__u32:29;
	union {
		struct {
			__u32 pred :5;
			__u32 base :5;
			__u32 dreg_offset_lsb :14;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32 dreg_offset_msb :2;
		};
		__u32 ctl;
	};
	__u64 offset;
};

struct pqm_packet_ch_wreg32 {
	__u32 value;
	union {
		struct {
			__u32 pred :5;
			__u32:1;
			__u32 reg_offset :12;
			__u32 a_or_b :1;
			__u32:5;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32:2;
		};
		__u32 ctl;
	};
};

struct pqm_packet_ch_wreg_bulk {
	__u32 size64 :16;
	__u32:16;
	union {
		struct {
			__u32 pred :5;
			__u32:1;
			__u32 reg_offset :12;
			__u32 a_or_b :1;
			__u32:5;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32:2;
		};
		__u32 ctl;
	};
	__u64 values[0];
};

struct pqm_packet_ch_wreg64_short {
	__u32 offset;
	union {
		struct {
			__u32 pred :5;
			__u32 base :5;
			__u32 dreg_offset_lsb :10;
			__u32 a_or_b :1;
			__u32:3;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32 dreg_offset_msb :2;
		};
		__u32 ctl;
	};
};

struct pqm_packet_ch_wreg64_long {
	__u32 dw_enable :2;
	__u32 rel :1;
	__u32:29;
	union {
		struct {
			__u32 pred :5;
			__u32 base :5;
			__u32 dreg_offset_lsb :10;
			__u32 a_or_b: 1;
			__u32:3;
			__u32 opcode :5;
			__u32 eng_barrier :1;
			__u32 dreg_offset_msb :2;
		};
		__u32 ctl;
	};
	__u64 offset;
};

#endif /* GAUDI3_PQM_PACKETS_H */
