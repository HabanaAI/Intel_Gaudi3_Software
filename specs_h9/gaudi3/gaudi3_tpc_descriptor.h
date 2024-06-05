#ifndef _GAUDI3_TPC_DESC_H_
#define _GAUDI3_TPC_DESC_H_

#include <stdint.h>

#include "asic_reg/gaudi3_blocks.h"
#include "asic_reg_structs/tpc_regs.h"
#include "asic_reg_structs/tpc_tensor_base_regs.h"
#include "asic_reg_structs/tpc_tensor_shared_regs.h"

#pragma pack(push, 4)

struct TensorDescGaudi3
{
    gaudi3::block_tpc_tensor_base   base;
    gaudi3::block_tpc_tensor_shared shared;
};

struct Gaudi3TpcDesc
{
    static const int c_max_tensor_dims = 5;
    static const int c_max_tensors_nr = 16;
    static const int c_max_threads    = 4;

    TensorDescGaudi3                            m_tensors[c_max_tensors_nr];
    gaudi3::block_sync_object                   m_so_th0;
    gaudi3::block_tpc_non_tensor_descriptor_smt m_desc_smt_th0;
    gaudi3::block_tpc_non_tensor_descriptor     m_desc;
    gaudi3::block_sync_object                   m_so_th1;
    gaudi3::block_tpc_non_tensor_descriptor_smt m_desc_smt_th1;
    gaudi3::block_sync_object                   m_so_th2;
    gaudi3::block_tpc_non_tensor_descriptor_smt m_desc_smt_th2;
    gaudi3::block_sync_object                   m_so_th3;
    gaudi3::block_tpc_non_tensor_descriptor_smt m_desc_smt_th3;
};

#pragma pack(pop)
#endif // _GAUDI3_TPC_DESC_H_
