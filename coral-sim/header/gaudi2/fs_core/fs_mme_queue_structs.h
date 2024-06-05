#pragma once
#include <stdint.h>

#include "gaudi2/mme.h"

#include "fs_mme_desc.h"
#include "fs_mme_eu_fp.h"
#include "fs_mme_md.h"

namespace Gaudi2
{
namespace Mme
{

typedef struct
{
    uint64_t                  addr;
    uint64_t                  addr1;
    Gaudi2::Mme::MetaData::SB md;
    uint64_t                  cmd_id;

} QS_Agu2Sb;

typedef struct
{
    uint8_t                       data[Gaudi2::Mme::c_cl_size];
    Gaudi2::Mme::MetaData::SB::TE md;
    uint64_t                      addrId;
    uint64_t                      addr;
} QS_Sb2Te, QS_Sbte2Eus, QS_Te2Dec, QS_Dec2Eus;

typedef struct
{
    Gaudi2::Mme::EUSDescriptor eusDesc;
} QS_EusBrain;

typedef struct
{
    Gaudi2::Mme::ACCAPDescriptor accapDesc;
} QS_AccApBrain;

typedef struct
{
    float    rollupFp[Gaudi2::Mme::EU_fp::c_matrix_height][Gaudi2::Mme::EU_fp::c_matrix_width];
    uint8_t  accIdx;
    bool     doubleAccums;
    uint64_t ucmd_id;
    uint64_t cmd_id;
} QS_Eus2Acc;

typedef struct
{
    uint8_t      data[Gaudi2::Mme::c_cl_size];
    uint64_t     ucmd_id;
    uint64_t     cmd_id;
    uint64_t     accum_id;
    EMmeDataType dataType;
} QS_Ap2Wbc;

} // namespace Mme
} // namespace Gaudi2
