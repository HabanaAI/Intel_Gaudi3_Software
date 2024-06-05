#include <cstddef>
#include <cassert>

#include "sync_mgr.hpp"
#include "scal_macros.h"

#include "gaudi2/asic_reg_structs/sob_objs_regs.h"
#include "gaudi3/asic_reg_structs/sob_objs_regs.h"

#include "gaudi2/asic_reg/gaudi2_blocks.h"
#include "gaudi3/asic_reg/gaudi3_blocks.h"


/****************************************************************************************/
/****************************************************************************************/
/*                                 G2                                                   */
/****************************************************************************************/
/****************************************************************************************/
uint64_t SyncMgrG2::getSmMappingSize()
{
    return varoffsetof(gaudi2::block_sob_objs, sm_sec);
}

uint64_t SyncMgrG2::getSmBase(unsigned dcoreID)
{
    uint64_t smBase = 0;
    switch (dcoreID)
    {
        case 0:
            smBase = mmDCORE0_SYNC_MNGR_OBJS_BASE;
            break;
        case 1:
            smBase = mmDCORE1_SYNC_MNGR_OBJS_BASE;
            break;
        case 2:
            smBase = mmDCORE2_SYNC_MNGR_OBJS_BASE;
            break;
        case 3:
            smBase = mmDCORE3_SYNC_MNGR_OBJS_BASE;
            break;
        default:
            assert(0);
    }
    return smBase;
}

/****************************************************************************************/
/****************************************************************************************/
/*                                 G3                                                   */
/****************************************************************************************/
/****************************************************************************************/
uint64_t SyncMgrG3::getSmMappingSize()
{
    return varoffsetof(gaudi3::block_sob_objs, cq_direct);
}

uint64_t SyncMgrG3::getSmBase(unsigned smIdx)
{
    uint64_t smBase;
    switch(smIdx / 2)
    {
        case 0:
            smBase = mmHD0_SYNC_MNGR_OBJS_BASE;
            break;
        case 1:
            smBase = mmHD1_SYNC_MNGR_OBJS_BASE;
            break;
        case 2:
            smBase = mmHD2_SYNC_MNGR_OBJS_BASE;
            break;
        case 3:
            smBase = mmHD3_SYNC_MNGR_OBJS_BASE;
            break;
        case 4:
            smBase = mmHD4_SYNC_MNGR_OBJS_BASE;
            break;
        case 5:
            smBase = mmHD5_SYNC_MNGR_OBJS_BASE;
            break;
        case 6:
            smBase = mmHD6_SYNC_MNGR_OBJS_BASE;
            break;
        case 7:
            smBase = mmHD7_SYNC_MNGR_OBJS_BASE;
            break;
        default:
            assert(0);
            return 0;
    }

    if (smIdx & 0x1)
    {
        smBase += offsetof(gaudi3::block_sob_objs, sob_obj_1);
    }

    return smBase;
}
