#include <cstddef>
#include "logger.h"
//
#include "sob.hpp"
#include "scal_macros.h"

#include "gaudi2/asic_reg_structs/sob_objs_regs.h"
#include "gaudi3/asic_reg_structs/sob_objs_regs.h"

#include "sync_mgr.hpp"
//
//
/****************************************************************************************/
/****************************************************************************************/
/*                                 G2                                                   */
/****************************************************************************************/
/****************************************************************************************/
uint64_t SobG2::getAddr(uint64_t smBase, uint32_t sobIdx)
{
    return smBase + varoffsetof(gaudi2::block_sob_objs, sob_obj[sobIdx]);
}

uint32_t SobG2::buildVal(int16_t val, SobLongSobEn longSobEn, SobOp op)
{
    if ((op == SobOp::set) && (val < 0))
    {
        LOG_ERR(SCAL, "Negative number {} with set request is illegal", val);
        assert(0);
    }

    gaudi2::sob_objs::reg_sob_obj sob {};

    sob.val         = val & 0x7FFF;
    sob._reserved24 = (val < 0);
    sob.long_sob    = (longSobEn == SobLongSobEn::on);
//    _reserved30 : 5,
    sob.trace_evict = 0;
    sob.inc         = (op == SobOp::inc);

    return sob._raw;
}

uint32_t SobG2::buildEmpty()
{
    return buildVal(0, SobLongSobEn::off, SobOp::set);
}

uint32_t SobG2::setVal(uint32_t givenSob, uint16_t val)
{
    gaudi2::sob_objs::reg_sob_obj sob {};

    sob._raw = givenSob;
    sob.val  = val;

    return sob._raw;
}

uint32_t SobG2::setLongEn(uint32_t givenSob, SobLongSobEn longSobEn)
{
    gaudi2::sob_objs::reg_sob_obj sob {};

    sob._raw      = givenSob;
    sob.long_sob  = (longSobEn == SobLongSobEn::on);

    return sob._raw;
}

/****************************************************************************************/
/****************************************************************************************/
/*                                 G3                                                   */
/****************************************************************************************/
/****************************************************************************************/
uint64_t SobG3::getAddr(uint64_t smBase, uint32_t sobIdx)
{
    constexpr int sosPerSyncMgr = sizeof(gaudi3::block_sob_objs::sob_obj_0) / sizeof(gaudi3::block_sob_objs::sob_obj_0[0]);

    if (sobIdx < sosPerSyncMgr)
        return smBase + varoffsetof(gaudi3::block_sob_objs, sob_obj_0[sobIdx]);
    else
        return smBase + varoffsetof(gaudi3::block_sob_objs, sob_obj_1[sobIdx - sosPerSyncMgr]);
}

uint32_t SobG3::buildVal(int16_t val, SobLongSobEn longSobEn, SobOp op)
{
    if ((op == SobOp::set) && (val < 0))
    {
        LOG_ERR(SCAL, "Negative number {} with set request is illegal", val);
        assert(0);
    }

    gaudi3::sob_objs::reg_sob_obj_0 sob {};

    sob.val         = val & 0x7FFF;
    sob._reserved24 = (val < 0); // 9 bits
    sob.long_sob    = (longSobEn == SobLongSobEn::on);
//    _reserved28 : 3
//  sob.zero_sob_cnt = 0;
//   _reserved30 : 1
    sob.trace_evict = 0;
    sob.inc         = (op == SobOp::inc);

    return sob._raw;
}

uint32_t SobG3::buildEmpty()
{
    return buildVal(0, SobLongSobEn::off, SobOp::set);
}

uint32_t SobG3::setVal(uint32_t givenSob, uint16_t val)
{
    gaudi3::sob_objs::reg_sob_obj_0 sob {};

    sob._raw = givenSob;
    sob.val  = val;

    return sob._raw;
}

uint32_t SobG3::setLongEn(uint32_t givenSob, SobLongSobEn longSobEn)
{
    gaudi3::sob_objs::reg_sob_obj_0 sob {};

    sob._raw      = givenSob;
    sob.long_sob  = (longSobEn == SobLongSobEn::on);

    return sob._raw;
}
