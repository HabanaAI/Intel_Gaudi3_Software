#pragma once

#include <cstdint>

// |Field Name |Bits on Write |Data Comments
// |----------------------------------------
// |Value      |15:0          |If Op=0 & Long=0, the SOB is written with bits [14:0]
// |           |              |If Op=0 & Long=1, 4 consecutive SOBs are written with ZeroExtendTo60bitsof bits [15:0]
// |           |              |If Op=1 & Long=0, an atomic add is performed such that SOB[14:0]+= Signed[15:0]. As the
// |           |              |incoming data is S16, one can perform a subtraction of the monitor.
// |           |              |If Op=1 & Long=1, The 60 bits SOB which aggregates 4x15bits physical SOB is atomically added
// |           |              |with Signed[15:0]
// |Long       |24            |See value field description
// |Trace Event|30            |When set, a trace event is triggered
// |Op         |31            |See value field description

enum class SobLongSobEn : bool {on, off};
enum class SobOp        : bool {set, inc};

class SobG2
{
public:
    static uint64_t getAddr(uint64_t smBase, uint32_t sobIdx);

    static uint32_t buildVal(int16_t val, SobLongSobEn longSobEn, SobOp op);
    static uint32_t buildEmpty();
    static uint32_t setVal(uint32_t givenSob, uint16_t val);
    static uint32_t setLongEn(uint32_t givenSob, SobLongSobEn longSobEn);
};

class SobG3
{
public:
    static uint64_t getAddr(uint64_t smBase, uint32_t sobIdx);

    static uint32_t buildVal(int16_t val, SobLongSobEn longSobEn, SobOp op);
    static uint32_t buildEmpty();
    static uint32_t setVal(uint32_t givenSob, uint16_t val);
    static uint32_t setLongEn(uint32_t givenSob, SobLongSobEn longSobEn);
};
