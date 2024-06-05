#pragma once

#include <string>
#include <map>
#include "chromium/small_map.h"

typedef enum
{
    ID_0, // main fence for GC program
    ID_1, // fence for Wait node
    ID_2, // fence used by driver
    ID_3  // fence used by goya2 HW bug WA mechanism
} WaitID;

typedef enum
{
    MONITOR_SO_OP_GREQ,
    MONITOR_SO_OP_EQ
} MonitorOp;


enum Barrier
{
    ENGINE_BARRIER    = 0x10, // 1 << 5
    REGISTER_BARRIER  = 0x20, // 1 << 6
    MESSAGE_BARRIER   = 0x40, // 1 << 7

    ALL_BARRIERS      = ENGINE_BARRIER | REGISTER_BARRIER | MESSAGE_BARRIER
};

// We utilize a small map of 5 elements to eliminate memory allocations.
// currently we have 5 logical queues for Gaudi2 and 4 for Gaudi3.
// Gaudi and Greco have more but are less of a concern since those are
// not supported by Eager.
// key = engine logical ID, value = signal value
using DependencyMap = chromium_small_map::small_map<std::map<unsigned, unsigned>, 5>;