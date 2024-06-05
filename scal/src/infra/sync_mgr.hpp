#pragma once

#include <cstdint>

class SyncMgrG2
{
public:
    static uint64_t getSmMappingSize();
    static uint64_t getSmBase(unsigned dcoreID);
};

class SyncMgrG3
{
public:
    static uint64_t getSmMappingSize();
    static uint64_t getSmBase(unsigned smIdx);
};