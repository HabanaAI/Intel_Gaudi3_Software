#pragma once
#include <stdint.h>
#include <string.h>
#include "gaudi/mme.h"

typedef enum
{
    MME_REG_WRITE,
    MME_ARM_MON,
    MME_FENCE
} EMmeCommandType;

class MmeRegWriteCmd
{
public:
    static const unsigned MAX_MON_ARM_IN_FLIGHT = 4;

    MmeRegWriteCmd() : cmd_type(MME_REG_WRITE) {};
    MmeRegWriteCmd(const MmeRegWriteCmd &other)
    {
        reg_offset      = other.reg_offset;
        num_regs        = other.num_regs;
        cmd_type        = other.cmd_type;
        so_target_value = other.so_target_value;
        memcpy(reg_values, other.reg_values, num_regs * sizeof(uint32_t));
    }

    unsigned        reg_offset = 0;
    unsigned        num_regs = 0;
    unsigned        reg_values[sizeof(Mme::RegBlock)/sizeof(uint32_t)];
    unsigned        so_target_value = 0;
    EMmeCommandType cmd_type;
};

