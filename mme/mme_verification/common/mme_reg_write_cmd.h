#pragma once
#include <cstdint>
#include <vector>

enum EMmeCommandType
{
    MME_REG_WRITE,
    MME_FENCE,
    MME_WAIT
};

class MmeQmanCmd
{
public:
    MmeQmanCmd() : cmd_type(MME_REG_WRITE) {};
    MmeQmanCmd(const MmeQmanCmd& other)
    {
        cmd_type = other.cmd_type;
        reg_offset = other.reg_offset;
        fence_idx = other.fence_idx;
        fence_value = other.fence_value;
        reg_values = other.reg_values;
        wait_idx = other.wait_idx;
        wait_value = other.wait_value;
        wait_cycles = other.wait_cycles;
        power_last_setup_cmd = other.power_last_setup_cmd;
    }

    // for MME_REG_WRITE
    unsigned reg_offset = 0;
    std::vector<uint32_t> reg_values;

    // for MME_FENCE
    unsigned fence_idx = 0;
    unsigned fence_value = 0;

    // for MME_WAIT
    unsigned wait_idx = 0;
    unsigned wait_value = 0;
    unsigned wait_cycles = 0;

    EMmeCommandType cmd_type;
    bool power_last_setup_cmd = false;
};
