#pragma once
#include <stdint.h>
#include <queue>
#include "../headers/mme_reg_write_cmd.h"

#define _IN_
#define _OUT_
#define _IO_

typedef struct
{
    unsigned base_offset;
    unsigned size;
} MmeSramRange;

typedef enum
{
    e_mme_gen_params_success = 0,
    e_mme_gen_params_out_of_sram = 1,
    e_mme_gen_params_not_applicable = 2,
} EGenTestParamsRet;

void getValidConvTests(std::queue<unsigned>* testsIds);

EGenTestParamsRet genTestParams(
    _IN_ unsigned test_id,
    _IN_ bool mult3x3,
    _IN_ unsigned sync_object_id,
    _IN_ uint64_t sram_base,
    _IN_ void * sram_buff, // must be 128 bytes aligned. Doron to allocate the srame space.
    _IN_ unsigned sram_buff_size,  // Doron to provide the allocated size, Amos return the actual size.
    _OUT_ MmeSramRange *inputRange,  // The sram range of input, bias, weights - everything that needs to be copied.
    _OUT_ MmeSramRange *outputRange, // The sram range of the output, Doron should DMA it out to the host and compare it with the range returned by Amos.
    _OUT_ std::queue<MmeRegWriteCmd>* reg_writes,
    _OUT_ unsigned* sync_object_value);
