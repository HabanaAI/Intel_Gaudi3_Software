#ifndef SYNAPSE_TYPES_H
#define SYNAPSE_TYPES_H

#include <stdint.h>
//TEMPORARY
#include "synapse_api_types.h"

struct EnqueueTensorInfo
{
    const char*     tensorName;
    char*           pTensorData;
    uint32_t        tensorSize;
};

#endif /*SYNAPSE_TYPES_H*/
