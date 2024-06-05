#pragma once
#include "synapse_common_types.h"
struct hlthunk_wait_multi_cs_in;
struct hlthunk_wait_multi_cs_out;

class WcmCsQuerierInterface
{
public:
    virtual ~WcmCsQuerierInterface() = default;

    // Todo modify the interface in a way that will hide the LKD's structures.
    virtual synStatus query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams) = 0;

    virtual void dump() const = 0;

    virtual void report() = 0;
};