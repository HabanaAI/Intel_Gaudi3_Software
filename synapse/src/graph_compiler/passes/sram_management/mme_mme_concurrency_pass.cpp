#include "backward_operand_chunk_setter.h"
#include "defs.h"
#include "sram_management.h"

#include <unordered_set>
#include "mme_concurrency_identifier.h"
#include "mme_concurrency_memset.h"

bool identifyMmeConcurrency(HabanaGraph& g)
{
    MmeConcurrencyIdentifier MmeConcurrencyIdentifier(g);
    return MmeConcurrencyIdentifier.scanGraph();
}

bool applyMmeConcurrencyMemset(HabanaGraph& g)
{
    MmeConcurrencyMemset MmeConcurrencyMemset(g);
    return MmeConcurrencyMemset.addMemsetNodes();
}

bool enableMmeNodeCommonDimConcurrency(HabanaGraph& g)
{
    bool ret = identifyMmeConcurrency(g);
    if (!ret) return ret;
    return applyMmeConcurrencyMemset(g);
}