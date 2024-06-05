#ifndef MME__GAUDI_AGU_STAT_H
#define MME__GAUDI_AGU_STAT_H

#include "gaudi/mme.h"
#include "include/sync/overlap.h"

namespace gaudi
{
typedef SegmentsSpace<unsigned> AguRanges;

unsigned aguStatsCountSBReads(
    bool isShared,
    const Mme::Desc *northDesc,
    const Mme::Desc *southDesc);

void aguStatsGetRanges(const bool isInput,
                       const bool isShared,
                       const Mme::Desc* desc,
                       const unsigned eventBase,
                       std::vector<AguRanges>* ranges);
}  // namespace gaudi

#endif //MME__GAUDI_AGU_STAT_H
