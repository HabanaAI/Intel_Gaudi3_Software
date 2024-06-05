#pragma once

#include <vector>

#include "gaudi2/mme.h"
#include "include/sync/segments_space.h"
#include "include/gaudi2/mme_descriptor_generator.h"

namespace Gaudi2
{
typedef SegmentsSpace<unsigned> AguRanges;

typedef enum
{
    e_mme_agu0_idx = 0,
    e_mme_agu1_idx = 1,
    e_mme_agu2_idx = 2,
    e_mme_agu3_idx = 3,
    e_mme_agu4_idx = 4,
    e_mme_agu_cout0_idx = 5,
    e_mme_agu_cout1_idx = 6,

} EMmeOperandIdx;

void genAddresses(const Mme::Desc* desc,
                  const EMmeOperandIdx aguId,
                  const MmeCommon::EMmeOpType opType,
                  const bool master,
                  std::vector<AguRanges>* ranges);

}  // namespace Gaudi2
