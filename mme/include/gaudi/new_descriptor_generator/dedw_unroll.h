#ifndef MME__GAUDI_DEDW_UNROLL_H
#define MME__GAUDI_DEDW_UNROLL_H

#include "include/gaudi/new_descriptor_generator/mme_common.h"

namespace gaudi
{
class GeoAttr;

// Calc DEDW unroll optimization by finding out unroll factor and dimension
class DedwUnroll
{
public:
    DedwUnroll(const MmeCommon::MmeLayerParams& params, const GeoAttr& geoAttr);
    unsigned getUnrollFactor() const { return m_unrollFactor; }
    unsigned getUnrollDim() const { return m_unrollDim; }
    bool shouldUnroll() { return (m_unrollFactor > 1); }

private:
    unsigned m_unrollFactor = 1;
    unsigned m_unrollDim = Mme::c_mme_max_tensor_dims;
};

}  // namespace gaudi

#endif //MME__GAUDI_DEDW_UNROLL_H
