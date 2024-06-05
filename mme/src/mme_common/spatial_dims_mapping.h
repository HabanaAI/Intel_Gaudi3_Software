#ifndef MME__SPATIAL_DIMS_MAPPING_H
#define MME__SPATIAL_DIMS_MAPPING_H

#include <cstddef>

namespace MmeCommon::AccessPatternDetails::Conv
{
using Dim = size_t;

// This class allows executing code with synchronized index space, filters and tensors spatial dims. The code is
// passed in a callable that recieves the spatial dim index (width, height, depth and optionally batch), with
// SpatialDimsIndices object that contains the respective dimensions
class SpatialDimsMapping
{
public:
    // local names and order of the spatial dimensions
    enum class SpatialDim : Dim
    {
        FIRST = 0,

        WIDTH = SpatialDim::FIRST,
        HEIGHT,
        DEPTH,

        // In some cases it's useful to treat 'batch' as one of the spatial dims.
        // When doing so, it's important not to use the mapped wDim and filterDim, as they are meaningless.
        BATCH,
    };

    struct SpatialDimIndices
    {
        Dim idxSpcDim;
        Dim filterDim;  // also an index space dim, that fit the wDim
        Dim xyDim;
        Dim wDim;
    };

    SpatialDimsMapping(unsigned spatialDimsNr);

    // Returns the mapped tensor dims and index space dims correlated with spatial dimension 'spDim'
    SpatialDimIndices getIndices(SpatialDim spDim) const;

    // Executes the passed callable for each spatial dimension from the inner-most out and the batch.
    // Note - filterDim and wDim should not be used when spDim == SpatialDim::BATCH
    template <typename Callable>
    void forEachWithBatch(Callable callable) const
    {
        forEachWithoutBatch(callable);
        callable(SpatialDim::BATCH, getIndices(SpatialDim::BATCH));
    }

    // Executes the passed callable for each spatial dimension but not the batch
    template <typename Callable>
    void forEachWithoutBatch(Callable callable) const
    {
        for (auto spDim = SpatialDim::FIRST; Dim(spDim) < m_spatialDimsNr; ((Dim&) spDim)++)
        {
            callable(spDim, getIndices(spDim));
        }
    }

private:
    const Dim m_spatialDimsNr;

};
}  // namespace MmeCommon::AccessPatternDetails::Conv

#endif //MME__SPATIAL_DIMS_MAPPING_H
