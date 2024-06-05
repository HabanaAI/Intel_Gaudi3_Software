#pragma once

#include "perf_lib_layer_params.h"
#include "habana_graph.h"

class ConvToFilterConverter
{
public:

    ConvToFilterConverter() = default;

    virtual ~ConvToFilterConverter() = default;

    static bool replaceConvNodeWithFilter2D(NodePtr foundNode, HabanaGraph& g);

private:
    static ns_SpatialReduction::Params
    getFilter2DParamsFromConvParams(const synConvolution3DParamsV2& convolutionParams);
};
