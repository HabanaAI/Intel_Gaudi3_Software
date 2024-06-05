#pragma once

#include "synapse_common_types.hpp"
#include "synapse_types.h"
#include "node.h"
#include "types.h"

namespace StridedOpUtils
{
// return a string describing synStridedOpParams
std::string stridedOpParamsString(const synStridedOpParams& params, unsigned dim);

bool verifyStridedAccess(const TensorPtr& real, const TensorPtr& alias, const synStridedOpParams& params);

synStridedOpParams createExpandedParams(const synStridedOpParams& params, unsigned dim);

// create new params that fit the 64bit -> 32bit reinterpret transformation
synStridedOpParams createReinterpretedParams(const synStridedOpParams& params, unsigned originalDim);

synStridedOpParams createParamsFromShapeTensors(const TensorPtr& strides, const TensorPtr& offset);

// check if t1 and t2 overlap, when using their synStridedOpParams.
bool isOverlap(const TensorPtr& t1, const TensorPtr& t2, const synStridedOpParams& p1, const synStridedOpParams& p2);

// check if p1 and p2 are equal up to dim
bool compareParams(const synStridedOpParams& p1, const synStridedOpParams& p2, unsigned dim);

bool isDenseStridedOpParams(const synStridedOpParams& params, const TensorPtr& view);

void convertShapeToH2D(NodeList& nodes, TensorVector& inputs, TensorVector& outputs, synStridedOpParams& params, const std::string& name);

std::tuple<TensorPtr, NodePtr> expandH2DTensor(const TensorPtr& tensor, unsigned dim);

// create new H2D params tensor that fits the 64bit -> 32bit reinterpret transformation
std::tuple<TensorPtr, NodePtr> reinterpretH2DTensor(const TensorPtr& tensor, unsigned factor);

};  // namespace StridedOpUtils