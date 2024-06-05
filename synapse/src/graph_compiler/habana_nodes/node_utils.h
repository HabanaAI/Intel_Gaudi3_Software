#pragma once

#include "synapse_types.h"
#include "types.h"

constexpr unsigned CONVERT_INV_SCALE_IDX  = 1;

/**
 * Get convolution params for MME nodes
 */
synConvolution3DParamsV2 getConvolutionParams(const Node& node);
float getConvertNodeScale(const NodePtr& n);

bool isConvertFp8Node(NodePtr node);
bool isReshapeNode(const NodePtr& node);
bool isLogicalReshape(const NodePtr& node);
bool isConvertToFp8Node(NodePtr node);
bool isConvertFromFp8Node(NodePtr node);
bool isFp8GemmGuid(NodePtr node);
bool isFp8ConvGuid(NodePtr node);
bool isFp8MmeCguid(NodePtr node);
bool hasScalarInput(const NodePtr& node);
float getScaleFromTensor(TensorPtr tensor);
