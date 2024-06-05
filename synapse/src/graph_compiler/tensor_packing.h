#pragma once
#include "habana_nodes.h"

bool canPack(const pTensor t, unsigned dim, unsigned factor);
void packBias(ConvolutionNode* n, std::shared_ptr<Tensor> t, const std::array<TSize, MME_MAX_CONV_DIMS>& packing);
void packWeights(ConvolutionNode* n, std::shared_ptr<Tensor> t, const std::array<TSize, MME_MAX_CONV_DIMS>& packing);
void packOFM(ConvolutionNode* n, std::shared_ptr<Tensor> t, const std::array<TSize, MME_MAX_CONV_DIMS>& packing);