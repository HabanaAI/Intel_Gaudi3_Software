#pragma once

#include "habana_graph.h"
#include "include/mme_common/mme_brain.h"
#include "include/mme_common/mme_common_enum.h"
#include "synapse_common_types.h"
#include "types.h"

using DcoreTile = gc::access_pattern::Tile<uint64_t, int64_t>;

void getTensorRolesCommon(const MmeNode&        node,
                          MmeCommon::EMmeOpType opType,
                          TensorPtr&            xTensor,
                          TensorPtr&            wTensor,
                          TensorPtr&            yTensor,
                          TensorPtr&            oTensor);

void setTensorViewByOp(const MmeNode&                  node,
                       MmeCommon::MmeLayerParams&      params,
                       const MmeCommon::MmeTensorView& aView,
                       const MmeCommon::MmeTensorView& bView,
                       const MmeCommon::MmeTensorView& outView);

void flattenContiguousBatchDims(const MmeNode& mmeNode, MmeCommon::MmeLayerParams& params);
void unifyBroadCastRepresentation(MmeCommon::MmeLayerParams& params);
void normalizeTensorDims(const MmeNode& mmeNode, MmeCommon::ChipType chipType, MmeCommon::MmeLayerParams& params);

void setTracing(MmeCommon::MmeLayerParams& params);

MmeCommon::EMmeOpType getOperationTypeCommon(MmeCommon::ChipType chipType, const MmeNode& node);

MmeCommon::MmeTensorView getTensorViewCommon(MmeCommon::ChipType chipType,
                                             const Tensor& tensor,
                                             const HalReader& halReader,
                                             bool isDmaOp);

MmeCommon::MmeTensorView getTensorViewFromTile(MmeCommon::ChipType chipType,
                                               const TensorPtr& tensor,
                                               const DcoreTile& tensorTile,
                                               const HalReader& halReader,
                                               bool isDmaOp);

// Sets only data type and sizes, no memory properties like strides.
MmeCommon::MmeTensorView
getSemanticTensorView(MmeCommon::ChipType chipType, const Tensor& tensor, const HalReader& halReader, bool isDmaOp);

MmeCommon::EMmeDataType getMmeElementTypeCommon(synDataType elementType, bool fp32ForcedToIEEE);

bool getAlignedAddresses(const MmeNode* mmeNode, MmeCommon::EMmeOpType opType, bool ignoreTensorAliasing);

bool isTensorAddressCacheLineAligned(TensorPtr tensor, bool ignoreTensorAliasing = true);

bool isAllocPolicySuitableForTensor(CacheDirective allocPolicy, const TensorPtr& tensor);
