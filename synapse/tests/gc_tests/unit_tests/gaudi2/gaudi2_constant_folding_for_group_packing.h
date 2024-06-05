#pragma once

#include "graph_compiler/types.h"
#include "graph_optimizer_test.h"
#include "define_synapse_common.hpp"
#include "synapse_common_types.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"

using SizeArray4dims = std::array<TSize, 4>;  // To build valid tensors for grouped convolution packing
using Vec4Dims       = std::vector<std::vector<std::vector<std::vector<float>>>>;

struct Gaudi2ConstantFoldingGconvFwdInGroupPackingParams
{
    unsigned       nGroups;
    unsigned       kPerGroup;
    SizeArray4dims wSizes;
    SizeArray4dims xSizes;
    bool           isDynamic;
};

class Gaudi2ConstantFoldingGconvFwdInGroupPacking
: public GraphOptimizerTest
, public testing::WithParamInterface<Gaudi2ConstantFoldingGconvFwdInGroupPackingParams>
{
protected:
    // create an arbitrary tensor
    TSize          getTotalSize(const SizeArray4dims& sizes);
    SizeArray4dims getSizeArray4dims(const SizeArray& sizes);
    TensorPtr      createTensor(const SizeArray4dims& maxSizes,
                                const SizeArray4dims& minSizes,
                                synDataType           dataType,
                                bool                  isPersistent = true,
                                char*                 data         = 0);
    Vec4Dims       createPackedWeightsExpectedData(const TensorPtr& weightsTensor,
                                                   const unsigned   firstKernelToPack,
                                                   const unsigned   amountOfKernelsToPack,
                                                   const unsigned   groupsPerNewGroup,
                                                   const unsigned   kPerGroup);
    void           validateWeightsOfConvNodesReturnedByPacker(const TensorPtr&  origWeights,
                                                              const NodeVector& convNodes,
                                                              const unsigned    mmeVectorSize,
                                                              const unsigned    nGroups,
                                                              const unsigned    kPerGroup);
    void           createAndSetConstSectionToTensor(const TensorPtr& t, const Gaudi2Graph& g);

private:
    unsigned m_groupsPerVector;
    unsigned m_groupsQuotient;
    unsigned m_groupsReminder;
    unsigned m_memorySectionId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1;
};
