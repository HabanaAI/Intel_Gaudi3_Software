#include "gaudi2_constant_folding_for_group_packing.h"
#include "defs.h"
#include "syn_singleton.hpp"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "types.h"
#include "node_factory.h"
#include "gtest/gtest.h"
#include "pack_grouped_convolutions.h"
#include <iostream>
#include "tensor_data_manipulation_util.h"
#include "scoped_configuration_change.h"

extern ConcurrentSlotMapAlloc<InternalSectionHandle> sectionHndlSlopMap;

TensorPtr Gaudi2ConstantFoldingGconvFwdInGroupPacking::createTensor(const SizeArray4dims& maxSizes,
                                                                    const SizeArray4dims& minSizes,
                                                                    synDataType           dataType,
                                                                    bool                  isPersistent /*= true*/,
                                                                    char*                 data)
{
    synMemoryDescriptor memDesc(isPersistent);

    auto tensor = std::make_shared<Tensor>(4U, maxSizes.data(), dataType, minSizes.data());
    if (data != 0 && maxSizes == minSizes)
    {
        tensor->setTensorBuffer(data, getTotalSize(maxSizes) * dataTypeSizeInBytes(dataType), dataType, false);
    }
    tensor->setMemoryDescriptor(memDesc);
    if (isPersistent)
    {
        tensor->setMemorySectionID(m_memorySectionId++);
    }
    tensor->map();
    return tensor;
}

TSize Gaudi2ConstantFoldingGconvFwdInGroupPacking::getTotalSize(const SizeArray4dims& sizes)
{
    TSize totalSize = 1;
    for (auto size : sizes)
    {
        totalSize *= size;
    }
    return totalSize;
}

SizeArray4dims Gaudi2ConstantFoldingGconvFwdInGroupPacking::getSizeArray4dims(const SizeArray& sizes)
{
    SizeArray4dims result;
    for (unsigned i = 0; i < result.size(); ++i)
    {
        result[i] = sizes[i];
    }
    return result;
}

void Gaudi2ConstantFoldingGconvFwdInGroupPacking::createAndSetConstSectionToTensor(const TensorPtr& t, const Gaudi2Graph& g)
{
    // reference for how to create and set a const section: recipe_generator_test.cpp
    synMemoryDescriptor persistentMemoryDesc(true);
    auto sectionHandle1 = sectionHndlSlopMap.insert(0, 0);
    sectionHandle1.second->setConst(true);
    uint32_t mysectionId1 = g.getCodeGenerator()->getNextMemorySectionID(SectionIDGenerator::USER_ALLOCATED_SECTIONS);
    sectionHandle1.second->setIDAndLock(mysectionId1);
    t->setMemoryDescriptor(persistentMemoryDesc);
    t->setSectionHandle(sectionHandle1.second);
    t->setMemorySectionID(mysectionId1);
    t->setAsStaticParam();
}

/**
 * @brief This function is used to create data for validation
 *
 * @param weightsTensor - the weights tensor to pack.
 * @param firstKernelToPack - first kernel index to pack.
 * @param amountOfKernelsToPack - amount of kernels to be packed (counted from the given first kernel).
 * @param groupsPerNewGroup - amount of groups to pack in a packed group.
 * @param kPerGroup - amount of kernels in one group.
 * @return Vec4Dims - data of the expected packed weights (after gconv_fwd)
 */
Vec4Dims
Gaudi2ConstantFoldingGconvFwdInGroupPacking::createPackedWeightsExpectedData(const TensorPtr& weightsTensor,
const unsigned firstKernelToPack, const unsigned amountOfKernelsToPack,
const unsigned groupsPerNewGroup,
const unsigned kPerGroup)
{
    const auto& weightsTensorSizes = weightsTensor->getAllSizesInElements();
    SizeArray4dims packedWeightsSizes = getSizeArray4dims(weightsTensorSizes);
    packedWeightsSizes[WEIGHT_DIM_C] *= groupsPerNewGroup;
    packedWeightsSizes[WEIGHT_DIM_K] = amountOfKernelsToPack;

    TSize rPacked = packedWeightsSizes[WEIGHT_DIM_R];
    TSize sPacked = packedWeightsSizes[WEIGHT_DIM_S];
    TSize cPacked = packedWeightsSizes[WEIGHT_DIM_C];
    TSize kPacked = packedWeightsSizes[WEIGHT_DIM_K];
    Vec4Dims v4Dims
    (rPacked, std::vector<std::vector<std::vector<float>>>(sPacked, std::vector<std::vector<float>>(cPacked, std::vector<float>(kPacked,0))));

    for (unsigned r = 0; r < weightsTensorSizes[WEIGHT_DIM_R]; ++r)
    {
        for (unsigned s = 0; s < weightsTensorSizes[WEIGHT_DIM_S]; ++s)
        {
            for (unsigned c = 0; c < weightsTensorSizes[WEIGHT_DIM_C]; ++c)
            {
                for (unsigned k = firstKernelToPack; k < firstKernelToPack+amountOfKernelsToPack; ++k)
                {
                    auto      tensorElem    = TensorElementAccess::getElement<float>(weightsTensor, {k,c,s,r});
                    // oldGroupIndexInPackedGroup = index of old group in the packed group,
                    // groupsPerNewGroup * kPerGroup = amount of kernels in the packed group
                    unsigned kOut = k - firstKernelToPack;
                    unsigned kernelsAmountInPackedGroup = groupsPerNewGroup * kPerGroup;
                    unsigned oldGroupIndexInPackedGroup = (kOut % kernelsAmountInPackedGroup) / kPerGroup;
                    unsigned cOut          = c + oldGroupIndexInPackedGroup * weightsTensor->getAllSizesInElements()[WEIGHT_DIM_C];
                    v4Dims[r][s][cOut][kOut] = tensorElem;
                }
            }
        }
    }
    return v4Dims;
}


void Gaudi2ConstantFoldingGconvFwdInGroupPacking::validateWeightsOfConvNodesReturnedByPacker(const TensorPtr& origWeights, const NodeVector& convNodes,const unsigned mmeVectorSize,const unsigned nGroups, const unsigned kPerGroup)
{
    // Create the expected solution meta data:
    const unsigned groupsPerVector = std::min(nGroups, std::max(1U, mmeVectorSize / kPerGroup));
    // Number of the new packed groups (without remainder packed groups, means that each of them packs equal number of groups)
    const unsigned groupsQuotient  = nGroups / groupsPerVector;
    // In case nGroup % groupsPerVector != 0, there is one more new packed group that packs the remainder groups
    const unsigned groupsRemainder = nGroups % groupsPerVector;
    // Extended explanation: in case groupsRemainder!=0 the packer split the original group convolution node to 2 group convolution nodes,
    // such that the first one's weights tensor contains: groupsQuotient * groupsPerVector * kPerGroup kernels,
    // and the second's weights tensor contains: groupsRemainder * kPerGroup kernels.

    // original weights tensor kernels total amount is equal to groupsQuotient*groupsPerVector*kPerGroup + groupsRemainder*kPerGroups.
    // also, groupsQuotient * groupsPerVector + groupsRemainder = nGroups (original groups amount).
    // number of new groups = groupsRemainder + groupsQuotient

    auto compareVec4DimsToTensorWeightsData = [](const TensorPtr& weights, const Vec4Dims& vec){
        for (int r=0; r<vec.size(); ++r)
        {
            for (int s=0; s<vec[r].size(); ++s)
            {
                for (int c=0; c<vec[r][s].size(); ++c)
                {
                    for (int k=0; k<vec[r][s][c].size(); ++k)
                    {
                        if (vec[r][s][c][k] != TensorElementAccess::getElement<float>(weights, {k,c,s,r}))
                        {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    };

    //for debug:
    // auto printWeightsTensor = [](TensorPtr& w){
    //     const auto& sizes = w->getAllSizesInElements();
    //     for (TSize r = 0; r < sizes[WEIGHT_DIM_R]; ++r)
    //     {
    //         for (TSize s = 0; s < sizes[WEIGHT_DIM_S]; ++s)
    //         {
    //             for (TSize c = 0; c < sizes[WEIGHT_DIM_C]; ++c)
    //             {
    //                 for (TSize k = 0; k < sizes[WEIGHT_DIM_K]; ++k)
    //                 {
    //                     std::cout<< TensorElementAccess::getElement<float>(w, {k,c,s,r});
    //                 }
    //                 std::cout<<""<<std::endl;
    //             }
    //         }
    //     }
    // };
    // auto printVec4Dims = [](const Vec4Dims& vec){
    //     for (auto vDim0: vec)
    //         {
    //             for (auto vDim1: vDim0)
    //             {
    //                 for (auto vDim2: vDim1)
    //                 {
    //                     for (auto elem: vDim2)
    //                     {
    //                         std::cout<<elem;
    //                     }
    //                     std::cout<<""<<std::endl;
    //                 }
    //             }
    //         };
    // };

    Vec4Dims expectedPackedWeightsToGroupsQuotient;
    Vec4Dims expectedPackedWeightsToGroupsRemainder;
    TensorPtr packedWeightsToGroupsQuotient;
    TensorPtr packedWeightsToGroupsRemainder;

    if (groupsRemainder==0)
    {
        ASSERT_TRUE(convNodes.size()==1) << "Expecting that the groups packer would return just one group convolution node";
        expectedPackedWeightsToGroupsQuotient = createPackedWeightsExpectedData(origWeights, 0, origWeights->getAllSizesInElements()[WEIGHT_DIM_K], groupsPerVector, kPerGroup);
        packedWeightsToGroupsQuotient = convNodes[0]->getInput(1);
        ASSERT_TRUE(compareVec4DimsToTensorWeightsData(packedWeightsToGroupsQuotient, expectedPackedWeightsToGroupsQuotient)) << "Weights were not packed as expected";
    }
    else
    {
        ASSERT_TRUE(convNodes.size()==2) << "Expecting that the groups packer would return two group convolution nodes";
        expectedPackedWeightsToGroupsQuotient = createPackedWeightsExpectedData(origWeights, 0, groupsQuotient*groupsPerVector*kPerGroup, groupsPerVector, kPerGroup);
        expectedPackedWeightsToGroupsRemainder = createPackedWeightsExpectedData(origWeights, groupsQuotient*groupsPerVector*kPerGroup, groupsRemainder*kPerGroup, groupsRemainder, kPerGroup);

        if (convNodes[0]->getInput(1)->getAllSizesInElements()[WEIGHT_DIM_K] > convNodes[1]->getInput(1)->getAllSizesInElements()[WEIGHT_DIM_K])
        {
            packedWeightsToGroupsQuotient = convNodes[0]->getInput(1);
            packedWeightsToGroupsRemainder = convNodes[1]->getInput(1);
        }
        else
        {
            packedWeightsToGroupsQuotient = convNodes[1]->getInput(1);
            packedWeightsToGroupsRemainder = convNodes[0]->getInput(1);
        }
        ASSERT_TRUE(compareVec4DimsToTensorWeightsData(packedWeightsToGroupsQuotient, expectedPackedWeightsToGroupsQuotient)) << "Weights were not packed as expected";
        ASSERT_TRUE(compareVec4DimsToTensorWeightsData(packedWeightsToGroupsRemainder, expectedPackedWeightsToGroupsRemainder)) << "Weights were not packed as expected";

    }

    //For debug:
    // std::cout<<"print tensors weights:"<<std::endl;
    // printWeightsTensor(packedWeightsToGroupsQuotient);
    // std::cout<<""<<std::endl;
    // std::cout<<""<<std::endl;
    // if (groupsRemainder>0)
    // {
    //     printWeightsTensor(packedWeightsToGroupsRemainder);
    //     std::cout<<""<<std::endl;
    //     std::cout<<""<<std::endl;
    // }
    // std::cout<<"print expected weights:"<<std::endl;
    // printVec4Dims(expectedPackedWeightsToGroupsQuotient);
    // std::cout<<""<<std::endl;
    // if (groupsRemainder>0)
    // {
    //     printVec4Dims(expectedPackedWeightsToGroupsRemainder);
    // }
}

TEST_P(Gaudi2ConstantFoldingGconvFwdInGroupPacking, gconv_fwd_constant_folding_for_static_weights)
{
    Gaudi2Graph g;
    ScopedConfigurationChange fuserCfg("ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING", "true");
    const unsigned mmeVectorSize        = g.getHALReader()->getMmeMinimalWidthInElems(syn_type_float);
    const unsigned kPerGroup = GetParam().kPerGroup;
    const unsigned nGroups = GetParam().nGroups;
    constexpr unsigned stride = 1;
    constexpr unsigned padBefore =0;
    constexpr unsigned padAfter = 0;
    constexpr unsigned dilation = 1;

    const SizeArray4dims& wSizes    = GetParam().wSizes;
    const SizeArray4dims& xMaxSizes = GetParam().xSizes;
    SizeArray4dims        yMaxSizes;
    yMaxSizes[DIM_C] = wSizes[WEIGHT_DIM_K];
    yMaxSizes[DIM_W] =
        convOutputDimSize(xMaxSizes[DIM_W], wSizes[WEIGHT_DIM_S], stride, padBefore + padAfter, dilation);
    yMaxSizes[DIM_H] =
        convOutputDimSize(xMaxSizes[DIM_H], wSizes[WEIGHT_DIM_R], stride, padBefore + padAfter, dilation);
    yMaxSizes[DIM_B] = xMaxSizes[DIM_B];

    const bool isDynamic = GetParam().isDynamic;
    // in case isDynamic is true, the batch dimension will be dynamic.
    const unsigned minBatch  = isDynamic ? 1 : xMaxSizes[DIM_B];
    SizeArray4dims xMinSizes = xMaxSizes;
    xMinSizes[DIM_B]         = minBatch;
    SizeArray4dims yMinSizes = yMaxSizes;
    yMinSizes[DIM_B]         = minBatch;

    synConvolutionParams params(wSizes[DIM_W],
                                wSizes[DIM_H],
                                stride,
                                stride,
                                padBefore,
                                padAfter,
                                padBefore,
                                padAfter,
                                dilation,
                                dilation);
    params.nGroups = nGroups;

    // Create inputs, output and convolution node:
    const auto x = createTensor(xMaxSizes, xMinSizes, syn_type_float);
    // Create the weights data, each kernel is filled with one value only (value = kernel index +1),
    // For example, tensor weights with these sizes: [3,2,2,2] looks like that:
    /*
        123
        123
        123
        123     RSC
        123
        123
        123
        123

        K
    */
    std::vector<float> vData(getTotalSize(wSizes));
    for (int i = 0; i<getTotalSize(wSizes) ; ++i)
    {
        vData[i] = i%wSizes[0] +1;
    }

    const auto w = createTensor(wSizes, wSizes, syn_type_float, true, reinterpret_cast<char*>(vData.data()));
    const auto w_for_validation =
        createTensor(wSizes, wSizes, syn_type_float, true, reinterpret_cast<char*>(vData.data()));
    createAndSetConstSectionToTensor(w, g);

    const auto y = createTensor(yMaxSizes, yMinSizes, syn_type_float, true);
    const auto conv2 =
        NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::convolutionNodeTypeName, "FwdConvolution");

    ASSERT_TRUE(GraphEditor::addNode(g, conv2));
    // GroupedConvolutionPackingManager is the object that handles the groups packing.
    GroupedConvolutionPackingManager gConvPacker(dynamic_cast<const ConvBaseNode*>(conv2.get()), g);
    NodeVector                       packGroupedConvNodes = gConvPacker.packGroupedConvNode();

    NodeVector convNodesReturnedByPacker;
    std::copy_if(packGroupedConvNodes.begin(), packGroupedConvNodes.end(), std::back_inserter(convNodesReturnedByPacker),[](const NodePtr& node) {
        return node && node->getNodeType() == Node::TYPE_CONVOLUTION;
    });

    validateWeightsOfConvNodesReturnedByPacker(w_for_validation, convNodesReturnedByPacker, mmeVectorSize, nGroups, kPerGroup);
    for (const auto& convNode : convNodesReturnedByPacker)
    {
        ASSERT_TRUE(convNode->getInput(1)->isStaticParam()) << "Expecting weights tensors to be static";
        ASSERT_TRUE(convNode->getInput(1)->inConstSection()) << "Expecting weights tensors to be a part of const section";
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         Gaudi2ConstantFoldingGconvFwdInGroupPacking,
                         ::testing::Values(Gaudi2ConstantFoldingGconvFwdInGroupPackingParams {5,2,{10,3,2,2},{15,5,5,1},false},
                         Gaudi2ConstantFoldingGconvFwdInGroupPackingParams {40,8,{320,2,2,2},{80,3,3,2},false},
                         Gaudi2ConstantFoldingGconvFwdInGroupPackingParams {5,2,{10,3,2,2},{15,5,5,2},true},
                         Gaudi2ConstantFoldingGconvFwdInGroupPackingParams {40,8,{320,2,2,2},{80,3,3,2},true}));
