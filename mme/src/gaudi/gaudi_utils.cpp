#include "include/gaudi/gaudi_utils.h"
#include <iostream>
#include "mme_assert.h"
#include "include/mme_common/mme_common_enum.h"
#include "mme_params_dumper.h"

using namespace MmeCommon;

template<typename T>
static void printArray(const T* a, const uint32_t size, std::string str)
{
    std::cout << "    " << str << " array: ";
    for (int i = 0; i < size; i++)
    {
        std::cout << a[i];
        if (i < (size - 1))
        {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

template<typename T>
static bool compareArray(const T* a, const T* ra, unsigned size, std::string str)
{
    bool match = true;
    for (int i = 0; i < size; i++)
    {
        if (a[i] != ra[i]) match = false;
    }
    if (match)
    {
        return true;
    }

    std::cout << "Mismatch in array " << str << std::endl;
    printArray<T>(a, size, "src " + str);
    printArray<T>(ra, size, "ref " + str);
    return false;
}

static bool compareBlock(const void* b, const void* rb, unsigned size, std::string str)
{
    if (memcmp(b, rb, size))
    {
        std::cout << "Mismatch in " << str << std::endl;
        return false;
    }
    return true;
}

static bool compareField(unsigned v, unsigned rv, std::string str)
{
    if (v != rv)
    {
        std::cout << "Mismatch in " << str << ". Actual " << v << " , reference " << rv << " ." << std::endl;
        return false;
    }
    return true;
}

static bool compareAssociatedDimsSingle(Mme::MmeAssociatedDims ad, Mme::MmeAssociatedDims ref_ad, std::string str)
{
    if (ad.dimS != ref_ad.dimS)
    {
        std::cout << "Mismatch in " << str << ".dimS Got " << ad.dimS << " , reference " << ref_ad.dimS << std::endl;
        return false;
    }
    if (ad.dimL != ref_ad.dimL)
    {
        std::cout << "Mismatch in " << str << ".dimL Got " << ad.dimL << " , reference " << ref_ad.dimL << std::endl;
        return false;
    }
    if (ad.dimO != ref_ad.dimO)
    {
        std::cout << "Mismatch in " << str << ".dimO Got " << ad.dimO << " , reference " << ref_ad.dimO << std::endl;
        return false;
    }
    return true;
}
bool compareAssciatedDims(const Mme::MmeAssociatedDims* ad, const Mme::MmeAssociatedDims* ref_ad, std::string str)
{
    bool isEqual = true;
    for (int d = 0; d < Mme::c_mme_max_conv_dims; d++)
    {
        isEqual &= compareAssociatedDimsSingle(ad[d], ref_ad[d], str + "[" + std::to_string(d) + "]");
    }
    return isEqual;
}

bool compareAxi(Mme::MmeUserData axiUserData, Mme::MmeUserData refAxiUserData, std::string str)
{
    if (compareBlock(&axiUserData, &refAxiUserData, sizeof(axiUserData), "axiUserData"))
    {
        return true;
    }

    if (axiUserData.first != refAxiUserData.first)
    {
        std::cout << "Mismatch in axiUserData.first . Got " << axiUserData.first << " , reference "
                  << refAxiUserData.first << std::endl;
    }
    if (axiUserData.steady != refAxiUserData.steady)
    {
        std::cout << "Mismatch in axiUserData.steady . Got " << axiUserData.steady << " , reference "
                  << refAxiUserData.steady << std::endl;
    }
    if (axiUserData.mask != refAxiUserData.mask)
    {
        std::cout << "Mismatch in axiUserData.mask . Got " << axiUserData.mask << " , reference " << refAxiUserData.mask
                  << std::endl;
    }

    return false;
}

bool compareDescriptors(const Mme::Desc& desc, const Mme::Desc& refDesc, std::string str)
{
    bool result;
    result = compareBlock(&desc, &refDesc, sizeof(desc), "full descriptors");
    if (result)
    {
        std::cout << str << " : Both descriptors match" << std::endl;
        return true;
    }

    std::cout << "===== Compare descriptors " << str << " =====" << std::endl;
    compareBlock(&desc.header, &refDesc.header, sizeof(desc.header), "header");
    compareBlock(&desc.conv, &refDesc.conv, sizeof(desc.conv), "conv");
    compareBlock(&desc.outerLoop, &refDesc.outerLoop, sizeof(desc.outerLoop), "outerLoop");
    compareBlock(&desc.tensorS, &refDesc.tensorS, sizeof(desc.tensorS), "tensorS");
    compareBlock(&desc.aguS, &refDesc.aguS, sizeof(desc.aguS), "aguS");
    compareBlock(&desc.tensorL, &refDesc.tensorL, sizeof(desc.tensorL), "tensorL");
    compareBlock(&desc.aguL, &refDesc.aguL, sizeof(desc.aguL), "aguL");
    compareBlock(&desc.tensorO, &refDesc.tensorO, sizeof(desc.tensorO), "tensorO");
    compareBlock(&desc.aguO, &refDesc.aguO, sizeof(desc.aguO), "aguO");
    compareBlock(&desc.sbRepeat, &refDesc.sbRepeat, sizeof(desc.sbRepeat), "sbRepeat");
    compareBlock(&desc.rateLimiter, &refDesc.rateLimiter, sizeof(desc.rateLimiter), "rateLimiter");
    compareBlock(&desc.syncObject, &refDesc.syncObject, sizeof(desc.syncObject), "syncObject");
    compareBlock(&desc.axiUserData, &refDesc.axiUserData, sizeof(desc.axiUserData), "axiUserData");
    compareBlock(&desc.perfEvtS, &refDesc.perfEvtS, sizeof(desc.perfEvtS), "perfEvtS");
    compareBlock(&desc.perfEvtL, &refDesc.perfEvtL, sizeof(desc.perfEvtL), "perfEvtL");
    compareBlock(&desc.perfEvtO, &refDesc.perfEvtO, sizeof(desc.perfEvtO), "perfEvtO");
    compareBlock(&desc.metaData, &refDesc.metaData, sizeof(desc.metaData), "metaData");
    compareBlock(&desc.pcu, &refDesc.pcu, sizeof(desc.pcu), "pcu");
    compareBlock(&desc.sw, &refDesc.sw, sizeof(desc.sw), "sw");

    compareField(desc.numIterationsMinus1, refDesc.numIterationsMinus1, "numIterationsMinus1");

    compareField(desc.baseAddrHighS, refDesc.baseAddrHighS, "baseAddrHighS");
    compareField(desc.baseAddrHighL, refDesc.baseAddrHighL, "baseAddrHighL");
    compareField(desc.baseAddrHighO, refDesc.baseAddrHighO, "baseAddrHighO");
    compareField(desc.baseAddrLowS, refDesc.baseAddrLowS, "baseAddrLowS");
    compareField(desc.baseAddrLowL, refDesc.baseAddrLowL, "baseAddrLowL");
    compareField(desc.baseAddrLowO, refDesc.baseAddrLowO, "baseAddrLowO");

    compareField(desc.header.transS, refDesc.header.transS, "header.transS");
    compareField(desc.header.transL, refDesc.header.transL, "header.transL");
    compareField(desc.header.transO, refDesc.header.transO, "header.transO");
    compareField(desc.header.advanceS, refDesc.header.advanceS, "header.advanceS");
    compareField(desc.header.advanceL, refDesc.header.advanceL, "header.advanceL");
    compareField(desc.header.advanceO, refDesc.header.advanceO, "header.advanceO");

    compareField(desc.header.partialHeightLoopS, refDesc.header.partialHeightLoopS, "header.partialHeightLoopS");
    compareField(desc.header.partialHeightLoopLLocal,
                 refDesc.header.partialHeightLoopLLocal,
                 "header.partialHeightLoopLLocal");
    compareField(desc.header.partialHeightLoopLRemote,
                 refDesc.header.partialHeightLoopLRemote,
                 "header.partialHeightLoopLRemote");
    compareField(desc.header.partialHeightLoopOLocal,
                 refDesc.header.partialHeightLoopOLocal,
                 "header.partialHeightLoopOLocal");
    compareField(desc.header.partialHeightLoopORemote,
                 refDesc.header.partialHeightLoopORemote,
                 "header.partialHeightLoopORemote");

    compareField(desc.header.lowerS, refDesc.header.lowerS, "header.lowerS");
    compareField(desc.header.lowerL, refDesc.header.lowerL, "header.lowerL");
    compareField(desc.header.storeEn, refDesc.header.storeEn, "header.storeEn");
    compareField(desc.header.rollAccums, refDesc.header.rollAccums, "header.rollAccums");
    compareField(desc.header.accum, refDesc.header.accum, "header.accum");

    compareField(desc.sbRepeat.aguSLoopMask, refDesc.sbRepeat.aguSLoopMask, "desc.sbRepeat.aguSLoopMask");
    compareField(desc.sbRepeat.aguLLoopMask, refDesc.sbRepeat.aguLLoopMask, "desc.sbRepeat.aguLLoopMask");
    compareField(desc.sbRepeat.repeatLMinus1, refDesc.sbRepeat.repeatLMinus1, "desc.sbRepeat.repeatLMinus1");
    compareField(desc.sbRepeat.repeatSMinus1, refDesc.sbRepeat.repeatSMinus1, "desc.sbRepeat.repeatSMinus1");
    compareField(desc.sbRepeat.loadS, refDesc.sbRepeat.loadS, "desc.sbRepeat.loadS");
    compareField(desc.sbRepeat.teEnS, refDesc.sbRepeat.teEnS, "desc.sbRepeat.teEnS");
    compareField(desc.sbRepeat.loadL, refDesc.sbRepeat.loadL, "desc.sbRepeat.loadL");
    compareField(desc.sbRepeat.teEnL, refDesc.sbRepeat.teEnL, "desc.sbRepeat.teEnL");

    compareArray<uint8_t>(desc.conv.kernelSizeMinus1.dim,
                          refDesc.conv.kernelSizeMinus1.dim,
                          4,
                          "conv.kernelSizeMinus1.dim");
    compareAssciatedDims(desc.conv.associatedDims, refDesc.conv.associatedDims, "conv.associatedDims");

    compareAssociatedDimsSingle(desc.outerLoop.associatedDims,
                                refDesc.outerLoop.associatedDims,
                                "outerLoop.associatedDims");
    compareField(desc.outerLoop.sizeMinus1, refDesc.outerLoop.sizeMinus1, "outerLoop.sizeMinus1");

    compareArray<uint32_t>(desc.tensorS.validElements, refDesc.tensorS.validElements, 5, "tensorS.validElements");
    compareArray<int32_t>(desc.tensorS.loopStride, refDesc.tensorS.loopStride, 5, "tensorS.loopStride");
    compareArray<int32_t>(desc.tensorS.roiSize, refDesc.tensorS.roiSize, 4, "tensorS.roiSize");
    compareArray<uint32_t>(desc.tensorS.spatialStrides, refDesc.tensorS.spatialStrides, 4, "tensorS.spatialStrides");
    compareField(desc.tensorS.spatialSizeMinus1, refDesc.tensorS.spatialSizeMinus1, "tensorS.spatialSizeMinus1");

    compareArray<uint32_t>(desc.tensorL.validElements, refDesc.tensorL.validElements, 5, "tensorL.validElements");
    compareArray<int32_t>(desc.tensorL.loopStride, refDesc.tensorL.loopStride, 5, "tensorL.loopStride");
    compareArray<int32_t>(desc.tensorL.roiSize, refDesc.tensorL.roiSize, 4, "tensorL.roiSize");
    compareArray<uint32_t>(desc.tensorL.spatialStrides, refDesc.tensorL.spatialStrides, 4, "tensorL.spatialStrides");
    compareField(desc.tensorL.spatialSizeMinus1, refDesc.tensorL.spatialSizeMinus1, "tensorL.spatialSizeMinus1");

    compareArray<uint32_t>(desc.tensorO.validElements, refDesc.tensorO.validElements, 5, "tensorO.validElements");
    compareArray<int32_t>(desc.tensorO.loopStride, refDesc.tensorO.loopStride, 5, "tensorO.loopStride");
    compareArray<int32_t>(desc.tensorO.roiSize, refDesc.tensorO.roiSize, 4, "tensorO.roiSize");
    compareArray<uint32_t>(desc.tensorO.spatialStrides, refDesc.tensorO.spatialStrides, 4, "tensorO.spatialStrides");
    compareField(desc.tensorO.spatialSizeMinus1, refDesc.tensorO.spatialSizeMinus1, "tensorO.spatialSizeMinus1");

    compareArray<int32_t>(desc.aguS.roiBaseOffset, refDesc.aguS.roiBaseOffset, 5, "aguS.roiBaseOffset");
    compareArray<uint32_t>(desc.aguS.startOffset, refDesc.aguS.startOffset, 4, "aguS.startOffset");

    compareArray<int32_t>(desc.aguL[0].roiBaseOffset, refDesc.aguL[0].roiBaseOffset, 5, "aguL[0].roiBaseOffset");
    compareArray<uint32_t>(desc.aguL[0].startOffset, refDesc.aguL[0].startOffset, 4, "aguL[0].startOffset");
    compareArray<int32_t>(desc.aguL[1].roiBaseOffset, refDesc.aguL[1].roiBaseOffset, 5, "aguL[1].roiBaseOffset");
    compareArray<uint32_t>(desc.aguL[1].startOffset, refDesc.aguL[1].startOffset, 4, "aguL[1].startOffset");

    compareArray<int32_t>(desc.aguO[0].roiBaseOffset, refDesc.aguO[0].roiBaseOffset, 5, "aguO[0].roiBaseOffset");
    compareArray<uint32_t>(desc.aguO[0].startOffset, refDesc.aguO[0].startOffset, 4, "aguO[0].startOffset");
    compareArray<int32_t>(desc.aguO[1].roiBaseOffset, refDesc.aguO[1].roiBaseOffset, 5, "aguO[1].roiBaseOffset");
    compareArray<uint32_t>(desc.aguO[1].startOffset, refDesc.aguO[1].startOffset, 4, "aguO[1].startOffset");

    compareAxi(desc.axiUserData, refDesc.axiUserData, "axiUserData");

    return false;
}

void compareActivationSizes(const std::list<gaudi::MmeActivation>& newFlowActivations,
                            const std::list<gaudi::MmeActivation>& activations,
                            std::string nodeNameStr)
{
    std::cout << "Node " << nodeNameStr << ": " << newFlowActivations.size() << " descriptors vs. "
              << activations.size() << " in the reference" << std::endl;
}

void dumpDescriptor(const Mme::Desc& desc, std::string fileName)
{
    FILE* fp = fopen(fileName.c_str(), "w");
    if (fp == nullptr)
    {
        printf("Error! Cannot open descriptor output dump file\n");
    }
    else
    {
        fwrite(&desc, 1, sizeof(Mme::Desc), fp);
        fclose(fp);
    }
}

static bool getEnvFlag(std::string envFlag)
{
    auto flagStr = getenv(envFlag.c_str());
    if (flagStr != nullptr)
    {
        return std::atoi(flagStr);
    }
    return false;
}

void readDescriptor(Mme::Desc& desc, std::string fileName)
{
    FILE* fp = fopen(fileName.c_str(), "r");
    if (fp == nullptr)
    {
        printf("Error! Cannot open descriptor input dump file\n");
    }
    else
    {
        size_t bytesRead = fread(&desc, sizeof(Mme::Desc), 1, fp);
        MME_ASSERT(bytesRead == sizeof(Mme::Desc), "didnt read enough bytes");
        fclose(fp);
    }
}

void dumpCompareDescriptors(const std::list<gaudi::MmeActivation>& activations, unsigned dumpIdx)
{
    bool isDumpDesc = getEnvFlag("DUMP_MME_DESC");
    bool isCompareDesc = getEnvFlag("COMPARE_MME_DESC");

    auto it = activations.begin();
    for (int actIdx = 0; (actIdx < activations.size()); actIdx++)
    {
        const Mme::Desc& southDesc = it->getDesc(1);
        const Mme::Desc& northDesc = it->getDesc(0);

        std::string southFileName = "refDescSouth";
        std::string northFileName = "refDescNorth";
        if (dumpIdx != DEFAULT_DUMP_IDX)
        {
            southFileName = southFileName + "_" + std::to_string(dumpIdx);
            northFileName = northFileName + "_" + std::to_string(dumpIdx);
        }
        southFileName = southFileName + ".dump";
        northFileName = northFileName + ".dump";

        if (isDumpDesc)
        {
            dumpDescriptor(southDesc, southFileName);
            dumpDescriptor(northDesc, northFileName);
        }
        if (isCompareDesc)
        {
            Mme::Desc refSouthDesc;
            readDescriptor(refSouthDesc, southFileName);
            std::string southMsg = "South Desc " + std::to_string(dumpIdx == DEFAULT_DUMP_IDX ? 0 : dumpIdx);
            compareDescriptors(southDesc, refSouthDesc, southMsg);

            Mme::Desc refNorthDesc;
            readDescriptor(refNorthDesc, northFileName);
            std::string northMsg = "North Desc " + std::to_string(dumpIdx == DEFAULT_DUMP_IDX ? 0 : dumpIdx);
            compareDescriptors(northDesc, refNorthDesc, northMsg);
        }
    }
}

void dumpParams(MmeCommon::MmeLayerParams params, std::string nodeName)
{
    MmeCommon::MmeParamsDumper(params).dumpMmeParamsForGaudiCfg("all", nodeName);
}
