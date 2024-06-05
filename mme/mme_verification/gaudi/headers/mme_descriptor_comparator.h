#pragma once
#include "gaudi/mme_descriptor_generator.h"

class MmeDescriptorComparator
{
public:
    MmeDescriptorComparator() = default;
    ~MmeDescriptorComparator() = default;
    bool compare(const std::list<std::pair<gaudi::MmeActivation, gaudi::MmeActivation>>& newActivations,
                 const std::list<std::pair<gaudi::MmeActivation, gaudi::MmeActivation>>& oldActivations);
private:
    bool compareActivation(const gaudi::MmeActivation& newAct,
                           const gaudi::MmeActivation& oldAct);
    bool compareDesc(const Mme::Desc& newDesc, const Mme::Desc& oldDesc);
    bool compareOverLapRoi(const OverlapRoi& newRoi, const OverlapRoi& oldRoi);
    void printError();
    template <typename T>
    bool compareAttribute(const T& newAttrib, const T& oldAttrib, std::string description);
    bool compareTensor(const Mme::MmeTensorDesc& newTensor, const Mme::MmeTensorDesc& oldTensor, std::string tensorName);
    bool compareAgu(const Mme::MmeAguCoreDesc& newAgu, const Mme::MmeAguCoreDesc& oldAgu, std::string aguName);
    std::string m_where;
};