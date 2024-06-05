#ifndef MME__GAUDI_UTILS_H
#define MME__GAUDI_UTILS_H

#include "include/gaudi/new_descriptor_generator/mme_common.h"

#define DEFAULT_DUMP_IDX 0xFFFFFFFF

void compareActivationSizes(const std::list<gaudi::MmeActivation>& newFlowActivations,
                            const std::list<gaudi::MmeActivation>& activations,
                            std::string nodeNameStr);
bool compareDescriptors(const Mme::Desc& desc, const Mme::Desc& refDesc, std::string str);
void dumpDescriptor(Mme::Desc& desc, std::string fileName);
void readDescriptor(Mme::Desc& desc, std::string fileName);
void dumpCompareDescriptors(const std::list<gaudi::MmeActivation>& activations, unsigned dumpIdx = DEFAULT_DUMP_IDX);

void dumpParams(MmeCommon::MmeLayerParams layerParams, std::string cfgName);

#endif //MME__GAUDI_UTILS_H
