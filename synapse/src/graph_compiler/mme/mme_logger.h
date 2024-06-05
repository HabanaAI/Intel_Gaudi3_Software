#pragma once

#include "include/mme_common/mme_brain.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/mme_descriptor_generator_base.h"
#include "mme_node.h"

namespace gaudi { class MmeDescriptorGenerator; }
class MmeLogger
{
public:
    MmeLogger() = default;
    void printDebugInfo(const MmeCommon::PerfAttr&             perfAttr,
                        MmeCommon::MmeDescriptorGeneratorBase* descGenerator);
    void printDebugInfoGaudi(gaudi::MmeDescriptorGenerator* descGenerator);
    void printMmeParams(const MmeCommon::MmeLayerParams& params);
    std::string getMmeStrategyInfo(const MmeCommon::MmeLayerParams&       params,
                                   MmeCommon::MmeDescriptorGeneratorBase& descGenerator,
                                   MmeCommon::ChipType                    chipType);
    void        printMmePerf(const MmeCommon::PerfAttr& perfAttr);
private:
    void        printMmeRecipeInfo(MmeCommon::MmeDescriptorGeneratorBase* descGenerator);
    void        printMmeRecipeInfo(gaudi::MmeDescriptorGenerator* descGenerator);
    void        printMmeDescriptor(const MmeCommon::MmeDescriptorGeneratorBase* descGenerator);
    void        printMmeDescriptor(const gaudi::MmeDescriptorGenerator* descGenerator);
    std::string createTensorString(const MmeCommon::MmeTensorView& tensor);
    std::string getGeometryInfo(const MmeCommon::MmeLayerParams&       params,
                                MmeCommon::MmeDescriptorGeneratorBase& descGenerator);
    std::string getRecipeInfo(MmeCommon::MmeDescriptorGeneratorBase& descGenerator,
                              const MmeCommon::MmeLayerParams&       params);
    std::string getRecurringMisalignmentInfo(MmeCommon::MmeDescriptorGeneratorBase& descGenerator,
                                             const MmeCommon::MmeLayerParams&       params,
                                             const MmeCommon::ChipType              chipType);
};
