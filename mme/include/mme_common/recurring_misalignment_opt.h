#ifndef MME__RECURRING_MISALIGNMENT_OPT_H
#define MME__RECURRING_MISALIGNMENT_OPT_H

#include "include/mme_common/mme_common_enum.h"

namespace MmeCommon
{
class CommonGeoAttr;
class ConvSubProblemContainer;
class MmeHalReader;

class RecurringMisalignmentOptimization
{
public:
    static unsigned getNumSubProblems(const MmeCommon::MmeLayerParams& params,
                                      const CommonGeoAttr& geoAttr,
                                      const MmeHalReader& mmeHalReader);
    static unsigned getCutPointPerSubProblem(const MmeCommon::MmeLayerParams& params,
                                             const CommonGeoAttr& geoAttr,
                                             ChipType chipType);
    static unsigned getCutPointPerSubProblem(const MmeCommon::MmeLayerParams& params,
                                             const CommonGeoAttr& geoAttr,
                                             const MmeHalReader& mmeHalReader);
    static bool isMultipleAccessToSameCL(const MmeLayerParams& params,
                                         const MmeHalReader& mmeHalReader,
                                         EMmeInternalOperand operand);
    static bool isRecurringMisalignment(const MmeLayerParams& params,
                                        EMmeInternalOperand operand,
                                        ChipType chipType);
    static bool isRecurringMisalignment(const MmeLayerParams& params,
                                        const MmeHalReader& mmeHalReader,
                                        EMmeInternalOperand operand);
    static std::string getDebugInfo(const ConvSubProblemContainer& convSubProblems,
                                    const CommonGeoAttr& geoAttr,
                                    const MmeHalReader& mmeHal,
                                    const MmeLayerParams& originalParams);
    static void makeParamsForSubProblem(const MmeLayerParams& originalParams,
                                        unsigned numSubProblems,
                                        unsigned subProblemIdx,
                                        MmeCommon::MmeLayerParams& subProblemParams,
                                        OffsetArray& descAddrOffset);
    static unsigned calcNumSubProblems(const MmeCommon::MmeLayerParams& params,
                                       const MmeHalReader& mmeHalReader,
                                       MmeCommon::EMmeInternalOperand operand);

protected:
    static bool canApplyOptimization(const MmeCommon::MmeLayerParams& params,
                                     const CommonGeoAttr& geoAttr,
                                     const MmeHalReader& mmeHalReader);
    static bool isNumSubProblemsValid(unsigned numSubProblems) { return ((numSubProblems > 0)); }
};
}  // namespace MmeCommon

#endif //MME__RECURRING_MISALIGNMENT_OPT_H
