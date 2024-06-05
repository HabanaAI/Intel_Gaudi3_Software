#ifndef MME__PARAMS_DUMPER_H
#define MME__PARAMS_DUMPER_H

#include "json.hpp"
#include "include/mme_common/mme_common_enum.h"

using Json = nlohmann::json;

namespace MmeCommon
{
class MmeParamsDumper
{
public:
    MmeParamsDumper(const MmeLayerParams& params) : m_params(params) {};
    void dumpMmeParamsJson();
    void dumpMmeParamsForGaudiCfg(std::string opStrCfg, std::string nodeName = "");
    static std::string getGeometryName(const EMmeGeometry& geometry);

private:
    void createMmeParamsJson();
    void createTensorJson(const MmeTensorView& tensor, const std::string& tensorName);
    void createControlsJson();
    void createStrategyJson();

    // enum to string converters
    static std::string getOpTypeName(const EMmeOpType& opType);
    static std::string getDataTypeName(const EMmeDataType& dataType);
    static std::string getReductionOpName(const EMmeReductionOp& reductionOp);
    static std::string getReductionRoundingModeName(const EMmeReductionRm& reductionRm);
    static std::string getRoundingModeName(const RoundingMode& roundingMode);
    static std::string getSignalingModeName(const EMmeSignalingMode& signalingMode);
    static std::string getPatternName(const EMmePattern& pattern);

    const MmeLayerParams& getParams() { return m_params; }

    MmeLayerParams m_params;
    Json m_paramsJson;
};
}  // namespace MmeCommon

#endif //MME__PARAMS_DUMPER_H
