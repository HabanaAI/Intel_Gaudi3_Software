#pragma once

#include "json.hpp"
#include "general_utils.h"
#include "mme_mem_access_checker.h"
#include "sim_tensor.h"
#include <random>

typedef struct MmeSyncObjectsArry
{
    // first value is the so Idx, second value is its address
    std::pair<uint32_t, uint64_t> Primary = {0, 0};  // primary SO
    std::pair<uint32_t, uint64_t> Secondary = {0, 0};  // secondary SO
    std::pair<uint32_t, uint64_t> PrimarySlave = {0, 0};  // slave SO
    std::pair<uint32_t, uint64_t> SecondarySlave = {0, 0};  // slave secondary SO
} MmeSyncObjects;
typedef struct MmeTensorAttributes_t
{
    std::vector<MmeCommon::SizeArray> sizes;
    std::vector<MmeCommon::SizeArray> strides;
    unsigned dims;
    MmeCommon::InfNanMode infNanMode;
    MmeCommon::EMmeDataType dataType;
    unsigned fpBias;
} MmeTensorAttributes;
struct AuxTensorData
{
    std::shared_ptr<MmeSimTensor> pTensor = nullptr;
    uint64_t                      addr = 0;
    bool                          isSram = false;
    bool                          isInput;
};
using MmeAuxTensors = std::array<AuxTensorData, MmeCommon::MmeAuxTensorIdx::AUX_TENSOR_MAX_NUM>;

struct MmeTensorParams
{
    std::shared_ptr<MmeSimTensor> xHost = nullptr;
    std::shared_ptr<MmeSimTensor> wHost = nullptr;
    std::shared_ptr<MmeSimTensor> yHost = nullptr;
    std::shared_ptr<MmeSimTensor> oHost = nullptr;
    std::shared_ptr<MmeSimTensor> outRef = nullptr;
    MmeAuxTensors auxTensors = {};

    std::shared_ptr<MmeSimTensor> outDevSim0 = nullptr;
    std::shared_ptr<MmeSimTensor> outDevSim1 = nullptr;
    std::vector<std::shared_ptr<MmeSimTensor>> outDevChip0 = {};
    std::vector<std::shared_ptr<MmeSimTensor>> outDevChip1 = {};

    uint64_t xAddr;
    uint64_t wAddr;
    uint64_t yAddr;
    uint64_t oAddr;
};
using MmeDataParams = struct MmeDataParams_t
{
    std::vector<MmeTensorParams> tensorParams;

    // indication of location of operands. order is - {X, W , Y, O}
    std::array<bool, 4> operandInSram = {true, true, true, true};
    std::vector<bool> auxTensorInSram = {};    // location of aux tensors
    std::vector<MmeSyncObjects> syncObjects;
    std::vector<unsigned> soValues;

    unsigned seed = 0;
    unsigned wkldId = 0;
    MmeCommon::ChipType m_chipType;

    nlohmann::json* testJson;

    std::shared_ptr<MmeCommon::MmeMemAccessChecker> memAccessChecker = nullptr;
};

class DataGenerator
{
public:
    DataGenerator(MmeCommon::ChipType chipType, unsigned wkldId, unsigned seed)
    {
        m_dataParams.wkldId = wkldId;
        m_dataParams.seed = seed;
        m_dataParams.m_chipType = chipType;
    }
    ~DataGenerator() = default;
    bool createAndInitDataParams(const nlohmann::json& testJson,
                                 bool runOnSim,
                                 unsigned driverDevicesNr);
    MmeDataParams& getParams() { return m_dataParams; }
    static unsigned
    getFpBias(const MmeCommon::EMmeDataType type, const unsigned fpBias, MmeCommon::ChipType m_chipType);

private:
    bool createParams(const nlohmann::json& testJson,
                      bool enableSecondOutput,
                      bool runOnSim,
                      unsigned driverDevicesNr);
    void generateData(const nlohmann::json& testJson, bool enableSecondOutput, bool hasReduction);

    void getTensorAttrByOp(const nlohmann::json& testJson,
                           MmeTensorAttributes& xAttr,
                           MmeTensorAttributes& wAttr,
                           MmeTensorAttributes& yAttr,
                           MmeTensorAttributes& oAttr) const;
    MmeCommon::SizeArray getSizeArrayFromVector(const std::vector<unsigned>& configArray) const;
    unsigned getFpBias(const MmeCommon::EMmeDataType type, const unsigned fpBias) const;
    bool verifyTensorSizes(const MmeCommon::SizeArray& xSizes,
                           const MmeCommon::SizeArray& wSizes,
                           const MmeCommon::SizeArray& ySizes,
                           const MmeCommon::EMmeOpType& op,
                           unsigned packingFactor,
                           unsigned reductionLevel) const;

    void fillTensorData(RandomSimTensorGenerator& randomGen,
                        pMMESimTensor& tensor,
                        MmeCommon::EMmeOpType op,
                        bool normalDist,
                        bool unitMatrix,
                        float minValue,
                        float maxValue,
                        bool fillForReductionPacking = false,
                        unsigned packingFactor = 1,
                        unsigned reductionLevel = 1,
                        float mean = 0.0,
                        float stdDev = 1.0);

    bool shouldMemsetOutput(const nlohmann::json& testJson);
    pMMESimTensor getActualOutputTensor(MmeCommon::EMmeOpType op, unsigned gemm) const;
    void getRefTensorMapper(MmeCommon::EMmeOpType op, unsigned gemm, pMMESimTensor mapperTensors[]) const;

    MmeDataParams m_dataParams;
};