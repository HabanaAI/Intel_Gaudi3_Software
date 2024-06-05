#include "mme_unit_test.h"
#include "include/mme_common/mme_brain.h"
#include "json.hpp"

using json = nlohmann::json;
using namespace MmeCommon;

struct MmeBwTestParams
{
    // node definition
    std::vector<unsigned> nodeDims;
    EMmeOpType opType;
    // solution params
    float accessBwA;
    float accessBwB;
    unsigned fetchNrA;
    unsigned fetchNrB;
};

struct MmeUtilzationTestParams
{
    std::vector<unsigned> nodeDims;
    std::vector<unsigned> granularity;
    float maxUtilization;
    float mmeUtilization;
};

struct MmeSolutionParams
{
    std::vector<unsigned> nodeDims;
    std::vector<unsigned> granularity;
    bool broadcastB = false;
    EMmeOpType opType = MmeCommon::e_mme_ab;
    EMmeDataType dataType = MmeCommon::e_type_bf16;
    unsigned solutionNr = 0;
};

struct MmeInflationParams
{
    std::vector<unsigned> nodeDims;
    std::vector<unsigned> granularity;
    float requiredPerf;
    bool broadcastB = false;
    EMmeOpType opType = MmeCommon::e_mme_fwd;
    EMmeDataType dataType = MmeCommon::e_type_bf16;
};

class MmeUTBrainTest : public MMEUnitTest
{
public:
    void ConvGeomtryTest(size_t nOFM,
                         size_t nIFM,
                         size_t wOFM,
                         size_t hOFM,
                         size_t batch,
                         MmeCommon::EMmeOpType operation,
                         MmeCommon::EMmeDataType dataType,
                         MmeCommon::EMmeGeometry expectedGeometry,
                         MmeCommon::EMmePattern expectedPattern,
                         unsigned fetchA,
                         unsigned fetchB);

    void BrainSolutionTest(MmeSolutionParams testParams);
    void BrainBwTest(MmeBwTestParams testParams);
    void BrainUtilizaitonTest(MmeUtilzationTestParams testParams);
    void BrainRedundantStrategiesTest(MmeSolutionParams testParams);
    void BrainSolutionInflationTest(MmeInflationParams testParams);

    void checkFlattening(const std::vector<unsigned>& aShape,
                         const std::vector<unsigned>& bShape,
                         const std::vector<unsigned>& cShape,
                         const std::vector<unsigned>& aStrides,
                         const std::vector<unsigned>& bStrides,
                         const std::vector<unsigned>& cStrides,
                         MmeCommon::EMmeDataType dataType,
                         MmeCommon::EMmeGeometry expectedGeometry,
                         unsigned expectedFlatteningFactor);

    MmeLayerParams
    setParams(std::vector<unsigned>& nodeDims, EMmeOpType operation, EMmeDataType dataType, bool broadcastB = false);
};
