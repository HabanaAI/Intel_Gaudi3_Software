#include "gaudi_mem_test.hpp"
#include "habana_global_conf.h"
#include "synapse_api.h"

class SynGaudiFlowMemTestsWithLessDcs : public SynFlowMemTests
{
public:
    SynGaudiFlowMemTestsWithLessDcs() : SynFlowMemTests() { setSupportedDevices({synDeviceGaudi}); }
    void SetUp() override;
    void afterSynInitialize() override
    {
        synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
        SynFlowMemTests::afterSynInitialize();
    }
};

REGISTER_SUITE(SynGaudiFlowMemTestsWithLessDcs, synTestPackage::ASIC_CI);

void SynGaudiFlowMemTestsWithLessDcs::SetUp()
{
    uint8_t cpDmasDcAmountForRecipe   = 15;
    uint8_t commandsDcAmountForRecipe = 1;

    GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP.setValue(cpDmasDcAmountForRecipe);
    GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP.setValue(cpDmasDcAmountForRecipe);
    GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP.setValue(cpDmasDcAmountForRecipe);

    GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_LOWER_CP.setValue(commandsDcAmountForRecipe);
    GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_LOWER_CP.setValue(commandsDcAmountForRecipe);
    GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_LOWER_CP.setValue(commandsDcAmountForRecipe);

    SynFlowMemTests::SetUp();
}

TEST_F_SYN(SynGaudiFlowMemTestsWithLessDcs, check_wcm_free_dc_memory)
{
    mem_test_internal(memTestErrorCodeDeviceMallocFail);
}