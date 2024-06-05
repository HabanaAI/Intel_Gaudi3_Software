#include "gc_gaudi_test_infra.h"
#include "../src/platform/gaudi/graph_compiler/descriptor_generator.h"
#include "gaudi/mme_descriptor_generator.h"
#include "include/mme_common/mme_brain.h"

using namespace gaudi;

const uint32_t cacheSize = 3;

class SynGaudiDesCacheTest : public SynTest
{
public:
    SynGaudiDesCacheTest()
    {
        if (m_deviceType == synDeviceTypeInvalid)
        {
            LOG_WARN(SYN_TEST,
                     "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi");
            m_deviceType = synDeviceGaudi;
        }
        setSupportedDevices({synDeviceGaudi});
        DES_CACHE.DescriptorsCacheInit(m_sizeOfCache);
    };
    ~SynGaudiDesCacheTest()
    {

    };

    virtual void SetUpTest() override;
    virtual void TearDownTest() override;

    MmeCommon::MmeLayerParams setDefaultConvParams()
    {
        //Init convolution params with some default values
        MmeCommon::MmeLayerParams layerParams =
            MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi);  // Initialized to default

        MmeCommon::MmeTensorView temp;
        for (unsigned dim=0; dim<Mme::c_mme_max_tensor_dims; dim++)
        {
            temp.sizes[dim]   = 1;  // Default value for this test class
            temp.strides[dim] = 1;  // Default value for this test class
            temp.bases[dim] = 0;
        }
        temp.elementType = MmeCommon::e_type_bf16;
        memcpy(&layerParams.x, &temp, sizeof(MmeCommon::MmeTensorView));
        memcpy(&layerParams.y, &temp, sizeof(MmeCommon::MmeTensorView));
        memcpy(&layerParams.w, &temp, sizeof(MmeCommon::MmeTensorView));

        layerParams.strategy.pattern  = MmeCommon::e_mme_z_reduction_ksf;
        layerParams.strategy.geometry = MmeCommon::e_mme_geometry_2wx2h;

        return layerParams;
    }

    uint32_t m_sizeOfCache = cacheSize;
};

void SynGaudiDesCacheTest::SetUpTest()
{
    if (!shouldRunTest()) GTEST_SKIP() << m_testConfig.skipReason();
    SetTestFileName();
    m_setupStatus = true;
}

void SynGaudiDesCacheTest::TearDownTest()
{
    printProfileInformation();
    CleanTestIntermediatesFiles();
}

TEST_F_GC(SynGaudiDesCacheTest, add_to_cache_same_element)
{
    //Creating 2 identical convolution params and insert both to MME descriptor's cache.
    //Test verify the cache include only 1 entry
    std::list<MmeActivation> activations;

    MmeCommon::MmeLayerParams params = setDefaultConvParams();
    DES_CACHE.generateDescriptorsCache(params, activations);

    MmeCommon::MmeLayerParams params2 = setDefaultConvParams();
    DES_CACHE.generateDescriptorsCache(params2, activations);

    uint32_t currCacheSize =  DES_CACHE.getCacheSize();

    LOG_INFO(GC,"Current cache size {}", currCacheSize);

    ASSERT_EQ(currCacheSize,1) << "Test Failed! expected num of elements = 1";
}

TEST_F_GC(SynGaudiDesCacheTest, differnt_ctxId)
{
    //Creating 2 convolution params with different ctxId and insert both to MME descriptor's cache.
    //Test verify the cache include only 1 entry
    std::list<MmeActivation> activations;

    MmeCommon::MmeLayerParams params = setDefaultConvParams();
    DES_CACHE.generateDescriptorsCache(params, activations);

    MmeCommon::MmeLayerParams params2 = setDefaultConvParams();
    params2.tracing.ctxId = 7;
    DES_CACHE.generateDescriptorsCache(params2, activations);

    uint32_t currCacheSize =  DES_CACHE.getCacheSize();

    LOG_INFO(GC,"Current cache size {}", currCacheSize);

    ASSERT_EQ(currCacheSize,1) << "Test Failed! expected num of elements = 1";
}

TEST_F_GC(SynGaudiDesCacheTest, differnt_attomicAdd_and_ctxId)
{
    //Creating 2 convolution params with different ctxId and attomicAdd value. Insert both to MME descriptor's cache.
    //Test verify the cache include only 1 entry
    std::list<MmeActivation> activations;

    MmeCommon::MmeLayerParams params = setDefaultConvParams();
    DES_CACHE.generateDescriptorsCache(params, activations);

    MmeCommon::MmeLayerParams params2 = setDefaultConvParams();
    params2.tracing.ctxId = 7;
    params2.controls.atomicAdd = true;
    DES_CACHE.generateDescriptorsCache(params2, activations);

    uint32_t currCacheSize =  DES_CACHE.getCacheSize();

    LOG_INFO(GC,"Current cache size {}", currCacheSize);

    ASSERT_EQ(currCacheSize,1) << "Test Failed! expected num of elements = 1" << "and got {}" << currCacheSize;
}

TEST_F_GC(SynGaudiDesCacheTest, max_size_save_lru)
{
    //Creating (max cache size +2) convolution params with different values (paddingValues). Insert all to MME descriptor's cache.
    //Test verify the cache is full (max size) and consist the least recently used elements.
    std::list<MmeActivation> activations;

    MmeCommon::MmeLayerParams params = setDefaultConvParams();
    for (unsigned i=0; i< cacheSize+2; i++)
    {
        params.conv.paddingValue = i;
        DES_CACHE.generateDescriptorsCache(params, activations);
    }

    params.conv.paddingValue = 1;
    DES_CACHE.generateDescriptorsCache(params, activations);

    uint32_t currCacheSize =  DES_CACHE.getCacheSize();

    LOG_INFO(GC,"Current cache size {}", currCacheSize);

    ASSERT_EQ(currCacheSize,cacheSize) << "Test Failed! expected num of elements = 3" << "and got {}" << currCacheSize;

    descriptorCacheIt cachedDesIter;

    ASSERT_EQ(DES_CACHE.isElementInDesCache(params, cachedDesIter), true) << "Test Failed! element 1 not in cache";
    params.conv.paddingValue = 3;
    ASSERT_EQ(DES_CACHE.isElementInDesCache(params, cachedDesIter), true) << "Test Failed! element 3 not in cache";
    params.conv.paddingValue = 4;
    ASSERT_EQ(DES_CACHE.isElementInDesCache(params, cachedDesIter), true) << "Test Failed! element 4 not in cache";
}
