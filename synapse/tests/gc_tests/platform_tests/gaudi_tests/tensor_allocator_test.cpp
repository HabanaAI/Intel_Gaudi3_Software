#include "gc_gaudi_test_infra.h"
#include "habana_global_conf.h"
#include "synapse_api.h"

class SynGaudiTensorAllocationTest : public SynGaudiTestInfra
{
    std::string m_fusingVal;
public:
    void SetUpTest() override
    {
        SynGaudiTestInfra::SetUpTest();
        m_fusingVal = GCFG_RUN_TPC_FUSER.getValueStr();

        // Prevent fusing
        GCFG_RUN_TPC_FUSER.setValue(false);
    }

    void TearDownTest() override
    {
        // Restore global config
        GCFG_RUN_TPC_FUSER.setFromString(m_fusingVal);

        for (auto section: m_sections)
        {
            ASSERT_EQ(synSuccess, synSectionDestroy(section));
        }
        m_sections.clear();

        SynGaudiTestInfra::TearDownTest();
    }

    synTensor createTrainingTensor(unsigned               dims,
                                   synDataType            data_type,
                                   const unsigned*        tensor_size,
                                   bool                   is_presist,
                                   const char*            name,
                                   const synGraphHandle   graph_handle);

    void testDramReuse(bool persistent_output);

    private:
        std::vector<synSectionHandle>       m_sections;

};

synTensor SynGaudiTensorAllocationTest::createTrainingTensor(unsigned               dims,
                                                             synDataType            data_type,
                                                             const unsigned*        tensor_size,
                                                             bool                   is_presist,
                                                             const char*            name,
                                                             const synGraphHandle   graph_handle)
{
    synStatus             status;
    synTensorDescriptor desc {};

    // input
    desc.m_dataType     = data_type;
    desc.m_dims         = dims;
    desc.m_name         = name;
    memset(desc.m_strides, 0, sizeof(desc.m_strides));

    for (unsigned i = 0; i < dims; ++i)
    {
        desc.m_sizes[i] = tensor_size[dims - 1 - i];
    }

    synSectionHandle pSectionHandle = nullptr;
    if (is_presist)
    {
        synSectionCreate(&pSectionHandle, 0, graph_handle);
        m_sections.push_back(pSectionHandle);
    }
    synTensor tensor;
    status = synTensorCreate(&tensor, &desc, pSectionHandle, 0);
    assert(status == synSuccess && "Create tensor failed!");

    UNUSED(status);

    return tensor;
}

void SynGaudiTensorAllocationTest::testDramReuse(bool persistent_output)
{
    // Tensors properties
    const unsigned dims     = 1U;
    const unsigned nofNodes = 31; // nof tensors would be nofNodes + 1

    uint64_t freeDramSpace = 0;
    uint64_t total = 0;
    ASSERT_EQ(synSuccess, synDeviceGetMemoryInfo(_getDeviceId(), &freeDramSpace, &total));

    unsigned sizes[dims];
    sizes[0] = freeDramSpace / (10 * sizeof(float)); // each tensor grabs 1/10 of available DRAM

    synGraphHandle graphHandle = nullptr;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, synDeviceGaudi));

    std::list<synTensor> tensors;

    // Create persistent input tensor
    tensors.push_back(createTrainingTensor(dims, syn_type_float, sizes, true, "input", graphHandle));

    unsigned idx;
    for (idx = 0; idx <  nofNodes - 1; idx++)
    {
        // Create node + workspace tensor
        std::stringstream tensorName;
        tensorName << "ws_tensor_" << idx;
        synTensor outputTensor = createTrainingTensor(dims, syn_type_float, sizes, false, tensorName.str().c_str(), graphHandle);

        std::stringstream nodeName;
        nodeName << "relu_" << idx;
        ASSERT_EQ(synSuccess, synNodeCreate(graphHandle, &tensors.back(), &outputTensor, 1, 1, nullptr, 0,
                                            "relu_fwd_f32", nodeName.str().c_str(), nullptr, nullptr));

        tensors.push_back(outputTensor);
    }

    // Create persistent output tensor
    synTensor outputTensor = createTrainingTensor(dims, syn_type_float, sizes, persistent_output, "output", graphHandle);

    std::stringstream nodeName;
    nodeName << "relu_" << idx;
    ASSERT_EQ(synSuccess, synNodeCreate(graphHandle, &tensors.back(), &outputTensor, 1, 1, nullptr, 0,
                                        "relu_fwd_f32", nodeName.str().c_str(), nullptr, nullptr));

    tensors.push_back(outputTensor);

    synRecipeHandle recipeHandle;
    ASSERT_EQ(synSuccess, synGraphCompile(&recipeHandle, graphHandle, GetTestFileName().c_str(), nullptr));

    uint64_t workspaceSize;
    ASSERT_EQ(synSuccess, synWorkspaceGetSize(&workspaceSize, recipeHandle));
    LOG_DEBUG(SYN_TEST, "Workspace size: {}", workspaceSize);

    for (synTensor tensor : tensors)
    {
        ASSERT_EQ(synSuccess, synTensorDestroy(tensor));
    }
    ASSERT_EQ(synSuccess, synGraphDestroy(graphHandle));
    ASSERT_EQ(synSuccess, synRecipeDestroy(recipeHandle));

    uint64_t persistentSize = 2 * multiplyElements(sizes, sizes + dims) * sizeof(float);
    LOG_DEBUG(SYN_TEST, "Persistent tensors size: {}", persistentSize);

    uint64_t totalHbmSize = persistentSize + workspaceSize;
    LOG_DEBUG(SYN_TEST, "Total size to allocate: {}", totalHbmSize);

    uint64_t allocatedAddress;
    ASSERT_EQ(synSuccess, synDeviceMalloc(_getDeviceId(), totalHbmSize, 0, 0, &allocatedAddress));
    ASSERT_EQ(synSuccess, synDeviceFree(_getDeviceId(), allocatedAddress, 0));

}

TEST_F_GC(SynGaudiTensorAllocationTest, dram_reuse_L2, {synDeviceGaudi})
{
    testDramReuse(true);
}

TEST_F_GC(SynGaudiTensorAllocationTest, dram_reuse_workspace_output_L2, {synDeviceGaudi})
{
    testDramReuse(false);
}
