#pragma once
#include "gc_autogen_test.h"
#include "synapse_api_types.h"

#include <numeric>

template<typename C>
inline size_t prod(const C& c)
{
    return c.empty() ? 0 : std::accumulate(begin(c), end(c), (size_t)1, std::multiplies<size_t>());
}

class SynTrainingResNetTest : public SynGaudiAutoGenTest
{
public:
    SynTrainingResNetTest();
    void SetUpTest() override;

    static void SetUpTestSuite() { ReleaseDevice(); }

protected:
    static const uint64_t c_max_tensor_size_to_dump = 8192;

    synTensor createTensor(unsigned        dims,
                           synDataType     data_type,
                           const unsigned* tensor_size,
                           bool            is_presist,
                           const char*     name,
                           synGraphHandle  graphHandle = nullptr,
                           uint64_t        deviceAddr  = -1) override;
    void
    executeTraining(const LaunchInfo&     launchInfo,
                    const TensorInfoList& inputs,
                    const TensorInfoList& outputs,
                    bool skipValidation = false) override;  // TODO SW-59132 - enable validation for fp8 resnet tests

    void executeTrainingRuntime(const LaunchInfo&     launchInfo,
                                const TensorInfoList& tensors,
                                synEventHandle*       eventHandle              = nullptr,
                                size_t                numOfEvents              = 0,
                                size_t                externalTensorsIndices[] = nullptr);

    class TensorInfo
    {
    public:
        TensorInfo(unsigned               dims,
                   synDataType            dataType,
                   uint64_t               tensorSize,
                   bool                   isPresist,
                   std::string            tensorName,
                   synTensor              originalTensor,
                   SynTrainingResNetTest* pTest,
                   bool                   isIntermediate = false);

        bool initFromFile(unsigned int deviceId, std::string basePath);
        bool validateResult(unsigned int deviceId, std::string basePath) const;
        bool allocateAndPrepareReference(unsigned int deviceId, std::string basePath, synStreamHandle streamHandle);
        void uploadAndCheck(unsigned int deviceId, synStreamHandle streamHandle);
        bool cleanup();
        void setIsInput(bool isInput);
        bool isPersist() const { return m_isPresist; }
        bool isInput() const { return m_isInput; }
        void print(unsigned int deviceId, FILE* fd) const;

    private:
        unsigned m_dims;
        synDataType m_dataType;
        uint64_t m_tensorSize;
        bool m_isPresist;
        bool m_isInput;
        std::string m_tensorName;
        synTensor m_pOriginalTensor;
        SynTrainingResNetTest* m_pTest;
        void*               m_ref_arr = nullptr;
        void*               m_data    = nullptr;
        bool                   m_isIntermediate;
    };

    void waitUploadCheck(synEventHandle* eventHandle, TensorInfo* tensorInfo);
    virtual void cleanup();

    TensorInfoList                    m_graphIntermediates;
    std::map<std::string, TensorInfo> m_tensorInfoMap;
    std::vector<std::string>          m_tensorOrder;
    std::string m_pathPrefix;
};

class SynGaudi2ResNetTestEager : public SynTrainingResNetTest
{
public:
    SynGaudi2ResNetTestEager()
    {
        m_testConfig.m_compilationMode      = COMP_EAGER_MODE_TEST;
        setSupportedDevices({synDeviceGaudi2});
        setTestPackage(TEST_PACKAGE_EAGER);
    }
};

class SynGaudiResNetTestArc
: public SynTrainingResNetTest
, public testing::WithParamInterface<unsigned>
{
};