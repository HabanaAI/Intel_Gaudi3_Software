#pragma once

#include "gc_gaudi_test_infra.h"
#include "infra/gc_test_configuration.h"

/**
 * Use to create two identical graph
 * Compile and run them with different configuration and compare results
 */
class SynTrainingTwoRunCompareTest : public SynTrainingTestInfra
{
public:
    virtual ~SynTrainingTwoRunCompareTest() = default;
    SynTrainingTwoRunCompareTest();
    virtual void SetUpTest() override;
    enum RunIndex
    {
        FIRST_RUN = 0,
        SECOND_RUN = 1
    };

    void addConfigurationToRun(RunIndex runIdx, std::string&& globalConf, std::string&& value);

    void compareRunsResults(const std::vector<unsigned>& tensorIdxs);

    virtual TensorIndices createTensors(unsigned        numTensors,
                                        TensorUsage     usage,
                                        bool            isPersistent    = false,
                                        const char*     name            = nullptr,
                                        MemInitType     initSelect      = MEM_INIT_ALL_ZERO,
                                        const float*    initializer     = nullptr,
                                        unsigned*       sizes           = nullptr,
                                        unsigned        dims            = DEFAULT_SIZES,
                                        synDataType     dataType        = syn_type_single,
                                        unsigned*       strides         = nullptr,
                                        unsigned        graphIndex      = 0,
                                        unsigned        offsetInSection = 0,
                                        const unsigned* sectionIndex    = nullptr,
                                        bool            isConst         = false,
                                        unsigned*       minSizes        = nullptr,
                                        synTensorType   tensorType      = synTensorType::DATA_TENSOR,
                                        synTensor       existingTensor  = nullptr) override;

    virtual void addNodeToGraph(const char*          guid,
                                const TensorIndices& inputTensorIndices,   // Indices of m_inTensors
                                const TensorIndices& outputTensorIndices,  // Indices of m_outTensors
                                void*                userParams    = nullptr,
                                unsigned             paramSize     = 0,
                                const char*          nodeName      = nullptr,
                                unsigned             graphIndex    = 0,
                                synNodeId*           nodeId        = nullptr,
                                const char**         inputLayouts  = nullptr,
                                const char**         outputLayouts = nullptr) override;

    virtual void setNodeDependency(const synNodeId*       pBlockingNodesIdList,
                                   const synNodeId*       pBlockedNodesIdList,
                                   const uint32_t         numberblocking,
                                   const uint32_t         numberblocked,
                                   unsigned               graphIndex = 0) override;

    virtual void compileAndRun() override;

    void setActualSizes(unsigned tensorIndex, const unsigned* tensorSizes);
    void setActualSizes(unsigned tensorIndex, const unsigned* tensorSizes, unsigned graphIndex) = delete;
    void
    setActualSizes(unsigned tensorIndex, const std::vector<unsigned>& tensorSizes, unsigned graphIndex = 0) = delete;
    unsigned createSection(uint64_t size);
    virtual unsigned createConstSection(uint64_t graphIndex) override;
    void setGraphsAttributes(synGraphAttribute* attributes, synGraphAttributeVal* values, uint32_t size);
    void setGraphsInferenceMode();
    void setGraphsInferenceModeAndQuantizationEnabled();
    void setNodeDeterminstic(const synNodeId& nodeId);
    void             setPermutationForTensor(unsigned tensorIndex, const std::vector<uint8_t>& permutation);

private:

    void compileAndRun(RunIndex runIdx);
    // Not supported method for this test
    virtual void addNodeToGraph(const char*  guid,
                                void*        userParams    = nullptr,
                                unsigned     paramSize     = 0,
                                const char*  nodeName      = nullptr,
                                unsigned     graphIndex    = 0,
                                const char** inputLayouts  = nullptr,
                                const char** outputLayouts = nullptr) override;

    using RunConfigurations = std::list<std::pair<std::string, std::string>>;
    std::array<RunConfigurations, 2> m_runsConfig;
    std::map<unsigned, unsigned> m_firstToSecondTensorIdx;
    std::map<unsigned, unsigned>     m_firstToSecondSectionIdx;
    std::map<synNodeId, synNodeId> m_firstToSecondNodeId;
    std::array<std::vector<std::pair<unsigned /* tensor index*/, const unsigned* /*actual sizes*/>>, 2> m_actualSizes;
    std::map<unsigned /* tensor index*/, unsigned /*total size*/> m_actualSizesForValidation;
    bool m_executed = false;
};

class SynGaudiTwoRunCompareTest : public SynTrainingTwoRunCompareTest
{
public:
    SynGaudiTwoRunCompareTest() { ReleaseDevice(); }
};
