#pragma once

#include <list>
#include "types.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "program_data_blob.h"
#include "queue_command.h"
#include "command_queue.h"

class ScopedConfigurationChange;

class RecipeTestBase : public GraphOptimizerTest
{
public:

protected:
    virtual void SetUp();
    virtual void TearDown();
    void         makeQueues(unsigned              numQueues,
                            HabanaGraph*          g,
                            bool                  identicalCmds      = false,
                            bool                  identicalQueues    = false,
                            std::vector<unsigned> activateCmdIndexes = {},
                            bool                  useMonArmCmd       = true);

    virtual QueueCommand* getWriteReg(unsigned valueCmd) const = 0;
    virtual QueueCommand* getMonArm() const = 0;
    virtual CommandQueue* getTpcQueue(unsigned engineId,
                                      unsigned engineIndex,
                                      unsigned stream,
                                      bool sendSyncEvents) const = 0;

    void setDummyAddressAndSection(TensorPtr tensor);

    // test implementations
    void blob_basic(HabanaGraph *g);
    void blob_container_basic(HabanaGraph *g);
    void blob_container_4_different_blobs(HabanaGraph *g);
    void blob_container_blobs_chunks_test(HabanaGraph *g);
    void generator_basic(HabanaGraph *g);
    void const_sections(HabanaGraph* g);
    void generator_node_exe(HabanaGraph *g);
    void generator_continuous_blobs(HabanaGraph *g);
    void generator_single_blob(HabanaGraph* g);
    void generator_two_queues(HabanaGraph *g);
    void generator_activate_execute_jobs(HabanaGraph *g);
    void patch_point_container();
    void section_type_patch_point_container();
    void program_basic();
    void program_serialize();
    void unique_hash_system_base_test(HabanaGraph *g);

    static const unsigned              m_numCmdsInQueue = 20;
    static const unsigned              m_blobCommitterInterval = 5;
    static const unsigned              m_patchPointInterval = 8;
    std::vector<CommandQueuePtr>       m_queuePtrs;
    TensorSet                          m_persistTensors;
    ScopedConfigurationChange*         m_pScopedChangeBlobOptimization = nullptr;
    ScopedConfigurationChange*         m_pScopedChangeEnableExperimentalFlags = nullptr;
    ScopedConfigurationChange*         m_pScopedAllowDuplicateKernel          = nullptr;
    uint32_t                           m_sectionID;
    uint64_t                           m_fakeAddress;
};
