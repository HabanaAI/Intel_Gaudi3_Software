#include <memory>
#include <string>
#include "infra/scoped_configuration_change.h"
#include "recipe_test_base.h"
#include "habana_global_conf.h"
#include "platform/gaudi/graph_compiler/command_queue.h"
#include "platform/gaudi/graph_compiler/queue_command.h"
#include "tensor.h"
#include "habana_graph.h"

void RecipeTestBase::SetUp()
{
    GraphOptimizerTest::SetUp();
    m_pScopedChangeEnableExperimentalFlags = new ScopedConfigurationChange("ENABLE_EXPERIMENTAL_FLAGS", "true");
    m_pScopedAllowDuplicateKernel          = new ScopedConfigurationChange("ALLOW_DUPLICATE_KERNELS", "1");

    // Create single persistent tensor
    TensorPtr           persistTensor = std::make_shared<Tensor>(syn_type_int8);
    synMemoryDescriptor memDescPersist(true);
    persistTensor->setMemoryDescriptor(memDescPersist);
    persistTensor->setName("persist_tensor_with_FULL_afci_at_index_0");
    m_persistTensors.insert(persistTensor);
    m_sectionID = 3;
    m_fakeAddress = 0x1000;
}

void RecipeTestBase::TearDown()
{
    for (auto ptr : m_queuePtrs)
    {
        ptr.reset();
    }
    delete m_pScopedChangeBlobOptimization;
    delete m_pScopedChangeEnableExperimentalFlags;
    delete m_pScopedAllowDuplicateKernel;
    ShapeFuncRegistry::instance().destroy();
}

void RecipeTestBase::makeQueues(unsigned              numQueues,
                                HabanaGraph*          g,
                                bool                  identicalCmds,
                                bool                  identicalQueues,
                                std::vector<unsigned> activateCmdIndexes,
                                bool                  useMonArmCmd)
{
    unsigned valueQ = 0;

    // Create queues and fill each one of them with commands
    for (unsigned i = 0; i < numQueues; i++)
    {
        CommandQueue*  queue = getTpcQueue(i, i, 0, false);
        unsigned       valueCmd = valueQ;
        unsigned       committerCounter = 0;

        for (unsigned j = 0; j < m_numCmdsInQueue; j++)
        {
            QueueCommand *cmd = nullptr;

            // First command is monitor arm unless we were asked to avoid it
            if (j == 0 && useMonArmCmd)
            {
                cmd = getMonArm();
            }
            else
            {
                cmd = getWriteReg(valueCmd);
                valueCmd += identicalCmds ? 0 : 1;
            }

            // every 5th command is blob committer
            // every 8th command has patch point
            // every third committer is also execute
            if ((j + 1) % m_blobCommitterInterval == 0)
            {
                cmd->setAsBlobCommitter(UNDEFINED_NODE_EXE_INDEX);
                committerCounter++;
                if (committerCounter % 3 == 0)
                {
                    cmd->setAsExe();
                }
            }
            if ((j+1) % m_patchPointInterval == 0)
            {
                BasicFieldsContainerInfo  afci;

                afci.addAddressEngineFieldInfo(nullptr,
                                        (*m_persistTensors.begin())->getName(),
                                        3,                     // memory id / section index
                                        0,                     // virtual address
                                        (uint32_t) 0,          // field index offset
                                        FIELD_MEMORY_TYPE_DRAM);

                cmd->SetContainerInfo(afci);
            }

            bool isSetup = std::count(activateCmdIndexes.begin(), activateCmdIndexes.end(), j);
            queue->PushBack(QueueCommandPtr{cmd}, isSetup);
        }

        // mark last command in each queue (activate/execute) as a committer
        if (queue->Size(true))
        {
            auto &lastCommand = queue->getCommands(true).back();
            lastCommand->setAsBlobCommitter(UNDEFINED_NODE_EXE_INDEX);
        }

        if (queue->Size(false))
        {
            auto &lastCommand = queue->getCommands(false).back();
            lastCommand->setAsBlobCommitter(UNDEFINED_NODE_EXE_INDEX);
        }

        CommandQueuePtr qPtr(queue);
        m_queuePtrs.push_back(qPtr);
        valueQ += identicalQueues ? 0 : 100;
    }

    std::map<uint32_t, CommandQueuePtr>& commandQueues = g->getCodeGenerator()->getCommandQueueByIdForTesting();
    commandQueues.clear();

    for (int i = 0; i < m_queuePtrs.size(); i++)
    {
        commandQueues[i] = m_queuePtrs[i];
    }
}

void RecipeTestBase::setDummyAddressAndSection(TensorPtr tensor)
{
    tensor->setDramOffset(m_fakeAddress);
    m_fakeAddress += 0x2000;

    synMemoryDescriptor memDescPersist(true);
    tensor->setMemoryDescriptor(memDescPersist);
    tensor->setMemorySectionID(m_sectionID++);
}
