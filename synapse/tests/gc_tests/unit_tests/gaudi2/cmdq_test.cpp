#include <string>
#include <gtest/gtest.h>
#include "gaudi2_graph.h"
#include "node.h"
#include "platform/gaudi2/graph_compiler/queue_command.h"
#include "platform/gaudi2/graph_compiler/command_queue.h"
#include "platform/gaudi2/graph_compiler/queue_command.h"
#include "gaudi2/gaudi2_packets.h"
#include "graph_optimizer_test.h"
#include "compilation_hal_reader.h"

using namespace gaudi2;

class Gaudi2BaseRegsCacheTest : public GraphOptimizerTest {};

class Gaudi2DmaDescQueueForTest : public gaudi2::DmaDescQueue
{
public:
    Gaudi2DmaDescQueueForTest() : gaudi2::DmaDescQueue(0, 0, 0, false)
    {
        memset(&desc, 0, sizeof(desc));

        setAllPropertiesToData();

        // Fill up the shadow with history of all zeros
        getDescShadow().updateLoadedSegment(0, sizeof(desc)/sizeof(uint32_t), pDesc);
    }

    void setAllPropertiesToData()
    {
        // Fill up the shadow with Data property for the entire descriptor
        DescriptorShadow::AllRegistersProperties writeAll =
                std::make_shared<std::vector<DescriptorShadow::RegisterProperties>>(
                        sizeof(gaudi2::DmaDesc), DescriptorShadow::RegisterProperties::createFromHandling(
                                DescriptorShadow::RegisterDataHandling::Data));
        getDescShadow().setAllRegProperties(writeAll);
    }

    void optimizePatchpoints(BasicFieldsContainerInfo& bfci, const std::vector<uint64_t>& cache)
    {
        gaudi2::DmaDescQueue::optimizePatchpoints(nullptr, desc, &bfci, cache);
    }

    DescriptorShadow& shadow() { return getDescShadow(); }

    gaudi2::DmaDesc desc;
    uint32_t*       pDesc = (uint32_t*)&desc;

};

TEST_F(Gaudi2BaseRegsCacheTest, verify_base_regs_cache_usage)
{
    Gaudi2Graph g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    BasicFieldsContainerInfo   bfci;
    Gaudi2DmaDescQueueForTest  queue;

    //-----------------------------------------------------------------------------------
    // Patchpoint:  #1
    // memoryID:    10
    // Parts:       Low + High
    // Positions:   2, 4
    // Cache:       Will contain sectionID 10 at index 0
    // Results:     Expecting to drop patchpoints and have 2 WriteReg64 (wreg64_long) in the queue for low and high
    bfci.addAddressEngineFieldInfo(nullptr,
                                   std::string("dontcare"),
                                   10,
                                   (uint64_t)(queue.pDesc+2),
                                   (uint64_t)(queue.pDesc+4),
                                   (uint64_t)(queue.pDesc),
                                   FIELD_MEMORY_TYPE_DRAM);
    // Set the values in the descriptor to something else than 0 in order to have a history change
    queue.pDesc[2] = 1;

    //-----------------------------------------------------------------------------------
    // Patchpoint:  #2
    // memoryID:    11
    // Parts:       Full
    // Positions:   6, 7
    // Cache:       Will contain sectionID 11 at index 1
    // Results:     Expecting to drop patchpoints and have 1 WriteReg64 (wreg64_short) in the queue (full)
    bfci.addAddressEngineFieldInfo(nullptr,
                                   std::string("dontcare"),
                                   11,
                                   0x1234,        // address
                                   (uint32_t)6,   // offset of address field in the descriptor
                                   FIELD_MEMORY_TYPE_DRAM);
    // Set the values in the descriptor to something else than 0 in order to have a history change
    queue.pDesc[6] = 0x1234;

    //-----------------------------------------------------------------------------------
    // Patchpoint:  #3
    // memoryID:    12
    // Parts:       Full
    // Positions:   10, 11
    // Cache:       Will contain sectionID 12 at index 2
    // offset:      Larger than 2^32
    // Results:     Expecting to drop patchpoints and have 1 WriteReg64 (wreg64_long) in the queue (full)
    bfci.addAddressEngineFieldInfo(nullptr,
                                   std::string("dontcare"),
                                   12,
                                   (uint64_t)0x1FFFFFFFF,    // address, larger than 2^32
                                   (uint32_t)10,   // offset of address field in the descriptor
                                   FIELD_MEMORY_TYPE_DRAM);
    // Set the values in the descriptor to something else than 0 in order to have a history change
    *((uint64_t*)(queue.pDesc + 10)) = 0x1FFFFFFFF;

    //-----------------------------------------------------------------------------------
    // Patchpoint:  #4
    // memoryID:    13
    // Parts:       Low + High
    // Positions:   13, 15
    // Cache:       Will NOT contain sectionID 13
    // Results:     Expecting keep the two legacy patchpoints
    bfci.addAddressEngineFieldInfo(nullptr,
                                   std::string("dontcare"),
                                   13,
                                   (uint64_t)(queue.pDesc+13),
                                   (uint64_t)(queue.pDesc+15),
                                   (uint64_t)(queue.pDesc),
                                   FIELD_MEMORY_TYPE_DRAM);
    // Set the values in the shadow at positions 13 to something else that 0 to have history change
    queue.shadow().updateLoadedReg(13, 1);
    //-----------------------------------------------------------------------------------

    std::vector<uint64_t> cache;
    cache.push_back(10);
    cache.push_back(11);
    cache.push_back(12);

    // So we set up all the framing for the test, let the job be done
    queue.optimizePatchpoints(bfci, cache);

    // Verify all the expected results stated above
    const AddressFieldInfoSet& afis = bfci.retrieveAddressFieldInfoSet();
    ASSERT_EQ(afis.size(), 2); // patchpoint #4 which is actually low PP and high PP

    for (auto afip : afis) // afip = address-field-info-pair :-)
    {
        ASSERT_TRUE(afip.first == 13 || afip.first == 15); // don't assume the order of elements in the set
    }

    ASSERT_EQ(queue.shadow().propertiesAt(2),  DescriptorShadow::RegisterProperties::getOptOutPatching());
    ASSERT_EQ(queue.shadow().propertiesAt(4),  DescriptorShadow::RegisterProperties::getOptOutPatching());
    ASSERT_EQ(queue.shadow().propertiesAt(6),  DescriptorShadow::RegisterProperties::getOptOutPatching());
    ASSERT_EQ(queue.shadow().propertiesAt(7),  DescriptorShadow::RegisterProperties::getOptOutPatching());
    ASSERT_EQ(queue.shadow().propertiesAt(10), DescriptorShadow::RegisterProperties::getOptOutPatching());
    ASSERT_EQ(queue.shadow().propertiesAt(11), DescriptorShadow::RegisterProperties::getOptOutPatching());
    ASSERT_EQ(queue.shadow().propertiesAt(13), DescriptorShadow::RegisterProperties::getPatching());
    ASSERT_EQ(queue.shadow().propertiesAt(15), DescriptorShadow::RegisterProperties::getPatching());

    const auto& cmds = queue.getCommands(false); // get the execute part of the queue

    // As explained above we expect to find in the queue 2 WriteReg64 (wreg64_long), 1 WriteReg64 (wreg64_short)
    // and finally 1 WriteReg64 (wreg64_long), so total of 4 commands

    // verify first command
    ASSERT_EQ(cmds.size(), 4);
    gaudi2::WriteReg64* cmd = dynamic_cast<WriteReg64*>(cmds[0].get());
    ASSERT_NE(cmd, nullptr);
    uint64_t cmdVal = cmd->getBinForTesting();
    packet_wreg64_long* packetL = (packet_wreg64_long*)(&cmdVal);
    ASSERT_EQ(packetL->opcode, PACKET_WREG_64_LONG);
    ASSERT_EQ(packetL->dw_enable, 0x01); // only low
    ASSERT_EQ(packetL->base, 0);

    // verify second command
    cmd = dynamic_cast<WriteReg64*>(cmds[1].get());
    ASSERT_NE(cmd, nullptr);
    cmdVal = cmd->getBinForTesting();
    packetL = (packet_wreg64_long*)(&cmdVal);
    ASSERT_EQ(packetL->opcode, PACKET_WREG_64_LONG);
    ASSERT_EQ(packetL->dw_enable, 0x02); // only high
    ASSERT_EQ(packetL->base, 0);

    // verify third command
    cmd = dynamic_cast<WriteReg64*>(cmds[2].get());
    ASSERT_NE(cmd, nullptr);
    cmdVal = cmd->getBinForTesting();
    packet_wreg64_short* packetS = (packet_wreg64_short*)(&cmdVal);
    ASSERT_EQ(packetS->opcode, PACKET_WREG_64_SHORT);
    ASSERT_EQ(packetS->base, 1);

    // verify fourth command
    cmd = dynamic_cast<WriteReg64*>(cmds[3].get());
    ASSERT_NE(cmd, nullptr);
    cmdVal = cmd->getBinForTesting();
    packetL = (packet_wreg64_long*)(&cmdVal);
    ASSERT_EQ(packetL->opcode, PACKET_WREG_64_LONG);
    ASSERT_EQ(packetL->dw_enable, 0x03); // both low and high
    ASSERT_EQ(packetL->base, 2);

    queue.Print(); // just to be able to observe the log file
}
