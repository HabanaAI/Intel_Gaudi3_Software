#include "cache_types.h"
#include "mcid_converter.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "platform/gaudi3/graph_compiler/tpc_descriptor_generator.h"
#include "platform/gaudi3/graph_compiler/gaudi3_types.h"
#include "graph_optimizer_test.h"
#include "gaudi3_code_generator.h"
#include "node_factory.h"
#include <string>

using std::tuple;

class Gaudi3McidTpcConfTest
: public GraphOptimizerTest
, public testing::WithParamInterface<tuple<std::vector<CacheMaintenanceAction>,
                                           std::vector<LogicalMcid>,
                                           gaudi3::TpcDescriptorGenerator::McidTpcUsage,  // Expected McidTpcUsage
                                           std::vector<unsigned>,                         // Expected srfs mcid + mask
                                           unsigned,                                      // Expected fast_cfg mcid
                                           unsigned,                                      // Expected fast_cfg mask
                                           unsigned,                                      // numInputs
                                           unsigned,                                      // numOutputs
                                           unsigned,                                      // mcid config mask
                                           unsigned>>                                     // numSRFs

{
};

static void getPhysicalMcid(McidConverter& mcidConverter, CacheMetaData cacheMD, uint64_t &physicalMcid)
{
    physicalMcid = 0;

    switch (cacheMD.cmAction)
    {
        case NOP:
            break;
        case DEGRADE:
            mcidConverter.convertDegrade(cacheMD.mcid, (uint16_t&)physicalMcid);
            break;
        case DISCARD:
            unsigned dummyRolloverIndication;
            mcidConverter.convertDiscard(cacheMD.mcid, (uint16_t&)physicalMcid, dummyRolloverIndication);
            break;
        default:
            HB_ASSERT(false, "Cache Maintenance Action not supportted");
    }
}

TEST_P(Gaudi3McidTpcConfTest, desc_conf_tests)
{
    // Get tests params
    std::vector<CacheMaintenanceAction> cmActions = std::get<0>(GetParam());
    std::vector<LogicalMcid> mcids                = std::get<1>(GetParam());
    std::vector<unsigned> expectedSRFs            = std::get<3>(GetParam());
    unsigned expextedFastCfgMcid                  = std::get<4>(GetParam());
    unsigned expextedFastCfgMask                  = std::get<5>(GetParam());
    unsigned numInputs                            = std::get<6>(GetParam());
    unsigned numOutputs                           = std::get<7>(GetParam());
    unsigned configMask                           = std::get<8>(GetParam());
    unsigned numSRFs                              = std::get<9>(GetParam());

    gaudi3::TpcDescriptorGenerator::McidTpcUsage expectedTpcUsage = std::get<2>(GetParam());

    setGlobalConfForTest(GCFG_TPC_MCID_NUM_SRF, std::to_string(numSRFs));
    setGlobalConfForTest(GCFG_TPC_MCID_CONFIG_MASK, std::to_string(configMask));

    /************************************************************************************
     *                   Create a graph with a single TPC NOP node                      *
     ************************************************************************************/

    const unsigned dim  = 1;
    const TSize    size = 1;
    Gaudi3Graph    g;

    // Create TPC node with input and output tensors according to params
    TensorVector inputTensors;
    TensorVector outputTensors;

    unsigned i;
    for (i = 0; i < numInputs; i++)
    {
        inputTensors.push_back(TensorPtr(new Tensor(dim, &size, syn_type_single)));
    }
    for (; i < numInputs + numOutputs; i++)
    {
        outputTensors.push_back(TensorPtr(new Tensor(dim, &size, syn_type_single)));
    }
    NodePtr node = NodeFactory::createNode(inputTensors, outputTensors, nullptr, NOP_KERNEL_NAME, "tpcNode");

    // Create Node ROIs with cache meta data
    NodeROI* tpcNodeRoi = new NodeROI();
    std::list<NodeROI>* nodeRois = new std::list<NodeROI>();
    McidConverter mcidConverter = g.getCodeGenerator()->getMcidConverter();
    std::vector<gaudi3::TpcDescriptorGenerator::McidInfo> mcidTpcInfo;
    gaudi3::TpcDescriptorGenerator::McidTpcUsage mcidTpcUsage;

    for (i = 0; i < numInputs + numOutputs; i++)
    {
        CacheMetaData md;
        md.cmAction = cmActions[i];
        md.mcid = mcids[i];
        if (i < numInputs)
        {
            tpcNodeRoi->inputsCacheMetaData.push_back(md);
        }
        else
        {
            tpcNodeRoi->outputsCacheMetaData.push_back(md);
        }

        // Fill cache data for later use
        gaudi3::TpcDescriptorGenerator::McidInfo info;
        info.tensorIndex = i;
        info.cacheMD = md;
        getPhysicalMcid(mcidConverter, md, info.physicalMcid);
        mcidTpcInfo.push_back(info);
    }
    nodeRois->push_back(*tpcNodeRoi);

    /************************************************************************************
     *       Call MCID TPC descriptor generator function and get McidTPcUsage           *
     ************************************************************************************/

    TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
    ValidityMask<gaudi3::TpcDesc> descMask {false};
    gaudi3::TpcDesc desc;
    memset(&desc, 0, sizeof(desc));

    gaudi3::TpcDescriptorGenerator::fillMcidTpcConfiguration(*tpcNode, desc, descMask, mcidTpcInfo, mcidTpcUsage);

    /************************************************************************************
     *                Validate McidTPcUsage and MCIDs configuration                     *
     ************************************************************************************/

    // Fast CFG
    if (GCFG_TPC_MCID_CONFIG_MASK.value() & 0x1)
    {
        ASSERT_EQ(expectedTpcUsage.fastCfg, mcidTpcUsage.fastCfg);
        ASSERT_EQ(expextedFastCfgMcid, desc.m_desc.mcid_fast_cfg.mcid);
        ASSERT_EQ(expextedFastCfgMask, desc.m_desc.mcid_fast_cfg.mask);
    }

    // SRFs
    if (GCFG_TPC_MCID_CONFIG_MASK.value() & 0x2)
    {
        unsigned srfBaseId = Gaudi3HalReader::instance()->getNumSRFs() - Gaudi3HalReader::instance()->getNumFastConfigMcidSRFs();
        for (unsigned i = 0; i < numSRFs; i++)
        {
            ASSERT_EQ(expectedTpcUsage.srf[i], mcidTpcUsage.srf[i]);
            ASSERT_EQ(expectedSRFs[i],desc.m_desc.srf[srfBaseId + i].v);
        }
    }

    // AXI_CFG (Private tensor conf) - TPC Usage
    ASSERT_EQ(expectedTpcUsage.tensorPrivateCfg.size(), mcidTpcUsage.tensorPrivateCfg.size());
    for (auto itExpected = expectedTpcUsage.tensorPrivateCfg.begin(); itExpected != expectedTpcUsage.tensorPrivateCfg.end(); itExpected++)
    {
        auto itResult = mcidTpcUsage.tensorPrivateCfg.find(itExpected->first);
        ASSERT_NE(itResult, mcidTpcUsage.tensorPrivateCfg.end());
        ASSERT_EQ(itResult->second, itExpected->second);
    }

    // AXI_CFG (Private tensor conf) - MCIDs
    for (unsigned i = 0; i < numInputs + numOutputs; i++)
    {
        TensorDescGaudi3* tpcDesc = desc.m_tensors + mcidTpcInfo[i].tensorIndex;
        if (mcidTpcUsage.tensorPrivateCfg.find(mcidTpcInfo[i].tensorIndex) != mcidTpcUsage.tensorPrivateCfg.end())
        {
            CacheMaintenanceAction cmAction = mcidTpcInfo[i].cacheMD.cmAction;
            uint64_t physicalMcid = cmAction == DISCARD ? mcidTpcInfo[i].physicalMcid - UINT32_MAX : mcidTpcInfo[i].physicalMcid;
            ASSERT_EQ(physicalMcid, tpcDesc->shared.hbw_axi_cfg.mcid);
        }
    }

    delete tpcNodeRoi;
    delete nodeRois;
}

INSTANTIATE_TEST_SUITE_P(,
                         Gaudi3McidTpcConfTest,
                         ::testing::Values(
                            // ---------------- Test case - All 16 tensors are configured ------------------- //
                            std::make_tuple(
                                std::vector<CacheMaintenanceAction>          {DISCARD,DEGRADE,NOP,DEGRADE,
                                                                             DEGRADE,DISCARD,NOP,DEGRADE,
                                                                             DISCARD,DISCARD,DISCARD,NOP,
                                                                             NOP,DEGRADE,DEGRADE,NOP},
                                std::vector<LogicalMcid>                     {4,2,0,2,9,3,0,2,6,1,1,0,0,8,2,0},
                                gaudi3::TpcDescriptorGenerator::McidTpcUsage {DEGRADE, {},
                                                                             {{0,DISCARD},{2,NOP},
                                                                             {4,DEGRADE},{5,DISCARD},{6,NOP},
                                                                             {8,DISCARD},{9,DISCARD},{10,DISCARD},{11,NOP},
                                                                             {12,NOP},{13,DEGRADE},{15,NOP}}},
                                std::vector<unsigned>                        {0,0,0,0}, //Expected srfs mcid + mask
                                                                             2,         //Expected fast_cfg mcid
                                                                             0x408a,    //Expected fast_cfg mask
                                                                             15,        // numInputs
                                                                             1,         // numOutputs
                                                                             1,         // mcid config mask
                                                                             0),        // numSRFs
                            // ---------------- Test case - MCID = 0 is most frequent ------------------- //
                            std::make_tuple(
                                std::vector<CacheMaintenanceAction>          {DEGRADE,NOP,NOP,NOP,
                                                                             NOP,NOP,NOP,DISCARD},
                                std::vector<LogicalMcid>                     {2,0,0,0,0,0,0,4},
                                gaudi3::TpcDescriptorGenerator::McidTpcUsage {DISCARD, {},
                                                                             {{0,DEGRADE},{1,NOP},{2,NOP},{3,NOP},
                                                                             {4,NOP},{5,NOP},{6,NOP}}},
                                std::vector<unsigned>                        {0,0,0,0}, //Expected srfs mcid + mask
                                                                             4,         //Expected fast_cfg mcid
                                                                             0x80,      //Expected fast_cfg mask
                                                                             5,         // numInputs
                                                                             3,         // numOutputs
                                                                             1,         // mcid config mask
                                                                             0),        // numSRFs
                            // ---------------- Test case - MCID = 0 only ------------------- //
                            std::make_tuple(
                                std::vector<CacheMaintenanceAction>          {NOP,NOP,NOP,NOP},
                                std::vector<LogicalMcid>                     {0,0,0,0},
                                gaudi3::TpcDescriptorGenerator::McidTpcUsage {NOP, {},
                                                                             {}},
                                std::vector<unsigned>                        {0,0,0,0}, //Expected srfs mcid + mask
                                                                             0,         //Expected fast_cfg mcid
                                                                             0xf,       //Expected fast_cfg mask
                                                                             3,         // numInputs
                                                                             1,         // numOutputs
                                                                             1,         // mcid config mask
                                                                             0),        // numSRFs

                            // ---------------- Test case - Use SRFs ------------------- //
                            std::make_tuple(
                                std::vector<CacheMaintenanceAction>          {DEGRADE,DEGRADE,NOP,DEGRADE,
                                                                             DEGRADE,DISCARD,NOP,DEGRADE,
                                                                             DISCARD,DISCARD,DISCARD,NOP,
                                                                             NOP,DEGRADE,DEGRADE,NOP},
                                std::vector<LogicalMcid>                     {4,2,0,2,9,3,0,2,6,1,1,0,0,8,2,0},
                                gaudi3::TpcDescriptorGenerator::McidTpcUsage {DEGRADE, {DISCARD,DISCARD,DISCARD,DEGRADE},
                                                                             {{0,DEGRADE},{2,NOP},
                                                                             {6,NOP},
                                                                             {11,NOP},{12,NOP},
                                                                             {13,DEGRADE},{15,NOP}}},
                                std::vector<unsigned>                        {0x6000001,0x1000006,0x200003,0x100009}, //Expected srfs mcid + mask
                                                                             2,         //Expected fast_cfg mcid
                                                                             0x408a,    //Expected fast_cfg mask
                                                                             15,        // numInputs
                                                                             1,         // numOutputs
                                                                             3,         // mcid config mask
                                                                             4),        // numSRFs

                            // ---------------- Test case - Use part of SRFs ------------------- //
                            std::make_tuple(
                                std::vector<CacheMaintenanceAction>          {DEGRADE,DEGRADE,DEGRADE,DEGRADE,
                                                                             DEGRADE,DEGRADE,DEGRADE,DEGRADE},
                                std::vector<LogicalMcid>                     {1,2,1,2,2,3,4,1},
                                gaudi3::TpcDescriptorGenerator::McidTpcUsage {DEGRADE, {DEGRADE,DEGRADE,NOP,NOP},
                                                                             {{5,DEGRADE},{6,DEGRADE}}},
                                std::vector<unsigned>                        {0x1a0002,0x850001,0,0}, //Expected srfs mcid + mask
                                                                             0,         //Expected fast_cfg mcid
                                                                             0,         //Expected fast_cfg mask
                                                                             6,         // numInputs
                                                                             2,         // numOutputs
                                                                             2,         // mcid config mask
                                                                             2),        // numSRFs

                            // ---------------- Test case - No SRF and No FAST_CFG ------------------- //
                            std::make_tuple(
                                std::vector<CacheMaintenanceAction>          {DISCARD,DEGRADE,DEGRADE,DISCARD,
                                                                             DEGRADE,DISCARD,DISCARD,DEGRADE,
                                                                             NOP,DEGRADE,DISCARD,NOP},
                                std::vector<LogicalMcid>                     {7,8,1,2,9,3,1,7,0,7,7,0},
                                gaudi3::TpcDescriptorGenerator::McidTpcUsage {NOP, {},
                                                                             {{0,DISCARD},{1,DEGRADE},{2,DEGRADE},{3,DISCARD},
                                                                             {4,DEGRADE},{5,DISCARD},{6,DISCARD},{7,DEGRADE},
                                                                             {8,NOP},{9,DEGRADE},{10,DISCARD},{11,NOP}}},
                                std::vector<unsigned>                        {0,0,0,0}, //Expected srfs mcid + mask
                                                                             0,         //Expected fast_cfg mcid
                                                                             0,     //Expected fast_cfg mask
                                                                             6,        // numInputs
                                                                             6,         // numOutputs
                                                                             0,         // mcid config mask
                                                                             0)         // numSRFs
                       ));