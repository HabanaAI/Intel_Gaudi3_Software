#include "gaudi3_code_generator.h"
#include "gaudi3_eng_arc_command.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "platform/gaudi3/graph_compiler/passes/generate_cache_maintenance_pass.h"
#include "scoped_configuration_change.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "transpose_permutation.h"
#include "transpose_utils.h"
#include "types_exception.h"

#include <gtest/gtest.h>

#include <iostream>

class GenerateCmeTaskPassTest : public GraphOptimizerTest
{
// Graph contains 4 nodes as detailed below. We will populate the ROIs
// inputsCacheMetaData & outputsCacheMetaData structures and run various test scenarios

// Transpode Node: mme_transpose_node_t0 , Logical Queue: 3 exe index: 1
// =====================================================================
// Num of rois     : 1
// roi input size  : 2
// roi output size : 2

// TPC Node: tensor-0_c71_memcpy_88_internal , Logical Queue: 1 exe index: 3
// =========================================================================
// Num of rois      : 3
// roi1 input size  : 1
// roi1 output size : 1
// roi2 input size  : 1
// roi2 output size : 1
// roi3 input size  : 1
// roi3 output size : 1

// MME Node: mme_node , Logical Queue: 0 exe index: 5
// ==================================================
// Num of rois     : 1
// roi input size  : 2
// roi output size : 2

// TPC Node: add , Logical Queue: 1 exe index: 6
// =============================================
// Num of rois     : 1
// roi input size  : 2
// roi output size : 1

private:
    NodePtr transposeNode;
    NodePtr tpcMemcpyNode;
    NodePtr mmeNode;
    NodePtr tpcNode;

public:
    void createGraph(Gaudi3Graph& g)
    {
        uint8_t dim      = 4;
        auto    guid     = NodeFactory::transposeNodeTypeName;
        auto    dataType = syn_type_single;

        // ADD TRANSPOSE NODE
        // outputFcd>inputFcd induces a physical->logical transpose sequence
        TSize inSize[]  = {64, 16, 128, 1};
        TSize outSize[] = {128, 16, 64, 1};

        TensorPtr in  = TensorPtr(new Tensor(dim, inSize, dataType));
        TensorPtr out = TensorPtr(new Tensor(dim, outSize, dataType));

        synMemoryDescriptor memDesc(true);  // persistent

        // set some boguse addresses to the tensors and allocate host memory so we won't assert
        in->setDramOffset(0x1000);
        out->setDramOffset(0x5000);
        in->setMemoryDescriptor(memDesc);
        out->setMemoryDescriptor(memDesc);
        in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
        out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

        // Create transpose node
        using D = TransposePermutationDim;
        synTransposeParamsNDims transposeParams(permutationToParams({D(2), D(1), D(0), D(3)}));

        NodePtr transpose = NodeFactory::createNode({in}, {out}, &transposeParams, guid, "mme_transpose_node");
        GraphEditor::addNode(g, transpose);

        // ADD MME NODE
        auto guid2     = NodeFactory::convolutionNodeTypeName;
        auto dataType2 = syn_type_fp16;

        const TSize sizes_x[] = {256, 256, 1, 1};
        const TSize sizes_w[] = {256, 256, 1, 1};
        const TSize sizes_y[] = {256, 256, 1, 1};

        TensorPtr x = TensorPtr(new Tensor(4U, sizes_x, dataType2));
        TensorPtr w = TensorPtr(new Tensor(4U, sizes_w, dataType2));
        TensorPtr y = TensorPtr(new Tensor(4U, sizes_y, dataType2));

        synMemoryDescriptor memDesc2(true);  // persistent

        // set some boguse addresses to the tensors and allocate host memory so we won't assert
        x->setDramOffset(0x10000);
        w->setDramOffset(0x20000);
        y->setDramOffset(0x30000);
        x->setMemoryDescriptor(memDesc2);
        w->setMemoryDescriptor(memDesc2);
        y->setMemoryDescriptor(memDesc2);
        x->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
        w->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
        y->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);

        synConvolutionParams params {};

        NodePtr conv = NodeFactory::createNode({x, w}, {y}, &params, guid2, "mme_node");
        GraphEditor::addNode(g, conv);

        // ADD TPC NODE
        const unsigned      tensor_dim = 1;
        const TSize         size       = 1;
        TensorPtr           i1         = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr           i2         = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        TensorPtr           o          = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        NodePtr             n          = NodeFactory::createNode({i1, i2}, {o}, nullptr, "add_fwd_f32", "add");
        synMemoryDescriptor memDesc3(true);  // persistent

        // set some boguse addresses to the tensors and allocate host memory so we won't assert
        i1->setDramOffset(0x40000);
        i2->setDramOffset(0x50000);
        o->setDramOffset(0x60000);
        i1->setMemoryDescriptor(memDesc3);
        i2->setMemoryDescriptor(memDesc3);
        o->setMemoryDescriptor(memDesc3);

        ASSERT_TRUE(GraphEditor::addNode(g, n));
    }

    void clear(Gaudi3Graph& g)
    {
        for (const NodePtr& n : g.getExeSortedNodes())
        {
            if (n == nullptr) continue;
            if (n->isLogicalOperation()) continue;

            for (auto& roi: (*g.GetNodeROIs(n)))
            {
                roi.inputsCacheMetaData.clear();
                roi.outputsCacheMetaData.clear();
                roi.cmeTasks.cmCmds.clear();
            }
        }
    }

    void run_test_case1(Gaudi3Graph& g)
    {
        // scenario: Fixed op type (DEGRADE). Random/Incremented MCID for all ROIs
        for (const NodePtr& n : g.getExeSortedNodes())
        {
            if (n == nullptr) continue;

            if (n->isLogicalOperation()) continue;

            LogicalMcid mcid = 6;

            for (auto& roi: (*g.GetNodeROIs(n)))
            {
                CacheMetaData md;

                md.mcid = mcid;
                md.cmAction = DEGRADE;

                for (int i = 0; i < roi.inputRois.size(); i++)
                {
                    md.mcid++;
                    roi.inputsCacheMetaData.push_back(md);
                }

                for (int i = 0; i < roi.outputRois.size(); i++)
                {
                    md.mcid++;
                    roi.outputsCacheMetaData.push_back(md);
                }
            }
        }

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        std::list<NodeROI>* tpcNodeROIs = g.GetNodeROIs(tpcNode);
        NodeROI tpcNodeROI = (*tpcNodeROIs).front();

        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on tpc node";

        // MCID = 7 --> 2 (MCIDs 8 & 9 has same dependency map as MCID 7)
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].mcid, 2);
        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds[0].op, DEGRADE);
        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 4);

        clear(g);
    }

    void run_test_case2(Gaudi3Graph& g)
    {
        // scenario: Fixed MCID/DEGRADE to all ROIs
        for (const NodePtr& n : g.getExeSortedNodes())
        {
            if (n == nullptr) continue;

            if (n->isLogicalOperation()) continue;

            CacheMetaData md;
            md.mcid = 6;
            md.cmAction = DEGRADE;

            for (auto& roi: (*g.GetNodeROIs(n)))
            {
                for (int i = 0; i < roi.inputRois.size(); i++)
                {
                    roi.inputsCacheMetaData.push_back(md);
                }

                for (int i = 0; i < roi.outputRois.size(); i++)
                {
                    roi.outputsCacheMetaData.push_back(md);
                }
            }
        }

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        std::list<NodeROI>* tpcNodeROIs = g.GetNodeROIs(tpcNode);
        NodeROI tpcNodeROI = (*tpcNodeROIs).front();

        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme tasks on tpc node";

        // MCID = 6 --> 1
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].mcid, 1);
        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds[0].op, DEGRADE);
        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 4);

        clear(g);
    }

    void run_test_case3(Gaudi3Graph& g)
    {
        // scenario: Fixed MCID(6) to all ROIs. OP type is alternating
        for (const NodePtr& n : g.getExeSortedNodes())
        {
            if (n == nullptr) continue;

            if (n->isLogicalOperation()) continue;

            CacheMetaData md;
            md.mcid = 6;
            md.cmAction = DEGRADE;

            for (auto& roi: (*g.GetNodeROIs(n)))
            {
                for (int i = 0; i < roi.inputRois.size(); i++)
                {
                    roi.inputsCacheMetaData.push_back(md);
                    md.cmAction = DISCARD;
                }

                md.cmAction = DEGRADE;
                for (int i = 0; i < roi.outputRois.size(); i++)
                {
                    roi.outputsCacheMetaData.push_back(md);
                    md.cmAction = DISCARD;
                }
            }
        }

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        std::list<NodeROI>* tpcNodeROIs = g.GetNodeROIs(tpcNode);
        NodeROI tpcNodeROI = (*tpcNodeROIs).front();

        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds.size(), 2) << "Expecting 2 cme tasks on tpc node";

        // MCID = 6 --> 1 / DEGRADE
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].mcid, 1);
        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds[0].op, DEGRADE);
        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 4);

        // MCID = 6 --> 1 / DISCARD
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[1].mcid, 1);
        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds[1].op, DISCARD);
        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds[1].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 4);

        clear(g);
    }

    void run_test_case4(Gaudi3Graph& g)
    {
        // scenario: MCID 7/DEGRADE on first roi output of transpose node
        //           MCID 7/DEGRADE of first roi input of MME node
        //           MCID 8/DISCARD on second roi output of transpose node
        //           MCID 8/DISCARD on input of 2nd roi of TPCMemcpy node

        // Expecting 2 tasks:
        // On first roi input of MME node
        // On input of 2nd roi of TPCMemcpy node

        std::list<NodeROI>* transposeNodeROIs = g.GetNodeROIs(transposeNode);
        NodeROI& transposeNodeROI = (*transposeNodeROIs).front();

        std::list<NodeROI>* mmeNodeROIs = g.GetNodeROIs(mmeNode);
        NodeROI& mmeNodeROI = (*mmeNodeROIs).front();

        std::list<NodeROI>* tpcMemcpyNodeROIs = g.GetNodeROIs(tpcMemcpyNode);

        auto it = (*tpcMemcpyNodeROIs).begin();
        std::advance(it, 1);
        NodeROI& tpcMemcpyNodeROI = (*it); // 2nd roi

        CacheMetaData md;
        md.mcid = 7;
        md.cmAction = DEGRADE;

        transposeNodeROI.outputsCacheMetaData.push_back(md);
        mmeNodeROI.inputsCacheMetaData.push_back(md);

        md.mcid = 8;
        md.cmAction = DISCARD;

        transposeNodeROI.outputsCacheMetaData.push_back(md);
        tpcMemcpyNodeROI.inputsCacheMetaData.push_back(md);

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        ASSERT_EQ (mmeNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on mme node";
        ASSERT_EQ (tpcMemcpyNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on TPCMemcpy node";

        // MCID = 7 --> 1 / DEGRADE
        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds[0].mcid, 1);
        ASSERT_EQ (mmeNodeROI.cmeTasks.cmCmds[0].op, DEGRADE);
        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_MME_LOGICAL_QUEUE], 1);

        // MCID = 8 --> 1 / DISCARD
        ASSERT_EQ(tpcMemcpyNodeROI.cmeTasks.cmCmds[0].mcid, 1);
        ASSERT_EQ (tpcMemcpyNodeROI.cmeTasks.cmCmds[0].op, DISCARD);
        ASSERT_EQ (tpcMemcpyNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 2);

        clear(g);
    }

void run_test_case5(Gaudi3Graph& g)
    {
        // scenario: MCID 7/DEGRADE on first roi output of transpose node
        //           MCID 7/DEGRADE of first roi input of MME node
        //           MCID 8/DISCARD of first roi output of MME node
        //           MCID 8/DISCARD of second roi output of MME node
        //           MCID 8/DISCARD on second roi output of transpose node
        //           MCID 8/DISCARD on input of 2nd roi of TPCMemcpy node
        //           MCID 8/DISCARD of first roi input of TPC node
        //           MCID 9/DISCARD on output of 2nd roi of TPCMemcpy node
        //           MCID 9/DISCARD of 2nd roi input of TPC node

        // Expecting 2 tasks:
        // One on first roi input of MME node
        // One for second roi input of TPC node

        std::list<NodeROI>* transposeNodeROIs = g.GetNodeROIs(transposeNode);
        NodeROI& transposeNodeROI = (*transposeNodeROIs).front();

        std::list<NodeROI>* mmeNodeROIs = g.GetNodeROIs(mmeNode);
        NodeROI& mmeNodeROI = (*mmeNodeROIs).front();

        std::list<NodeROI>* tpcNodeROIs = g.GetNodeROIs(tpcNode);
        NodeROI& tpcNodeROI = (*tpcNodeROIs).front();

        std::list<NodeROI>* tpcMemcpyNodeROIs = g.GetNodeROIs(tpcMemcpyNode);

        auto it = (*tpcMemcpyNodeROIs).begin();
        std::advance(it, 1);
        NodeROI& tpcMemcpyNodeROI = (*it); // 2nd roi

        CacheMetaData md;
        md.mcid = 7;
        md.cmAction = DEGRADE;

        transposeNodeROI.outputsCacheMetaData.push_back(md);
        mmeNodeROI.inputsCacheMetaData.push_back(md);

        md.mcid = 8;
        md.cmAction = DISCARD;

        mmeNodeROI.outputsCacheMetaData.push_back(md);
        mmeNodeROI.outputsCacheMetaData.push_back(md);
        transposeNodeROI.outputsCacheMetaData.push_back(md);
        tpcMemcpyNodeROI.inputsCacheMetaData.push_back(md);
        tpcNodeROI.inputsCacheMetaData.push_back(md);

        md.mcid = 9;

        tpcMemcpyNodeROI.outputsCacheMetaData.push_back(md);
        tpcNodeROI.inputsCacheMetaData.push_back(md);

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        ASSERT_EQ (mmeNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on mme node";
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on TPC node";

        // MCID = 7 --> 1 / DEGRADE
        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds[0].mcid, 1);
        ASSERT_EQ (mmeNodeROI.cmeTasks.cmCmds[0].op, DEGRADE);
        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_MME_LOGICAL_QUEUE], 1);

        // MCID = 8 --> 1 / DISCARD
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].mcid, 1);
        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds[0].op, DISCARD);
        ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 4);

        clear(g);
    }

    void run_test_case6(Gaudi3Graph& g)
    {
        // scenario: Fixed MCID(6) , ROIs input get DEGRADES while outputs set to DISCARD
        for (const NodePtr& n : g.getExeSortedNodes())
        {
            if (n == nullptr) continue;

            if (n->isLogicalOperation()) continue;

            CacheMetaData md;
            md.mcid = 6;

            for (auto& roi: (*g.GetNodeROIs(n)))
            {
                md.cmAction = DEGRADE;
                for (int i = 0; i < roi.inputRois.size(); i++)
                {
                    roi.inputsCacheMetaData.push_back(md);
                }

                md.cmAction = DISCARD;
                for (int i = 0; i < roi.outputRois.size(); i++)
                {
                    roi.outputsCacheMetaData.push_back(md);
                }
            }
        }

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        std::list<NodeROI>* tpcNodeROIs = g.GetNodeROIs(tpcNode);
        NodeROI& tpcNodeROI = (*tpcNodeROIs).front();

         ASSERT_EQ (tpcNodeROI.cmeTasks.cmCmds.size(), 2) << "Expecting 2 cme tasks on TPC node";

         // MCID = 6 --> 1 / DEGRADE
         ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].mcid, 1);
         ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].op, DEGRADE);
         ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 4);

         // MCID = 6 --> 1 / DISCARD
         ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[1].mcid, 1);
         ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[1].op, DISCARD);
         ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[1].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 4);

         clear(g);
    }

void run_test_case7(Gaudi3Graph& g)
    {
        // scenario: MCID 7/DEGRADE on first roi input and output of transpose node
        //           MCID 7/DISCARD of second roi input and output of transpose node
        //           MCID 8/DEGRADE on input of 1st roi of TPCMemcpy node
        //           MCID 8/DEGRADE on output of 1st roi of TPCMemcpy node
        //           MCID 9/DISCARD on input of 2nd roi of TPCMemcpy node
        //           MCID 9/DISCARD on output of 2nd roi of TPCMemcpy node
        //           MCID 10/DEGRADE of first roi output of MME node
        //           MCID 10/DISCARD of second roi output of MME node

        // Expecting 4 tasks:
        // One on MME roi node
        // One on transpose node
        // Two on TPCMemcpy node - one for each of first 2 rois

        std::list<NodeROI>* transposeNodeROIs = g.GetNodeROIs(transposeNode);
        NodeROI& transposeNodeROI = (*transposeNodeROIs).front();

        std::list<NodeROI>* mmeNodeROIs = g.GetNodeROIs(mmeNode);
        NodeROI& mmeNodeROI = (*mmeNodeROIs).front();

        std::list<NodeROI>* tpcMemcpyNodeROIs = g.GetNodeROIs(tpcMemcpyNode);

        auto it = (*tpcMemcpyNodeROIs).begin();

        NodeROI& tpcMemcpyNodeROI1 = (*it); // 1st roi

        std::advance(it, 1);
        NodeROI& tpcMemcpyNodeROI2 = (*it); // 2nd roi

        CacheMetaData md;
        md.mcid = 7;
        md.cmAction = DEGRADE;

        transposeNodeROI.inputsCacheMetaData.push_back(md);
        transposeNodeROI.outputsCacheMetaData.push_back(md);

        md.cmAction = DISCARD;
        transposeNodeROI.inputsCacheMetaData.push_back(md);
        transposeNodeROI.outputsCacheMetaData.push_back(md);

        md.mcid = 8;
        md.cmAction = DEGRADE;

        tpcMemcpyNodeROI1.inputsCacheMetaData.push_back(md);
        tpcMemcpyNodeROI1.outputsCacheMetaData.push_back(md);

        md.mcid = 9;
        md.cmAction = DISCARD;

        tpcMemcpyNodeROI2.inputsCacheMetaData.push_back(md);
        tpcMemcpyNodeROI2.outputsCacheMetaData.push_back(md);

        md.mcid = 10;
        md.cmAction = DEGRADE;

        mmeNodeROI.outputsCacheMetaData.push_back(md);

        md.cmAction = DISCARD;
        mmeNodeROI.outputsCacheMetaData.push_back(md);

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        ASSERT_EQ (mmeNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on mme node";
        ASSERT_EQ (transposeNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on transpose node";
        ASSERT_EQ (tpcMemcpyNodeROI1.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on first TPCMemcpy roi";
        ASSERT_EQ (tpcMemcpyNodeROI2.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on second TPCMemcpy roi";

        // MCID = 7 --> 1 / DEGRADE
        ASSERT_EQ(transposeNodeROI.cmeTasks.cmCmds[0].mcid, 1);
        ASSERT_EQ (transposeNodeROI.cmeTasks.cmCmds[0].op, DEGRADE);
        ASSERT_EQ (transposeNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_XPS_LOGICAL_QUEUE], 1);

        // MCID = 8 --> 2 / DEGRADE
        ASSERT_EQ(tpcMemcpyNodeROI1.cmeTasks.cmCmds[0].mcid, 2);
        ASSERT_EQ (tpcMemcpyNodeROI1.cmeTasks.cmCmds[0].op, DEGRADE);
        ASSERT_EQ (tpcMemcpyNodeROI1.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 1);

        // MCID = 9 --> 2 / DISCARD
        ASSERT_EQ(tpcMemcpyNodeROI2.cmeTasks.cmCmds[0].mcid, 2);
        ASSERT_EQ (tpcMemcpyNodeROI2.cmeTasks.cmCmds[0].op, DISCARD);
        ASSERT_EQ (tpcMemcpyNodeROI2.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 2);

        // MCID = 10 --> 3 / DEGRADE
        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds[0].mcid, 3);
        ASSERT_EQ (mmeNodeROI.cmeTasks.cmCmds[0].op, DEGRADE);
        ASSERT_EQ (mmeNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_MME_LOGICAL_QUEUE], 1);

        clear(g);
    }

    void run_test_case8(Gaudi3Graph& g)
    {
        // scenario: Fail on illegal brain input: MCID = 6 and NOP action
        std::list<NodeROI>* transposeNodeROIs = g.GetNodeROIs(transposeNode);
        NodeROI& transposeNodeROI = (*transposeNodeROIs).front();

        CacheMetaData md;
        md.mcid = 6;
        md.cmAction = NOP;

        transposeNodeROI.inputsCacheMetaData.push_back(md);

        try
        {
            CacheMaitenanceTasks cm(g);

            cm.executePass();

            // Shouldn't get here
            HB_ASSERT(0, "Test should fail on illegal brain input");
        }
        catch (const SynapseException& e)
        {
            // Failed as expected
            LOG_CRITICAL(GC, "generateCacheMaitenanceTasks pass failed: {}", e.what());
        }

        clear(g);
    }

    void run_test_case9(Gaudi3Graph& g)
    {
        // scenario: Fail on illegal brain input: MCID = 0 and DEGRADE action
        std::list<NodeROI>* transposeNodeROIs = g.GetNodeROIs(transposeNode);
        NodeROI& transposeNodeROI = (*transposeNodeROIs).front();

        CacheMetaData md;
        md.mcid = 0;
        md.cmAction = DEGRADE;

        transposeNodeROI.inputsCacheMetaData.push_back(md);

        try
        {
            CacheMaitenanceTasks cm(g);

            cm.executePass();

            // Shouldn't get here
            HB_ASSERT(0, "Test should fail on illegal brain input");
        }
        catch (const SynapseException& e)
        {
            // Failed as expected
            LOG_CRITICAL(GC, "generateCacheMaitenanceTasks pass failed: {}", e.what());
        }

        clear(g);
    }

    void run_test_case10(Gaudi3Graph& g)
    {
        // scenario: single user for MCID:
        //           MCID 7/DEGRADE for transpose node
        //           MCID 8/DISCARD for MME node

        // Expecting 2 tasks:
        // On first roi input of MME node
        // On first roi input of transpose  node

        std::list<NodeROI>* transposeNodeROIs = g.GetNodeROIs(transposeNode);
        NodeROI& transposeNodeROI = (*transposeNodeROIs).front();

        std::list<NodeROI>* mmeNodeROIs = g.GetNodeROIs(mmeNode);
        NodeROI& mmeNodeROI = (*mmeNodeROIs).front();

        CacheMetaData md;
        md.mcid = 7;
        md.cmAction = DEGRADE;

        transposeNodeROI.outputsCacheMetaData.push_back(md);

        md.mcid = 8;
        md.cmAction = DISCARD;

        mmeNodeROI.inputsCacheMetaData.push_back(md);

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        ASSERT_EQ (mmeNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on mme node";
        ASSERT_EQ (transposeNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on transpose node";

        // MCID = 7 --> 1 / DEGRADE
        ASSERT_EQ(transposeNodeROI.cmeTasks.cmCmds[0].mcid, 1);
        ASSERT_EQ (transposeNodeROI.cmeTasks.cmCmds[0].op, DEGRADE);
        ASSERT_EQ (transposeNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_XPS_LOGICAL_QUEUE], 1);

        // MCID = 8 --> 1 / DISCARD
        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds[0].mcid, 1);
        ASSERT_EQ (mmeNodeROI.cmeTasks.cmCmds[0].op, DISCARD);
        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_MME_LOGICAL_QUEUE], 1);

        clear(g);
    }

    void run_test_case11(Gaudi3Graph& g)
    {
        // scenario: single user for MCID:
        //           MCID 7/DEGRADE for transpose node
        //           MCID 8/DISCARD for MME node
        //           Call CodeGenerator::generateCmeCommands() the 2 nodes

        Gaudi3CodeGenerator& codeGen = dynamic_cast<Gaudi3CodeGenerator&>(*(g.getCodeGenerator()));

        std::list<EngArcCmdPtr>& cmeCmds = codeGen.getCmeCommands();

        ASSERT_EQ(cmeCmds.size(), 0);

        std::list<NodeROI>* transposeNodeROIs = g.GetNodeROIs(transposeNode);
        NodeROI& transposeNodeROI = (*transposeNodeROIs).front();

        std::list<NodeROI>* mmeNodeROIs = g.GetNodeROIs(mmeNode);
        NodeROI& mmeNodeROI = (*mmeNodeROIs).front();

        CacheMetaData md;
        md.mcid = 7;
        md.cmAction = DEGRADE;

        transposeNodeROI.outputsCacheMetaData.push_back(md);

        md.mcid = 8;
        md.cmAction = DISCARD;

        mmeNodeROI.inputsCacheMetaData.push_back(md);

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Expecting 2 cmeTasks - one on each mme & transpose nodes
        codeGen.generateCmeCommands(transposeNode);
        ASSERT_EQ(codeGen.getCmeCommands().size(), 1);

        EngArcCmdPtr cmd1 = codeGen.getCmeCommands().back();
        HB_ASSERT(cmd1 != nullptr, "Failed to get cme command");

        Gaudi3CmeDegradeArcCommand* cmdDegrade =  (Gaudi3CmeDegradeArcCommand *)cmd1.get();
        HB_ASSERT(cmdDegrade != nullptr, "Failed to get degrade command");

        codeGen.generateCmeCommands(mmeNode);
        ASSERT_EQ(codeGen.getCmeCommands().size(), 2);

        cme_arc_cmd_degrade_cls_t degradeCmdBinary;
        cmdDegrade->serialize(&degradeCmdBinary);

        ASSERT_EQ(degradeCmdBinary.cmd_type, CME_ECBL_CMD_DEGRADE_CLS);
        ASSERT_EQ(degradeCmdBinary.mcid_offset, 1);
        ASSERT_EQ(degradeCmdBinary.use_discard_base, 0);
        ASSERT_EQ(degradeCmdBinary.target_bitmap, 4);

        EngArcCmdPtr cmd2 = codeGen.getCmeCommands().back();
        HB_ASSERT(cmd2 != nullptr, "Failed to get cme command");

        Gaudi3CmeDiscardArcCommand* cmdDiscard = (Gaudi3CmeDiscardArcCommand *)cmd2.get();
        HB_ASSERT(cmdDiscard != nullptr, "Failed to get discard command");

        cme_arc_cmd_discard_cls_t discardCmdBinary;
        cmdDiscard->serialize(&discardCmdBinary);

        ASSERT_EQ(discardCmdBinary.cmd_type, CME_ECBL_CMD_DISCARD_CLS);
        ASSERT_EQ(discardCmdBinary.mcid_offset, 1);
        ASSERT_EQ(discardCmdBinary.target_bitmap, 2);

        clear(g);
    }

    void run_test_case12(Gaudi3Graph& g)
    {
        // scenario: Put DEGRADE action and 2 different MCIDs on tpc and transpose nodes
        // We expect 1 task on the tpc node as the Dependency Map should be identical for MCIDS 3 & 4
        // Also - MCIDs 3/4 should be replaced with MCID 1
        // In addition - validate the cache meta data modifications:
        // MCID = 1 / DEGRADE on tpcNode, MCID = 0 / NOP on transposeNode

        std::list<NodeROI>* transposeNodeROIs = g.GetNodeROIs(transposeNode);
        NodeROI&            transposeNodeROI  = (*transposeNodeROIs).front();

        std::list<NodeROI>* tpcNodeROIs = g.GetNodeROIs(tpcNode);
        NodeROI&            tpcNodeROI  = (*tpcNodeROIs).front();

        CacheMetaData md3;
        md3.mcid     = 3;
        md3.cmAction = DEGRADE;

        CacheMetaData md4;
        md4.mcid     = 4;
        md4.cmAction = DEGRADE;

        transposeNodeROI.inputsCacheMetaData.push_back(md3);
        transposeNodeROI.outputsCacheMetaData.push_back(md4);

        tpcNodeROI.inputsCacheMetaData.push_back(md3);
        tpcNodeROI.outputsCacheMetaData.push_back(md4);

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on tpc node";

        // MCID = 1 / DEGRADE
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].mcid, 1);
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].op, DEGRADE);
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 4);

        // validate cache meta data modifications
        ASSERT_EQ(tpcNodeROI.inputsCacheMetaData[0].mcid, 1);
        ASSERT_EQ(tpcNodeROI.outputsCacheMetaData[0].mcid, 1);
        ASSERT_EQ(tpcNodeROI.inputsCacheMetaData[0].cmAction, DEGRADE);
        ASSERT_EQ(tpcNodeROI.outputsCacheMetaData[0].cmAction, DEGRADE);

        ASSERT_EQ(transposeNodeROI.inputsCacheMetaData[0].mcid, 0);
        ASSERT_EQ(transposeNodeROI.outputsCacheMetaData[0].mcid, 0);
        ASSERT_EQ(transposeNodeROI.inputsCacheMetaData[0].cmAction, NOP);
        ASSERT_EQ(transposeNodeROI.outputsCacheMetaData[0].cmAction, NOP);

        clear(g);
    }

    void run_test_case1_with_signal_limit(Gaudi3Graph& g)
    {
        // scenario: Fixed op type (DEGRADE). Random/Incremented MCID for all ROIs
        // This is the same test as test_case1 above. However, since we set the sync scheme
        // signal limit to 3 the expected TPC sob value should be 1 instead of 4
        for (const NodePtr& n : g.getExeSortedNodes())
        {
            if (n == nullptr) continue;

            if (n->isLogicalOperation()) continue;

            LogicalMcid mcid = 6;

            for (auto& roi : (*g.GetNodeROIs(n)))
            {
                CacheMetaData md;

                md.mcid     = mcid;
                md.cmAction = DEGRADE;

                for (int i = 0; i < roi.inputRois.size(); i++)
                {
                    md.mcid++;
                    roi.inputsCacheMetaData.push_back(md);
                }

                for (int i = 0; i < roi.outputRois.size(); i++)
                {
                    md.mcid++;
                    roi.outputsCacheMetaData.push_back(md);
                }
            }
        }

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        std::list<NodeROI>* tpcNodeROIs = g.GetNodeROIs(tpcNode);
        NodeROI             tpcNodeROI  = (*tpcNodeROIs).front();

        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on tpc node";

        // MCID = 7 --> 2 (MCIDs 8 & 9 has same dependency map as MCID 7)
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].mcid, 2);
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].op, DEGRADE);
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 1);

        clear(g);
    }

    void run_test_case7_with_rollover(Gaudi3Graph& g)
    {
        // scenario: MCID 7/DISCARD on first roi input and output of transpose node
        //           MCID 7/DISCARD of second roi input and output of transpose node
        //           MCID 8/DISCARD on input of 1st roi of TPCMemcpy node
        //           MCID 8/DISCARD on output of 1st roi of TPCMemcpy node
        //           MCID 9/DISCARD on input of 2nd roi of TPCMemcpy node
        //           MCID 9/DISCARD on output of 2nd roi of TPCMemcpy node
        //           MCID 10/DISCARD of first roi output of MME node
        //           MCID 10/DISCARD of second roi output of MME node

        // Expecting 4 tasks:
        // One on MME roi node
        // One on transpose node
        // Two on TPCMemcpy node - one for each of first 2 rois
        // Also, expecting rollover task with Id 1 on last (3rd) TPCMemcpy

        std::list<NodeROI>* transposeNodeROIs = g.GetNodeROIs(transposeNode);
        NodeROI&            transposeNodeROI  = (*transposeNodeROIs).front();

        std::list<NodeROI>* mmeNodeROIs = g.GetNodeROIs(mmeNode);
        NodeROI&            mmeNodeROI  = (*mmeNodeROIs).front();

        std::list<NodeROI>* tpcMemcpyNodeROIs = g.GetNodeROIs(tpcMemcpyNode);

        auto it = (*tpcMemcpyNodeROIs).begin();

        NodeROI& tpcMemcpyNodeROI1 = (*it);  // 1st roi

        std::advance(it, 1);
        NodeROI& tpcMemcpyNodeROI2 = (*it);  // 2nd roi

        std::advance(it, 1);
        NodeROI& tpcMemcpyNodeROI3 = (*it);  // 3rd roi

        CacheMetaData md;
        md.mcid     = 7;
        md.cmAction = DISCARD;
        transposeNodeROI.inputsCacheMetaData.push_back(md);
        transposeNodeROI.outputsCacheMetaData.push_back(md);

        transposeNodeROI.inputsCacheMetaData.push_back(md);
        transposeNodeROI.outputsCacheMetaData.push_back(md);

        md.mcid = 8;
        tpcMemcpyNodeROI1.inputsCacheMetaData.push_back(md);
        tpcMemcpyNodeROI1.outputsCacheMetaData.push_back(md);

        md.mcid = 9;
        tpcMemcpyNodeROI2.inputsCacheMetaData.push_back(md);
        tpcMemcpyNodeROI2.outputsCacheMetaData.push_back(md);

        md.mcid = 10;
        mmeNodeROI.outputsCacheMetaData.push_back(md);
        mmeNodeROI.outputsCacheMetaData.push_back(md);

        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on mme node";
        ASSERT_EQ(transposeNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on transpose node";
        ASSERT_EQ(tpcMemcpyNodeROI1.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on first TPCMemcpy roi";
        ASSERT_EQ(tpcMemcpyNodeROI2.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on second TPCMemcpy roi";

        // MCID = 7 --> 1 / DISCARD
        ASSERT_EQ(transposeNodeROI.cmeTasks.cmCmds[0].mcid, 1);
        ASSERT_EQ(transposeNodeROI.cmeTasks.cmCmds[0].op, DISCARD);
        ASSERT_EQ(transposeNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_XPS_LOGICAL_QUEUE], 1);

        // MCID = 8 --> 2 / DISCARD
        ASSERT_EQ(tpcMemcpyNodeROI1.cmeTasks.cmCmds[0].mcid, 2);
        ASSERT_EQ(tpcMemcpyNodeROI1.cmeTasks.cmCmds[0].op, DISCARD);
        ASSERT_EQ(tpcMemcpyNodeROI1.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 1);

        // MCID = 9 --> 3 / DISCARD
        ASSERT_EQ(tpcMemcpyNodeROI2.cmeTasks.cmCmds[0].mcid, 3);
        ASSERT_EQ(tpcMemcpyNodeROI2.cmeTasks.cmCmds[0].op, DISCARD);
        ASSERT_EQ(tpcMemcpyNodeROI2.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE], 2);

        // MCID = 10 --> 4 / DISCARD
        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds[0].mcid, 4);
        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds[0].op, DISCARD);
        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds[0].deps[gaudi3::DEVICE_MME_LOGICAL_QUEUE], 1);

        // expecting rollover task on tpcMemcpyNodeROI3
        ASSERT_EQ(tpcMemcpyNodeROI3.cmeTasks.rollover.doRollover, true);
        ASSERT_EQ(tpcMemcpyNodeROI3.cmeTasks.rollover.rolloverId, 1);
        ASSERT_EQ(tpcMemcpyNodeROI3.cmeTasks.rollover.rolloverEngineBitmap, 1);

        clear(g);
    }

    void run_test_case7_with_rollover_on_mme_node(Gaudi3Graph& g)
    {
        // scenario: Fixed op type (DISCARD). Incremented MCID for first input/output in ROI
        // CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING = 5 - expecting rollover task on first mmeNode ROI

        CacheMetaData md;
        md.mcid     = 6;
        md.cmAction = DISCARD;

        for (const NodePtr& n : g.getExeSortedNodes())
        {
            if (n == nullptr || n->isLogicalOperation()) continue;

            for (auto& roi : (*g.GetNodeROIs(n)))
            {
                for (int i = 0; i < roi.inputRois.size(); i++)
                {
                    md.mcid++;
                    roi.inputsCacheMetaData.push_back(md);
                    break;
                }

                for (int i = 0; i < roi.outputRois.size(); i++)
                {
                    md.mcid++;
                    roi.outputsCacheMetaData.push_back(md);
                    break;
                }
            }
        }
        CacheMaitenanceTasks cmTasksPass(g);

        cmTasksPass.executePass();

        // Validate NodeROI cmeTasks & cmCmds
        std::list<NodeROI>* transposeNodeROIs = g.GetNodeROIs(transposeNode);
        NodeROI&            transposeNodeROI  = (*transposeNodeROIs).front();

        std::list<NodeROI>* mmeNodeROIs = g.GetNodeROIs(mmeNode);
        NodeROI&            mmeNodeROI  = (*mmeNodeROIs).front();

        std::list<NodeROI>* tpcMemcpyNodeROIs = g.GetNodeROIs(tpcMemcpyNode);
        NodeROI&            tpcMemcpyROI      = (*tpcMemcpyNodeROIs).front();

        std::list<NodeROI>* tpcNodeROIs = g.GetNodeROIs(tpcNode);
        NodeROI             tpcNodeROI  = (*tpcNodeROIs).front();

        ASSERT_EQ(mmeNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on mme node";
        ASSERT_EQ(transposeNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on transpose node";
        ASSERT_EQ(tpcNodeROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on tpc node";
        ASSERT_EQ(tpcMemcpyROI.cmeTasks.cmCmds.size(), 1) << "Expecting 1 cme task on tpc memcpy node";

        // expecting rollover task on mmeNode first ROI
        ASSERT_EQ(mmeNodeROI.cmeTasks.rollover.doRollover, true);
        ASSERT_EQ(mmeNodeROI.cmeTasks.rollover.rolloverId, 1);
        ASSERT_EQ(mmeNodeROI.cmeTasks.rollover.rolloverEngineBitmap, 1);

        clear(g);
    }

    void prepareTests(Gaudi3Graph& g)
    {
        // gather all interesting nodes for this test, so we can manipulate them over and over in the various tests
        for (const NodePtr& n : g.getExeSortedNodes())
        {
            if (n == nullptr) continue;

            if (n->isLogicalOperation()) continue;

            unsigned logicalQueueId = gaudi3::deviceTypeToLogicalQueue(g.getNodeUtility().getNodeDeviceType(n), *n);
            if (n->getExecutionOrderedIndex() == 1 && logicalQueueId == gaudi3::DEVICE_XPS_LOGICAL_QUEUE)
                transposeNode = n;
            if (n->getExecutionOrderedIndex() == 3 && logicalQueueId == gaudi3::DEVICE_TPC_LOGICAL_QUEUE)
                tpcMemcpyNode = n;
            if (n->getExecutionOrderedIndex() == 5 && logicalQueueId == gaudi3::DEVICE_MME_LOGICAL_QUEUE) mmeNode = n;
            if (n->getExecutionOrderedIndex() == 6 && logicalQueueId == gaudi3::DEVICE_TPC_LOGICAL_QUEUE) tpcNode = n;
        }

        for (const auto& n : {transposeNode, tpcMemcpyNode, mmeNode, tpcNode})
        {
            ASSERT_NE(n, nullptr);
        }

        ASSERT_EQ(g.GetNodeROIs(transposeNode)->size(), 1) << "Expecting 1 roi for transpose node";
        ASSERT_EQ(g.GetNodeROIs(tpcMemcpyNode)->size(), 3) << "Expecting 3 rois for tpcMemcpy node";
        ASSERT_EQ(g.GetNodeROIs(mmeNode)->size(), 1) << "Expecting 1 roi for mme node";
        ASSERT_EQ(g.GetNodeROIs(tpcNode)->size(), 1) << "Expecting 1 roi for tpc node";

        // clear NodeROI from compilation results
        clear(g);
    }

    void runAllTests(Gaudi3Graph& g)
    {
        prepareTests(g);

        run_test_case1(g);
        run_test_case2(g);
        run_test_case3(g);
        run_test_case4(g);
        run_test_case5(g);
        run_test_case6(g);
        run_test_case7(g);

        run_test_case8(g);
        run_test_case9(g);
        run_test_case10(g);
        run_test_case11(g);
        run_test_case12(g);
    }
};

TEST_F(GenerateCmeTaskPassTest, gaudi3_cm_test_with_tpc_mme_transpose_nodes)
{
    setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "true");

    Gaudi3Graph g;

    createGraph(g);

    ASSERT_TRUE(g.compile());
    ASSERT_EQ(g.getNodes().size(), 7) << "Expecting 7 nodes in graph: physical and logical transposes, flatten and "
                                         "unflatten, convolution node, tpc node and memcpy";

    uint32_t mmeNodeCounter = 0;
    uint32_t rspNodeCounter = 0;
    uint32_t tpcNodeCounter = 0;

    for (const auto& node : g.getNodes())
    {
        if (g.runsOnMME(node))
        {
            mmeNodeCounter++;
        }

        if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            rspNodeCounter++;
        }

        if (g.runsOnTPC(node))
        {
            tpcNodeCounter++;
        }
    }

    ASSERT_EQ(mmeNodeCounter, 2) << "Expecting two MME nodes in post graph";
    ASSERT_EQ(rspNodeCounter, 2) << "Expecting two reshape nodes in post graph";
    ASSERT_EQ(tpcNodeCounter, 2) << "Expecting one tpc memcpy node and one tpc add node in post graph";

    runAllTests(g);
}

TEST_F(GenerateCmeTaskPassTest, gaudi3_cm_test_with_sync_scheme_signal_limit)
{
    setGlobalConfForTest(GCFG_ARC_SYNC_SCHEME_SIGNAL_LIMIT, "3");

    Gaudi3Graph g;

    createGraph(g);

    ASSERT_TRUE(g.compile());
    ASSERT_EQ(g.getNodes().size(), 7) << "Expecting 7 nodes in graph: physical and logical transposes, flatten and "
                                         "unflatten, convolution node, tpc node and memcpy";

    prepareTests(g);

    run_test_case1_with_signal_limit(g);
}

TEST_F(GenerateCmeTaskPassTest, gaudi3_cm_test_with_rollover)
{
    setGlobalConfForTest(GCFG_CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING, "3");

    Gaudi3Graph g;

    createGraph(g);

    ASSERT_TRUE(g.compile());
    ASSERT_EQ(g.getNodes().size(), 7) << "Expecting 7 nodes in graph: physical and logical transposes, flatten and "
                                         "unflatten, convolution node, tpc node memcpy";

    prepareTests(g);

    run_test_case7_with_rollover(g);
}

TEST_F(GenerateCmeTaskPassTest, gaudi3_cm_test_with_rollover_on_mme_node)
{
    setGlobalConfForTest(GCFG_CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING, "5");

    Gaudi3Graph g;

    createGraph(g);

    ASSERT_TRUE(g.compile());
    ASSERT_EQ(g.getNodes().size(), 7) << "Expecting 7 nodes in graph: physical and logical transposes, flatten and "
                                         "unflatten, convolution node, tpc node and memcpy";

    prepareTests(g);

    run_test_case7_with_rollover_on_mme_node(g);
}

TEST_F(GenerateCmeTaskPassTest, gaudi3_no_cme_task_for_alias_tensor)
{
    ScopedConfigurationChange experimentalFlagsCfg("ENABLE_EXPERIMENTAL_FLAGS", "true");
    ScopedConfigurationChange skipBundleCheckCfg("LITE_PERFORATION_SKIP_BUNDLE_CHECK", "true");

    const unsigned      tensor_dim = 1;
    const TSize         size       = 1;
    Gaudi3Graph         g;
    TensorPtr           i1 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr           i2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr           o  = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    NodePtr             n  = NodeFactory::createNode({i1, i2}, {o}, nullptr, "add_fwd_f32", "add");
    synMemoryDescriptor memDesc(true);  // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    i1->setDramOffset(0x1000);
    i2->setDramOffset(0x2000);
    i1->setMemoryDescriptor(memDesc);
    i2->setMemoryDescriptor(memDesc);

    o->setAsAliasSubTensor(i1);

    ASSERT_TRUE(GraphEditor::addNode(g, n));
    ASSERT_TRUE(g.compile()) << "failed to compile graph";
    ASSERT_EQ(g.getNodes().size(), 1) << "Expecting a single node in graph";

    // sum the number of cmeTasks. Expecting 0 tasks as the alias tensor is aliased to persistent one
    unsigned numOfCMETasks = 0;
    for (const NodePtr& n : g.getExeSortedNodes())
    {
        if (n == nullptr || n->isLogicalOperation()) continue;

        for (auto& roi : (*(g.GetNodeROIs(n))))
        {
            numOfCMETasks += roi.cmeTasks.cmCmds.size();
        }
    }
    ASSERT_EQ(numOfCMETasks, 0) << "Expecting no cmeTasks in graph";
}
