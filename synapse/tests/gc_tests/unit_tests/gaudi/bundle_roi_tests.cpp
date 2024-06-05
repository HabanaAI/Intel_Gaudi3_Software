#include <platform/gaudi/graph_compiler/passes.h>
#include "sram_management_fe_test.h"
#include "tpc_slice.h"

class BundleRoiTest : public gaudi::SRAMManagementTest
{
public:
    void SetUp() override
    {
        gaudi::SRAMManagementTest::SetUp();
        m_origPipelineEdgesEnabled = GCFG_PIPELINE_BUNDLE_EDGES_ENABLED.value();
        GCFG_PIPELINE_BUNDLE_EDGES_ENABLED.setValue(true);
    }
    void TearDown() override
    {
        GCFG_PIPELINE_BUNDLE_EDGES_ENABLED.setValue(m_origPipelineEdgesEnabled);
        gaudi::SRAMManagementTest::TearDown();
    }

private:
    bool m_origPipelineEdgesEnabled;
};

TEST_F(BundleRoiTest, only_first_and_last_node_of_an_engine_should_have_roi_split_annotation_true)
{
    pTensor r = createTensor({256, 32, 32, 64}, syn_type_bf16);
    pTensor x = createTensor({256, 32, 32, 64}, syn_type_bf16);
    pTensor w = createTensor({512, 256, 1, 1}, syn_type_bf16);
    pTensor o = createTensor({512, 32, 32, 64}, syn_type_bf16);

    synConvolutionParams params{};

    pNode relu = NodeFactory::createGenericTPCNode({r}, {x}, nullptr, "relu_fwd_bf16", "relu");
    pNode conv = NodeFactory::createNode({x, w}, {o}, &params, NodeFactory::convolutionNodeTypeName, "conv");

    GraphEditor::addNode(getGraph(), relu);
    GraphEditor::addNode(getGraph(), conv);

    ASSERT_TRUE(getGraph().compile());

    uint32_t numOfMmeNodes = 0;
    uint32_t numOfTpcNodes = 0;
    uint32_t numOfDmaNodes = 0;
    for (const pNode& n : getGraph().getExeSortedNodes())
    {
        if (!n->getNodeAnnotation().bundleInfo.is_set()) continue;
        if (GaudiGraph::runsOnMME(n))                           numOfMmeNodes++;
        if (GaudiGraph::runsOnTPC(n))                           numOfTpcNodes++;
        if (std::dynamic_pointer_cast<DMANode>(n) != nullptr)   numOfDmaNodes++;
    }

    uint32_t curMmeNode = 0;
    uint32_t curTpcNode = 0;
    uint32_t curDmaNode = 0;
    for (const pNode& n : getGraph().getExeSortedNodes())
    {
        const auto& annotations = n->getNodeAnnotation();
        if (!annotations.bundleInfo.is_set()) continue;
        if (GaudiGraph::runsOnMME(n))
        {
            if (curMmeNode == 0 || curMmeNode == numOfMmeNodes - 1)
            {
                ASSERT_TRUE(annotations.splitToLogicalROIs)
                                        << "Expect split to ROI indication to be true for node " << n->getNodeName();
            }
            else
            {
                ASSERT_FALSE(annotations.splitToLogicalROIs)
                                        << "Expect split to ROI indication to be false for node " << n->getNodeName();
            }
            curMmeNode++;
        }
        if (GaudiGraph::runsOnTPC(n))
        {
            if (curTpcNode == 0 || curTpcNode == numOfTpcNodes - 1)
            {
                ASSERT_TRUE(annotations.splitToLogicalROIs)
                                        << "Expect split to ROI indication to be true for node " << n->getNodeName();
            }
            else
            {
                ASSERT_FALSE(annotations.splitToLogicalROIs)
                                        << "Expect split to ROI indication to be false for node " << n->getNodeName();
            }
            curTpcNode++;
        }
        if (std::dynamic_pointer_cast<DMANode>(n) != nullptr)
        {
            if (curDmaNode == 0 || curDmaNode == numOfDmaNodes - 1)
            {
                ASSERT_TRUE(annotations.splitToLogicalROIs)
                                        << "Expect split to ROI indication to be true for node " << n->getNodeName();
            }
            else
            {
                ASSERT_FALSE(annotations.splitToLogicalROIs)
                                        << "Expect split to ROI indication to be false for node " << n->getNodeName();
            }
            curDmaNode++;
        }
    }
}

TEST_F(BundleRoiTest, fused_tpc)
{
    // SRAM slicing pass happens after the TPC fuser pass.
    // Therefore we can't assume that the tensors sizes that were used to generate
    // the kernel remain the same.
    // The purpose of this test is to verify that the index-space of a fused
    // TPC kernel that is sliced as part of a bundle is calculated correctly.
    setGlobalConfForTest(GCFG_PIPELINE_BUNDLE_EDGES_ENABLED, "false");
    setGlobalConfForTest(GCFG_RUN_TPC_FUSER, "true");

    // Create a graph with GEMM -> Add -> Cast nodes.
    // The Add+Cast will be fused.
    pTensor a = createTensor({768,4096}, syn_type_bf16, true);
    pTensor b = createTensor({768,768}, syn_type_bf16, true);
    pTensor gemmOut = createTensor({768,4096}, syn_type_bf16, false);
    synGEMMParams gemmParams;
    pNode gemm = NodeFactory::createNode({a, b}, {gemmOut}, &gemmParams, "gemm", "GEMM");
    GraphEditor::addNode(getGraph(), gemm);

    pTensor addIn = createTensor({768,4096}, syn_type_bf16, true);
    pTensor addOut = createTensor({768,4096}, syn_type_bf16, false);
    pNode add = NodeFactory::createGenericTPCNode({gemmOut, addIn}, {addOut}, nullptr, "add_fwd_bf16", "ADD");
    GraphEditor::addNode(getGraph(), add);

    pTensor castOut = createTensor({768,4096}, syn_type_single, true);
    pNode cast = NodeFactory::createGenericTPCNode({addOut}, {castOut}, nullptr, "cast_bf16_to_f32", "CAST");
    GraphEditor::addNode(getGraph(), cast);

    ASSERT_TRUE(getGraph().compile());

    for (const pNode& node : getGraph().getExeSortedNodes())
    {
        if (GaudiGraph::runsOnTPC(node))
        {
            auto tpcSlice = std::dynamic_pointer_cast<TPCSlice>(node);
            ASSERT_TRUE(tpcSlice != nullptr);

            const std::list<NodeROI>* rois = getGraph().GetNodeROIs(node);
            ASSERT_EQ(rois->size(), 1);

            const NodeROI& roi = rois->front();

            const auto& kernelInstanse = tpcSlice->getInstance();
            for (unsigned inputIdx = 0; inputIdx < node->getNumInputs(); ++inputIdx)
            {
                const auto& inAccessPattern = kernelInstanse.inputTensorAccessPattern[inputIdx];
                const pTensor inputTensor = node->getInput(inputIdx);

                for (unsigned int dim = 0; dim < inputTensor->getDim(); ++dim)
                {
                    // Which index-space dimension relates to this tensor dimension
                    const auto indexSpaceDim = inAccessPattern.mapping[dim].indexSpaceDim;
                    // Verify that the final activation on each dimension covers the new tensor size
                    const auto startPixel = (double)inAccessPattern.mapping[dim].a *
                                                (roi.size[indexSpaceDim] + roi.baseOffset[indexSpaceDim] - 1) +
                                            (double)inAccessPattern.mapping[dim].start_b;
                    const auto endPixel = (double)inAccessPattern.mapping[dim].a *
                                              (roi.size[indexSpaceDim] + roi.baseOffset[indexSpaceDim] - 1) +
                                          (double)inAccessPattern.mapping[dim].end_b;
                    const auto tensorEndPixel =
                        inputTensor->getSizeInElements(dim) - 1 + tpcSlice->getTensorSliceOffsetInDim(inputTensor, dim);

                    // Verify that the ROI size matches the sliced tensor dimenstions.
                    ASSERT_TRUE(startPixel <= tensorEndPixel);
                    ASSERT_TRUE(tensorEndPixel <= endPixel);
                }
            }
            for (unsigned outputIdx = 0; outputIdx < node->getNumOutputs(); ++outputIdx)
            {
                const auto& outAccessPattern = kernelInstanse.outputTensorAccessPattern[outputIdx];
                const pTensor outputTensor = node->getOutput(outputIdx);

                for (unsigned int dim = 0; dim < outputTensor->getDim(); ++dim)
                {
                    // Which index-space dimension relates to this tensor dimension
                    const auto indexSpaceDim = outAccessPattern.mapping[dim].indexSpaceDim;
                    const auto startPixel    = (double)outAccessPattern.mapping[dim].a *
                                                (roi.size[indexSpaceDim] + roi.baseOffset[indexSpaceDim] - 1) +
                                            (double)outAccessPattern.mapping[dim].start_b;
                    const auto endPixel = (double)outAccessPattern.mapping[dim].a *
                                              (roi.size[indexSpaceDim] + roi.baseOffset[indexSpaceDim] - 1) +
                                          (double)outAccessPattern.mapping[dim].end_b;
                    const auto tensorEndPixel = outputTensor->getSizeInElements(dim) - 1 +
                                                tpcSlice->getTensorSliceOffsetInDim(outputTensor, dim);

                    // Verify that the ROI size matches the sliced tensor dimensions.
                    ASSERT_TRUE(startPixel <= tensorEndPixel);
                    ASSERT_TRUE(tensorEndPixel <= endPixel);
                }
            }
        }
    }
}