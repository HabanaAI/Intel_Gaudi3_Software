#include <gtest/gtest.h>
#include <memory>
#include "compilation_hal_reader.h"
#include "graph_optimizer_test.h"
#include "mme_brain_ifc.h"
#include "node_annotation.h"
#include "tensor.h"
#include "node_factory.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "synapse_common_types.h"
#include "platform/gaudi3/graph_compiler/passes.h"
#include "hal_reader/gaudi3/hal_reader.h"

namespace gaudi3
{
using TestParamTuple = std::tuple<unsigned,    // cDim
                                  unsigned,    // kDim
                                  unsigned,    // wDim
                                  unsigned,    // hDim
                                  unsigned,    // dDim
                                  unsigned,    // bDim
                                  const char*  // opType
                                  >;

typedef struct
{
    TensorPtr a;
    TensorPtr b;
    TensorPtr c;
} TensorsData;

class Gaudi3CDParallelTest
: public GraphOptimizerTest
, public testing::WithParamInterface<TestParamTuple>
{
public:
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(Gaudi3HalReader::instance());
        setGlobalConfForTest(GCFG_ENABLE_CD_PARALLEL, "true");
    }

    static TensorsData setTensorsByOp(TensorPtr x, TensorPtr w, TensorPtr y, const char* opType)
    {
        TensorsData tensors;

        if (!strcmp(opType, NodeFactory::convolutionNodeTypeName))
        {
            tensors.a     = x;
            tensors.b     = w;
            tensors.c     = y;
        }
        else if (!strcmp(opType, NodeFactory::deDwNodeTypeName))
        {
            tensors.a     = y;
            tensors.b     = x;
            tensors.c     = w;
        }
        else
        {
            tensors.a     = y;
            tensors.b     = w;
            tensors.c     = x;
        }

        return tensors;
    }
};

// TODO [SW-158409] - enable this
TEST_P(Gaudi3CDParallelTest, DISABLED_cd_parallel_test)
{
    Gaudi3Graph g;
    unsigned    dcoreNr = CompilationHalReader::getHalReader()->getNumDcores();

    synConvolution3DParams params {};
    params.kernel[CONV_KERNEL_WIDTH]  = 1;
    params.kernel[CONV_KERNEL_HEIGHT] = 1;
    params.kernel[CONV_KERNEL_DEPTH]  = 1;

    unsigned cDim = std::get<0>(GetParam());
    unsigned kDim = std::get<1>(GetParam());
    unsigned wDim = std::get<2>(GetParam());
    unsigned hDim = std::get<3>(GetParam());
    unsigned dDim = std::get<4>(GetParam());
    unsigned bDim = std::get<5>(GetParam());

    const char* opType = std::get<6>(GetParam());

    const unsigned wOFM = convOutputDimSize(wDim,
                                            params.kernel[CONV_KERNEL_WIDTH],
                                            params.stride[CONV_STRIDE_WIDTH],
                                            params.padding[CONV_PAD_LEFT],
                                            params.dilation[CONV_DIL_WIDTH]);

    const unsigned hOFM = convOutputDimSize(hDim,
                                            params.kernel[CONV_KERNEL_HEIGHT],
                                            params.stride[CONV_STRIDE_HEIGHT],
                                            params.padding[CONV_PAD_TOP],
                                            params.dilation[CONV_DIL_HEIGHT]);
    const unsigned dOFM = convOutputDimSize(dDim,
                                            params.kernel[CONV_KERNEL_DEPTH],
                                            params.stride[CONV_STRIDE_DEPTH],
                                            params.padding[CONV_PAD_FRONT],
                                            params.dilation[CONV_DIL_DEPTH]);

    const TSize xSizes[] = {cDim, wDim, hDim, dDim, bDim};
    const TSize wSizes[] = {kDim,
                            cDim,
                            params.kernel[CONV_KERNEL_WIDTH],
                            params.kernel[CONV_KERNEL_HEIGHT],
                            params.kernel[CONV_KERNEL_DEPTH]};
    const TSize ySizes[] = {kDim, wOFM, hOFM, dOFM, bDim};

    TensorPtr x = TensorPtr(new Tensor(5U, xSizes, syn_type_single));
    TensorPtr w = TensorPtr(new Tensor(5U, wSizes, syn_type_single));
    TensorPtr y = TensorPtr(new Tensor(5U, ySizes, syn_type_single));

    TensorsData tensors = setTensorsByOp(x, w, y, opType);

    const TSize auxScratchpadSizes[] = {tensors.c->getTotalElements(), dcoreNr};  // partials output
    const TSize auxReductionSizes[]  = {dcoreNr, 1};                              // 1's tensor for reductionAdd
    TensorPtr   auxScratchpad        = TensorPtr(new Tensor(2U, auxScratchpadSizes, syn_type_single));
    TensorPtr   auxReduction         = TensorPtr(new Tensor(2U, auxReductionSizes, syn_type_single));

    synMemoryDescriptor memDesc(true);  // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    x->setDramOffset(0x10000);
    w->setDramOffset(0x20000);
    y->setDramOffset(0x30000);
    auxScratchpad->setDramOffset(0x40000);
    auxReduction->setDramOffset(0x50000);

    x->setMemoryDescriptor(memDesc);
    w->setMemoryDescriptor(memDesc);
    y->setMemoryDescriptor(memDesc);
    auxScratchpad->setMemoryDescriptor(memDesc);
    auxReduction->setMemoryDescriptor(memDesc);

    x->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    w->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    y->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);

    auxScratchpad->getTensorAnnotation().isAuxTensor = true;
    auxReduction->getTensorAnnotation().isAuxTensor  = true;

    NodePtr conv = NodeFactory::createInternalNode({tensors.a, tensors.b}, {tensors.c}, &params, opType, "mme_node");
    conv->addInput(TENSOR_AUX_CD_SCRATCHPAD, auxScratchpad, Node::TENSOR_TYPE_DATA, true);
    conv->addInput(TENSOR_AUX_CD_REDUCTION, auxReduction, Node::TENSOR_TYPE_DATA, true);
    GraphEditor::addNode(g, conv);

    CacheMetaData cacheMetaData;
    cacheMetaData.cacheDirective = CacheDirective::HomeAllocate;
    conv->getNodeAnnotation().inputsCacheMetaData.resize(6);
    conv->getNodeAnnotation().outputsCacheMetaData.resize(1);
    conv->getNodeAnnotation().inputsCacheMetaData[0]  = cacheMetaData;
    conv->getNodeAnnotation().inputsCacheMetaData[1]  = cacheMetaData;
    conv->getNodeAnnotation().inputsCacheMetaData[4]  = cacheMetaData;
    conv->getNodeAnnotation().inputsCacheMetaData[5]  = cacheMetaData;
    conv->getNodeAnnotation().outputsCacheMetaData[0] = cacheMetaData;

    DcoreROI dcoreRoiObj;
    std::fill(std::begin(dcoreRoiObj.size), std::end(dcoreRoiObj.size), 1);

    for (int i = 0; i < dcoreNr; i++)
    {
        conv->getNodeAnnotation().m_dcoreROIs.push_back(dcoreRoiObj);
    }
    generateROIs(g);


    auto                                     ap = conv->getNodeAccessPattern();
    std::shared_ptr<MmeBrainIfc> brainIfc = std::dynamic_pointer_cast<MmeNode>(conv)->getMmeBrainIfc();
    conv->getNodeAnnotation().perforationDim = brainIfc->getCDDims().front();
    ASSERT_TRUE(generateMmeDescriptors(g));

    MmeDescriptorGenerator& descGen = g.getMmeNodeDescriptorGenerator(conv);

    unsigned numSignals = 0;
    for (auto& activation : descGen.getMmeActivations())
    {
        ASSERT_EQ(activation.descriptors.size(), 8);
        numSignals += activation.numSignals;
    }

    ASSERT_EQ(numSignals, 2);  // one signal for partials and one for reductionAdd

    // check operand roles
    MmeCommon::OperandRoles partialsOpRoles  = descGen.getMmeActivations().front().operandRoles;
    MmeCommon::OperandRoles reductionOpRoles = descGen.getMmeActivations().back().operandRoles;

    ASSERT_EQ(partialsOpRoles[0], MmeCommon::INPUT_TENSOR_A);
    ASSERT_EQ(partialsOpRoles[1], MmeCommon::INPUT_TENSOR_B);
    ASSERT_EQ(partialsOpRoles[2], MmeCommon::AUX_TENSOR_SCRATCHPAD);

    ASSERT_EQ(reductionOpRoles[0], MmeCommon::AUX_TENSOR_REDUCTION);
    ASSERT_EQ(reductionOpRoles[1], MmeCommon::AUX_TENSOR_SCRATCHPAD);
    ASSERT_EQ(reductionOpRoles[2], MmeCommon::OUTPUT_TENSOR_C);

    // check patching
    ASSERT_TRUE(patchMmeDescriptors(g));
    ASSERT_EQ(descGen.getMmeActivations().front().descriptors[0].baseAddrCOut0.addr, auxScratchpad->getDramOffset());
    ASSERT_EQ(descGen.getMmeActivations().back().descriptors[0].baseAddrA.addr, auxReduction->getDramOffset());
    ASSERT_EQ(descGen.getMmeActivations().back().descriptors[0].baseAddrB.addr, auxScratchpad->getDramOffset());
}

INSTANTIATE_TEST_SUITE_P(,
                         Gaudi3CDParallelTest,
                         ::testing::Values(std::make_tuple(64, 64, 64, 1, 1, 1, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(64, 64, 64, 2, 1, 1, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(64, 64, 64, 2, 4, 5, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(100, 64, 64, 1, 1, 1, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(128, 64, 64, 1, 1, 1, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(128, 64, 64, 1, 3, 2, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(256, 64, 64, 1, 1, 1, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(32, 256, 512, 1, 1, 1, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(64, 256, 512, 1, 1, 1, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(31, 256, 512, 1, 1, 1, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(31, 256, 512, 1, 2, 1, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(50, 256, 512, 1, 1, 1, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(50, 256, 512, 1, 1, 3, NodeFactory::convolutionNodeTypeName),
                                           std::make_tuple(64, 64, 64, 1, 1, 1, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(64, 64, 64, 2, 1, 1, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(64, 64, 64, 2, 4, 5, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(100, 64, 64, 1, 1, 1, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(128, 64, 64, 1, 1, 1, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(128, 64, 64, 1, 3, 2, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(256, 64, 64, 1, 1, 1, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(32, 256, 512, 1, 1, 1, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(64, 256, 512, 1, 1, 1, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(31, 256, 512, 1, 1, 1, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(31, 256, 512, 1, 2, 1, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(50, 256, 512, 1, 1, 1, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(50, 256, 512, 1, 1, 3, NodeFactory::deDwNodeTypeName),
                                           std::make_tuple(64, 64, 64, 1, 1, 1, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(64, 64, 64, 2, 1, 1, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(64, 64, 64, 2, 4, 5, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(100, 64, 64, 1, 1, 1, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(128, 64, 64, 1, 1, 1, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(128, 64, 64, 1, 3, 2, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(256, 64, 64, 1, 1, 1, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(32, 256, 512, 1, 1, 1, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(64, 256, 512, 1, 1, 1, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(31, 256, 512, 1, 1, 1, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(31, 256, 512, 1, 2, 1, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(50, 256, 512, 1, 1, 1, NodeFactory::deDxNodeTypeName),
                                           std::make_tuple(50, 256, 512, 1, 1, 3, NodeFactory::deDxNodeTypeName)));
}  // namespace gaudi3