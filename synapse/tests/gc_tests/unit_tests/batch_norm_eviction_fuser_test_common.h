#pragma once

#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "passes/sram_management/batch_norm_eviction_fuser.h"
#include "passes/sram_management/bundle.h"
#include "passes/sram_management/bundlizer.h"
#include "passes/sram_management/mme_slicing_strategy.h"
#include "passes/sram_management/pattern_solvers.h"
#include "passes/sram_management/slice_mapping.h"
#include "passes/sram_management/slicing_brain.h"
#include "passes/sram_management/slicing_strategy.h"
#include "perf_lib_layer_params.h"
#include "synapse_common_types.hpp"
#include "tensor.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include <cassert>
#include <memory>
#include "passes/sram_management/slicing_utils.h"
#include "types.h"
#include "compilation_hal_reader.h"

template<class GraphType>
class BatchNormEvictionFuserTest :
    public GraphOptimizerTest,
    public testing::WithParamInterface<std::tuple<bool,   // 1st param: Is mme operand persistent? Refers to mme input in BN1
                                                          // fusion case, mme output in BN2
                                                 bool>>   // 2nd param: add external consumer to mme output/input ? (same distinction
                                                          // as above)
                                                          // Those 2 parameters have indluence on whether eviction
                                                          //fusion should happen
{
public:
    virtual ~BatchNormEvictionFuserTest() = default;

protected:
    using SizeVec = std::vector<TSize>;

    static TensorPtr createTensor(SizeVec shape = {})
    {
        if (shape.empty())
        {
            shape = {128, 128, 1, 1};
        }
        TensorPtr tensor = std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_float, shape.data());
        return tensor;
    }

    static TensorPtr createPersistentTensor(const SizeVec& shape = {})
    {
        auto tensor = createTensor(shape);

        synMemoryDescriptor memDesc(true);
        tensor->setMemoryDescriptor(memDesc);
        return tensor;
    }

    virtual void buildGraph(bool mmeOperandPersistent, bool addExternalConsumer) = 0;
    virtual void preBundleHandling()                                             = 0;
    virtual void bundleGraph()                                                   = 0;
    virtual void fuseEviction()                                                  = 0;
    virtual void validateFusion()                                                = 0;
    virtual void validateNoFusion()                                              = 0;

    virtual void runBnEvictionFuserTest()
    {
        bool                       mmeOperandPersistent = std::get<0>(GetParam());
        bool                       addExternalConsumer  = std::get<1>(GetParam());
        CompilationHalReaderSetter compHalReaderSetter(&m_graph);

        buildGraph(mmeOperandPersistent, addExternalConsumer);
        preBundleHandling();
        bundleGraph();

        // When
        fuseEviction();

        if (mmeOperandPersistent || addExternalConsumer)
        {
            // Then expect mmeOut to be produced by the BN and the strategy updated accordingly
            validateFusion();
        }
        else
        {
            // Expect no change
            validateNoFusion();
        }
    }
    GraphType m_graph;

    NodePtr m_gemm;
    NodePtr m_bn1;
    NodePtr m_bn2;

    TensorPtr m_mmeIn;
    TensorPtr m_mmeOut;

    pBundle             m_bundle;
    pMmeSlicingStrategy m_strategy;
};

template<class GraphType>

class BatchNormStage1FwdEvictionFuserTestCommon : public BatchNormEvictionFuserTest<GraphType>
{
public:
    BatchNormStage1FwdEvictionFuserTestCommon()  = default;
    ~BatchNormStage1FwdEvictionFuserTestCommon() = default;

    using Base =
        BatchNormEvictionFuserTest<GraphType>;  // needed using because the parent class is a template. Base Class
                                                // methods calls in Derived Class requires info about specific type.
    virtual void buildGraph(bool mmeOperandPersistent, bool addExternalConsumer) override
    {
        // Build:  MME -> [mmeOut] -> BN_stage1
        // mmeOut may be persistent depending on param 0 and may be memcpied depending on param 1

        TensorPtr inA  = Base::createPersistentTensor();
        TensorPtr inB  = Base::createPersistentTensor();
        Base::m_mmeOut = mmeOperandPersistent ? Base::createPersistentTensor() : Base::createTensor();
        synGEMMParams gemmParams {};
        Base::m_gemm =
            NodeFactory::createNode({inA, inB}, {Base::m_mmeOut}, &gemmParams, NodeFactory::gemmNodeTypeName, "gemm");
        ASSERT_TRUE(GraphEditor::addNode(Base::m_graph, Base::m_gemm));

        ns_BatchNormStage1Kernel::ParamsV2 bnParams {};
        bnParams.isTraining = true;
        bnParams.N          = Base::m_mmeOut->getDenseSizeInElements() / Base::m_mmeOut->getSizeInElements(0);
        TensorPtr sigmas    = Base::createTensor({128, 3});
        TensorPtr k         = Base::createPersistentTensor({128});
        Base::m_bn1 =
            NodeFactory::createNode({Base::m_mmeOut, k}, {sigmas}, &bnParams, "batch_norm_stage1_fwd_f32", "bn1");
        ASSERT_TRUE(GraphEditor::addNode(Base::m_graph, Base::m_bn1));

        if (addExternalConsumer)
        {
            TensorPtr mmeOutCopy = Base::createTensor();
            NodePtr   memcpy     = NodeFactory::createNode({Base::m_mmeOut},
                                                     {mmeOutCopy},
                                                     nullptr,
                                                     NodeFactory::memcpyNodeTypeName,
                                                     "memcpy");
            ASSERT_TRUE(GraphEditor::addNode(Base::m_graph, memcpy));
        }
    }

    virtual void bundleGraph() override
    {
        AllBrains brains(Base::m_graph);

        // A bundle containing the GEMM
        Bundlizer  bundlizer(Base::m_graph);
        BundleList mmeBundles, dummy[4];
        bundlizer.generateBundles(mmeBundles, dummy[0], dummy[1], dummy[2], dummy[3]);
        ASSERT_EQ(mmeBundles.size(), 1);
        Base::m_bundle = mmeBundles.front();

        // A strategy to use mmeOut in SRAM
        TrivialSolver solver(*Base::m_graph.getHALReader(), Base::m_bundle);
        solver.createAllStrategies();
        ASSERT_FALSE(solver.getStrategies().empty());
        Base::m_strategy = std::static_pointer_cast<MmeSlicingStrategy>(solver.getStrategies().front());
        Base::m_strategy->getSlicingData().masterOperand->resideInSRAM = true;
        const auto& mmeOutTensor = Base::m_strategy->getSlicingData().masterOperand->originalTensor;
        const auto& consumerGranularity =
            Base::m_bn1->getNodeAccessPattern()->getTensorGranularity(mmeOutTensor).geometry;
        ASSERT_EQ(consumerGranularity.size(), mmeOutTensor->getDim());
        Base::m_strategy->getSlicingData().masterOperand->chunkDimensions = {consumerGranularity[0],
                                                                             consumerGranularity[1],
                                                                             1,
                                                                             1,
                                                                             1};

        // With the BN added to the strategy to consume the mmeOut from SRAM
        pBundleExpansion bundleExp = std::make_shared<BundleExpansion>();
        bundleExp->nodeToStitch    = Base::m_bn1;
        bundleExp->bundleNode      = Base::m_gemm;
        bundleExp->stitchedOperand = Base::m_strategy->getSlicingData().masterOperand;
        brains.m_tpcSlaveBrain->addConsumerToStrategy(bundleExp, Base::m_strategy);
        Base::m_bundle->addNode(Base::m_bn1);
    }

    virtual void fuseEviction() override
    {
        BatchNormStagesEvictionFuser fuser(Base::m_graph, Base::m_bundle, Base::m_strategy);
        ASSERT_NO_THROW(fuser.fuseEvictions());
    }

    virtual void validateFusion() override
    {
        // Expects that mmeOut will become an output from the BN node in the graph
        ASSERT_EQ(Base::m_bn1->getNumOutputs(), 2);
        EXPECT_EQ(Base::m_bn1->getOutput(1), Base::m_mmeOut);

        // Expect the GEMM to have a new WS output
        auto newMmeOutput = Base::m_gemm->getOutput(0);
        EXPECT_NE(newMmeOutput, Base::m_mmeOut);
        EXPECT_FALSE(newMmeOutput->isPersistent());

        // The master operand of the strategy is the new GEMM output
        EXPECT_EQ(Base::m_strategy->getMmeSlicingData().masterOperand->originalTensor, newMmeOutput);

        // And the strategy will (fwd)map the new intermediate mmeOut-copy to mmeOut slices.
        auto newMmeOutSlice = std::make_shared<SliceReference>(Base::m_strategy->getMmeSlicingData().masterOperand);
        // Get the last valid coordinate (to avoid assumptions on number of slices)
        for (auto i = 0; i < Base::m_strategy->getMmeSlicingData().masterOperand->originalTensor->getDim(); i++)
        {
            newMmeOutSlice->coordinates[i] =
                SlicedOperandUtils::nofSlices(Base::m_strategy->getMmeSlicingData().masterOperand, i) - 1;
        }
        // mapping returns list< pair< InputSliceReferencesList, OutputSliceReferencesList > >
        auto mappedSlices = Base::m_strategy->getSlicingData().getFwdMappedSlices(newMmeOutSlice);
        ASSERT_EQ(mappedSlices.size(), 1);
        SliceReferenceList inputSlices, outputSlices;
        std::tie(inputSlices, outputSlices) = mappedSlices.front();
        EXPECT_EQ(inputSlices.size(), 2);
        EXPECT_EQ(inputSlices.front()->operand->originalTensor, newMmeOutput);
        EXPECT_EQ(outputSlices.size(), 2);
        const auto& newSlicedOutput = outputSlices.back()->operand;
        EXPECT_EQ(newSlicedOutput->originalTensor, Base::m_mmeOut);
        EXPECT_FALSE(newSlicedOutput->resideInSRAM);
        EXPECT_EQ(outputSlices.back()->coordinates, newMmeOutSlice->coordinates);
    }

    virtual void validateNoFusion() override
    {
        // Expects no change in the operands or strategy
        ASSERT_EQ(Base::m_bn1->getNumOutputs(), 1);
        EXPECT_EQ(Base::m_gemm->getOutput(0), Base::m_mmeOut);
        EXPECT_EQ(Base::m_strategy->getMmeSlicingData().masterOperand->originalTensor, Base::m_mmeOut);
    }
};

template<class GraphType>
class BatchNormStage2FwdEvictionFuserTestCommon : public BatchNormEvictionFuserTest<GraphType>
{
public:
    BatchNormStage2FwdEvictionFuserTestCommon()  = default;
    ~BatchNormStage2FwdEvictionFuserTestCommon() = default;
    using Base =
        BatchNormEvictionFuserTest<GraphType>;  // needed using because the parent class is a template. Base Class
                                                // methods calls in Derived Class requires info about specific type.
    virtual void buildGraph(bool mmeOperandPersistent, bool addExternalConsumer) override
    {
        // Build:  BN_stage2 --> [mme_In] --> MME
        // mmeOut may be persistent depending on param 0 and may be memcpied depending on param 1

        ns_BatchNormStage2Kernel::ParamsV2 bnParams {};
        TensorPtr                          batchNormIfm         = Base::createPersistentTensor();
        TensorPtr                          meanTensor           = Base::createTensor({128});
        TensorPtr                          sigmasTensor         = Base::createTensor({128, 2});
        TensorPtr                          betaGammaTensor      = Base::createTensor({128, 2});
        TensorPtr                          RunningMeanVarTensor = Base::createTensor({128, 2});
        TensorPtr                          runningMeanVarTensor = Base::createTensor({128, 2});
        TensorPtr                          IstdTensor           = Base::createTensor({128, 2});

        TensorPtr kernel       = Base::createPersistentTensor();
        Base::m_mmeIn          = mmeOperandPersistent ? Base::createPersistentTensor() : Base::createTensor();
        TensorPtr bundleOutput = Base::createPersistentTensor();

        Base::m_bn2 =
            NodeFactory::createNode({batchNormIfm, meanTensor, sigmasTensor, betaGammaTensor, RunningMeanVarTensor},
                                    {Base::m_mmeIn, runningMeanVarTensor, IstdTensor},
                                    &bnParams,
                                    "batch_norm_stage2_fwd_f32",
                                    "bn2");
        ASSERT_TRUE(GraphEditor::addNode(Base::m_graph, Base::m_bn2));

        synGEMMParams gemmParams {};
        Base::m_gemm = NodeFactory::createNode({Base::m_mmeIn, kernel},
                                               {bundleOutput},
                                               &gemmParams,
                                               NodeFactory::gemmNodeTypeName,
                                               "gemm");
        ASSERT_TRUE(GraphEditor::addNode(Base::m_graph, Base::m_gemm));

        if (addExternalConsumer)
        {
            TensorPtr mmeInCopy = Base::createTensor();
            NodePtr   memcpy    = NodeFactory::createNode({Base::m_mmeIn},
                                                     {mmeInCopy},
                                                     nullptr,
                                                     NodeFactory::memcpyNodeTypeName,
                                                     "memcpy");
            ASSERT_TRUE(GraphEditor::addNode(Base::m_graph, memcpy));
        }
    }

    virtual void bundleGraph() override
    {
        AllBrains brains(Base::m_graph);

        // A bundle containing the GEMM
        Bundlizer  bundlizer(Base::m_graph);
        BundleList mmeBundles, dummy[4];
        bundlizer.generateBundles(mmeBundles, dummy[0], dummy[1], dummy[2], dummy[3]);
        ASSERT_EQ(mmeBundles.size(), 1);
        Base::m_bundle = mmeBundles.front();

        // A strategy to use mmeIn in SRAM
        TrivialSolver solver(*Base::m_graph.getHALReader(), Base::m_bundle);
        solver.createAllStrategies();
        ASSERT_FALSE(solver.getStrategies().empty());
        Base::m_strategy        = std::static_pointer_cast<MmeSlicingStrategy>(solver.getStrategies().front());
        auto m_mmeInOp          = Base::m_strategy->getSlicingData().getSlicedOperand(Base::m_mmeIn);
        m_mmeInOp->resideInSRAM = true;
        const auto& producerGranularity =
            Base::m_bn2->getNodeAccessPattern()->getTensorGranularity(Base::m_mmeIn).geometry;
        ASSERT_EQ(producerGranularity.size(), Base::m_mmeIn->getDim());
        m_mmeInOp->chunkDimensions = {producerGranularity[0], producerGranularity[1], 1, 1, 1};

        // With the BN added to the strategy to consume the mmeOut from SRAM
        pBundleExpansion bundleExp = std::make_shared<BundleExpansion>();
        bundleExp->nodeToStitch    = Base::m_bn2;
        bundleExp->bundleNode      = Base::m_gemm;
        bundleExp->stitchedOperand = Base::m_strategy->getSlicingData().getSlicedOperand(Base::m_bn2->getOutput(0));
        brains.m_tpcSlaveBrain->addProducerToStrategy(bundleExp, Base::m_strategy);
        Base::m_bundle->addNode(Base::m_bn2);
    }

    virtual void fuseEviction() override
    {
        BatchNormStagesEvictionFuser fuser(Base::m_graph, Base::m_bundle, Base::m_strategy);
        ASSERT_NO_THROW(fuser.fuseEvictions());
    }

    virtual void validateFusion() override
    {
        ASSERT_EQ(Base::m_bn2->getNumOutputs(), 4);

        // Expect the GEMM to have a new WS input
        auto newMmeInput = Base::m_gemm->getInput(0);
        EXPECT_EQ(Base::m_bn2->getOutput(0), newMmeInput);
        EXPECT_EQ(Base::m_bn2->getOutput(1), Base::m_mmeIn);
        EXPECT_NE(newMmeInput, Base::m_mmeIn);
        EXPECT_FALSE(newMmeInput->isPersistent());

        // check if the new tensor is in the BWD mapping of BundleOutput
        auto bundleOutputSlicedOperand =
            Base::m_strategy->getSlicingData().getSlicedOperand(Base::m_gemm->getOutput(0));
        std::list<pSlicedOperand> inputList, outputList;
        std::tie(inputList, outputList) =
            Base::m_strategy->getSlicingData().getBwdMappedSlicedOperands(bundleOutputSlicedOperand);
        const auto& bnSramOutput = Base::m_bn2->getOutput(0);
        auto        newTensorIsMapped =
            std::any_of(inputList.begin(), inputList.end(), [&bnSramOutput](const pSlicedOperand& so) {
                return (so->originalTensor == bnSramOutput);
            });
        EXPECT_EQ(newTensorIsMapped, true);

        // check if the new sliced operand resides in SRAM.
        const auto& BnSramOutputSlicedOperand =
            Base::m_strategy->getSlicingData().getSlicedOperand(Base::m_bn2->getOutput(0));
        ASSERT_TRUE(BnSramOutputSlicedOperand->resideInSRAM) << "BnOutputSlicedOperand is not in SRAM";

        // check if the old tensor (which is now in HBM) is still in the bundle.
        const auto& bundleTensors = Base::m_strategy->getSlicingData().bundleTensors;
        auto        oldTensorIsInBundle =
            std::any_of(bundleTensors.begin(), bundleTensors.end(), [&](const pSlicedOperand& so) {
                return (so->originalTensor == Base::m_mmeIn);
            });
        EXPECT_EQ(oldTensorIsInBundle, true);

        // And the strategy will (bwd)map the new intermediate mmeIn-copy to mmeIn slices.

        auto newMmeInSlice = std::make_shared<SliceReference>(BnSramOutputSlicedOperand);

        for (auto i = 0; i < Base::m_strategy->getMmeSlicingData().masterOperand->originalTensor->getDim(); i++)
        {
            newMmeInSlice->coordinates[i] =
                SlicedOperandUtils::nofSlices(Base::m_strategy->getMmeSlicingData().masterOperand, i) - 1;
        }

        SliceReferenceList inputSlices  = Base::m_strategy->getSlicingData().getInputsForSlice(newMmeInSlice);
        SliceReferenceList outputSlices = Base::m_strategy->getSlicingData().getOutputsForSlice(newMmeInSlice);
        EXPECT_EQ(inputSlices.size(), 5);  // BN2 has 5 inputs
        EXPECT_EQ(outputSlices.size(), 2);
        const auto& newSlicedInput = outputSlices.back()->operand;
        EXPECT_EQ(newSlicedInput->originalTensor, Base::m_mmeIn);
        EXPECT_FALSE(newSlicedInput->resideInSRAM);
        EXPECT_EQ(outputSlices.back()->coordinates, newMmeInSlice->coordinates);
    }

    virtual void validateNoFusion() override
    {
        // Expects no change in the operands or strategy
        ASSERT_EQ(Base::m_bn2->getNumOutputs(), 3);
        EXPECT_EQ(Base::m_bn2->getOutput(0), Base::m_mmeIn);
        EXPECT_EQ(Base::m_gemm->getInput(0), Base::m_mmeIn);
    }
};