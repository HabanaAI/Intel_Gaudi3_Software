#include <graph_compiler/habana_nodes/node_factory.h>
#include "recipe_allocator.h"
#include "infra/scoped_configuration_change.h"
#include "recipe_test_base.h"
#include "graph_compiler/command_queue.h"
#include "graph_compiler/recipe_patch_point.h"
#include "graph_compiler/recipe_blob.h"
#include "graph_compiler/recipe_program.h"
#include "graph_compiler/recipe_generator.h"
#include "graph_compiler/habana_graph.h"
#include "graph_compiler/graph_traits.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "section_handle.hpp"
#include "syn_singleton.hpp"

void RecipeTestBase::generator_basic(HabanaGraph *g)
{
    CompilationHalReaderSetter compHalReaderSetter(g);
    makeQueues(1, g);
    RecipeGenerator generator(g);
    generator.generateRecipes(false);
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = generator.serializeDataPlaneGraph(&recipeAlloc);

    uint blobAlginment = g->getHALReader()->getCpDmaAlignment(); // 32 for Gaudi

    ASSERT_TRUE(recipe != nullptr);

    // 20 commands divided by 4 blob terminators yield 4 blobs, 2 out of which are further divided to 2 blobs,
    // 1 patching and one executing, so total expected blobs are 6
    ASSERT_EQ(recipe->blobs_nr, 6);

    // The first blob has 5 non-patching wreg32 commands with a total size of 40 bytes (each wreg32 is 8 bytes)
    // we pad each blob to a multiple of blobAlginment, so the expected size of the first blob is 64
    ASSERT_EQ(recipe->blobs[0].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[0].size, 64);

    // The second blob has 1 patching wreg32 commands with a total size of 8 bytes
    // we pad each blob to a multiple of blobAlginment, so the expected size of the first blob is blobAlginment
    ASSERT_EQ(recipe->blobs[1].blob_type.requires_patching, true);
    ASSERT_EQ(recipe->blobs[1].size, blobAlginment);

    // The third blob has the remaining 4 non-patching wreg32 commands with a total size of 32 bytes
    ASSERT_EQ(recipe->blobs[2].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[2].size, blobAlginment);

    // The forth blob has 5 non-patching wreg32 commands with a total size of 40 padded to 64
    ASSERT_EQ(recipe->blobs[3].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[3].size, 64);

    // The fifth blob has 1 patching wreg32 commands with a total size of 8 bytes padded to 32
    ASSERT_EQ(recipe->blobs[4].blob_type.requires_patching, true);
    ASSERT_EQ(recipe->blobs[4].size, blobAlginment);

    // The sixth blob has the remaining 4 non-patching wreg32 commands with a total size of 32 bytes
    ASSERT_EQ(recipe->blobs[5].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[5].size, blobAlginment);

    // We should have one program containing the indices 0 to 5
    ASSERT_EQ(recipe->programs_nr, 1);
    ASSERT_EQ(recipe->programs[0].program_length, 6);
    for (size_t i = 0; i < 6; i++)
    {
        ASSERT_EQ(recipe->programs[0].blob_indices[i], i);
    }

    // We should have one job containing program index 0
    ASSERT_EQ(recipe->execute_jobs_nr, 1);
    ASSERT_EQ(recipe->execute_jobs[0].engine_id, m_queuePtrs[0]->GetQueueID());
    ASSERT_EQ(recipe->execute_jobs[0].program_idx, 0);

    // We should have 2 patch points, one that patch the command in blob index 1, and the other in blob index 4
    ASSERT_EQ(recipe->patch_points_nr, 2);
    ASSERT_EQ(recipe->patch_points[0].blob_idx, 1);
    ASSERT_EQ(recipe->patch_points[1].blob_idx, 4);

    // We should have 1 data blob
    if (g->getDeviceType() == synDeviceGaudi)
    {
        ASSERT_EQ(recipe->program_data_blobs_nr, 1);
        ASSERT_EQ(recipe->program_data_blobs[0].offset_in_section, 0);
        ASSERT_EQ(recipe->program_data_blobs[0].size, 1024);
        ASSERT_EQ(recipe->program_data_blobs[0].section_idx, MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);

        ASSERT_EQ(recipe->workspace_sizes[1], 1024);
    }

    // We should have 3 fixed workspaces sections (workspace, program data and program)
    ASSERT_EQ(recipe->sections_nr, 3);
    ASSERT_EQ(recipe->workspace_nr, 3);

    ASSERT_EQ(recipe->workspace_sizes[0], 0);

    // Counting all blob sizes we get 32 * 4 + 4 * blobAlginment
    ASSERT_EQ(recipe->workspace_sizes[2], 32 * 4 + 4 * blobAlginment);

    // Validate gc conf params

    ASSERT_GE(recipe->recipe_conf_nr, 5);
    ASSERT_GT(recipe->recipe_conf_params[0].conf_value, 0);
    ASSERT_GT(recipe->recipe_conf_params[1].conf_value, 1000);
    ASSERT_GE(recipe->recipe_conf_params[2].conf_value, 1);
    ASSERT_GE(recipe->recipe_conf_params[3].conf_value, 1);
    ASSERT_GE(recipe->recipe_conf_params[4].conf_value, 3);

    generator.print();

}

extern ConcurrentSlotMapAlloc<InternalSectionHandle> sectionHndlSlopMap;

void RecipeTestBase::const_sections(HabanaGraph* g)
{
    CompilationHalReaderSetter compHalReaderSetter(g);
    makeQueues(1, g);
    NodeList emptyNodeList;

    const TSize n     = 1;
    const TSize w1    = 3;
    const TSize h1    = 3;
    const TSize batch = 1;
    const TSize w2    = 12;
    const TSize h2    = 36;

    char in1[n * w1 * h1 * batch];
    char in2[n * w2 * h2 * batch];

    const TSize sizes1[] = {n, w1, h1, batch};
    const TSize sizes2[] = {n, w2, h2, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    auto sectionHandle1 = sectionHndlSlopMap.insert(0, 0);
    auto sectionHandle2 = sectionHndlSlopMap.insert(0, 0);
    auto sectionHandle3 = sectionHndlSlopMap.insert(0, 0);

    sectionHandle1.second->setConst(true);
    sectionHandle2.second->setConst(true);
    sectionHandle3.second->setConst(true);

    uint32_t mysectionId1 = g->getNextMemorySectionID(SectionIDGenerator::USER_ALLOCATED_SECTIONS);
    uint32_t mysectionId2 = g->getNextMemorySectionID(SectionIDGenerator::USER_ALLOCATED_SECTIONS);
    uint32_t mysectionId3 = g->getNextMemorySectionID(SectionIDGenerator::USER_ALLOCATED_SECTIONS);

    sectionHandle1.second->setIDAndLock(mysectionId1);
    sectionHandle2.second->setIDAndLock(mysectionId2);
    sectionHandle3.second->setIDAndLock(mysectionId3);

    pTensor IN1 = pTensor(new Tensor(4U, sizes1, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setTensorInSram();
    IN1->setMemoryDescriptor(persistentMemoryDesc);
    IN1->setSectionHandle(sectionHandle1.second);
    IN1->setMemorySectionID(mysectionId1);
    IN1->setMemorySectionOffset(0);

    pTensor IN2 = pTensor(new Tensor(4U, sizes1, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    IN2->setTensorInSram();
    IN2->setMemoryDescriptor(persistentMemoryDesc);
    IN2->setSectionHandle(sectionHandle1.second);
    IN2->setMemorySectionID(mysectionId1);
    IN2->setMemorySectionOffset(IN1->getTotalSizeInBytes());

    pTensor OUT1 = pTensor(new Tensor(4U, sizes1, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    OUT1->setTensorInSram();
    OUT1->setMemoryDescriptor(persistentMemoryDesc);
    OUT1->setSectionHandle(sectionHandle2.second);
    OUT1->setMemorySectionID(mysectionId2);
    OUT1->setMemorySectionOffset(0);

    pNode elu_i8_1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "elu_i8", "elu_i8_1");
    GraphEditor::addNode(*g, elu_i8_1);

    pTensor IN3 = pTensor(new Tensor(4U, sizes2, syn_type_float, reinterpret_cast<char*>(in2)));
    IN3->setName("in3");
    IN3->setTensorInSram();
    IN3->setMemoryDescriptor(persistentMemoryDesc);
    IN3->setSectionHandle(sectionHandle3.second);
    IN3->setMemorySectionID(mysectionId3);
    IN3->setMemorySectionOffset(0);

    pTensor IN4 = pTensor(new Tensor(4U, sizes2, syn_type_float, reinterpret_cast<char*>(in2)));
    IN4->setName("in4");
    IN4->setTensorInSram();
    IN4->setMemoryDescriptor(persistentMemoryDesc);
    IN4->setSectionHandle(sectionHandle3.second);
    IN4->setMemorySectionID(mysectionId3);
    IN4->setMemorySectionOffset(IN3->getTotalSizeInBytes());

    pTensor OUT2 = pTensor(new Tensor(4U, sizes2, syn_type_float, reinterpret_cast<char*>(in2)));
    OUT2->setName("out2");
    OUT2->setTensorInSram();
    OUT2->setMemoryDescriptor(persistentMemoryDesc);
    OUT2->setSectionHandle(sectionHandle3.second);
    OUT2->setMemorySectionID(mysectionId3);
    OUT2->setMemorySectionOffset(IN3->getTotalSizeInBytes() + IN4->getTotalSizeInBytes());

    pNode elu_i8_2 = NodeFactory::createGenericTPCNode({IN3, IN4}, {OUT2}, nullptr, "elu_i8", "elu_i8_2");
    GraphEditor::addNode(*g, elu_i8_2);

    RecipeGenerator generator(g);
    generator.generateRecipes(false);
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = generator.serializeDataPlaneGraph(&recipeAlloc);

    ASSERT_TRUE(recipe != nullptr);

    // Expecting 3 sections: tensors IN1 and IN2 in section1, OUT1 in section2 and IN3, IN4 and OUT2 in section3
    // Validate section sizes as well
    ASSERT_EQ(recipe->const_sections_nr, 3);
    ASSERT_EQ(recipe->const_sections[0].size, IN1->getTotalSizeInBytes() * 2);
    ASSERT_EQ(recipe->const_sections[0].section_idx, MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    ASSERT_EQ(recipe->const_sections[1].size, OUT1->getTotalSizeInBytes());
    ASSERT_EQ(recipe->const_sections[1].section_idx, MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    ASSERT_EQ(recipe->const_sections[2].size, IN3->getTotalSizeInBytes() * 3);
    ASSERT_EQ(recipe->const_sections[2].section_idx, MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
}

void RecipeTestBase::generator_node_exe(HabanaGraph *g)
{
    CompilationHalReaderSetter compHalReaderSetter(g);

    // add 5 nodes
    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = { n, w, h, batch };

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setTensorInSram();
    IN1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    IN2->setTensorInSram();
    IN2->setMemoryDescriptor(persistentMemoryDesc);
    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    IN3->setTensorInSram();
    IN3->setMemoryDescriptor(persistentMemoryDesc);
    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    IN4->setTensorInSram();
    IN4->setMemoryDescriptor(persistentMemoryDesc);
    pTensor IN5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN5->setName("in5");
    IN5->setTensorInSram();
    IN5->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    OUT1->setTensorInSram();
    OUT1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");
    OUT2->setTensorInSram();
    OUT2->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setTensorInSram();
    OUT3->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    OUT4->setTensorInSram();
    OUT4->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");
    OUT5->setTensorInSram();
    OUT5->setMemoryDescriptor(persistentMemoryDesc);

    pNode neg1 = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, "neg_fwd_f32", "neg1");
    GraphEditor::addNode(*g, neg1);
    pNode neg2 = NodeFactory::createGenericTPCNode({IN2}, {OUT2}, nullptr, "neg_fwd_f32", "neg2");
    GraphEditor::addNode(*g, neg2);
    pNode neg3 = NodeFactory::createGenericTPCNode({IN3}, {OUT3}, nullptr, "neg_fwd_f32", "neg3");
    GraphEditor::addNode(*g, neg3);
    pNode neg4 = NodeFactory::createGenericTPCNode({IN4}, {OUT4}, nullptr, "neg_fwd_f32", "neg4");
    GraphEditor::addNode(*g, neg4);
    pNode neg5 = NodeFactory::createGenericTPCNode({IN5}, {OUT5}, nullptr, "neg_fwd_f32", "neg5");
    GraphEditor::addNode(*g, neg5);

    makeQueues(1, g);
    RecipeGenerator generator(g);
    generator.generateRecipes(false);
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = generator.serializeDataPlaneGraph(&recipeAlloc);

    ASSERT_TRUE(recipe != nullptr);

    ASSERT_EQ(recipe->blobs_nr, 6);
    ASSERT_EQ(recipe->node_nr, 5);
    ASSERT_EQ(recipe->node_exe_list[4].patch_points_nr, 2);

    // verify that each blob count in node is greater or equal to the equivalent in previous node
    for (unsigned i=0; i < recipe->programs_nr; i++)
    {
        ASSERT_GE(recipe->node_exe_list[1].program_blobs_nr[i], recipe->node_exe_list[0].program_blobs_nr[i]);
        ASSERT_GE(recipe->node_exe_list[2].program_blobs_nr[i], recipe->node_exe_list[1].program_blobs_nr[i]);
        ASSERT_GE(recipe->node_exe_list[3].program_blobs_nr[i], recipe->node_exe_list[2].program_blobs_nr[i]);
        ASSERT_GE(recipe->node_exe_list[4].program_blobs_nr[i], recipe->node_exe_list[3].program_blobs_nr[i]);
    }

    // We should have one program containing the indices 0 to 5
    ASSERT_EQ(recipe->programs_nr, 1);
    ASSERT_EQ(recipe->programs[0].program_length, 6);
    for (size_t i = 0; i < 6; i++)
    {
        ASSERT_EQ(recipe->programs[0].blob_indices[i], i);
    }

    // verify blob count in last node for program[0] is 6
    ASSERT_EQ(recipe->node_exe_list[4].program_blobs_nr[0], 6);

    generator.print();

}

void RecipeTestBase::generator_continuous_blobs(HabanaGraph *g)
{
    CompilationHalReaderSetter compHalReaderSetter(g);
    makeQueues(1, g);
    RecipeGenerator generator(g);
    generator.generateRecipes(false);
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = generator.serializeDataPlaneGraph(&recipeAlloc);

    uint blobAlginment = g->getHALReader()->getCpDmaAlignment(); // 32 for Gaudi

    ASSERT_TRUE(recipe != nullptr);

    // 20 commands divided by 4 blob terminators yield 4 blobs, 2 out of which are further divided to 2 blobs,
    // 1 patching and one executing, so total expected blobs are 6
    ASSERT_EQ(recipe->blobs_nr, 6);

    // The first blob has 5 non-patching wreg32 commands with a total size of 40 bytes (each wreg32 is 8 bytes)
    // we pad each blob to a multiple of 32, so the expected size of the first blob is 64
    ASSERT_EQ(recipe->blobs[0].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[0].size, 64);

    // The second blob has 1 patching wreg32 commands with a total size of 8 bytes
    // we pad each blob to a multiple of 32, so the expected size of the first blob is blobAlginment
    ASSERT_EQ(recipe->blobs[1].blob_type.requires_patching, true);
    ASSERT_EQ(recipe->blobs[1].size, blobAlginment);

    // The third blob has the remaining 4 non-patching wreg32 commands with a total size of 32 bytes
    ASSERT_EQ(recipe->blobs[2].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[2].size, blobAlginment);

    // The forth blob has 5 non-patching wreg32 commands with a total size of 40 padded to 64
    ASSERT_EQ(recipe->blobs[3].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[3].size, 64);

    // The fifth blob has 1 patching wreg32 commands with a total size of 8 bytes padded to blobAlginment
    ASSERT_EQ(recipe->blobs[4].blob_type.requires_patching, true);
    ASSERT_EQ(recipe->blobs[4].size, blobAlginment);

    // The sixth blob has the remaining 4 non-patching wreg32 commands with a total size of 32 bytes
    ASSERT_EQ(recipe->blobs[5].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[5].size, blobAlginment);

    // verify patching and execution blobs are separate and continuous
    // This should't be done in case blobs chunks mechanism is enabled since blobs are not necessarily continuous
    bool blobsChunksForPatchingBlobsEnabled  = (PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES != 0);
    bool blobsChunksForExecutionBlobsEnabled = (EXECUTION_BLOBS_CHUNK_SIZE_IN_BYTES != 0);
    // blobs 0, 2, 3, 5 are execution blobs
    if (!blobsChunksForExecutionBlobsEnabled)
    {
        ASSERT_EQ((char *)recipe->blobs[0].data + recipe->blobs[0].size, recipe->blobs[2].data);
        ASSERT_EQ((char *)recipe->blobs[2].data + recipe->blobs[2].size, recipe->blobs[3].data);
        ASSERT_EQ((char *)recipe->blobs[3].data + recipe->blobs[3].size, recipe->blobs[5].data);
    }

    // blobs 1, 4 are execution blobs
    if (!blobsChunksForPatchingBlobsEnabled)
    {
        ASSERT_EQ((char *)recipe->blobs[1].data + recipe->blobs[1].size, recipe->blobs[4].data);
    }

    // We should have 1 data blob
    if (g->getDeviceType() == synDeviceGaudi)
    {
        ASSERT_EQ(recipe->program_data_blobs_nr, 1);
        ASSERT_EQ(recipe->program_data_blobs[0].offset_in_section, 0);
        ASSERT_EQ(recipe->program_data_blobs[0].size, 1024);
        ASSERT_EQ(recipe->program_data_blobs[0].section_idx, MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);

        ASSERT_EQ(recipe->workspace_sizes[1], 1024);
    }

    // We should have 3 fixed workspaces sections (workspace, program data and program)
    ASSERT_EQ(recipe->sections_nr, 3);
    ASSERT_EQ(recipe->workspace_nr, 3);

    ASSERT_EQ(recipe->workspace_sizes[0], 0);

    // Counting all blob sizes we get 2 * 64 + 4 * blobAlginment
    ASSERT_EQ(recipe->workspace_sizes[2], 2 * 64 + 4 * blobAlginment);

    generator.print();

}

void RecipeTestBase::generator_single_blob(HabanaGraph* g)
{
    RecipeGenerator generator(g);
    generator.generateRecipes(false);
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = generator.serializeDataPlaneGraph(&recipeAlloc);
    ASSERT_GT(recipe->blobs_nr, 0);
    ASSERT_GT(recipe->programs_nr, 0);
    ASSERT_GT(recipe->execute_jobs_nr, 0);
    ASSERT_EQ(recipe->activate_patch_points_nr, 0);
    ASSERT_EQ(recipe->persist_tensors_nr, 0);
    ASSERT_EQ(recipe->sections_nr, 3);
    ASSERT_EQ(recipe->workspace_nr, 3);
    ASSERT_EQ(recipe->workspace_sizes[0], 0);
    ASSERT_GT(recipe->workspace_sizes[2], 0);

    if (g->getDeviceType() == synDeviceGaudi)
    {
        ASSERT_EQ(recipe->program_data_blobs_nr, 1);
        ASSERT_EQ(recipe->workspace_sizes[1], 1024);
        ASSERT_EQ(recipe->patch_points_nr, 8);
    }
}

void RecipeTestBase::generator_two_queues(HabanaGraph *g)
{
    CompilationHalReaderSetter compHalReaderSetter(g);
    makeQueues(2, g);
    RecipeGenerator generator(g);
    generator.generateRecipes(false);
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = generator.serializeDataPlaneGraph(&recipeAlloc);

    ASSERT_TRUE(recipe != nullptr);

    ASSERT_EQ(recipe->blobs_nr, 12);

    // We should have two programs containing the indices 0 to 5 and 6 to 11 respectively
    ASSERT_EQ(recipe->programs_nr, 2);
    ASSERT_EQ(recipe->programs[0].program_length, 6);
    for (size_t i = 0; i < 6; i++)
    {
        ASSERT_EQ(recipe->programs[0].blob_indices[i], i);
    }
    ASSERT_EQ(recipe->programs[1].program_length, 6);
    for (size_t i = 0; i < 6; i++)
    {
        ASSERT_EQ(recipe->programs[1].blob_indices[i], i+6);
    }

    // We should have two jobs containing program index 0 and 1
    ASSERT_EQ(recipe->execute_jobs_nr, 2);
    ASSERT_EQ(recipe->execute_jobs[0].engine_id, m_queuePtrs[0]->GetQueueID());
    ASSERT_EQ(recipe->execute_jobs[0].program_idx, 0);
    ASSERT_EQ(recipe->execute_jobs[1].engine_id, m_queuePtrs[1]->GetQueueID());
    ASSERT_EQ(recipe->execute_jobs[1].program_idx, 1);

    // We should have 4 patch points
    ASSERT_EQ(recipe->patch_points_nr, 4);
    ASSERT_EQ(recipe->patch_points[0].blob_idx, 1);
    ASSERT_EQ(recipe->patch_points[1].blob_idx, 4);
    ASSERT_EQ(recipe->patch_points[2].blob_idx, 7);
    ASSERT_EQ(recipe->patch_points[3].blob_idx, 10);

    uint blobAlginment = g->getHALReader()->getCpDmaAlignment(); // 32 for Gaudi

    // Counting all blob sizes we get 4 * 64 + 8 * blobAlginment
    ASSERT_EQ(recipe->workspace_sizes[2], 4 * 64 + 8 * blobAlginment);

}

void RecipeTestBase::generator_activate_execute_jobs(HabanaGraph *g)
{
    CompilationHalReaderSetter compHalReaderSetter(g);

    makeQueues(1, g, false, false, {1, 3});
    RecipeGenerator generator(g);
    generator.generateRecipes(false);
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = generator.serializeDataPlaneGraph(&recipeAlloc);

    ASSERT_TRUE(recipe != nullptr);

    /*
    1,3 -            patching (activate stage)
    0,2,4 -          non patching ended by blob terminator
    7 -              patching
    5,6,8,9 -        non patching ended by blob terminator

    10,11,12,13,14 - non patching ended by blob terminator
    15,17,18,19 -    patching ended by blob terminator
    16 -             patching
    */

    // 20 commands divided by 4 blob terminators yield 4 blobs, 2 out of which are further divided to 2 blobs,
    // 1 patching and one executing, so total expected blobs are 6
    // because 2 commands from the first 5 are activate we expect 1 more blob, so total of 7
    ASSERT_EQ(recipe->blobs_nr, 7);

    // The first blob has 2 non-patching wreg32 commands with a total size of 16 bytes (each wreg32 is 8 bytes)
    // we pad each blob to a multiple of 32, so the expected size of the first blob is 32
    ASSERT_EQ(recipe->blobs[0].blob_type.requires_patching, true);
    ASSERT_EQ(recipe->blobs[0].size, 32);

    // The second blob has 3 non-patching wreg32 commands with a total size of 24 bytes (each wreg32 is 8 bytes)
    // we pad each blob to a multiple of 32, so the expected size of the first blob is 32
    ASSERT_EQ(recipe->blobs[1].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[1].size, 32);

    // The third blob has 1 patching wreg32 commands with a total size of 32 bytes
    // we pad each blob to a multiple of 32, so the expected size of the first blob is 32
    ASSERT_EQ(recipe->blobs[2].blob_type.requires_patching, true);
    ASSERT_EQ(recipe->blobs[2].size, 32);

    // The fourth blob has the remaining 4 non-patching wreg32 commands with a total size of 32 bytes
    ASSERT_EQ(recipe->blobs[3].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[3].size, 32);

    // The fifth blob has 5 non-patching wreg32 commands with a total size of 40 padded to 64
    ASSERT_EQ(recipe->blobs[4].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[4].size, 64);

    // The sixth blob has 1 patching wreg32 commands with a total size of 8 bytes padded to 32
    ASSERT_EQ(recipe->blobs[5].blob_type.requires_patching, true);
    ASSERT_EQ(recipe->blobs[5].size, 32);

    // The seventh blob has the remaining 4 non-patching wreg32 commands with a total size of 32 bytes
    ASSERT_EQ(recipe->blobs[6].blob_type.requires_patching, false);
    ASSERT_EQ(recipe->blobs[6].size, 32);

    // We should have two programs, one index 0, and one containing the indices 1 to 6
    ASSERT_EQ(recipe->programs_nr, 2);

    ASSERT_EQ(recipe->programs[0].program_length, 1);
    ASSERT_EQ(recipe->programs[0].blob_indices[0], 0);

    ASSERT_EQ(recipe->programs[1].program_length, 6);
    for (size_t i = 1; i <= 6; i++)
    {
        ASSERT_EQ(recipe->programs[1].blob_indices[i-1], i);
    }

    // We should have one activate job containing program index 0
    ASSERT_EQ(recipe->activate_jobs_nr, 1);
    ASSERT_EQ(recipe->activate_jobs[0].engine_id, m_queuePtrs[0]->GetQueueID());
    ASSERT_EQ(recipe->activate_jobs[0].program_idx, 0);

    // We should have one execution job containing program index 1
    ASSERT_EQ(recipe->execute_jobs_nr, 1);
    ASSERT_EQ(recipe->execute_jobs[0].engine_id, m_queuePtrs[0]->GetQueueID());
    ASSERT_EQ(recipe->execute_jobs[0].program_idx, 1);

    // We should have 2 patch points, one that patch the command in blob index 2, and the other in blob index 5
    ASSERT_EQ(recipe->patch_points_nr, 2);
    ASSERT_EQ(recipe->patch_points[0].blob_idx, 2);
    ASSERT_EQ(recipe->patch_points[1].blob_idx, 5);

    // We should have 1 data blob
    ASSERT_EQ(recipe->program_data_blobs_nr, 1);
    ASSERT_EQ(recipe->program_data_blobs[0].offset_in_section, 0);
    ASSERT_EQ(recipe->program_data_blobs[0].size, 1024);
    ASSERT_EQ(recipe->program_data_blobs[0].section_idx, MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);

    // We should have 3 fixed workspaces sections (workspace, program data and program)
    ASSERT_EQ(recipe->sections_nr, 3);
    ASSERT_EQ(recipe->workspace_nr, 3);

    ASSERT_EQ(recipe->workspace_sizes[0], 0);

    ASSERT_EQ(recipe->workspace_sizes[1], 1024);

    // Counting all blob sizes we get 256
    ASSERT_EQ(recipe->workspace_sizes[2], 256);

    generator.print();

}

