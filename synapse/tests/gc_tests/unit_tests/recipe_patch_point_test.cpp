#include <graph_compiler/habana_nodes/node_factory.h>
#include "recipe_allocator.h"
#include "recipe_test_base.h"
#include "recipe.h"
#include "graph_compiler/recipe_patch_point.h"
#include "habana_global_conf.h"
#include "sim_graph.h"

void RecipeTestBase::patch_point_container()
{
    RecipePatchPointContainer   container;
    BasicFieldsContainerInfo    afci;
    uint32_t                    array[20] = {0};

    SizeArray size = {100, 100, 100, 2, 1};
    TensorPtr inT(new Tensor(4, size.data(), syn_type_bf16));
    TensorPtr outT(new Tensor(4, size.data(), syn_type_bf16));

    // Adding 2 dummy nodes to test the node execution index patch points association
    pNode dummyNode1 = std::make_shared<TPCNode>(TensorVector{inT}, TensorVector{outT}, "node1");
    dummyNode1->setExecutionOrderedIndex(0);
    pNode dummyNode2 = std::make_shared<TPCNode>(TensorVector{inT}, TensorVector{outT}, "node2");
    dummyNode2->setExecutionOrderedIndex(1);

    // Insert one FULL field
    afci.addAddressEngineFieldInfo(nullptr,
                            "FULL_at_index_3",
                            0,                     // memory id / section index
                            0,                     // virtual address
                            (uint32_t) 3,          // field index offset
                            FIELD_MEMORY_TYPE_DRAM);

    // Insert LOW and HIGH consecutive fields
    afci.addAddressEngineFieldInfo(dummyNode1,
                            "LOW_and_HIGH_at_index_7_and_8",
                            1,                      // memory id / section index
                            (uint64_t) &array[7],   // virtual address low
                            (uint64_t) &array[8],   // virtual address high
                            (uint64_t) &array[0],   // base
                            FIELD_MEMORY_TYPE_DRAM,
                            0);

    // Insert LOW and HIGH non-consecutive fields
    afci.addAddressEngineFieldInfo(dummyNode1,
                            "LOW_and_HIGH_at_index_10_and_12",
                            2,                      // memory id / section index
                            (uint64_t) &array[10],  // virtual address low
                            (uint64_t) &array[12],  // virtual address high
                            (uint64_t) &array[0],   // base
                            FIELD_MEMORY_TYPE_DRAM,
                            0);

    // Insert one FULL field
    afci.addAddressEngineFieldInfo(dummyNode2,
                            "FULL_at_index_15",
                            3,                      // memory id / section index
                            (uint64_t) &array[15],  // virtual address full
                            (uint64_t) &array[0],   // base
                            FIELD_MEMORY_TYPE_DRAM,
                            0);

    // We expect to have 5 patch-points:
    // 1. Full at offset 103
    // 2. Full at offset 107 - this is the optimization of low and high at 107 and 108 to a single full
    // 3. Low  at offset 110
    // 4. High at offset 112
    // 5. Full at offset 115

    // We should have 3 node execution indices: 0, 1, 2
    // patch point 0 - should have node index 0
    // patch point 1,2,4 - should have node index 1
    // patch point 3 - should have node index 2

    std::list<uint64_t> pps = container.insertPatchPoints(afci, 400); // 400 bytes are 100 32-bit elements
    ASSERT_EQ(pps.size(), 5);
    for (unsigned i = 0; i < pps.size(); i++)
    {
        container[i].setBlobIndex(7); // arbitrary index
    }

    uint32_t        numPatchPoints;
    uint32_t        numActivePatchPoints;
    patch_point_t*  pPatchPoints;

    uint32_t                     pNumSectionTypes;
    section_group_t*             pSectionTypesPatchPoints;
    std::map<uint64_t, uint8_t>  sectionIdToSectionType;
    section_blobs_t*             pSectionIdBlobIndices;
    section_group_t              sectionIdBlobIndices;
    uint32_t                     pNumSectionIds;
    RecipeAllocator              recipeAlloc;
    SimGraph                     g;
    const TensorSet              persistTensors;

    container.serialize(&numPatchPoints,
                        &numActivePatchPoints,
                        &pPatchPoints,
                        &pNumSectionTypes,
                        &pSectionTypesPatchPoints,
                        sectionIdToSectionType,
                        &pNumSectionIds,
                        &pSectionIdBlobIndices,
                        &sectionIdBlobIndices,
                        &recipeAlloc,
                        persistTensors);

    ASSERT_EQ(numPatchPoints, pps.size());
    ASSERT_EQ(pNumSectionTypes, 2);
    ASSERT_EQ(pSectionTypesPatchPoints[0].patch_points_nr, 4);  // default section type
    ASSERT_EQ(pSectionTypesPatchPoints[1].patch_points_nr, 1);  // program data section type

    for (unsigned i = 0; i < numPatchPoints; i++)
    {
        ASSERT_EQ(pPatchPoints[i].blob_idx, 7);

        if (pPatchPoints[i].memory_patch_point.section_idx == 0)
        {
            ASSERT_EQ(pPatchPoints[i].node_exe_index, 0);
            ASSERT_EQ(pPatchPoints[i].type, patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT);
            ASSERT_EQ(pPatchPoints[i].dw_offset_in_blob, 103);
        }
        else if (pPatchPoints[i].memory_patch_point.section_idx == 1)
        {
            ASSERT_EQ(pPatchPoints[i].node_exe_index, 1);
            ASSERT_EQ(pPatchPoints[i].type, patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT);
            ASSERT_EQ(pPatchPoints[i].dw_offset_in_blob, 107);
        }
        else if (pPatchPoints[i].memory_patch_point.section_idx == 2)
        {
            ASSERT_EQ(pPatchPoints[i].node_exe_index, 1);
            ASSERT_TRUE(pPatchPoints[i].type == patch_point_t::SIMPLE_DW_LOW_MEM_PATCH_POINT ||
                        pPatchPoints[i].type == patch_point_t::SIMPLE_DW_HIGH_MEM_PATCH_POINT);

            if (pPatchPoints[i].type == patch_point_t::SIMPLE_DW_LOW_MEM_PATCH_POINT)
            {
                ASSERT_EQ(pPatchPoints[i].dw_offset_in_blob, 110);
            }
            else
            {
                ASSERT_EQ(pPatchPoints[i].dw_offset_in_blob, 112);
            }
        }
        else if (pPatchPoints[i].memory_patch_point.section_idx == 3)
        {
            ASSERT_EQ(pPatchPoints[i].node_exe_index, 2);
            ASSERT_EQ(pPatchPoints[i].type, patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT);
            ASSERT_EQ(pPatchPoints[i].dw_offset_in_blob, 115);
        }
        else
        {
            ASSERT_TRUE(false) << "invalid section_idx";
        }
    }

    // validate section id blob indices - 4 sections - 1 blobs per section
    ASSERT_EQ(pNumSectionIds, 4);
    for (unsigned i = 0; i < pNumSectionIds; i++)
    {
        ASSERT_EQ(pSectionIdBlobIndices[i].blobs_nr, 1);
        ASSERT_EQ(pSectionIdBlobIndices[i].blob_indices[0], 7);
    }

    container.print();
}

void RecipeTestBase::section_type_patch_point_container()
{
    RecipePatchPointContainer   container;
    BasicFieldsContainerInfo    afci;
    uint32_t                    array[20] = {0};

    // Insert one FULL field
    afci.addAddressEngineFieldInfo(nullptr,
                                   "FULL_at_index_3",
                                   0,                     // memory id / section index
                                   0,                     // virtual address
                                   (uint32_t) 3,          // field index offset
                                   FIELD_MEMORY_TYPE_DRAM);

    // Insert LOW and HIGH consecutive fields
    afci.addAddressEngineFieldInfo(nullptr,
                                   "LOW_and_HIGH_at_index_7_and_8",
                                   1,                      // memory id / section index
                                   (uint64_t) &array[7],   // virtual address low
                                   (uint64_t) &array[8],   // virtual address high
                                   (uint64_t) &array[0],   // base
                                   FIELD_MEMORY_TYPE_DRAM,
                                   0);

    // Insert LOW and HIGH non-consecutive fields
    afci.addAddressEngineFieldInfo(nullptr,
                                   "LOW_and_HIGH_at_index_10_and_12",
                                   2,                      // memory id / section index
                                   (uint64_t) &array[10],  // virtual address low
                                   (uint64_t) &array[12],  // virtual address high
                                   (uint64_t) &array[0],   // base
                                   FIELD_MEMORY_TYPE_DRAM,
                                   0);

    // Insert one FULL field
    afci.addAddressEngineFieldInfo(nullptr,
                                   "FULL_at_index_15",
                                   3,                      // memory id / section index
                                   (uint64_t) &array[15],  // virtual address full
                                   (uint64_t) &array[0],   // base
                                   FIELD_MEMORY_TYPE_DRAM,
                                   0);

    afci.addAddressEngineFieldInfo(nullptr,
                                   "FULL_at_index_16",
                                   4,                      // memory id / section index
                                   (uint64_t) &array[15],  // virtual address full
                                   (uint64_t) &array[0],   // base
                                   FIELD_MEMORY_TYPE_DRAM,
                                   0);

    afci.addAddressEngineFieldInfo(nullptr,
                                   "FULL_at_index_17",
                                   5,                      // memory id / section index
                                   (uint64_t) &array[15],  // virtual address full
                                   (uint64_t) &array[0],   // base
                                   FIELD_MEMORY_TYPE_DRAM,
                                   0);
    afci.addAddressEngineFieldInfo(nullptr,
                                   "FULL_at_index_18",
                                   0,                      // memory id / section index
                                   (uint64_t) &array[15],  // virtual address full
                                   (uint64_t) &array[0],   // base
                                   FIELD_MEMORY_TYPE_DRAM,
                                   0);

    // We expect to have 8 patch-points:
    std::list<uint64_t> pps = container.insertPatchPoints(afci, 400); // 400 bytes are 100 32-bit elements
    ASSERT_EQ(pps.size(), 8);
    for (unsigned i = 0; i < pps.size(); i++)
    {
        container[i].setBlobIndex(7); // arbitrary index
    }

    uint32_t        numPatchPoints;
    uint32_t        numActivePatchPoints;
    patch_point_t*  pPatchPoints;

    uint32_t                     pNumSectionTypes;
    section_group_t*             pSectionTypesPatchPoints;
    section_group_t              sectionTypesPatchPoints;
    std::map<uint64_t, uint8_t>  sectionIdToSectionType;
    section_blobs_t*             pSectionIdBlobIndices;
    uint32_t                     pNumSectionIds;
    RecipeAllocator              recipeAlloc;
    SimGraph                     g;
    const TensorSet              persistTensors;

    sectionIdToSectionType[0] = 0;
    sectionIdToSectionType[1] = 1;
    sectionIdToSectionType[2] = 2;
    sectionIdToSectionType[3] = 3;
    sectionIdToSectionType[4] = 3;
    sectionIdToSectionType[5] = 0;

    container.serialize(&numPatchPoints,
                        &numActivePatchPoints,
                        &pPatchPoints,
                        &pNumSectionTypes,
                        &pSectionTypesPatchPoints,
                        sectionIdToSectionType,
                        &pNumSectionIds,
                        &pSectionIdBlobIndices,
                        &sectionTypesPatchPoints,
                        &recipeAlloc,
                        persistTensors);

    ASSERT_EQ(numPatchPoints, pps.size());
    ASSERT_EQ(pNumSectionTypes, 4);
    ASSERT_EQ(pSectionTypesPatchPoints[0].patch_points_nr, 3);  // default section type
    ASSERT_EQ(pSectionTypesPatchPoints[1].patch_points_nr, 1);  // program data section type
    ASSERT_EQ(pSectionTypesPatchPoints[2].patch_points_nr, 2);
    ASSERT_EQ(pSectionTypesPatchPoints[3].patch_points_nr, 2);

    // validate section id blob indices - 6 sections - 1 blobs per section
    ASSERT_EQ(pNumSectionIds, 6);
    for (unsigned i = 0; i < pNumSectionIds; i++)
    {
        ASSERT_EQ(pSectionIdBlobIndices[i].blobs_nr, 1);
        ASSERT_EQ(pSectionIdBlobIndices[i].blob_indices[0], 7);
    }

    container.print();
}
