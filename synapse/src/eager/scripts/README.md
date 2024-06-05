This folder contains scripts used during `Eager` development.

These are ad hoc scripts meant for internal usage and may break or be out of date.

Setup
=====

To enable it, if you don't have the `~/.gdbinit` file, create it. Add the following to this file,

    python
    import os
    SYNAPSE_ROOT = os.environ['SYNAPSE_ROOT']
    if SYNAPSE_ROOT is not None:
        gdb.execute(f'source {SYNAPSE_ROOT}/src/eager/scripts/gdb.eager_recipe_dump.py')
        # gdb.execute(f'source {SYNAPSE_ROOT}/src/eager/scripts/gdb.pprint.llvm.py')
        # gdb.execute(f'source {SYNAPSE_ROOT}/src/eager/scripts/gdb.pprint.nlohmann.py')
    end

gdb.eager_recipe_dump.py
==================

Used to dump `recipe_t` contents while debugging.

Usage
-----

To use it to dump a `recipe_t* recipe`, either

* Run `eager_recipe_dump <recipe_t*>` in `gdb` or
* If you're using `VSCode` while deubbing, use the `DEBUG CONSOLE` to run `-exec eager_recipe_dump <recipe_t*>` (The `-exec` is used to pass commands to the `gdb` backend)

Example
-------

We often break in `recipe_manager.cpp:338` right after the `recipe_t` is generated,

        basicRecipeHandle.recipe = graph->serializeDataPlane(basicRecipeHandle.recipeAllocator);
    ->  if (basicRecipeHandle.recipe == nullptr)
        {

And running the command in `VSCode` we get,

    -exec eager_recipe_dump basicRecipeHandle.recipe

    /===========================================================================\
    |                                                                           |
    |                                                                           |
    |       /===========================================================\       |
    |       |                                                           |       |
    |       |                     recipe_t content                      |       |
    |       |                                                           |       |
    |       \===========================================================/       |
    |                                                                           |
    |                                                                           |
    \===========================================================================/

    uint32_t                 version_major = 1
    uint32_t                 version_minor = 26

    uint32_t                 nameSize = 22
    char *                   name = 0x6030029a8620 "transpose_bf16.recipe"
        0x6030029a8620 "transpose_bf16.recipe"

    uint64_t                 execution_blobs_buffer_size = 320
    uint64_t *               execution_blobs_buffer = 0x7fff5f164000

        0x7fff5f164000:	0x81a2c80000000001	0x0800000080010001
        0x7fff5f164010:	0x92b8d80000000000	0x92b8e02000000000
        0x7fff5f164020:	0x81b8100000000000	0x82b868000000000e
        0x7fff5f164030:	0x0000000000000080	0x0000002100000000
        0x7fff5f164040:	0x0000008000000002	0x0000000100000002
        0x7fff5f164050:	0x0000000100000080	0x0000000100000080
        0x7fff5f164060:	0x0000000100000100	0x0000000100004000
        0x7fff5f164070:	0x0000000200000100	0x0000000100000080
        0x7fff5f164080:	0x0000000000004000	0x8000000100000000
        0x7fff5f164090:	0x0000000000000000	0x0000000000000000
        0x7fff5f1640a0:	0xc1b8e80000000080	0x81b8100000000000
        0x7fff5f1640b0:	0x82b8680000000010	0x0000000000000000
        0x7fff5f1640c0:	0x0000000000000000	0x0000000000000000
        0x7fff5f1640d0:	0x0000000000000000	0x0000000000000000
        0x7fff5f1640e0:	0x0000000000000000	0x0000000000000000
        0x7fff5f1640f0:	0x0000000000000000	0x0000000000000000
        0x7fff5f164100:	0x0000000000000000	0x0000000000000000
        0x7fff5f164110:	0x8000000100000000	0x0000000000000000
        0x7fff5f164120:	0x0000000000000000	0x00ffffff00000000
        0x7fff5f164130:	0x00ffffff00000000	0xc1b8e80000000000

    uint64_t                 patching_blobs_buffer_size = 24
    uint64_t *               patching_blobs_buffer = 0x7fff4703e000

        0x7fff4703e000:	0x82a9000000000002	0x0000040000000000
        0x7fff4703e010:	0x0000050000000000

    uint64_t                 dynamic_blobs_buffer_size = 156
    uint32_t *               dynamic_blobs_buffer = 0x7fff4703d000

        0x7fff4703d000:	0x00000201	0x0000000d	0x00000000	0x00000000
        0x7fff4703d010:	0x00000000	0x00000000	0x00000000	0x00000000
        0x7fff4703d020:	0x00000000	0x00000000	0x00000000	0x00000000
        0x7fff4703d030:	0x00000000	0x00000000	0x00000000	0x00000000
        0x7fff4703d040:	0x00000000	0x00000000	0x00000000	0x00000000
        0x7fff4703d050:	0x00000000	0x00000000	0x00000000	0x00000000
        0x7fff4703d060:	0x00000000	0x00000000	0x00000000	0x00000000
        0x7fff4703d070:	0x00000000	0x00000000	0x00000000	0x00000000
        0x7fff4703d080:	0x00000000	0x00000000	0x00000000	0x00000000
        0x7fff4703d090:	0x00000000	0x00000000	0x00000000


    uint64_t                 blobs_nr = 5
    blob_t *                 blobs = 0x60c00011d940

        #000: {unique_key = 5732602973900058883, {blob_type = {requires_patching = 1, static_exe = 0, dynamic_exe = 0, reserved = 0}, blob_type_all = blob_t::PATCHING, block_index = 1}, size = 24, {data = 0x7fff4703e000, offset_in_block = 1191436288}}
        #001: {unique_key = 13625687684486440965, {blob_type = {requires_patching = 0, static_exe = 1, dynamic_exe = 0, reserved = 0}, blob_type_all = blob_t::EXE, block_index = 2}, size = 16, {data = 0x7fff5f164000, offset_in_block = 1595293696}}
        #002: {unique_key = 10322926966301814398, {blob_type = {requires_patching = 0, static_exe = 1, dynamic_exe = 0, reserved = 0}, blob_type_all = blob_t::EXE, block_index = 2}, size = 152, {data = 0x7fff5f164010, offset_in_block = 1595293712}}
        #003: {unique_key = 7230461151160335494, {blob_type = {requires_patching = 0, static_exe = 0, dynamic_exe = 1, reserved = 0}, blob_type_all = blob_t::DYNAMIC, block_index = 4}, size = 156, {data = 0x7fff4703d000, offset_in_block = 1191432192}}
        #004: {unique_key = 3428154159411768589, {blob_type = {requires_patching = 0, static_exe = 1, dynamic_exe = 0, reserved = 0}, blob_type_all = blob_t::EXE, block_index = 2}, size = 152, {data = 0x7fff5f1640a8, offset_in_block = 1595293864}}

    uint32_t                 programs_nr = 5
    program_t *              programs = 0x6070006f59e0

        #000: {blob_indices = 0x6030029a82f0, program_length = 4, reserved = "\276\276\276\276"}
            0x6030029a82f0:	0x0000000000000000	0x0000000000000001
            0x6030029a8300:	0x0000000000000002	0x0000000000000003
        #001: {blob_indices = 0x6030029a8320, program_length = 4, reserved = "\276\276\276\276"}
            0x6030029a8320:	0x0000000000000000	0x0000000000000001
            0x6030029a8330:	0x0000000000000004	0x0000000000000003
        #002: {blob_indices = 0x6030029a8350, program_length = 4, reserved = "\276\276\276\276"}
            0x6030029a8350:	0x0000000000000000	0x0000000000000001
            0x6030029a8360:	0x0000000000000004	0x0000000000000003
        #003: {blob_indices = 0x6030029a8380, program_length = 4, reserved = "\276\276\276\276"}
            0x6030029a8380:	0x0000000000000000	0x0000000000000001
            0x6030029a8390:	0x0000000000000004	0x0000000000000003
        #004: {blob_indices = 0x6030029a83b0, program_length = 4, reserved = "\276\276\276\276"}
            0x6030029a83b0:	0x0000000000000000	0x0000000000000001
            0x6030029a83c0:	0x0000000000000004	0x0000000000000003

    uint32_t                 activate_jobs_nr = 0
    job_t *                  activate_jobs = 0x0


    uint32_t                 arc_jobs_nr = 1
    arc_job_t *              arc_jobs = 0x604000764a10

        #000: {static_ecb = {cmds_size = 1280, cmds_eng_offset = 256, cmds = 0x7fff4703c000 " "}, dynamic_ecb = {cmds_size = 256, cmds_eng_offset = 0, cmds = 0x7fff4703b000 " "}, engines_filter = 0, logical_engine_id = Recipe::DMA, reserved = "\276\276\276"}

            static_ecb:
            0x7fff4703c000:	0x00010020	0x00180004	0x00000000	0x20100004
            0x7fff4703c010:	0x00000000	0x20980014	0x00000010	0x00000101
            0x7fff4703c020:	0x00006e01	0x00000000	0x00000000	0x00000000
            0x7fff4703c030:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c040:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c050:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c060:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c070:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c080:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c090:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c0a0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c0b0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c0c0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c0d0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c0e0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c0f0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c100:	0x00010020	0x00180104	0x00000000	0x20100104
            0x7fff4703c110:	0x00000000	0x20980114	0x000000a8	0x00000101
            0x7fff4703c120:	0x00006e01	0x00000000	0x00000000	0x00000000
            0x7fff4703c130:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c140:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c150:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c160:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c170:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c180:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c190:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c1a0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c1b0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c1c0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c1d0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c1e0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c1f0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c200:	0x00010020	0x00180204	0x00000000	0x20100204
            0x7fff4703c210:	0x00000000	0x20980214	0x000000a8	0x00000101
            0x7fff4703c220:	0x00006e01	0x00000000	0x00000000	0x00000000
            0x7fff4703c230:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c240:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c250:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c260:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c270:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c280:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c290:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c2a0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c2b0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c2c0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c2d0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c2e0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c2f0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c300:	0x00010020	0x00180304	0x00000000	0x20100304
            0x7fff4703c310:	0x00000000	0x20980314	0x000000a8	0x00000101
            0x7fff4703c320:	0x00006e01	0x00000000	0x00000000	0x00000000
            0x7fff4703c330:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c340:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c350:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c360:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c370:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c380:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c390:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c3a0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c3b0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c3c0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c3d0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c3e0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c3f0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c400:	0x00010020	0x00180404	0x00000000	0x20100404
            0x7fff4703c410:	0x00000000	0x20980414	0x000000a8	0x00000101
            0x7fff4703c420:	0x00006e01	0x00000000	0x00000000	0x00000000
            0x7fff4703c430:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c440:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c450:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c460:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c470:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c480:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c490:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c4a0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c4b0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c4c0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c4d0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c4e0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703c4f0:	0x00000000	0x00000000	0x00000000	0x00000000

            dynamic_ecb:
            0x7fff4703b000:	0x00010020	0x00000101	0x00009c53	0x00000000
            0x7fff4703b010:	0x00000032	0x00007401	0x00000000	0x00000000
            0x7fff4703b020:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b030:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b040:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b050:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b060:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b070:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b080:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b090:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b0a0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b0b0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b0c0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b0d0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b0e0:	0x00000000	0x00000000	0x00000000	0x00000000
            0x7fff4703b0f0:	0x00000000	0x00000000	0x00000000	0x00000000

    uint32_t                 persist_tensors_nr = 2
    persist_tensor_info_t *  tensors = 0x614000c3b440

        #000: {name = 0x6030029a8590 "autoGenPersistInputTensorName_0", layout = 0x0, offset_in_section = 0, size = 256, zp = 0, scale = 1, multi_views_indices_nr = 0, multi_views_indices = 0x0, section_idx = 4, elementType = 2, dimensions = 4, dimensionsSize = {1, 128, 1 <repeats 23 times>}, batchSize = 0, tensorType = 0, extTensorExeOrder = 4294967295, permutation = "\000\001\002\003\004\005\006\a\b\t\n\v\f\r\016\017\020\021\022\023\024\025\026\027\030", section_type = 0 '\000', isInput = true, isExternal = false}
        #001: {name = 0x604000764990 "autoGenPersistOutputTensorName_1", layout = 0x0, offset_in_section = 0, size = 256, zp = 0, scale = 1, multi_views_indices_nr = 0, multi_views_indices = 0x0, section_idx = 5, elementType = 2, dimensions = 4, dimensionsSize = {128, 1 <repeats 24 times>}, batchSize = 0, tensorType = 0, extTensorExeOrder = 4294967295, permutation = "\000\001\002\003\004\005\006\a\b\t\n\v\f\r\016\017\020\021\022\023\024\025\026\027\030", section_type = 0 '\000', isInput = false, isExternal = false}

    uint64_t                 program_data_blobs_size = 0
    char *                   program_data_blobs_buffer = 0x0

    program_data_blobs_buffer sha1: da39a3ee5e6b4b0d3255bfef95601890afd80709

    uint32_t                 program_data_blobs_nr = 0
    program_data_blob_t *    program_data_blobs = 0x0


    uint32_t                 patch_points_nr = 2
    patch_point_t *          patch_points = 0x60600074de00

        000: {blob_idx = 0, dw_offset_in_blob = 2, {memory_patch_point = {effective_address = 0, section_idx = 4}, sob_patch_point = {tensor_db_index = 0}}, node_exe_index = 1, type = patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT, reserved = "\276\276\276"}
        001: {blob_idx = 0, dw_offset_in_blob = 4, {memory_patch_point = {effective_address = 0, section_idx = 5}, sob_patch_point = {tensor_db_index = 0}}, node_exe_index = 1, type = patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT, reserved = "\276\276\276"}

    uint32_t                 sections_nr = 5

    uint32_t                 section_groups_nr = 1
    section_group_t *        section_groups_patch_points = 0x6020000caa70

        #000: {patch_points_index_list = 0x6020000caa90, patch_points_nr = 2, section_group = 0 '\000', reserved = "\276\276\276"}
            0x6020000caa90:	0x00000000	0x00000001

    section_group_t          sobj_section_group_patch_points = {patch_points_index_list = 0x0, patch_points_nr = 0, section_group = 255 '\377', reserved = "\000\000"}



    uint32_t                 section_ids_nr = 2
    section_blobs_t *        section_blobs_indices = 0x6030029a8410

        #000: {blob_indices = 0x6020000caa30, section_idx = 4, blobs_nr = 1}
            0x6020000caa30:	0x00000000
        #001: {blob_indices = 0x6020000caa50, section_idx = 5, blobs_nr = 1}
            0x6020000caa50:	0x00000000

    uint32_t                 node_nr = 1
    node_program_t *         node_exe_list = 0x6020000caaf0

    #000: {program_blobs_nr = 0x6030029a8560, patch_points_nr = 2, reserved = "\276\276\276\276"}
            0x6030029a8560:	0x00000004	0x00000004

    uint32_t                 workspace_nr = 3
    uint64_t *               workspace_sizes = 0x6030029a85c0

        0x6030029a85c0:	0x0000000000000000	0x0000000000000000
        0x6030029a85d0:	0x0000000000000200

    debug_info_t             debug_profiler_info = {version_major = 1, version_minor = 1, recipe_id = 48830, num_nodes = 0, nodes = 0x0, printf_addr_nr = 0, printf_addr = 0x0, printf_section_idx = 0}

        WARN: stuff within debug info not printed recursively...

    uint32_t                 recipe_conf_nr = 5
    gc_conf_t *              recipe_conf_params = 0x6070006f5a50

        #000: {conf_id = gc_conf_t::DEVICE_TYPE, conf_value = 4}
        #001: {conf_id = gc_conf_t::TIME_STAMP, conf_value = 1658436150}
        #002: {conf_id = gc_conf_t::TPC_ENGINE_MASK, conf_value = 16777215}
        #003: {conf_id = gc_conf_t::MME_NUM_OF_ENGINES, conf_value = 2}
        #004: {conf_id = gc_conf_t::DMA_ENGINE_MASK, conf_value = 31}

    uint64_t                 nop_kernel_offset = 0
    uint64_t                 nop_kernel_section = 0
    bool                     valid_nop_kernel = false


    /===========================================================================\
    |                                                                           |
    |                                                                           |
    |       /===========================================================\       |
    |       |                                                           |       |
    |       |           static + patching command buffer decode         |       |
    |       |                                                           |       |
    |       \===========================================================/       |
    |                                                                           |
    |                                                                           |
    \===========================================================================/


    /===========================================================\
    |                                                           |
    |                       Blob #000000                        |
    |                                                           |
    \===========================================================/

    blob_t                   [0] = {unique_key = 5732602973900058883, {blob_type = {requires_patching = 1, static_exe = 0, dynamic_exe = 0, reserved = 0}, blob_type_all = blob_t::PATCHING, block_index = 1}, size = 24, {data = 0x7fff4703e000, offset_in_block = 1191436288}}

    #  0 0x82a9000000000002	 MB: 0x1, SWTC: 0x0, EB: 0x0, OP: 0x2  (WREG BULK      ), REG_OFFSET: 0xa900, PRED: 0x0 , SIZE64: 0x2   
      0x0000040000000000  0x0000050000000000

    wreg32 and wreg_bulk assignments:
    0xa900 = ['0x0'], 0xa904 = ['0x400'], 0xa908 = ['0x0'], 0xa90c = ['0x500']

    wreg32 and wreg_bulk sorted assignments:
    0xa900 = ['0x0']
    0xa904 = ['0x400']
    0xa908 = ['0x0']
    0xa90c = ['0x500']


    /===========================================================\
    |                                                           |
    |                       Blob #000001                        |
    |                                                           |
    \===========================================================/

    blob_t                   [1] = {unique_key = 13625687684486440965, {blob_type = {requires_patching = 0, static_exe = 1, dynamic_exe = 0, reserved = 0}, blob_type_all = blob_t::EXE, block_index = 2}, size = 16, {data = 0x7fff5f164000, offset_in_block = 1595293696}}

    #  0 0x81a2c80000000001	 MB: 0x1, SWTC: 0x0, EB: 0x0, OP: 0x1  (WREG 32        ), REG_OFFSET: 0xa2c8, RSV: 0x0, PRED: 0x0 , VALUE: 0x1       
    #  1 0x0800000080010001	 MB: 0x0, SWTC: 0x0, EB: 0x0, OP: 0x8  (FENCE          ), PRED: 0x0 , ID: 0x2, TARGET_VAL: 0x1   , DEC_VAL: 0x1

    wreg32 and wreg_bulk assignments:
    0xa2c8 = ['0x1']

    wreg32 and wreg_bulk sorted assignments:
    0xa2c8 = ['0x1']


    /===========================================================\
    |                                                           |
    |                       Blob #000002                        |
    |                                                           |
    \===========================================================/

    blob_t                   [2] = {unique_key = 10322926966301814398, {blob_type = {requires_patching = 0, static_exe = 1, dynamic_exe = 0, reserved = 0}, blob_type_all = blob_t::EXE, block_index = 2}, size = 152, {data = 0x7fff5f164010, offset_in_block = 1595293712}}

    #  0 0x92b8d80000000000	 MB: 0x1, SWTC: 0x0, EB: 0x0, OP: 0x12 (WREG_64_SHORT  ), DREG_OFFSET: 0x2e36, BASE: 0x0, PRED: 0x0 , OFFSET: 0x0       
    #  1 0x92b8e02000000000	 MB: 0x1, SWTC: 0x0, EB: 0x0, OP: 0x12 (WREG_64_SHORT  ), DREG_OFFSET: 0x2e38, BASE: 0x1, PRED: 0x0 , OFFSET: 0x0       
    #  2 0x81b8100000000000	 MB: 0x1, SWTC: 0x0, EB: 0x0, OP: 0x1  (WREG 32        ), REG_OFFSET: 0xb810, RSV: 0x0, PRED: 0x0 , VALUE: 0x0       
    #  3 0x82b868000000000e	 MB: 0x1, SWTC: 0x0, EB: 0x0, OP: 0x2  (WREG BULK      ), REG_OFFSET: 0xb868, PRED: 0x0 , SIZE64: 0xe   
      0x0000000000000080  0x0000002100000000  0x0000008000000002  0x0000000100000002
      0x0000000100000080  0x0000000100000080  0x0000000100000100  0x0000000100004000
      0x0000000200000100  0x0000000100000080  0x0000000000004000  0x8000000100000000
      0x0000000000000000  0x0000000000000000
    # 18 0xc1b8e80000000080	 MB: 0x1, SWTC: 0x1, EB: 0x0, OP: 0x1  (WREG 32        ), REG_OFFSET: 0xb8e8, RSV: 0x0, PRED: 0x0 , VALUE: 0x80      

    wreg32 and wreg_bulk assignments:
    0xb810 = ['0x0'], 0xb868 = ['0x80'], 0xb86c = ['0x0'], 0xb870 = ['0x0'], 0xb874 = ['0x21'], 0xb878 = ['0x2'], 0xb87c = ['0x80'], 0xb880 = ['0x2'], 0xb884 = ['0x1'], 0xb888 = ['0x80'], 0xb88c = ['0x1'], 0xb890 = ['0x80'], 0xb894 = ['0x1'], 0xb898 = ['0x100'], 0xb89c = ['0x1'], 0xb8a0 = ['0x4000'], 0xb8a4 = ['0x1'], 0xb8a8 = ['0x100'], 0xb8ac = ['0x2'], 0xb8b0 = ['0x80'], 0xb8b4 = ['0x1'], 0xb8b8 = ['0x4000'], 0xb8bc = ['0x0'], 0xb8c0 = ['0x0'], 0xb8c4 = ['0x80000001'], 0xb8c8 = ['0x0'], 0xb8cc = ['0x0'], 0xb8d0 = ['0x0'], 0xb8d4 = ['0x0'], 0xb8e8 = ['0x80']

    wreg32 and wreg_bulk sorted assignments:
    0xb810 = ['0x0']
    0xb868 = ['0x80']
    0xb86c = ['0x0']
    0xb870 = ['0x0']
    0xb874 = ['0x21']
    0xb878 = ['0x2']
    0xb87c = ['0x80']
    0xb880 = ['0x2']
    0xb884 = ['0x1']
    0xb888 = ['0x80']
    0xb88c = ['0x1']
    0xb890 = ['0x80']
    0xb894 = ['0x1']
    0xb898 = ['0x100']
    0xb89c = ['0x1']
    0xb8a0 = ['0x4000']
    0xb8a4 = ['0x1']
    0xb8a8 = ['0x100']
    0xb8ac = ['0x2']
    0xb8b0 = ['0x80']
    0xb8b4 = ['0x1']
    0xb8b8 = ['0x4000']
    0xb8bc = ['0x0']
    0xb8c0 = ['0x0']
    0xb8c4 = ['0x80000001']
    0xb8c8 = ['0x0']
    0xb8cc = ['0x0']
    0xb8d0 = ['0x0']
    0xb8d4 = ['0x0']
    0xb8e8 = ['0x80']


    /===========================================================\
    |                                                           |
    |                       Blob #000003                        |
    |                                                           |
    \===========================================================/

    blob_t                   [3] = {unique_key = 7230461151160335494, {blob_type = {requires_patching = 0, static_exe = 0, dynamic_exe = 1, reserved = 0}, blob_type_all = blob_t::DYNAMIC, block_index = 4}, size = 156, {data = 0x7fff4703d000, offset_in_block = 1191432192}}

    TODO: Dynamic blob parsing not yet supported


    /===========================================================\
    |                                                           |
    |                       Blob #000004                        |
    |                                                           |
    \===========================================================/

    blob_t                   [4] = {unique_key = 3428154159411768589, {blob_type = {requires_patching = 0, static_exe = 1, dynamic_exe = 0, reserved = 0}, blob_type_all = blob_t::EXE, block_index = 2}, size = 152, {data = 0x7fff5f1640a8, offset_in_block = 1595293864}}

    #  0 0x81b8100000000000	 MB: 0x1, SWTC: 0x0, EB: 0x0, OP: 0x1  (WREG 32        ), REG_OFFSET: 0xb810, RSV: 0x0, PRED: 0x0 , VALUE: 0x0       
    #  1 0x82b8680000000010	 MB: 0x1, SWTC: 0x0, EB: 0x0, OP: 0x2  (WREG BULK      ), REG_OFFSET: 0xb868, PRED: 0x0 , SIZE64: 0x10  
      0x0000000000000000  0x0000000000000000  0x0000000000000000  0x0000000000000000
      0x0000000000000000  0x0000000000000000  0x0000000000000000  0x0000000000000000
      0x0000000000000000  0x0000000000000000  0x0000000000000000  0x8000000100000000
      0x0000000000000000  0x0000000000000000  0x00ffffff00000000  0x00ffffff00000000
    # 18 0xc1b8e80000000000	 MB: 0x1, SWTC: 0x1, EB: 0x0, OP: 0x1  (WREG 32        ), REG_OFFSET: 0xb8e8, RSV: 0x0, PRED: 0x0 , VALUE: 0x0       

    wreg32 and wreg_bulk assignments:
    0xb810 = ['0x0'], 0xb868 = ['0x0'], 0xb86c = ['0x0'], 0xb870 = ['0x0'], 0xb874 = ['0x0'], 0xb878 = ['0x0'], 0xb87c = ['0x0'], 0xb880 = ['0x0'], 0xb884 = ['0x0'], 0xb888 = ['0x0'], 0xb88c = ['0x0'], 0xb890 = ['0x0'], 0xb894 = ['0x0'], 0xb898 = ['0x0'], 0xb89c = ['0x0'], 0xb8a0 = ['0x0'], 0xb8a4 = ['0x0'], 0xb8a8 = ['0x0'], 0xb8ac = ['0x0'], 0xb8b0 = ['0x0'], 0xb8b4 = ['0x0'], 0xb8b8 = ['0x0'], 0xb8bc = ['0x0'], 0xb8c0 = ['0x0'], 0xb8c4 = ['0x80000001'], 0xb8c8 = ['0x0'], 0xb8cc = ['0x0'], 0xb8d0 = ['0x0'], 0xb8d4 = ['0x0'], 0xb8d8 = ['0x0'], 0xb8dc = ['0xffffff'], 0xb8e0 = ['0x0'], 0xb8e4 = ['0xffffff'], 0xb8e8 = ['0x0']

    wreg32 and wreg_bulk sorted assignments:
    0xb810 = ['0x0']
    0xb868 = ['0x0']
    0xb86c = ['0x0']
    0xb870 = ['0x0']
    0xb874 = ['0x0']
    0xb878 = ['0x0']
    0xb87c = ['0x0']
    0xb880 = ['0x0']
    0xb884 = ['0x0']
    0xb888 = ['0x0']
    0xb88c = ['0x0']
    0xb890 = ['0x0']
    0xb894 = ['0x0']
    0xb898 = ['0x0']
    0xb89c = ['0x0']
    0xb8a0 = ['0x0']
    0xb8a4 = ['0x0']
    0xb8a8 = ['0x0']
    0xb8ac = ['0x0']
    0xb8b0 = ['0x0']
    0xb8b4 = ['0x0']
    0xb8b8 = ['0x0']
    0xb8bc = ['0x0']
    0xb8c0 = ['0x0']
    0xb8c4 = ['0x80000001']
    0xb8c8 = ['0x0']
    0xb8cc = ['0x0']
    0xb8d0 = ['0x0']
    0xb8d4 = ['0x0']
    0xb8d8 = ['0x0']
    0xb8dc = ['0xffffff']
    0xb8e0 = ['0x0']
    0xb8e4 = ['0xffffff']
    0xb8e8 = ['0x0']