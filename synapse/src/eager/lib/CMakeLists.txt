cmake_minimum_required(VERSION 3.5)

add_library(Eager STATIC
    eager_interface.cpp
    include/eager/eager_interface.h

    eager_graph.h
    eager_graph.cpp
    program_data_blob_manager.h
    tpc_node_handler.cpp
    tpc_node_handler.h
    chip_info.h
    chip_info.cpp
    eager_brain_base.cpp
    eager_brain_base.h

    node_info/const_tensor_optimizer.cpp
    node_info/const_tensor_optimizer.h
    node_info/duplication_map.cpp
    node_info/duplication_map.h
    node_info/eager_complex_guid_extractor.cpp
    node_info/eager_complex_guid_extractor.h
    node_info/eager_memset_node_output.h
    node_info/eager_node.cpp
    node_info/eager_node.h
    node_info/exec_schedule.cpp
    node_info/exec_schedule.h
    node_info/node_collector.cpp
    node_info/node_collector.h
    node_info/node_container.cpp
    node_info/node_container.h
    node_info/node_displacement.cpp
    node_info/node_displacement.h
    node_info/node_info_defs.cpp
    node_info/node_info_defs.h
    node_info/suggested_tensor_manipulation.cpp
    node_info/suggested_tensor_manipulation.h
    node_info/tensor_info.cpp
    node_info/tensor_info.h
    node_info/transpose_fuser.cpp
    node_info/transpose_fuser.h

    desc_gen/desc_gen_hal.h
    desc_gen/desc_base.h
    desc_gen/desc_base.cpp
    desc_gen/tpc_desc_base.h
    desc_gen/tpc_desc_base.cpp
    desc_gen/sync_scheme_manager_base.h
    desc_gen/sync_scheme_manager_base.cpp
    desc_gen/node2desc.h
    desc_gen/node2desc.cpp

    #[[
        Template Specific Implementation
        ================================
    ]]

        #[[ Recipe Generation: Offline ]]
        recipe_gen/recipe_templates.cpp
        recipe_gen/recipe_templates.h
        recipe_gen/recipe_templates_defs.h

        #[[ Recipe Analysis: Offline ]]
        recipe_gen/blob_to_desc_map.h
        recipe_gen/blob_to_desc_map.cpp
        recipe_gen/blob_to_desc_map_structs.h
        recipe_gen/blob_to_desc_map_structs.cpp

        #[[ Recipe Allocation: Hot Path ]]
        recipe_gen/eager_recipe_allocator.h
        recipe_gen/eager_recipe_allocator.cpp
        recipe_gen/eager_recipe_generator.h
        recipe_gen/eager_recipe_generator.cpp

        #[[ Recipe Instantiation: Hot Path ]]
        recipe_gen/recipe_arc_job_utils.cpp
        recipe_gen/recipe_arc_job_utils.h
        recipe_gen/recipe_arc_job_writer.cpp
        recipe_gen/recipe_arc_job_writer.h
        recipe_gen/recipe_arc_job_ecb_writer.cpp
        recipe_gen/recipe_arc_job_ecb_writer.h
        recipe_gen/recipe_instantiation.cpp
        recipe_gen/recipe_instantiation.h
        recipe_gen/recipe_instantiation_dma.cpp
        recipe_gen/recipe_instantiation_dma.h
        recipe_gen/recipe_instantiation_mme.cpp
        recipe_gen/recipe_instantiation_mme.h
        recipe_gen/recipe_instantiation_tpc.cpp
        recipe_gen/recipe_instantiation_tpc.h
        recipe_gen/template_structs.h
        recipe_gen/template_structs_utils.h

        #[[ Utils: Both offline and hot path ]]
        #[[ TODO: Split hot path functions (?) ]]
        recipe_gen/recipe_defs.h
        recipe_gen/recipe_hal_base.h

    utils/algorithm_utils.h
    utils/float_utils.h
    utils/general_defs.h
    utils/general_utils.h
    utils/memory_utils.h
    utils/numeric_utils.h
    utils/sequential_iter_tracker.h
    utils/string_utils.h

    chip_specification/gaudi2/general_defs.h
    chip_specification/gaudi2/mme_brain.h

    chip_specification/gaudi2/desc_gen/desc_gen_hal.h
    chip_specification/gaudi2/desc_gen/desc_gen_hal.cpp
    chip_specification/gaudi2/desc_gen/mme_desc.h
    chip_specification/gaudi2/desc_gen/mme_desc.cpp
    chip_specification/gaudi2/desc_gen/tpc_desc.h
    chip_specification/gaudi2/desc_gen/tpc_desc.cpp
    chip_specification/gaudi2/desc_gen/dma_desc.h
    chip_specification/gaudi2/desc_gen/dma_desc.cpp
    chip_specification/gaudi2/desc_gen/desc_factory.h
    chip_specification/gaudi2/desc_gen/desc_factory.cpp
    chip_specification/gaudi2/desc_gen/sync_scheme_manager.h
    chip_specification/gaudi2/desc_gen/sync_scheme_manager.cpp

    chip_specification/gaudi2/recipe/recipe_hal.cpp
    chip_specification/gaudi2/recipe/recipe_hal.h
    chip_specification/gaudi2/recipe/recipe_hal_defs.h
    chip_specification/gaudi2/recipe/command_packets_wrappers.h
    chip_specification/gaudi2/recipe/dma_template_gen.h
    chip_specification/gaudi2/recipe/mme_template_gen.h
    chip_specification/gaudi2/recipe/template_structs.h
    chip_specification/gaudi2/recipe/templates_creator.cpp
    chip_specification/gaudi2/recipe/templates_creator.h
    chip_specification/gaudi2/recipe/tpc_template_gen.h

    chip_specification/gaudi3/general_defs.h
    chip_specification/gaudi3/mme_brain.h

    chip_specification/gaudi3/desc_gen/desc_gen_hal.h
    chip_specification/gaudi3/desc_gen/desc_gen_hal.cpp
    chip_specification/gaudi3/desc_gen/mme_desc.h
    chip_specification/gaudi3/desc_gen/mme_desc.cpp
    chip_specification/gaudi3/desc_gen/tpc_desc.h
    chip_specification/gaudi3/desc_gen/tpc_desc.cpp
    chip_specification/gaudi3/desc_gen/desc_factory.h
    chip_specification/gaudi3/desc_gen/desc_factory.cpp
    chip_specification/gaudi3/desc_gen/sync_scheme_manager.h
    chip_specification/gaudi3/desc_gen/sync_scheme_manager.cpp

    chip_specification/gaudi3/recipe/recipe_hal.cpp
    chip_specification/gaudi3/recipe/recipe_hal.h
    chip_specification/gaudi3/recipe/recipe_hal_defs.h
    chip_specification/gaudi3/recipe/command_packets_wrappers.h
    chip_specification/gaudi3/recipe/mme_template_gen.h
    chip_specification/gaudi3/recipe/tpc_template_gen.h
    chip_specification/gaudi3/recipe/template_structs.h
    chip_specification/gaudi3/recipe/templates_creator.cpp
    chip_specification/gaudi3/recipe/templates_creator.h

    debug_tools/eager_graph_visualization.h
    debug_tools/eager_graph_visualization.cpp
)

target_include_directories(Eager
PUBLIC
    include/
PRIVATE
                                            #[[ In source comment used for include group: ]]
    $ENV{QMAN_FW_ROOT}/engines-arc/include/ #[[ "// relative to <qman_fw>/engines-arc/include/" ]]
    $ENV{SPECS_EXT_ROOT}/                   #[[ "// relative to <specs_external>/" ]]
    $ENV{SPECS_ROOT}/                       #[[ "// relative to <specs>/" ]]
    $ENV{SPECS_ROOT_H9}/                    #[[ "// relative to <specs_h9>/" ]]
    $ENV{SYNAPSE_ROOT}/include/             #[[ "// synapse api (relative to include/)" ]]
    $ENV{SYNAPSE_ROOT}/src/                 #[[ "// synapse-internal includes (relative to src/)" ]]
                                            #[[ "// synapse-internal passes includes (relative to src/)" ]]
                                            #[[ "// synapse-internal gaudi2-specific includes (relative to src/)" ]]
                                            #[[ "// synapse-internal gaudi3-specific includes (relative to src/)" ]]
    $ENV{SYNAPSE_ROOT}/src/eager/lib/       #[[ "// eager includes (relative to src/eager/lib/)" ]]
    $ENV{THIRD_PARTIES_ROOT}/               #[[ "// relative to <3rd-parties>/" ]]

    # [[ W/A for including graph_compiler passes headers ]]
    $ENV{SYNAPSE_ROOT}/src/graph_compiler/habana_nodes/
)

target_include_directories(Eager SYSTEM PRIVATE $<TARGET_PROPERTY:fma,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(Eager PRIVATE mme_stack SynapseInfra HalReader)