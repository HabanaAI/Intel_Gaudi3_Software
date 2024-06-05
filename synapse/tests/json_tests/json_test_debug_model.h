#pragma once

#include <string>
#include <fstream>

static constexpr const char* DEBUG_MODEL =
    "{\"graphs\": [{\"breakpoint_after_graph\": 0, \"id\": 0, \"name\": \"basic\", \"nodes\": [{\"blocking_nodes\": "
    "[], \"breakpoint_before_node\": 0, \"exec_order_idx\": 1, \"graph_index\": 0, \"guid\": \"spatial_convolution\", "
    "\"input_ctrl_tensors\": [], \"input_tensors\": [\"autoGenPersistInputTensorName_0\", "
    "\"autoGenPersistInputTensorName_1\"], \"is_logical\": false, \"name\": \"Convolution1\", \"output_ctrl_tensors\": "
    "[], \"output_tensors\": [\"autoGenPersistOutputTensorName_2\"], \"params\": [3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, "
    "1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 162, 191, 1, 0, 0, 0, "
    "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 255, 127, 0, 0], \"rollups\": [], \"type\": "
    "\"Convolution\"}], \"recipe_debug_id\": 0, \"tensors\": [{\"alias\": false, \"allocation\": \"DRAM\", \"dtype\": "
    "\"float32\", \"graph_index\": 0, \"is_const\": false, \"is_dense\": true, \"is_reduction\": false, \"max_shape\": "
    "[3, 6, 6, 1], \"min_shape\": [3, 6, 6, 1], \"name\": \"autoGenPersistInputTensorName_0\", \"offset\": "
    "18446744073709551615, \"persistent\": true, \"rmw_section\": false, \"strides\": [4, 12, 72, 432, 432], \"type\": "
    "\"DATA_TENSOR\", \"usage\": \"INPUT_TENSOR\", \"user_mem_offset\": 0, \"user_mem_section_index\": [3]}, "
    "{\"alias\": false, \"allocation\": \"DRAM\", \"dtype\": \"float32\", \"graph_index\": 0, \"is_const\": false, "
    "\"is_dense\": true, \"is_reduction\": false, \"max_shape\": [1, 3, 3, 3], \"min_shape\": [1, 3, 3, 3], \"name\": "
    "\"autoGenPersistInputTensorName_1\", \"offset\": 18446744073709551615, \"persistent\": true, \"rmw_section\": "
    "false, \"strides\": [4, 4, 12, 36, 108], \"type\": \"DATA_TENSOR\", \"usage\": \"INPUT_TENSOR\", "
    "\"user_mem_offset\": 0, \"user_mem_section_index\": [4]}, {\"alias\": false, \"allocation\": \"DRAM\", \"dtype\": "
    "\"float32\", \"graph_index\": 0, \"is_const\": false, \"is_dense\": true, \"is_reduction\": false, \"max_shape\": "
    "[1, 4, 4, 1], \"min_shape\": [1, 4, 4, 1], \"name\": \"autoGenPersistOutputTensorName_2\", \"offset\": "
    "18446744073709551615, \"persistent\": true, \"rmw_section\": false, \"strides\": [4, 4, 16, 64, 64], \"type\": "
    "\"DATA_TENSOR\", \"usage\": \"OUTPUT_TENSOR\", \"user_mem_offset\": 0, \"user_mem_section_index\": [5]}]}, "
    "{\"breakpoint_after_graph\": 0, \"id\": 1, \"name\": \"basic\", \"nodes\": [{\"blocking_nodes\": [], "
    "\"breakpoint_before_node\": 0, \"exec_order_idx\": 1, \"graph_index\": 1, \"guid\": \"spatial_convolution\", "
    "\"input_ctrl_tensors\": [], \"input_tensors\": [\"autoGenPersistInputTensorName_0\", "
    "\"autoGenPersistInputTensorName_1\"], \"is_logical\": false, \"name\": \"Convolution1\", \"output_ctrl_tensors\": "
    "[], \"output_tensors\": [\"autoGenPersistOutputTensorName_2\"], \"params\": [3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, "
    "1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 168, 212, 1, 0, 0, 0, "
    "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 255, 127, 0, 0], \"rollups\": [], \"type\": "
    "\"Convolution\"}], \"recipe_debug_id\": 0, \"tensors\": [{\"alias\": false, \"allocation\": \"DRAM\", \"dtype\": "
    "\"float32\", \"graph_index\": 1, \"is_const\": false, \"is_dense\": true, \"is_reduction\": false, \"max_shape\": "
    "[3, 6, 6, 1], \"min_shape\": [3, 6, 6, 1], \"name\": \"autoGenPersistInputTensorName_0\", \"offset\": "
    "18446744073709551615, \"persistent\": true, \"rmw_section\": false, \"strides\": [4, 12, 72, 432, 432], \"type\": "
    "\"DATA_TENSOR\", \"usage\": \"INPUT_TENSOR\", \"user_mem_offset\": 0, \"user_mem_section_index\": [3]}, "
    "{\"alias\": false, \"allocation\": \"DRAM\", \"dtype\": \"float32\", \"graph_index\": 1, \"is_const\": false, "
    "\"is_dense\": true, \"is_reduction\": false, \"max_shape\": [1, 3, 3, 3], \"min_shape\": [1, 3, 3, 3], \"name\": "
    "\"autoGenPersistInputTensorName_1\", \"offset\": 18446744073709551615, \"persistent\": true, \"rmw_section\": "
    "false, \"strides\": [4, 4, 12, 36, 108], \"type\": \"DATA_TENSOR\", \"usage\": \"INPUT_TENSOR\", "
    "\"user_mem_offset\": 0, \"user_mem_section_index\": [4]}, {\"alias\": false, \"allocation\": \"DRAM\", \"dtype\": "
    "\"float32\", \"graph_index\": 1, \"is_const\": false, \"is_dense\": true, \"is_reduction\": false, \"max_shape\": "
    "[1, 4, 4, 1], \"min_shape\": [1, 4, 4, 1], \"name\": \"autoGenPersistOutputTensorName_2\", \"offset\": "
    "18446744073709551615, \"persistent\": true, \"rmw_section\": false, \"strides\": [4, 4, 16, 64, 64], \"type\": "
    "\"DATA_TENSOR\", \"usage\": \"OUTPUT_TENSOR\", \"user_mem_offset\": 0, \"user_mem_section_index\": [5]}]}], "
    "\"name\": \"debug\", \"version\": 1}";

inline void generateDebugModel(const std::string& filePath)
{
    std::ofstream jsonFile(filePath);
    if (!jsonFile)
    {
        throw std::runtime_error("failed to open json file: " + filePath);
    }
    jsonFile << DEBUG_MODEL;
    jsonFile.close();
    if (!jsonFile)
    {
        throw std::runtime_error("failed to write json file: " + filePath);
    }
}