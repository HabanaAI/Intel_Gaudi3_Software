#pragma once

#include "basic_recipe_info.hpp"
#include "recipe.h"
#include "recipe_dynamic_info.hpp"
#include "runtime/common/recipe/device_agnostic_recipe_info.hpp"
#include "runtime/common/recipe/patching/define.hpp"
#include "runtime/common/recipe/recipe_verification.hpp"
#include "runtime/qman/common/recipe_static_information.hpp"
#include "runtime/qman/common/static_info_processor.hpp"
#include "runtime/scal/common/recipe_static_processor_scal.hpp"
#include "smf/shape_func_registry.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "tpc_kernel_lib_interface.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <memory>

//            Node0
//            ---         node1
//   In0----->| |  mid0  ---
//   In0----->| |------> | |
//   Shape0-->| |        | |         node2
//            ---        | |  mid1  ---
//   In1---------------->| |------->| |
//   Shape1------------->| |        | |
//                       ---        | |
//   In2--------------------------->| |--------->Out0 (tensor3)
//   Shape2------------------------>| |
//                                  ---
//
//   Launch tensors = {In0, In1, In2, out0, Shape0, Shape1, Shape2}
//   2 patch points per node
//   Tensors: 0-2 (in) 3(out) 4-5(Internal) 6-8(Shape)
//   Static tensors: In1, Shape2
//

using tensorType = shape_plane_basic_node_t::EShapePlanceTensorDb;

class DsdRecipe : public ::testing::Test
{
public:
    void createRecipeTensors();
    void createRecipeBlobs();
    void createSpgNodes();
    void createIn();
    void createSpgTensors();
    void createBlobs();
    void createShapeTensors();

    void registerFunctions();
    void verifyBlobs(std::string msg, const blob_t* blobsChunksBlobs);

    void verifyAddrPP(data_chunk_patch_point_t* pp, int n, std::string msg);

    void initTensorMap();
    void initDynamicPatchingTest(bool addFuser = false);
    void setFuser();

    void copyBetweenBlobsChunksAndDataChunks(uint8_t*                     patchingBlobsChunkBuffer,
                                             const std::vector<uint64_t>& dataChunksHostAddresses,
                                             uint64_t                     chunkSize,
                                             bool                         isToDataChunk);

    bool patch(DynamicRecipe&               dynamicRecipe,
               uint8_t*&                    patchingBlobsChunkBuffer,
               const std::vector<uint64_t>& dataChunksHostAddresses,
               uint64_t                     dcSizeCpDma,
               const std::vector<uint32_t>* tensorIdx2userIdx);

    void SetUp() override;
    void TearDown() override;

    typedef std::unique_ptr<uint8_t[]> SingleDataChunkHostBuffer;

    uint32_t static const DIMS = 3;

    uint32_t static const NUM_NODES            = 3;
    uint32_t static const NUM_PERSIST_TENSORS  = NUM_NODES + 1;  // 4: 0-2 (in), 3(out)
    uint32_t static const NUM_INTERNAL_TENSORS = NUM_NODES - 1;  // 2: 4-5
    uint32_t static const NUM_SHAPE_TENSORS    = NUM_NODES;      // 3: 6-8
    uint32_t static const NUM_TENSORS          = NUM_PERSIST_TENSORS + (NUM_NODES - 1) + NUM_SHAPE_TENSORS;  // 9: 4+2+3
    uint32_t static const FIRST_SHAPE_TENSOR   = NUM_PERSIST_TENSORS + NUM_INTERNAL_TENSORS;                 // 6: 4+2

    uint32_t static const NUM_NODE_INPUTS  = 3;
    uint32_t static const NUM_NODE_OUTPUTS = 1;
    uint32_t static const NUM_PP_NODE      = 2;
    uint32_t static const NUM_BLOBS        = NUM_NODES;
    uint32_t static const BLOB_DATA_SIZE   = 0x20;
    uint32_t static const TOTAL_DATA_SIZE  = NUM_BLOBS * BLOB_DATA_SIZE * sizeof(uint32_t);
    uint32_t static const MAX_TENSOR_SIZE  = 20;
    uint32_t static const MIN_TENSOR_SIZE  = 12;
    uint32_t static const SET_TENSOR_SIZE  = 16;  // Make sure it is even
    uint32_t static const SET_SHAPE_SIZE   = SET_TENSOR_SIZE + 1;
    uint32_t static const NUM_ADDR_PP      = 3;

    recipe_t                 recipe {};
    basicRecipeInfo          recipeInfo {&recipe, &spg};
    DeviceAgnosticRecipeInfo deviceAgnosticRecipeInfo;
    RecipeStaticInfo         recipeStaticInfo {};
    persist_tensor_info_t    recipeTensor[NUM_PERSIST_TENSORS] {};
    std::string              tensorName[NUM_PERSIST_TENSORS] {};
    synLaunchTensorInfoExt   launchTensors[NUM_PERSIST_TENSORS + NUM_SHAPE_TENSORS] {};
    roi_info_t               roiInfo {};
    tensor_roi_t             tensorRoi {};  // not used
    std::string              shapeTensorsName[NUM_SHAPE_TENSORS] {};
    shape_tensor_info_t      shapeTensorsInfo[NUM_SHAPE_TENSORS] {};

    shape_plane_graph_t      spg {};
    shape_plane_node_t       spn[NUM_NODES] {};
    shape_plane_basic_node_t spbn[NUM_NODES] {};
    uint32_t                 nodeIn[NUM_NODES][NUM_NODE_INPUTS] {};
    uint32_t                 nodeOut[NUM_NODES][NUM_NODE_OUTPUTS] {};
    tensorType               tensorInType[NUM_NODES][NUM_NODE_INPUTS] {};
    tensorType               tensorOutType[NUM_NODES][NUM_NODE_OUTPUTS] {};
    sm_patch_point_t         smPP[NUM_NODES][NUM_PP_NODE] {};
    uint8_t                  sif_params[NUM_NODES] {};
    tensor_info_t            spgTensors[NUM_TENSORS] {};
    blob_t                   blobs[NUM_BLOBS] {};
    uint32_t                 blobData[NUM_BLOBS][BLOB_DATA_SIZE] {};
    std::vector<uint32_t>    tensorIdx2userIdx[2] {};
    gc_conf_t                gcConf[5] {};

    patch_point_t addrPP[NUM_ADDR_PP] {};

    /********************* Fuser ****************/
    // On node 1:
    //            node1-1    node1-2
    //            ---        ---
    //   mid0---->| |  temp1 | |mid1
    //   In1----->| |------> | |------->
    //            | |  temp2 | |
    //            | |------> | |
    //            ---        ---

    shape_plane_basic_node_t fuserBasicNode[2] {};
    uint32_t                 fuserIn[2][3] {};
    uint32_t                 fuserOut[2][2] {};
    tensorType               fuserInType[2][3] {};
    tensorType               fuserOutType[2][2] {};
    uint8_t                  fuserSifParams[2][1] {};  // size of one
    tensor_info_t            fuserTensorDb[2];

    /******************** Fuser end **************/

    synDeviceType m_deviceType = synDeviceGaudi;
};

tpc_lib_api::_GlueCodeReturn sif2_n(const tpc_lib_api::ShapeInferenceParams* inputParams, tpc_lib_api::ShapeInferenceOutput* outputData, int numOut);
tpc_lib_api::_GlueCodeReturn sif2_1(tpc_lib_api::DeviceId deviceId,const tpc_lib_api::ShapeInferenceParams* inputParams,
                                     tpc_lib_api::ShapeInferenceOutput*       outputData);
tpc_lib_api::_GlueCodeReturn sif2_2(tpc_lib_api::DeviceId deviceId, const tpc_lib_api::ShapeInferenceParams* inputParams,
                                     tpc_lib_api::ShapeInferenceOutput*       outputData);
void                          smf0(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);