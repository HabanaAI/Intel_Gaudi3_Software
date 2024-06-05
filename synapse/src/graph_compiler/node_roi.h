#pragma once
#include "tensor.h"
#include "tensor_roi.h"
#include "roi_shape_type.h"
#include "cache_types.h"

struct MMEPartial
{
    enum MMEPartialType  //Ways of breaking down the calculation of a chunk of output
    {
        PARTIAL_NONE = 0,  //Just do the whole thing, potentially generating output in a couple of chunks
        PARTIAL_FILTER,    //Split a rectangular filter into chunks of 1x1s
        PARTIAL_IFM,       //accumulate the contribution of some of the IFMs
    };

    MMEPartialType type;
    union
    {
        struct  //for partials along Z
        {
            uint32_t firstIFM;
            uint32_t nIFM;
        } IFMPartial;
        struct  //for partials in filter
        {
            uint16_t filterSize[2];   //the rectangle this accumulates
            uint16_t startOffset[2];  //The offset from 0,0 of this part
            uint16_t wSpatialOffset;  //The width offset of the roi
        } FilterPartial;
    };

    bool canSignal;
    bool firstPartial;
    bool lastPartial;
};

// TPC specific work distribution context for GC
struct TpcWdCtx
{
    // Base coordinate of the grid
    TOffset baseCord[MAX_DIMENSIONS_NUM];
    // Actual or Total Size of the index space tensor
    TSize gridSize[MAX_DIMENSIONS_NUM];
    // Index space tensor is divided into small boxes, each is processed by a particular TPC
    // Box size is nothing but amount of work that is processed by a TPC
    TSize boxSize[MAX_DIMENSIONS_NUM];
    // Dim Slices means how many boxSize fit into the gridSize in each dim
    TSize dimSlices[MAX_DIMENSIONS_NUM];
    // Index of the 1st engine to start with
    uint8_t shuffleIndex;

    TpcWdCtx()
    {
        shuffleIndex = 0;
        for (int dim = 0; dim < MAX_DIMENSIONS_NUM; dim++)
        {
            boxSize[dim]   = 1;
            baseCord[dim]  = 0;
            gridSize[dim]  = 0;
            dimSlices[dim] = 0;
        }
    }
};

//This struct defines an ROI on a node's index space
//For MME: 2D accumulator space made of [Z/vector]x[BHW/vector]
//For DMA: Tensor-dim pixel space (linear DMA requires split along only one dimension)
//For TPC: index space as defined in the spec
struct NodeROI
{
    TOffset      baseOffset[Tensor::c_tensorMaxNDim]         = {0};
    TSize        size[Tensor::c_tensorMaxNDim]               = {0};
    int          spatialOffset[Tensor::c_numOfNStrides - 1]  = {0};  // TODO: Move to TOffset array [SW-117362]
    int          spatialStrides[Tensor::c_numOfNStrides - 1] = {0};
    unsigned int numIterations                               = 0;
    TSize        spatialSizeMinus1                           = 0;
    unsigned int vectorSize                                  = 0;
    MMEPartial   mmePartial                                  = {};
    bool         isLowered                                  = false;
    bool         isAux = false;  // should this descriptor be patched using regular inputs or aux inputs
    unsigned int pipelineLevel                              = 0;
    unsigned int engineIndex                                = 0;
    uint32_t     numSignals                                 = 0;
    RoiShapeType roiDsdType                                 = RoiShapeType::UNSPECIFIED;
    std::vector<TpcWdCtx>      tpcWdCtx;
    DcoreRoisVec               dcoreROIs;
    std::vector<CacheMetaData> inputsCacheMetaData;
    std::vector<CacheMetaData> outputsCacheMetaData;

    TensorROIVector        inputRois;
    TensorROIVector        outputRois;
    std::shared_ptr<void>  additionalData; // Here we can patch in information about the NodeROI (used later for splitting)

    CmeTasks              cmeTasks;
    std::vector<unsigned> rolloverIds;
};
