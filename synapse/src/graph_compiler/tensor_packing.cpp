#include "tensor_packing.h"
#include "utils.h"

bool canPack(const pTensor t, unsigned dim, unsigned factor)
{
    TensorAnnotation& ann = t->getTensorAnnotation();

    if (t->isPacked(dim) && ann.dataInfo.packing[dim] != factor)
    {
        return false;
    }

    return true;
}

void packBias(ConvolutionNode* n, TensorPtr t, const std::array<TSize, MME_MAX_CONV_DIMS>& packing)
{
    uint64_t inBiasDataSize   = t->getTotalSizeInBytes();
    TSize    packingX         = packing[PACKING_X];

    HB_ASSERT(canPack(t, PACKING_X, packingX), "can't packing {} in dim {} with factor {}",
              t->getName(), PACKING_X, packingX);

    // if tensor is packed, we know it was packed with the same factor
    if (packingX > 1 && !t->isPacked(PACKING_X))
    {
        TSize sizes[MME_MAX_TENSOR_DIMS];
        t->getAllSizesInElements(sizes, MME_MAX_TENSOR_DIMS);

        /* bias should be duplicated for each output channel */
        sizes[DIM_C] = sizes[DIM_C] * packingX;

        t->reshape(t->getDim(), sizes, nullptr);

        /* duplicate data */
        void     *data       = t->getAddress();
        uint64_t newDataSize = t->getTotalElements() * t->getElementSizeInBytes();
        uint8_t  *newData    = new uint8_t[newDataSize];

        for (TSize i = 0; i < packingX; ++i)
        {
            memcpy(newData + (i * inBiasDataSize), data, inBiasDataSize);
        }

        /* bind new data to tensor */
        t->unbind();
        t->bind((void *)newData, true);
        t->setPacked(packing);
    }
}

void packWeights(ConvolutionNode* n, pTensor t, const std::array<TSize, MME_MAX_CONV_DIMS>& packing)
{
    synConvolution3DParamsV2& convParams = n->getConvolutionParams();
    TSize                     packingX   = packing[PACKING_X];

    HB_ASSERT(canPack(t, PACKING_X, packingX), "can't packing {} in dim {} with factor {}",
              t->getName(), PACKING_X, packingX);

    // if tensor is packed, we know it was packed with the same factor
    if (packingX > 1 && !t->isPacked(PACKING_X))
    {
        SizeArray   newSizes   = t->getAllSizesInElements();
        SizeArray   orgSizes   = t->getAllSizesInElements();
        StrideArray orgStrides = t->getAllStridesInBytes();

        /* weights are being duplicated for each output channel */
        newSizes[WEIGHT_DIM_K] = newSizes[WEIGHT_DIM_K] * packingX;
        /* each duplication adds to S dimension, so it can be filled with zero
         * when shouldn't be taken into account */
        newSizes[WEIGHT_DIM_S] = newSizes[WEIGHT_DIM_S] + (convParams.stride[CONV_STRIDE_WIDTH] * (packingX - 1));

        t->reshape(t->getDim(), newSizes.data(), nullptr);

        /* duplicate data */
        void    *data       = t->getAddress();
        TSize   newDataSize = t->getDenseSizeInElements() * t->getElementSizeInBytes();
        uint8_t *newData    = new uint8_t[newDataSize];

        //init new data to zero
        padBuffWithValue(newData, t->getDenseSizeInElements(), t->getZeroPoint(), t->getElementType());

        LOG_DEBUG(GC,
                  "{}: packed weights changed from ({}) to ({})",
                  n->getNodeName(),
                  toString(orgSizes.data(), orgSizes.data() + t->getDim(), ','),
                  toString(newSizes.data(), newSizes.data() + t->getDim(), ','));

        //generate the weights data according to new dimensions
        TSize orgSizeR = orgSizes[WEIGHT_DIM_R];
        TSize orgSizeS = orgSizes[WEIGHT_DIM_S];
        TSize orgSizeC = orgSizes[WEIGHT_DIM_C];
        TSize orgSizeK = orgSizes[WEIGHT_DIM_K];
        TSize newSizeR = newSizes[WEIGHT_DIM_R];
        TSize newSizeS = newSizes[WEIGHT_DIM_S];
        TSize newSizeC = newSizes[WEIGHT_DIM_C];
        TSize newSizeK = newSizes[WEIGHT_DIM_K];

        for (TSize p = 0; p < packingX; ++p)
        {
            uint64_t sOffset = p * convParams.stride[CONV_STRIDE_WIDTH]; //how many zeros to have before the weights start
            uint64_t newOffK = p * orgSizeK;      //current instance of the duplicated weights
            uint64_t orgOffK = 0;

            for (TSize q = 0; q < orgSizes[WEIGHT_DIM_Q]; ++q)
            {
                //copy the original weights to the correct offset in the new weights, for each packing
                for (TSize r = 0; r < orgSizes[WEIGHT_DIM_R]; ++r)
                {
                    for (TSize s = 0; s < orgSizes[WEIGHT_DIM_S]; ++s)
                    {
                        //new weights S dimension size is larger, so the rest remains with zeros
                        for (TSize c = 0; c < orgSizes[WEIGHT_DIM_C]; ++c)
                        {
                            uint64_t srcFirstElementIdx = orgOffK +
                                                          orgSizeK * c +
                                                          orgSizeK * orgSizeC * s +
                                                          orgSizeK * orgSizeC * orgSizeS * r +
                                                          orgSizeK * orgSizeC * orgSizeS * orgSizeR * q;
                            uint64_t dstFirstElementIdx = newOffK +
                                                          newSizeK * c +
                                                          newSizeK * newSizeC * (s + sOffset) +
                                                          newSizeK * newSizeC * newSizeS * r +
                                                          newSizeK * newSizeC * newSizeS * newSizeR * q;

                            memcpy(newData + dstFirstElementIdx * t->getElementSizeInBytes(),
                                   (uint8_t*)data + srcFirstElementIdx * t->getElementSizeInBytes(),
                                   orgStrides[WEIGHT_DIM_C]);
                        }
                    }
                }
            }
        }
        /* bind new data to tensor */
        t->unbind();
        t->bind((void *)newData, true);
        t->setPacked(packing);
    }
}

void packOFM(ConvolutionNode* n, pTensor t, const std::array<TSize, MME_MAX_CONV_DIMS>& packing)
{
    TSize packingX = packing[PACKING_X];

    HB_ASSERT(canPack(t, PACKING_X, packingX), "can't packing {} in dim {} with factor {}",
              t->getName(), PACKING_X, packingX);

    // if tensor is packed, we know it was packed with the same factor
    if (packingX > 1 && !t->isPacked(PACKING_X))
    {
        TSize sizes[MME_MAX_TENSOR_DIMS];
        t->getAllSizesInElements(sizes, MME_MAX_TENSOR_DIMS);

        HB_ASSERT(sizes[DIM_W] % packingX == 0, "tensor W dimension size is not divisible by packing value");

        /* each output line now contains packingX outputs */
        sizes[DIM_C] = sizes[DIM_C] * packingX;
        /* the extra outputs per line is reduced from W dim */
        sizes[DIM_W] = sizes[DIM_W] / packingX;

        t->reshape(t->getDim(), sizes, nullptr);
        t->setPacked(packing);
    }
}
