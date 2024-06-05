#pragma once

#include "gaudi_tests/gc_dynamic_shapes_infra.h"

class SynGaudiDataLayoutTest : public SynGaudiDynamicShapesTestsInfra
{
protected:
    template<typename ArrType = TSize, class T>
    static void
    transposeBuffer(const ArrType* inputSizes, unsigned inputDim, T* buffer, const gc::Permutation& permutation)
    {
        std::vector<TSize> inSizes(inputSizes, inputSizes + inputDim);
        std::vector<TSize> outSizes(inputDim);
        synTransposeParams params;
        params.tensorDim = permutation.size();
        for (unsigned i = 0; i < inputDim; i++)
        {
            unsigned inDim        = permutation.getValues()[i];
            params.permutation[i] = (TransposePermutationDim)inDim;
            outSizes[i]           = inputSizes[inDim];
        }

        synDataType dtype   = asSynType<T>();
        TensorPtr   IFM     = TensorPtr(new Tensor(inputDim, inSizes.data(), dtype, reinterpret_cast<char*>(buffer)));
        TensorPtr   OFM_ref = TensorPtr(new Tensor(inputDim, outSizes.data(), dtype));

        NodePtr ref_n = NodeFactory::createNode({IFM}, {OFM_ref}, &params, NodeFactory::transposeNodeTypeName, "");
        ref_n->RunOnCpu();
        T* ref_resultArray = (T*)OFM_ref->map();

        uint64_t numElements = multiplyElements(outSizes);
        memcpy(buffer, ref_resultArray, numElements * sizeof(T));
    }

    bool isPermuted(const char* name, const gc::Permutation& expected);

    gc::Permutation getPermutation(const char* name);

    void setPermutation(unsigned t, const gc::Permutation& perm);

    using InOutLayouts = std::pair<const char**, const char**>;
    InOutLayouts getNodePtLayout(const char* guid);

    void addNodeWithLayouts(const char*   guid,
                            TensorIndices inputTensorIndices,
                            TensorIndices outputTensorIndices,
                            void*         userParams,
                            unsigned      paramsSize);

    template<typename T>
    static T relu(const T& n)
    {
        return std::max(static_cast<T>(0), n);
    }

    const gc::Permutation m_ptActivationPermutation5D = DimVector({3, 0, 1, 2, 4});
    const gc::Permutation m_ptWeightPermutation5D     = DimVector({4, 3, 0, 1, 2});
    const gc::Permutation m_ptActivationPermutation4D = DimVector({2, 0, 1, 3});
    const gc::Permutation m_ptWeightPermutation4D     = DimVector({3, 2, 0, 1});

    static const char* pt_conv2D_in_layouts[];
    static const char* pt_conv2D_out_layouts[];
    static const char* pt_conv3D_in_layouts[];
    static const char* pt_conv3D_out_layouts[];
    static const char* pt_dedx_in_layouts[];
    static const char* pt_dedx_out_layouts[];
    static const char* pt_dedx3d_in_layouts[];
    static const char* pt_dedx3d_out_layouts[];
    static const char* pt_dedw_in_layouts[];
    static const char* pt_dedw_out_layouts[];
    static const char* pt_dedw3d_in_layouts[];
    static const char* pt_dedw3d_out_layouts[];
};
