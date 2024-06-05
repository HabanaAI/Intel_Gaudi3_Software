#ifndef GRAPH_OPTIMIZER_BN_UTILS_H
#define GRAPH_OPTIMIZER_BN_UTILS_H

#include "node.h"
#include "types.h"

class BNUtils
{
public:
    enum eReductionRelatedNodes
    {
        eMemsetNode,
        eReductionNode,
        eNumOfReductionRelatedNodes
    };

    enum eReshapeNodeDirection
    {
        eDirectionForward,
        eDirectionBackward
    };

    typedef std::array<NodePtr, eNumOfReductionRelatedNodes> ReductionRelatedNodesArr;

    struct Bn1Bn2FwdInputs
    {
        TensorPtr xIn;
        TensorPtr beta;
        TensorPtr gamma;
        TensorPtr runningMeanIn;
        TensorPtr runningVarIn;
        TensorPtr residualAddIn;
    };

    struct Bn1Bn2FwdOutputs
    {
        TensorPtr xOut;
        TensorPtr meanOut;
        TensorPtr stdOut;
        TensorPtr runningMeanOut;
        TensorPtr runningVarOut;
    };

    struct Bn1Bn2BwdInputs
    {
        TensorPtr xIn;
        TensorPtr gradIn;
        TensorPtr mean;
        TensorPtr istd;
        TensorPtr gamma;
    };
    struct Bn1Bn2BwdOutputs
    {
        TensorPtr dX;
        TensorPtr dGamma;
        TensorPtr dBeta;
        TensorPtr dZ;
    };

    static ReductionRelatedNodesArr createNodesForSramReduction(unsigned           elements,
                                                                unsigned           minElements,
                                                                unsigned           numberOfDims,
                                                                const std::string& baseName,
                                                                bool               isBnBwd,
                                                                bool               locateInSram);

    static NodeList getMoments(TensorPtr        inputFM,
                               TensorPtr&       outputMean,
                               TensorPtr&       outputSigma,
                               TensorPtr&       outputStd,
                               std::string_view baseName);

    static bool createBn1Bn2NodesBwd(Bn1Bn2BwdInputs&  inputs,
                                     Bn1Bn2BwdOutputs& outputs,
                                     std::string       baseName,
                                     synDataType       dtype,
                                     bool              isTraining,
                                     NodeList&         nodesList,
                                     bool              locateInSram);

    static bool createBn1Bn2NodesFwd(Bn1Bn2FwdInputs&        inputs,
                                     Bn1Bn2FwdOutputs&       outputs,
                                     float                   momentum,
                                     float                   epsilon,
                                     std::string             baseName,
                                     synDataType             dtype,
                                     bool                    isTraining,
                                     NodeList&               nodesList,
                                     bool                    locateInSram,
                                     std::optional<unsigned> packingFactor = std::nullopt);

    static NodePtr createReshapeNode(TensorPtr inTensor, unsigned dims, eReshapeNodeDirection direction);

    static TensorPtr createWorkspaceVectorsTensor(unsigned elements, unsigned minElements, unsigned vectors, const std::string& name);

    static pNode createConcat1Dto2D(unsigned size, unsigned minsize,TensorPtr input1, TensorPtr input2);

    static void optimizeLowFCD(unsigned          factor,
                               Bn1Bn2FwdInputs&  inputs,
                               Bn1Bn2FwdOutputs& outputs,
                               NodeList&         nodesList,
                               std::string_view  baseName);

    static std::optional<unsigned> shouldOptimizeLowFCD(Bn1Bn2FwdInputs&  inputs,
                                                        Bn1Bn2FwdOutputs& outputs,
                                                        bool              isTraining,
                                                        std::string_view  baseName);

    static void      increaseDim(const TensorPtr& t, unsigned dim, unsigned factor);
    static void      reduceDim(const TensorPtr& t, unsigned dim, unsigned factor);
    static TensorPtr tileTensor(const TensorPtr& tensor, NodeList& nodesList, unsigned factor);
    static TensorPtr avgTensor(const TensorPtr& input, NodeList& nodesList, unsigned factor);
    static TensorPtr sliceTensor(const TensorPtr& output, NodeList& nodesList, unsigned factor);
    static TensorPtr
    packTensor(const TensorPtr& tensor, NodeList& nodesList, unsigned factor, eReshapeNodeDirection direction);
};

#endif //GRAPH_OPTIMIZER_BN_UTILS_H
