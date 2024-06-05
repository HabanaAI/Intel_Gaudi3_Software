#pragma once
#include "include/mme_common/mme_common_enum.h"
#include "graph_compiler/types.h"
#include "synapse_common_types.h"
#include "tensor_shape.h"
namespace MmeCommon
{
class MmeAuxTensorHandler
{
public:
    MmeAuxTensorHandler()  = default;
    ~MmeAuxTensorHandler() = default;
    void addAuxTensorsForCdParallel(MMENodePtr& mmeNode, unsigned concurrencyLevel);
    void addUnitMatrixToNode(MMENodePtr& mmeNode, synDataType dtype);

private:
    TensorShape
    getScratchPadShape(const MMENodePtr& mmeNode, const TensorShape& outputShape, unsigned concurrencyLevel);
    TensorShape getReduceShape(unsigned concurrencyLevel);
    void        addDataToReduceTensor(TensorPtr& reduceTensor, synDataType dtype, unsigned concurrencyLevel);
    TensorPtr   createSparseUnitTensor(synDataType dtype);
    char*       getOneByDtype(synDataType dataType);
};

class MmeServices
{
public:
    MmeServices()  = default;
    ~MmeServices() = default;
    MmeAuxTensorHandler& getAuxHandler() { return m_auxHandler; }
    void                 addAuxTensorToNode(MMENodePtr& mmeNode, const MmeStrategy& strategy);
    void                 adjustDcoreRoisForCdParallel(MMENodePtr& mmeNode, const MmeStrategy& strategy);
    static synDataType   getDtypeForTranspose(const Tensor& tensor);
    enum ePattern
    {
        // CD_PARALLEL: gaudi3, concurrency over 4 dcores
        CD_PARALLEL,
        // CD_CONCURRENCY: gaudi2,3 , concurrency over all MME cores - currently not used.
        CD_CONCURRENCY,
        // TRANSPOSE: gaudi3, transpose via gemm
        TRANSPOSE_VIA_GEMM,
        PATTERNS_NR
    };
    static ePattern matchPattern(const MMENodePtr& mmeNode, const MmeStrategy& strategy);
    static bool     checkTransposeViaGemmPattern(const MMENodePtr& mmeNode);

private:
    MmeAuxTensorHandler m_auxHandler;
};

}  // namespace MmeCommon
