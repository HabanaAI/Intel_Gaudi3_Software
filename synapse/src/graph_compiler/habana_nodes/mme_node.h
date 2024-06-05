#pragma once

#include "node.h"
#include "synapse_common_types.h"
#include "node_visitor.h"

enum eMmeDataType
{
    eMmeDataTypeUnknown = -1,
    eMmeDataTypeInt,
    eMmeDataTypeFloat,
};

struct MmeExpBias
{
    SmallVector<unsigned, 2> fp8BiasIn;
    unsigned fp8BiasOut;
};

class MmeBrainIfc;
class MmeNode : public Node
{
public:
    DEFINE_VISITOR_METHOD
    typedef Node BaseClass;

    MmeNode(const TensorVector& inputs,
            const TensorVector& outputs,
            std::string_view    name,
            eNodeType           type,
            ShapeFuncID         sifId = SHAPE_FUNC_MAX_ID);
    MmeNode(const MmeNode& other);

    virtual bool validateNode() const override;
    virtual bool validateNodeLayout() const override;
    bool validateNodeForGraph(const HabanaGraph& g) const override;

    virtual NodeROI generateRoi() const override;
    virtual synDataType getRequiredInputType(uint32_t tensorIdx)  const override;
    virtual synDataType getRequiredOutputType(uint32_t tensorIdx) const override;

    virtual bool hasBias() const;
    virtual bool hasCin() const;
    virtual void addMMETensor(const TensorPtr& tensor, unsigned tensorIndex); // Add live opB zpb/Bias/LUT tensor
    virtual bool isOperandTransposed(const TensorPtr& tensor) const = 0;

    virtual TensorSemanticType getParamSemanticType(const TensorPtr& param) const override;
    virtual unsigned           getKDimIndex() override;
    eMmeDataType getMmeDataType()                                  const;

    static void                     printMmeParams(const synConvolution3DParamsV2& mmeParams);
    virtual std::string_view        getEngineTypeStr() const override;
    static synConvolution3DParamsV2 convert2DconvTo3DconvStruct(const synConvolutionParamsV2& userConvParam);
    static synConvolutionParamsV2   convert3DconvTo2DconvStruct(const synConvolution3DParamsV2& cov3DParams);
    virtual bool canBeConvertedToGEMM() const = 0;

    static std::string synConvolutionParamsToString(const synConvolutionParamsV2& params);
    static std::string synConvolution3DParamsToString(const synConvolution3DParamsV2& params);

    virtual bool     isConvolution() const { return false; }
    virtual bool     is3DConvolution() const { return false; }
    HabanaDeviceType getNodeDeviceType() const override { return DEVICE_MME; }

    virtual bool isDynamicPaddingConvolution() const { return false; }
    static bool  isDmaOperation(const NodePtr& node);
    bool         isDmaOperation() const;
    virtual bool isTransposeViaGemm() const { return false; }

    void                               initMmeBrainIfc(synDeviceType deviceType);
    const std::shared_ptr<MmeBrainIfc> getMmeBrainIfc() const { return m_mmeBrainIfc; }
    bool                               isCdIndexSpaceDim(unsigned indexSpaceDim) const;
    void                               setCdPerforated(bool cdPerforated) { m_cdPerforated = cdPerforated; }
    bool                               isCdPerforated() const { return m_cdPerforated; };

    const MmeExpBias& getMmeExpBias() const { return m_mmeExpBias; }
    void              setMmeExpBias(const MmeExpBias& mmeExpBias) { m_mmeExpBias = mmeExpBias; }
    bool              isOutputTensorPartialWrites(unsigned tensorOutputIndex) const;

protected:
    synDataType getRequiredInputTypeInt(uint32_t tensorIdx) const;
    synDataType getRequiredInputTypeFloat(uint32_t tensorIdx) const;
    synDataType getRequiredOutputTypeInt(uint32_t tensorIdx) const;
    synDataType getRequiredOutputTypeFloat(uint32_t tensorIdx) const;
    synDataType getRequiredOutputType(uint32_t tensorIdx, synDataType weightsDataType) const;
    synDataType getRequiredInputType(uint32_t tensorIdx, synDataType weightsDataType) const;
    synDataType getRequiredWeightsDataType() const;
    synDataType getDefaultRequiredDataType(bool isFloat = true) const;
    synDataType getRequiredTensorType(uint32_t tensorIdx, bool isInput) const;

    bool isTensorDataTypeSupported(uint32_t tensorIdx, synDataType tensorDataType, bool isInput) const;
    // Set tensorDataType to match with the weights tensor constraints, return whether a change was made
    bool matchTensorDataTypeToWeights(uint32_t     tensorIdx,
                                      synDataType& tensorDataType,
                                      synDataType  weightsDataTypeAfterCast,
                                      bool         isInput) const;

    static bool areConvParamsGEMMConvertible(const synConvolution3DParamsV2& params);

private:
    MmeExpBias m_mmeExpBias;

    bool                         m_cdPerforated = false;
    std::shared_ptr<MmeBrainIfc> m_mmeBrainIfc;
};

typedef std::set<MMENodePtr, NodeComparator> MMENodeSet;