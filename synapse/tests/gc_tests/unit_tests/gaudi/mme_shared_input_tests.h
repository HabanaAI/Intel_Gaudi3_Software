#pragma once

#include "sram_management_fe_test.h"
#include "utils.h"
#include "tensor.h"

namespace gaudi
{
/* This test suit describe a situation where we have 2 MME nodes with shared input.
 * Conventions - input X,Y , output P are of the first node
 *               input Y,Z , output Q are if the second node
 *               (inputY is the shared input)
 * input order is determined by the methods - firstOperationTensors\secondOperationTensors - can be overridden.
 */
class MMEInterleaveTest : public SRAMManagementTest
{
public:
    MMEInterleaveTest() { getGraph().constructBPGraph(); };
    virtual ~MMEInterleaveTest() = default;
    void createTensors(const SizeArray& inputXSize,
                       const SizeArray& inputYSize /*shared*/,
                       const SizeArray& outputPSize,
                       const SizeArray& inputZSize,
                       const SizeArray& outputQSize);
    void addNodesToGraph(GaudiGraph& graph);
protected:
    virtual synDataType getType() {return m_type;}
    virtual void setType(synDataType type) {m_type = type;}
    virtual TensorVector firstOperationTensors() {return {m_inputX, m_inputY, m_outputP};}
    virtual TensorVector secondOperationTensors() {return {m_inputY, m_inputZ, m_outputQ};}
    virtual void* getParams(unsigned opIdx) = 0;
    virtual size_t getParamsSize() = 0;
    virtual const char* getFirstNodeGuid() = 0;
    virtual const char* getSecondNodeGuid() = 0;
    virtual const char* getFirstNodeName() {return "node1";}
    virtual const char* getSecondNodeName() {return "node2";}
    void findAndAddCandidate(pBundle& bundle, pMmeSlicingStrategy& strategy);
    pTensor m_inputX  = nullptr;
    pTensor m_inputY  = nullptr;
    pTensor m_outputP = nullptr;
    pTensor m_inputZ  = nullptr;
    pTensor m_outputQ = nullptr;
    synDataType m_type = syn_type_float;
};

class DedxDedwInterleaveTest : public MMEInterleaveTest
{
public:
    void createTensors(const SizeArray& xSize,
                       uint32_t yChannel,
                       const synConvolutionParams& convParams);

protected:
    virtual void* getParams(unsigned opIdx) override {return &m_convParams;}
    virtual size_t getParamsSize() override {return sizeof(m_convParams);}
    virtual const char* getFirstNodeGuid() override {return NodeFactory::deDwNodeTypeName;}
    virtual const char* getSecondNodeGuid() override {return NodeFactory::deDxNodeTypeName;}
    virtual const char* getFirstNodeName() override {return "dedw";}
    virtual const char* getSecondNodeName() override {return "dedx";}
private:
    synConvolutionParams m_convParams;
};

class GEMMInterleaveTest : public MMEInterleaveTest
{
protected:
    using ExpansionCandidatesSet = std::unordered_map<pNode, pBundleExpansion>;
    Solution runTest(unsigned commonSize, unsigned xChunk, unsigned yChunk, unsigned zChunk);
    void setSharedOperandPosition(unsigned firstGemm, unsigned secondGemm);
    virtual void* getParams(unsigned opIdx) override;
    virtual size_t getParamsSize() override {return sizeof(m_paramsA);}
    virtual const char* getFirstNodeGuid() override {return NodeFactory::gemmNodeTypeName;}
    virtual const char* getSecondNodeGuid() override {return NodeFactory::gemmNodeTypeName;}
    virtual const char* getFirstNodeName() override {return "gemm1";}
    virtual const char* getSecondNodeName() override {return "gemm2";}
    virtual TensorVector firstOperationTensors() override;
    virtual TensorVector secondOperationTensors() override;
    void transposeSharedInput(unsigned nodeIdx, unsigned operandIdx);
    pMmeSlicingStrategy makeStrategy(unsigned commonSize, unsigned inputXChunk,
                                  unsigned inputYChunk, pBundle& bundle);
private:
    synGEMMParams m_paramsA;
    synGEMMParams m_paramsB;
    unsigned m_sharedOperandPosition_firstGEMM = 0;
    unsigned m_sharedOperandPosition_secondGEMM = 0;
};

class ConvInterleaveTest : public GEMMInterleaveTest
{
protected:
    virtual void* getParams(unsigned opIdx) override {return &m_convParams;}
    virtual size_t getParamsSize() override {return sizeof(m_convParams);}
    virtual const char* getFirstNodeGuid() override {return NodeFactory::convolutionNodeTypeName;}
    virtual const char* getSecondNodeGuid() override {return NodeFactory::convolutionNodeTypeName;}
    virtual const char* getFirstNodeName() override {return "conv1";}
    virtual const char* getSecondNodeName() override {return "conv2";}

private:
    synConvolutionParams m_convParams;
};

} // namespace gaudi
