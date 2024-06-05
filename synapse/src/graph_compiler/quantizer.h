#pragma once

#include "types.h"
#include "utils.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include <map>

/**
 *  Node Quantizer
 *
 *   Base class with default behavior:
 *   For nodes that do not require any adjustment on input and output scale.
 *   The existing quantization info is being locked to inputs/outputs.
 **/
class Quantizer
{
public:
    virtual ~Quantizer() {}
    virtual void adjustRestrictions(pNode node, bool isForwardPass);
    virtual void adjustScales(HabanaGraph& g, pNode node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput);
    static const int index_na;

protected:
    bool isConflictedWith(pTensor tensor, QuantizationMap& newQuant);
    QuantizationMap getSingleScaleFromTensors(TensorVector tensors, int index=Quantizer::index_na);
    bool isAllLocked(TensorVector& tensors);
    bool isQuantMapEmpty(QuantizationMap& quantMap);
    void revertAndRequantLock(pTensor inputTensor, pNode inputNode, pNode node, QuantizationMap& quantMap);
    void setInputScale(HabanaGraph& g, pNode node, QuantizationMap& quantInfo,
                       std::vector<uint32_t> numSuccessorsPerInput, int index=Quantizer::index_na);
    void lockTensors(pNode node, TensorVector& tensors);
    void setOutputScale(pNode node, QuantizationMap& quantInfo, int index=Quantizer::index_na);
    bool shouldEnforceFixedPoint(TensorVector tensors, int index=Quantizer::index_na);
    bool shouldEnforceInt16Ltd(TensorVector tensors, int index=Quantizer::index_na);
    void enforceFixedPoint(TensorVector tensors, int index=Quantizer::index_na);
    void enforceInt16Ltd(TensorVector tensors, int index=Quantizer::index_na);
    void correctSpecialIntQuantization(TensorVector& tensors);
};

/**
 *   Node Backward Quantizer
 *
 *   For nodes that enforce their output scale on their inputs (e.g Concat).
 *   Only during backward passes.
 **/
class BackwardQuantizer : public Quantizer
{
public:
    void adjustRestrictions(pNode node, bool isForwardPass) override;
    void adjustScales(HabanaGraph& g, pNode node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput) override;
};

/**
 *   Node Backward Don't Care Quantizer
 *
 *   Lock inputs/outputs current quantization info.
 *   Only during backward passes.
 **/
class BackwardDontCareQuantizer : public BackwardQuantizer
{
public:
    void adjustScales(HabanaGraph& g, pNode node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput) override;
};

/**
 *   Node Selective Backward Quantizer
 *
 *   For nodes with multiple outputs that have different scales and enforce their output scale on specific inputs
 *   During backward pass. The rest of the inputs are handled during the forward pass.
 **/
class SelectiveBackwardQuantizer : public BackwardQuantizer
{
public:
    SelectiveBackwardQuantizer(std::map<uint32_t, uint32_t>& outputToInputIndexMap);
    void adjustRestrictions(pNode node, bool isForwardPass) override;
    void adjustScales(HabanaGraph& g, pNode node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput) override;

private:
    std::map<uint32_t, uint32_t> m_outputToInputIndexMap;
};

/**
 *   Align Node Inputs Quantizer
 *
 *   For nodes that enforce the same scale for all inputs.
 *   Only during backward passes (before inputs are locked).
 **/
class AlignInputsQuantizer : public BackwardQuantizer
{
public:
    void adjustRestrictions(pNode node, bool isForwardPass) override;
    void adjustScales(HabanaGraph& g, pNode node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput) override;
};

/**
 *   Node Forward Quantizer
 *
 *   For nodes that enforce inputs scale on their outputs (e.g Split).
 *   If specific input is given, use its scale to enforce.
 *   Only during forward passes.
 **/
class ForwardQuantizer : public Quantizer
{
public:
    ForwardQuantizer();
    ForwardQuantizer(int specificInput);
    void adjustRestrictions(pNode node, bool isForwardPass) override;
    void adjustScales(HabanaGraph& g, pNode node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput) override;

private:
    int m_specificInput;
};

/**
 *   Node "Don't Care" Quantizer
 *
 *   For nodes that do not affect quantization (e.g Flatten).
 *   Can run during forward/backward passes and pass the scale from inputs to outputs and vice versa.
 *   During backward passes, should only transfer scale from output to input if output is locked.
 *   During forward passes, should adjust all scales and make sure both inputs/outputs are locked.
 **/
class DontCareQuantizer : public Quantizer
{
public:
    void adjustRestrictions(pNode node, bool isForwardPass) override;
    void adjustScales(HabanaGraph& g, pNode node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput) override;
};

class BooleanOutputQuantizer : public AlignInputsQuantizer
{
public:
    void adjustRestrictions(pNode node, bool isForwardPass) override;
};

class BooleanInputQuantizer : public BooleanOutputQuantizer
{
public:
    void adjustRestrictions(pNode node, bool isForwardPass) override;
};

class CastQuantizer : public Quantizer
{
public:
    void adjustRestrictions(pNode node, bool isForwardPass) override;
};

class TopKQuantizer : public Quantizer
{
public:
    void adjustRestrictions(pNode node, bool isForwardPass) override;
    static const unsigned indicesInput;
};

class SequenceReverseQuantizer : public Quantizer
{
public:
    void adjustRestrictions(pNode node, bool isForwardPass) override;
    static const unsigned sequenceLensInput;
};

class SequenceLengthQuantizer : public BackwardQuantizer
{
public:
    void adjustRestrictions(pNode node, bool isForwardPass) override;
};

class EmbeddingQuantizer : public Quantizer
{
public:
    EmbeddingQuantizer();
    void adjustRestrictions(pNode node, bool isForwardPass) override;
    void adjustScales(HabanaGraph& g, pNode node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput) override;
    static const unsigned indicesInput;
    static const unsigned dataInput;
    static const unsigned dataOutput;

private:
    QuantizerPtr getEmbeddingQuantizer();
    QuantizerPtr m_byWeightsQuantizer;
    QuantizerPtr m_byOutputQuantizer;
};
