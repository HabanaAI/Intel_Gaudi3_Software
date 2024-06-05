#pragma once

#include "habana_graph.h"
#include "perf_lib_layer_params.h"


/*
 * This pass looks for a specific pattern as seen in BERT topology, and in order to improve performance.
 * in JIRA ticket SW-14390 there is visual diagram of the pattern. which divides in to two parts:
 * 1. the common part: exists once and consumed by each instance of (2)
 * 2. duplicated part: exists in every BERT layer
 *
 * The following assumptions have been made:
 * A. the input to (1) is input mask tensor which holds consecutive 1's to
 *    indicate valid values along the sentence it represents.
 * B. the sub-mul-add sequence in (2) is amplifying the the valid scores from the batch-gemm nodes
 *
 * according to (A) and (B) we can perform reduceSum over inputMask to get the sequenceLen of each sentence.
 * the sequenceLen can be provided to raggedSoftmax kernel
 */

class RaggedSoftmaxFuser {
    private:
        bool constructRaggedSoftmaxPattern(Graph* pattern, unsigned int var);

        bool validatePattern(pNode               reshape2Node,
                             NodeList&           patternNodes,
                             pTensor&            tensorToNorm,
                             pTensor&            normalizedTensor,
                             ns_Softmax::Params& softmaxParams,
                             pNode&              inputMaskMulNode,
                             pNode&              inputMaskCastNode,  // optional node in the pattern
                             pNode&              inputMaskReshapeNode,
                             unsigned int        var);

        void transposeRaggedSoftmaxTensors(pTensor& tensorToNorm,
                                        pTensor& normalizedTensor,
                                        unsigned numFusions,
                                        NodeList& newNodes);

        void createSeqLenTensor(NodeList& newNodes, pTensor inputMaskTensor, pTensor& seqLenTensor);

        HabanaGraph& m_graph;

    public:
        bool fuseRaggedSoftmax(Graph* pattern);
        RaggedSoftmaxFuser(HabanaGraph& g) : m_graph(g) {};

};