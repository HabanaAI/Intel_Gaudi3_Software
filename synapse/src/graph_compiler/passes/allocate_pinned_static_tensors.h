#pragma once
#include "habana_graph.h"


static const int NUM_OF_FIRST_LAYERS_TO_PIN       = 2;
static const int MAX_SIZE_IN_FIRST_LAYERS_TO_PIN  = 100 * 1024;   // 100KB - maximum size of static tensor in first layer to prioritize for pinning

struct TensorScore
{
    std::shared_ptr<Tensor> m_tensor;
    float                   m_score;
    unsigned                id = 0; // for debug
};

class Compare
{
public:
    bool operator() (const TensorScore &tensor1, const TensorScore &tensor2) const
    {
        return tensor1.m_score >= tensor2.m_score;
    }
};

bool allocatePinnedStaticTensors(HabanaGraph& g);
