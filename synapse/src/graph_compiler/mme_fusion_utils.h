#pragma once

#include "types.h"
#include "node.h"


class MMEFusionUtils
{
public:
    static unsigned getOutputChannels(const pNode& mmeNode, bool isConvNode);

    static bool canBeBias(uint8_t biasCandidate, const pNode& addNode, const pNode& mmeNode, HabanaGraph& g);

    static bool canBeCin(uint8_t cinCandidate, const pNode& addNode, const pNode& mmeNode);

};
