#pragma once

#include <optional>
#include "node.h"

class SequenceStrategy
{
public:
    // Start condition: returns true if n is valid as a start of a sequence
    virtual bool isSeqStart(const NodePtr& n) const = 0;

    // Continue condition: returns true if n is a valid node to continue current sequence
    virtual bool canContinueSeq(const NodePtr& n) const = 0;

    // End condition: returns true if the sequence is complete.
    virtual bool isIdentity(const NodeVector& seq) const = 0;

    virtual bool isFusibleSequence(const NodeVector& seq) const { return false; }

    virtual std::optional<NodePtr> fuseSequence(const NodeVector& seq) const
    {
        HB_ASSERT(false, "fuse method is not implemented for this strategy");
        return std::nullopt;
    }
};