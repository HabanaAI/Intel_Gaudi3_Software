#pragma once

class SlicingStrategy;
class MmeSlicingStrategy;

class StrategyVisitor
{
public:
    virtual ~StrategyVisitor() {}
    virtual void visit(SlicingStrategy&) = 0;
    virtual void visit(MmeSlicingStrategy&) = 0;
};
