#ifndef GRAPHCOMPILER_LOGICALOPNODE_H_
#define GRAPHCOMPILER_LOGICALOPNODE_H_

#include "node.h"

enum AliasDirection
{
    INPUT_TO_OUTPUT,
    OUTPUT_TO_INPUT,

    DIRECTION_MAX
};

class LogicalOpNode : public Node
{
public:
    typedef Node BaseClass;

    LogicalOpNode(const TensorVector& inputs,
                  const TensorVector& outputs,
                  std::string_view    name,
                  AliasDirection      direction,
                  eNodeType           type  = TYPE_DEBUG,
                  ShapeFuncID         sifId = SHAPE_FUNC_MAX_ID);
    LogicalOpNode(const LogicalOpNode& other);
    LogicalOpNode& operator=(const LogicalOpNode& other);

    virtual ~LogicalOpNode() = default;
    virtual void setMustBeDenseIfNeeded() const {} // used by Eager
    virtual bool isLogicalOperation() const override { return true; }
    virtual void runLogicalOperation() const override = 0;
    virtual bool incompatibleWithNextNode(Node::eNodeType type = Node::TYPE_MAX) const;
    virtual bool        isAliasStrided() const { return true; }
    virtual bool        isAliasStrided(unsigned idx) const { return isAliasStrided(); }
    virtual bool validateNode() const override;
    virtual synDataType getRequiredInputType(uint32_t tensorIdx) const override;
    virtual synDataType getRequiredOutputType(uint32_t tensorIdx) const override;
    bool validateAlias() const;
    AliasDirection getAliasDirection() const { return m_aliasTensorDirection; }
    void swapAliasDirection();
    virtual bool        validateNodeForGraph(const HabanaGraph& g) const override;
    virtual bool canSwapAliasDirection() const { return false; }
    virtual NStrideArray calculateAliasStrides(unsigned idx) const;
    virtual TensorPtr    getRealTensor() const;
    virtual TensorVector getAliasTensors() const;
    void runAndSetLogicalOp();
    void resetLogicalOp();

    // in case that logical node is marked as pure logical, then it forbidden to add memcpy to handle the node
    // and it will throw an error when it will be handled
    bool isPureLogical() { return m_isPureLogical; }
    void setIsPureLogical(const bool isPureLogical) { m_isPureLogical = isPureLogical; }

    bool getRunLogicalOperationDone() const { return m_runLogicalOperationDone; }
    virtual bool isRedundantNode() const { return false; };

    // Can handle logical operation on strided real tensor (which is alias to another tensor)
    // without insertion of memcpy node
    virtual bool canHandleStridedRealTensor() const { return false; }

    enum class ResolveStatus
    {
        Success,             // Alias direction is resolved
        MemcpyNeeded,        // Can't resolve alias direction as memcpy should be inserted
        AliasDirectionVaried // Alias direction can be swapped and should be determined from the outside
    };
    using IndicesVec = llvm_vecsmall::SmallVector<uint32_t, MAX_TENSOR_NR>;
    // Resolve alias direction
    // In case the alias direction can't be resolved an apropriate status code will be return
    // requireMemcpy will be filled in case MemcpyNeeded is returned
    ResolveStatus resolveAliasDirection(IndicesVec& requireInputMemcpy, IndicesVec& requireOutputMemcpy);

    bool aliasDirectionValid(AliasDirection direction,
                             IndicesVec&    requireInputMemcpy,
                             IndicesVec&    requireOutputMemcpy) const;

protected:
    void         setRunLogicalOperationDone();
    virtual bool hasNodeIOManagerSpecialization() const override { return true; }
    bool         isBasicRedundant() const;
    bool         aliasDirectionValid(const TensorPtr&    realTensor,
                                     const TensorVector& aliases,
                                     IndicesVec&         realMemcpy,
                                     IndicesVec&         aliasesMemcpy) const;
    bool
    canAllTensorsBeAlias(const TensorPtr& realTensor, const TensorVector& aliases, IndicesVec& requireMemcpy) const;
    bool         constFoldingForReshape();

private:
    AliasDirection m_aliasTensorDirection;
    bool           m_runLogicalOperationDone;
    bool           m_isPureLogical;
};


#endif /* GRAPHCOMPILER_LOGICALOPNODE_H_ */
