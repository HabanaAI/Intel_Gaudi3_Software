#pragma once
#include <type_traits>
#include <mutex>
/**
 * base node type for user node types
 * user nodes should be declared this way:
 * @code
 * struct UserNode : public IntrusiveStackNodeBase<UserNode>
 * {
 *   // members
 * };
 * @endcode
 * @tparam TDerivedNode user node type
 */
template<class TDerivedNode>
struct IntrusiveStackNodeBase
{
    TDerivedNode* next;
};

/**
 * lock-based intrusive stack
 * does not allocate/deallocate any memory
 * @tparam TNode node type that must be publicly derived from IntrusiveStackNodeBase
 */
template<class TNode>
class ConcurrentIntrusiveStack
{
public:
    ConcurrentIntrusiveStack() = default;

    /**
     * push a new node into stack
     * complexity : O(1)
     * @param newNode
     */
    bool push(TNode* newNode)
    {
        if (newNode != nullptr)
        {
            std::lock_guard<std::mutex> lck(m_mtx);
            newNode->next = m_head;
            m_head        = newNode;
            return true;
        }
        return false;
    }
    /**
     * pop a node from stack
     * complexity : O(1)
     * @param newNode pointer to a popped node or nullptr if stack is empty
     */
    TNode* pop()
    {
        std::lock_guard<std::mutex> lck(m_mtx);

        TNode* nodeToPop = m_head;
        if (nodeToPop != nullptr)
        {
            m_head = m_head->next;
        }
        return nodeToPop;
    }

    ConcurrentIntrusiveStack(ConcurrentIntrusiveStack const&) = delete;
    ConcurrentIntrusiveStack(ConcurrentIntrusiveStack&&)      = delete;

private:
    static_assert(std::is_base_of<IntrusiveStackNodeBase<TNode>, TNode>::value,
                  "TNode must be publicly derived from IntrusiveStackNodeBase");
    TNode*     m_head {nullptr};
    std::mutex m_mtx;
};
