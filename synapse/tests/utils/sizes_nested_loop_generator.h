#pragma once

#include <algorithm>
#include <iterator>
#include <vector>
#include <list>
#include <optional>

#include "defs.h"

/**
 * @brief Generator object that generalizes the concept of using nested loops to traverse all elements of a tensor.
 *
 * @param sizes The vector containing the number of elements in each axis
 * @param steps Optional vector of steps defining the traversal increment for each axis. Default is {1}.
 *
 * @details After creating a SizesNestedLoopGenerator, each call to nextState() will return a vector depicting
 *          a legit tensor element coordinate within the input sizes vector range. Order is equivalent to that
 *          of using nested loops to traverse the tensor with the same input sizes.
 *          After the last coordinate state has been returned, isDone() will return true, otherwise false;
 *          A generator can also be reset to it's initial state by invoking the reset() method.
 */
template<typename T = TSize>
class SizesNestedLoopGenerator
{
public:
    using Sizes         = std::vector<T>;
    using States        = std::list<Sizes>;
    using OptionalSteps = std::optional<Sizes>;

    SizesNestedLoopGenerator(const Sizes& sizes, const OptionalSteps steps = std::nullopt)
    : m_states({Sizes {}, Sizes(sizes.size(), static_cast<T>(0))})
    {
        bool isZeroSizedShape = std::any_of(sizes.begin(), sizes.end(), [](const auto& size) { return size == 0; });
        HB_ASSERT(!isZeroSizedShape, "Cannot validate a zero sized shape");

        m_steps.reserve(sizes.size());
        if (!steps.has_value())
        {
            std::fill_n(std::back_inserter(m_steps), sizes.size(), 1);
        }
        else
        {
            HB_ASSERT(sizes.size() == steps.value().size(),
                      "Expecting input sizes length {} == input steps length {}",
                      sizes.size(),
                      steps.value().size());
            std::copy_n(steps.value().begin(), sizes.size(), std::back_inserter(m_steps));
        }

        m_last.reserve(sizes.size());
        std::transform(sizes.begin(), sizes.end(), std::back_inserter(m_last), [](auto size) { return --size; });
    }

    SizesNestedLoopGenerator(const SizesNestedLoopGenerator& other) = delete;
    SizesNestedLoopGenerator(SizesNestedLoopGenerator&& other)      = delete;
    SizesNestedLoopGenerator operator=(const SizesNestedLoopGenerator& other) = delete;
    SizesNestedLoopGenerator operator=(SizesNestedLoopGenerator&& other) = delete;

    /**
     * @brief Whether the generator is done iterating over all items.
     *        The generator is done when states is one step passed
     *        the end:
     *        [prev: last state, next: first state]
     *
     */
    bool isDone() const { return !states().empty() && getPrevState() >= lastState() && getNextState() == firstState(); }

    /**
     * @brief Returns the next state and updates internal state
     */
    Sizes nextState() noexcept
    {
        const auto nextState = getNextState();
        update();
        return nextState;
    }

    /**
     * @brief Reset internal state to the initial state
     *
     */
    void reset()
    {
        states().clear();
        states().insert({Sizes {}, firstState()});
    }

protected:
    Sizes  m_last;
    Sizes  m_steps;
    States m_states;  //  [prev_state]->[next_state]

    const Sizes& steps() const noexcept { return m_steps; }
    const Sizes& lastState() const noexcept { return m_last; }

    void update() noexcept
    {
        if (isDone()) return;

        Sizes newNextState(getNextState());
        for (unsigned dim = 0; dim < newNextState.size(); ++dim)
        {
            if (newNextState.at(dim) >= lastState().at(dim))
            {
                newNextState.at(dim) = 0;
            }
            else
            {
                newNextState.at(dim) += steps().at(dim);
                break;
            }
        }

        states().pop_front();
        states().push_back(newNextState);
    }

    Sizes firstState() const
    {
        HB_ASSERT(!lastState().empty(), "Expecting last state member not empty");
        return Sizes(lastState().size(), static_cast<T>(0));
    }

    const States& states() const noexcept
    {
        return const_cast<const States&>(const_cast<SizesNestedLoopGenerator*>(this)->states());
    }
    States& states() noexcept { return m_states; }

    const Sizes& getPrevState() const noexcept
    {
        return const_cast<const Sizes&>(const_cast<SizesNestedLoopGenerator*>(this)->getPrevState());
    }

    Sizes& getPrevState() noexcept { return states().front(); }

    const Sizes& getNextState() const noexcept
    {
        return const_cast<const Sizes&>(const_cast<SizesNestedLoopGenerator*>(this)->getNextState());
    }

    Sizes& getNextState() noexcept { return states().back(); }
};