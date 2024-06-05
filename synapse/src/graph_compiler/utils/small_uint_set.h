#pragma once

#include <type_traits>
#include <cstdint>
#include <infra/defs.h>

// This template is a set of SMALL unsigned integers.
// The range of the integers that the set can contain
// is 0 <= x < Top, where Top <= 64.
// It is implemented in terms of bit mask.

template <unsigned Top>
class SmallUintSet
{
    public:
        SmallUintSet() : m_data(0) {};
        SmallUintSet(const SmallUintSet& other) : m_data(other.m_data) {};
        SmallUintSet(unsigned num) : m_data(BIT<<verify(num)) {}

        void insert(unsigned num) { verify(num); m_data |= (BIT<<num); }
        void remove(unsigned num) { verify(num); m_data &= ~(BIT<<num); }
        bool contains(unsigned num) { verify(num); return m_data & (BIT<<num); }

        friend SmallUintSet operator| (SmallUintSet a, SmallUintSet b) { return SmallUnitSet(a.m_data | b.m_data, true); }
        friend SmallUintSet operator& (SmallUintSet a, SmallUintSet b) { return SmallUnitSet(a.m_data & b.m_data, true); }
        friend SmallUintSet operator^ (SmallUintSet a, SmallUintSet b) { return SmallUnitSet(a.m_data ^ b.m_data, true); }
        friend SmallUintSet operator- (SmallUintSet a, SmallUintSet b) { return SmallUnitSet(a.m_data & ~b.m_data, true); }

        SmallUintSet& operator|= (SmallUintSet other) { m_data |= other.m_data; return *this; }
        SmallUintSet& operator&= (SmallUintSet other) { m_data &= other.m_data; return *this; }
        SmallUintSet& operator^= (SmallUintSet other) { m_data ^= other.m_data; return *this; }
        SmallUintSet& operator-= (SmallUintSet other) { m_data &= ~other.m_data; return *this; }

        // no iterators at this time!
        template <typename Func>
        bool any(Func&& f)
        {
            for (unsigned i = 0; i < Top; ++i)
            {
                if (m_data & (BIT<<i))
                {
                    if (f(i)) return true;
                }
            }
            return false;
        }

        template <typename Func>
        bool all(Func&& f)
        {
            for (unsigned i = 0; i < Top; ++i)
            {
                if (m_data & (BIT<<i))
                {
                    if (!f(i)) return false;
                }
            }
            return true;
        }


    private:
        template <unsigned X>
        struct Disallow
        {
            static_assert(X < 64, "Small uint set is too large");
            using type = void;
        };

        using Underlying =
        typename std::conditional<
            (Top < 32),
            uint32_t,
            typename std::conditional<
                (Top < 64),
                uint64_t,
                typename Disallow<Top>::type
            >::type
        >::type;
        Underlying m_data;
        static Underlying constexpr BIT = 1;

        // an explicit constructor from underlying bitset
        // To avoid conflict with the singleton constructor,
        // add a dummy argument
        SmallUintSet (Underlying s, bool dummy) : m_data(s) {}

        unsigned verify(unsigned num)
        {
            HB_ASSERT(num < Top, "Argument for a small set operation is too big!");
            return num;
        }
};
