#pragma once

#include "tpc_kernel_lib_interface.h"
#include "infra/defs.h"

#include "synapse_types.h"
#include <array>
#include <string>
#include <vector>
#include <numeric>
#include <ostream>
#include <exception>
#include "transpose_permutation.h"

class LayoutException : public std::exception
{
};

class InvalidPermutation : public LayoutException
{
};

class InvalidLayout : public LayoutException
{
};

namespace gc
{
    constexpr uint8_t PERMUTATION_DONTCARE_AXIS = 0xff;

    class Permutation
    {
    public:
        // a local buffer for 2 elements avoids memory allocations for most use cases
        // while not making the vector's local storage too big.
        using PermutationVector = llvm_vecsmall::SmallVector<Permutation, 2>;

        Permutation(unsigned dims) { resize(dims); }  // Get Identity Permutation
        Permutation() = default;
        Permutation(const DimVector& v) : m_value(v) {}
        Permutation(const TransposePermutationArray& perm);

        bool operator==(const Permutation& rhs) const { return m_value == rhs.m_value; }
        bool operator!=(const Permutation& rhs) const { return m_value != rhs.m_value; }
        Permutation& operator=(const Permutation& rhs) = default;

        static std::string toString(const PermutationVector& permutations);

        std::string      toString() const;
        unsigned         size() const { return m_value.size(); }
        const DimVector& getValues() const { return m_value; }
        void             clear() { m_value.clear(); }

        void getValues(TransposePermutationDim* perm, unsigned permArrSize) const;

        void setIdentityPermutation(unsigned dims) { resize(dims); }
        bool isIdentity() const;
        bool isEmpty() const { return m_value.empty(); }

        /* Get the index in the permutation of the given value (-1 if doesn't exist) */
        int getIndex(uint8_t value) const;

        Permutation getInversePermutation() const;

        /* Apply another permutation on the permutation */
        void permute(const Permutation& otherPermutation);

        void addValueToPermutation(uint8_t v) { m_value.push_back(v); }

        unsigned permuteDim(unsigned dim) const;

        template<typename T>
        void permute(const T* in, T* out, int size) const
        {
            HB_ASSERT(m_value.size() == size, "permutation length should be in the same size as the shape to permute");

            for (int i = 0; i < size; ++i)
            {
                out[i] = in[m_value[i]];
            }
        }

        template<typename T>
        void permuteShape(T* shape, int shapeSize) const
        {
            if (isIdentity()) return;
            using ShapeVector = llvm_vecsmall::SmallVector<T, HABANA_DIM_MAX>;
            ShapeVector origShape(shape, shape + shapeSize);
            permute(origShape.data(), shape, shapeSize);
        }

        template<typename T>
        void permuteShape(std::vector<T>& shape) const
        {
            permuteShape(shape.data(), shape.size());
        }

        void permuteShape(SizeVector& shape) const { permuteShape(shape.data(), shape.size()); }

        bool isValidPermutation() const;

    private:
        void resize(unsigned size);

        DimVector m_value;
    };

    class Layout
    {
    public:
        // a local buffer for 2 elements avoids memory allocations for most use cases
        // while not making the vector's local storage too big.
        using LayoutVector = llvm_vecsmall::SmallVector<Layout, 2>;

        Layout(std::string_view layout) : m_layout(layout) {}
        Layout() = default;
        bool operator==(const Layout &rhs)             const { return m_layout == rhs.m_layout; }
        bool operator!=(const Layout &rhs)             const { return m_layout != rhs.m_layout; }

        static std::string toString(const LayoutVector& layouts);

        bool             isDontCare() const { return m_layout.empty(); }
        std::string_view toString() const;

        unsigned    dims()                             const { return m_layout.size(); }

        void getPermutation(const Layout& to, Permutation& permutation) const;

        Layout      permute(const Permutation &p)      const;
        unsigned    getIndexByName(char name)          const;

        bool        isRestricted()                     const { return m_isRestricted; };
        void        setAsRestricted()                        { m_isRestricted = true; };
    private:
        std::string m_layout;
        bool        m_isRestricted = false;
    };
    }  // namespace gc

    using LayoutVector      = gc::Layout::LayoutVector;
    using PermutationVector = gc::Permutation::PermutationVector;
