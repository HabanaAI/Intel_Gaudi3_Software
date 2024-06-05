#include "layout.h"


#include "utils.h"

#include <algorithm>
#include <bitset>

using namespace gc;

template<typename T>
static std::string toStringVec(const llvm_vecsmall::SmallVectorImpl<T>& elements)
{
    return fmt::format("({})", toString(elements, ',', [](const auto& e) { return e.toString(); }));
}

Permutation::Permutation(const TransposePermutationArray& perm) : m_value(perm.begin(), perm.end()) {}

bool Permutation::isIdentity() const
{
    for (size_t i = 0; i < m_value.size(); ++i)
    {
        if (m_value[i] != i) return false;
    }
    return true;
}

void Permutation::permute(const Permutation& otherPermutation)
{
    if (otherPermutation.size() != size())
    {
        LOG_ERR(DATA_LAYOUT,
                "Cannot permute. Given permutation {} doesn't match the size of {}",
                otherPermutation.toString(),
                toString());
        throw InvalidPermutation();
    }
    DimVector newValue;
    newValue.reserve(size());
    for (auto axis : otherPermutation.m_value)
    {
        if (axis > size())
        {
            LOG_ERR(DATA_LAYOUT, "Cannot permute. Given permutation includes invalid value {}", axis);
            throw InvalidPermutation();
        }
        newValue.push_back(m_value[axis]);
    }
    m_value = std::move(newValue);
}

Permutation Permutation::getInversePermutation() const
{
    Permutation inverse = Permutation(*this);
    // The inverse permutation is the one that creates the identity permutation,
    // when performed on the original permutation.
    for (unsigned i = 0; i < size(); i++)
    {
        inverse.m_value[m_value[i]] = i;
    }
    return inverse;
}

int Permutation::getIndex(uint8_t value) const
{
    auto it = std::find(m_value.begin(), m_value.end(), value);

    if (it != m_value.end())
    {
        return std::distance(m_value.begin(), it);
    }
    return -1;
}

std::string Permutation::toString() const
{
    return fmt::format("({})", fmt::join(m_value, ","));
}

std::string Permutation::toString(const PermutationVector& permutations)
{
    return toStringVec(permutations);
}

void Permutation::resize(unsigned size)
{
    m_value.resize(size);
    std::iota(std::begin(m_value), std::end(m_value), 0);
}

bool Permutation::isValidPermutation() const
{
    std::bitset<HABANA_DIM_MAX> seenDims;
    auto dims = m_value.size();
    for (auto element : m_value)
    {
        if (element >= dims) return false;
        seenDims[element] = true;
    }
    return (seenDims.count() == dims);
}

void Permutation::getValues(TransposePermutationDim* perm, unsigned permArrSize) const
{
    HB_ASSERT(perm != nullptr && permArrSize >= m_value.size(), "invalid permutation pointer");
    for (unsigned i = 0; i < m_value.size(); i++)
    {
        perm[i] = (TransposePermutationDim)m_value[i];
    }
}

unsigned Permutation::permuteDim(unsigned dim) const
{
    if (isIdentity())
    {
        return dim;
    }

    int indexInPerm = getIndex(dim);
    if (indexInPerm != -1)
    {
        return indexInPerm;
    }

    LOG_ERR(DATA_LAYOUT, "Cannot permute. Given dimension ({}) was not found in the permutation ({})", dim, toString());
    throw InvalidPermutation();
}

std::string_view Layout::toString() const {
    if (isDontCare())
    {
        return "Don'tCare";
    }
    else
    {
        return m_layout;
    }
}

std::string Layout::toString(const LayoutVector& layouts)
{
    return toStringVec(layouts);
}

void Layout::getPermutation(const Layout& to, Permutation& permutation) const
{
    if (!isDontCare() && to.isDontCare())
    {
        permutation.setIdentityPermutation(dims());
        return;
    }

    HB_ASSERT(isDontCare() == to.isDontCare(),
              "Cannot permute to {}. Unable to permute {} layout",
              to.toString(),
              toString());
    HB_ASSERT(dims() == to.dims(),
              "Cannot permute {}. Given layout ({}) does not have the same amount of dims.",
              toString(),
              to.toString());

    for (const char& c : to.m_layout)
    {
        std::size_t toIndex = m_layout.find(c);

        HB_ASSERT(toIndex != std::string::npos,
                  "Cannot permute {}. Given layout ({}) is not a permutation of {}",
                  toString(),
                  to.toString(),
                  toString());

        permutation.addValueToPermutation(toIndex);
    }
    HB_ASSERT(permutation.size() == dims(), "Permutation dim is not equal to Layout dims");
}

Layout Layout::permute(const Permutation &p) const
{
    if (isDontCare())
    {
        return Layout();
    }

    if (p.isIdentity())
    {
        return Layout(m_layout);
    }

    HB_ASSERT(p.getIndex(PERMUTATION_DONTCARE_AXIS) == -1,
              "Cannot permute {}. Given permutation ({}) is invalid",
              toString(),
              p.toString());
    HB_ASSERT(p.size() == m_layout.size(),
              "Cannot permute {}. Given permutation ({}) has a different length",
              toString(),
              p.toString());

    std::string newLayout;

    for (auto axis : p.getValues())
    {
        newLayout.push_back(m_layout[axis]);
    }

    return Layout(newLayout);
}

unsigned Layout::getIndexByName(char name) const
{
    for (int i = 0; i < m_layout.size(); i++)
    {
        if (m_layout[i] == name)
        {
            return i;
        }
    }
    throw LayoutException();
}
