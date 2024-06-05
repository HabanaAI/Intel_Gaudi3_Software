// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef BASE_CONTAINERS_small_set_H_
#define BASE_CONTAINERS_small_set_H_

#include <stddef.h>

#include <limits>
#include <algorithm>
#include <set>
#include <new>
#include <utility>
#include <cassert>

namespace chromium_small_set {

inline constexpr size_t kUsingFullSetSentinel =
    std::numeric_limits<size_t>::max();

// This is an adaptation of small_map to set use case.
// small_set implementation is not found in chromium containers.
// for more information please read small_map description found in small_map.h file.

namespace internal {

template <typename NormalSet>
class small_set_default_init {
 public:
  void operator()(NormalSet* set) const { new (set) NormalSet(); }
};

// has_key_equal<M>::value is true iff there exists a type M::key_equal. This is
// used to dispatch to one of the select_equal_key<> metafunctions below.
template <typename M>
struct has_key_equal {
  typedef char sml;  // "small" is sometimes #defined so we use an abbreviation.
  typedef struct { char dummy[2]; } big;
  // Two functions, one accepts types that have a key_equal member, and one that
  // accepts anything. They each return a value of a different size, so we can
  // determine at compile-time which function would have been called.
  template <typename U> static big test(typename U::key_equal*);
  template <typename> static sml test(...);
  // Determines if M::key_equal exists by looking at the size of the return
  // type of the compiler-chosen test() function.
  static const bool value = (sizeof(test<M>(0)) == sizeof(big));
};
template <typename M> const bool has_key_equal<M>::value;

// Base template used for set types that do NOT have an M::key_equal member,
// e.g., std::set<>. These sets have a strict weak ordering comparator rather
// than an equality functor, so equality will be implemented in terms of that
// comparator.
//
// There's a partial specialization of this template below for set types that do
// have an M::key_equal member.
template <typename M, bool has_key_equal_value>
struct select_equal_key {
  struct equal_key {
    bool operator()(const typename M::key_type& left,
                    const typename M::key_type& right) {
      // Implements equality in terms of a strict weak ordering comparator.
      typename M::key_compare comp;
      return !comp(left, right) && !comp(right, left);
    }
  };
};

// Partial template specialization handles case where M::key_equal exists, e.g.,
// unordered_set<>.
template <typename M>
struct select_equal_key<M, true> {
  typedef typename M::key_equal equal_key;
};

}  // namespace internal

template <typename NormalSet,
          size_t kArraySize = 4,
          typename EqualKey = typename internal::select_equal_key<
              NormalSet,
              internal::has_key_equal<NormalSet>::value>::equal_key,
          typename SetInit = internal::small_set_default_init<NormalSet>>
class small_set {
  static_assert(kArraySize > 0, "Initial size must be greater than 0");
  static_assert(kArraySize != kUsingFullSetSentinel,
                "Initial size out of range");

 public:

  class const_iterator;

  typedef typename NormalSet::key_type key_type;
  typedef typename NormalSet::value_type value_type;
  typedef EqualKey key_equal;
  typedef const_iterator iterator;

  small_set() : size_(0), functor_(SetInit()) {}

  explicit small_set(const SetInit& functor) : size_(0), functor_(functor) {}

  // Allow copy-constructor and assignment, since STL allows them too.
  small_set(const small_set& src) {
    // size_ and functor_ are initted in InitFrom()
    InitFrom(src);
  }

  void operator=(const small_set& src) {
    if (&src == this) return;

    // This is not optimal. If src and dest are both using the small array, we
    // could skip the teardown and reconstruct. One problem to be resolved is
    // that the value_type itself is pair<const K, V>, and const K is not
    // assignable.
    Destroy();
    InitFrom(src);
  }

  ~small_set() { Destroy(); }

  class const_iterator {
   public:
    typedef typename NormalSet::const_iterator::iterator_category
        iterator_category;
    typedef typename NormalSet::const_iterator::value_type value_type;
    typedef typename NormalSet::const_iterator::difference_type difference_type;
    typedef typename NormalSet::const_iterator::pointer pointer;
    typedef typename NormalSet::const_iterator::reference reference;

    inline const_iterator() : array_iter_(nullptr) {}

    // Non-explicit constructor lets us convert regular iterators to const
    // iterators.
    inline const_iterator(const iterator& other)
        : array_iter_(other.array_iter_), set_iter_(other.set_iter_) {}

    inline const_iterator& operator++() {
      if (array_iter_ != nullptr) {
        ++array_iter_;
      } else {
        ++set_iter_;
      }
      return *this;
    }

    inline const_iterator operator++(int /*unused*/) {
      const_iterator result(*this);
      ++(*this);
      return result;
    }

    inline const_iterator& operator--() {
      if (array_iter_ != nullptr) {
        --array_iter_;
      } else {
        --set_iter_;
      }
      return *this;
    }

    inline const_iterator operator--(int /*unused*/) {
      const_iterator result(*this);
      --(*this);
      return result;
    }

    inline const value_type* operator->() const {
      return array_iter_ ? array_iter_ : set_iter_.operator->();
    }

    inline const value_type& operator*() const {
      return array_iter_ ? *array_iter_ : *set_iter_;
    }

    inline bool operator==(const const_iterator& other) const {
      if (array_iter_ != nullptr) {
        return array_iter_ == other.array_iter_;
      }
      return other.array_iter_ == nullptr && set_iter_ == other.set_iter_;
    }

    inline bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

   private:
    friend class small_set;
    inline explicit const_iterator(const value_type* init)
        : array_iter_(init) {}
    inline explicit const_iterator(
        const typename NormalSet::const_iterator& init)
        : array_iter_(nullptr), set_iter_(init) {}

    const value_type* array_iter_;
    typename NormalSet::const_iterator set_iter_;
  };

  const_iterator find(const key_type& key) const {
    key_equal compare;

    if (UsingFullSet()) {
      return const_iterator(set()->find(key));
    }

    for (size_t i = 0; i < size_; ++i) {
      if (compare(array_[i], key)) {
        return const_iterator(array_ + i);
      }
    }
    return const_iterator(array_ + size_);
  }

  // Invalidates iterators.
  std::pair<iterator, bool> insert(const value_type& x) {
    key_equal compare;

    if (UsingFullSet()) {
      std::pair<typename NormalSet::iterator, bool> ret = set_.insert(x);
      return std::make_pair(iterator(ret.first), ret.second);
    }

    for (size_t i = 0; i < size_; ++i) {
      if (compare(array_[i], x)) {
        return std::make_pair(iterator(array_ + i), false);
      }
    }

    if (size_ == kArraySize) {
      ConvertToRealSet();  // Invalidates all iterators!
      std::pair<typename NormalSet::iterator, bool> ret = set_.insert(x);
      return std::make_pair(iterator(ret.first), ret.second);
    }

    assert(size_ < kArraySize);
    new (&array_[size_]) value_type(x);
    return std::make_pair(iterator(array_ + size_++), true);
  }

  // Invalidates iterators.
  template <class InputIterator>
  void insert(InputIterator f, InputIterator l) {
    if (UsingFullSet()) {
      set_.insert(f, l);
      return;
    }
    assert(size_ <= kArraySize);
    key_equal compare;
    for(auto current = f; current != l; ++current)
    {
        if (std::none_of(std::begin(array_), 
                        std::begin(array_) + size_, 
                        [&current, &compare](const auto& element){ 
                          return compare(element, *current);
                        })) {
          if (size_ == kArraySize) {
            ConvertToRealSet();
            set_.insert(current, l);
            return;              
          }
          new (&array_[size_++]) value_type(*current);
        }
    }
  }

  // Invalidates iterators.
  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    key_equal compare;

    if (UsingFullSet()) {
      std::pair<typename NormalSet::iterator, bool> ret =
          set_.emplace(std::forward<Args>(args)...);
      return std::make_pair(iterator(ret.first), ret.second);
    }

    value_type x(std::forward<Args>(args)...);
    for (size_t i = 0; i < size_; ++i) {
      if (compare(array_[i], x)) {
        return std::make_pair(iterator(array_ + i), false);
      }
    }

    if (size_ == kArraySize) {
      ConvertToRealSet();  // Invalidates all iterators!
      std::pair<typename NormalSet::iterator, bool> ret =
          set_.emplace(std::move(x));
      return std::make_pair(iterator(ret.first), ret.second);
    }

    assert(size_ < kArraySize);
    new (&array_[size_]) value_type(std::move(x));
    return std::make_pair(iterator(array_ + size_++), true);
  }

  iterator begin() {
    return UsingFullSet() ? iterator(set_.begin()) : iterator(array_);
  }

  const_iterator begin() const {
    return UsingFullSet() ? const_iterator(set_.begin())
                          : const_iterator(array_);
  }

  iterator end() {
    return UsingFullSet() ? iterator(set_.end()) : iterator(array_ + size_);
  }

  const_iterator end() const {
    return UsingFullSet() ? const_iterator(set_.end())
                          : const_iterator(array_ + size_);
  }

  void clear() {
    if (UsingFullSet()) {
      set_.~NormalSet();
    } else {
      for (size_t i = 0; i < size_; ++i) {
        array_[i].~value_type();
      }
    }
    size_ = 0;
  }

  // Invalidates iterators. Returns iterator following the last removed element.
  iterator erase(const iterator& position) {
    if (UsingFullSet()) {
      return iterator(set_.erase(position.set_iter_));
    }

    size_t i = static_cast<size_t>(position.array_iter_ - array_);
    // TODO(crbug.com/817982): When we have a checked iterator, this CHECK might
    // not be necessary.
    assert(i <= size_);
    array_[i].~value_type();
    --size_;
    if (i != size_) {
      new (&array_[i]) value_type(std::move(array_[size_]));
      array_[size_].~value_type();
      return iterator(array_ + i);
    }
    return end();
  }

  size_t erase(const key_type& key) {
    iterator iter = find(key);
    if (iter == end()) {
      return 0;
    }
    erase(iter);
    return 1;
  }

  size_t count(const key_type& key) const {
    return (find(key) == end()) ? 0 : 1;
  }

  size_t size() const { return UsingFullSet() ? set_.size() : size_; }

  bool empty() const { return UsingFullSet() ? set_.empty() : size_ == 0; }

  // Returns true if we have fallen back to using the underlying set
  // representation.
  bool UsingFullSet() const { return size_ == kUsingFullSetSentinel; }

  inline NormalSet* set() {
    assert(UsingFullSet());
    return &set_;
  }

  inline const NormalSet* set() const {
    assert(UsingFullSet());
    return &set_;
  }

 private:
  // When `size_ == kUsingFullSetSentinel`, we have switched storage strategies
  // from `array_[kArraySize] to `NormalSet set_`. See ConvertToRealSet and
  // UsingFullSet.
  size_t size_;

  SetInit functor_;

  // We want to call constructors and destructors manually, but we don't want
  // to allocate and deallocate the memory used for them separately. Since
  // array_ and set_ are mutually exclusive, we'll put them in a union.
  union {
    value_type array_[kArraySize];
    NormalSet set_;
  };

  void ConvertToRealSet() {
    // Storage for the elements in the temporary array. This is intentionally
    // declared as a union to avoid having to default-construct |kArraySize|
    // elements, only to move construct over them in the initial loop.
    union Storage {
      Storage() {}
      ~Storage() {}
      value_type array[kArraySize];
    } temp;

    // Move the current elements into a temporary array.
    for (size_t i = 0; i < kArraySize; ++i) {
      new (&temp.array[i]) value_type(std::move(array_[i]));
      array_[i].~value_type();
    }

    // Initialize the set.
    size_ = kUsingFullSetSentinel;
    functor_(&set_);

    // Insert elements into it.
    for (size_t i = 0; i < kArraySize; ++i) {
      set_.insert(std::move(temp.array[i]));
      temp.array[i].~value_type();
    }
  }

  // Helpers for constructors and destructors.
  void InitFrom(const small_set& src) {
    functor_ = src.functor_;
    size_ = src.size_;
    if (src.UsingFullSet()) {
      functor_(&set_);
      set_ = src.set_;
    } else {
      for (size_t i = 0; i < size_; ++i) {
        new (&array_[i]) value_type(src.array_[i]);
      }
    }
  }

  void Destroy() {
    if (UsingFullSet()) {
      set_.~NormalSet();
    } else {
      for (size_t i = 0; i < size_; ++i) {
        array_[i].~value_type();
      }
    }
  }
};

template <typename NormalSet, size_t kArraySize, typename EqualKey, typename SetInit>
bool operator==(const small_set<NormalSet, kArraySize, EqualKey, SetInit> &lhs,
                const small_set<NormalSet, kArraySize, EqualKey, SetInit> &rhs) {
  if (lhs.size() != rhs.size()) return false;
  return std::all_of(lhs.begin(), lhs.end(), [&rhs](const typename NormalSet::value_type &element) { return rhs.find(element) != rhs.end(); });
}

template <typename NormalSet, size_t kArraySize, typename EqualKey, typename SetInit>
bool operator!=(const small_set<NormalSet, kArraySize, EqualKey, SetInit> &lhs,
                const small_set<NormalSet, kArraySize, EqualKey, SetInit> &rhs) {
  return !(lhs == rhs);
}

}  // namespace base

#endif  // BASE_CONTAINERS_small_set_H_