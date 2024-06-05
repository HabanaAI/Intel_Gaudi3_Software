#pragma once

template <class T>
struct copiable_atomic : std::atomic<T>
{
    using std::atomic<T>::atomic;
    copiable_atomic() : std::atomic<T>() {};
    copiable_atomic(const copiable_atomic& other) : std::atomic<T>(other.load()) {}
    copiable_atomic& operator=(const copiable_atomic& other) { this->store(other.load()); return *this; }
};

