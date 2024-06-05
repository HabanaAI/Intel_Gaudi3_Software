#pragma once

#include <variant>
#include <cstdint>

#include "scal_internal/sched_pkts.hpp"
#include "scal_internal/eng_pkts.hpp"

template<class Tfw>
struct Packets : Tfw
{
};

template struct Packets<g2fw>;
template struct Packets<g3fw>;

using G2Packets = Packets<g2fw>;
using G3Packets = Packets<g3fw>;


// for now they are the same, if it changes in the future,
static_assert((uint32_t)g2fw::STATIC_COMPUTE_ECB_LIST_BUFF_SIZE == (uint32_t)g3fw::STATIC_COMPUTE_ECB_LIST_BUFF_SIZE);
inline auto STATIC_COMPUTE_ECB_LIST_BUFF_SIZE = g2fw::STATIC_COMPUTE_ECB_LIST_BUFF_SIZE;


template<template<class> class PKT, typename... Args>
auto fillPkt(const std::variant<G2Packets , G3Packets> buildPkt, char* buff, Args&& ...args)
{
    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            PKT<T>::build((void*)(buff), std::forward<Args>(args)...);
            return PKT<T>::getSize();
            },
            buildPkt);
}

template<template<class> class PKT, typename... Args>
auto fillScalPkt(const std::variant<G2Packets , G3Packets> buildPkt, char* buff, Args&& ...args)
{
    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            PKT<T>::buildScal((void*)(buff), std::forward<Args>(args)...);
            return PKT<T>::getSize();
            },
            buildPkt);
}

template<template<class> class PKT, typename... Args>
auto fillScalPktNoSize(const std::variant<G2Packets , G3Packets> buildPkt, char* buff, Args&& ...args)
{
    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            PKT<T>::buildScal((void*)(buff), std::forward<Args>(args)...);
            },
            buildPkt);
}

template<template<class> class PKT, typename... Args>
auto fillPktNoSize(const std::variant<G2Packets , G3Packets> buildPkt, char* buff, Args&& ...args)
{
    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            PKT<T>::build((void*)(buff), std::forward<Args>(args)...);
            },
            buildPkt);
}

template<template<class> class PKT, typename... Args>
auto fillPktForceOpcode(const std::variant<G2Packets , G3Packets> buildPkt, char* buff, Args&& ...args)
{
    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            PKT<T>::buildForceOpcode((void*)(buff), std::forward<Args>(args)...);
            return PKT<T>::getSize();
            },
            buildPkt);
}

template<template<class> class PKT, typename... Args>
auto fillPktForceOpcodeNoSize(const std::variant<G2Packets , G3Packets> buildPkt, char* buff, Args&& ...args)
{
    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            PKT<T>::buildForceOpcode((void*)(buff), std::forward<Args>(args)...);
            },
            buildPkt);
}

template<template<class> class PKT, typename... Args>
auto getPktSize(const std::variant<G2Packets , G3Packets> buildPkt, Args&& ...args)
{
    return std::visit(
        [&](auto pkts) {
                using T = decltype(pkts);
                return PKT<T>::getSize(std::forward<Args>(args)...);
            },
            buildPkt);
}
