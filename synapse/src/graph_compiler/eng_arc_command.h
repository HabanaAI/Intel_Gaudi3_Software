#pragma once

#include "types.h"
#include "defs.h"
#include "define_synapse_common.hpp"

#define DWORD_SIZE     sizeof(uint32_t)

// A base class for Engine ARC commands
class EngArcCommand
{
public:
    EngArcCommand()          = default;
    virtual ~EngArcCommand() = default;

    virtual void     setYield(bool y)           = 0;
    virtual void     print() const              = 0;
    virtual unsigned sizeInBytes() const        = 0;
    virtual uint64_t serialize(void* dst) const = 0;  // returns how many bytes were written
    virtual bool     isArcExeWd() { return false; }
    virtual void     setSwitchCQ(bool switchCQ) { HB_ASSERT(0, "shouldn't get here"); }  // immediate commands override
};

using EngArcCmdPtr = std::shared_ptr<EngArcCommand>;

class StaticCpDmaEngArcCommand : public EngArcCommand
{
public:
    StaticCpDmaEngArcCommand() = default;
    virtual ~StaticCpDmaEngArcCommand() = default;
    virtual void     print() const override              = 0;
    virtual unsigned sizeInBytes() const override        = 0;
    virtual uint64_t serialize(void* dst) const override = 0;
    virtual void     setEngId(unsigned engId)            = 0;
    virtual void     setYield(bool y) override           = 0;
};

class NopEngArcCommand : public EngArcCommand
{
public:
    NopEngArcCommand() = default;
    virtual ~NopEngArcCommand() = default;

    virtual void     print() const override              = 0;
    virtual unsigned sizeInBytes() const override        = 0;
    virtual uint64_t serialize(void* dst) const override = 0;
    virtual void     setPadding(unsigned padding)        = 0;
    virtual void     setSwitchCQ(bool switchCQ) override = 0;
    virtual void     setYield(bool y) override           = 0;
};

class DynamicWorkDistEngArcCommand : public EngArcCommand
{
public:
    DynamicWorkDistEngArcCommand() = default;
    virtual ~DynamicWorkDistEngArcCommand() = default;

    virtual void     print() const override              = 0;
    virtual unsigned sizeInBytes() const override        = 0;
    virtual uint64_t serialize(void* dst) const override = 0;
    virtual void     setYield(bool y) override           = 0;
    virtual bool     isArcExeWd() override { return true; }
};

class ScheduleDmaEngArcCommand : public EngArcCommand
{
public:
    ScheduleDmaEngArcCommand() = default;
    virtual ~ScheduleDmaEngArcCommand() = default;

    virtual void     print() const override              = 0;
    virtual unsigned sizeInBytes() const override        = 0;
    virtual uint64_t serialize(void* dst) const override = 0;
    virtual void     setYield(bool y) override           = 0;
};

class ListSizeEngArcCommand : public EngArcCommand
{
public:
    ListSizeEngArcCommand() = default;
    virtual ~ListSizeEngArcCommand() = default;

    virtual void     print() const override              = 0;
    virtual unsigned sizeInBytes() const override        = 0;
    virtual uint64_t serialize(void* dst) const override = 0;
    virtual void     setListSize(unsigned listSize)      = 0;
    virtual void     setTopologyStart()                  = 0;
    virtual void     setYield(bool y) override           = 0;
};

class SignalOutEngArcCommand : public EngArcCommand
{
public:
    SignalOutEngArcCommand()          = default;
    virtual ~SignalOutEngArcCommand() = default;

    virtual void     print() const override              = 0;
    virtual unsigned sizeInBytes() const override        = 0;
    virtual uint64_t serialize(void* dst) const override = 0;
    virtual void     setYield(bool y) override           = 0;
    virtual void     setSwitchCQ(bool switchCQ) override = 0;
};

class ResetSobsArcCommand : public EngArcCommand
{
public:
    ResetSobsArcCommand()          = default;
    virtual ~ResetSobsArcCommand() = default;

    virtual void     print() const override              = 0;
    virtual unsigned sizeInBytes() const override        = 0;
    virtual uint64_t serialize(void* dst) const override = 0;
    virtual void     setYield(bool y) override           = 0;
    virtual void     setSwitchCQ(bool switchCQ) override = 0;
};
