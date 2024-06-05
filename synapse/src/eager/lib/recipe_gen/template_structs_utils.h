#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_defs.h"

namespace eager_mode
{
// Classes to define the required QMAN commands to write a bulk of registers.
//
// WREG_BULK command has two constraints:
//  1. REG_OFFSET[15:0]: Registers start address offset in Bytes, , 3LSBs must be zero (8 Byte aligned)
//  2. SIZE64[15:0]: Number of 64bit registers duplets to write (in 8 bytes resolution)
//
// The following covers all possible cases to write the list of registers using WREG_32 and WREG_BULK commands:
//
// Case id    Illustration    QMAN commands
// ~~~~~~~    ~~~~~~~~~~~~    ~~~~~~~~~~~~~
//    1        |--|  |  |       WREG_BULK
//    2        | -|- |  |       WREG_32 -> WREG_32
//    3        |--|- |  |       WREG_BULK -> WREG_32
//    4        | -|--|  |       WREG_32 -> WREG_BULK
//    1        |--|--|  |       WREG_BULK
//    5        | -|--|- |       WREG_32 -> WREG_BULK -> WREG_32

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct WriteRegBulkCommandsBase
///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Wreg32, typename WregBulk, AsicRegType FirstRegOffset, BlobSizeType RegBulkSize>
struct WriteRegBulkCommandsBase
{
    // Calculate number of required WREG_32 commands
    static constexpr unsigned calcWreg32CmdNr()
    {
        static_assert(RegBulkSize % sizeOfAsicRegVal == 0);
        // Check for possible WREG_32 as prefix:
        unsigned wreg32CmdNr = (FirstRegOffset % sizeOfAddressVal == 0) ? 0 : 1;
        // Check for possible WREG_32 as suffix:
        if ((RegBulkSize - wreg32CmdNr * sizeOfAsicRegVal) % sizeOfAddressVal != 0)
        {
            ++wreg32CmdNr;
        }
        return wreg32CmdNr;  // Possible values: 0, 1, 2
    }
    // Calculate size of the register bulk
    static constexpr unsigned calcRegBlkSize(unsigned wreg32CmdNr)
    {
        if (RegBulkSize < sizeOfAddressVal) return 0;
        BlobSizeType netSize = RegBulkSize - sizeOfAsicRegVal * wreg32CmdNr;  // Size without WREG_32
        return netSize;
    }

    static constexpr unsigned     wreg32CmdNr           = calcWreg32CmdNr();
    static constexpr BlobSizeType regBlkSize            = calcRegBlkSize(wreg32CmdNr);
    static constexpr bool         isFirstWReg32Required = (FirstRegOffset % sizeOfAddressVal) != 0;

protected:
    // Init commands based on template parameters. It's a generic function to cover all cases described above
    static void init(Wreg32* firstWreg32, Wreg32* secondWreg32, WregBulk* wreg_bulk, Byte regBulkBuf[regBlkSize])
    {
        [[maybe_unused]] unsigned curWreg32CmdNr  = wreg32CmdNr;
        [[maybe_unused]] unsigned curWregBlkCmdNr = (regBlkSize == 0) ? 0 : 1;
        [[maybe_unused]] unsigned totalCmdNr =
            ((firstWreg32 == nullptr) ? 0 : 1) + ((secondWreg32 == nullptr) ? 0 : 1) + ((wreg_bulk == nullptr) ? 0 : 1);
        EAGER_ASSERT(totalCmdNr != 0, "Invalid inputs to write a bulk of registers");
        EAGER_ASSERT((curWreg32CmdNr + curWregBlkCmdNr) == totalCmdNr, "Invalid inputs to write a bulk of registers");

        AsicRegType curRegOffset = FirstRegOffset;
        // Init WREG_32 as prefix
        if constexpr (isFirstWReg32Required)
        {
            EAGER_ASSERT(wreg32CmdNr >= 1, "Invalid template parameters");
            EAGER_ASSERT_PTR(firstWreg32);
            firstWreg32->init(FirstRegOffset);
            curRegOffset += sizeOfAsicRegVal;
            --curWreg32CmdNr;
        }
        // Init WREG_BULK
        if constexpr (regBlkSize != 0)
        {
            EAGER_ASSERT_PTR(wreg_bulk);
            EAGER_ASSERT_PTR(regBulkBuf);
            static constexpr TensorsNrType tensorsNr = regBlkSize / sizeOfAddressVal;
            wreg_bulk->init(curRegOffset, tensorsNr);
            curRegOffset += regBlkSize;
            --curWregBlkCmdNr;
        }
        // Init WREG_32 as suffix. There are two possible cases:
        if constexpr ((wreg32CmdNr == 2) ||                                      // (1) Two WREG_32 are required
                      ((wreg32CmdNr != 0) && (isFirstWReg32Required == false)))  // (2) Only WREG_32 as suffix
        {
            EAGER_ASSERT(curWreg32CmdNr == 1, "Invalid template parameters");
            EAGER_ASSERT(curRegOffset != 0, "Invalid template parameters");
            EAGER_ASSERT_PTR(secondWreg32);
            secondWreg32->init(curRegOffset);
            --curWreg32CmdNr;
        }
        EAGER_ASSERT((curWreg32CmdNr == 0) && (curWregBlkCmdNr == 0), "Invalid setup to write a bulk of registers");
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct WrRegBulk
///////////////////////////////////////////////////////////////////////////////////////////////////

// Case 1: Use it when only one WREG_BULK command is needed
template<typename Wreg32Cmd, typename WregBulkCmd, AsicRegType firstRegOffset, BlobSizeType regBulkSize>
struct WrRegBulk final : public WriteRegBulkCommandsBase<Wreg32Cmd, WregBulkCmd, firstRegOffset, regBulkSize>
{
    using base = WriteRegBulkCommandsBase<Wreg32Cmd, WregBulkCmd, firstRegOffset, regBulkSize>;

    WregBulkCmd wreg_bulk;
    Byte        regBulkBuf[base::regBlkSize];

    void init(bool switchBit)
    {
        base::init(nullptr, nullptr, &wreg_bulk, regBulkBuf);
        wreg_bulk.swtc = switchBit ? 1 : 0;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct WrReg32x2
///////////////////////////////////////////////////////////////////////////////////////////////////

// Case 2: Use it for WREG_32 -> WREG_32 sequence
template<typename Wreg32Cmd, typename WregBulkCmd, AsicRegType firstRegOffset, BlobSizeType regBulkSize>
struct WrReg32x2 final : public WriteRegBulkCommandsBase<Wreg32Cmd, WregBulkCmd, firstRegOffset, regBulkSize>
{
    using base = WriteRegBulkCommandsBase<Wreg32Cmd, WregBulkCmd, firstRegOffset, regBulkSize>;

    Wreg32Cmd firstWreg32;
    Wreg32Cmd secondWreg32;

    void init(bool switchBit)
    {
        base::init(&firstWreg32, &secondWreg32, nullptr, nullptr);
        secondWreg32.swtc = switchBit ? 1 : 0;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct WrRegBulk_WrReg32
///////////////////////////////////////////////////////////////////////////////////////////////////

// Case 3: Use it for WREG_BULK -> WREG_32 sequence
template<typename Wreg32Cmd, typename WregBulkCmd, AsicRegType firstRegOffset, BlobSizeType regBulkSize>
struct WrRegBulk_WrReg32 final : public WriteRegBulkCommandsBase<Wreg32Cmd, WregBulkCmd, firstRegOffset, regBulkSize>
{
    using base = WriteRegBulkCommandsBase<Wreg32Cmd, WregBulkCmd, firstRegOffset, regBulkSize>;

    WregBulkCmd wreg_bulk;
    Byte        regBulkBuf[base::regBlkSize];
    Wreg32Cmd   wreg32;

    void init(bool switchBit)
    {
        base::init(nullptr, &wreg32, &wreg_bulk, regBulkBuf);
        wreg32.swtc = switchBit ? 1 : 0;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct WrReg32_WrRegBulk
///////////////////////////////////////////////////////////////////////////////////////////////////

// Case 4: Use it for WREG_32 -> WREG_BULK sequence
template<typename Wreg32Cmd, typename WregBulkCmd, AsicRegType firstRegOffset, BlobSizeType regBulkSize>
struct WrReg32_WrRegBulk final : public WriteRegBulkCommandsBase<Wreg32Cmd, WregBulkCmd, firstRegOffset, regBulkSize>
{
    using base = WriteRegBulkCommandsBase<Wreg32Cmd, WregBulkCmd, firstRegOffset, regBulkSize>;

    Wreg32Cmd   wreg32;
    WregBulkCmd wreg_bulk;
    Byte        regBulkBuf[base::regBlkSize];

    void init(bool switchBit)
    {
        base::init(&wreg32, nullptr, &wreg_bulk, regBulkBuf);
        wreg_bulk.swtc = switchBit ? 1 : 0;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// struct WrReg32_WrRegBulk_WrReg32
///////////////////////////////////////////////////////////////////////////////////////////////////

// Case 5: Use it for WREG_32 -> WREG_BULK -> WREG_32 sequence
template<typename Wreg32Cmd, typename WregBulkCmd, AsicRegType firstRegOffset, BlobSizeType regBulkSize>
struct WrReg32_WrRegBulk_WrReg32 final
: public WriteRegBulkCommandsBase<Wreg32Cmd, WregBulkCmd, firstRegOffset, regBulkSize>
{
    using base = WriteRegBulkCommandsBase<Wreg32Cmd, WregBulkCmd, firstRegOffset, regBulkSize>;

    Wreg32Cmd   firstWreg32;
    WregBulkCmd wreg_bulk;
    Byte        regBulkBuf[base::regBlkSize];
    Wreg32Cmd   secondWreg32;

    void init(bool switchBit)
    {
        base::init(&firstWreg32, &secondWreg32, &wreg_bulk, regBulkBuf);
        secondWreg32.swtc = switchBit ? 1 : 0;
    }
};

}  // namespace eager_mode