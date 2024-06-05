#include <list>
#include <iostream>
#include <stddef.h>
#include "gaudi/gaudi.h"
#include "mme_test_chip.h"
#include "gaudi/asic_reg_structs/acc_regs.h"
#include "gaudi/asic_reg_structs/sync_mngr_regs.h"
#include "gaudi/asic_reg/gaudi_blocks.h"
#include "gaudi/mme.h"
#include "gaudi_device_handler.h"
#include "coral_user_program.h"
#include "coral_user_utils.h"
#include "mme_test.h"
#include "mme_assert.h"
#include "tensor_utils.h"

using namespace MmeCommon;

namespace gaudi
{
#ifndef varoffsetof
#define varoffsetof(t, f) ((uint64_t)(&((t*)0)->f))
#endif

static inline uint64_t roundUp(uint64_t size, uint64_t round)
{
    return (size + round - 1) & ~(round - 1);
}

static void addMmeLfsrInitSequance(
    CPProgram &prog,
    const testLfsrState_t *lfsrState)
{
    uint64_t masterAPBase;
    uint64_t slaveAPBase;
    Mme::EMmeCore masterCore;
    Mme::EMmeCore slaveCore;

    switch (prog.getQId())
    {
    case GAUDI_QUEUE_ID_MME_0_0:
    case GAUDI_QUEUE_ID_MME_0_1:
    case GAUDI_QUEUE_ID_MME_0_2:
    case GAUDI_QUEUE_ID_MME_0_3:
        // north master
        masterAPBase = mmMME2_ACC_BASE;
        slaveAPBase = mmMME3_ACC_BASE;
        masterCore = Mme::MME_CORE_NORTH_MASTER;
        slaveCore = Mme::MME_CORE_NORTH_SLAVE;
        break;
    case GAUDI_QUEUE_ID_MME_1_0:
    case GAUDI_QUEUE_ID_MME_1_1:
    case GAUDI_QUEUE_ID_MME_1_2:
    case GAUDI_QUEUE_ID_MME_1_3:
        // south master
        masterAPBase = mmMME0_ACC_BASE;
        slaveAPBase = mmMME1_ACC_BASE;
        masterCore = Mme::MME_CORE_SOUTH_MASTER;
        slaveCore = Mme::MME_CORE_SOUTH_SLAVE;
        break;
    default:
        MME_ASSERT(0, "invalid Queue ID");
        return;
    }

    prog.addCommandsBack(CPCommand::MsgLong(
        masterAPBase + offsetof(block_acc, ap), // addr
        lfsrState->poly[masterCore], // value,
        true, // mb
        true, // rb
        true)); // eb

    prog.addCommandsBack(CPCommand::MsgLong(
        slaveAPBase + offsetof(block_acc, ap), // addr
        lfsrState->poly[slaveCore])); // value,

    for (int i = 0; i < Mme::c_mme_lfsr_seeds_nr; i++)
    {
        prog.addCommandsBack(CPCommand::MsgLong(
            masterAPBase + offsetof(block_acc, ap_lfsr), // addr
            i)); // value,

        prog.addCommandsBack(CPCommand::MsgLong(
            slaveAPBase + offsetof(block_acc, ap_lfsr), // addr
            i)); // value,

        prog.addCommandsBack(CPCommand::MsgLong(
            masterAPBase + offsetof(block_acc, ap_lfsr_seed), // addr
            lfsrState->seeds[masterCore][i])); // value,

        prog.addCommandsBack(CPCommand::MsgLong(
            slaveAPBase + offsetof(block_acc, ap_lfsr_seed), // addr
            lfsrState->seeds[slaveCore][i])); // value,
    }

    prog.addCommandsBack(CPCommand::Nop(true)); // mb;
}

bool checkTensorSizesInSram(testInfo_t& ti)
{
    uint64_t sramAddr = ti.params.sramBase;
    // This block assists in identifying whether the tensors defined in the test are too large
    //  to fit into the sram
    union
    {
        uint64_t addr;
        uint32_t words[2];
    } sramBaseFS, sramBase, sramEnd, xStart, xEnd, wStart, wEnd, yStart, yEnd;
    sramBase.addr = sramAddr;
    sramBaseFS.addr = SRAM_BASE_ADDR;
    sramEnd.addr = sramAddr + SRAM_SIZE;
    uint32_t xSize, wSize, ySize;
    xStart.addr = (uint64_t) ti.data.in.data();
    xSize = ti.data.in.getMemorySize();
    xEnd.addr = xStart.addr + xSize;
    wStart.addr = (uint64_t) ti.data.weights.data();
    wSize = ti.data.weights.getMemorySize();
    wEnd.addr = wStart.addr + wSize;
    yStart.addr = (uint64_t) ti.data.out.data();
    ySize = ti.data.out.getMemorySize();
    yEnd.addr = yStart.addr + ySize;

    uint32_t totalSizesInSram =
        (ti.params.xInSram ? xSize : 0) + (ti.params.wInSram ? wSize : 0) + (ti.params.yInSram ? ySize : 0);

    if (totalSizesInSram > SRAM_SIZE)
    {
        atomicColoredPrint(COLOR_RED, "Tensors are too large to fit into the SRAM");

        atomicColoredPrint(COLOR_RED,
                           "sram address: %lu (%x %x, FS: %lu, (%x, %x), sramSize: %u, sramEnd: %lu (%x %x)\n",
                           sramBase.addr,
                           sramBase.words[1],
                           sramBase.words[0],
                           sramBaseFS.addr,
                           sramBaseFS.words[1],
                           sramBaseFS.words[0],
                           SRAM_SIZE,
                           sramEnd.addr,
                           sramEnd.words[1],
                           sramEnd.words[0]);
        atomicColoredPrint(COLOR_RED,
                           "X tensor: start %lu (%x %x), End %lu (%x %x), size: %u, %s\n",
                           xStart.addr,
                           xStart.words[1],
                           xStart.words[0],
                           xEnd.addr,
                           xEnd.words[1],
                           xEnd.words[0],
                           xSize,
                           ti.params.xInSram ? "in Sram" : "in HBM");
        atomicColoredPrint(COLOR_RED,
                           "W tensor: start %lu (%x %x), End %lu (%x %x), size %u, %s\n",
                           wStart.addr,
                           wStart.words[1],
                           wStart.words[0],
                           wEnd.addr,
                           wEnd.words[1],
                           wEnd.words[0],
                           wSize,
                           ti.params.xInSram ? "in Sram" : "in HBM");
        atomicColoredPrint(COLOR_RED,
                           "Y tensor: start %lu (%x %x), End %lu (%x %x), size %u, %s\n",
                           yStart.addr,
                           yStart.words[1],
                           yStart.words[0],
                           yEnd.addr,
                           yEnd.words[1],
                           yEnd.words[0],
                           ySize,
                           ti.params.yInSram ? "in Sram" : "in HBM");
        atomicColoredPrint(COLOR_RED, "Total tensor sizes in Sram: %u\n", totalSizesInSram);

        exit(1);
    }
    return true;
}

static bool canRunCDTests(const MmeTestParams_t& ti, bool runOnChip)
{
    bool isCDTest = multiplyElements(std::begin(ti.inputShape), std::end(ti.inputShape)) == 0 ||
                    multiplyElements(std::begin(ti.weightsShape), std::end(ti.weightsShape)) == 0;
    return isCDTest && runOnChip;
}

bool runChipTests(std::vector<MmeTestParams_t>* testsParams,
                  const std::string& dumpDir,
                  const std::string& lfsrDir,
                  const MmeCommon::DeviceType devTypeA,
                  const MmeCommon::DeviceType devTypeB,
                  const unsigned devAIdx,
                  const unsigned devBIdx,
                  bool verifMode,
                  bool gaudiM)
{
    CoralMmeHBWSniffer hbwSniffer;
    CoralMmeLBWSniffer lbwSniffer;

    std::vector<unsigned> driverDeviceIdxs;
    if (devTypeA == e_chip) driverDeviceIdxs.push_back(devAIdx);
    else if (devTypeB == e_chip)
        driverDeviceIdxs.push_back(devBIdx);
    auto devHandler = std::make_unique<GaudiDeviceHandler>(devTypeA, devTypeB, driverDeviceIdxs);
    devHandler->setChipAlternative(gaudiM);
    devHandler->createDevices(0);
    if (!dumpDir.empty())
    {
        devHandler->configureMeshSniffersAndDumpDir(EMmeDump::e_mme_dump_all, 0, hbwSniffer, lbwSniffer, dumpDir, "");
    }

    if (!devHandler->openDevices()) return false;

    if (!dumpDir.empty())
    {
        dumpTestInfo(testsParams, dumpDir);
    }

    bool canDoStaticConfig = devHandler->isRunOnSim() && !devHandler->isRunOnChip();
    if (canDoStaticConfig)
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: Initializing LFSRs.\n");
    }
    else
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: Skipping Initializing LFSRs - cannot do static config of regs.\n");
    }

    testLfsrState_t lfsrState;

    getLfsrState(
        (*testsParams)[0].tid,
        (*testsParams)[0].seed,
        lfsrDir,
        &lfsrState);

    static const unsigned c_reserved_so_num = GAUDI_QUEUE_ID_CPU_PQ;
    static const unsigned c_so_group_size = 8;

    unsigned testCounter = 0;
    unsigned soIdx = c_reserved_so_num;
    unsigned stream = 0;

    for (auto & test : *testsParams)
    {
        testInfo_t ti;
        ti.params = test;

        if (ti.params.incDec)
        {
            MME_ASSERT((ti.params.repeats % 2) == 0, "repeats should be even in incDec mode");
            MME_ASSERT(((int) ti.params.rm & (int) RoundingMode::StochasticRounding) == 0,
                      "rounding mode should be the same as stochastic rounding");
        }
        bool isRunOnChip = devTypeA == e_chip || devTypeB == e_chip;
        if (canRunCDTests(ti.params, isRunOnChip))
        {
            // TODO: SW-89675 re-enable CD tests.
            atomicColoredPrint(COLOR_YELLOW,
                               "zero CD tests are currently not supported in mme_test_gaudi - skipping test[%u]\n",
                               testCounter);
            testCounter++;
            continue;
        }

        ti.params.randomMD = 0;
        ti.params.dumpEn = dumpDir.empty() ? 0 : 1;

        MME_ASSERT((soIdx / c_so_group_size) == ((soIdx + Mme::MME_CORES_NR - 1) / c_so_group_size), "");

        SyncInfo si;
        si.smBase = ti.params.smBase;
        si.outputSOIdx = soIdx / c_so_group_size;
        si.outputSOSel = ((1 << Mme::MME_CORES_NR) - 1) << (soIdx % c_so_group_size);

        for (unsigned i = 0; i < Mme::MME_CORES_NR; i++)
        {
            uint64_t addr = si.smBase + varoffsetof(block_sync_mngr, sync_mngr_objs.sob_obj[soIdx]);
            ti.params.soAddrLow[i] = (uint32_t)addr;
            soIdx++;
            if (soIdx == (sizeof(block_sync_mngr::sync_mngr_objs.sob_obj) / sizeof(uint32_t)))
            {
                soIdx = c_reserved_so_num;
            }
        }
        ti.params.soAddrHigh = (uint32_t)(si.smBase >> 32);

        std::string testInfoStr = test2text(&ti.params);
        atomicColoredPrint(COLOR_CYAN, "INFO: Starting test: (test #%u)\n", testCounter);
        atomicColoredPrint(COLOR_MAGENTA, "%s", testInfoStr.c_str());

        allocTestHostTensors(
            &ti.params,
            &ti.data.hostIn,
            &ti.data.hostWeights,
            &ti.data.hostOut,
            &ti.data.hostRef);

        std::list<Buffer> inputBuffers;
        std::list<Buffer> outputBuffers;
        std::list<Buffer> initBuffers;

        uint64_t sramAddr = ti.params.sramBase;
        uint64_t hbmAddr = ti.params.hbmBase;
        uint64_t firstPredAddr = 0;
        uint64_t steadyPredAddr = 0;
        uint64_t* memAddr;

        uint8_t predBuff[2*c_cl_size];

        if (ti.params.loop)
        {
            Buffer tensor;
            memset(predBuff, 1, sizeof(predBuff));
            tensor.deviceAddr = sramAddr;
            tensor.hostAddr = predBuff;
            tensor.size = sizeof(predBuff);
            initBuffers.push_back(tensor);
            steadyPredAddr = tensor.deviceAddr;
            firstPredAddr = steadyPredAddr + c_cl_size;
            sramAddr += tensor.size;
        }

        Buffer tensorIn;
        ti.data.in = *ti.data.hostIn;
        memAddr = ti.params.xInSram ? &sramAddr : &hbmAddr;
        ti.data.in.setDeviceAddress((char*) *memAddr, false);
        (*memAddr) += roundUp(ti.data.in.getMemorySize(), c_cl_size);
        tensorIn.deviceAddr = (uint64_t) ti.data.in.data();
        tensorIn.hostAddr = ti.data.hostIn->data();
        tensorIn.size = ti.data.in.getMemorySize();
        ((ti.params.op == E_CONV_TEST_DEDX) ? &outputBuffers : &inputBuffers)->push_back(tensorIn);

        Buffer tensorW;
        ti.data.weights = *ti.data.hostWeights;
        memAddr = ti.params.wInSram ? &sramAddr : &hbmAddr;
        ti.data.weights.setDeviceAddress((char*) *memAddr, false);
        (*memAddr) += roundUp(ti.data.weights.getMemorySize(), c_cl_size);
        tensorW.deviceAddr = (uint64_t) ti.data.weights.data();
        tensorW.hostAddr = ti.data.hostWeights->data();
        tensorW.size = ti.data.weights.getMemorySize();
        ((ti.params.op == E_CONV_TEST_DEDW) ? &outputBuffers : &inputBuffers)->push_back(tensorW);

        Buffer tensorOut;
        ti.data.out = *ti.data.hostOut;
        memAddr = ti.params.yInSram ? &sramAddr : &hbmAddr;
        ti.data.out.setDeviceAddress((char*) *memAddr, false);
        (*memAddr) += roundUp(ti.data.out.getMemorySize(), c_cl_size);
        tensorOut.deviceAddr = (uint64_t) ti.data.out.data();
        tensorOut.hostAddr = ti.data.hostOut->data();
        tensorOut.size = ti.data.out.getMemorySize();
        (((ti.params.op == E_CONV_TEST_FWD) || (ti.params.op == E_CONV_TEST_AB) || (ti.params.op == E_CONV_TEST_ABT) ||
          (ti.params.op == E_CONV_TEST_ATB) || (ti.params.op == E_CONV_TEST_ATBT))
             ? &outputBuffers
             : &inputBuffers)
            ->push_back(tensorOut);

        checkTensorSizesInSram(ti);

        std::list<MmeRegWriteCmd> cmds[Mme::MME_MASTERS_NR];
        genConvTest(&ti.params, &ti.data, &ti.soValue, cmds, 0, 0, verifMode, testCounter);
        si.outputSOTarget = ti.soValue;

        unsigned northQId;
        unsigned southQId;
        switch (stream)
        {
        case 0: northQId = GAUDI_QUEUE_ID_MME_0_0; southQId = GAUDI_QUEUE_ID_MME_1_0; break;
        case 1: northQId = GAUDI_QUEUE_ID_MME_0_1; southQId = GAUDI_QUEUE_ID_MME_1_1; break;
        case 2: northQId = GAUDI_QUEUE_ID_MME_0_2; southQId = GAUDI_QUEUE_ID_MME_1_2; break;
        case 3: northQId = GAUDI_QUEUE_ID_MME_0_3; southQId = GAUDI_QUEUE_ID_MME_1_3; break;
        default:
            MME_ASSERT(0, "invalid stream id");
        }

        std::list<CPProgram> progs;
        for (unsigned mstr = 0; mstr < Mme::MME_MASTERS_NR; mstr++)
        {
            progs.push_back(CPProgram((mstr == Mme::MME_CORE_NORTH_MASTER) ? northQId : southQId));
            CPProgram &prog = progs.back();
            if (canDoStaticConfig && !testCounter && !ti.params.loop)
            {
                addMmeLfsrInitSequance(prog, &lfsrState);
            }
            for (auto & cmd : cmds[mstr])
            {
                prog.addTclSeq(cmd.reg_offset, cmd.reg_values, cmd.num_regs);
            }

            if (ti.params.loop)
            {

                prog.setPredModeInfo(1, firstPredAddr, steadyPredAddr);
                std::pair<uint64_t, uint32_t> addrValPair;
                addrValPair.second = si.outputSOTarget;
                for (unsigned i=0; i<c_so_group_size; i++)
                {
                    if ((i % Mme::MME_MASTERS_NR == mstr) && (si.outputSOSel & (1<<i)))
                    {
                        unsigned soIdx = (si.outputSOIdx * c_so_group_size) + i;
                        addrValPair.first = si.smBase + varoffsetof(block_sync_mngr, sync_mngr_objs.sob_obj[soIdx]);
                        prog.addPostExecWrite(addrValPair);
                    }
                }
            }
        }

        if (ti.params.memsetOutput)
        {
            inputBuffers.push_front(outputBuffers.back());
            inputBuffers.front().hostAddr = 0;
        }

        bool firstDeviceIsChip = !devHandler->isRunOnSim() && devHandler->isRunOnChip();
        bool compareTwoDevices = devHandler->isRunOnChip() && !firstDeviceIsChip;

        if (devHandler->isRunOnSim())
        {
            atomicColoredPrint(COLOR_CYAN, "INFO: Sending workload to Simulator. (test #%u)\n", testCounter);
            createAndExecuteProgram(devHandler->getSimDevice(),
                                    stream,
                                    ti.params.programInSram ? sramAddr : hbmAddr,
                                    &progs,
                                    &si,
                                    0,  // Arbitration
                                    &inputBuffers,
                                    &outputBuffers,
                                    &initBuffers);

            if (hbwSniffer.isEnabled())
            {
                MME_ASSERT(!dumpDir.empty(), "dump dir should not be empty");
                hbwSniffer.generateDump(dumpDir);
                hbwSniffer.disable();  // dump only the first test
            }

            if (lbwSniffer.isEnabled())
            {
                MME_ASSERT(!dumpDir.empty(), "dump dir should not be empty");
                lbwSniffer.generateDumpFile(dumpDir + "/act_seq.txt");
                lbwSniffer.disable();  // dump only the first test
            }
            atomicColoredPrint(COLOR_CYAN, "INFO: Simulator run completed. (test #%u)\n", testCounter);
        }

        MME_ASSERT(devHandler->getNumOfDriverDevices() <= 1, "currently doesnt support mode then single driver device");
        if (devHandler->isRunOnChip())
        {
            atomicColoredPrint(COLOR_CYAN, "INFO: Sending workload to the Device[%u]. (test #%u)\n", 0, testCounter);
            if (compareTwoDevices)
            {
                outputBuffers.front().hostAddr = malloc(outputBuffers.front().size);
            }

            createAndExecuteProgram(devHandler->getDriverDevices()[0],
                                    stream,
                                    ti.params.programInSram ? sramAddr : hbmAddr,
                                    &progs,
                                    &si,
                                    0,  // Arbitration
                                    &inputBuffers,
                                    &outputBuffers);

            atomicColoredPrint(COLOR_CYAN, "INFO: Device[%u] run completed. (%u)\n", 0, testCounter);
        }
        // Comparing the results
        bool equal = compareResults(ti,
                                    firstDeviceIsChip,
                                    compareTwoDevices ? (char*) outputBuffers.front().hostAddr : nullptr,
                                    testCounter);
        if (equal)
        {
            atomicColoredPrint(COLOR_GREEN, "INFO: Success. (test #%u)\n", testCounter);
        }
        else
        {
            return false;
        }

        if (ti.data.hostRef) delete ti.data.hostRef;
        if (ti.data.hostOut) delete ti.data.hostOut;
        if (ti.data.hostWeights) delete ti.data.hostWeights;
        if (ti.data.hostIn) delete ti.data.hostIn;
        if (compareTwoDevices)
        {
            free((void*)outputBuffers.front().hostAddr);
        }

        testCounter++;
    }

    atomicColoredPrint(COLOR_YELLOW, "INFO: Test done.\n");
    return true;
}

}  // namespace gaudi
