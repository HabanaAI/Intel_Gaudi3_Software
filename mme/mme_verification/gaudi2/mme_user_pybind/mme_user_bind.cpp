#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

#include "convolution_params.h"
#include "gaudi2/asic_reg_structs/acc_regs.h"
#include "gaudi2/asic_reg_structs/mme_ctrl_lo_regs.h"
#include "gaudi2/asic_reg_structs/sob_objs_regs.h"
#include "gaudi2/mme.h"
#include "gaudi2/mme_descriptor_generator.h"
#include "gaudi2/mme_user/gaudi2_mme_user.h"
#include "mme_verification/gaudi2/mme_user/gaudi2_sync_object_manager.h"
#include "mme_reg_write_cmd.h"
#include "coral_user_sim_device.h"
#include "coral_user_utils.h"
#include "mme_verification/common/mme_test_manager.h"
#include "sim_tensor.h"
#include "tensor_comparator.h"

namespace Gaudi2
{
namespace py = pybind11;
using namespace MmeCommon;
using namespace gaudi2;

PYBIND11_MODULE(mme_user_utils_gaudi2, m)
{
    py::class_<MmeTestParams> mmeTestParams(m, "MmeTestParams", py::dynamic_attr());
    mmeTestParams.def(py::init<>())
        .def_readwrite("randomMD", &MmeTestParams::randomMD)
        .def_readwrite("wkldIdMD", &MmeTestParams::wkldIdMD)
        .def_readwrite("repeats", &MmeTestParams::repeats)
        //.def_readwrite("sbResuseInStripes", &MmeTestParams::sbResuseInStripes)
        .def_readwrite("incDec", &MmeTestParams::incDec)
        .def_readwrite("maskSignals", &MmeTestParams::maskSignals)
        .def("__repr__",
             [](const MmeTestParams& mmeTestParams) { return "<struct containing all params for Mme Test>"; });

    py::class_<MmeStrategy> mmeStrategy(m, "MmeStrategy", py::dynamic_attr());
    mmeStrategy.def(py::init<>())
        .def_readwrite("geometry", &MmeStrategy::geometry)
        .def_readwrite("pattern", &MmeStrategy::pattern)
        .def_readwrite("loweringEn", &MmeStrategy::loweringEn)
        .def_readwrite("sbReuse", &MmeStrategy::sbReuse)
        .def_readwrite("memsetxVoidPixels", &MmeStrategy::memsetDedxVoidPixels)
        .def("__repr__",
             [](const MmeStrategy& mmeStrategy) { return "<struct containing all params for Mme Test Strategy>"; });

    py::class_<MmeSimTensor> mmeSimTensor(m, "MmeSimTensor", py::dynamic_attr());
    mmeSimTensor.def(py::init<>())
        .def(py::init<>([](std::vector<int32_t> sizes,
                           int32_t dim,
                           EMmeDataType type,
                           uint64_t devAddr,
                           std::vector<int32_t> strides) {
            uint64_t tensor_size = 1;
            for (auto size : sizes)
            {
                tensor_size *= size;
            }
            char* memAddr = new char[tensor_size * 4];  // todo: element size
            MmeSimTensor simTensor = MmeSimTensor(sizes.data(), dim, type, (char*) (memAddr), strides.data());
            delete[] memAddr;

            return simTensor;
        }))
        .def(py::init<>([](MmeSimTensor tensor, std::vector<int32_t> offsets, int32_t dim, std::vector<int32_t> sizes) {
            MmeSimTensor simTensor = MmeSimTensor(&tensor, offsets.data(), dim, sizes.data());

            return simTensor;
        }))
        .def("set_device_addr",
             [](MmeSimTensor& tensor, uint64_t devAddr) {
                 uint64_t* memAddr = &devAddr;
                 tensor.setDeviceAddress((char*) (*memAddr));
             })
        .def("get_size", &MmeSimTensor::getSize)
        .def("get_stride", &MmeSimTensor::getStride)
        .def("get_dim", &MmeSimTensor::getDim)
        .def("get_elem_size", &MmeSimTensor::getElementSize)
        .def("get_elem_type", &MmeSimTensor::getElementType)
        .def("get_stride_list",
             [](MmeSimTensor& tensor, std::vector<int32_t> strides) { tensor.copyStrides(strides.data()); })
        .def("__repr__",
             [](const MmeSimTensor& mmeSimTensor) { return "<class containing all params for Mme Tensor>"; });

    py::enum_<EMmeDataType>(mmeSimTensor, "EMmeDataType")
        // TODO: other enum types
        .value("e_type_bf16", EMmeDataType::e_type_bf16)
        .value("e_type_f32", EMmeDataType::e_type_fp32_ieee)
        .export_values();

    py::enum_<EMmeGeometry>(m, "EMmeGeometry").export_values();
    py::enum_<EMmePattern>(m, "EMmePattern").export_values();

    py::class_<ConvolutionParams> convolution(m, "Convolution", py::dynamic_attr());
    convolution.def(py::init<>())
        .def_readwrite("dim", &ConvolutionParams::dim)
        .def_readwrite("paddingValue", &ConvolutionParams::paddingValue)
        .def("get_max_conv_dims", [](ConvolutionParams& convParams) { return convParams.maxConvDim + 1; })
        .def("add_padding",
             [](ConvolutionParams& convParams, std::vector<int> padVec) {
                 for (int32_t i = 0; i < padVec.size(); i++)
                 {
                     convParams.padding.at(i) = padVec.at(i);
                 }
             })
        .def("add_conv_stride",
             [](ConvolutionParams& convParams, std::vector<int> convStrideVec) {
                 for (int32_t i = 0; i < convStrideVec.size(); i++)
                 {
                     convParams.convStride.at(i) = convStrideVec.at(i);
                 }
             })
        .def("add_dilation",
             [](ConvolutionParams& convParams, std::vector<int> dilationVec) {
                 for (int32_t i = 0; i < dilationVec.size(); i++)
                 {
                     convParams.dilation.at(i) = dilationVec.at(i);
                 }
             })
        .def_readwrite("dim", &ConvolutionParams::dim)
        .def("add_pad_val",
             [](ConvolutionParams& convParams, float padVal) { convParams.paddingValue.int32 = *(uint32_t*) &padVal; })
        .def("__repr__",
             [](const ConvolutionParams& convolution) { return "<struct containing all params for Convolution>"; });

    m.def("gen_mme_cmd_desc",
          [](uint32_t opType,
             uint32_t ctxId,
             ConvolutionParams convParams,
             std::vector<uint32_t> soAddrLow,
             uint32_t soAddrHigh,
             MmeSimTensor inp,
             MmeSimTensor wt,
             MmeSimTensor op,
             uint32_t rMode,
             MmeStrategy strategy,
             uint64_t smBase,
             void* progs_ptr) {
              const MmeCommon::MmeHalReader& mmeHal = Gaudi2::MmeHalReader::getInstance();
              // Create SO manager
              Gaudi2SyncObjectManager soMgr(smBase);

              // Setup all the parameters for createActivations() mme user entrypoint
              TestResources testResources({});

              // Create mmeUser
              Gaudi2MmeUser mmeUser;

              // MmeDataParams
              MmeDataParams& testDataParams = testResources.testDataParams;
              testDataParams.tensorParams.resize(1);
              testDataParams.tensorParams.front().xHost = std::make_shared<MmeSimTensor>(inp);
              testDataParams.tensorParams.front().wHost = std::make_shared<MmeSimTensor>(wt);
              testDataParams.tensorParams.front().yHost = std::make_shared<MmeSimTensor>(op);
              testResources.numOfSoIdx = Gaudi2::Mme::MME_CORE_MASTERS_NR;  // no slave signaling, no secondary output
              testDataParams.syncObjects.resize(Gaudi2::Mme::MME_CORE_MASTERS_NR);
              testDataParams.soValues.resize(Gaudi2::Mme::MME_CORE_MASTERS_NR);
              soMgr.getSoIdx(testDataParams.syncObjects[0].Primary);
              soMgr.getSoIdx(testDataParams.syncObjects[1].Primary);
              testDataParams.soValues[0] = 0;
              testDataParams.soValues[1] = 0;

              // MmeTensorMemoryAttrib
              MmeTensorMemoryAttrib memAttrib;
              memAttrib.sramBase = 0x1000fffffd000000ull;  // sram base
              memAttrib.sramSize = 0x3000000ull;
              memAttrib.hbmBase = 0x1001000000000000ull;
              testResources.testMemUsage.sramUsage =
                  inp.getMemorySize() + wt.getMemorySize() + op.getMemorySize();  // todo: align up

              // other params structs
              MmeControls controls;
              MmeMemoryConfig memoryConfig;

              // Op type and descGenerator
              EMmeOpType mmeOpType = static_cast<EMmeOpType>(opType);
              unsigned actNum;

              MmeTestParams testParams = {0};
              // First entrypoint into mme user
              mmeUser.createActivations(mmeOpType,
                                        convParams,
                                        memAttrib,
                                        strategy,
                                        controls,
                                        memoryConfig,
                                        testDataParams,
                                        testResources.testMemUsage,
                                        actNum,
                                        testParams,
                                        testResources.testId);

              // Second entrypoint into mme user to Patch the generated descriptor
              // and generate the cmds
              testParams.repeats = 1;
              testParams.mmeLimit = 0;
              testParams.fullDesc = true;
              bool firstValidTest = true;

              mmeUser.patchTensors(testDataParams,
                                   0,  // sramOffset
                                   0);  // hbmOffset
              mmeUser.buildCmds(testParams,
                                testResources.cmds.data(),
                                testResources.testDataParams,
                                firstValidTest,
                                scalFw,
                                testResources.testMemUsage);

              // Finally convert the commands to CP programs
              std::list<CPProgram> mme_progs;
              std::list<CPProgram> powerProgs;
              GroupType group;
              bool firstValidGroup = true;
              bool nullDescInGroup = false;
              unsigned stream = 0;
              uint32_t seed = 0;
              uint32_t lfsrNumRegs = Gaudi2::Mme::c_mme_lfsr_seeds_nr;
              LfsrData lfsrData;
              bool duplicateLfsrValuesForAllCores = false;
              bool configLfsr = false;
              MmeTestManager::createLfsrValues(seed,
                                               lfsrData,
                                               Gaudi2::Mme::MME_CORES_NR,
                                               lfsrNumRegs,
                                               duplicateLfsrValuesForAllCores,
                                               configLfsr);
              // add test to group
              group.push_back(std::make_unique<TestResources>(testResources));
              auto& test = group.back();
              test->testJson = new nlohmann::json();
              auto& testJson = *(test->testJson);
              testJson["geometry"] = strategy.geometry;
              const PmuConfig pmucfgmode = PMUCFGNONE;
              // Return the test sync info, i.e. final output SO and its target value
              // that the upload DMA should wait for
              SyncInfo groupSI = soMgr.createGroupSyncObj(group.size());
              MmeTestManager::generateProgram(mmeUser,
                                              group,
                                              soMgr,
                                              mme_progs,
                                              powerProgs,
                                              testParams.mmeLimit,
                                              firstValidGroup,
                                              lfsrData,
                                              seed,
                                              stream,
                                              pmucfgmode,
                                              false,
                                              false);
              std::list<CPProgram>& progs = *(std::list<CPProgram>*) progs_ptr;
              for (auto& prog : mme_progs)
              {
                  progs.emplace_back(prog);
              }
              delete test->testJson;

              return groupSI;
          });
}

}  // namespace Gaudi2
