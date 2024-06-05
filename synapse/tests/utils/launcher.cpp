#include "launcher.h"

template<class T>
std::string listToString(const T& list)
{
    std::string ret       = "[";
    std::string delimiter = "";
    for (const auto& v : list)
    {
        ret += delimiter + std::to_string(v);
        delimiter = ", ";
    }
    ret += "]";
    return ret;
}

LauncherBase::TensorMemType LauncherBase::getTensorMemType(synTensorType type)
{
    switch (type)
    {
        case HOST_SHAPE_TENSOR:
        case HOST_TO_DEVICE_TENSOR:
            return TensorMemType::HOST;
        case DATA_TENSOR:
        case DATA_TENSOR_DYNAMIC:
        case DEVICE_SHAPE_TENSOR:  // device shape tensor has data and need to be allocated
            return TensorMemType::DEVICE;
        case OUTPUT_DESCRIBING_SHAPE_TENSOR:
        case INPUT_DESCRIBING_SHAPE_TENSOR:
        case TENSOR_TYPE_MAX:
            return TensorMemType::NONE;
    }
    throw std::runtime_error("unsupported tensor type: " + std::to_string(type));
}

LauncherBase::LauncherBase(syn::Device&                         device,
                           const syn::Recipe&                   recipe,
                           const std::shared_ptr<DataProvider>& dataProvider,
                           const OnTensor&                      onOutputTensor)
: m_recipe(recipe),
  m_dataProvider(dataProvider),
  m_device(device),
  m_streamDownload(device.createStream()),
  m_onOutputTensor(onOutputTensor)
{
}

LauncherBase::LauncherBase(syn::Device&                          device,
                           const syn::Recipe&                    recipe,
                           const std::shared_ptr<DataProvider>&  dataProvider,
                           const std::shared_ptr<DataCollector>& dataCollector)
: LauncherBase(device, recipe, dataProvider, [&dataCollector](const std::string& name, const syn::HostBuffer& buffer) {
      dataCollector->setBuffer(name, buffer);
  })
{
}

void LauncherBase::verifyTensorShape(const std::vector<TSize>& shape, const synRetrievedLaunchTensorInfoExt& info)
{
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] < info.tensorMinSize[i] || shape[i] > info.tensorMaxSize[i])
        {
            m_result.warnings.push_back(fmt::format(
                "invalid shape for tensor: {} in dim {}, actual size: {}, min size is: {} and max size is: {}",
                info.tensorName,
                i,
                shape[i],
                info.tensorMinSize[i],
                info.tensorMaxSize[i]));
        }
    }
}

void LauncherBase::getSectionInfo(synSectionId sectionId,
                                  bool&        isConstSection,
                                  uint64_t&    section_size,
                                  uint64_t&    section_data)
{
    isConstSection = (bool)m_recipe.sectionGetProp(sectionId, IS_CONST);

    if (isConstSection)
    {
        section_size = m_recipe.sectionGetProp(sectionId, SECTION_SIZE);
        if (section_size > 0)
        {
            section_data = m_recipe.sectionGetProp(sectionId, SECTION_DATA);
        }
    }
}

void LauncherBase::downloadConstSectionDataToDevice(const ConstSectionInfo& sectionInfo,
                                                    DeviceBuffersMap&       constSectionDeviceBuf)
{
    if (constSectionDeviceBuf.find(sectionInfo.sectionId) == constSectionDeviceBuf.end())
    {
        // allocate buffer on device (and copy data) of const section once
        syn::DeviceBuffer buff = m_device.malloc(sectionInfo.sectionSize);

        m_device.hostMap((void*)sectionInfo.sectionData, sectionInfo.sectionSize);

        m_streamDownload.memCopyAsync(sectionInfo.sectionData,
                                      buff.getAddress(),
                                      sectionInfo.sectionSize,
                                      HOST_TO_DRAM);
        m_streamDownload.synchronize();

        constSectionDeviceBuf[sectionInfo.sectionId] = buff;

        m_device.hostUnmap((void*)sectionInfo.sectionData);
    }
}

void LauncherBase::splitConstSectionTensors(TensorsData& tensors)
{
    // separate the const section tensors from the general tensors list
    bool     isConstSection = false;
    uint64_t section_size = 0, section_data = 0;

    auto it = tensors.input.begin();

    for (; it != tensors.input.end();)
    {
        const auto& info = it->second.retrievedlaunchTensorInfo;
        getSectionInfo(info.tensorSectionId, isConstSection, section_size, section_data);

        if (isConstSection)
        {
            if (section_size > 0)
            {
                ConstSectionInfo tInfo {info.tensorSectionId,
                                        section_size,
                                        section_data,
                                        info.tensorId,
                                        info.tensorOffsetInSection,
                                        info.tensorType};
                memcpy(tInfo.tensorName, info.tensorName, ENQUEUE_TENSOR_NAME_MAX_SIZE);
                it->second.tensorConstSectionInfo = tInfo;
                it->second.launchInfo             = generateConstSectionLaunchInfo(tInfo, tensors.constSectionsBuffers);
                tensors.concatTensors.push_back(it->second.launchInfo);
            }
            it = tensors.input.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

synLaunchTensorInfoExt LauncherBase::generateLaunchInfo(const synRetrievedLaunchTensorInfoExt& info,
                                                        uint64_t                               tensorAddress)
{
    synLaunchTensorInfoExt launchInfo = {info.tensorName, tensorAddress, info.tensorType, {}, info.tensorId};
    memset(launchInfo.tensorSize, 0, sizeof(launchInfo.tensorSize));
    if (m_dataProvider)
    {
        const auto shape = m_dataProvider->getShape(info.tensorName);
        static_assert(sizeof(launchInfo.tensorSize[0]) == sizeof(shape[0]));
        uint32_t shapeSizeInBytes = sizeof(shape[0]) * shape.size();
        if (info.tensorDims != shape.size() || shapeSizeInBytes > sizeof(launchInfo.tensorSize))
        {
            m_result.warnings.push_back(
                fmt::format("invalid shape info for tensor: {}, dims count: {}", info.tensorName, shape.size()));
        }
        verifyTensorShape(shape, info);
        if (!shape.empty())
        {
            memcpy(launchInfo.tensorSize, shape.data(), shapeSizeInBytes);
        }
    }
    else
    {
        static_assert(sizeof(launchInfo.tensorSize) == sizeof(info.tensorMaxSize));
        memcpy(launchInfo.tensorSize, info.tensorMaxSize, sizeof(launchInfo.tensorSize[0]) * info.tensorDims);
    }
    return launchInfo;
}

synLaunchTensorInfoExt LauncherBase::generateConstSectionLaunchInfo(const ConstSectionInfo& sectionInfo,
                                                                    DeviceBuffersMap&       constSectionDeviceBuf)
{
    downloadConstSectionDataToDevice(sectionInfo, constSectionDeviceBuf);

    uint64_t tensorAddr =
        constSectionDeviceBuf.at(sectionInfo.sectionId).getAddress() + sectionInfo.tensorOffsetInSection;

    synLaunchTensorInfoExt launchInfo = {sectionInfo.tensorName,
                                         tensorAddr,
                                         sectionInfo.tensorType,
                                         {},
                                         sectionInfo.tensorId};
    memset(launchInfo.tensorSize, 0, sizeof(launchInfo.tensorSize));
    return launchInfo;
}

uint64_t LauncherBase::getTensorAddress(const TensorData& tensor, const DeviceBuffersMap sectionsBuffers)
{
    if (tensor.memType == TensorMemType::DEVICE)
    {
        auto deviceBuffer = sectionsBuffers.at(tensor.retrievedlaunchTensorInfo.tensorSectionId);
        return deviceBuffer.getAddress() + tensor.retrievedlaunchTensorInfo.tensorOffsetInSection;
    }
    if (tensor.hostBuffer)
    {
        return tensor.hostBuffer.getAddress();
    }
    return 0;
}

void LauncherBase::generateLaunchInfos(TensorsData& tensors)
{
    for (auto& t : tensors.input)
    {
        auto&       tensor = t.second;
        const auto& info   = tensor.retrievedlaunchTensorInfo;
        if (tensor.memType == TensorMemType::HOST)
        {
            uint64_t tensorSizeInBytes = getActualTensorSize(info, m_dataProvider);
            tensor.hostBuffer          = m_device.hostMalloc(tensorSizeInBytes);
            setHostBuffer(tensor.retrievedlaunchTensorInfo.tensorName, tensor.hostBuffer);
        }
        uint64_t tensorAddress = getTensorAddress(tensor, tensors.sectionsBuffers);
        tensor.launchInfo      = generateLaunchInfo(info, tensorAddress);
        tensors.concatTensors.push_back(tensor.launchInfo);
    }

    for (auto& t : tensors.output)
    {
        auto& tensor = t.second;

        const auto& info = tensor.retrievedlaunchTensorInfo;
        if (tensor.memType == TensorMemType::HOST)
        {
            uint64_t tensorSizeInBytes = getActualTensorSize(info, m_dataProvider);
            tensor.hostBuffer          = m_device.hostMalloc(tensorSizeInBytes);
        }
        uint64_t tensorAddress = getTensorAddress(tensor, tensors.sectionsBuffers);
        tensor.launchInfo      = generateLaunchInfo(info, tensorAddress);
        tensors.concatTensors.push_back(tensor.launchInfo);
    }
}

void LauncherBase::initTensors(TensorsData& tensors)
{
    const auto tensorsInfo = m_recipe.getLaunchTensorsInfoExt();
    for (const auto& info : tensorsInfo)
    {
        auto& tensorsMap = info.isInput ? tensors.input : tensors.output;
        auto  it         = tensorsMap.insert(std::pair<uint64_t, TensorData> {info.tensorId, TensorData {}});

        it.first->second.retrievedlaunchTensorInfo = info;
        it.first->second.memType                   = getTensorMemType(info.tensorType);
    }
}

void LauncherBase::setHostBuffer(const std::string& tensorName, syn::HostBuffer& hostBuffer)
{
    if (m_dataProvider)
    {
        m_dataProvider->copyBuffer(tensorName, hostBuffer);
    }
    else
    {
        memset(hostBuffer.get(), 0, hostBuffer.getSize());
    }
}

void LauncherBase::allocateBuffers(TensorData& tensor, const DeviceBuffersMap sectionsBuffers)
{
    if (tensor.memType == TensorMemType::DEVICE)
    {
        const auto& info              = tensor.retrievedlaunchTensorInfo;
        uint64_t    tensorSizeInBytes = getActualTensorSize(info, m_dataProvider);
        tensor.hostBuffer             = m_device.hostMalloc(tensorSizeInBytes);
        tensor.deviceBuffer           = sectionsBuffers.at(info.tensorSectionId);
    }
}

void LauncherBase::allocateInputBuffers(TensorsData& tensors)
{
    for (auto& t : tensors.input)
    {
        auto& tensor = t.second;
        if (!tensor.hostBuffer && (tensor.memType != TensorMemType::NONE))
        {
            allocateBuffers(tensor, tensors.sectionsBuffers);
            setHostBuffer(tensor.retrievedlaunchTensorInfo.tensorName, tensor.hostBuffer);
        }
    }
}

void LauncherBase::queryTensors(TensorsData& tensors)
{
    initTensors(tensors);
    splitConstSectionTensors(tensors);
    allocateSectionsMem(tensors);
    generateLaunchInfos(tensors);
    allocateInputBuffers(tensors);
}

void LauncherBase::download(syn::Stream stream, const TensorsData& tensors)
{
    // Copy input tensors from host memory to HBM
    for (const auto& t : tensors.input)
    {
        const auto& tensor = t.second;
        if (tensor.memType == TensorMemType::DEVICE)
        {
            stream.memCopyAsync(tensor.hostBuffer.getAddress(),
                                tensor.deviceBuffer.getAddress() +
                                    tensor.retrievedlaunchTensorInfo.tensorOffsetInSection,
                                tensor.hostBuffer.getSize(),
                                HOST_TO_DRAM);
        }
    }
}

void LauncherBase::upload(syn::Stream stream, const TensorsData& tensors)
{
    if (!m_onOutputTensor) return;

    // Copy graph outputs from HBM to host memory
    for (auto& t : tensors.output)
    {
        // the tensor is destroyed on each iteration and release the hostBuffer unless the buffer ref-count is increasd
        // in the callback (m_onOutputTensor)
        auto tensor = t.second;

        allocateBuffers(tensor, tensors.sectionsBuffers);

        stream.memCopyAsync(tensor.deviceBuffer.getAddress() + tensor.retrievedlaunchTensorInfo.tensorOffsetInSection,
                            tensor.hostBuffer.getAddress(),
                            tensor.hostBuffer.getSize(),
                            DRAM_TO_HOST);

        stream.synchronize();
        m_onOutputTensor(tensor.retrievedlaunchTensorInfo.tensorName, tensor.hostBuffer);
    }
}

uint64_t LauncherBase::getActualTensorSize(const synRetrievedLaunchTensorInfoExt& info,
                                           const std::shared_ptr<DataProvider>&   dataProvider)
{
    if (info.tensorType == DATA_TENSOR || !dataProvider)
    {
        return syn::Tensor::getMaxSizeInBytes(info);
    }

    return dataProvider->getBufferSize(info.tensorName);
}

// called from query tensors
void LauncherBase::allocateSectionsMem(TensorsData& tensors)
{
    std::map<uint32_t, uint64_t> sectionsSizes;

    for (auto& t : tensors.input)
    {
        const auto& info = t.second.retrievedlaunchTensorInfo;
        if (t.second.memType == TensorMemType::DEVICE || info.tensorType == HOST_TO_DEVICE_TENSOR)
        {
            uint64_t offset = getActualTensorSize(info, m_dataProvider) + info.tensorOffsetInSection;
            auto     it     = sectionsSizes.insert(std::pair<uint32_t, uint64_t>(info.tensorSectionId, offset));
            if (!it.second)
            {
                it.first->second = std::max(it.first->second, offset);
            }
        }
    }

    for (auto& t : tensors.output)
    {
        const auto& info = t.second.retrievedlaunchTensorInfo;
        if (t.second.memType == TensorMemType::DEVICE || info.tensorType == HOST_TO_DEVICE_TENSOR)
        {
            uint64_t offset = getActualTensorSize(info, m_dataProvider) + info.tensorOffsetInSection;
            auto     it     = sectionsSizes.insert(std::pair<uint32_t, uint64_t>(info.tensorSectionId, offset));
            if (!it.second)
            {
                it.first->second = std::max(it.first->second, offset);
            }
        }
    }

    const uint32_t alignTo = m_device.getAttribute(DEVICE_ATTRIBUTE_ADDRESS_ALIGNMENT_SIZE);

    // align sections to device requirements and get total size
    for (auto& e : sectionsSizes)
    {
        if ((e.second % alignTo) != 0)
        {
            e.second += alignTo - (e.second % alignTo);
        }
        tensors.sectionsBuffers.emplace(e.first, m_device.malloc(e.second));
    }
}

const std::string Launcher::timeMeasurementToString(TimeMeasurement timeMeasurement)
{
    switch (timeMeasurement)
    {
        case TimeMeasurement::NONE:
            return "none";
        case TimeMeasurement::EVENETS:
            return "events";
        case TimeMeasurement::PROFILER:
            return "profiler";
        case TimeMeasurement::TIME_MEASUREMENT_MAX:
            return "invalid";
    }
    return "invalid";
}

const Launcher::TimeMeasurement Launcher::timeMeasurementFromString(const std::string& name)
{
    if (name == "none") return TimeMeasurement::NONE;
    if (name == "events") return TimeMeasurement::EVENETS;
    if (name == "profiler") return TimeMeasurement::PROFILER;
    return TimeMeasurement::TIME_MEASUREMENT_MAX;
}

Launcher::Result Launcher::launch(syn::Context&                  ctx,
                                  const synDeviceType            deviceType,
                                  const syn::Recipe&             recipe,
                                  const uint32_t                 iterations,
                                  const std::set<synDeviceType>& optionalDeviceTypes,
                                  TimeMeasurement                timeMeasurement,
                                  bool                           keepGoing)
{
    return Launcher::launch(ctx,
                            deviceType,
                            recipe,
                            iterations,
                            optionalDeviceTypes,
                            nullptr,
                            OnTensor {},
                            timeMeasurement,
                            keepGoing);
}

Launcher::Result Launcher::launch(syn::Device&       device,
                                  const syn::Recipe& recipe,
                                  const uint32_t     iterations,
                                  TimeMeasurement    timeMeasurement,
                                  bool               keepGoing)
{
    return Launcher::launch(device, recipe, iterations, nullptr, OnTensor {}, timeMeasurement, keepGoing);
}

Launcher::Result Launcher::launch(syn::Device&                         device,
                                  const syn::Recipe&                   recipe,
                                  const uint32_t                       iterations,
                                  const std::shared_ptr<DataProvider>& dataProvider,
                                  const OnTensor&                      onOutputTensor,
                                  TimeMeasurement                      timeMeasurement,
                                  bool                                 keepGoing)
{
    Launcher launcher(device, recipe, iterations, dataProvider, onOutputTensor);
    launcher.run(timeMeasurement, keepGoing);
    return launcher.m_result;
}

Launcher::Result Launcher::launch(syn::Context&                        ctx,
                                  const synDeviceType                  deviceType,
                                  const syn::Recipe&                   recipe,
                                  const uint32_t                       iterations,
                                  const std::set<synDeviceType>&       optionalDeviceTypes,
                                  const std::shared_ptr<DataProvider>& dataProvider,
                                  const OnTensor&                      onOutputTensor,
                                  TimeMeasurement                      timeMeasurement,
                                  bool                                 keepGoing)
{
    syn::Device device = acquireDevice(ctx, deviceType, optionalDeviceTypes);
    return Launcher::launch(device, recipe, iterations, dataProvider, onOutputTensor, timeMeasurement, keepGoing);
}

Launcher::Result Launcher::launch(syn::Context&                         ctx,
                                  const synDeviceType                   deviceType,
                                  const syn::Recipe&                    recipe,
                                  const uint32_t                        iterations,
                                  const std::set<synDeviceType>&        optionalDeviceTypes,
                                  const std::shared_ptr<DataProvider>&  dataProvider,
                                  const std::shared_ptr<DataCollector>& dataCollector,
                                  TimeMeasurement                       timeMeasurement,
                                  bool                                  keepGoing)
{
    return Launcher::launch(
        ctx,
        deviceType,
        recipe,
        iterations,
        optionalDeviceTypes,
        dataProvider,
        [&dataCollector](const std::string& name, const syn::HostBuffer& buffer) {
            dataCollector->setBuffer(name, buffer);
        },
        timeMeasurement,
        keepGoing);
}

Launcher::Result Launcher::launch(syn::Device&                          device,
                                  const syn::Recipe&                    recipe,
                                  const uint32_t                        iterations,
                                  const std::shared_ptr<DataProvider>&  dataProvider,
                                  const std::shared_ptr<DataCollector>& dataCollector,
                                  TimeMeasurement                       timeMeasurement,
                                  bool                                  keepGoing)
{
    return Launcher::launch(
        device,
        recipe,
        iterations,
        dataProvider,
        [&dataCollector](const std::string& name, const syn::HostBuffer& buffer) {
            dataCollector->setBuffer(name, buffer);
        },
        timeMeasurement,
        keepGoing);
}

syn::Device Launcher::tryAcquireDevice(syn::Context& ctx, const std::set<synDeviceType>& deviceTypes)
{
    for (const auto& t : deviceTypes)
    {
        if (ctx.getDeviceCount(t) == 0) continue;
        try
        {
            return ctx.acquire(t);
        }
        catch (...)
        {
            // failed to acquire
        }
    }
    return syn::Device();
}

syn::Device Launcher::acquireDevice(syn::Context&                  ctx,
                                    const synDeviceType            deviceType,
                                    const std::set<synDeviceType>& optionalDeviceTypes)
{
    syn::Device dev = tryAcquireDevice(ctx, {deviceType});
    if (dev) return dev;
    dev = tryAcquireDevice(ctx, optionalDeviceTypes);
    if (dev) return dev;
    throw std::runtime_error("failed to acquire device");
}

std::vector<double> Launcher::filterElapsedTime(std::vector<double> durations, uint64_t maxValidDuration)
{
    std::vector<double> ret;
    if (durations.empty()) return ret;

    const double varianceThreshold = 0.3;

    size_t median_index = durations.size() / 2;
    std::nth_element(durations.begin(), durations.begin() + median_index, durations.end());

    double median = durations[median_index];

    for (const auto& d : durations)
    {
        // invalid result, force re-run
        if (d > maxValidDuration) return {};
        // 0 is invalid result, Jira: SW-[116547]
        if (d == 0) continue;
        // remove outliers
        if ((d < (1 - varianceThreshold) * median) || (d > (1 + varianceThreshold) * median)) continue;
        ret.push_back(d);
    }
    return ret;
}

std::vector<double> Launcher::pollElapsedTime(const std::vector<TimeEvents>& timeEvents)
{
    std::vector<double> times;
    times.reserve(timeEvents.size());
    for (const auto& e : timeEvents)
    {
        try
        {
            times.push_back(e.begin.getElapsedTime(e.end));
        }
        catch (const syn::Exception& e)
        {
            if (e.status() != synUnavailable)
            {
                return {};  // try to recover from getElapsedTime failure
            }
        }
    }

    return times;
}

Launcher::Launcher(syn::Device&                         device,
                   const syn::Recipe&                   recipe,
                   const uint32_t                       iterations,
                   const std::shared_ptr<DataProvider>& dataProvider,
                   const OnTensor&                      onOutputTensor)
: LauncherBase(device, recipe, dataProvider, onOutputTensor),
  m_iterations(iterations),
  m_event(device.createEvent()),
  m_streamUpload(device.createStream()),
  m_streamCompute(device.createStream())
{
}

Launcher::Launcher(syn::Device&                          device,
                   const syn::Recipe&                    recipe,
                   const uint32_t                        iterations,
                   const std::shared_ptr<DataProvider>&  dataProvider,
                   const std::shared_ptr<DataCollector>& dataCollector)
: Launcher(device,
           recipe,
           iterations,
           dataProvider,
           [&dataCollector](const std::string& name, const syn::HostBuffer& buffer) {
               dataCollector->setBuffer(name, buffer);
           })
{
}

void Launcher::launchAndMeasureDuration(TimeMeasurement timeMeasurement, bool keepGoing)
{
    bool shouldMeasureTime =
        timeMeasurement == TimeMeasurement::EVENETS || timeMeasurement == TimeMeasurement::PROFILER;
    const uint32_t minRequiredIterations = shouldMeasureTime ? m_iterations : 0;
    size_t         maxAttempts           = 100;  // w/a elapsed time == 0 bug
    size_t         attempt               = 0;
    uint32_t       orgIterationsCount    = m_iterations;
    do
    {
        if (attempt && timeMeasurement == TimeMeasurement::EVENETS)
        {
            m_iterations = std::min(m_iterations * 2, orgIterationsCount * 10);
        }
        launch(timeMeasurement);
    } while (++attempt < maxAttempts && m_result.durations.size() < minRequiredIterations);

    if (!keepGoing && attempt == maxAttempts && m_result.durations.size() < minRequiredIterations)
    {
        throw std::runtime_error(fmt::format("failed to capture device run time"));
    }

    if (m_iterations != orgIterationsCount)
    {
        m_result.warnings.push_back(
            fmt::format("failed to capture run time of few iterations, requested iterations count: {}, actual "
                        "iterations count: {}, valid iteration count: {}, launch attempts: {}",
                        orgIterationsCount,
                        m_iterations,
                        m_result.durations.size(),
                        attempt));
    }
}

void Launcher::allocateAndDownloadInputTensors()
{
    size_t                       activeDownloadSize    = 0;
    const size_t                 maxActiveDownloadSize = 1e9;
    std::vector<syn::HostBuffer> release;

    for (auto& it : m_tensors.input)
    {
        auto& tensor = it.second;

        if (tensor.memType != TensorMemType::DEVICE) continue;

        allocateBuffers(tensor, m_tensors.sectionsBuffers);
        setHostBuffer(tensor.retrievedlaunchTensorInfo.tensorName, tensor.hostBuffer);
        activeDownloadSize += tensor.hostBuffer.getSize();
        m_streamDownload.memCopyAsync(tensor.hostBuffer.getAddress(),
                                      tensor.deviceBuffer.getAddress() +
                                          tensor.retrievedlaunchTensorInfo.tensorOffsetInSection,
                                      tensor.hostBuffer.getSize(),
                                      HOST_TO_DRAM);

        release.push_back(tensor.hostBuffer);
        tensor.hostBuffer = {};

        if (activeDownloadSize > maxActiveDownloadSize)
        {
            m_streamDownload.synchronize();
            release.clear();
            activeDownloadSize = 0;
        }
    }
    m_streamDownload.synchronize();
}

void Launcher::releaseInputBuffers()
{
    m_streamDownload.synchronize();
    m_tensors.input.clear();
}

void Launcher::run(TimeMeasurement timeMeasurement, bool keepGoing)
{
    prepareTensors();
    allocateAndDownloadInputTensors();
    launchAndMeasureDuration(timeMeasurement, keepGoing);
    releaseInputBuffers();
    upload();

    m_device.synchronize();
}

void Launcher::prepareTensors()
{
    initTensors(m_tensors);
    splitConstSectionTensors(m_tensors);
    allocateSectionsMem(m_tensors);
    generateLaunchInfos(m_tensors);
}

void Launcher::upload()
{
    // Wait until all uploads are done before copy from HBM
    m_streamUpload.waitEvent(m_event);

    LauncherBase::upload(m_streamUpload, m_tensors);
}

void Launcher::launch(TimeMeasurement timeMeasurement)
{
    switch (timeMeasurement)
    {
        case TimeMeasurement::NONE:
            return launchWihtoutMeasurement();
        case TimeMeasurement::EVENETS:
            return launchAndMeasureTimeEvents();
        case TimeMeasurement::PROFILER:
            return launchAndMeasureProfiler();
        default:
            throw std::runtime_error(fmt::format("unsupported TimeMeasurement value: {}", int(timeMeasurement)));
    }
}

void Launcher::launchWihtoutMeasurement()
{
    uint64_t          workspaceSize   = m_recipe.getWorkspaceSize();
    syn::DeviceBuffer workspaceBuffer = workspaceSize > 0 ? m_device.malloc(workspaceSize) : syn::DeviceBuffer();

    for (size_t i = 0; i < m_iterations; ++i)
    {
        m_streamCompute.launch(m_recipe, m_tensors.concatTensors, workspaceBuffer ? workspaceBuffer.getAddress() : 0);
    }
    // Wait for the completion of the compute
    m_streamCompute.synchronize();
}

void Launcher::delayExecution(const syn::Event&        event,
                              const syn::DeviceBuffer& deviceBuffer,
                              const syn::HostBuffer&   hostBuffer)
{
    m_streamDownload.memCopyAsync(hostBuffer.getAddress(),
                                  deviceBuffer.getAddress(),
                                  hostBuffer.getSize(),
                                  HOST_TO_DRAM);
    m_streamDownload.record(event);
    m_streamCompute.waitEvent(event);
}

void Launcher::launchAndMeasureTimeEvents()
{
    uint64_t          workspaceSize   = m_recipe.getWorkspaceSize();
    syn::DeviceBuffer workspaceBuffer = workspaceSize > 0 ? m_device.malloc(workspaceSize) : syn::DeviceBuffer();

    std::vector<TimeEvents> timeEvents(m_iterations);

    for (size_t i = 0; i < m_iterations; ++i)
    {
        timeEvents[i].begin = m_device.createEvent(EVENT_COLLECT_TIME);
        timeEvents[i].end   = m_device.createEvent(EVENT_COLLECT_TIME);
    }

    syn::Event delayExecutionEvents = m_device.createEvent();

    // Make sure all the launches are submitted before execution starts (using big dummy PDMA copy) -
    //   for event elapsed time accuracy
    constexpr unsigned BIG_COPY_SIZE = 1024 * 1024;
    syn::DeviceBuffer  deviceBuffer  = m_device.malloc(BIG_COPY_SIZE);
    syn::HostBuffer    hostBuffer    = m_device.hostMalloc(BIG_COPY_SIZE);
    delayExecution(delayExecutionEvents, deviceBuffer, hostBuffer);

    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < m_iterations; ++i)
    {
        m_streamCompute.record(timeEvents[i].begin);

        m_streamCompute.launch(m_recipe, m_tensors.concatTensors, workspaceBuffer ? workspaceBuffer.getAddress() : 0);

        m_streamCompute.record(timeEvents[i].end);
    }

    // Wait for the completion of the compute
    m_streamCompute.synchronize();

    const auto stop             = std::chrono::steady_clock::now();
    uint64_t   maxValidDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    m_result.durations = filterElapsedTime(pollElapsedTime(timeEvents), maxValidDuration);
}

void Launcher::printEvent(const synTraceEvent& e, bool increasedRange)
{
#if 0
      fmt::print("{{ name: \"{}\", catergory: \"{}\", type: \"{}\", timestamp: "
                 "\"{}\", duration: \"{}\", engineType: \"{}\", engineIndex: "
                 "\"{}\", contextId: \"{}\", arguments.name: \"{}\" , arguments.value: \"{}\" }} -> {}\n",
                 e.name, e.category, e.type, e.timestamp, e.duration,
                 e.engineType, e.engineIndex, e.contextId, e.arguments.name, e.arguments.value, increasedRange);
#endif
}

long double Launcher::getOverallDurationNs(const std::pair<std::size_t, std::unique_ptr<char[]>>& events)
{
    const auto& [numEvents, rawBuffer] = events;

    long double first = {};
    long double last  = {};
    for (std::size_t i = 0; i < numEvents; ++i)
    {
        synTraceEvent e;
        static_assert(std::is_trivially_copyable_v<decltype(e)>, "");
        std::memcpy(&e, &rawBuffer[i * sizeof(e)], sizeof(e));

        static constexpr std::string_view busy = "Busy";
        auto                              name = std::string_view(e.name);
        if (!startsWith(name, busy)) continue;

        bool increasedRange = false;
        if (e.timestamp < first || first == 0)
        {
            first          = e.timestamp;
            increasedRange = true;
        }
        if (e.timestamp > last || last == 0)
        {
            last           = e.timestamp;
            increasedRange = true;
        }
        printEvent(e, increasedRange);
    }
    return 1e3 * (last - first);
}

void Launcher::launchAndMeasureProfiler()
{
    uint64_t          workspaceSize   = m_recipe.getWorkspaceSize();
    syn::DeviceBuffer workspaceBuffer = workspaceSize > 0 ? m_device.malloc(workspaceSize) : syn::DeviceBuffer();

    auto prof = m_device.createProfiler(synTraceType::synTraceDevice);

    for (size_t i = 0; i < m_iterations; ++i)
    {
        prof.start();
        m_streamCompute.launch(m_recipe, m_tensors.concatTensors, workspaceBuffer ? workspaceBuffer.getAddress() : 0);
        // Wait for the completion of the compute
        m_streamCompute.synchronize();
        prof.stop();

        auto events   = prof.getEvents(synTraceFormat::synTraceFormatTEF);
        auto duration = getOverallDurationNs(events);
        if (duration > 0)
        {
            m_result.durations.push_back(duration);
        }
    }
}