#pragma once

#include "data_collector.h"
#include "data_provider.h"
#include "hpp/syn_context.hpp"
#include "hpp/syn_device.hpp"
#include "hpp/syn_device_buffer.hpp"
#include "hpp/syn_host_buffer.hpp"
#include "synapse_api_types.h"
#include <map>
#include <vector>

/***************************************************************************/
/*                            LauncherBase                                 */
/***************************************************************************/
class LauncherBase
{
public:
    enum class TensorMemType
    {
        NONE,
        DEVICE,
        HOST
    };

    using TensorsInfo      = std::vector<synRetrievedLaunchTensorInfoExt>;
    using DeviceBuffersMap = std::map<uint32_t, syn::DeviceBuffer>;
    using OnTensor         = std::function<void(const std::string&, const syn::HostBuffer&)>;

    struct ConstSectionInfo
    {
        uint32_t      sectionId;
        uint64_t      sectionSize;
        uint64_t      sectionData;
        uint64_t      tensorId;
        uint64_t      tensorOffsetInSection;
        synTensorType tensorType;
        char          tensorName[ENQUEUE_TENSOR_NAME_MAX_SIZE];
    };
    struct TensorData
    {
        synRetrievedLaunchTensorInfoExt retrievedlaunchTensorInfo;
        synLaunchTensorInfoExt          launchInfo;
        syn::HostBuffer                 hostBuffer;
        syn::DeviceBuffer               deviceBuffer;
        std::optional<ConstSectionInfo> tensorConstSectionInfo;
        TensorMemType                   memType;
    };

    using TensorsMap = std::map<uint64_t, TensorData>;

    struct TensorsData
    {
        TensorsMap                          input;
        TensorsMap                          output;
        DeviceBuffersMap                    sectionsBuffers;
        DeviceBuffersMap                    constSectionsBuffers;
        std::vector<synLaunchTensorInfoExt> concatTensors;
    };

    struct Result
    {
        std::vector<double>      durations;
        std::vector<std::string> warnings;
    };

    LauncherBase(syn::Device&                         device,
                 const syn::Recipe&                   recipe,
                 const std::shared_ptr<DataProvider>& dataProvider   = nullptr,
                 const OnTensor&                      onOutputTensor = {});

    LauncherBase(syn::Device&                          device,
                 const syn::Recipe&                    recipe,
                 const std::shared_ptr<DataProvider>&  dataProvider,
                 const std::shared_ptr<DataCollector>& dataCollector);

    static TensorMemType getTensorMemType(synTensorType type);

    void verifyTensorShape(const std::vector<TSize>& shape, const synRetrievedLaunchTensorInfoExt& info);

    void getSectionInfo(synSectionId sectionId, bool& isConstSection, uint64_t& section_size, uint64_t& section_data);

    void downloadConstSectionDataToDevice(const ConstSectionInfo& sectionInfo, DeviceBuffersMap& constSectionDeviceBuf);

    void splitConstSectionTensors(TensorsData& tensors);

    synLaunchTensorInfoExt generateLaunchInfo(const synRetrievedLaunchTensorInfoExt& info, uint64_t tensorAddress);

    synLaunchTensorInfoExt generateConstSectionLaunchInfo(const ConstSectionInfo& sectionInfo,
                                                          DeviceBuffersMap&       constSectionDeviceBuf);

    uint64_t getTensorAddress(const TensorData& tensor, const DeviceBuffersMap sectionsBuffers);

    void generateLaunchInfos(TensorsData& tensors);

    void initTensors(TensorsData& tensors);

    void setHostBuffer(const std::string& tensorName, syn::HostBuffer& hostBuffer);

    void allocateBuffers(TensorData& tensor, const DeviceBuffersMap sectionsBuffers);

    void allocateInputBuffers(TensorsData& tensors);

    void queryTensors(TensorsData& tensors);

    void download(syn::Stream stream, const TensorsData& tensors);

    void upload(syn::Stream stream, const TensorsData& tensors);

protected:
    // used by query
    static uint64_t getActualTensorSize(const synRetrievedLaunchTensorInfoExt& info,
                                        const std::shared_ptr<DataProvider>&   dataProvider);

    // called from query tensors
    void allocateSectionsMem(TensorsData& tensors);

    syn::Recipe                   m_recipe;
    std::shared_ptr<DataProvider> m_dataProvider;
    syn::Device                   m_device;
    syn::Stream                   m_streamDownload;
    Result                        m_result;
    const OnTensor                m_onOutputTensor;
};

/***************************************************************************/
/*                                Launcher                                 */
/***************************************************************************/
class Launcher : public LauncherBase
{
public:
    enum class TimeMeasurement
    {
        NONE,
        EVENETS,
        PROFILER,
        TIME_MEASUREMENT_MAX
    };

    static const std::string timeMeasurementToString(TimeMeasurement timeMeasurement);

    static const TimeMeasurement timeMeasurementFromString(const std::string& name);

    static Result launch(syn::Context&                  ctx,
                         const synDeviceType            deviceType,
                         const syn::Recipe&             recipe,
                         const uint32_t                 iterations          = 1,
                         const std::set<synDeviceType>& optionalDeviceTypes = {},
                         TimeMeasurement                timeMeasurement     = TimeMeasurement::EVENETS,
                         bool                           keepGoing           = false);

    static Result launch(syn::Device&       device,
                         const syn::Recipe& recipe,
                         const uint32_t     iterations      = 1,
                         TimeMeasurement    timeMeasurement = TimeMeasurement::EVENETS,
                         bool               keepGoing       = false);

    static Result launch(syn::Context&                        ctx,
                         const synDeviceType                  deviceType,
                         const syn::Recipe&                   recipe,
                         const uint32_t                       iterations,
                         const std::set<synDeviceType>&       optionalDeviceTypes,
                         const std::shared_ptr<DataProvider>& dataProvider,
                         const OnTensor&                      onOutputTensor,
                         TimeMeasurement                      timeMeasurement = TimeMeasurement::EVENETS,
                         bool                                 keepGoing       = false);

    static Result launch(syn::Device&                         device,
                         const syn::Recipe&                   recipe,
                         const uint32_t                       iterations,
                         const std::shared_ptr<DataProvider>& dataProvider,
                         const OnTensor&                      onOutputTensor,
                         TimeMeasurement                      timeMeasurement = TimeMeasurement::EVENETS,
                         bool                                 keepGoing       = false);

    static Result launch(syn::Context&                         ctx,
                         const synDeviceType                   deviceType,
                         const syn::Recipe&                    recipe,
                         const uint32_t                        iterations,
                         const std::set<synDeviceType>&        optionalDeviceTypes,
                         const std::shared_ptr<DataProvider>&  dataProvider,
                         const std::shared_ptr<DataCollector>& dataCollector,
                         TimeMeasurement                       timeMeasurement = TimeMeasurement::EVENETS,
                         bool                                  keepGoing       = false);

    static Result launch(syn::Device&                          device,
                         const syn::Recipe&                    recipe,
                         const uint32_t                        iterations,
                         const std::shared_ptr<DataProvider>&  dataProvider,
                         const std::shared_ptr<DataCollector>& dataCollector,
                         TimeMeasurement                       timeMeasurement = TimeMeasurement::EVENETS,
                         bool                                  keepGoing       = false);

    static syn::Device tryAcquireDevice(syn::Context& ctx, const std::set<synDeviceType>& deviceTypes);

    static syn::Device acquireDevice(syn::Context&                  ctx,
                                     const synDeviceType            deviceType,
                                     const std::set<synDeviceType>& optionalDeviceTypes);

private:
    class TimeEvents
    {
    public:
        syn::Event begin;
        syn::Event end;
    };

    static std::vector<double> filterElapsedTime(std::vector<double> durations, uint64_t maxValidDuration);

    static std::vector<double> pollElapsedTime(const std::vector<TimeEvents>& timeEvents);

    Launcher(syn::Device&                         device,
             const syn::Recipe&                   recipe,
             const uint32_t                       iterations,
             const std::shared_ptr<DataProvider>& dataProvider,
             const OnTensor&                      onOutputTensor);

    Launcher(syn::Device&                          device,
             const syn::Recipe&                    recipe,
             const uint32_t                        iterations,
             const std::shared_ptr<DataProvider>&  dataProvider,
             const std::shared_ptr<DataCollector>& dataCollector);

    void launchAndMeasureDuration(TimeMeasurement timeMeasurement, bool keepGoing);

    void releaseInputBuffers();

    void run(TimeMeasurement timeMeasurement, bool keepGoing);

    void prepareTensors();

    void allocateAndDownloadInputTensors();

    void upload();

    void launch(TimeMeasurement timeMeasurement);

    void launchWihtoutMeasurement();

    void
    delayExecution(const syn::Event& event, const syn::DeviceBuffer& deviceBuffer, const syn::HostBuffer& hostBuffer);

    void launchAndMeasureTimeEvents();

    void printEvent(const synTraceEvent& e, bool increasedRange);

    long double getOverallDurationNs(const std::pair<std::size_t, std::unique_ptr<char[]>>& events);

    void launchAndMeasureProfiler();

    uint32_t    m_iterations;
    TensorsData m_tensors;
    syn::Event  m_event;
    syn::Stream m_streamUpload;
    syn::Stream m_streamCompute;
};
