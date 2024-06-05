#include "infra/gc_synapse_test.h"
#include "synapse_api.h"
#include "infra/event_triggered_logger.hpp"

#include <thread>
#include <condition_variable> // std::condition_variable
#include <functional>

uint32_t deviceId;
#define BLOCK_SIZE 1024
const char* tensor1_name = "tensor_0";
const char* tensor2_name = "tensor_1";
const char* tensor3_name = "tensor_2";

bool silentConsole = true;

class SynGaudiThreads : public SynTest
{
public:
    SynGaudiThreads()
    {
        if (m_deviceType == synDeviceTypeInvalid)
        {
            LOG_WARN(SYN_TEST, "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi");
            m_deviceType = synDeviceGaudi;
        }
        setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
    }
};

TEST_F_GC(SynGaudiThreads, tf_memcpy_flow_ASIC_CI)
{
    synStatus status;
    synStreamHandle computationStream, h2dStream, d2hStream;

    status = synStreamCreateGeneric(&computationStream, deviceId, 0);
    ASSERT_EQ(synSuccess, status) << "Failed to create compute stream";

    status = synStreamCreateGeneric(&h2dStream, deviceId, 0);
    ASSERT_EQ(synSuccess, status) << "Failed to create DL stream";

    status = synStreamCreateGeneric(&d2hStream, deviceId, 0);
    ASSERT_EQ(synSuccess, status) << "Failed to create UL stream";

    synEventHandle recvEventT1, recvEventT2, compEvent, d2hEvent;

    std::vector<uint8_t> buffer, returned;
    buffer.resize(BLOCK_SIZE * BLOCK_SIZE);
    returned.resize(BLOCK_SIZE * BLOCK_SIZE);
    void *buf1 = buffer.data();
    void *buf2 = buffer.data() + BLOCK_SIZE;

    for (int i = 0; i < buffer.size() / 4; i++)
    {
        reinterpret_cast<int32_t *>(buffer.data())[i] = i;
    }

    uint64_t devMem;
    status = synDeviceMalloc(/*deviceId=*/deviceId,
            /*size=*/BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE,
            /*reqAddr=*/0x0,
            /*flags=*/0,
            /*buffer=*/&devMem);
    ASSERT_EQ(synSuccess, status) << "Failed to allocate device memory";


    status = synHostMap(/*deviceId=*/deviceId,
            /*size=*/BLOCK_SIZE * 2,
            /*buffer=*/(const void *) returned.data());
    ASSERT_EQ(synSuccess, status) << "Failed to host memory";

    status = synHostMap(/*deviceId=*/deviceId,
            /*size=*/BLOCK_SIZE,
            /*buffer=*/(const void *) buf1);
    ASSERT_EQ(synSuccess, status) << "Failed to host memory";

    status = synHostMap(/*deviceId=*/deviceId,
            /*size=*/BLOCK_SIZE,
            /*buffer=*/(const void *) buf2);
    ASSERT_EQ(synSuccess, status) << "Failed to host memory";

    synGraphHandle graphHandle;
    status = synGraphCreate(&graphHandle, m_deviceType);
    ASSERT_EQ(synSuccess, status) << "Failed to create concat graph";

    ASSERT_EQ(synSuccess, status) << "Failed to create first tensor section";

    unsigned dims1[5] = {256, 0, 0, 0, 0};
    synTensor tensor1  = createTrainingTensor(1, syn_type_int32, dims1, true, tensor1_name, graphHandle);

    unsigned dims2[5] = {256, 0, 0, 0, 0};
    synTensor tensor2  = createTrainingTensor(1, syn_type_int32, dims2, true, tensor2_name, graphHandle);

    unsigned dims3[5] = {512, 0, 0, 0, 0};
    synTensor tensor3  = createTrainingTensor(1, syn_type_int32, dims3, true, tensor3_name, graphHandle);

    synTensor node1_inputs[] = {tensor1, tensor2};
    synTensor node1_outputs[] = {tensor3};
    uint8_t object_1_data[] = {0x0, 0x0, 0x0, 0x0};

    status = synNodeCreate(/*graphHandle=*/graphHandle, /*pInputsTensorList=*/node1_inputs,
            /*pOutputsTensorList=*/node1_outputs, /*numberInputs=*/2, /*numberOutputs=*/1,
            /*pUserParams=*/(const void *) object_1_data, /*paramsSize=*/0x4, /*pGuid=*/"concat", /*pName=*/"",
            /*inputLayouts=*/nullptr, /*outputLayouts=*/nullptr);
    ASSERT_EQ(synSuccess, status) << "Failed to create concat node";

    synRecipeHandle recipe1;
    status = synGraphCompile(/*pRecipeHandle=*/&recipe1, /*graphHandle=*/graphHandle,
            /*pRecipeName=*/GetTestFileName().c_str(), /*pBuildLog=*/nullptr);
    ASSERT_EQ(synSuccess, status) << "Failed to compile graph";

    auto robota = [&](synEventHandle *ev, void *host_data, int offset)
    {
        uint64_t device_data = devMem + offset;
        status = synMemCopyAsync(/*streamHandle=*/h2dStream,
                /*src=*/(uint64_t) host_data,
                /*size=*/BLOCK_SIZE,
                /*dst=*/device_data,
                /*direction=*/synDmaDir::HOST_TO_DRAM);
        ASSERT_EQ(synSuccess, status) << "Failed to copy tensor to device";

        status = synEventCreate(ev, deviceId, 0);
        ASSERT_EQ(synSuccess, status) << "Failed to create event";

        status = synEventRecord(/*eventHandle=*/*ev,
                /*streamHandle=*/h2dStream);
        ASSERT_EQ(synSuccess, status) << "Failed to record event";
    };
    uint32_t totalNumOfTensors = 3;

    synLaunchTensorInfo launch_tensors_info1[totalNumOfTensors] = {{tensor1_name, 0, DATA_TENSOR, {0, 0, 0, 0, 0}, 0},
                                                                   {tensor2_name, 0, DATA_TENSOR, {0, 0, 0, 0, 0}, 0},
                                                                   {tensor3_name, 0, DATA_TENSOR, {0, 0, 0, 0, 0}, 0}};
    prepareTensorInfo(recipe1, launch_tensors_info1, totalNumOfTensors);
    for (int N = 0; N < 10000; N++)
    {
        void *ptr1 = N % 2 ? buf1 : buf2;
        void *ptr2 = N % 2 ? buf2 : buf1;

        std::thread fred1(robota, &recvEventT1, ptr1, 0);
        std::thread fred2(robota, &recvEventT2, ptr2, BLOCK_SIZE);

        fred1.join();
        fred2.join();

        status = synStreamWaitEvent(/*streamHandle=*/computationStream,
                /*eventHandle=*/recvEventT1,
                /*flags=*/0);
        ASSERT_EQ(synSuccess, status) << "Failed to wait on T1 operation";

        status = synStreamWaitEvent(/*streamHandle=*/computationStream,
                /*eventHandle=*/recvEventT2,
                /*flags=*/0);
        ASSERT_EQ(synSuccess, status) << "Failed to wait on T2 operation";
        launch_tensors_info1[0].pTensorAddress = devMem + (N % 2 ? 0 : BLOCK_SIZE);
        launch_tensors_info1[1].pTensorAddress = devMem + (N % 2 ? BLOCK_SIZE : 0);
        launch_tensors_info1[2].pTensorAddress = devMem + 2 * BLOCK_SIZE;

        status = synLaunch(/*streamHandle=*/computationStream,
                           /*launchTensorsInfo=*/launch_tensors_info1,
                           /*numberTensors=*/totalNumOfTensors,
                           /*pWorkspace=*/devMem + BLOCK_SIZE * 20,
                           /*pRecipehandle=*/recipe1,
                           0);
        ASSERT_EQ(synSuccess, status) << "Failed to create compute stream";

        status = synEventCreate(&compEvent, deviceId, 0);
        ASSERT_EQ(synSuccess, status) << "Failed to compute";

        status = synEventRecord(/*eventHandle=*/compEvent,
                /*streamHandle=*/computationStream);
        ASSERT_EQ(synSuccess, status) << "Failed to record on compute stream";

        status = synStreamWaitEvent(/*streamHandle=*/d2hStream,
                /*eventHandle=*/compEvent,
                /*flags=*/0);
        ASSERT_EQ(synSuccess, status) << "Failed to wait on compute stream";

        status = synMemCopyAsync(/*streamHandle=*/d2hStream,
                /*src=*/devMem + 2 * BLOCK_SIZE,
                /*size=*/BLOCK_SIZE * 2,
                /*dst=*/(uint64_t) returned.data(),
                /*direction=*/DRAM_TO_HOST);
        ASSERT_EQ(synSuccess, status) << "Failed to copy to device";

        status = synEventCreate(&d2hEvent, deviceId, 0);
        ASSERT_EQ(synSuccess, status) << "Failed to create event for compute stream";

        status = synEventRecord(/*eventHandle=*/d2hEvent,
                /*streamHandle=*/d2hStream);
        ASSERT_EQ(synSuccess, status) << "Failed to record compute stream";

        status = synEventSynchronize(/*eventHandle=*/d2hEvent);
        ASSERT_EQ(synSuccess, status) << "Failed to synchronize event";

        if (memcmp(returned.data() + (N % 2 ? 0 : BLOCK_SIZE), ptr1, BLOCK_SIZE) != 0 ||
            memcmp(returned.data() + (N % 2 ? BLOCK_SIZE : 0), ptr2, BLOCK_SIZE) != 0)
        {
            if (!silentConsole)
            {
                std::cout << std::endl << std::endl;
                std::cout << "T1: \n";
                for (int i = 0; i < BLOCK_SIZE / 4; i++)
                {
                    if (i % 32 == 0 && i)
                        std::cout << "\n";
                    std::cout << std::setw(3) << std::setfill(' ');
                    std::cout << reinterpret_cast<int32_t *>(buf1)[i] << " ";
                }
                std::cout << std::endl << std::endl;
                std::cout << "T2: \n";
                for (int i = 0; i < BLOCK_SIZE / 4; i++)
                {
                    if (i % 32 == 0 && i)
                        std::cout << "\n";
                    std::cout << std::setw(3) << std::setfill(' ');
                    std::cout << reinterpret_cast<int32_t *>(buf2)[i] << " ";
                }
                std::cout << std::endl << std::endl;
                std::cout << "Tout: \n";
                for (int i = 0; i < BLOCK_SIZE / 4; i++)
                {
                    if (i % 32 == 0 && i)
                        std::cout << "\n";
                    std::cout << std::setw(3) << std::setfill(' ');
                    std::cout << reinterpret_cast<int32_t *>(returned.data())[i] << " ";
                }
                std::cout << "\n-------------------------------------------------\n";
                for (int i = BLOCK_SIZE / 4; i < 2048 / 4; i++)
                {
                    if (i % 32 == 0 && i != BLOCK_SIZE / 4)
                        std::cout << "\n";
                    std::cout << std::setw(3) << std::setfill(' ');
                    std::cout << reinterpret_cast<int32_t *>(returned.data())[i] << " ";
                }
                std::cout << std::endl << std::endl;
            }
            ASSERT_EQ("T1 and T2 Concat", "Some other value");
        }

        status = synEventDestroy(/*eventHandle=*/recvEventT1);
        ASSERT_EQ(synSuccess, status) << "Failed to destroy event";

        status = synEventDestroy(/*eventHandle=*/recvEventT2);
        ASSERT_EQ(synSuccess, status) << "Failed to destroy event";

        status = synEventDestroy(/*eventHandle=*/compEvent);
        ASSERT_EQ(synSuccess, status) << "Failed to destroy event";

        status = synEventDestroy(/*eventHandle=*/d2hEvent);
        ASSERT_EQ(synSuccess, status) << "Failed to destroy event";
    }

    status = synStreamDestroy(/*streamHandle=*/computationStream);
    ASSERT_EQ(synSuccess, status) << "Failed to destroy stream";

    status = synStreamDestroy(/*streamHandle=*/h2dStream);
    ASSERT_EQ(synSuccess, status) << "Failed to destroy stream";

    status = synStreamDestroy(/*streamHandle=*/d2hStream);
    ASSERT_EQ(synSuccess, status) << "Failed to destroy stream";

    status = synDeviceFree(deviceId, devMem, 0);
    ASSERT_EQ(synSuccess, status) << "Failed to free device mem";

    status = synHostUnmap(deviceId, buf1);
    ASSERT_EQ(synSuccess, status) << "Failed to unmap host mem";

    status = synHostUnmap(deviceId, buf2);
    ASSERT_EQ(synSuccess, status) << "Failed to unmap host mem";

    status = synHostUnmap(deviceId, returned.data());
    ASSERT_EQ(synSuccess, status) << "Failed to unmap host mem";

    status = synRecipeDestroy(recipe1);
    ASSERT_EQ(synSuccess, status) << "Failed to destroy recipe";

    status = synTensorDestroy(tensor1);
    ASSERT_EQ(synSuccess, status) << "Failed to destroy tensor";

    status = synTensorDestroy(tensor2);
    ASSERT_EQ(synSuccess, status) << "Failed to destroy tensor";

    status = synTensorDestroy(tensor3);
    ASSERT_EQ(synSuccess, status) << "Failed to destroy tensor";

    status = synGraphDestroy(graphHandle);
    ASSERT_EQ(synSuccess, status) << "Failed to destroy graph";
}

TEST_F_GC(SynGaudiThreads, tf_multi_memcpy_threads_ASIC_CI)
{
    auto memcopyOperation = [&](synStreamHandle*    streamHandle,
                                uint64_t            srcData,
                                uint64_t            dstData,
                                uint64_t            copySize,
                                synDmaDir           direction)
    {
        synStatus status = synMemCopyAsync(/*streamHandle=*/    *streamHandle,
                                           /*src=*/             srcData,
                                           /*size=*/            copySize,
                                           /*dst=*/             dstData,
                                           /*direction=*/       direction);
        ASSERT_EQ(synSuccess, status) << "Failed to perform synMemCopyAsync direction " << direction;
    };

    synStatus       status = synSuccess;
    synStreamHandle uploadStream;
    synStreamHandle innerDeviceCopyStream;

    status = synStreamCreateGeneric(&uploadStream, deviceId, 0);
    ASSERT_EQ(synSuccess, status) << "Failed to create upload stream";

    status = synStreamCreateGeneric(&innerDeviceCopyStream, deviceId, 0);
    ASSERT_EQ(synSuccess, status) << "Failed to create upload stream";

    uint64_t bufferSize = 500 * 1024 * 1024; // 500 MB

    uint8_t* hostBuffer = new uint8_t[bufferSize];

    status = synHostMap(/*deviceId=*/ deviceId,
                        /*size=*/     bufferSize,
                        /*buffer=*/   (const void *) hostBuffer);
    ASSERT_EQ(synSuccess, status) << "Failed to map host memory";

    uint64_t deviceMemoryHostUsage;
    status = synDeviceMalloc(/*deviceId=*/ deviceId,
                             /*size=*/     bufferSize,
                             /*reqAddr=*/  0x0,
                             /*flags=*/    0,
                             /*buffer=*/   &deviceMemoryHostUsage);
    ASSERT_EQ(synSuccess, status) << "Failed to allocate device memory (deviceMemoryHostUsage)";

    uint64_t deviceMemoryInnerSrc, deviceMemoryInnerDst;
    status = synDeviceMalloc(/*deviceId=*/ deviceId,
                             /*size=*/     bufferSize,
                             /*reqAddr=*/  0x0,
                             /*flags=*/    0,
                             /*buffer=*/   &deviceMemoryInnerSrc);
    ASSERT_EQ(synSuccess, status) << "Failed to allocate device memory (deviceMemoryInnerSrc)";

    status = synDeviceMalloc(/*deviceId=*/ deviceId,
                             /*size=*/     bufferSize,
                             /*reqAddr=*/  0x0,
                             /*flags=*/    0,
                             /*buffer=*/   &deviceMemoryInnerDst);
    ASSERT_EQ(synSuccess, status) << "Failed to allocate device memory (deviceMemoryInnerDst)";

    for (int i = 0; i < 100; i++)
    {
        std::thread thread1(memcopyOperation,
                            &uploadStream,
                            (uint64_t) deviceMemoryHostUsage,
                            (uint64_t) hostBuffer,
                            bufferSize,
                            DRAM_TO_HOST);

        std::thread thread2(memcopyOperation,
                            &innerDeviceCopyStream,
                            (uint64_t) deviceMemoryInnerSrc,
                            (uint64_t) deviceMemoryInnerDst,
                            bufferSize,
                            DRAM_TO_DRAM);

        thread1.join();
        thread2.join();
    }

    status = synStreamSynchronize(/*synStreamHandle=*/ uploadStream);
    ASSERT_EQ(synSuccess, status) << "Failed to synchronize stream (uploadStream))";

    status = synStreamSynchronize(/*synStreamHandle=*/ innerDeviceCopyStream);
    ASSERT_EQ(synSuccess, status) << "Failed to synchronize stream (innerDeviceCopyStream))";

    status = synDeviceFree(deviceId, deviceMemoryInnerDst, 0);
    ASSERT_EQ(synSuccess, status) << "Failed to free device memory (deviceMemoryInnerDst)";

    status = synDeviceFree(deviceId, deviceMemoryInnerSrc, 0);
    ASSERT_EQ(synSuccess, status) << "Failed to free device memory (deviceMemoryInnerSrc)";

    status = synDeviceFree(deviceId, deviceMemoryHostUsage, 0);
    ASSERT_EQ(synSuccess, status) << "Failed to free device memory (deviceMemoryHostUsage)";

    status = synHostUnmap(deviceId, hostBuffer);
    ASSERT_EQ(synSuccess, status) << "Failed to unmap host memory";

    delete[] hostBuffer;

    status = synStreamDestroy(/*streamHandle=*/ uploadStream);
    ASSERT_EQ(synSuccess, status) << "Failed to destroy upload stream";

    status = synStreamDestroy(/*streamHandle=*/ innerDeviceCopyStream);
    ASSERT_EQ(synSuccess, status) << "Failed to destroy upload stream";
}

/***************************************************************************/
//Some functions/structures used for TEST_F_GC(SynGaudiThreads, multi_threaded_multi_graph)
//This struct holds all the needed information for one run
struct oneRun
{
    std::vector<testEventHandle>    event_vec;
    std::vector<std::vector<float>> inDataVec;
    std::vector<uint64_t>           inDevAddrVec;
    uint64_t                        outDevAddr;
    testEventHandle                 eventCompute;
    std::mutex                      mtx;
    std::condition_variable         cv;
    int                             inputSent   = 0;
    bool                            computeSent = false;
    int                             graphId;
};


struct tfSimpleFlowData
{
    tfSimpleFlowData(synDeviceType deviceType);
    ~tfSimpleFlowData();

    static constexpr int IN_SIZE            =  256;
    static constexpr int DEV_MEM_SIZE       = 1024 * 1024; //Needed: (IN_SIZE*NUM_INPUTS+OUT_SIZE*NUM_OUTPUTS)*sizeof(float)*NUM_RUNS = (256*2+512*1)*4*50=200K
    static constexpr int NUM_INPUTS         = 2;   //assume same number of inputs for all graphs
    static constexpr int NUM_OUTPUTS        = 1;
    static constexpr int NUM_RUNS           = 50;  //limiting to 50. When we get to around 80 we hit the limit: "static const uint64_t maximalCacheAmount = 1024;"
    static constexpr int NUM_LAUNCH_THREADS = 2;   // Num of threads that call launch
    static constexpr int NUM_RES_THREADS    = 1;   //Num of threads to get the result
    static constexpr int NUM_GRAPHS         = 2;   //Num of different graphs

    testStreamHandle computeStream {deviceId, "compute"};
    testStreamHandle h2dStream {deviceId, "h2d"};
    testStreamHandle d2hStream {deviceId, "d2h"};

    oneRun           runArr[NUM_RUNS];
    uint64_t         devMem = 0;
    testGraphHandle*                                               graph_arr[NUM_GRAPHS];
    std::function<bool(std::vector<std::vector<float>>&, float[])> resultFunction[NUM_GRAPHS];
};

tfSimpleFlowData::~tfSimpleFlowData()
{
    for (int i = 0; i < NUM_GRAPHS; i++)
    {
        delete graph_arr[i];
    }
}

tfSimpleFlowData::tfSimpleFlowData(synDeviceType deviceType)
{
    for (int i = 0; i < NUM_GRAPHS; i++)
    {
        graph_arr[i] = new testGraphHandle(deviceType);
        if (graph_arr[i] == nullptr)
        {
            printf("error allocating graph_arr[%d]\n", i);
        }
    }
    for(int run = 0; run < NUM_RUNS; run++)
    {
        auto& singleRun = runArr[run];
        singleRun.event_vec.resize(NUM_INPUTS);
        singleRun.inDevAddrVec.resize(NUM_INPUTS);
        singleRun.eventCompute.init("event_comp" + std::to_string(run), deviceId);
        singleRun.graphId = run % NUM_GRAPHS;
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            std::vector<float> v(IN_SIZE);  // will become empty afterward
            singleRun.inDataVec.emplace_back(std::move(v));  // can be replaced by
            singleRun.event_vec[j].init("event" + std::to_string(run) + "-" + std::to_string(j), deviceId);
        }
    }
}

void host2dev_thread(tfSimpleFlowData& data, int thread_num)
{
    //Each thread sets one input of every run
    for(int run = 0; run < data.NUM_RUNS; run++)
    {
        auto& singleRun = data.runArr[run];
        std::vector<float> &in = singleRun.inDataVec[thread_num];
        for(int i = 0; i < data.IN_SIZE; i++)
        {
            in[i] = i + run + thread_num;
        }
        size_t inByteSize = in.size() * sizeof(in[0]);

        synStatus status = synHostMap(deviceId, inByteSize, in.data());
        ASSERT_EQ(synSuccess, status) << "Failed to hostMap memory" << thread_num << " " << std::this_thread::get_id() << "\n";

        status = synMemCopyAsync(data.h2dStream.get(), (uint64_t)in.data(), inByteSize, singleRun.inDevAddrVec[thread_num], synDmaDir::HOST_TO_DRAM);
        ASSERT_EQ(synSuccess, status) << "Failed to copy tensor to device" << thread_num;

        singleRun.event_vec[thread_num].record(data.h2dStream);
        std::unique_lock<std::mutex> lck(singleRun.mtx);
        singleRun.inputSent++;
        singleRun.cv.notify_all(); //notify compute thread that event was record
    }
}

void compute_thread(tfSimpleFlowData& data, int thread_num)
{
    // Each thread waits for the inputs to be ready and launch a graph-> Each thread handles differnt runs
    for(int run = 0; run < data.NUM_RUNS; run++)
    {
        if(run % data.NUM_LAUNCH_THREADS != thread_num) continue;
        //wait for all inputs before lunch
        auto& singleRun = data.runArr[run];
        {
            std::unique_lock<std::mutex> lck(singleRun.mtx);
            while (singleRun.inputSent != data.NUM_INPUTS) singleRun.cv.wait(lck); //wait for all inputs to be dma start
        }
        for(int i = 0; i < data.NUM_INPUTS; i++)
        {
            singleRun.event_vec[i].wait(data.computeStream); //wait for dma end of both inputs
        }
        auto& graph = data.graph_arr[singleRun.graphId];
        const int totalNumTensors = data.NUM_INPUTS + data.NUM_OUTPUTS;
        if(singleRun.graphId == 0)
        {
            synLaunchTensorInfo launchTensors[] = {{"in0_1", singleRun.inDevAddrVec[0], DATA_TENSOR, {0, 0, 0, 0, 0}, 0},
                                                   {"in0_2", singleRun.inDevAddrVec[1], DATA_TENSOR, {0, 0, 0, 0, 0}, 1},
                                                   {"out0", singleRun.outDevAddr, DATA_TENSOR, {0, 0, 0, 0, 0}, 2}};

            //synLaunch - Enqueue into the graph the inputs that the input_ptr points to, the device will execute on any input it can
            graph->launch(data.computeStream, launchTensors, totalNumTensors, SYN_FLAGS_TENSOR_NAME);
        }
        else
        {
            synLaunchTensorInfo launchTensors[] = {{"in1_1", singleRun.inDevAddrVec[0], DATA_TENSOR, {0, 0, 0, 0, 0}, 0},
                                                   {"in1_2", singleRun.inDevAddrVec[1], DATA_TENSOR, {0, 0, 0, 0, 0}, 1},
                                                   {"out1", singleRun.outDevAddr, DATA_TENSOR, {0, 0, 0, 0, 0}, 2}};

            //synLaunch - Enqueue into the graph the inputs that the input_ptr points to, the device will execute on any input it can

            graph->launch(data.computeStream, launchTensors, totalNumTensors, 0);
        }
        singleRun.eventCompute.record(data.computeStream);
        std::unique_lock<std::mutex> lck(singleRun.mtx);
        singleRun.computeSent = true;
        singleRun.cv.notify_all();  // notify result thread that graph was launched
    }
}

void result_thread(tfSimpleFlowData& data, int thread_num)
{
    //The thread waits for the results and checks the results
    std::array<float, tfSimpleFlowData::IN_SIZE> output = {0};
    synStatus status = synHostMap(deviceId, sizeof(output), output.data());
    ASSERT_EQ(status, synSuccess) << "Failed to map outpout";

    for(int run = 0; run < data.NUM_RUNS; run++)
    {
        if(run % data.NUM_RES_THREADS != thread_num) continue;

        auto& singleRun = data.runArr[run];
        {
            std::unique_lock<std::mutex> lck(singleRun.mtx);
            while (!singleRun.computeSent) singleRun.cv.wait(lck);
        }

        singleRun.eventCompute.wait(data.d2hStream); //wait for data

        for(auto& x : output) x = 0.1234; //just put garbage
        synStatus status = synMemCopyAsync(data.d2hStream.get(), singleRun.outDevAddr, data.IN_SIZE * sizeof(float), (uint64_t)output.data(), DRAM_TO_HOST);
        ASSERT_EQ(status, synSuccess) << "Failed copy from the device to output";

        status = synStreamSynchronize(data.d2hStream.get()); //wait for the MemCopy
        ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream (copy from the device)";

        if(data.resultFunction[singleRun.graphId] != nullptr) //if check function is set, call it
        {
            bool res = data.resultFunction[singleRun.graphId](singleRun.inDataVec, output.data());
            ASSERT_EQ(res, false) << "result func failed " << run << " " << singleRun.graphId;
        }
        for(int inNum = 0; inNum <data.NUM_INPUTS; inNum++)
        {
            std::vector<float> &in = singleRun.inDataVec[inNum];
            status = synHostUnmap(deviceId, in.data());
            ASSERT_EQ(status, synSuccess) << "Failed to unmap input tensor in run " << run << "inNum " << inNum;
        }
    } //run
    status = synHostUnmap(deviceId, output.data());
    ASSERT_EQ(status, synSuccess) << "Failed to unmap outpout";
}

void create_graphs(tfSimpleFlowData &data)
{
    //This function creates the two graph that are later ran
    //Graph0
    {
        auto& graph0 = data.graph_arr[0];
        graph0->set_name("gemm");
        const int matrixSize = 16;

        unsigned sizesIn1[SYN_MAX_TENSOR_DIM] = {1, 1, matrixSize, matrixSize};
        unsigned sizesIn2[SYN_MAX_TENSOR_DIM] = {1, 1, matrixSize, matrixSize};
        unsigned sizesOut[SYN_MAX_TENSOR_DIM] = {1, 1, matrixSize, matrixSize};

        graph0->addTensor(4U, syn_type_single, sizesIn1, true /*isPersist*/, "in0_1");
        graph0->addTensor(4U, syn_type_single, sizesIn1, false /*isPersist*/, "in0_1dma");
        graph0->addTensor(4U, syn_type_single, sizesIn2, true /*isPersist*/, "in0_2");
        graph0->addTensor(4U, syn_type_single, sizesOut, true /*isPersist*/, "out0");

        // DMA node
        const char* dmaNodeGuid   = "memcpy";
        char        dmaNodeName[] = "dma_node";

        synTensor inputs1[]  = {graph0->getTensorHandle("in0_1")};
        synTensor outputs1[] = {graph0->getTensorHandle("in0_1dma")};

        synStatus status = synNodeCreate(graph0->get(),
                                         inputs1,
                                         outputs1,
                                         1,
                                         1,
                                         nullptr,
                                         0,
                                         dmaNodeGuid,
                                         dmaNodeName,
                                         nullptr,
                                         nullptr);
        ASSERT_EQ(status, synSuccess) << "Failed to create Add Node";


        const char* addNodeGuid   = "gemm";
        char        addNodeName[] = "gemmnode";

        synTensor inputs2[]  = {graph0->getTensorHandle("in0_1dma"), graph0->getTensorHandle("in0_2")};
        synTensor outputs2[] = {graph0->getTensorHandle("out0")};
        status               = synNodeCreate(graph0->get(),
                               inputs2,
                               outputs2,
                               2,
                               1,
                               nullptr,
                               0,
                               addNodeGuid,
                               addNodeName,
                               nullptr,
                               nullptr);
        ASSERT_EQ(status, synSuccess) << "Failed to create Add Node";
        graph0->compile(deviceId);

        //Function to check the results
#define ROW_COL2ELM(row, col) (row * matrixSize + col)
        data.resultFunction[0] = [](std::vector<std::vector<float>>& in, float out[])
                {
                    for(int row = 0; row < matrixSize; row++)
                        for(int col = 0; col < matrixSize ; col++)
                    {
                        float sum = 0;
                        for(int i = 0; i < matrixSize; i++)
                        {
                            sum += in[0][ROW_COL2ELM(row,i)] * in[1][ROW_COL2ELM(i,col)];
                        }
                        if(sum != out[ROW_COL2ELM(row, col)]) return true;
                    }
                    return false;
                };
    }
    //Graph1
    {
        auto& graph1 = data.graph_arr[1];
        graph1->set_name("gemm2");

        const int vecSize = data.IN_SIZE / 2;
        unsigned sizesIn1[SYN_MAX_TENSOR_DIM] = {vecSize,     0, 0, 0, 0};
        unsigned sizesIn2[SYN_MAX_TENSOR_DIM] = {vecSize,     0, 0, 0, 0};
        unsigned sizesOut[SYN_MAX_TENSOR_DIM] = {vecSize * 2, 0, 0, 0, 0};

        graph1->addTensor(1U, syn_type_single, sizesIn1, true /*isPersist*/, "in1_1");
        graph1->addTensor(1U, syn_type_single, sizesIn2, true /*isPersist*/, "in1_2");
        graph1->addTensor(1U, syn_type_single, sizesOut, true /*isPersist*/, "out1");

        // DMA node
        const char* concatNodeGuid   = "concat";
        char        concatNodeName[] = "concat_node";

        synTensor inputs2[]   = {graph1->getTensorHandle("in1_1"), graph1->getTensorHandle("in1_2")};
        synTensor outputs2[]  = {graph1->getTensorHandle("out1")};
        uint32_t object1Data = 0;
        synStatus status      = synNodeCreate(graph1->get(),
                                         inputs2,
                                         outputs2,
                                         2,
                                         1,
                                         &object1Data,
                                         4,
                                         concatNodeGuid,
                                         concatNodeName,
                                         nullptr,
                                         nullptr);
        ASSERT_EQ(status, synSuccess) << "Failed to create Add Node";
        graph1->compile(deviceId);

        //Function to check the results
        data.resultFunction[1] = [=](std::vector<std::vector<float>>& in, float out[]) {
            for (int i = 0; i < vecSize; i++)
            {
                float expected0 = in[0][i];
                float expected1 = in[1][i];
                float actual0   = out[i];
                float actual1   = out[vecSize + i];
                if ((expected0 != actual0) || (expected1 != actual1)) return true;
            }
            return false;
        };
    }
}

TEST_F_GC(SynGaudiThreads, multi_threaded_multi_graph, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    tfSimpleFlowData data(m_deviceType);

    //Allocate memory on device once. I assume each graph has 2 inputs, both same size, both floats
    //graph/input = run0: in0, in1, out; run1: in0, in1, out; and so on
    //I also assume that the output max size of the input*2 (for concatenation)
    synStatus status = synDeviceMalloc(deviceId, data.DEV_MEM_SIZE, 0x0, 0, &data.devMem);
    ASSERT_EQ(synSuccess, status) << "Failed to allocate device memory";

    //populate vectors in memory
    size_t inByteSize       = data.IN_SIZE  * sizeof(float);
    size_t outByteSize      = inByteSize * 2;

    size_t totalInByteSize  = inByteSize  * data.NUM_INPUTS;
    size_t totalOutByteSize = outByteSize * data.NUM_OUTPUTS;
    size_t totalSizePerRun  = totalInByteSize + totalOutByteSize;


    for(int run = 0; run < data.NUM_RUNS; run++)
    {
        auto& singleRun = data.runArr[run];
        for(int in = 0; in < data.NUM_INPUTS; in++)
        {
            singleRun.inDevAddrVec[in] = data.devMem + (run * totalSizePerRun + in) * inByteSize;
        }
        singleRun.outDevAddr      = data.devMem + (run * totalSizePerRun + data.NUM_INPUTS) * inByteSize;
    }

    create_graphs(data);

    //Run threads to copy the data to device
    std::vector<std::thread> threads;
    for(int i = 0; i < data.NUM_INPUTS; i++)
    {
        threads.push_back(std::thread(host2dev_thread, std::ref(data), i));
    }
    //Start threads to run graphs
    for(int i = 0; i < data.NUM_LAUNCH_THREADS; i++)
    {
        threads.push_back(std::thread(compute_thread, std::ref(data), i));
    }
    //Start threads to get results
    for(int i = 0; i < data.NUM_RES_THREADS; i++)
    {
        threads.push_back(std::thread(result_thread, std::ref(data), i));
    }

    for(auto& thread : threads)
    {
        thread.join();
    }

    status = synDeviceFree(deviceId, data.devMem, 0);
    ASSERT_EQ(synSuccess, status) << "Failed to free device memory";
}
