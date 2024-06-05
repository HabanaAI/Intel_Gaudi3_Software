#include "gaudi2_types.h"
#include "node_factory.h"
#include "rotator_utils.h"

#include "../gaudi_tests/gc_gaudi_test_infra.h"

#include "platform/gaudi2/graph_compiler/passes/generate_mme_descriptors.h"

#include "runtime/common/syn_singleton.hpp"
#include "synapse_common_types.h"

#include <stdlib.h>

static constexpr int backgroundPixel = 22;

class SynGaudi2TestRotate : public SynGaudiTestInfra
{
public:
    void run_test(int                      batch_size,
                  int                      channel_size,
                  float                    rotationAngle,
                  std::vector<std::string> inputFileName,
                  int                      output_height,
                  int                      output_width,
                  int                      inputCenterOffsetX,
                  int                      inputCenterOffsetY,
                  int                      outputCenterOffsetX,
                  int                      outputCenterOffsetY,
                  const char*              referenceFileName,
                  uint8_t                  background = 0,
                  unsigned                 crd_mode   = 0);
    void run_test_base(int                      batch_size,
                       int                      channel_size,
                       float                    rotationAngle,
                       int                      input_height,
                       int                      input_width,
                       int                      output_height,
                       int                      output_width,
                       int                      inputCenterOffsetX,
                       int                      inputCenterOffsetY,
                       int                      outputCenterOffsetX,
                       int                      outputCenterOffsetY,
                       unsigned char*           inputBuffer,
                       unsigned char*           expectedOutput,
                       std::vector<std::string> inputFileName,
                       const char*              referenceFileName,
                       uint8_t                  background = 0,
                       unsigned                 crd_mode   = 0);

    SynGaudi2TestRotate()
    {
        if (m_deviceType == synDeviceTypeInvalid)
        {
            LOG_WARN(SYN_TEST,
                     "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi2");
            m_deviceType = synDeviceGaudi2;
        }
        setSupportedDevices({synDeviceGaudi2});

        const char* env = getenv("SOFTWARE_LFS_DATA");
        if (env)
        {
            m_referencePrefix = std::string(env) + "/synapse/tests/gc_rotator_tests/gaudi2/";
        }
    }

    const std::string& getReferencePrefix() const { return m_referencePrefix; }

private:
    std::string m_referencePrefix;
};

// define RUN_ROTATOR_SIM if you want to run new tests and generate reference results using the reference simulator.
// Once the reference result is validated, simply copy the bin/bmp file to
// $SOFTWARE_LFS_DATA/synapse/tests/gc_rotator_tests/gaudi2/ so it can be compared against in subsequent runs

//#define RUN_ROTATOR_SIM
#ifdef RUN_ROTATOR_SIM

static char        rotatorReferenceFileName[200] = "/tmp/gaudi2_rotate_result.bin";
static std::string ROTATOR_EXE(std::getenv("ROTATOR_DEBUG_BUILD") + std::string("/bin/rotator_sim"));
static const char* OUT_IMAGE_FILE_PREFIX = "image_";
static const char* OUT_IMAGE_DIR         = "/tmp/";

static void buildRotatorSimCmd(std::string& simCmd,
                               std::string  outFileName,
                               std::string  inputFileName,
                               int          stripeHeight,
                               int          stripeWidth,
                               int          inputCenterY,
                               int          inputCenterX,
                               int          stripeCenterY,
                               int          stripeCenterX,
                               float        rotationAngle,
                               uint8_t      background,
                               unsigned     crd_mode)

{
    // extract the input file name and directory - required for rotator standalone simulator activation
    std::string inFileName(inputFileName.substr(inputFileName.find_last_of("/") + 1));

    std::string  inFileDir("");
    const size_t last_slash_idx = inputFileName.rfind('/');
    if (std::string::npos != last_slash_idx)
    {
        inFileDir = inputFileName.substr(0, last_slash_idx) + "/";
    }

    simCmd += ROTATOR_EXE + std::string(" -fi ") + inFileName + std::string(" -di ") + inFileDir +
              std::string(" -print_images ");
    simCmd += std::string(" -do ") + OUT_IMAGE_DIR + std::string(" -fo ") + outFileName;
    simCmd += " -Ho " + std::to_string(stripeHeight);
    simCmd += " -Wo " + std::to_string(stripeWidth);
    simCmd += " -Yci " + std::to_string(inputCenterY);
    simCmd += " -Xci " + std::to_string(inputCenterX);
    simCmd += " -Yco " + std::to_string(stripeCenterY);
    simCmd += " -Xco " + std::to_string(stripeCenterX);
    simCmd += " -rot_angle " + std::to_string(rotationAngle);
    simCmd += " -bg " + std::to_string(background);
    simCmd += " -crd_mode " + std::to_string(crd_mode);
    simCmd += " -rate_mode 1";  // e_irt_rate_adaptive_2x2

    LOG_DEBUG(GC, "rotator sim command to run: {}", simCmd);
}

// Get the expected results by calling the reference sim code
static void runRotatorSim(HabanaGraph*                    graph,
                          std::shared_ptr<RotateNode>&    rotateNode,
                          const std::vector<std::string>& inputFileName,
                          int                             batch_size,
                          int                             channel_size,
                          int                             inputHeight,
                          int                             inputWidth,
                          int                             outputHeight,
                          int                             outputWidth,
                          int                             inputCenterX,
                          int                             inputCenterY,
                          int                             outputCenterX,
                          int                             outputCenterY,
                          float                           rotationAngle,
                          uint8_t                         background,
                          unsigned char*                  rotatorExpectedOutput,
                          unsigned char*                  inputBuffer,
                          unsigned                        crd_mode)
{
    // The reference code can be used at this stage for an output of one stripe only per 2d image
    // So we break the output into multiple blocks each consists of a single stripe and copy
    // the result into the proper place in the expected output.
    // In each iteration we only change the horizontal output center Xc and the output width.
    int num_horizontal_stripes = (outputWidth + 127) / 128;
    int num_vertical_stripes   = 1;
    int maxStripeHeight        = outputHeight;
    int maxStripeWidth         = 128;
    if ((rotationAngle == 90.0) || (rotationAngle == 270.0))
    {
        num_vertical_stripes = (outputHeight + 255) / 256;
        maxStripeHeight      = 256;
    }

    if (!graph->getHALReader()->isRotateAngleSupported(rotationAngle))
    {
        unsigned calculatedStripeWidth = graph->getRotateStripeWidth(rotateNode);

        num_horizontal_stripes = (outputWidth + calculatedStripeWidth - 1) / calculatedStripeWidth;
        maxStripeWidth         = calculatedStripeWidth;
    }

    int fileIndex = 0;
    for (int j = 0; j < num_vertical_stripes; j++)
    {
        int stripeHeight  = (j == (num_vertical_stripes - 1)) ? outputHeight - j * maxStripeHeight : maxStripeHeight;
        int stripeCenterY = outputCenterY - maxStripeHeight * j;
        for (int i = 0; i < num_horizontal_stripes; i++)
        {
            fileIndex++;
            int  stripeWidth = (i == (num_horizontal_stripes - 1)) ? outputWidth - i * maxStripeWidth : maxStripeWidth;
            int  stripeCenterX = outputCenterX - maxStripeWidth * i;
            int  res           = 0;
            char filename[200];
            strcpy(filename, inputFileName[0].c_str());
            char* baseName;
            (baseName = strrchr(filename, '/')) ? ++baseName : (baseName = filename);

            // build rotator sim command
            std::string outFileName(OUT_IMAGE_FILE_PREFIX + std::to_string(fileIndex) + ".bmp");
            std::string simCmd("");
            buildRotatorSimCmd(simCmd,
                               outFileName,
                               inputFileName[0],
                               stripeHeight,
                               stripeWidth,
                               inputCenterY,
                               inputCenterX,
                               stripeCenterY,
                               stripeCenterX,
                               rotationAngle,
                               background,
                               crd_mode);

            res = system(simCmd.c_str());

            ASSERT_EQ(res, 0);

            // read BMP into inputBuffer
            unsigned char* inputBufferBMP    = nullptr;
            unsigned char* inputImageDataBMP = nullptr;
            int            imageWidthBMP, imageHeightBMP;
            std::string    outFileNameFullPath(OUT_IMAGE_DIR + outFileName);

            bool res = ReadBMPplain(outFileNameFullPath, imageWidthBMP, imageHeightBMP, &inputImageDataBMP, true);
            HB_ASSERT(res, "Failed to read BMP file");
            unsigned int numInputImageElementsBMP = channel_size * imageHeightBMP * imageWidthBMP;

            // Allocate the buffer for the full input
            inputBufferBMP = new unsigned char[numInputImageElementsBMP];
            memcpy(inputBufferBMP, inputImageDataBMP, numInputImageElementsBMP);
            delete[] inputImageDataBMP;

            // Copy current stripe into the proper place in the output
            // If the batch size is larger than 1, we will duplicate the first batch onto the other batches
            for (int b = 0; b < batch_size; b++)
            {
                copyStripeToFullTensor(batch_size,
                                       channel_size,
                                       stripeHeight,
                                       stripeWidth,
                                       b,
                                       j * maxStripeHeight,
                                       i * maxStripeWidth,
                                       outputHeight,
                                       outputWidth,
                                       inputBufferBMP,
                                       rotatorExpectedOutput);
            }
        }
    }

    // save the simulator bmp output file for debug purposes
    char simOutputImageFile[40] = "/tmp/sim_out.bmp";
    WriteBMP(simOutputImageFile, outputWidth, outputHeight, rotatorExpectedOutput, true);
}

#endif

// The common code: sets the input and output tensors, create the single-node graph, run the test
// and compare the results.
// If simulation is enabled, the content of rotatorExpectedOutput is ignored and set by calling the
// simulation reference.
void SynGaudi2TestRotate::run_test_base(int                      batch_size,
                                        int                      channel_size,
                                        float                    rotationAngle,
                                        int                      input_height,
                                        int                      input_width,
                                        int                      output_height,
                                        int                      output_width,
                                        int                      inputCenterOffsetX,
                                        int                      inputCenterOffsetY,
                                        int                      outputCenterOffsetX,
                                        int                      outputCenterOffsetY,
                                        unsigned char*           inputBuffer,
                                        unsigned char*           rotatorExpectedOutput,
                                        std::vector<std::string> inputFileName,
                                        const char*              referenceFileName,
                                        uint8_t                  background,
                                        unsigned                 crd_mode)
{
    int      outputTensorElements = batch_size * channel_size * output_height * output_width;
    unsigned inputCenterX         = input_width / 2 + inputCenterOffsetX;
    unsigned inputCenterY         = input_height / 2 + inputCenterOffsetY;
    unsigned outputCenterX        = output_width / 2 + outputCenterOffsetX;
    unsigned outputCenterY        = output_height / 2 + outputCenterOffsetY;

    synRotateParams params(rotationAngle, inputCenterX, inputCenterY, outputCenterX, outputCenterY, background);

    unsigned inputDims[4]     = {input_width, input_height, channel_size, batch_size};
    unsigned inputTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                    MEM_INIT_FROM_INITIALIZER,
                                                    (float*)inputBuffer,
                                                    inputDims,
                                                    4,
                                                    syn_type_uint8,
                                                    nullptr,
                                                    "inputData");
    memset(rotatorExpectedOutput, 44, outputTensorElements);
    unsigned outputDims[4]     = {output_width, output_height, channel_size, batch_size};
    unsigned outputTensorIndex = createPersistTensor(OUTPUT_TENSOR,
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     outputDims,
                                                     4,
                                                     syn_type_uint8,
                                                     nullptr,
                                                     "outputData");

    synNodeId nodeId;
    addNodeToGraph(NodeFactory::rotateNodeTypeName,
                   {inputTensorIndex},
                   {outputTensorIndex},
                   &params,
                   sizeof(params),
                   "image_rotate_node",
                   0,
                   &nodeId);

    HabanaGraph*                graph = synSingleton::getInstanceInternal()->getGraph(m_graphs.front().graphHandle);
    std::shared_ptr<RotateNode> rotateNode = std::dynamic_pointer_cast<RotateNode>(graph->getNodeByID(nodeId));

    rotateNode->setCoordinateMode(crd_mode);
    rotateNode->setInputLayouts({gc::Layout("NCHW")});
    rotateNode->setOutputLayouts({gc::Layout("NCHW")});

#ifdef RUN_ROTATOR_SIM
    runRotatorSim(graph,
                  rotateNode,
                  inputFileName,
                  batch_size,
                  channel_size,
                  input_height,
                  input_width,
                  output_height,
                  output_width,
                  inputCenterX,
                  inputCenterY,
                  outputCenterX,
                  outputCenterY,
                  rotationAngle,
                  background,
                  rotatorExpectedOutput,
                  inputBuffer,
                  crd_mode);

    writeBufferToFile(referenceFileName, rotatorExpectedOutput, outputTensorElements);
#else
    readBufferFromFile(referenceFileName, rotatorExpectedOutput, outputTensorElements);
#endif

    compileAndRun();

    float* outputData = (float*)m_hostBuffers[outputTensorIndex];

    // Set to 1 in order to create a new bmp & bin file for latest test.
    // Otherwise, we just compare the buffers in-memory
#if 0
    // write the actual output buffer into output BMP
    char actualOutFileName[40] = "/tmp/image_actual_out.bmp";
    WriteBMP(actualOutFileName, output_width, output_height, (unsigned char*)outputData, true);

    // save the actual output buffer as binary
    char outDataFileName[40] = "/tmp/image_actual_out.bin";
    writeBufferToFile(outDataFileName, (unsigned char *)outputData, outputTensorElements);
#endif

    int numErrors = compareResults(batch_size,
                                   channel_size,
                                   output_height,
                                   output_width,
                                   rotatorExpectedOutput,
                                   (unsigned char*)outputData);
    if (numErrors > 0)
    {
        std::cout << "Total " << numErrors << " mismatches out of " << outputTensorElements
                  << " elements in output tensor" << std::endl;
    }
    ASSERT_EQ(numErrors, 0);
}
void SynGaudi2TestRotate::run_test(int                      batch_size,
                                   int                      channel_size,
                                   float                    rotationAngle,
                                   std::vector<std::string> inputFileName,
                                   int                      output_height,
                                   int                      output_width,
                                   int                      inputCenterOffsetX,
                                   int                      inputCenterOffsetY,
                                   int                      outputCenterOffsetX,
                                   int                      outputCenterOffsetY,
                                   const char*              referenceFileName,
                                   uint8_t                  background,
                                   unsigned                 crd_mode)
{
    int inputWidth = 0, inputHeight = 0;  // Derived from the input bmp
    int numImages = inputFileName.size();
    // Read the input image
    unsigned char* inputBuffer = nullptr;
    int            imageIdx;
    for (imageIdx = 0; imageIdx < numImages; imageIdx++)
    {
        unsigned char* inputImageData = nullptr;
        int            imageWidth, imageHeight;
        bool           res = ReadBMPplain(inputFileName[imageIdx], imageWidth, imageHeight, &inputImageData, true);
        HB_ASSERT(res, "Failed to read BMP file");
        unsigned int numInputImageElements = channel_size * imageHeight * imageWidth;
        if (imageIdx == 0)
        {
            inputWidth  = imageWidth;
            inputHeight = imageHeight;
            // Allocate the buffer for the full input
            inputBuffer = new unsigned char[batch_size * numInputImageElements];
        }
        else
        {  // Sanity check: verify that all images are of the same size
            if ((imageWidth != inputWidth) || (imageHeight != inputHeight))
            {
                HB_ASSERT(false, "Error! Not all input bmp images are of the same dimensions");
            }
        }
        memcpy(inputBuffer + imageIdx * numInputImageElements, inputImageData, numInputImageElements);
        delete[] inputImageData;
    }
    // Duplicate the 3-channel available input images onto rest of batch dims
    unsigned int numInputImageElements = channel_size * inputHeight * inputWidth;
    for (; imageIdx < batch_size; imageIdx++)
    {
        memcpy(inputBuffer + imageIdx * numInputImageElements,
               inputBuffer + (imageIdx % numImages) * numInputImageElements,
               numInputImageElements);
    }
    unsigned char* rotatorExpectedOutput = new unsigned char[batch_size * channel_size * output_height * output_width];
    run_test_base(batch_size,
                  channel_size,
                  rotationAngle,
                  inputHeight,
                  inputWidth,
                  output_height,
                  output_width,
                  inputCenterOffsetX,
                  inputCenterOffsetY,
                  outputCenterOffsetX,
                  outputCenterOffsetY,
                  inputBuffer,
                  rotatorExpectedOutput,
                  inputFileName,
                  referenceFileName,
                  background,
                  crd_mode);
    delete[] inputBuffer;
    delete[] rotatorExpectedOutput;
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_30_angle, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 30.0;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_positive_30_angle_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_positive_angle_32_27_crd_mode_0, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 32.27;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + ""
                                                           "gaudi2_rotate_positive_32_27_angle_crd_mode_0_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 0;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}
TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_positive_angle_45, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 45.0;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_positive_45_angle_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_0_angle, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 0.0;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_0_angle_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_positive_angle_small_output, {synDeviceGaudi2})
{
    // Setup the test parameters

    float                    rotationAngle = 32.05;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_positive_angle_small_output_result.bin";
#endif

    int      output_height = 420, output_width = 350;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_hd_image, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 22.05;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "nature1920x1200.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_hd_image_result.bin";
#endif

    int      output_height = 1200, output_width = 1920;
    int      inputCenterOffsetX  = 1;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 3;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_positive_angle_large_output, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 32.05;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_positive_angle_large_output_result.bin";
#endif

    int      output_height = 800, output_width = 620;
    int      inputCenterOffsetX  = 5;
    int      inputCenterOffsetY  = -2;
    int      outputCenterOffsetX = 3;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_multi_batch, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 45.0;
    int                      batch_size = 4, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_multi_batch_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_negative_small_angle, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 352.05;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_negative_small_angle_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -10;
    int      inputCenterOffsetY  = -3;
    int      outputCenterOffsetX = -5;
    int      outputCenterOffsetY = -12;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_negative_big_angle, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 308.05;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_negative_big_angle_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_negative_single_stripe_batch16, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 190.0;
    int                      batch_size = 16, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + ""
                                                           "gaudi2_rotate_negative_single_stripe_batch16_result.bin";
#endif

    int      output_height = 280, output_width = 128;
    int      inputCenterOffsetX  = 5;
    int      inputCenterOffsetY  = -2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_small_output_batch32, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 128.0;
    int                      batch_size = 32, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_small_output_batch32_result.bin";
#endif

    int      output_height = 150, output_width = 232;
    int      inputCenterOffsetX  = 4;
    int      inputCenterOffsetY  = -6;
    int      outputCenterOffsetX = 3;
    int      outputCenterOffsetY = 2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_less_than_180_angle, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 161.52;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_less_than_180_angle_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = 0;
    int      inputCenterOffsetY  = 9;
    int      outputCenterOffsetX = 13;
    int      outputCenterOffsetY = 17;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_180_angle, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 180.0;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_180_angle_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_180_angle_padding, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 180.0;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_180_angle_padding_result.bin";
#endif

    int      output_height = 690, output_width = 550;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}
TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_larger_than_180_angle, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 212.35;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_larger_than_180_angle_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 1;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_90deg, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 90.0;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_90deg_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -25;
    int      inputCenterOffsetY  = -18;
    int      outputCenterOffsetX = 15;
    int      outputCenterOffsetY = 18;
    unsigned crd_mode            = 0;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_90deg_single_roi, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 90.0;
    int                      batch_size = 4, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_90deg_single_roi_result.bin";
#endif

    int      output_height = 240, output_width = 120;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 0;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_90deg_single_stripe, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 90.0;
    int                      batch_size = 4, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_90deg_single_stripe_result.bin";
#endif

    int      output_height = 667, output_width = 120;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 0;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_270deg, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 270.0;
    int                      batch_size = 4, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_270deg_result.bin";
#endif

    int      output_height = 120, output_width = 128;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 0;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_angle_in_1st_q, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 72.0;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_angle_in_1st_q_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 0;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_angle_in_2nd_q, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 112.5;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_angle_in_2nd_q_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 0;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_angle_in_3rd_q, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 241.2;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_angle_in_3rd_q_result.bin";
#endif

    std::string descFilePrefix = "rotate_pos_angle_desc";
    int         output_height = 667, output_width = 500;
    int         inputCenterOffsetX  = -5;
    int         inputCenterOffsetY  = 2;
    int         outputCenterOffsetX = 5;
    int         outputCenterOffsetY = -2;
    unsigned    crd_mode            = 0;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_angle_in_4th_q, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 285.0;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_angle_in_4th_q_result.bin";
#endif

    int      output_height = 667, output_width = 500;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 0;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_angle_in_4th_q_hd_images, {synDeviceGaudi2})
{
    // Setup the test parameters
    float rotationAngle = 285.0;
    int   batch_size = 2, channel_size = 3;

    std::vector<std::string> inputFileName = {getReferencePrefix() + "nature1920x1200.bmp",
                                              getReferencePrefix() + "nature1920x1200.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + "gaudi2_rotate_angle_in_4th_q_hd_images_result.bin";
#endif

    int      output_height = 1200, output_width = 1920;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 0;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}

TEST_F_GC(SynGaudi2TestRotate, gaudi2_rotate_unsupported_angle_single_roi, {synDeviceGaudi2})
{
    // Setup the test parameters
    float                    rotationAngle = 241.2;
    int                      batch_size = 1, channel_size = 3;
    std::vector<std::string> inputFileName = {getReferencePrefix() + "input_image_500_667.bmp"};

#ifdef RUN_ROTATOR_SIM
    char* referenceFileName = rotatorReferenceFileName;
#else
    std::string referenceFileName = getReferencePrefix() + ""
                                                           "gaudi2_rotate_unsupported_angle_single_roi_result.bin";
#endif

    int      output_height = 667, output_width = 128;
    int      inputCenterOffsetX  = -5;
    int      inputCenterOffsetY  = 2;
    int      outputCenterOffsetX = 5;
    int      outputCenterOffsetY = -2;
    unsigned crd_mode            = 0;

    run_test(batch_size,
             channel_size,
             rotationAngle,
             inputFileName,
             output_height,
             output_width,
             inputCenterOffsetX,
             inputCenterOffsetY,
             outputCenterOffsetX,
             outputCenterOffsetY,
             referenceFileName.c_str(),
             backgroundPixel,
             crd_mode);
}
