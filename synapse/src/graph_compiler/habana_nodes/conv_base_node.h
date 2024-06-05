#pragma once

#include "mme_node.h"
#include "node_visitor.h"
#include "graph_compiler/sif/shape_inference_metadata.h"

enum eConvSpatialIndex
{
    CONV_WIDTH_INDEX = 0,
    CONV_HEIGHT_INDEX,
    CONV_DEPTH_INDEX,
    CONV_MAX_SPATIAL_INDEX
};

struct ConvParamsIndices
{
    eConvSpatialIndex spatialIndex;
    unsigned paddingBeforeIndex;
    unsigned paddingAfterIndex;
};

// Store data of the slice position in the original tensor, which was sliced by Gaudi SRAM bundle slicer
struct ROIPosition
{
    ROIPosition(std::array<bool, SYN_MAX_TENSOR_DIM> posFirst, std::array<bool, SYN_MAX_TENSOR_DIM> posLast) :
    isFirst(posFirst), isLast(posLast) {}

    std::array<bool, SYN_MAX_TENSOR_DIM> isFirst;
    std::array<bool, SYN_MAX_TENSOR_DIM> isLast;
};

class ConvBaseNode: public MmeNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef MmeNode BaseClass;

    ConvBaseNode(const TensorVector& inputs,
                 const TensorVector& outputs,
                 std::string_view    name,
                 Node::eNodeType     type,
                 ShapeFuncID         sifId = SHAPE_FUNC_MAX_ID);

    virtual NodePtr clone() const override = 0;

    virtual bool equalTo(const Node& o) const override;

    virtual void print() const override;
    virtual std::string getNodeParametersStr() const override;

    virtual void printParamsRawData() const override;

    virtual bool validateNode() const override;
    virtual bool validateNodeLayout() const override;

    // Return the maximal shape of the X operand, corresponding to the given Y operand
    virtual TensorShape getXOperandShape(const TensorShape& yOperandShape) const;
    // Return the maximal shape of the Y operand, corresponding to the given X operand
    virtual TensorShape getYOperandShape(const TensorShape& xOperandShape) const;
    // Return the overlap between 2 adjucent ROIs of the input operand
    virtual int getInputROIOverlapForDim(uint32_t inputIdx, unsigned dim) const;
    // Return the padding of X operand ROI, which is required for MME operation with the corresponding Y operand shape.
    // The ROI must start in a stride-aligned offset.
    void getXStrideAlignedROIPaddingForDim(unsigned dim, unsigned xRoiSize, int& padBefore, int& padAfter) const;

    virtual bool isOperandTransposed(const TensorPtr& t) const override { return false; }

    const synConvolution3DParamsV2& getConvolutionParams() const { return m_params; }
    synConvolution3DParamsV2&       getConvolutionParams() { return m_params; }

    const synConvolution3DParamsV2& getOriginalConvolutionParams() const { return m_originalParams; }

    virtual bool isConvolution() const override { return true; }
    virtual bool is3DConvolution() const override;
    virtual bool is3DConvolutionGuid() const = 0;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

    bool canBeConvertedToGEMM() const override;

    virtual TensorPtr getXOperand() const = 0;
    virtual TensorPtr getYOperand() const = 0;
    virtual TensorPtr getWOperand() const = 0;
    virtual TensorPtr getShapeOperand() const { return nullptr; };

    unsigned getDimActualKernelSize(unsigned dim) const;

    void updateConvForXOperandROI(ROIPosition& roiPos, const OffsetArray& padBefore, const OffsetArray& padAfter);

    // returns W/H/D according to convolution 2D/3D
    DimVector getSpatialDims() const;

    // convert tensor dimension index to convolution parameters indices
    static ConvParamsIndices dimIndexToConvParamsIndices(unsigned tensorDim);
    // convert an output coordinate in place to its corresponding input coordinate in a conv operation
    static void convertToInputCoord(int*                            coord,
                                    const synConvolution3DParamsV2& convParams,
                                    int                             kernelIndex[MAX_CONV_DIMS],
                                    unsigned int                    numDims);
    static int  convertToInputCoordForDim(int                             coord,
                                          const synConvolution3DParamsV2& convParams,
                                          int                             kernelIndex,
                                          unsigned int                    tensorDim);
    // convert an input coordinate in place to its corresponding output coordinate in a conv operation
    static void convertToOutputCoord(int*                            coord,
                                     const synConvolution3DParamsV2& convParams,
                                     int                             kernelIndex[MAX_CONV_DIMS],
                                     unsigned int                    numDims,
                                     bool                            ceil);
    static int  convertToOutputCoordForDim(int                             coord,
                                           const synConvolution3DParamsV2& convParams,
                                           int                             kernelIndex,
                                           unsigned int                    tensorDim,
                                           bool                            ceil,
                                           bool*                           pIsIntOutputCoord = nullptr);

    virtual bool isDynamicPaddingConvolution() const override;
    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

    virtual bool  isSpatialSlicingSupported(unsigned dim) const;
    virtual TSize getMinSpatialDimOutputROI(unsigned dim) const = 0;

protected:
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t getShapeInferenceFunctionUserParamsSize() const override;
    int getXOverlapForDim(unsigned dim) const;
    int getYOverlapForDim(unsigned dim) const;
    bool getMinValidYCoordForDim(unsigned dim, int xCoord, int& yCoord, int& validKernelIndex, int& dilOffset) const;
    bool getMaxValidYCoordForDim(unsigned dim, int xCoord, int& yCoord, int& validKernelIndex, int& dilOffset) const;
    void updateDimPaddingByPosition(unsigned dim, const ROIPosition& pos, const OffsetArray& padBefore, const OffsetArray& padAfter);
    synConvolution3DParamsV2                 convertUserParams(UserParams userParams, size_t userParamsSize) const;
    gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override;

    synConvolution3DParamsV2 m_params;
    SifConvolutionMetadata m_sifMetadata;

    // m_params can be change during graph optimization
    // Those params stay constant from construction
    synConvolution3DParamsV2 m_originalParams;
};
