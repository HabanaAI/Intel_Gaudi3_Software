#include "gaudi2_types.h"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "syn_logging.h"
#include "descriptor_generator.h"
#include "habana_nodes.h"
#include "tensor.h"
#include "utils.h"
#include "defs.h"
#include "queue_command.h"
#include "habana_global_conf.h"
#include "../../../../func-sim/agents/irt/IRTsim6/src/IRT.h"
#include "../../../../func-sim/agents/irt/IRTsim6/src/IRTutils.h"
#include "block_data.h"

uint8_t*     ext_mem;
uint16_t***  input_image;
uint16_t**** output_image;

using namespace gaudi2;

//=======  Aux functions ========================
static uint32_t low(uint64_t v)
{
    return (uint32_t)(v & 0x00000000FFFFFFFF);
}

static uint32_t high(uint64_t v)
{
    return (uint32_t)((v & 0xFFFFFFFF00000000) >> 32);
}

static uint64_t getImageOffset(const uint64_t* strides, const int* coord, bool imageLevel)
{
    // If imageLevel is true, we should ignore x and y coordinates
    uint64_t stripe_offset = coord[NCHW_C_DIM] * strides[NCHW_C_DIM] + coord[NCHW_N_DIM] * strides[NCHW_N_DIM];
    if (!imageLevel)
    {
        HB_ASSERT(strides[NCHW_W_DIM] == 1, "unexpected strided on FCD!");
        stripe_offset += coord[NCHW_W_DIM] * strides[NCHW_W_DIM] + coord[NCHW_H_DIM] * strides[NCHW_H_DIM];
    }
    return stripe_offset;
}

// This function calculates the max stripe width for "unsupported" angles
unsigned DescriptorGenerator::getRotateStripeWidth(std::shared_ptr<RotateNode>& rotateNode)
{
    irt_desc_par irt_desc;
    irt_cfg_pars irt_cfg;
    rotation_par rot_pars;

    memset(&irt_desc, 0, sizeof(irt_desc_par));
    memset(&irt_cfg, 0, sizeof(irt_cfg_pars));
    irt_cfg_init(irt_cfg);
    memset(&rot_pars, 0, sizeof(rotation_par));
    irt_par_init(rot_pars);

    irt_desc.irt_mode                    = e_irt_rotation;
    rot_pars.irt_angles[e_irt_angle_rot] = rotateNode->getRotationAngle();
    irt_desc.image_par[OIMAGE].S         = 128;  // initial value (max)
    irt_desc.image_par[OIMAGE].W         = rotateNode->getOutput(0)->getSizeInBytes(0);
    irt_desc.image_par[OIMAGE].H         = rotateNode->getOutput(0)->getSizeInBytes(1);

    // pixel size: 0 - 1 byte, 1 - 2 byte, 2 - 4 bytes, 3 - 8 bytes
    unsigned getInputPixelWidth   = rotateNode->getInputPixelWidth();
    irt_desc.image_par[IIMAGE].Ps = getInputPixelWidth >> 4;

    unsigned stripeWidth = irt_out_stripe_calc(irt_cfg, rot_pars, irt_desc, 0);

    LOG_DEBUG(
        GC,
        "Angle: {}, Out image width: {}, Out image height: {}, In image pixel width: {}, calculated stripe width: {}",
        rot_pars.irt_angles[e_irt_angle_rot],
        irt_desc.image_par[OIMAGE].W,
        irt_desc.image_par[OIMAGE].H,
        irt_desc.image_par[IIMAGE].Ps,
        stripeWidth);

    return stripeWidth;
}

// This function targets overriding descriptor fields that cannot be set by the generic code
static void overrideDescParams(RotatorDesc& desc, int outputImageWidth, uint64_t inputLastAddress)
{
    desc.out_stride.val = outputImageWidth;
}

static void SetImageParams(uint64_t      inputImageAddr,
                           uint32_t      inputImageWidth,
                           uint32_t      inputImageHeight,
                           uint32_t      inputCenterX,
                           uint32_t      inputCenterY,
                           uint64_t      inputLastAddress,
                           uint64_t      outputStripeAddr,
                           uint32_t      outputStripeWidth,
                           uint32_t      outputStripeHeight,
                           uint32_t      outputCenterX,
                           uint32_t      outputCenterY,
                           double        rotationAngle,
                           int           super_irt,
                           irt_desc_par& irt_desc)
{
    irt_desc.image_par[OIMAGE].addr_start = outputStripeAddr;
    irt_desc.image_par[OIMAGE].H          = outputStripeHeight;
    irt_desc.image_par[OIMAGE].W          = outputStripeWidth;
    irt_desc.image_par[OIMAGE].Yc         = outputCenterY << 1;
    irt_desc.image_par[OIMAGE].Xc         = outputCenterX << 1;
    irt_desc.image_par[OIMAGE].S          = irt_desc.image_par[OIMAGE].W;
    int So8                               = irt_desc.image_par[OIMAGE].S;
    if (super_irt == 0)
    {
        if ((irt_desc.image_par[OIMAGE].S % 8) != 0)  // not multiple of 8, round up to multiple of 8
        {
            So8 += (8 - (irt_desc.image_par[OIMAGE].S % 8));
        }
    }
    irt_desc.image_par[IIMAGE].addr_start = inputImageAddr;
    irt_desc.image_par[IIMAGE].addr_end   = inputLastAddress;
    irt_desc.image_par[IIMAGE].H          = inputImageHeight;
    irt_desc.image_par[IIMAGE].W          = inputImageWidth;
    irt_desc.image_par[IIMAGE].Yc         = inputCenterY << 1;
    irt_desc.image_par[IIMAGE].Xc         = inputCenterX << 1;
    irt_desc.image_par[IIMAGE].S          = (rotationAngle == 90 || rotationAngle == -90)
                                                ? 256
                                                : abs((long)ceil(So8 / cos((double)rotationAngle * M_PI / 180.0))) + 1;
}

//============ Move the content to the actual descriptor ======================
static void setDescriptorParams(const irt_desc_par& irt_desc,
                                const rotation_par& rot_par,
                                const irt_cfg_pars& irt_cfg,
                                int                 context_id,
                                RotatorDesc&        desc)
{
    desc.context_id.val               = context_id;
    desc.in_img_start_addr_l.val      = low(irt_desc.image_par[IIMAGE].addr_start);
    desc.in_img_start_addr_h.val      = high(irt_desc.image_par[IIMAGE].addr_start);
    desc.out_img_start_addr_l.val     = low(irt_desc.image_par[OIMAGE].addr_start);
    desc.out_img_start_addr_h.val     = high(irt_desc.image_par[OIMAGE].addr_start);
    desc.cfg.rot_dir                  = irt_desc.rot_dir;
    desc.cfg.read_flip                = irt_desc.read_hflip;
    desc.cfg.rot_90                   = irt_desc.rot90;
    desc.cfg.out_image_line_wr_format = irt_desc.oimage_line_wr_format;
    desc.cfg.v_read_flip              = irt_desc.read_vflip;
    desc.im_read_slope.val            = irt_desc.im_read_slope;
    if (irt_desc.crd_mode == e_irt_crd_mode_fp32)
    {
        desc.sin_d.val = *(uint32_t*)&irt_desc.sinf;
        desc.cos_d.val = *(uint32_t*)&irt_desc.cosf;
    }
    else
    {
        desc.sin_d.val = irt_desc.sini;
        desc.cos_d.val = irt_desc.cosi;
    }
    desc.in_img.width                  = irt_desc.image_par[IIMAGE].W;
    desc.in_img.height                 = irt_desc.image_par[IIMAGE].H;
    desc.in_stride.val                 = irt_desc.image_par[IIMAGE].Hs;
    desc.in_stripe.width               = irt_desc.image_par[IIMAGE].S;
    desc.in_center.x                   = irt_desc.image_par[IIMAGE].Xc;
    desc.in_center.y                   = irt_desc.image_par[IIMAGE].Yc;
    desc.out_img.height                = irt_desc.image_par[OIMAGE].H;
    desc.out_img._reserved16           = irt_desc.image_par[OIMAGE].W;
    desc.out_stride.val                = irt_desc.image_par[OIMAGE].Hs;
    desc.out_stripe.width              = irt_desc.image_par[OIMAGE].W;
    desc.out_center.x                  = irt_desc.image_par[OIMAGE].Xc;
    desc.out_center.y                  = irt_desc.image_par[OIMAGE].Yc;
    desc.background.pxl_val            = irt_desc.bg;
    desc.idle_state.start_en           = 0;
    desc.idle_state.end_en             = 0;
    desc.x_i_start_offset.val          = irt_desc.Xi_start_offset;
    desc.x_i_start_offset_flip.val     = irt_desc.Xi_start_offset_flip;
    desc.x_i_first.val                 = irt_desc.Xi_first_fixed;
    desc.y_i_first.val                 = irt_desc.Yi_first_fixed;
    desc.y_i.end_val                   = irt_desc.Yi_end;
    desc.y_i.start_val                 = irt_desc.Yi_start;
    desc.out_stripe_size.val           = irt_desc.image_par[OIMAGE].Size;
    desc.rsb_cfg_0.cache_inv           = 1;
    desc.rsb_cfg_0.uncacheable         = 0;
    desc.rsb_cfg_0.perf_evt_start      = 1;
    desc.rsb_cfg_0.perf_evt_end        = 1;
    desc.rsb_cfg_0.pad_duplicate       = 0;
    desc.rsb_pad_val.val               = 0;
    desc.owm_cfg.perf_evt_start        = 1;
    desc.owm_cfg.perf_evt_end          = 1;
    desc.ctrl_cfg.bg_mode              = irt_desc.bg_mode;
    desc.ctrl_cfg.irt_mode             = irt_desc.irt_mode;
    desc.ctrl_cfg.img_pwi              = irt_desc.image_par[IIMAGE].Ps;
    desc.ctrl_cfg.img_pwo              = irt_desc.image_par[OIMAGE].Ps;
    desc.ctrl_cfg.buf_cfg_mode         = irt_cfg.buf_mode[0];
    desc.ctrl_cfg.buf_fmt_mode         = irt_cfg.buf_format[0];
    desc.ctrl_cfg.rate_mode            = irt_desc.rate_mode;
    desc.ctrl_cfg.proc_size            = irt_desc.proc_size;
    desc.ctrl_cfg.coord_calc_data_type = irt_desc.crd_mode;
    desc.pixel_pad.mask_pwi            = irt_desc.Msi;
    desc.pixel_pad.lsb_o               = irt_desc.Ppo;
    desc.prec_shift.bli                = irt_desc.bli_shift;
    desc.prec_shift.coord_align        = irt_desc.prec_align;
    desc.max_val.sat                   = irt_desc.MAX_VALo;
    if (irt_desc.irt_mode == e_irt_projection)
    {
        desc.a0_m11.val = *(uint32_t*)&irt_desc.prj_Af[0];
        desc.a1_m12.val = *(uint32_t*)&irt_desc.prj_Af[1];
        desc.a2.val     = *(uint32_t*)&irt_desc.prj_Af[2];
        desc.b0_m21.val = *(uint32_t*)&irt_desc.prj_Bf[0];
        desc.b1_m22.val = *(uint32_t*)&irt_desc.prj_Bf[1];
        desc.b2.val     = *(uint32_t*)&irt_desc.prj_Bf[2];
        desc.c0.val     = *(uint32_t*)&irt_desc.prj_Cf[0];
        desc.c1.val     = *(uint32_t*)&irt_desc.prj_Cf[1];
        desc.c2.val     = *(uint32_t*)&irt_desc.prj_Cf[2];
        desc.d0.val     = *(uint32_t*)&irt_desc.prj_Df[0];
        desc.d1.val     = *(uint32_t*)&irt_desc.prj_Df[1];
        desc.d2.val     = *(uint32_t*)&irt_desc.prj_Df[2];
    }
    else if (irt_desc.irt_mode == e_irt_affine)
    {
        if (irt_desc.crd_mode == e_irt_crd_mode_fp32)
        {
            desc.a0_m11.val = *(uint32_t*)&irt_desc.M11f;
            desc.a1_m12.val = *(uint32_t*)&irt_desc.M12f;
            desc.b0_m21.val = *(uint32_t*)&irt_desc.M21f;
            desc.b1_m22.val = *(uint32_t*)&irt_desc.M22f;
        }
        else
        {
            desc.a0_m11.val = *(uint32_t*)&irt_desc.M11i;
            desc.a1_m12.val = *(uint32_t*)&irt_desc.M12i;
            desc.b0_m21.val = *(uint32_t*)&irt_desc.M21i;
            desc.b1_m22.val = *(uint32_t*)&irt_desc.M22i;
        }
    }
    float proc_size                = 1.0 / (desc.ctrl_cfg.proc_size - 1);
    desc.inv_proc_size_m_1.val     = *(uint32_t*)&proc_size;
    desc.mesh_img_start_addr_l.val = low(irt_desc.image_par[MIMAGE].addr_start);
    desc.mesh_img_start_addr_h.val = high(irt_desc.image_par[MIMAGE].addr_start);
    desc.mesh_img.height           = irt_desc.image_par[MIMAGE].H;
    desc.mesh_stride.val           = irt_desc.image_par[MIMAGE].Hs;
    desc.mesh_stripe.width         = irt_desc.image_par[MIMAGE].S;
    desc.mesh_ctrl.data_type       = irt_desc.image_par[MIMAGE].Ps - 2;
    desc.mesh_ctrl.rel_mode        = irt_desc.mesh_rel_mode;
    desc.mesh_ctrl.sparse_mode     = (uint32_t)irt_desc.mesh_sparse_h | ((uint32_t)irt_desc.mesh_sparse_v << 1);
    desc.mesh_ctrl.fxd_pt_frac_w   = irt_desc.mesh_point_location;
    desc.mesh_gh.val               = irt_desc.mesh_Gh;
    desc.mesh_gv.val               = irt_desc.mesh_Gv;
    desc.mrsb_cfg_0.cache_inv      = 1;
    desc.mrsb_cfg_0.uncacheable    = 0;
    desc.mrsb_cfg_0.perf_evt_start = 1;
    desc.mrsb_cfg_0.perf_evt_end   = 1;
    desc.mrsb_cfg_0.pad_duplicate  = 0;
    desc.mrsb_pad_val.val          = 0;
    desc.buf_cfg.rot_epl           = irt_cfg.rm_cfg[0][irt_cfg.buf_mode[0]].Buf_EpL;
    desc.buf_cfg.mesh_epl          = irt_cfg.rm_cfg[1][irt_cfg.buf_mode[1]].Buf_EpL;
    desc.buf_cfg.mesh_mode         = irt_desc.mesh_rel_mode;
    desc.buf_cfg.mesh_fmt          = irt_desc.mesh_format;
    desc.cid_offset.val            = 1;
    desc.push_desc.ind             = 1;
}

void DescriptorGenerator::generateRotatorDescriptors(const RotateNode&         node,
                                                     const std::list<NodeROI>& physicalRois,
                                                     uint64_t                  sramBase,
                                                     RotDescriptorsList&       descriptors)
{
    pTensor src = node.getInput(0);
    pTensor dst = node.getOutput(0);
    HB_ASSERT(src->tensorIsAllocated(), "{}: source tensor has no address, cannot rotate it", node.getNodeName());
    HB_ASSERT(dst->tensorIsAllocated(), "{}: destination tensor has no address, cannot rotate it", node.getNodeName());
    const SizeArray inputDims         = src->getAllSizesInElements();
    const SizeArray outputDims        = dst->getAllSizesInElements();
    gc::Layout      inputLayout       = node.getInputLayouts()[0];
    gc::Layout      outputLayout      = node.getOutputLayouts()[0];
    uint32_t        batchSize         = inputDims[NCHW_N_DIM];
    uint32_t        channelSize       = inputDims[NCHW_C_DIM];
    uint32_t        inputImageHeight  = inputDims[NCHW_H_DIM];
    uint32_t        inputImageWidth   = inputDims[NCHW_W_DIM];
    uint32_t        inputImageSize    = inputImageWidth * inputImageHeight;
    uint32_t        outputBatchSize   = outputDims[NCHW_N_DIM];
    uint32_t        outputChannelSize = outputDims[NCHW_C_DIM];
    uint32_t        outputImageWidth  = outputDims[NCHW_W_DIM];

    // Verify the arguments
    HB_ASSERT(inputLayout.toString() == "NCHW" && outputLayout.toString() == "NCHW",
              "Rotator supports tensors in NCHW format only");
    HB_ASSERT(batchSize == outputBatchSize && channelSize == outputChannelSize,
              "Error. Batch and channel dimensions must be identical for the rotator input and output tensors");
    unsigned inputCenterX  = node.getInputCenterX();
    unsigned inputCenterY  = node.getInputCenterY();
    unsigned outputCenterX = node.getOutputCenterX();
    unsigned outputCenterY = node.getOutputCenterY();
    uint64_t inputAddress  = src->tensorAllocatedInSram() ? src->getSramOffset() : src->getDramOffset();
    uint64_t outputAddress = dst->tensorAllocatedInSram() ? dst->getSramOffset() : dst->getDramOffset();

    uint64_t inputStrides[SYN_MAX_TENSOR_DIM + 1];
    uint64_t outputStrides[SYN_MAX_TENSOR_DIM + 1];
    src->getAllStridesInElements(inputStrides);
    dst->getAllStridesInElements(outputStrides);
    // Base descriptor, will be used for all rois
    RotatorDesc desc;
    memset(&desc, 0, sizeof(desc));
    ValidityMask<RotatorDesc> descMask {false};  // turn everything off
    SET_MASK_BULK_ON(std::begin(descMask),
                     MASK_OFFSET(RotatorDesc, context_id),
                     MASK_OFFSET(RotatorDesc, cpl_msg_addr));

    SET_MASK_BULK_ON(std::begin(descMask),
                     MASK_OFFSET(RotatorDesc, x_i_start_offset),
                     MASK_OFFSET(RotatorDesc, hbw_aruser_hi));
    SET_MASK_BULK_ON(std::begin(descMask), MASK_OFFSET(RotatorDesc, owm_cfg), MASK_OFFSET(RotatorDesc, push_desc));

    uint16_t context_id     = node.getContextId();
    int      super_irt      = 0;
    int      line_wr_format = 0;

    // init intermediate data structures
    irt_desc_par irt_desc;
    irt_cfg_pars irt_cfg;
    rotation_par rot_pars;

    memset(&irt_desc, 0, sizeof(irt_desc_par));
    memset(&irt_cfg, 0, sizeof(irt_cfg_pars));
    irt_cfg_init(irt_cfg);
    memset(&rot_pars, 0, sizeof(rotation_par));
    irt_par_init(rot_pars);

    float rot_angle                      = node.getRotationAngle();
    rot_pars.Pwi                         = node.getInputPixelWidth();
    rot_pars.Pwo                         = node.getOutputPixelWidth();
    rot_pars.irt_angles[e_irt_angle_rot] = rot_angle;

    irt_desc.oimage_line_wr_format = line_wr_format;
    irt_desc.bg                    = node.getBackgroundPixel();
    irt_desc.bg_mode               = e_irt_bg_prog_value;
    irt_desc.irt_mode              = (Eirt_tranform_type)node.getRotationMode();
    irt_desc.int_mode              = (Eirt_int_mode_type)node.getInterpolationMode();
    irt_desc.rate_mode             = e_irt_rate_adaptive_2x2;
    irt_desc.crd_mode              = (Eirt_coord_mode_type)node.getCoordinateMode();
    irt_desc.proc_size             = 8;

    for (const NodeROI& roi : physicalRois)
    {
        int roiCoordinate[Tensor::c_tensorMaxNDim];  // TODO: Move to TOffset array [SW-117362]
        castNcopy(roiCoordinate, roi.baseOffset, Tensor::c_tensorMaxNDim);

        uint64_t   stripeOffset        = getImageOffset(outputStrides, roiCoordinate, false);
        uint64_t   inputImageOffset    = getImageOffset(inputStrides, roiCoordinate, true);
        uint64_t   inputImageAddr      = inputAddress + inputImageOffset;
        uint64_t   inputLastAddress    = inputImageAddr + inputImageSize;
        uint64_t   outputStripeAddr    = outputAddress + stripeOffset;
        uint32_t   outputStripeCenterX = outputCenterX - roi.baseOffset[NCHW_W_DIM];
        uint32_t   outputStripeCenterY = outputCenterY - roi.baseOffset[NCHW_H_DIM];
        SetImageParams(inputImageAddr,
                       inputImageWidth,
                       inputImageHeight,
                       inputCenterX,
                       inputCenterY,
                       inputLastAddress,
                       outputStripeAddr,
                       roi.size[NCHW_W_DIM],
                       roi.size[NCHW_H_DIM],
                       outputStripeCenterX,
                       outputStripeCenterY,
                       rot_pars.irt_angles[e_irt_angle_rot],
                       super_irt,
                       irt_desc);
        uint32_t Hsi = irt_desc.image_par[IIMAGE].W << irt_desc.image_par[IIMAGE].Ps;
        irt_descriptor_gen(irt_cfg, rot_pars, irt_desc, Hsi, 0);
        setDescriptorParams(irt_desc, rot_pars, irt_cfg, context_id, desc);
        overrideDescParams(desc, outputImageWidth, inputLastAddress);
        // Finally, add the descriptor to the list of descriptors
        descriptors.push_back({desc, descMask, rot_wd_ctxt_t {0}});
    }
}

void DescriptorGenerator::generateRotatorEmptyJobDescriptor(RotatorDesc&               desc,
                                                            ValidityMask<RotatorDesc>& descMask,
                                                            uint64_t                   inImgAddr,
                                                            uint64_t                   outImgAddr)
{
    memset(&desc, 0, sizeof(desc));
    descMask.fill(false);  // turn everything off
    SizeArray inputDims  = {2, 2, 1, 1};
    SizeArray outputDims = {2, 2, 1, 1};

    uint32_t inputImageHeight = inputDims[NCHW_H_DIM];
    uint32_t inputImageWidth  = inputDims[NCHW_W_DIM];
    uint32_t inputImageSize   = inputImageWidth * inputImageHeight;
    uint32_t outputImageWidth = outputDims[NCHW_W_DIM];

    unsigned inputCenterX = 0;
    unsigned inputCenterY = 0;

    SET_MASK_BULK_ON(std::begin(descMask),
                     MASK_OFFSET(RotatorDesc, context_id),
                     MASK_OFFSET(RotatorDesc, cpl_msg_addr));

    SET_MASK_BULK_ON(std::begin(descMask),
                     MASK_OFFSET(RotatorDesc, x_i_start_offset),
                     MASK_OFFSET(RotatorDesc, hbw_aruser_hi));
    SET_MASK_BULK_ON(std::begin(descMask), MASK_OFFSET(RotatorDesc, owm_cfg), MASK_OFFSET(RotatorDesc, push_desc));

    uint16_t context_id     = 0;
    int      super_irt      = 0;
    int      line_wr_format = 0;

    // init intermediate data structures
    irt_desc_par irt_desc;
    irt_cfg_pars irt_cfg;
    rotation_par rot_pars;

    memset(&irt_desc, 0, sizeof(irt_desc_par));
    memset(&irt_cfg, 0, sizeof(irt_cfg_pars));
    irt_cfg_init(irt_cfg);
    memset(&rot_pars, 0, sizeof(rotation_par));
    irt_par_init(rot_pars);

    float rot_angle                      = 0;
    rot_pars.Pwi                         = 8;
    rot_pars.Pwo                         = 8;
    rot_pars.irt_angles[e_irt_angle_rot] = rot_angle;

    irt_desc.oimage_line_wr_format = line_wr_format;
    irt_desc.bg                    = 22;
    irt_desc.bg_mode               = e_irt_bg_prog_value;
    irt_desc.rate_mode             = e_irt_rate_adaptive_2x2;
    irt_desc.proc_size             = 8;

    uint64_t inputImageAddr      = inImgAddr;
    uint64_t inputLastAddress    = inputImageAddr + inputImageSize;
    uint64_t outputStripeAddr    = outImgAddr;
    uint32_t outputStripeCenterX = 0;
    uint32_t outputStripeCenterY = 0;
    SetImageParams(inputImageAddr,
                   inputImageWidth,
                   inputImageHeight,
                   inputCenterX,
                   inputCenterY,
                   inputLastAddress,
                   outputStripeAddr,
                   2,
                   2,
                   outputStripeCenterX,
                   outputStripeCenterY,
                   rot_pars.irt_angles[e_irt_angle_rot],
                   super_irt,
                   irt_desc);
    uint32_t Hsi = irt_desc.image_par[IIMAGE].W << irt_desc.image_par[IIMAGE].Ps;
    irt_descriptor_gen(irt_cfg, rot_pars, irt_desc, Hsi, 0);
    setDescriptorParams(irt_desc, rot_pars, irt_cfg, context_id, desc);
    overrideDescParams(desc, outputImageWidth, inputLastAddress);
}
