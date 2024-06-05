#pragma once
#include <stdint.h>

#include <string>
#include <vector>

#include <gaudi2/mme.h>

#include "fs_mme_desc.h"
#include "fs_mme_unit.h"

namespace Gaudi2
{
namespace Mme
{
class EU_fp : public Gaudi2::Mme::Unit
{
   public:
    static const unsigned c_matrix_height = Gaudi2::Mme::c_mme_dcore_matrix_height_in_bytes / sizeof(uint16_t);
    static const unsigned c_matrix_width  = Gaudi2::Mme::c_mme_dcore_matrix_width_in_bytes / sizeof(uint16_t);

    static const unsigned c_fma_adder_depth     = 8;
    static const unsigned c_fma_fp8_adder_depth = c_fma_adder_depth * 2;
    static const unsigned c_operand_b_vec_size  = c_matrix_width / c_fma_adder_depth;

    union float8
    {
        uint8_t b;
        struct
        {
            uint16_t man : 3;
            uint16_t exp : 4;
            uint16_t sign : 1;
        } mode_143;
        struct
        {
            uint16_t man : 2;
            uint16_t exp : 5;
            uint16_t sign : 1;
        } mode_152;
    };

    union bfloat
    {
        uint16_t w;
        uint8_t  b[2];
        struct
        {
            uint16_t man : 7;
            uint16_t exp : 8;
            uint16_t sign : 1;
        };
    };

    union float16
    {
        uint16_t w;
        uint8_t  b[2];
        struct
        {
            uint16_t man : 10;
            uint16_t exp : 5;
            uint16_t sign : 1;
        };
    };

    // TODO: define FP8 union

    struct Config
    {
        uint32_t widthMinus1 : 8; // effective symetric matrix width in elements
        uint32_t heightMinus1 : 8; // effective symetric matrix height in elements.
        uint32_t dataType : 7; // EMmeDataType
        uint32_t hx2 : 1; // 2xH mode
        uint32_t rollupEn : 1; // enable rollup
        uint32_t fp8BiasOpA : 4;
        uint32_t fp8BiasOpB : 4;
        uint32_t clipFP : 1;
    };

    union VecA
    {
        uint16_t vec16[c_matrix_height][c_fma_adder_depth];
        bfloat   vecBf[c_matrix_height][c_fma_adder_depth];
        float16  vecF16[c_matrix_height][c_fma_adder_depth];
        float8   vecF8[c_matrix_height][c_fma_adder_depth][2];
        float    vecF32[c_matrix_height][c_fma_adder_depth / 2];
        uint32_t vecU32[c_operand_b_vec_size][c_fma_adder_depth / 2];
        uint16_t data16[c_matrix_height * c_fma_adder_depth];
    };

    union VecB
    {
        uint16_t vec16[c_operand_b_vec_size][c_fma_adder_depth];
        bfloat   vecBf[c_operand_b_vec_size][c_fma_adder_depth];
        float16  vecF16[c_operand_b_vec_size][c_fma_adder_depth];
        float8   vecFP8[c_operand_b_vec_size][c_fma_adder_depth][2];
        uint32_t vecF32[c_fma_adder_depth / 2][c_operand_b_vec_size];
        uint32_t vecU32[c_operand_b_vec_size][c_fma_adder_depth / 2];
        uint16_t data16[c_operand_b_vec_size * c_fma_adder_depth];
    };

    union RollupVec
    {
        float f[c_matrix_width];
    };

    union Accums
    {
        float    f[c_fma_adder_depth][c_matrix_height][c_matrix_width / c_fma_adder_depth];
        uint32_t i[c_fma_adder_depth][c_matrix_height][c_matrix_width / c_fma_adder_depth];
    };

    EU_fp(FS_Mme* mme, const std::string& name) : Unit(mme, name), m_cycCtr(0)
    {
        memset(m_accums.i, 0, sizeof(m_accums));
    }

    virtual void
    mul(const Config& config, const VecA& vecA, const VecB& vecB, RollupVec* rollup, uint32_t& rollupIndex) = 0;

   protected:
    Accums   m_accums;
    uint64_t m_cycCtr;
};

class EU_fp_AVX512 : public virtual EU_fp
{
   public:
    EU_fp_AVX512(FS_Mme* mme = nullptr, const std::string& name = std::string()) : EU_fp(mme, name){};
    virtual ~EU_fp_AVX512() override = default;
    virtual void
    mul(const Config& config, const VecA& vecA, const VecB& vecB, RollupVec* rollup, uint32_t& rollupIndex) override;

   private:
    void mulFloat(const Config& config,
                  const VecA&   vecA,
                  const VecB&   vecB,
                  const uint8_t accIdx,
                  RollupVec*    rollup,
                  uint32_t&     rollupIndex);
};

class EU_fp_AVX256 : public EU_fp
{
   public:
    EU_fp_AVX256(FS_Mme* mme = nullptr, const std::string& name = std::string()) : EU_fp(mme, name){};
    virtual ~EU_fp_AVX256() override = default;
    virtual void
    mul(const Config& config, const VecA& vecA, const VecB& vecB, RollupVec* rollup, uint32_t& rollupIndex) override;

   private:
    void mulFloat(const Config& config,
                  const VecA&   vecA,
                  const VecB&   vecB,
                  const uint8_t accIdx,
                  RollupVec*    rollup,
                  uint32_t&     rollupIndex);
};

class EU_fp_NOAVX : public EU_fp
{
   public:
    EU_fp_NOAVX(FS_Mme* mme = nullptr, const std::string& name = std::string()) : EU_fp(mme, name){};
    virtual ~EU_fp_NOAVX() override = default;
    virtual void
    mul(const Config& config, const VecA& vecA, const VecB& vecB, RollupVec* rollup, uint32_t& rollupIndex) override;

   private:
    void mulFloat(const Config& config,
                  const VecA&   vecA,
                  const VecB&   vecB,
                  const uint8_t accIdx,
                  RollupVec*    rollup,
                  uint32_t&     rollupIndex);
};

} // namespace Mme
} // namespace Gaudi2
