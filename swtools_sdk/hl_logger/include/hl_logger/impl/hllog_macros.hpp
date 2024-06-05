#pragma once

// clang-format off
/**
 * @brief calculate the number of parameters
 *
 * @param parameters
 * @return the number of parameters
 * @examples
 * HLLOG_COUNTOF()        => 0
 * HLLOG_COUNTOF(a, b, c) => 3
 */
#define HLLOG_COUNTOF(...) HLLOG_COUNTOF_CAT( HLLOG_COUNTOF_A, ( 0, ##__VA_ARGS__, 120, \
    119, 118, 117, 116, 115, 114, 113, 112, 111, 110,\
    109, 108, 107, 106, 105, 104, 103, 102, 101, 100,\
    99, 98, 97, 96, 95, 94, 93, 92, 91, 90,\
    89, 88, 87, 86, 85, 84, 83, 82, 81, 80,\
    79, 78, 77, 76, 75, 74, 73, 72, 71, 70,\
    69, 68, 67, 66, 65, 64, 63, 62, 61, 60,\
    59, 58, 57, 56, 55, 54, 53, 52, 51, 50,\
    49, 48, 47, 46, 45, 44, 43, 42, 41, 40,\
    39, 38, 37, 36, 35, 34, 33, 32, 31, 30,\
    29, 28, 27, 26, 25, 24, 23, 22, 21, 20,\
    19, 18, 17, 16, 15, 14, 13, 12, 11, 10,\
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0 ) )
#define HLLOG_COUNTOF_CAT( a, b ) a b
#define HLLOG_COUNTOF_A( a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,\
    a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,\
    a20, a21, a22, a23, a24, a25, a26, a27, a28, a29,\
    a30, a31, a32, a33, a34, a35, a36, a37, a38, a39,\
    a40, a41, a42, a43, a44, a45, a46, a47, a48, a49,\
    a50, a51, a52, a53, a54, a55, a56, a57, a58, a59,\
    a60, a61, a62, a63, a64, a65, a66, a67, a68, a69,\
    a70, a71, a72, a73, a74, a75, a76, a77, a78, a79,\
    a80, a81, a82, a83, a84, a85, a86, a87, a88, a89,\
    a90, a91, a92, a93, a94, a95, a96, a97, a98, a99,\
    a100, a101, a102, a103, a104, a105, a106, a107, a108, a109,\
    a110, a111, a112, a113, a114, a115, a116, a117, a118, a119,\
    a120, n, ... ) n
// clang-format on

#define HLLOG_CONCAT(a, b)  HLLOG_CONCAT_(a, b)
#define HLLOG_CONCAT_(a, b) a##b

#define HLLOG_APPLY_0(comma, sep, OP)
#define HLLOG_APPLY_1(comma, sep, OP, v, ...)  comma OP(v)
#define HLLOG_APPLY_2(comma, sep, OP, v, ...)  comma OP(v) sep() HLLOG_APPLY_1(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_3(comma, sep, OP, v, ...)  comma OP(v) sep() HLLOG_APPLY_2(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_4(comma, sep, OP, v, ...)  comma OP(v) sep() HLLOG_APPLY_3(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_5(comma, sep, OP, v, ...)  comma OP(v) sep() HLLOG_APPLY_4(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_6(comma, sep, OP, v, ...)  comma OP(v) sep() HLLOG_APPLY_5(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_7(comma, sep, OP, v, ...)  comma OP(v) sep() HLLOG_APPLY_6(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_8(comma, sep, OP, v, ...)  comma OP(v) sep() HLLOG_APPLY_7(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_9(comma, sep, OP, v, ...)  comma OP(v) sep() HLLOG_APPLY_8(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_10(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_9(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_11(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_10(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_12(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_11(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_13(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_12(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_14(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_13(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_15(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_14(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_16(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_15(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_17(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_16(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_18(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_17(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_19(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_18(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_20(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_19(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_21(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_20(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_22(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_21(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_23(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_22(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_24(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_23(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_25(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_24(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_26(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_25(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_27(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_26(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_28(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_27(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_29(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_28(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_30(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_29(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_31(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_30(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_32(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_31(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_33(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_32(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_34(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_33(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_35(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_34(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_36(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_35(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_37(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_36(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_38(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_37(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_39(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_38(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_40(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_39(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_41(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_40(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_42(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_41(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_43(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_42(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_44(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_43(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_45(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_44(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_46(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_45(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_47(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_46(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_48(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_47(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_49(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_48(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_50(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_49(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_51(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_50(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_52(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_51(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_53(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_52(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_54(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_53(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_55(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_54(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_56(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_55(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_57(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_56(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_58(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_57(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_59(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_58(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_60(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_59(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_61(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_60(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_62(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_61(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_63(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_62(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_64(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_63(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_65(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_64(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_66(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_65(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_67(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_66(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_68(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_67(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_69(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_68(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_70(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_69(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_71(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_70(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_72(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_71(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_73(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_72(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_74(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_73(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_75(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_74(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_76(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_75(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_77(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_76(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_78(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_77(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_79(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_78(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_80(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_79(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_81(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_80(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_82(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_81(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_83(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_82(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_84(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_83(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_85(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_84(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_86(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_85(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_87(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_86(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_88(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_87(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_89(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_88(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_90(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_89(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_91(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_90(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_92(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_91(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_93(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_92(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_94(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_93(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_95(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_94(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_96(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_95(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_97(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_96(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_98(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_97(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_99(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_98(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_100(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_99(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_101(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_100(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_102(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_101(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_103(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_102(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_104(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_103(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_105(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_104(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_106(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_105(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_107(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_106(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_108(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_107(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_109(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_108(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_110(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_109(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_111(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_110(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_112(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_111(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_113(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_112(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_114(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_113(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_115(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_114(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_116(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_115(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_117(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_116(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_118(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_117(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_119(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_118(, sep, OP, ##__VA_ARGS__)
#define HLLOG_APPLY_120(comma, sep, OP, v, ...) comma OP(v) sep() HLLOG_APPLY_119(, sep, OP, ##__VA_ARGS__)



// comma for passing to macros as a parameter
#define HLLOG_COMMA() ,
#define HLLOG_EMPTY()
/**
 * @brief apply an operation to all the variadic parameters
 *
 * @param OP oparation that is applied to all the parameters
 * @param ... variadic parameters
 *
 * @example
 * HLLOG_APPLY(op) =>
 * HLLOG_APPLY(op, a, b, c) => op(a), op(b), op(c)
 */
#define HLLOG_APPLY(sep, OP, ...) HLLOG_CONCAT(HLLOG_APPLY_, HLLOG_COUNTOF(__VA_ARGS__))(, sep, OP, ##__VA_ARGS__)

/**
 * @brief apply an operation to all the variadic parameters
 *         add a leading comma if the number of parameters is more than 0
 *
 * @param OP oparation that is applied to all the parameters
 * @param ... variadic parameters
 *
 * @example
 * HLLOG_APPLY_WITH_LEADING_COMMA(op) =>
 * HLLOG_APPLY_WITH_LEADING_COMMA(op, a, b, c) => , op(a), op(b), op(c)
 */
#define HLLOG_APPLY_WITH_LEADING_COMMA(OP, ...)                                                                        \
    HLLOG_CONCAT(HLLOG_APPLY_, HLLOG_COUNTOF(__VA_ARGS__))(HLLOG_COMMA(), HLLOG_COMMA, OP, ##__VA_ARGS__)
