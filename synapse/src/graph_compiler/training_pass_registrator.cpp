// clang-format off

#include "training_pass_registrator.h"
#include "habana_graph.h"
#include "pass_dependencies/training/passes_dependencies.h"

#define GRAPH_REGISTER_GROUP(name_, id_, groupMembers_, depSet_)                                                       \
    graph.addPass(pPass(new PassGroup(#name_, (id_), groupMembers_, depSet_)))

void TrainingPassRegistrator::registerGroups(HabanaGraph& graph) const
{
    //                      NAME                                   ID                                                     GROUP MEMBERS                                                 Dependency Set
    //                       ===                                   ===                                                     ========                                                     ==============
    GRAPH_REGISTER_GROUP(PreCompileGroup,                       GROUP_ID_PRE_COMPILE_GROUP,                             {PRE_COMPILE_GROUP_MEMBERS},                                {PRE_COMPILE_GROUP_DEPENDENCY_SET}                              );
    GRAPH_REGISTER_GROUP(PreCompilePreProccess,                 GROUP_ID_PRE_COMPILE_PRE_PROCCESS_GROUP,                {PRE_COMPILE_PRE_PROCCESS_GROUP_MEMBERS},                   {PRE_COMPILE_PRE_PROCCESS_GROUP_DEPENDENCY_SET}                 );
    GRAPH_REGISTER_GROUP(PreCompileCorrectUserGraph,            GROUP_ID_PRE_COMPILE_CORRECT_USER_LEVEL,                {PRE_COMPILE_CORRECT_USER_LEVEL_GROUP_MEMBERS},             {PRE_COMPILE_CORRECT_USER_LEVEL_GROUP_DEPENDENCY_SET}           );
    GRAPH_REGISTER_GROUP(PreCompileCorrectUserGraphDataLayout,  GROUP_ID_CORRECT_USER_GRAPH_DATA_LAYOUT,                {CORRECT_USER_GRAPH_DATA_LAYOUT_GROUP_MEMBERS},             {CORRECT_USER_GRAPH_DATA_LAYOUT_DEPENDENCY_SET}                 );
    GRAPH_REGISTER_GROUP(PreCompileOptimizeUserLevelGraph,      GROUP_ID_PRE_COMPILE_OPTIMIZE_USER_LEVEL_GRAPH,         {PRE_COMPILE_OPTIMIZE_USER_LEVEL_GRAPH_GROUP_MEMBERS},      {PRE_COMPILE_OPTIMIZE_USER_LEVEL_GRAPH_GROUP_DEPENDENCY_SET}    );
    GRAPH_REGISTER_GROUP(PreCompileOptimizeControlEdges,        GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES,            {PRE_COMPILE_OPTIMIZE_CONTROL_EDGES_GROUP_MEMBERS},         {PRE_COMPILE_OPTIMIZE_CONTROL_EDGES_GROUP_DEPENDENCY_SET}       );
    GRAPH_REGISTER_GROUP(PreCompileOptimizeDataLayout,          GROUP_ID_PRE_COMPILE_PRE_COMPILE_OPTIMIZE_DATA_LAYOUT,  {PRE_COMPILE_OPTIMIZE_DATA_LAYOUT_GROUP_MEMBERS},           {PRE_COMPILE_OPTIMIZE_DATA_LAYOUT_GROUP_DEPENDENCY_SET}         );
    GRAPH_REGISTER_GROUP(LoweringGroup,                         GROUP_ID_LOWERING_GROUP,                                {LOWERING_GROUP_MEMBERS},                                   {LOWERING_GROUP_DEPENDENCY_SET}                                 );
    GRAPH_REGISTER_GROUP(LoweringNormOps,                       GROUP_ID_LOWERING_NORM_OPS,                             {LOWERING_NORM_OPS_GROUP_MEMBERS},                          {LOWERING_NORM_OPS_GROUP_DEPENDENCY_SET}                        );
    GRAPH_REGISTER_GROUP(GconvHandling,                         GROUP_ID_GCONV_HANDLING_GROUP,                          {GCONV_HANDLING_GROUP_MEMBERS},                             {GCONV_HANDLING_GROUP_DEPENDENCY_SET}                           );
    GRAPH_REGISTER_GROUP(HugeTensorsHandling,                   GROUP_ID_HUGE_TENSORS_HANDLING_GROUP,                   {HUGE_TENSORS_HANDLING_GROUP_MEMBERS},                      {HUGE_TENSORS_HANDLING_GROUP_DEPENDENCY_SET}                    );
    GRAPH_REGISTER_GROUP(BeGroup,                               GROUP_ID_BE_GROUP,                                      {BE_GROUP_MEMBERS},                                         {BE_GROUP_DEPENDENCY_SET}                                       );
    GRAPH_REGISTER_GROUP(BeGraphOptGroup,                       GROUP_ID_BE_GRAPH_OPT_GROUP,                            {BE_GRAPH_OPT_GROUP_MEMBERS},                               {BE_GRAPH_OPT_GROUP_DEPENDENCY_SET}                             );
    GRAPH_REGISTER_GROUP(BeMainGroup,                           GROUP_ID_BE_MAIN_GROUP,                                 {BE_MAIN_GROUP_MEMBERS},                                    {BE_MAIN_GROUP_DEPENDENCY_SET}                                  );
    GRAPH_REGISTER_GROUP(BeTPCGroup,                            GROUP_ID_BE_TPC_GROUP,                                  {BE_TPC_GROUP_MEMBERS},                                     {BE_TPC_GROUP_DEPENDENCY_SET}                                   );
    GRAPH_REGISTER_GROUP(BeMMEOptimizationGroup,                GROUP_ID_BE_MME_OPTIMIZATION_GROUP,                     {BE_MME_OPTIMIZATION_GROUP_MEMBERS},                        {BE_MME_OPTIMIZATION_GROUP_DEPENDENCY_SET}                                   );
    GRAPH_REGISTER_GROUP(BeGCBrainGroup,                        GROUP_ID_BE_GC_BRAIN_GROUP,                             {BE_GC_BRAIN_GROUP_MEMBERS},                                {BE_GC_BRAIN_GROUP_DEPENDENCY_SET}                              );
    GRAPH_REGISTER_GROUP(BeGCBrainPreSlicingGroup,              GROUP_ID_BE_GC_BRAIN_PRE_SLICING_GROUP,                 {BE_GC_BRAIN_PRE_SLICING_GROUP_MEMBERS},                    {BE_GC_BRAIN_PRE_SLICING_DEPENDENCY_SET}                        );
    GRAPH_REGISTER_GROUP(BeGCBrainPostSlicingGroup,             GROUP_ID_BE_GC_BRAIN_POST_SLICING_GROUP,                {BE_GC_BRAIN_POST_SLICING_GROUP_MEMBERS},                   {BE_GC_BRAIN_POST_SLICING_DEPENDENCY_SET}                       );
    GRAPH_REGISTER_GROUP(BeGCBrainLogicalOpsGroup,              GROUP_ID_BE_GC_BRAIN_LOGICAL_OPS_GROUP,                 {BE_GC_BRAIN_LOGICAL_OPS_GROUP_MEMBERS},                    {BE_GC_BRAIN_LOGICAL_OPS_DEPENDENCY_SET}                        );
    GRAPH_REGISTER_GROUP(BeGCBrainMemoryGroup,                  GROUP_ID_BE_GC_BRAIN_MEMORY_GROUP,                      {BE_GC_BRAIN_MEMORY_GROUP_MEMBERS},                         {BE_GC_BRAIN_MEMORY_GROUP_DEPENDENCY_SET}                       );
    GRAPH_REGISTER_GROUP(BePostGraphGroup,                      GROUP_ID_BE_POST_GRAPH_GROUP,                           {BE_POST_GRAPH_GROUP_MEMBERS},                              {BE_POST_GRAPH_GROUP_DEPENDENCY_SET}                            );
    GRAPH_REGISTER_GROUP(PreCompileCalcQuantInfo,               GROUP_ID_PRE_COMPILE_CALC_QUANT_INFO,                   {PRE_COMPILE_CALC_QUANT_INFO_GROUP_MEMBERS},                {PRE_COMPILE_CALC_QUANT_INFO_GROUP_DEPENDENCY_SET}              );
    GRAPH_REGISTER_GROUP(PreCompileDataTypeSelection,           GROUP_ID_PRE_COMPILE_DATA_TYPE_SELECTION,               {PRE_COMPILE_DATA_TYPE_SELECTION_GROUP_MEMBERS},            {PRE_COMPILE_DATA_TYPE_SELECTION_GROUP_DEPENDENCY_SET}          );
    GRAPH_REGISTER_GROUP(PreCompileCastsInjection,              GROUP_ID_PRE_COMPILE_CASTS_INJECTION,                   {PRE_COMPILE_CASTS_INJECTION_GROUP_MEMBERS},                {PRE_COMPILE_CASTS_INJECTION_GROUP_DEPENDENCY_SET}              );
    GRAPH_REGISTER_GROUP(PostGraphGroup,                        GROUP_ID_POST_GRAPH_GROUP,                              {POST_GRAPH_GROUP_MEMBERS},                                 {POST_GRAPH_GROUP_DEPENDENCY_SET}                               );
    GRAPH_REGISTER_GROUP(PostGraphMemoryGroup,                  GROUP_ID_POST_GRAPH_MEMORY,                             {POST_GRAPH_MEMORY_GROUP_MEMBERS},                          {POST_GRAPH_MEMORY_GROUP_DEPENDENCY_SET}                        );
    GRAPH_REGISTER_GROUP(PostGraphValidationsGroup,             GROUP_ID_POST_GRAPH_VALIDATIONS,                        {POST_GRAPH_VALIDATION_GROUP_MEMBERS},                      {POST_GRAPH_VALIDATION_GROUP_DEPENDENCY_SET}                    );
    GRAPH_REGISTER_GROUP(PostGraphCodeGenGroup,                 GROUP_ID_POST_GRAPH_CODE_GEN,                           {POST_GRAPH_CODE_GEN_GROUP_MEMBERS},                        {POST_GRAPH_CODE_GEN_GROUP_DEPENDENCY_SET}                      );
    GRAPH_REGISTER_GROUP(PostGraphCodeGenMainGroup,             GROUP_ID_POST_GRAPH_CODE_GEN_MAIN_GROUP,                {POST_GRAPH_CODE_GEN_MAIN_GROUP_MEMBERS},                   {POST_GRAPH_CODE_GEN_MAIN_GROUP_DEPENDENCY_SET}                 );
    GRAPH_REGISTER_GROUP(PostGraphCodeGenLinearRangesGroup,     GROUP_ID_POST_GRAPH_CODE_GEN_LINEAR_RANGES,             {POST_GRAPH_CODE_GEN_LINEAR_RANGES_GROUP_MEMBERS},          {POST_GRAPH_CODE_GEN_LINEAR_RANGES_DEPENDENCY_SET}              );
    GRAPH_REGISTER_GROUP(PostGraphCodeGenLogicalRoiGroup,       GROUP_ID_POST_GRAPH_CODE_GEN_LOGICAL_ROI,               {POST_GRAPH_CODE_GEN_LOGICAL_ROI_GROUP_MEMBERS},            {POST_GRAPH_CODE_GEN_LOGICAL_ROI_DEPENDENCY_SET}                );
    GRAPH_REGISTER_GROUP(TpcFuserGroup,                         GROUP_ID_BE_TPC_FUSER_GROUP,                               {BE_TPC_FUSER_GROUP_MEMBERS},                            {BE_TPC_FUSER_GROUP_DEPENDENCY_SET}                             );
}

// clang-format on
