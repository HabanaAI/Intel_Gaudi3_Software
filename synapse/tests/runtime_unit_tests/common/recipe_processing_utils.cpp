#include "recipe_processing_utils.hpp"
#include "runtime/qman/gaudi/master_qmans_definition.hpp"

void RecipeProcessingUtils::getExecutionJobs(synDeviceType deviceType,
                                             uint32_t&     r_execute_jobs_nr,
                                             job_t*&       r_execute_jobs)
{
    switch (deviceType)
    {
        case synDeviceGaudi:
        {
            r_execute_jobs_nr = sizeof(executionJobsGaudi) / sizeof(job_t);
            r_execute_jobs    = &executionJobsGaudi[0];
            break;
        }
        default:
        {
            r_execute_jobs_nr = 0;
            r_execute_jobs    = nullptr;
        }
    }
}

job_t RecipeProcessingUtils::executionJobsGaudi[16] {{999, 0},
                                                     {GAUDI_QUEUE_ID_DMA_2_0, 0},
                                                     {GAUDI_QUEUE_ID_DMA_3_0, 0},
                                                     {GAUDI_QUEUE_ID_DMA_4_0, 0},
                                                     {GAUDI_QUEUE_ID_DMA_6_0, 0},
                                                     {GAUDI_QUEUE_ID_DMA_7_0, 0},
                                                     {GAUDI_QUEUE_ID_MME_0_0, 0},
                                                     {GAUDI_QUEUE_ID_MME_1_0, 0},
                                                     {GAUDI_QUEUE_ID_TPC_0_0, 0},
                                                     {GAUDI_QUEUE_ID_TPC_1_0, 0},
                                                     {GAUDI_QUEUE_ID_TPC_2_0, 0},
                                                     {GAUDI_QUEUE_ID_TPC_3_0, 0},
                                                     {GAUDI_QUEUE_ID_TPC_4_0, 0},
                                                     {GAUDI_QUEUE_ID_TPC_5_0, 0},
                                                     {GAUDI_QUEUE_ID_TPC_6_0, 0},
                                                     {GAUDI_QUEUE_ID_TPC_7_0, 0}};
