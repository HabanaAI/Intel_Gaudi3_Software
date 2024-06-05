#pragma once

#include "recipe.h"

class RecipeProcessingUtils
{
public:
    static void getExecutionJobs(synDeviceType deviceType, uint32_t& r_execute_jobs_nr, job_t*& r_execute_jobs);

private:
    static job_t executionJobsGaudi[16];
};