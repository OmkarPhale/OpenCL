// headers
#include <stdio.h>
#include <stdlib.h> //exit()
#include <string.h> //strlen()
#include <math.h>   //fabs()

#include <CL/opencl.h> //standard OpenCL header

#include "helper_timer.h"

// global OpenCL variables
// const int iNumberOfArrayElements = 5;
const int iNumberOfArrayElements = 11444777;

cl_platform_id oclPlatformID;
cl_device_id oclDeviceID;

cl_context oclContext;
cl_command_queue oclCommandQueue;

cl_program oclProgram;
cl_kernel oclKernel;

float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;
float *gold = NULL;

cl_mem deviceInput1 = NULL;
cl_mem deviceInput2 = NULL;
cl_mem deviceOutput = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;

// OpenCL kernel
char *oclSourceCode =
    "__kernel void vecAddGPU(__global float *input1, __global float *input2, __global float *output, int length)        \n"
    "{                                                                                                                  \n"
    "    int index = get_global_id(0);                                                                                  \n"
    "    if(index < length)                                                                                             \n"
    "    {                                                                                                              \n"
    "        output[index] = input1[index] + input2[index];                                                             \n"
    "    }                                                                                                              \n"
    "}                                                                                                                  \n";

// main() definition
int main(void)
{
    // local function declaration
    void fillArrayWithRandomNumbers(float *, int);
    size_t roundGlobalSizeToNearestMultipleOfLocalSize(int, unsigned int);
    void vecAddCPU(const float *, const float *, float *, int);
    void cleanup(void);

    // local variable declaration
    int size = iNumberOfArrayElements * sizeof(float);
    cl_int result;

    // code
    // host memory allocation
    hostInput1 = (float *)malloc(size);
    if (hostInput1 == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostInput1 Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float *)malloc(size);
    if (hostInput2 == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostInput2 Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutput = (float *)malloc(size);
    if (hostOutput == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostOutput Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    gold = (float *)malloc(size);
    if (gold == NULL)
    {
        printf("error>> Host Memory Allocation Failed For gold Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // filling values into host arrays
    fillArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
    fillArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

    // get OpenCL supporting platform's ID
    result = clGetPlatformIDs(1, &oclPlatformID, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clGetPlatformIDs() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // get OpenCL supporting GPU device's ID
    result = clGetDeviceIDs(oclPlatformID, CL_DEVICE_TYPE_GPU, 1, &oclDeviceID, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clGetDeviceIDs() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // create OpenCL compute context
    oclContext = clCreateContext(NULL, 1, &oclDeviceID, NULL, NULL, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateContext() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // create command queue
    oclCommandQueue = clCreateCommandQueue(oclContext, oclDeviceID, 0, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateCommandQueue() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // create OpenCL program from .cl
    oclProgram = clCreateProgramWithSource(oclContext, 1, (const char **)&oclSourceCode, NULL, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateProgramWithSource() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // build OpenCL program
    result = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clBuildProgram() Failed : %d. Terminating Now ...\n", result);

        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(oclProgram, oclDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("OpenCL Program Build Log : %s\n", buffer);
        printf("error>> clBuildProgram() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // create OpenCL kernel by passing kernel function name that we used in .cl file
    oclKernel = clCreateKernel(oclProgram, "vecAddGPU", &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateKernel() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // allocate device memory
    size = iNumberOfArrayElements * sizeof(cl_float);
    deviceInput1 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateBuffer() Failed For 1st Input Array : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceInput2 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateBuffer() Failed For 2nd Input Array : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceOutput = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, size, NULL, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateBuffer() Failed For Output Array : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 0th argument i.e deviceInput
    result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void *)&deviceInput1);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 1st Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 1st argument i.e deviceInpu2
    result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void *)&deviceInput2);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 2nd Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 2nd argument i.e deviceOutput
    result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void *)&deviceOutput);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 3rd Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 3rd argument i.e length
    result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void *)&iNumberOfArrayElements);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 4th Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // write above "input" device buffer to device memory
    result = clEnqueueWriteBuffer(oclCommandQueue, deviceInput1, CL_FALSE, 0, size, hostInput1, 0, NULL, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clEnqueueWriteBuffer() Failed For 1st Input Device Buffer : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = clEnqueueWriteBuffer(oclCommandQueue, deviceInput2, CL_FALSE, 0, size, hostInput2, 0, NULL, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clEnqueueWriteBuffer() Failed For 2nd Input Device Buffer : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // kernel configuration
    // size_t localWorkSize = 5;
    size_t localWorkSize = 256;
    size_t globalWorkSize;

    globalWorkSize = roundGlobalSizeToNearestMultipleOfLocalSize(localWorkSize, iNumberOfArrayElements);

    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    result = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clEnqueueNDRangeKernel() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // finish OpenCL command queue
    clFinish(oclCommandQueue);

    // stop timer
    sdkStopTimer(&timer);
    timeOnGPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;

    // read back result from the device (i.e from deviceOutput) into cpu variable (i.e hostOutput)
    result = clEnqueueReadBuffer(oclCommandQueue, deviceOutput, CL_TRUE, 0, size, hostOutput, 0, NULL, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clEnqueueReadBuffer() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // vector addition on host
    vecAddCPU(hostInput1, hostInput2, gold, iNumberOfArrayElements);

    // comparison
    const float epsilon = 0.000001f;
    int breakValue = -1;
    bool bAccuracy = true;
    int index;
    for (index = 0; index < iNumberOfArrayElements; index++)
    {
        float val1 = gold[index];
        float val2 = hostOutput[index];

        if (fabs(val1 - val2) > epsilon)
        {
            bAccuracy = false;
            breakValue = index;
            break;
        }
    }

    char stringMessage[125];
    if (bAccuracy == false)
    {
        sprintf(stringMessage, "%s", "# Comparison Of CPU And GPU Vector Addition Is Not With Accuracy Of Limit Of 0.000001 At Array Index %d", breakValue);
    }
    else
    {
        sprintf(stringMessage, "%s", "# Comparison Of CPU And GPU Vector Addition Is With Accuracy Of Limit Of 0.000001.");
    }

    printf("\n==================================================================================\n");
    printf("+ DISPLAYING THE RESULT OF ADDITION FROM DEVICE TO HOST +\n");
    printf("==================================================================================\n");

    printf("- Array1 Begins From 0th Index %0.6f To %dth Index %0.6f\n", hostInput1[0], (iNumberOfArrayElements - 1), hostInput1[iNumberOfArrayElements - 1]);
    printf("- Array2 Begins From 0th Index %0.6f To %dth Index %0.6f\n\n", hostInput2[0], (iNumberOfArrayElements - 1), hostInput2[iNumberOfArrayElements - 1]);
    printf("- OpenCL Kernel Global Work Size = %zu And Local Work Size = %zu\n\n", globalWorkSize, localWorkSize);
    printf("- Output Array Begins From 0th Index %0.6f To %dth Index %0.6f\n\n", hostOutput[0], (iNumberOfArrayElements - 1), hostOutput[iNumberOfArrayElements - 1]);
    printf("- The Time Taken To Do Above Addition On CPU = %0.6f (ms)\n", timeOnCPU);
    printf("- The Time Taken To Do Above Addition On GPU = %0.6f (ms)\n", timeOnGPU);
    printf("%s\n", stringMessage);
    printf("==================================================================================\n");

    // total cleanup
    cleanup();

    return (0);
}

// cleanup() definition
void cleanup(void)
{
    // code
    // OpenCL cleanup
    if (oclSourceCode)
    {
        free((void *)oclSourceCode);
        oclSourceCode = NULL;
    }

    if (oclKernel)
    {
        clReleaseKernel(oclKernel);
        oclKernel = NULL;
    }

    if (oclProgram)
    {
        clReleaseProgram(oclProgram);
        oclProgram = NULL;
    }

    if (oclCommandQueue)
    {
        clReleaseCommandQueue(oclCommandQueue);
        oclCommandQueue = NULL;
    }

    if (oclContext)
    {
        clReleaseContext(oclContext);
        oclContext = NULL;
    }

    // free allocated device memory
    if (deviceOutput)
    {
        clReleaseMemObject(deviceOutput);
        deviceOutput = NULL;
    }

    if (deviceInput2)
    {
        clReleaseMemObject(deviceInput2);
        deviceInput2 = NULL;
    }

    if (deviceInput1)
    {
        clReleaseMemObject(deviceInput1);
        deviceInput1 = NULL;
    }

    // free allocated host memory
    if (gold)
    {
        free(gold);
        gold = NULL;
    }

    if (hostOutput)
    {
        free(hostOutput);
        hostOutput = NULL;
    }

    if (hostInput2)
    {
        free(hostInput2);
        hostInput2 = NULL;
    }

    if (hostInput1)
    {
        free(hostInput1);
        hostInput1 = NULL;
    }
}

// fillArrayWithRandomNumbers() definition
void fillArrayWithRandomNumbers(float *pFloatArray, int iSize)
{
    // code
    int index;
    const float fScale = 1.0f / (float)RAND_MAX;
    for (index = 0; index < iSize; index++)
    {
        pFloatArray[index] = fScale * rand();
    }
}

// roundGlobalSizeToNearestMultipleOfLocalSize() definition
size_t roundGlobalSizeToNearestMultipleOfLocalSize(int local_size, unsigned int global_size)
{
    // code
    unsigned int r = global_size % local_size;

    if (r == 0)
        return (global_size);
    else
        return (global_size + local_size - r);
}

// vecAddCPU() definition
void vecAddCPU(const float *in1, const float *in2, float *out, int iNumElements)
{
    int index;

    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (index = 0; index < iNumElements; index++)
    {
        out[index] = in1[index] + in2[index];
    }

    // stop timer
    sdkStopTimer(&timer);
    timeOnCPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;
}
