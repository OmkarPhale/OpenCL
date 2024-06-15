// headers
#include <stdio.h>
#include <stdlib.h> // exit()
#include <math.h>   // fabs()

#include <CL/opencl.h> // standard OpenCL header

#include "helper_timer.h"

// macros
#define BLOCK_WIDTH 64

// global variables declaration
cl_platform_id oclPlatformID;
cl_device_id oclComputeDeviceID;

cl_context oclContext;
cl_command_queue oclCommandQueue;

cl_program oclProgram;
cl_kernel oclKernel;

int *hostA = NULL;
int *hostB = NULL;
int *hostC = NULL;
int *gold = NULL;

cl_mem deviceA = NULL;
cl_mem deviceB = NULL;
cl_mem deviceC = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;

// OpenCL kernel
char *oclSourceCode =
    " __kernel void matrixMultiplyGPU(__global int *A, __global int *B, __global int *C, int numberOfARows, int numberOfAColumns, int numberOfBColumns, int numberOfCColumns)       \n"
    "{                                                                                                                                                                              \n"
    "   int rowIndex = get_global_id(0);                                                                                                                                            \n"
    "   int columnIndex = get_global_id(1);                                                                                                                                         \n"
    "   if ((rowIndex < numberOfARows) && (columnIndex < numberOfBColumns))                                                                                                         \n"
    "   {                                                                                                                                                                           \n"
    "       float value = 0.0f;                                                                                                                                                     \n"
    "       for (int depth = 0; depth < numberOfAColumns; depth++)                                                                                                                  \n"
    "       {                                                                                                                                                                       \n"
    "           int a = A[rowIndex * numberOfAColumns + depth];                                                                                                                     \n"
    "           int b = B[depth * numberOfBColumns + columnIndex];                                                                                                                  \n"
    "           value += (a * b);                                                                                                                                                   \n"
    "       }                                                                                                                                                                       \n"
    "       C[rowIndex * numberOfCColumns + columnIndex] = value;                                                                                                                   \n"
    "   }                                                                                                                                                                           \n"
    "}                                                                                                                                                                              \n";

// main() definition
int main(void)
{
    // local function declaration
    void InitA(int *data, int, int);
    void InitB(int *data, int, int);
    void matMulCPU(int *, int *, int *, int, int, int, int);
    void cleanup(void);

    // local variable declaration
    int numberOfARows = BLOCK_WIDTH;
    int numberOfAColumns = BLOCK_WIDTH;
    int numberOfBRows = BLOCK_WIDTH;
    int numberOfBColumns = BLOCK_WIDTH;

    int numberOfCRows = numberOfARows;
    int numberOfCColumns = numberOfBColumns;

    int numberOfGoldRows = numberOfARows;
    int numberOfGoldColumns = numberOfBColumns;

    cl_int result;

    // code
    int sizeA = (numberOfARows * numberOfAColumns * sizeof(int));
    int sizeB = (numberOfBRows * numberOfBColumns * sizeof(int));
    int sizeC = (numberOfCRows * numberOfCColumns * sizeof(int));
    int sizeGold = (numberOfGoldRows * numberOfGoldColumns * sizeof(int));

    // host memory allocation
    hostA = (int *)malloc(sizeA);
    if (hostA == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostA Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostB = (int *)malloc(sizeB);
    if (hostB == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostB Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostC = (int *)malloc(sizeC);
    if (hostC == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostC Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    gold = (int *)malloc(sizeGold);
    if (gold == NULL)
    {
        printf("error>> Host Memory Allocation Failed For gold Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // print matrix dimensions and sizes
    printf("\n==============================================================================================\n");
    printf("+ DISPLAYING THE MEXTRIX DIMENSIONS AND SIZES +\n");
    printf("==============================================================================================\n");
    printf("- The Dimensions Of Matrix 'hostA' Are : %d x %d\n", numberOfARows, numberOfAColumns);
    printf("  Size Of Matrix 'hostA'               : %d\n\n", sizeA);

    printf("- The Dimensions Of Matrix 'hostB' Are : %d x %d\n", numberOfBRows, numberOfBColumns);
    printf("  Size Of Matrix 'hostB'               : %d\n\n", sizeB);

    printf("- The Dimensions Of Matrix 'hostC' Are : %d x %d\n", numberOfCRows, numberOfCColumns);
    printf("  Size Of Matrix 'hostC'               : %d\n\n", sizeC);

    printf("- The Dimensions Of Matrix 'gold' Are  : %d x %d\n", numberOfGoldRows, numberOfGoldColumns);
    printf("  Size Of Matrix 'gold'                : %d\n\n", sizeGold);

    // fill source matrices
    InitA(hostA, numberOfARows, numberOfAColumns);
    InitA(hostB, numberOfBRows, numberOfBColumns);

    // get OpenCL supporting platform's ID
    result = clGetPlatformIDs(1, &oclPlatformID, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clGetPlatformIDs() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // get OpenCL supporting GPU device's ID
    result = clGetDeviceIDs(oclPlatformID, CL_DEVICE_TYPE_GPU, 1, &oclComputeDeviceID, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clGetDeviceIDs() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // create OpenCL compute context
    oclContext = clCreateContext(NULL, 1, &oclComputeDeviceID, NULL, NULL, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateContext() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // create command queue
    oclCommandQueue = clCreateCommandQueue(oclContext, oclComputeDeviceID, 0, &result);
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
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(oclProgram, oclComputeDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("OpenCL Program Build Log : %s\n", buffer);
        printf("error>> clBuildProgram() Failed : %d. Terminating Now ...\n", result);

        cleanup();
        exit(EXIT_FAILURE);
    }

    // create OpenCL kernel by passing kernel function name that we used in .cl file
    oclKernel = clCreateKernel(oclProgram, "matrixMultiplyGPU", &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateKernel() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // device memory allocation
    deviceA = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeA, NULL, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateBuffer() Failed For 1st Input Array : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceB = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeB, NULL, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateBuffer() Failed For 2nd Input Array : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceC = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, sizeC, NULL, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateBuffer() Failed For Output Array : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 0th argument i.e deviceA
    result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void *)&deviceA);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 1st Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 1st argument i.e deviceB
    result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void *)&deviceB);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 2nd Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 2nd argument i.e deviceC
    result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void *)&deviceC);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 3rd Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 3rd argument i.e numberOfARows
    result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void *)&numberOfARows);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 4th Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 4th argument i.e numberOfAColumns
    result = clSetKernelArg(oclKernel, 4, sizeof(cl_int), (void *)&numberOfAColumns);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 5th Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 6th argument i.e numberOfBColumns
    result = clSetKernelArg(oclKernel, 5, sizeof(cl_int), (void *)&numberOfBColumns);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 6th Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 8th argument i.e numberOfCColumns
    result = clSetKernelArg(oclKernel, 6, sizeof(cl_int), (void *)&numberOfCColumns);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 7th Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // write above "input" device buffer to device memory
    result = clEnqueueWriteBuffer(oclCommandQueue, deviceA, CL_FALSE, 0, sizeA, hostA, 0, NULL, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clEnqueueWriteBuffer() Failed For 1st Input Device Buffer : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = clEnqueueWriteBuffer(oclCommandQueue, deviceB, CL_FALSE, 0, sizeB, hostB, 0, NULL, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clEnqueueWriteBuffer() Failed For 2nd Input Device Buffer : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // run the kernel
    size_t globalWorkSize[2];
    globalWorkSize[0] = BLOCK_WIDTH;
    globalWorkSize[1] = BLOCK_WIDTH;

    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    result = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
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

    // read back result from the device (i.e from deviceOutput) intp cpu vairiable (i.e hostOutput)
    result = clEnqueueReadBuffer(oclCommandQueue, deviceC, CL_TRUE, 0, sizeC, hostC, 0, NULL, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clEnqueueReadBuffer() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // matrix multiplication on host
    matMulCPU(hostA, hostB, gold, numberOfARows, numberOfAColumns, numberOfBColumns, numberOfCColumns);

    // comparison
    int breakValue = -1;
    bool bAccuracy = true;
    int index;
    for (index = 0; index < (numberOfARows * numberOfAColumns); index++)
    {
        float val1 = gold[index];
        float val2 = hostC[index];

        if (val1 != val2)
        {
            bAccuracy = false;
            breakValue = index;
            break;
        }
    }

    if (bAccuracy == false)
    {
        printf("Break Value = %d\n", breakValue);
    }

    char stringMessage[125];
    if (bAccuracy == false)
    {
        sprintf(stringMessage, "%s", "# Comparison Of CPU And GPU Matrix Multiplication Is Not Accurate At Array %d", breakValue);
    }
    else
    {
        sprintf(stringMessage, "%s", "# Comparison Of CPU And GPU Matrix Multiplication Is Not Accurate At Array.");
    }

    printf("\n==============================================================================================\n");
    printf("+ DISPLAYING THE RESULT OF ADDITION FROM DEVICE TO HOST +\n");
    printf("==============================================================================================\n");

    printf("- The Time Taken To Do Above Calculations On CPU = %0.6f (ms)\n", timeOnCPU);
    printf("- The Time Taken To Do Above Calculations On GPU = %0.6f (ms)\n", timeOnGPU);
    printf("%s\n", stringMessage);
    printf("==============================================================================================\n");

    // cleanup
    cleanup();

    return (0);
}

// InitA() definition
void InitA(int *data, int row, int column)
{
    // local variable declaration
    int number = 1;
    int rowIndex;
    int columnIndex;

    // code
    for (rowIndex = 0; rowIndex < row; rowIndex++)
    {
        for (columnIndex = 0; columnIndex < column; columnIndex++)
        {
            *(data + rowIndex * column + columnIndex) = number;
            number++;
        }
    }
}

// InitB() definition
void InitB(int *data, int row, int column)
{
    // local variable declaration
    int number = BLOCK_WIDTH;
    int rowIndex;
    int columnIndex;

    // code
    for (rowIndex = 0; rowIndex < row; rowIndex++)
    {
        for (columnIndex = 0; columnIndex < column; columnIndex++)
        {
            *(data + rowIndex * column + columnIndex) = number;
            number--;
        }
    }
}

// matMulCPU() definition
void matMulCPU(int *A, int *B, int *C, int iARows, int iAColumns, int iBColumns, int iCColumns)
{
    // local variable declaration
    int index;
    int column;
    int depth;

    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (index = 0; index < iARows; index++)
    {
        for (column = 0; column < iBColumns; column++)
        {
            float value = 0.0f;
            for (depth = 0; depth < iAColumns; depth++)
            {
                float a = A[index * iAColumns + depth];
                float b = B[depth * iCColumns + column];

                value += a * b;
            }

            C[index * iCColumns + column] = value;
        }
    }

    // stop timer
    sdkStopTimer(&timer);
    timeOnCPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;
}

// cleanup() definition
void cleanup(void)
{
    // code
    if (oclSourceCode)
    {
        free((void *)oclSourceCode);
        oclSourceCode = NULL;
    }

    if (deviceC)
    {
        clReleaseMemObject(deviceC);
        deviceC = NULL;
    }

    if (deviceB)
    {
        clReleaseMemObject(deviceB);
        deviceB = NULL;
    }

    if (deviceA)
    {
        clReleaseMemObject(deviceA);
        deviceA = NULL;
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

    if (gold)
    {
        free(gold);
        gold = NULL;
    }

    if (hostC)
    {
        free(hostC);
        hostC = NULL;
    }

    if (hostB)
    {
        free(hostB);
        hostB = NULL;
    }

    if (hostA)
    {
        free(hostA);
        hostA = NULL;
    }
}
