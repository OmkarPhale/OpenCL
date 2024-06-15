// headers
#include <stdio.h>
#include <stdlib.h> //exit()
// #include <string.h>                      //strlen()

#include <CL/opencl.h> //standard OpenCL header

// global variables
const int iNumberOfArrayElements = 5;

cl_platform_id oclPlatformID;
cl_device_id oclDeviceID;

cl_context oclContext;
cl_command_queue oclCommandQueue;

cl_program oclProgram;
cl_kernel oclKernel;

float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;

cl_mem deviceInput1 = NULL;
cl_mem deviceInput2 = NULL;
cl_mem deviceOutput = NULL;

// OpenCL kernel
const char *openclSourceCode =
    "__kernel void vecAddGPU(__global float *input1, __global float *input2, __global float *output, int length)        \n"
    "{                                                                                                                  \n"
    "   int index = get_global_id(0);                                                                                   \n"
    "   if(index < length)                                                                                              \n"
    "   {                                                                                                               \n"
    "       output[index] = input1[index] + input2[index];                                                              \n"
    "   }                                                                                                               \n"
    "}                                                                                                                  \n";

// main() definition
int main(void)
{
    // local function declaration
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

    // filling values into host arrays
    hostInput1[0] = 1001.0f;
    hostInput1[1] = 1002.0f;
    hostInput1[2] = 1003.0f;
    hostInput1[3] = 1004.0f;
    hostInput1[4] = 1005.0f;

    hostInput2[0] = 2001.0f;
    hostInput2[1] = 2002.0f;
    hostInput2[2] = 2003.0f;
    hostInput2[3] = 2004.0f;
    hostInput2[4] = 2005.0f;

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
    oclProgram = clCreateProgramWithSource(oclContext, 1, (const char **)&openclSourceCode, NULL, &result);
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
        clGetProgramBuildInfo(oclProgram, oclDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), &buffer, &len);
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

    // device memory allocation
    size = iNumberOfArrayElements * sizeof(cl_float);

    deviceInput1 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateBuffer() Failed For 1st Array Input : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceInput2 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
    if (result != CL_SUCCESS)
    {
        printf("error>> clCreateBuffer() Failed For 2nd Array Input : %d. Terminating Now ...\n", result);
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

    // set 0 based 0th argument i.e. deviceInput1
    result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void *)&deviceInput1);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 1st Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 1st argument i.e. deviceInput2
    result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void *)&deviceInput2);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 2nd Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 2nd argument i.e. deviceOutput
    result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void *)&deviceOutput);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 3rd Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 3rd argument i.e. length
    result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void *)&iNumberOfArrayElements);
    if (result != CL_SUCCESS)
    {
        printf("error>> clSetKernelArg() Failed For 4th Argument : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // write above 'input' device buffer to device memory
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
    size_t global_size = 5; // 1-D 5 element array operation
    result = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clEnqueueNDRangeKernel() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // finish OpenCL command queue
    clFinish(oclCommandQueue);

    // read back result from the device (i.e from deviceOutput) into cpu variable (i.e hostOutput)
    result = clEnqueueReadBuffer(oclCommandQueue, deviceOutput, CL_TRUE, 0, size, hostOutput, 0, NULL, NULL);
    if (result != CL_SUCCESS)
    {
        printf("error>> clEnqueueReadBuffer() Failed : %d. Terminating Now ...\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // display results
    int index;
    printf("\n==================================================================\n");
    printf("+ DISPLAYING THE RESULT OF ADDITION FROM DEVICE TO HOST +\n");
    printf("==================================================================\n");

    for (index = 0; index < iNumberOfArrayElements; index++)
    {
        printf("- Array Index '%d' >> %f + %f = %f\n", index, hostInput1[index], hostInput2[index], hostOutput[index]);
    }
    printf("==================================================================\n");

    // cleanup
    cleanup();

    return (0);
}

void cleanup(void)
{
    // code
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

    if (hostOutput)
    {
        free(hostOutput);
        hostOutput = NULL;
    }

    if (hostInput1)
    {
        free(hostInput1);
        hostInput1 = NULL;
    }

    if (hostInput2)
    {
        free(hostInput2);
        hostInput2 = NULL;
    }
}
