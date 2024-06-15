//Headers
#include <stdio.h>
#include <stdlib.h>

#include <CL\OpenCL.h>

//main()
int main(void)
{
    //function declarations
    void printOpenCLDeviceProperties(void);
    
    //code
    printOpenCLDeviceProperties();
}

//printOpenCLDeviceProperties()
void printOpenCLDeviceProperties(void)
{
    //variable declarations
    cl_platform_id *ocl_platform_ids = NULL;
    cl_uint         platform_count;
    cl_device_id   *ocl_device_ids = NULL;
    cl_uint         dev_count;
    cl_int          ret_ocl;
    
    cl_device_type  oclDeviceType;
    cl_uint         oclDeviceInfo, clockFrequency;
    cl_bool         errorCorrectionSupport;
    
    cl_ulong        memSize, maxMemAllocSize;
    
    cl_uint         computeUnits;
    
    size_t          workGroupSize, workItemDims;
    size_t          workItemSize[3], szMaxDims[5];
    
    char  oclPlatformInfo[512], oclPlatformExtension[2048];;
    int   i, j;
    
    //code
    printf("OpenCL INFORMATION : \n");
    printf("===============================================================\n");
    
        //get platform count
    ret_ocl = clGetPlatformIDs(0, NULL, &platform_count);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error: clGetPlatformIDs() failed. Exiting Now...\n");
        exit(EXIT_FAILURE);
    }
    if(platform_count < 1)
    {
        printf("No Platform Support OpenCL\n");
        exit(EXIT_FAILURE);
    }
    
    
        //allocate memory for platform id
    ocl_platform_ids = (cl_platform_id *) malloc(platform_count * sizeof(cl_platform_id));
    if(ocl_platform_ids == NULL)
    {
        printf("Cannot allocate memory\n");
        exit(EXIT_FAILURE);
    }
    
        //get platform ids
    ret_ocl = clGetPlatformIDs(platform_count, ocl_platform_ids, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error: clGetPlatformIDs() Failed. Exiting Noew...\n");
        exit(EXIT_FAILURE);
    }
    
    
    for(i = 0; i < platform_count; i++)
    {        
            //Get All Device count
        ret_ocl = clGetDeviceIDs(ocl_platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &dev_count);
        if(ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error: clGetDeviceIDs() Failed. Exiting Noew...\n");
            exit(EXIT_FAILURE);
        }
        
        else if(dev_count < 1)
        {
            printf("No Device Support OpenCL\n");
            continue;
        }
        
        printf("\n\t/******** PLAFORM INFORMATION *********/\n");
            //get platform profile
        clGetPlatformInfo(ocl_platform_ids[i], CL_PLATFORM_PROFILE, sizeof(oclPlatformInfo), &oclPlatformInfo, NULL);
        printf("Platform Profile: \t %s\n", oclPlatformInfo);
        
            //get platform name
        clGetPlatformInfo(ocl_platform_ids[i], CL_PLATFORM_NAME, sizeof(oclPlatformInfo), &oclPlatformInfo, NULL);
        printf("Platform Name:    \t %s\n", oclPlatformInfo);
        
            //get platform version
        clGetPlatformInfo(ocl_platform_ids[i], CL_PLATFORM_VERSION, sizeof(oclPlatformInfo), &oclPlatformInfo, NULL);
        printf("Platform Version: \t %s\n", oclPlatformInfo);
        
            //get platform vendor
        clGetPlatformInfo(ocl_platform_ids[i], CL_PLATFORM_VENDOR, sizeof(oclPlatformInfo), &oclPlatformInfo, NULL);
        printf("Platform Vendor:  \t %s\n", oclPlatformInfo);
        
            //get platform extension
        clGetPlatformInfo(ocl_platform_ids[i], CL_PLATFORM_EXTENSIONS, sizeof(oclPlatformExtension), &oclPlatformExtension, NULL);
        printf("Platform Supported Extension : \n\t\t");
                
        int k = 0;
        while(oclPlatformExtension[k] != '\0')
        {
            if(oclPlatformExtension[k] == ' ')
                printf("\n\t\t");
            else
                printf("%c", oclPlatformExtension[k]);
            k++;
        }
        printf("\n\n");
        
            //number of Device supported
        printf("Total Number Of OpenCL Supporting Device/Device : %d\n", dev_count);
                
        
            //Allocate memory to hold device ids
        ocl_device_ids = (cl_device_id *) malloc(dev_count * sizeof(cl_device_id));
        if(ocl_device_ids == NULL)
        {
            printf("Cannot allocate memory for device.\n");
            free(ocl_platform_ids);
            exit(EXIT_FAILURE);
        }
        
            //get device ids
        ret_ocl = clGetDeviceIDs(ocl_platform_ids[i], CL_DEVICE_TYPE_ALL, dev_count, ocl_device_ids, NULL);
        if(ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error: clGetDeviceIDs() Failed. Exiting Now...\n");
            free(ocl_device_ids);
            free(ocl_platform_ids);
            exit(EXIT_FAILURE);
        }
        
        char oclDevProp[1024];
        
        for(j = 0; j < dev_count; j++)
        {
            printf("\n\t\t/******** DEVICE INFORMATION *********/\n");
            printf("\t--------- General Information -------------\n");
            printf("\tDevice Numer: %d\n", j);
            
                //get device type
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &oclDeviceType, NULL);
            printf("\tDevice Type : ");
            
            switch(oclDeviceType)
            {
                case CL_DEVICE_TYPE_CPU:
                    printf("CL_DEVICE_TYPE_CPU\n");
                break;
                
                case CL_DEVICE_TYPE_GPU:
                    printf("CL_DEVICE_TYPE_GPU\n");
                break;
                
                case CL_DEVICE_TYPE_ACCELERATOR:
                    printf("CL_DEVICE_TYPE_ACCELERATOR\n");
                break;
                
                case CL_DEVICE_TYPE_DEFAULT:
                    printf("CL_DEVICE_TYPE_DEFAULT\n");
                break;
                
                case CL_DEVICE_TYPE_CUSTOM:
                    printf("CL_DEVICE_TYPE_CUSTOM\n");
                break;
                
                default:
                    printf("UNKNOWN\n");
                break;
            }
            
                //get device name
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_NAME, sizeof(oclDevProp), &oclDevProp, NULL);
            printf("\tDevice Name : %s\n", oclDevProp);
            
                //get device vendor id
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &oclDeviceInfo, NULL);
            printf("\tDevice Vendor ID: %u\n", oclDeviceInfo);
            
                //get device vendor
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_VENDOR, sizeof(oclDevProp), &oclDevProp, NULL);
            printf("\tDevice Vendor : %s\n", oclDevProp);
            
                //get driver version
            clGetDeviceInfo(ocl_device_ids[j], CL_DRIVER_VERSION, sizeof(oclDevProp), &oclDevProp, NULL);
            printf("\tDevice Driver Version : %s\n", oclDevProp);
            
                //get device version
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_VERSION, sizeof(oclDevProp), &oclDevProp, NULL);
            printf("\tDevice OpenCL Version : %s\n", oclDevProp);
            
                //get clock frequency
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFrequency), &clockFrequency, NULL);
            printf("\tDevice Clock Rate : %u\n", clockFrequency);
            
                //check device support error correction
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(errorCorrectionSupport), &errorCorrectionSupport, NULL);
            printf("\tDevice Error Correction Code (ECC) Support : %s\n", errorCorrectionSupport == CL_TRUE ? "Yes" : "No");
            
            
            printf("\n\n\t--------- Memory Information -------------\n");
                //get device global memory
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, NULL);
            printf("\tDevice Global Memory : %llu Bytes (%f MB)\n", memSize, ((float)memSize/1024.0f)/1024.0f);
            
                //get device local memory
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(memSize), &memSize, NULL);
            printf("\tDevice Local Memory : %llu Bytes (%f MB)\n", memSize, ((float)memSize/1024.0f)/1024.0f);
            
                //get constant buffer size
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(memSize), &memSize, NULL);
            printf("\tDevice Constant Buffer Size : %llu Bytes (%f MB)\n", memSize, ((float)memSize/1024.0f)/1024.0f);
            
                //get Maximum Memory Allocation Size
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAllocSize), &maxMemAllocSize, NULL);
            printf("\tDevice Memory Allocation Size : %llu Bytes (%f MB)\n", maxMemAllocSize, ((float)maxMemAllocSize/1024.0f)/1024.0f);
            
            
            printf("\n\n\t--------- Compute Information -------------\n");
                //Max Compute Units
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
            printf("\tDevice Number of Parallel Processors Cores : %u\n", computeUnits);
            
                //max work size
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workGroupSize), &workGroupSize, NULL);
            printf("\tDevice Work Group Size : %u\n", (unsigned int)workGroupSize);
            
                //work dimensions
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workItemDims), &workItemDims, NULL);
            printf("\tDevice Work Item Dimensions : %u\n", (unsigned int)workItemDims);
            
                //Max work item sizes
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workItemSize), &workItemSize, NULL);
            printf("\tDevice Work Item Sizes : %u/%u/%u\n", (unsigned int) workItemSize[0], (unsigned int) workItemSize[1], (unsigned int) workItemSize[2]);
            

            printf("\n\n\t--------- Image Support Information -------------\n");
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
            printf("\tDevice Supported 2-D Image W x H : %u x %u\n", (unsigned int) szMaxDims[0], (unsigned int) szMaxDims[1]);
            
            
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
            clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
            printf("\tDevice Supported 3-D Image W x H x D : %u x %u x %u\n", (unsigned int) szMaxDims[2], (unsigned int) szMaxDims[3], (unsigned int) szMaxDims[4]);
            
            printf("\n");
            
            free(ocl_device_ids);
        }
    }    
    free(ocl_platform_ids);
}
