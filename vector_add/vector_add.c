// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// OpenCL kernel to perform an element–wise addition
const char* programSource =
"__kernel                                       \n"
"void vecadd(__global int *A,                   \n"
" __global int *B,                              \n"
" __global int *C)                              \n"
"{                                              \n"
"                                               \n"
" // Get the work-item's unique ID              \n"
" int idx = get_global_id(0);                   \n"
"                                               \n"
" // Add the corresponding locations of         \n"
" // 'A' and 'B', and store the resuult in 'C'. \n"
" C[idx] = A[idx] + B[idx];                     \n"
"}                                              \n"
;

void check(cl_int status) {
  if (status < 0) {
    printf("ERROR. Exiting...\n");
    exit(-1);
  }
}

int main() {
  // This code executes on the OpenCL host

  // Elements in each array
  const int elements = 2048;

  // Compute the size of the data
  size_t datasize = sizeof(int)*elements;

  // Allocate space for input/output host data
  int *A = (int*)malloc(datasize); // Input array
  int *B = (int*)malloc(datasize); // Input array
  int *C = (int*)malloc(datasize); // Output array

  // Initialize the input data
  int i;
  for (i = 0; i < elements; i++) {
    A[i] = 1;
    B[i] = 1;
  }

  // Use this to check the output of each API call
  cl_int status;

  // Get the first platform
  cl_platform_id platform;
  status = clGetPlatformIDs(1, &platform, NULL);

  // Get the first device
  cl_device_id device;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

  // Create a context and associate it with the device
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);

  // Create a command−queue and associate it with the device
  cl_command_queue cmdQueue = clCreateCommandQueueWithPropertiesAPPLE(context, device , 0, &status);

  // Allocate two input buffers and one output buffer for the three vectors in the vector addition
  cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
  cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
  cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);

  // Write data from the input arrays to the buffers
  status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE, 0, datasize , A, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 0, datasize , B, 0, NULL, NULL);

  // Create a program with source code
  printf("Create a program with source code\n");
  cl_program program = clCreateProgramWithSource(context , 1, (const char**)&programSource, NULL, &status);
  check(status);

  // Build (compile) the program for the device
  printf("Build (compile) the program for the device\n");
  status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  check(status);

  // Create the vector addition kernel
  printf("Create the vector addition kernel\n");
  cl_kernel kernel = clCreateKernel(program, "vecadd", &status);
  check(status);

  // Set the kernel arguments
  printf("Set the kernel arguments\n");
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
  check(status);

  // Define an index space of work−items for execution.
  // A work−group size is not required, but can be used.
  size_t indexSpaceSize [1] , workGroupSize [1];

  // There are ‘elements’ work–items
  indexSpaceSize [0] = elements;
  workGroupSize [0] = 256;

  // Execute the kernel
  printf("Execute the kernel\n");
  status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL);
  check(status);

  printf("The kernel has finished execution on the device\n");
  printf("Read the device output buffer to the host output array\n");

  // Read the device output buffer to the host output array
  status = clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);

  // Free OpenCL resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(cmdQueue);
  clReleaseMemObject(bufA);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufC);
  clReleaseContext(context);

  for (int i = 0; i < elements; i++) {
    printf("%d", C[i]);
    printf("\n");
  }

  // Free host resources
  free(A);
  free(B);
  free(C);

  return 0;
}