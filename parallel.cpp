#define errCheck(code) { errorCheck(code, __FILE__, __LINE__); }

#include <iostream>
#include <fstream>
#include <CL/opencl.h>

#include "windows.h"
#include "matrix.hpp"
#include "parallel.hpp"

char * readFile(const char * fileName);
void addWithOpenCl(const int * a, const int * b, int * c, int elements, const char * kernelSource);

using std::cout;
using std::flush;
using std::endl;

inline void errorCheck(cl_int code, const char* file, int line) {
	if (CL_SUCCESS != code) {
		std::cout << "[" << file << ", line " << line << "]" << std::flush;
		std::cout << " OpenCL error code <" << code << "> received." << std::endl << std::flush;
		Sleep(3000);
		exit(EXIT_FAILURE);
	}
}

void solvePar(int rows, int cols, int iterations, double td, double h, double ** matrix, const char * kernelFileName) {

	double** newMatrix = allocateMatrix(rows, cols);

	char* kernelSource = readFile(kernelFileName);

	for (int k = 0; k < iterations; ++k) {
		heatMapTimeJump(rows - 2, cols - 2, matrix, td, h, newMatrix, kernelSource);
		memcpy(matrix, newMatrix, sizeof(double) * rows * cols);
	}
	
	//cout << "Do OpenCl related stuff here!" << endl << flush;

	// Example.
	/*const int elements = 5;
	const int a[elements] = { 1, 2, 3, 4, 5 };
	const int b[elements] = { 5, 4, 3, 2, 1 };
	int c[elements] = { 0 };

	char * kernelSource = readFile(kernelFileName);
	printf("%s\n", kernelSource);
	
	Sleep(3000);

	addWithOpenCl(a, b, c, elements, kernelSource);*/
}

char * readFile(const char * fileName) {
	int length;

	std::ifstream file(fileName, std::ifstream::in | std::ios::binary);
	file.seekg(0, std::ios::end);
	length = file.tellg();
	file.seekg(0, std::ios::beg);

	char * buffer = (char *) malloc(length + 1);
	file.read(buffer, length);
	file.close();

	buffer[length] = '\0';

	return buffer;
}

void heatMapTimeJump(int rows, int cols, double ** oldMatrix, double td, double h, double ** newMatrix, const char * kernelSource) {
	int bytes = rows * cols * sizeof(double);
	cl_int err = CL_SUCCESS;

	std::cout << "CL program create start";

	// Get execution platform.
	cl_platform_id platform;
	errCheck(clGetPlatformIDs(1, &platform, NULL));

	// Get available gpus on platform.
	cl_device_id device_id;
	errCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL));

	// Create an execution context.
	cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	errCheck(err);

	// Create the command queue.
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
	errCheck(err);

	// Compile the source program.
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
	errCheck(err);

	errCheck(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

	std::cout << "Program OK";

	// Setup an execution kernel from the source program.
	cl_kernel kernel = clCreateKernel(program, "addKernel", &err);
	errCheck(err);

	std::cout << "Kernel OK";

	// Create device buffers.
	cl_mem dev_old_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
	errCheck(err);
	cl_mem dev_new_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
	errCheck(err);

	errCheck(clEnqueueWriteBuffer(queue, dev_old_matrix, CL_TRUE, 0, bytes, oldMatrix, 0, NULL, NULL));

	// Setup function arguments.
	errCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_old_matrix));
	errCheck(clSetKernelArg(kernel, 1, sizeof(double), &td));
	errCheck(clSetKernelArg(kernel, 2, sizeof(double), &h));
	errCheck(clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev_new_matrix));
	errCheck(clSetKernelArg(kernel, 4, sizeof(int), &rows));
	errCheck(clSetKernelArg(kernel, 5, sizeof(int), &cols));

	// Execute the kernel.
	size_t localSize = rows * cols;
	size_t globalSize = rows * cols;
	errCheck(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL));

	// Wait for the kernel the terminate.
	errCheck(clFinish(queue));

	// Write device data in our output buffer.
	errCheck(clEnqueueReadBuffer(queue, dev_new_matrix, CL_TRUE, 0, bytes, newMatrix, 0, NULL, NULL));
	
	// Clear memory.
	errCheck(clReleaseMemObject(dev_old_matrix));
	errCheck(clReleaseMemObject(dev_new_matrix));
	errCheck(clReleaseKernel(kernel));
	errCheck(clReleaseProgram(program));
	errCheck(clReleaseCommandQueue(queue));
	errCheck(clReleaseContext(context));
}

void addWithOpenCl(const int * a, const int * b, int * c, int elements, const char * kernelSource) {
	int bytes = elements * sizeof(int);
	cl_int err = CL_SUCCESS;

	// Get execution platform.
	cl_platform_id platform;
	errCheck(clGetPlatformIDs(1, &platform, NULL));

	// Get available gpus on platform.
	cl_device_id device_id;
	errCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL));

	// Create an execution context.
	cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	errCheck(err);

	// Create the command queue.
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
	errCheck(err);

	// Compile the source program.
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	errCheck(err);

	errCheck(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

	// Setup an execution kernel from the source program.
	cl_kernel kernel = clCreateKernel(program, "addKernel", &err);
	errCheck(err);

	// Create device buffers.
	cl_mem dev_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
	errCheck(err);
	cl_mem dev_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
	errCheck(err);
	cl_mem dev_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
	errCheck(err);

	// Write host data to the device.
	errCheck(clEnqueueWriteBuffer(queue, dev_a, CL_TRUE, 0, bytes, a, 0, NULL, NULL));
	errCheck(clEnqueueWriteBuffer(queue, dev_b, CL_TRUE, 0, bytes, b, 0, NULL, NULL));

	// Setup function arguments.
	errCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_a));
	errCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev_b));
	errCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev_c));
	errCheck(clSetKernelArg(kernel, 3, sizeof(int), &elements));

	// Execute the kernel.
	size_t localSize = elements;
	size_t globalSize = elements;
	errCheck(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL));

	// Wait for the kernel the terminate.
	errCheck(clFinish(queue));

	// Write device data in our output buffer.
	errCheck(clEnqueueReadBuffer(queue, dev_c, CL_TRUE, 0, bytes, c, 0, NULL, NULL));

	// Clear memory.
	errCheck(clReleaseMemObject(dev_a));
	errCheck(clReleaseMemObject(dev_b));
	errCheck(clReleaseMemObject(dev_c));
	errCheck(clReleaseKernel(kernel));
	errCheck(clReleaseProgram(program));
	errCheck(clReleaseCommandQueue(queue));
	errCheck(clReleaseContext(context));

	cout << "c = { " << c[0] << flush;
	for (int i = 1; i < elements; i++) {
		cout << ", " << c[i] << flush;
	}

	cout << " }" << endl << flush;
}