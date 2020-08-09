#define errCheck(code) { errorCheck(code, __FILE__, __LINE__); }

#include <iostream>
#include <fstream>
#include <CL/opencl.h>

#include "windows.h"
#include "matrix.hpp"
#include "parallel.hpp"

char * readFile(const char * fileName);

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

	int elements = rows * cols;
	double* flatMatrix1 = flatten(matrix, rows, cols);
	double* flatMatrix2 = new double[elements];

	char* kernelSource = readFile(kernelFileName);

	heatMapTimeJump(rows, cols, iterations, flatMatrix1, flatMatrix2, td, h, kernelSource);

	double** newTallMatrix = return2d(flatMatrix1, rows, cols);
	memcpy(matrix, newTallMatrix, elements * sizeof(double));

	delete[] flatMatrix1;
	delete[] flatMatrix2;
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

void heatMapTimeJump(int rows, int cols, int iterations, double* flatMatrix1, double* flatMatrix2, double td, double h, const char* kernelSource) {
	int matrix_mem_size = rows * cols * sizeof(double);
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
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
	errCheck(err);

	errCheck(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

	// Setup an execution kernel from the source program.
	cl_kernel kernel = clCreateKernel(program, "addKernel", &err);
	errCheck(err);

	// Create device buffers.
	cl_mem dev_old_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_mem_size, NULL, &err);
	errCheck(err);
	cl_mem dev_new_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_mem_size, NULL, &err);
	errCheck(err);

	errCheck(clEnqueueWriteBuffer(queue, dev_old_matrix, CL_TRUE, 0, matrix_mem_size, flatMatrix1, 0, NULL, NULL));
	errCheck(clEnqueueWriteBuffer(queue, dev_new_matrix, CL_TRUE, 0, matrix_mem_size, flatMatrix2, 0, NULL, NULL));

	// Setup function arguments.
	errCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_old_matrix));
	errCheck(clSetKernelArg(kernel, 1, sizeof(double), &td));
	errCheck(clSetKernelArg(kernel, 2, sizeof(double), &h));
	errCheck(clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev_new_matrix));
	errCheck(clSetKernelArg(kernel, 4, sizeof(int), &rows));
	errCheck(clSetKernelArg(kernel, 5, sizeof(int), &cols));
	errCheck(clSetKernelArg(kernel, 6, sizeof(int), &iterations));

	// Execute the kernel.
	size_t localSize = 1;// (size_t)cols;
	size_t globalSize = (size_t)rows * (size_t)cols;
	errCheck(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL));

	// Wait for the kernel the terminate.
	errCheck(clFinish(queue));

	// Write device data in our output buffer.
	errCheck(clEnqueueReadBuffer(queue, dev_new_matrix, CL_TRUE, 0, matrix_mem_size, flatMatrix2, 0, NULL, NULL));
	
	// Clear memory.
	errCheck(clReleaseMemObject(dev_old_matrix));
	errCheck(clReleaseMemObject(dev_new_matrix));
	errCheck(clReleaseKernel(kernel));
	errCheck(clReleaseProgram(program));
	errCheck(clReleaseCommandQueue(queue));
	errCheck(clReleaseContext(context));
}

double* flatten(double** inArray, int rows, int cols) {

	int elements = rows * cols;
	double* flatArray = new double[elements];

	int rowOffset = 0;


	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			flatArray[rowOffset + col] = inArray[col][row];
		}

		rowOffset += cols;
	}

	return flatArray;
}

double** return2d(double* inArray, int rows, int cols) {
	double** tallArray = allocateMatrix(rows, cols);

	int rowOffset = 0;

	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			tallArray[col][row] = inArray[rowOffset + col];
		}

		rowOffset += cols;
	}

	return tallArray;
}

