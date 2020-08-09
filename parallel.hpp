#ifndef PARALLEL_HPP
#define PARALLEL_HPP

void solvePar(int rows, int cols, int iterations, double td, double h, double ** matrix, const char * kernelFileName);
void heatTransferOclCall(int rows, int cols, int iterations, double * flatMatrix1, double * flatMatrix2, double td, double h, const char* kernelSource);

/*
* Return 1d equivalent of inArray (row-major)
*/
double* flatten(double** inArray, int rows, int cols);

/*
* Return 2d equivalent of inArray (row-major)
*/
double** return2d(double* inArray, int rows, int cols);

#endif