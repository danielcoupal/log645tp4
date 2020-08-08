#ifndef PARALLEL_HPP
#define PARALLEL_HPP

void solvePar(int rows, int cols, int iterations, double td, double h, double ** matrix, const char * kernelFileName);
void heatMapTimeJump(int rows, int cols, int iterations, double* oldMatrix, double td, double h, double* newMatrix, const char* kernelSource);
double* flatten(double** inArray, int rows, int cols);
double** return2d(double* inArray, int rows, int cols);

#endif