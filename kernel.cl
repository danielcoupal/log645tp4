__kernel void addKernel(__global double ** oldMatrix, __global double td, __global double h, __global double ** newMatrix, int rows, int cols) {
    
    int id = get_global_id(0);
    int x = id / cols + 1;
    int y = id % cols + 1;
    
    if(id < elements) {
        double c = oldMatrix[x]    [y];
        double b = oldMatrix[x]    [y-1];
        double t = oldMatrix[x]    [y+1];
        double l = oldMatrix[x-1]  [y];
        double r = oldMatrix[x+1]  [y];

        double k = td / (h * h);
    
        newMatrix[x][y] = c * (1.0 - 4.0 * k) + (t + b + l + r) * (k);
    }
}