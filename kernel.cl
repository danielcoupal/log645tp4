__kernel void addKernel(__global double * oldMatrix, 
                        __global double td, 
                        __global double h, 
                        __global double * newMatrix, 
                        __global int rows,
                        __global int cols) {
    
    int id = get_global_id(0) + cols;
    int x = id % cols;
    int y = id / cols;

    if(x > 0 && id < (cols - 1)) {
        if (y > 0 && y < (rows - 1)) {
            double c = oldMatrix[id];
            double b = oldMatrix[id - cols];
            double t = oldMatrix[id + cols];
            double l = oldMatrix[id - 1];
            double r = oldMatrix[id + 1];

            double k = td / (h * h);
    
            newMatrix[id] = c * (1.0 - 4.0 * k) + (t + b + l + r) * (k);
        }
    }
}