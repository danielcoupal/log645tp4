__kernel void addKernel(__global double * oldMatrix, 
                        double td, 
                        double h, 
                        __global double * newMatrix, 
                        int rows,
                        int cols,
                        int iterations) {
    
    int id = get_global_id(0);
    int x = id % cols;
    int y = id / cols;

    for (int i = 0; i < iterations; ++i) {
        if(x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
            double c = oldMatrix[id];
            double b = oldMatrix[id - cols];
            double t = oldMatrix[id + cols];
            double l = oldMatrix[id - 1];
            double r = oldMatrix[id + 1];

            double k = td / (h * h);
        
            newMatrix[id] = c * (1.0 - 4.0 * k) + (t + b + l + r) * (k);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        oldMatrix[id] = newMatrix[id];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}