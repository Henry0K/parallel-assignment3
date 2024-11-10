#include <stdio.h>
#include <time.h>
#include <omp.h>


/*
 * OpenMP Parallelization Strategy:
 * 
 * 1. Row-wise Chunking:
 *    - Each thread processes one complete row (chunk_size=1) of the image at a time
 *    - This approach minimizes false sharing as threads write to separate cache lines
 *    - Rows are independent, eliminating need for synchronization between threads
 * 
 * 2. Dynamic Scheduling:
 *    - Used because computation time varies significantly between rows
 *    - Rows intersecting the Mandelbrot set require more iterations
 *    - Dynamic scheduling allows faster threads to pick up new rows immediately
 *    - Reduces thread idle time compared to static scheduling
 */

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

// Function to calculate whether a point is in the Mandelbrot set
int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));
    
    return iter;
}

// Function to save the result as a PGM image
void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg;
    int temp;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n");  // Writing Magic Number to the File
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);  // Writing Width and Height
    fprintf(pgmimg, "255\n");  // Writing the maximum gray value
    
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            temp = image[i][j];
            fprintf(pgmimg, "%d ", temp);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main() {
    int image[HEIGHT][WIDTH];
    double AVG = 0;
    int N = 10;  // number of trials
    double total_time[N];
    struct complex c;
    
    // Run N trials
    for (int k = 0; k < N; k++) {
        double start_time = omp_get_wtime();
        
        /*
         * schedule(dynamic, 1)
         * - Dynamic: Threads request new chunks when they finish their current chunk
         * - 1: Each chunk is one row (better load balancing for Mandelbrot)
         * 
         * private(c)
         * - Each thread needs its own copy of c to avoid race conditions
         * - Complex coordinates are calculated independently for each pixel
         * 
         * shared(image)
         * - All threads write to different parts of image array
         * - No race conditions as threads write to different rows
         */
        #pragma omp parallel for schedule(dynamic, 1) private(c) shared(image)
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                // Map pixel coordinates to the complex plane
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[i][j] = cal_pixel(c);
            }
        }
        
        double end_time = omp_get_wtime();
        total_time[k] = end_time - start_time;
        
        printf("Execution time of trial [%d]: %f seconds\n", k, total_time[k]);
        AVG += total_time[k];
    }

    save_pgm("mandelbrot_parallel.pgm", image);
    printf("The average execution time of %d trials is: %f ms\n", N, (AVG/N)*1000);
    
    return 0;
}