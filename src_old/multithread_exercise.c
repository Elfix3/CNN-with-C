#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define WIDTH  1200
#define HEIGHT 800
#define MAX_ITER 2000

int mandelbrot_iter(double cx, double cy) {
    double zx = 0.0, zy = 0.0;
    int iter = 0;
    while (zx*zx + zy*zy < 4.0 && iter < MAX_ITER) {
        double tmp = zx*zx - zy*zy + cx;
        zy = 2.0*zx*zy + cy;
        zx = tmp;
        iter++;
    }
    return iter;
}

int main() {
    int *image = malloc(WIDTH * HEIGHT * sizeof(int));

    double start = omp_get_wtime();
    
    #pragma omp parallel for schedule(dynamic, 1)
    
    for (int py = 0; py < HEIGHT; py++) {
        for (int px = 0; px < WIDTH; px++) {
            double cx = (px - WIDTH/2.0)  * 4.0 / WIDTH;
            double cy = (py - HEIGHT/2.0) * 4.0 / HEIGHT;
            image[py * WIDTH + px] = mandelbrot_iter(cx, cy);
        }
    }

    double end = omp_get_wtime();
    printf("Temps : %f secondes\n", end - start);

    free(image);
    return 0;
}