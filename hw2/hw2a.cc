#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <png.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>

int num_threads;
int64_t* image;
int width;
int height;
int iters;
double left;
double right;
double lower;
double upper;
double y_scale, x_scale;

void write_png(const char* filename, int iters, int width, int height, const int64_t* buffer) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int64_t p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    // free(row);
    png_write_end(png_ptr, NULL);
    // png_destroy_write_struct(&png_ptr, &info_ptr);
    // fclose(fp);
}

__m512d v_2 = _mm512_set1_pd(2.0);
__m512d v_4 = _mm512_set1_pd(4.0);
// __m512d v_i_pad = _mm512_set_pd(7.0,6.0,5.0,4.0,3.0,2.0,1.0,0.0);
__m512i v_1 = _mm512_set1_epi64(1);
const __m512d v_increment = _mm512_set1_pd(8.0);

void* mandelbrot(void* arg) {
    int id = *(int*)arg;
    __m512i v_iter = _mm512_set1_epi64(iters);
    for (int j = id; j < height; j += num_threads) {
        double y0 = j * y_scale + lower;
        __m512d v_y0 = _mm512_set1_pd(y0);
        __m512d v_i = _mm512_set_pd(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
        for (int i = 0; i < width; i += 8) {
            if (i + 7 < width) {
                // Process 8 pixels at once
                __m512d v_x0 = _mm512_fmadd_pd(v_i, _mm512_set1_pd(x_scale), _mm512_set1_pd(left));

                __m512d v_x = _mm512_setzero_pd();
                __m512d v_y = _mm512_setzero_pd();
                __m512d v_length_squared = _mm512_setzero_pd();
                __m512i v_repeat = _mm512_setzero_si512();

                while (1) {
                    // Compute mask of elements where length_squared < 4
                    __mmask8 cmp_mask = _mm512_cmp_pd_mask(v_length_squared, v_4, _CMP_LT_OS);

                    // If no elements are active, break
                    if (cmp_mask == 0)
                        break;

                    // Increment repeats where cmp_mask is set
                    v_repeat = _mm512_mask_add_epi64(v_repeat, cmp_mask, v_repeat, v_1);

                    // Break if repeats >= iters
                    __mmask8 repeats_mask = _mm512_cmp_epi64_mask(v_repeat, v_iter, _MM_CMPINT_LT);

                    // Update cmp_mask
                    cmp_mask = _mm512_kand(cmp_mask, repeats_mask);

                    if (cmp_mask == 0)
                        break;

                    // Compute temp = v_x * v_x - v_y * v_y + v_x0
                    __m512d v_x_sq = _mm512_mul_pd(v_x, v_x);
                    __m512d v_y_sq = _mm512_mul_pd(v_y, v_y);
 
                    __m512d temp = _mm512_add_pd(_mm512_sub_pd(v_x_sq, v_y_sq), v_x0);

                    // Compute v_y = 2 * v_x * v_y + v_y0
                    v_y = _mm512_fmadd_pd(_mm512_mul_pd(v_x, v_y), v_2, v_y0);

                    // Update v_x = temp
                    v_x = temp;

                    // Compute v_length_squared = v_x * v_x + v_y * v_y
                    v_length_squared = _mm512_fmadd_pd(v_x, v_x, _mm512_mul_pd(v_y, v_y));
                }

                // Convert v_repeat to 32-bit integers
                // __m256i v_repeat32 = _mm512_cvtepi64_epi32(v_repeat);

                // Store v_repeat32 into image buffer
                _mm512_storeu_si512((__m512i*)(image + j * width + i), v_repeat);
                v_i = _mm512_add_pd(v_i, v_increment);
            } else {
                // Process remaining pixels one by one
                for (int k = i; k < width; ++k) {
                    double x0 = k * x_scale + left;
                    int repeats = 0;
                    double x = 0;
                    double y = 0;
                    double length_squared = 0;
                    while (repeats < iters && length_squared < 4) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    image[j * width + k] = repeats;
                }
            }
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = CPU_COUNT(&cpu_set);

    /* argument parsing */
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int64_t*)malloc(width * height * sizeof(int64_t));

    pthread_t threads[num_threads];
    int thread_ids[num_threads];

    y_scale = ((upper - lower) / height);
    x_scale = ((right - left) / width);

    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, mandelbrot, &thread_ids[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    // free(image);

    return 0;
}
