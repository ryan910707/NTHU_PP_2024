#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <png.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <immintrin.h>

int world_size;
int num_threads;

int* image;
int width;
int height;
int iters;
double left;
double right;
double lower;
double upper;
double y_scale, x_scale;
int partition;
// Constants for SIMD operations
__m512d v_2;
__m512d v_4;
__m256i v_1;
__mmask8 mask = 0xFF;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    // assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    // assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    // assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    int new_y=0;
    int range = partition*world_size;
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[new_y * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                int p_shift = (p & 0xF)<<4;
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p_shift;
                } else {
                    color[0] = p_shift;
                }
            }
        }
        png_write_row(png_ptr, row);
        new_y += partition;
		if (new_y >= range)
			new_y = new_y%partition + 1;
    }
    png_write_end(png_ptr, NULL);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* Initialize constants for SIMD */
    v_2 = _mm512_set1_pd(2.0);
    v_4 = _mm512_set1_pd(4.0);
    v_1 = _mm256_set1_epi32(1);

    if (height < world_size)
	{
        if(rank >= height){
			return 0;
		}
		world_size = height;
	}


    partition = ceil((double)height / world_size);
    int image_size = width * partition;
    image = (int*)malloc(image_size * sizeof(int));

    y_scale = ((upper - lower) / height);
    x_scale = ((right - left) / width);
    int row_count,offset, remaining;
    double y0;
    __m512d v_y0;
    int i,j;
    #pragma omp parallel for schedule(dynamic) default(shared) private(j, i, row_count, offset, y0, v_y0, remaining)
    for (j = height-1-rank; j >= 0; j -= world_size) {
        row_count = (height-1-rank - j) / world_size;
        offset = row_count*width;
        y0 = j * y_scale + lower;
        __m512d v_y0 = _mm512_set1_pd(y0);
        for (i = 0; i < width; i+=8) {
            remaining = width - i;
            if (remaining >= 8) {
                __m512d v_i = _mm512_set_pd(i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i + 0);
                __m512d v_x0 = _mm512_fmadd_pd(v_i, _mm512_set1_pd(x_scale), _mm512_set1_pd(left));
                __m512d v_x = _mm512_setzero_pd();
                __m512d v_y = _mm512_setzero_pd();
                __m512d v_length_squared = _mm512_setzero_pd();
                __m256i v_repeat = _mm256_setzero_si256();
                __m512d v_x_sq = _mm512_setzero_pd();
                __m512d v_y_sq = _mm512_setzero_pd();
                for(int k=0;k<iters;k++) {
                    __mmask8 cmp_mask = _mm512_cmp_pd_mask(v_length_squared, v_4, _CMP_LT_OS);
                    if (cmp_mask == 0)
                        break;

                    v_repeat = _mm256_mask_add_epi32(v_repeat, cmp_mask, v_repeat, v_1);

                    __m512d temp = _mm512_add_pd(_mm512_sub_pd(v_x_sq, v_y_sq), v_x0);
                    v_y = _mm512_fmadd_pd(_mm512_mul_pd(v_x, v_y), v_2, v_y0);
                    v_x = temp;
                    v_x_sq = _mm512_mul_pd(v_x, v_x);
                    v_y_sq = _mm512_mul_pd(v_y, v_y);
                    v_length_squared = _mm512_fmadd_pd(v_x, v_x, v_y_sq);
                }
                _mm256_mask_storeu_epi32((__m256i*)(image + offset + i), mask, v_repeat);
            }
            else {
                for (int k = i; k < width; ++k) {
                    double x0 = k * x_scale + left;
                    double x = 0;
                    double y = 0;
                    double x_2 = 0;
                    double y_2 = 0;
                    double length_squared = 0;
                    int repeats = 0;
                    for(int k =0;k<iters;k++) {
                        if(length_squared>4)
                            break;
                        double temp = x_2 - y_2 + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        x_2 = x*x;
                        y_2 = y*y;
                        length_squared = x_2 + y_2;
                        repeats++;
                    }
                    image[offset + k] = repeats;
                }
            }
        }
        
    }

    int *final_image = (int*)malloc(image_size*world_size * sizeof(int));
    MPI_Gather(image, image_size, MPI_INT, final_image, image_size, MPI_INT, 0,MPI_COMM_WORLD);
    if(rank==0){
        write_png(filename, iters, width, height, final_image);
    }
    return 0;
}