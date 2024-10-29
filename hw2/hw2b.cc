#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
// #include <assert.h>
#include <png.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <emmintrin.h>

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
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    int new_y=0;
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[new_y * width + x];
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
        new_y += partition;
		if (new_y >= partition * world_size)
			new_y = new_y%partition + 1;
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    //  cpu_set_t cpu_set;
    // sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* argument parsing */
    // assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    if (height < world_size)
	{
        if(rank >= height){
			MPI_Finalize();
			return 0;
		}
		world_size = height;
	}


    partition = ceil((double)height / world_size);
    /* allocate memory for image */
    image = (int*)malloc(width * partition * sizeof(int));
    // assert(image);

    y_scale = ((upper - lower) / height);
    x_scale = ((right - left) / width);

    int odd = width&1 ? 1 : 0;

    __m128d v_2 = _mm_set_pd1(2);
	__m128d v_4 = _mm_set_pd1(4);

    // struct timespec start, end, temp;
    // double time_used;   
    // clock_gettime(CLOCK_MONOTONIC, &start); 
    #pragma omp parallel for schedule(dynamic) default(shared) 
    for (int j = height-1-rank; j >= 0; j -= world_size) {
        int row_count = (height-1-rank - j) / world_size;
        double y0 = j * y_scale + lower;
        __m128d v_y0 = _mm_load1_pd(&y0);

        #pragma omp parallel for schedule(dynamic) default(shared) 
        for (int i = 0; i < width-1; i+=2) {
            double x0[2] = {i * x_scale + left, (i + 1) * x_scale + left};
            __m128d v_x0 = _mm_load_pd(x0);
            __m128d v_x = _mm_setzero_pd();
            __m128d v_y = _mm_setzero_pd();
            __m128d v_sq_x = _mm_setzero_pd();
            __m128d v_sq_y = _mm_setzero_pd();
            __m128i v_repeat = _mm_setzero_si128();
            __m128d v_length_squared = _mm_setzero_pd();
            int repeats = 0;
            while(repeats < iters){
                __m128d v_cmp = _mm_cmpgt_pd(v_4, v_length_squared);
                //if two > 4 break
                if (_mm_movemask_pd(v_cmp) == 0)
                    break;
                repeats++;
                __m128d temp = _mm_add_pd(_mm_sub_pd(v_sq_x, v_sq_y), v_x0);
                v_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(v_x, v_y), v_2), v_y0);
                v_x = temp;
                v_sq_x = _mm_mul_pd(v_x,v_x);
                v_sq_y = _mm_mul_pd(v_y, v_y);
                v_length_squared = _mm_or_pd(_mm_andnot_pd(v_cmp, v_length_squared), _mm_and_pd(v_cmp, _mm_add_pd(v_sq_x, v_sq_y)));
                v_repeat = _mm_add_epi64(v_repeat, _mm_srli_epi64(_mm_castpd_si128(v_cmp), 63));
            }
            _mm_storel_epi64((__m128i*)(image + row_count*width+i), _mm_shuffle_epi32(v_repeat, 0b01000));
        }
        if(odd){
            //handle odd width 
            int i = width-1;
            double x0 = i * x_scale + left;
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
            image[row_count * width + i] = repeats;
        }
    }

    int *final_image = (int*)malloc(width * partition*world_size * sizeof(int));
    MPI_Gather(image, partition * width, MPI_INT, final_image, partition * width, MPI_INT, 0,MPI_COMM_WORLD);
    if(rank==0){
        write_png(filename, iters, width, height, final_image);
    }
    free(image);
    free(final_image);
    MPI_Finalize();

    return 0;
}